"""Run a rehearsal spec: the real run commands against the fake provider, then the speed report.

A spec (YAML; paths relative to the spec file):

    profile: board.profile.yaml        # `haenv loadtest profile`
    library: board.library.jsonl
    time_scale: 0.05                   # service times x this
    k: 3                               # repeats per job (tools/reliability_passk.py --k)
    workers: 24                        # lanes per job (--workers), divided over the k samples
    budget_usd: 300                    # the shared ledger's limit
    models: [gemini-3.7-flash, ...]    # default: config.default_models
    config: {backend_limits: {...}}    # merged into the run's configuration (an overlay)
    probes: {slot: [module:function, ...], ledger: [...], send: [...], cell: [...]}
    acceptance: {overhead_share: 0.05, ledger_p99_s: 0.05, judge_lag_share: 0.10}
    max_wall_s: 7200                   # stop (SIGKILL) after this long; the report says it was cut
    burst_s: 300                       # the start-up burst the report gives its own ledger latency
    today: 2026-10-05                  # the day tariff checks see; default: the real day
    jobs:
      - {job: m2.job.yaml, batch: built/m2/<stamp>}   # a built batch to sample from

`scale` multiplies `k` and `workers` (2 = twice the board, twice the lanes). `kill_at` kills
every run process (SIGKILL) once that fraction of the solver cells is answered, then starts
the same commands again; the report checks that nothing was lost or bought twice.

The run happens in its own output root (`HAENV_OUTPUT_ROOT`) with its own ledger and a key
file of dummy keys; with `transport.rehearsal_url` set, no request leaves the machine.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from haenv import ROOT
from haenv.runtime.loadtest.profile import Profile
from haenv.runtime.loadtest.responder import Responder
from haenv.runtime.loadtest.server import FakeProvider

#: Dummy credentials: every backend has a key, none of them is real.
DUMMY_KEYS = {"OPENROUTER_API_KEY": "sk-rehearsal", "GOOGLE_GENERATIVE_AI_API_KEY": "rehearsal",
              "DASHSCOPE_API_KEY": "sk-rehearsal", "OPENAI_API_KEY": "sk-rehearsal"}
#: Seconds between progress checks.
POLL_S = 2.0
#: Modules whose presence on the stack tags a probe event (judge calls vs. solver calls).
PROBE_TAGS = {"judge": "haenv.semantic.transport"}
_PROXY_VARS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY")


@dataclass
class Job:
    job: Path
    batch: Path
    start_after_s: float = 0.0

    @property
    def doc(self) -> dict:
        from haenv.resources.yamlcache import load_yaml
        return load_yaml(self.job)


@dataclass
class Rehearsal:
    profile: Path
    library: Path
    jobs: list[Job]
    out: Path
    time_scale: float = 0.05
    k: int = 3
    workers: int = 24
    budget_usd: float = 300.0
    models: list[str] | None = None
    config: dict = field(default_factory=dict)
    probes: dict = field(default_factory=dict)
    acceptance: dict = field(default_factory=dict)
    max_wall_s: float | None = None
    burst_s: float | None = None
    today: str | None = None
    kill_at: float | None = None
    seed: int = 0

    @classmethod
    def from_spec(cls, spec: Path, *, out: Path | None = None, scale: int | None = None,
                  time_scale: float | None = None, kill_at: float | None = None) -> "Rehearsal":
        spec = Path(spec).resolve()
        from haenv.resources.yamlcache import load_yaml
        doc = load_yaml(spec)
        here = lambda p: (spec.parent / p).resolve()  # noqa: E731
        factor = int(scale or 1)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return cls(profile=here(doc["profile"]), library=here(doc["library"]),
                   jobs=[Job(here(j["job"]), here(j["batch"]), float(j.get("start_after_s", 0)))
                         for j in doc["jobs"]],
                   out=Path(out).resolve() if out else spec.parent / "runs" / f"{stamp}-x{factor}",
                   time_scale=float(time_scale if time_scale is not None else doc.get("time_scale", 0.05)),
                   k=int(doc.get("k", 3)) * factor, workers=int(doc.get("workers", 24)) * factor,
                   budget_usd=float(doc.get("budget_usd", 300)), models=doc.get("models"),
                   config=doc.get("config") or {}, probes=doc.get("probes") or {},
                   acceptance=doc.get("acceptance") or {}, kill_at=kill_at,
                   max_wall_s=doc.get("max_wall_s"), burst_s=doc.get("burst_s"),
                   today=str(doc["today"]) if doc.get("today") else None, seed=int(doc.get("seed", 0)))

    # ---- setup ----
    def _check_clean_tree(self) -> None:
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True).stdout
        if dirty:
            raise SystemExit("haenv loadtest: commit or stash changes first -- paid judging "
                             "(rehearsed too) refuses an uncommitted tree:\n" + dirty)

    def _prepare(self, url: str) -> dict:
        root = self.out / "root"
        for job in self.jobs:
            doc = job.doc
            dest = root / "results" / doc["task_type"] / doc["job_id"] / job.batch.name
            if not dest.exists():
                shutil.copytree(job.batch, dest)
        (self.out / "dummy.env").write_text("".join(f"{k}={v}\n" for k, v in DUMMY_KEYS.items()))
        overlay = {**self.config, "env_file": str(self.out / "dummy.env"),
                   "transport": {**(self.config.get("transport") or {}), "rehearsal_url": url}}
        (self.out / "overlay.yaml").write_text(yaml.safe_dump(overlay, sort_keys=False))
        env = {k: v for k, v in os.environ.items() if k not in _PROXY_VARS}
        env.update(no_proxy="127.0.0.1,localhost", NO_PROXY="127.0.0.1,localhost",
                   HAENV_OUTPUT_ROOT=str(root), HAENV_CONFIG_OVERLAY=str(self.out / "overlay.yaml"),
                   HAENV_ENV_FILE=str(self.out / "dummy.env"), TMPDIR=str(self.out / "tmp"),
                   PYTHONPATH=os.pathsep.join([str(Path(__file__).parent / "probe_site"), str(ROOT),
                                               str(ROOT / "plugins")]),
                   HAENV_LOADTEST_PROBES=json.dumps(self.probes),
                   HAENV_LOADTEST_PROBE_LOG=str(self.out / 'probes'),
                   HAENV_LOADTEST_PROBE_TAGS=json.dumps(PROBE_TAGS))
        if self.today:
            env["HAENV_LOADTEST_TODAY"] = self.today
        (self.out / "tmp").mkdir(exist_ok=True)
        return env

    def _command(self, job: Job) -> list[str]:
        cmd = [sys.executable, "-m", "haenv.reporting.tools.reliability_passk", str(job.job),
               "--from-batch", job.batch.name, "--k", str(self.k), "--workers", str(self.workers),
               "--judge-budget-usd", str(self.budget_usd),
               "--judge-budget-ledger", str(self.out / "ledger" / "budget.json")]
        return cmd + (["--models", ",".join(self.models)] if self.models else [])

    # ---- progress ----
    def sample_dirs(self) -> list[Path]:
        root = self.out / "root" / "results"
        return [d for job in self.jobs
                for d in sorted((root / job.doc["task_type"] / job.doc["job_id"]).glob(f"{job.batch.name}-pk*"))]

    def answered(self) -> set[tuple]:
        """(sample, case, model, raw sha) of every answered cell of a rehearsed model."""
        import hashlib
        models = set(self.rehearsed_models())
        out = set()
        for d in self.sample_dirs():
            path = d / "responses.jsonl"
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue                           # a line cut by a kill
                if row.get("solver") in models and row.get("raw"):
                    out.add((d.name, row["case"], row["solver"],
                             hashlib.sha256(str(row["raw"]).encode()).hexdigest()[:16]))
        return out

    def rehearsed_models(self) -> list[str]:
        from haenv.app.config import load_cfg
        return list(self.models or load_cfg().get("default_models") or [])

    def expected_cells(self) -> int:
        models = self.rehearsed_models()
        n = 0
        for job in self.jobs:
            with (job.batch / "cases.jsonl").open(encoding="utf-8") as f:
                n += sum(1 for line in f if line.strip())
        return n * len(models) * self.k

    # ---- run ----
    def _launch(self, env: dict, attempt: int) -> list[subprocess.Popen]:
        logs = self.out / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        procs, started = [], time.time()
        for job in sorted(self.jobs, key=lambda j: j.start_after_s):
            delay = job.start_after_s - (time.time() - started)
            if delay > 0 and attempt == 0:
                time.sleep(delay)
            log = (logs / f"{job.doc['job_id']}.{attempt}.log").open("w")
            procs.append(subprocess.Popen(self._command(job), cwd=ROOT, env=env, stdout=log,
                                          stderr=subprocess.STDOUT, start_new_session=True))
        return procs

    def run(self):
        from haenv.runtime.loadtest.report import SpeedReport
        self._check_clean_tree()
        self.out.mkdir(parents=True, exist_ok=True)
        entries = [json.loads(line) for line in self.library.read_text(encoding="utf-8").splitlines() if line]
        fake = FakeProvider(Profile.load(self.profile), Responder(entries, seed=self.seed),
                            self.out / "requests.jsonl", time_scale=self.time_scale, seed=self.seed).start()
        record = {"spec": {"k": self.k, "workers": self.workers, "time_scale": self.time_scale,
                           "budget_usd": self.budget_usd, "models": self.models,
                           "jobs": [str(j.job) for j in self.jobs], "kill_at": self.kill_at,
                           "acceptance": self.acceptance, "probes": self.probes,
                           "backend_limits": self.config.get("backend_limits") or {},
                           **({"burst_s": self.burst_s} if self.burst_s is not None else {})},
                  "fake_url": fake.url, "started": time.time(), "attempts": []}
        procs: list[subprocess.Popen] = []
        try:
            env = self._prepare(fake.url)
            expected = self.expected_cells()
            record["expected_cells"], record["models"] = expected, self.rehearsed_models()
            procs, attempt = self._launch(env, 0), 0
            while any(p.poll() is None for p in procs):
                time.sleep(POLL_S)
                if self.max_wall_s and time.time() - record["started"] > self.max_wall_s:
                    record["stopped"] = f"max_wall_s={self.max_wall_s}"
                    break
                if self.kill_at is not None and attempt == 0 and \
                        len(self.answered()) >= self.kill_at * expected:
                    for p in procs:
                        os.killpg(p.pid, signal.SIGKILL)
                    for p in procs:
                        p.wait()
                    before = sorted(self.answered())
                    record["attempts"].append({"killed_at": time.time(), "rcs": [p.returncode for p in procs],
                                               "answered_before_kill": len(before)})
                    (self.out / "answered_before_kill.json").write_text(json.dumps(before))
                    attempt = 1
                    procs = self._launch(env, attempt)
            if record.get("stopped"):
                for p in procs:
                    if p.poll() is None:
                        os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
            record["attempts"].append({"ended": time.time(), "rcs": [p.returncode for p in procs]})
            record["ended"] = time.time()
            record["answered"] = len(self.answered())
        finally:
            for p in procs:
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGKILL)
            fake.stop()
            (self.out / "run.json").write_text(json.dumps(record, indent=1))
        return SpeedReport.read(self.out)
