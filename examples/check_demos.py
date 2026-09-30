#!/usr/bin/env python3
"""Run every example demo next to its negative control and check the pair.

    python examples/check_demos.py            # all demos
    python examples/check_demos.py plugin_demo attrib_demo

The example packages must be installed (`uv pip install --no-deps -e examples/<name>`); a
demo whose package is missing is reported as NOT RUN and fails. A demo passes only when
the demo batch carries the plugin's output and the control batch (the same job without
`plugins:`) does not. Offline, no cost; batches go to `HAENV_OUTPUT_ROOT` (default: the
repository root). Exit code 0 when every demo passes, 1 otherwise (2 for an unknown demo name).

SYNTHETIC data, for evaluation only, not medical advice.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
EX = ROOT / "examples"
OUT = pathlib.Path(os.environ.get("HAENV_OUTPUT_ROOT") or ROOT)


def _haenv(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "haenv", *args], cwd=ROOT,
                          capture_output=True, text=True, timeout=1800)


def _latest_batch(task: str, job_id: str) -> pathlib.Path | None:
    d = OUT / "results" / task / job_id
    bs = sorted(p for p in d.glob("*") if p.is_dir() and p.name[:8].isdigit()) if d.is_dir() else []
    return bs[-1] if bs else None


def _rows(task: str, job_id: str) -> list[dict]:
    b = _latest_batch(task, job_id)
    if b is None or not (b / "eval.jsonl").is_file():
        return []
    return [json.loads(x) for x in (b / "eval.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()]


def _keys_pair(name: str, task: str, keys: tuple[str, ...], every_row: bool = False,
               extra=None) -> tuple[bool, str]:
    """Run demo + control with `haenv run --offline --fresh` and compare the key sets."""
    base = EX / name
    jid = _job_id(base / "demo.job.yaml")
    jid0 = _job_id(base / "demo-noplugin.job.yaml")
    r1 = _haenv("run", str(base / "demo.job.yaml"), "--offline", "--fresh")
    r0 = _haenv("run", str(base / "demo-noplugin.job.yaml"), "--offline", "--fresh")
    if r1.returncode or r0.returncode:
        return False, f"exit codes demo={r1.returncode} control={r0.returncode}: {(r1.stderr or r0.stderr)[-300:]}"
    rows1, rows0 = _rows(task, jid), _rows(task, jid0)
    if not rows1 or not rows0:
        return False, f"no rows read (demo {len(rows1)}, control {len(rows0)})"
    with_keys = [r for r in rows1 if all(k in r for k in keys)]
    leaked = sorted({k for r in rows0 for k in keys if k in r})
    ok_demo = (len(with_keys) == len(rows1)) if every_row else bool(with_keys)
    msg = (f"demo: {len(with_keys)}/{len(rows1)} rows carry all {len(keys)} keys · "
           f"control: {len(rows0)} rows, plugin keys present: {leaked or 'none'}")
    ok = ok_demo and not leaked
    if ok and extra:
        ok, more = extra(rows1)
        msg += f" · {more}"
    return ok, msg


def _job_id(p: pathlib.Path) -> str:
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.startswith("job_id:"):
            return line.split(":", 1)[1].strip()
    raise ValueError(f"no job_id in {p}")


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def plugin_demo():
    return _keys_pair("plugin_demo", "early_warning",
                      ("pdemo_n_cited_ev", "pdemo_drivers_total", "pdemo_drivers_sourced",
                       "pdemo_evidence_ok"), every_row=True)


def llm_judge_demo():
    keys = ("llmj_disc_status", "llmj_disc_via", "llmj_disc_n_rivals", "llmj_disc_n_claims",
            "llmj_disc_n_judged", "llmj_disc_n_discriminative", "llmj_disc_rate",
            "llmj_disc_verdicts")

    def _offline_rate_is_null(rows):
        bad = [r.get("llmj_disc_rate") for r in rows if "llmj_disc_rate" in r
               and r.get("llmj_disc_rate") is not None]
        return (not bad), f"offline llmj_disc_rate non-null in {len(bad)} rows (must be 0)"
    return _keys_pair("llm_judge_demo", "joint_dx", keys, extra=_offline_rate_is_null)


def external_subject_demo():
    return _keys_pair("external_subject_demo", "tracking_review",
                      ("xdemo_n_stages", "xdemo_risk_monotonic", "xdemo_stage_note"))


def world_plugin_demo():
    base = EX / "world_plugin_demo"
    shas = {}
    for f in ("demo.job.yaml", "demo-noplugin.job.yaml"):
        r = _haenv("build", str(base / f), "--gen", "deterministic", "--fresh")
        if r.returncode:
            return False, f"build {f} exit {r.returncode}: {r.stderr[-300:]}"
        b = _latest_batch("early_warning", _job_id(base / f))
        meta = json.loads((b / "batch.json").read_text(encoding="utf-8")) if b else {}
        shas[f] = meta.get("world_sha")
    a, c = shas["demo.job.yaml"], shas["demo-noplugin.job.yaml"]
    return (bool(a) and bool(c) and a != c), f"world_sha demo={a} control={c} (must differ)"


def attrib_demo():
    def _gold_delivered(rows):
        kinds = sorted({r.get("attrib_gold_kind") for r in rows})
        return kinds == ["event", "no_event"], f"attrib_gold_kind values {kinds}"
    base = EX / "attrib_demo"
    r1 = _haenv("run", str(base / "demo.job.yaml"), "--offline", "--fresh")
    if r1.returncode:
        return False, f"demo exit {r1.returncode}: {r1.stderr[-300:]}"
    rows = _rows("early_warning", _job_id(base / "demo.job.yaml"))
    keys = ("attrib_hit", "attrib_answer_kind", "attrib_gold_kind", "attrib_n_candidates")
    n = sum(1 for r in rows if all(k in r for k in keys))
    ok_g, gmsg = _gold_delivered(rows)
    r0 = _haenv("run", str(base / "demo-noplugin.job.yaml"), "--offline", "--fresh")
    refused = r0.returncode != 0 and "attrib_payload" in (r0.stderr + r0.stdout)
    return (bool(rows) and n == len(rows) and ok_g and refused), (
        f"demo: {n}/{len(rows)} rows carry the four attrib_* keys · {gmsg} · "
        f"control exit {r0.returncode}, names `attrib_payload`: {refused}")


DEMOS = {
    "plugin_demo": ("haenv_plugin_demo", plugin_demo),
    "llm_judge_demo": ("haenv_llm_judge_demo", llm_judge_demo),
    "external_subject_demo": ("haenv_subject_demo", external_subject_demo),
    "world_plugin_demo": ("haenv_world_demo", world_plugin_demo),
    "attrib_demo": ("haenv_attrib_demo", attrib_demo),
}


def main(argv: list[str]) -> int:
    names = argv or list(DEMOS)
    unknown = [n for n in names if n not in DEMOS]
    if unknown:
        print(f"unknown demo(s): {unknown}; known: {list(DEMOS)}")
        return 2
    failed = 0
    for name in names:
        module, fn = DEMOS[name]
        if not _installed(module):
            print(f"[demos] 🔴 {name}: NOT RUN -- package `{module}` is not installed "
                  f"(uv pip install --no-deps -e examples/{name})")
            failed += 1
            continue
        try:
            ok, msg = fn()
        except Exception as e:                       # noqa: BLE001 -- report, never swallow
            ok, msg = False, f"{type(e).__name__}: {e}"
        print(f"[demos] {'✅' if ok else '🔴'} {name}: {msg}")
        failed += 0 if ok else 1
    print(f"[demos] {len(names) - failed}/{len(names)} demos pass")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
