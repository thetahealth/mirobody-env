"""One command runs the whole pipeline: generate, emission gate, evaluate, report.

    haenv run inputs/example-ew.job.yaml             # everything, resuming the latest batch
    haenv build inputs/example-ew.job.yaml           # generate and self-check only, offline
    haenv report inputs/example-ew.job.yaml          # re-render from existing JSONL
    haenv run inputs/example-ew.job.yaml --fresh     # start a new batch
    haenv run inputs/example-ew.job.yaml --offline   # smoke test: offline reference only

For long runs, detach so a closed session does not stop the run::

    nohup haenv run inputs/example-ew.job.yaml > run.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

from . import kernel_path as _kernel_path

# Resources (config, registries) are read from the data root.
from haenv import data_root as _data_root
from haenv import output_root as _output_root
ROOT = _data_root()

# Artefacts go to the output root. The two roots coincide in a source tree; after a
# non-editable install the data root is inside site-packages, so batches are always looked
# up under `OUT`.
OUT = _output_root()


def _bootstrap(cfg: dict) -> None:
    """Check that the L0 kernel is on the import path (`haenv.__init__` mounts it) and exit
    with a readable message if it is missing.
    """
    kp = _kernel_path(cfg)
    if kp is None or not kp.exists():
        sys.exit(f"[haenv] kernel path does not exist: {kp}"
                 f" (check config.yaml:kernel_path, or the HAENV_KERNEL_PATH env var)")
    if str(kp) not in sys.path:
        sys.path.insert(0, str(kp))


def _deep_merge(base: dict, over: dict) -> dict:
    """Merge `over` into `base` in place: mappings recurse, everything else replaces."""
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_cfg() -> dict:
    """`config.yaml`, with the untracked `config.local.yaml` (deployment settings) merged over it."""
    # Parsed once per process and returned as deep copies (callers mutate them).
    from .yamlcache import load_yaml as _cached
    cfg = _cached(ROOT / "config.yaml")
    local = ROOT / "config.local.yaml"
    if local.is_file():
        _deep_merge(cfg, _cached(local) or {})
    from . import settings
    return settings.validate(settings.apply_set(apply_config_overlay(cfg)))


def config_layers() -> list[dict]:
    """Every file merged into `load_cfg`, in order, with its SHA-256 (`--set` is recorded
    separately, see `settings.record`)."""
    import hashlib
    paths = [ROOT / "config.yaml", ROOT / "config.local.yaml"]
    named = os.environ.get(CONFIG_OVERLAY_ENV, "").strip()
    if named:
        paths.append(Path(named))
    return [{"file": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in paths if p.is_file()]


CONFIG_OVERLAY_ENV = "HAENV_CONFIG_OVERLAY"


def apply_config_overlay(cfg: dict) -> dict:
    """Merge the per-run overlay named by `$HAENV_CONFIG_OVERLAY` over `cfg`, in place.

    The overlay is a YAML file in the shape of `config.yaml`; it is applied after
    `config.local.yaml`, so one process (one `haenv run`) can route a model to a different
    backend without touching either file. What was routed is recorded in `batch.json`
    (`sampling.per_model.<model>.backend` and `solving_sha16`). A named overlay that does not
    exist is an error: a silent fallback would run the cells on the wrong provider.
    """
    named = os.environ.get(CONFIG_OVERLAY_ENV, "").strip()
    if not named:
        return cfg
    from .yamlcache import load_yaml as _cached
    path = Path(named)
    if not path.is_file():
        raise FileNotFoundError(f"{CONFIG_OVERLAY_ENV} names a file that does not exist: {path}")
    return _deep_merge(cfg, _cached(path) or {})



#: Extensions treated as case text. Anything that is not a job file is text.
TEXT_SUFFIXES = (".md", ".txt")


def _resolve_input(a, root: Path) -> str:
    """Resolve the command-line input to a job file path.

    A ``*.job.yaml`` is returned as is. A text file or directory is extracted into
    ``derived/<stem>.job.yaml`` and that path is returned. ``--gold`` has no default:
    the gold source decides how far the batch can be trusted, and it is recorded in the
    artefacts and the report.
    """
    p = Path(a.job)
    if str(p).endswith((".job.yaml", ".job.yml")) or (p.is_file() and p.suffix in (".yaml", ".yml")):
        return str(p)
    files = (sorted(x for x in p.iterdir() if x.suffix in TEXT_SUFFIXES)
             if p.is_dir() else [p])
    if not files:
        sys.exit(f"[haenv] {p} has no {TEXT_SUFFIXES} files")
    if not getattr(a, "gold", None):
        sys.exit("[haenv] `--gold` is required when the input is case text "
                 "(absent|text|llm|clinician) -- where the gold comes from decides how "
                 "trustworthy this batch is, so there is no default. `--gold llm` hands gold "
                 "to a model; the artefact will be flagged "
                 "field by field and called out separately in the report.")
    import json as _json

    import yaml as _yaml

    from .extract import ExtractionError, text_to_case
    cfg = load_cfg()
    ann = (_json.loads(Path(a.annotations).read_text(encoding="utf-8"))
           if a.gold == "clinician" and a.annotations else {})
    if a.gold == "clinician" and not a.annotator:
        sys.exit("[haenv] `--gold clinician` requires `--annotator`"
                 " (a human anchor with no \"who\" is not an anchor)")
    # `--gold clinician` without annotations would stamp clinician provenance on empty gold.
    if a.gold == "clinician" and not a.annotations:
        sys.exit("[haenv] `--gold clinician` requires `--annotations` (clinician annotation "
                 "JSON). Without it all 14 gold fields default to `absent`, while the artefact "
                 "still stamps `gold_path: clinician`, which would misstate the gold's provenance. Use `--gold absent` "
                 "for empty gold.")
    cases, prov, srcs = [], {}, {}
    print(f"[haenv] input is case text: {len(files)} file(s) · gold source `{a.gold}`")
    for i, f in enumerate(files, 1):
        text = f.read_text(encoding="utf-8")
        if a.cut_at:
            _ls = text.splitlines()
            for j, ln in enumerate(_ls):
                if ln.startswith(a.cut_at):
                    text = "\n".join(_ls[:j])
                    break
        cid = f.stem.upper().replace("RC-", "").replace("_", "-")
        try:
            ec = text_to_case(text, cid, gold_path=a.gold, model=a.extract_model,
                              annotations=ann.get(cid), annotator=a.annotator, cfg=cfg,
                              input_kind=a.input_kind,
                              offline=bool(getattr(a, "extract_offline", False)))
        except ExtractionError as e:
            print(f"  [{i}/{len(files)}] {cid}: extraction failed {str(e)[:80]} -- skipping")
            continue
        cases.append(ec.to_case_dict())
        srcs[cid] = ec.extract_source or "unknown"
        prov[cid] = ec.ledger.summary()
        _nl = len(prov[cid]["gold_llm_fields"])
        print(f"  [{i}/{len(files)}] {cid}: raw {len(ec.raw)} item(s) · latent {len(ec.latent)} item(s)"
              + (f" · {_nl} gold field(s) LLM-judged" if _nl else ""))
        for w in ec.warnings:
            print(f"        {w[:120]}")
    if not cases:
        sys.exit("[haenv] not a single case was extracted -- refusing to write an empty job")
    # Provenance sources come from their registry, so counts cover any new source.
    from .provenance import PROVENANCES as _PROVS
    agg = {k: sum(v["gold_provenance_counts"].get(k, 0) for v in prov.values())
           for k in _PROVS}
    n_llm = sum(1 for v in prov.values() if v["gold_llm_fields"])
    print(f"[haenv] extracted {len(cases)} case(s) · gold field sources {agg}"
          + (f" · gold for {n_llm}/{len(cases)} case(s) was LLM-judged" if n_llm else ""))
    # Case facts from a placeholder replay fixture are announced at the top of the job file:
    # such a job only proves the mechanism runs.
    _ph = sorted(c for c, v in srcs.items() if str(v).startswith("replay:placeholder"))
    if _ph:
        print(f"[haenv] patient facts for {len(_ph)}/{len(cases)} case(s) come from a "
              f"placeholder replay fixture -- this job must not be used for any reading, "
              f"it only proves this gold path works end to end.")
    jid = p.stem if p.is_dir() else p.parent.name
    out = root / "derived" / f"{jid}.job.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        (("# " + "=" * 74 + "\n"
          "# PLACEHOLDER ARTEFACT -- MUST NOT BE USED FOR ANY READING\n"
          f"# {len(_ph)}/{len(cases)} case(s) in this job have patient facts taken from a "
          f"hand-written replay fixture\n"
          f"# (`origin: placeholder-handwritten`),\n"
          "# not extracted from case text. If gold goes through `--gold clinician`, the "
          "annotations are placeholder too.\n"
          "# Its only purpose is proving this gold path works end to end with zero model "
          "calls.\n"
          "# Per-case source: see `_provenance.extract_sources`.\n"
          "# " + "=" * 74 + "\n") if _ph else "")
        + f"# Generated from case text by `haenv <cmd> {a.job} --gold {a.gold}` -- do not "
        f"hand-edit.\n"
        "# `_provenance` records where each gold field came from; it carries through to the "
        "report.\n"
        "# SYNTHETIC, evaluation only, not medical advice.\n"
        + _yaml.safe_dump({
            "job_id": jid, "task_type": "joint_dx", "findings": True,
            "multiround": False, "include_baseline": True, "models": [],
            "sample_cases": min(6, len(cases)), "report": f"eval-{jid}.md",
            # Input kind and cut point decide whether a transcribed outcome label can be trusted
            # (a prospective text has no outcome; an uncut file shows the extractor the answers).
            "_provenance": {"gold_path": a.gold, "extractor_model": a.extract_model,
                            "annotator": a.annotator,
                            "extract_offline": bool(getattr(a, "extract_offline", False)),
                            "extract_sources": srcs, "placeholder_facts": _ph,
                            "input_kind": a.input_kind, "cut_at": a.cut_at or None,
                            "gold_field_source_totals": agg,
                            "cases_with_llm_gold": n_llm, "per_case": prov},
            "cases": cases}, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    print(f"[haenv] pack spec -> {out}")
    return str(out)


def _gen_model_of(syn: dict) -> str:
    """The generation model, from `config.synth.gen_model` only; raises if it is missing
    rather than falling back to a built-in default.
    """
    m = (syn or {}).get("gen_model")
    if not m:
        raise KeyError(
            "config.synth.gen_model is missing; set it in config.yaml (there is no built-in "
            "default)")
    return str(m)


def rebuild_blockers(results_dir: Path) -> list[str]:
    """Reasons ``--rebuild`` would mix new questions into old answers on this batch; an empty
    list allows it.
    """
    out = []
    for name in ("eval.jsonl", "responses.jsonl"):
        p = Path(results_dir) / name
        if p.is_file() and p.stat().st_size > 0:
            out.append(name)
    return out



def _rel(p) -> str:
    """Path relative to the output or data root for printing, else absolute (the roots can be
    on different volumes).
    """
    from . import output_root as _o
    for base in (_o(), ROOT):
        try:
            return str(Path(p).relative_to(base))
        except ValueError:
            continue
    return str(p)


def main(argv=None) -> int:
    import sys as _sys
    _argv = list(_sys.argv[1:] if argv is None else argv)
    if _argv and _argv[0] in ("ops", "drain"):
        # Operator commands on the shared ledger (runtime knobs, drain); no job is involved.
        from .ops.cli import main as _ops_main
        return _ops_main(_argv[1:] if _argv[0] == "ops" else _argv)
    if _argv[:1] == ["spend"]:
        from .spend import main as spend_main
        return spend_main(_argv[1:])
    try:
        return _main(argv)
    except BaseException as error:  # noqa: BLE001 -- only a drain is a clean exit
        from .ops.runtime import Draining
        if isinstance(error, Draining):
            import logging as _logging
            _logging.getLogger("haenv").warning("[run] drained: no new paid request started; finished rows are on disk; "
                        "resume replays every recorded request")
            return 0
        raise


def _main(argv=None) -> int:
    # Maintenance modules are excluded from the public snapshot and distribution.
    # Advertise only commands that are actually included in this installation.
    maintenance = {name for name in ("inputs", "tasks", "board_freeze")
                   if Path(__file__).with_name(name + ".py").is_file()}
    ap = argparse.ArgumentParser(prog="haenv", description="Health agent environment: generate, evaluate, and report in one pipeline")
    ap.add_argument("cmd", choices=["run", "build", "verify", "report"]
                    + [name for name in ("inputs", "tasks") if name in maintenance])
    ap.add_argument("job", nargs="?", default=None,
                    help="path to job.yaml")
    ap.add_argument("--check", action="store_true",
                    help=("`tasks` only: compare each card with the gold, geometry, framing and primary judge "
                          "its jobs produce; exit 1 on a mismatch") if "tasks" in maintenance else argparse.SUPPRESS)
    ap.add_argument("--fresh", action="store_true",
                    help="start a new batch (new datetime directory) instead of resuming the previous one; the old batch is left untouched")
    ap.add_argument("--batch", metavar="YYYYmmdd-HHMMSS",
                    help="target a specific batch directory (resume, or re-render its report); default: the job's most recent batch")
    # Toggling the physiology layer from the command line avoids copying a job file that carries gold.
    ap.add_argument("--physio", dest="physio", action="store_true", default=None,
                    help="force the physiology layer on (overrides job.yaml's `physio`). "
                         "Paired with `--no-physio` for an A/B: run the same job twice without copying the yaml")
    ap.add_argument("--no-physio", dest="physio", action="store_false",
                    help="force the physiology layer off (overrides job.yaml's `physio`)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override one configuration value after config.local.yaml and the overlay "
                         "(dotted key, YAML value; repeatable), e.g. --set eval.workers=4")
    ap.add_argument("--offline", action="store_true",
                    help="run only the offline deterministic reference (no real model calls); for smoke tests/CI")
    ap.add_argument("--judge-budget-usd", help="approved combined solver and semantic-judge API budget ceiling")
    ap.add_argument("--judge-budget-ledger", type=Path,
                    help="shared persistent API cost ledger; required for online run")
    ap.add_argument("--semantic-run", type=Path,
                    help="sealed semantic backfill directory to use when rendering a report (read-only)")
    ap.add_argument("--models", metavar="A,B",
                    help="run only these models (comma-separated, overrides job.yaml/config.default_models); "
                         "combine with resume to backfill different models in separate batches")
    ap.add_argument("--limit", type=int, metavar="N",
                    help="take only the first N cases (for a small-scale check before running real models)")
    ap.add_argument("--cases", metavar="ID,ID",
                    help="run only the named cases (comma-separated); combine with resume to advance in batches by payload size/priority")
    # ---- input forms: a job file, or case text, through one entry point ----
    ap.add_argument("--gold", choices=["absent", "text", "llm", "clinician"],
                    help="only used when the input is case text: where the gold comes from. "
                         "Required, no default: it decides how far this batch can be trusted")
    ap.add_argument("--annotations", metavar="JSON",
                    help="--gold clinician: clinician annotation file (`{case_id: {gold field: value}}`). "
                         "Placeholder annotations exist for the mechanism self-check only; "
                         "that set must not be used for any reading")
    ap.add_argument("--annotator", metavar="ID", help="--gold clinician: annotator id")
    ap.add_argument("--extract-model", metavar="KEY", default="gemini-3.1-pro",
                    help="which model to use for case-text extraction (default gemini-3.1-pro)")
    ap.add_argument("--extract-offline", action="store_true",
                    help="extraction may only replay: continue only on a hit in the "
                         "`docs/anchor/extract-replay/` fixtures or the `cases/_llm_cache/` "
                         "generation cache -- fail if neither has it (no network calls, no "
                         "falling back to an empty extraction -- an empty raw case would be "
                         "backfilled by downstream defaults into a patient that did not come "
                         "from this text). Used by CI / zero-cost reproduction")
    ap.add_argument("--input-kind", choices=["prospective", "retrospective"],
                    default="prospective",
                    help="which kind of case text is being fed in. `prospective` (default, "
                         "strict) = chief complaint + baseline at enrollment, no outcome; "
                         "`retrospective` = a full retrospective case report that states the "
                         "outcome verbatim, so transcribing the outcome label is allowed. The "
                         "strict default is deliberate: misdeclaring a prospective text as "
                         "retrospective would make the extractor force an outcome out of text "
                         "that has none and stamp it `source_text` (forged provenance)")
    ap.add_argument("--cut-at", metavar="HEADING", default="",
                    help="truncate the input at this heading -- our own raw case markdown has a "
                         "verifier-only section, and feeding the whole file would show the "
                         "extractor the answer; real case text does not need this")
    ap.add_argument("--gen", choices=["llm", "deterministic"],
                    help="question generator (default: config.synth.generator); deterministic = formulaic, free and CI-safe")
    ap.add_argument("--gen-model", metavar="KEY",
                    help="model key used for question generation (config.synth.gen_model is the only default source; there is no fallback in the code)")
    ap.add_argument("--gen-workers", type=int, metavar="N", default=0,
                    help="number of concurrent question-generation workers (0 = use `config.synth.gen_workers`). "
                         "Deterministic generation sends no requests and automatically falls back to serial")
    ap.add_argument("--regen", action="store_true",
                    help="ignore the question-generation LLM cache and force fresh model calls (default: hits cases/_llm_cache/)")
    ap.add_argument("--workers", type=int, default=0,
                    help="cells in flight per model (0 = use `config.eval.workers`; 1 = serial). Each model's "
                         "pool is min(this value, `backend_limits[backend]`, its cell count), and the models "
                         "on one backend share that backend's cap; rows are identical to a serial run apart "
                         "from timing fields")
    ap.add_argument("--pooling", choices=["model", "backend"], default=None,
                    help="scheduling: one pool per model, longest expected cell first (default, "
                         "`config.eval.pooling`), or one pool per backend in task order")
    ap.add_argument("--allow-retired", action="store_true",
                    help="allow probes marked retired-defective. For reproducing historical batches only -- "
                         "those probes were retired because their questions had confounds, and running new data "
                         "through them only produces another unusable batch of numbers")
    ap.add_argument("--rebuild", action="store_true",
                    help="ignore the case bodies already persisted in the batch and regenerate questions within this batch, overwriting cases.jsonl")
    ap.add_argument("--override-batch-gate", action="store_true",
                    help="force run/report on a batch rejected by a batch-level gate (or "
                         "missing batch.json). This batch's questions may be answerable "
                         "without reasoning, so its scores may be meaningless. The override "
                         "is recorded in batch.json")
    a = ap.parse_args(argv)
    if a.cmd != "tasks" and not a.job:
        ap.error(f"`{a.cmd}` needs a job.yaml path")
    if a.cmd == "tasks" and a.job:
        ap.error("`tasks` takes no job path; it reads every card under tasks/ and every job under inputs/")
    if a.check and a.cmd != "tasks":
        ap.error("`--check` is only valid with `tasks`")

    if getattr(a, "pooling", None):
        a.set.append(f"eval.pooling={a.pooling}")    # `--pooling` is `--set eval.pooling=...`
    if a.set:
        from .settings import export_set
        export_set(a.set)
    cfg = load_cfg()
    logging.basicConfig(level=[logging.WARNING, logging.INFO, logging.DEBUG][int(cfg.get("verbose", 1))],
                        format="%(levelname)s %(name)s: %(message)s")
    _bootstrap(cfg)

    from . import batch as batch_mod
    from .job import load_job  # import after the kernel is on the path
    from .build import build_case
    from .evaluate import run_eval
    from .report import write_report

    from .evaluate import RUN
    # Worker count comes from the config when the flag is unset; there is no default in code.
    _w = int(getattr(a, "workers", 0) or 0)
    if _w <= 0:
        _w = int((cfg.get("eval") or {}).get("workers") or 1)
    RUN.workers = max(1, _w)

    # `inputs` is read-only: it lists every input the framework reads (config, job file,
    # registries, code constants) and where each comes from.
    # `tasks` is read-only too: it assigns each case under inputs/ to a task card by its gold;
    # with --check it exits 1 when a card and the jobs disagree.
    if a.cmd == "tasks":
        from . import tasks as _tasks
        try:
            return _tasks.main(check_only=bool(a.check))
        except FileNotFoundError as e:
            print(f"[haenv tasks] {e}", file=sys.stderr)
            return 2

    if a.cmd == "inputs":
        from . import inputs as _inputs
        _p = Path(a.job)
        print(_inputs.render(_p if _p.suffix in (".yaml", ".yml") and _p.exists() else None))
        # Exit 1 when the register and the code disagree in either direction.
        _unreg = [c for c in _inputs.scan_code_constants() if not c["class"]]
        return 1 if (_unreg or _inputs.absent_entries()) else 0

    if a.cmd == "run" and not a.offline:
        if a.judge_budget_usd is None or a.judge_budget_ledger is None:
            print("[haenv] online solvers and default LLM semantic judging share one budget; provide "
                  "--judge-budget-usd and --judge-budget-ledger before any paid calls")
            return 2

    # A job file or case text; text is extracted to `derived/<stem>.job.yaml` under the output root.
    _job_path = _resolve_input(a, OUT)
    from . import output_root as _out_root
    job = load_job(_job_path, root=_out_root())
    job.allow_retired = bool(a.allow_retired)
    # Applied after loading, so the flag overrides the job file.
    if a.physio is not None:
        from . import events as _ev_sw
        _ev_sw.PHYSIO_ENABLED[0] = bool(a.physio)
        print(f"[haenv] physiology layer {'enabled' if a.physio else 'disabled'} from the command line (overrides job.yaml)")
    if a.models:
        job.models = [m.strip() for m in a.models.split(",") if m.strip()]
    if a.cases:
        want = {x.strip() for x in a.cases.split(",") if x.strip()}
        job.cases = [c for c in job.cases if c.case_id in want]
        missing = want - {c.case_id for c in job.cases}
        if missing:
            print(f"[haenv] job.yaml has no such case(s): {sorted(missing)}"); return 2
    if a.limit:
        job.cases = job.cases[:a.limit]
    if a.offline:  # offline deterministic reference only
        job.models = []
        cfg = {**cfg, "default_models": []}
        job.include_baseline = True

    # ---- board-freeze gate ----
    if "board_freeze" in maintenance:
        from . import board_freeze
        _bf = board_freeze.refusal(
            ROOT, cmd=a.cmd,
            models=(job.models or cfg.get("default_models") or []))
        if _bf:
            print(_bf)
            return 3                               # 3 = refused by a gate, distinct from 2 (usage error)

    # ---- batch directory: archived by run timestamp; the latest batch is reused to resume ----
    try:
        # `--batch <old> --fresh` would delete that batch's evaluation rows, so the
        # combination is refused.
        if a.batch and a.fresh:
            print("[haenv] `--batch` and `--fresh` cannot be given together -- the two "
                  "intents conflict, and combined they actually delete eval.jsonl for the "
                  "batch named by --batch.\n"
                  f"        · To re-run cells already in {a.batch}: give only `--batch {a.batch}`"
                  " (resumes by default, nothing is deleted)\n"
                  "        · To discard old results and start over: give only `--fresh` (opens "
                  "a new timestamped directory, the old batch is left untouched)")
            return 2
        job.batch, is_new = batch_mod.resolve(OUT, job.task_type, job.job_id,
                                              fresh=a.fresh, batch=a.batch)
    except ValueError as e:
        print(f"[haenv] {e}"); return 2
    if a.cmd == "report" and is_new and not a.batch:
        print(f"[haenv] this job has no batch artefacts yet; run `run` first."); return 2
    print(f"[haenv] batch {job.batch} ({'new' if is_new else 'reused, resuming'})"
          f" · details directory {_rel(job.results_dir)}")

    # ---- reuse the batch's cases, so what is scored and reported are the questions that were run ----
    from .store import load_cases, save_cases
    # `--rebuild` on a batch with answers would overwrite the questions and their hash while
    # the recorded answers stay, so it is refused; `--fresh` opens a new batch instead.
    if a.rebuild or a.cmd in ("build", "verify"):
        _dirty = rebuild_blockers(job.results_dir)
        if _dirty:
            _how = "--rebuild" if a.rebuild else a.cmd
            print(f"[haenv] refusing {_how}: batch {job.batch} already has answers"
                  f" ({', '.join(_dirty)}).\n"
                  f"        Regenerating questions in place would mix new questions with old "
                  f"answers in the same batch -- old cells are marked \"done\" by `case|solver` "
                  f"and are not re-run,\n"
                  f"        while the report's fold-out prints the new question and the score "
                  f"comes from the old question's answer.\n"
                  f"        · To change the questions and re-run: `--fresh` (opens a new "
                  f"timestamped directory, batch {job.batch} is left untouched)\n"
                  f"        · To regenerate questions without keeping old answers: move that "
                  f"batch directory aside, or use `--batch <new-stamp>`")
            return 2
    stored = {} if a.rebuild else load_cases(job.cases_file)
    if a.cmd == "report" and not stored:
        print(f"[haenv] batch {job.batch} has no cases.jsonl; run `verify` or `run` first."); return 2
    if stored and a.cmd in ("run", "report"):
        # Only the cases this job lists (`--limit` sampling).
        built = {cs.case_id: stored[cs.case_id] for cs in job.cases if cs.case_id in stored}
        if not built:
            print(f"[haenv] the batch's cases do not match job.yaml; use --rebuild to regenerate."); return 2
        # Every case in `cases.jsonl` passed premise and leak checks, so both are set explicitly
        # (the report reads a missing key as a failure). Other audit columns come from `audit.jsonl`.
        audits = [{"case_id": cid, "emitted": True, "T": int(raw.prediction_context["prediction_time_T"]),
                   "from_batch": True, "outcome_label": raw.outcome_label,
                   "premise_ok": True, "leak_ok": True,
                   "gold_drivers": list(raw.gold_drivers)} for cid, raw in built.items()]
        # Fingerprints are checked on read-back.
        from .store import digest_of
        meta = json.loads((job.results_dir / "batch.json").read_text(encoding="utf-8")) \
            if (job.results_dir / "batch.json").is_file() else {}
        # Cases the emission gate stopped are read back from `batch.json` (`blocked_kinds`,
        # `per_case_kinds_blocked`); `premise_ok` / `leak_ok` stay unset because the leak probe
        # never ran on them.
        _eg = meta.get("emission_gates") or {}
        _why_blocked = _eg.get("blocked_kinds") or {}
        _warns_blocked = _eg.get("per_case_kinds_blocked") or {}
        _want = {cs.case_id for cs in job.cases}
        for _cid in sorted(set(_why_blocked) | set(_warns_blocked)):
            if _cid in built or _cid not in _want:
                continue
            audits.append({"case_id": _cid, "emitted": False, "from_batch": True,
                           "post_noise_conflicts": list(_why_blocked.get(_cid) or []),
                           "gate_warnings": list(_warns_blocked.get(_cid) or []),
                           "premise_error": "(blocked case · read back from the batch: the "
                                            "premise and leakage columns were never recorded, "
                                            "so those two cells are not a verdict; the "
                                            "reason it was stopped is in the post-noise "
                                            "conflicts column)"})
        # The recorded per-case audit (`store.AUDIT_FILENAME`) replaces a stub when both agree on
        # `emitted`; rows follow the job's case order.
        from .store import AUDIT_FILENAME as _AUD_NAME, load_audits as _load_audits
        from .store import merge_recorded_audits
        _aud_path = job.cases_file.parent / _AUD_NAME
        audits = merge_recorded_audits(
            audits, _load_audits(_aud_path) if _aud_path.is_file() else [],
            [cs.case_id for cs in job.cases])
        # ---- batch-level gates fail closed: a missing `batch.json` or a recorded failure is refused ----
        _override_trace = None
        _bg = job.results_dir / "batch.json"
        _blocked = list((meta.get("emission_gates") or {}).get("blocked_by_batch_gate") or [])
        if (meta.get("emission_gates") or {}).get("a5_blocked"):
            _blocked = sorted(set(_blocked) | {"a5_nonclinical_shortcut"})
        _why = (f"batch carries a failure marker {_blocked}" if _blocked
                else ("batch has no batch.json (no failure marker, no fingerprint, no "
                      "emission_gates at all -- on disk this looks identical to \"blocked "
                      "before register ran\")"
                      if not _bg.is_file() else None))
        if _why:
            if not getattr(a, "override_batch_gate", False):
                print(f"[haenv] refusing {a.cmd}: {_why}.")
                print(f"[haenv]    batch {job.batch} did not pass the batch-level emission gate; "
                      f"its scores may be meaningless.")
                print(f"[haenv]    fix the questions and regenerate with `--fresh`; to "
                      f"actually run this batch, add `--override-batch-gate` explicitly "
                      f"(recorded in batch.json).")
                return 4
            print(f"[haenv] --override-batch-gate: {_why} -- explicitly bypassed, recorded in batch.json")
            # The original marker is kept alongside the override.
            _override_trace = {"blocked_by_batch_gate": _blocked,
                               "overridden_batch_gate": _blocked or ["missing_batch_json"],
                               "overridden_by_cmd": a.cmd}
        want, got = meta.get("cases_sha256") or {}, digest_of(built)
        # Compare only the cases this run used.
        drift = sorted(c for c in got if c in want and want[c] != got[c]) if want else []
        print(f"[haenv] reusing case bodies from the batch: {len(built)} case(s) (no "
              f"regeneration; use --rebuild to regenerate)"
              + (f" · fingerprint mismatch {drift[:3]}" if drift else
                 (" · fingerprint matches ✓" if want else " · this batch recorded no fingerprint (older batch)")))
        k_want, (_, k_got) = meta.get("kernel_sha256"), batch_mod.kernel_fingerprint()
        if k_want and k_want != k_got:
            print(f"[haenv] the kernel is not the one used at generation time: batch "
                  f"recorded {k_want} · current {k_got}; results may not be comparable with "
                  f"this batch, --rebuild is recommended")
        elif k_want:
            print(f"[haenv] kernel fingerprint matches ✓ {k_got}")
        batch_mod.register(job, cfg, len(job.cases), len(built), a.cmd,
                           gate_report=_override_trace)
        return _eval_and_report(a, job, cfg, built, audits, run_eval, write_report)

    # ---- generator: a model for the course and daily events, or deterministic ----
    syn = cfg.get("synth", {}) or {}
    gen_mode = a.gen or ("deterministic" if a.offline else syn.get("generator", "deterministic"))
    dispatch = None
    if gen_mode == "llm":
        from .llm import make_dispatcher
        gen_model = a.gen_model or _gen_model_of(syn)
        try:
            # The response cache lives under the output root.
            dispatch = make_dispatcher(cfg, gen_model, OUT,
                                       use_cache=bool(syn.get("gen_cache", True)) and not a.regen)
        except ValueError as e:
            print(f"[haenv] generation model unavailable: {e}"); return 2
        print(f"[haenv] question generator: LLM `{gen_model}` (course + day-by-day event "
              f"indicators; cache {'on' if dispatch.use_cache else 'off'})")
    else:
        print("[haenv] question generator: deterministic (no model calls)")

    # ---- generate: premise check, conditioned generation, converge, verify, emission gate.
    # Parallel by default; parallel and serial runs produce byte-identical cases. ----
    built, audits = {}, []
    _n_workers = max(1, int(getattr(a, "gen_workers", 0) or syn.get("gen_workers", 1) or 1))
    # Deterministic generation sends no requests, so it runs serially.
    if dispatch is None:
        _n_workers = 1

    def _one_case(cs):
        try:
            return cs, build_case(cs, max_rounds=int(syn.get("max_rounds", 6)),
                                  dispatch=dispatch, event_dispatch=dispatch), None
        except Exception as e:  # one failed case must not sink the batch
            return cs, None, e

    if _n_workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        from . import events as _ev_warm
        # Warm lazily loaded registries before threading; failure is harmless.
        for _warm in ("_load_coupling_rules", "_load_symptom_topics", "_sync_pools"):
            _f = getattr(_ev_warm, _warm, None)
            if callable(_f):
                try:
                    _f()
                except Exception:                            # noqa: BLE001
                    pass
        print(f"[haenv] generating with {_n_workers}-way concurrency (results collected in input order, byte-identical to serial)")
        with ThreadPoolExecutor(max_workers=_n_workers) as _ex:
            _results = list(_ex.map(_one_case, job.cases))   # map preserves order
    else:
        _results = [_one_case(cs) for cs in job.cases]

    for cs, _ra, _err in _results:
        if _err is not None:
            log = logging.getLogger("haenv.cli")
            log.error("[haenv] %s generation failed: %s: %s", cs.case_id, type(_err).__name__, _err)
            audits.append({"case_id": cs.case_id, "emitted": False, "T": cs.index_time_T,
                           "gen_error": f"{type(_err).__name__}: {str(_err)[:200]}"})
            continue
        raw, audit = _ra
        audits.append(audit)
        if raw is not None:
            built[cs.case_id] = raw
    print(f"[haenv] generation: {len(built)}/{len(job.cases)} case(s) passed the emission gate")
    if dispatch is not None:
        print(f"[haenv] question-generation LLM calls: {dispatch.n_calls} (cache hits {dispatch.n_hits})")
    if not built:
        print("[haenv] no usable cases (all blocked by the emission gate); stopping."); return 2

    # ---- save the generated cases ----
    gen_tag = f"llm:{a.gen_model or _gen_model_of(syn)}" if dispatch else "deterministic"
    if a.limit:
        # `--limit` samples must not overwrite the batch's cases.
        print(f"[haenv] --limit {a.limit}: skipping save (to avoid overwriting the batch's cases.jsonl with a partial result)")
    else:
        # Batch-level injection-coverage check (`wq`).
        from . import wq as _wq
        for _m in _wq.check_injection_coverage(built):
            print(f"[haenv] iron law two: {_m}")
        _, digests = save_cases(job.cases_file, built)
        from .payloads import FILENAME as _PL_NAME
        from .payloads import save_payloads as _save_payloads
        try:
            _, _pl_digests = _save_payloads(job.cases_file.parent / _PL_NAME, built)
            print(f"[haenv] payloads persisted for {len(_pl_digests)} case(s) "
                  f"-- re-judging no longer depends on the generation code ({_PL_NAME})")
        except Exception as _e:                              # noqa: BLE001
            # Reported, not silent: re-judging would otherwise assume the payloads exist.
            print(f"[haenv] payload persist failed ({type(_e).__name__}: {_e})"
                  f" -- re-judging this batch will fall back to rebuilding it, read this "
                  f"line before reading any numbers")
        from .store import AUDIT_FILENAME, save_audits
        try:
            _n_aud = save_audits(job.cases_file.parent / AUDIT_FILENAME, audits)
            print(f"[haenv] build audit written for {_n_aud} cases, emitted or not "
                  f"({AUDIT_FILENAME})")
        except Exception as _e:                              # noqa: BLE001
            print(f"[haenv] build audit not written ({type(_e).__name__}: {_e})")
        from build import build_instance                     # kernel
        from .gates import check_batch, check_shortcut, case_features
        from .gates import check_hardwired_horizons, check_tier_surface
        from .gates import check_footprint_not_discriminative
        # Physiological footprints must not identify real symptoms.
        _fp_hits = check_footprint_not_discriminative(built)
        # Coupling-layer observability; only "this layer did not run" fails.
        from .gates import check_coupling_observable
        _fp_hits += check_coupling_observable(built)
        _batch_hits = (check_batch(built) + check_tier_surface(built)
                       + check_hardwired_horizons(built) + _fp_hits)
        for w in _batch_hits:
            print(f"[gates] batch {'failed' if w['severity'] == 'gate' else 'warning'} "
                  f"{w['kind']}: {w['detail']}")
        _batch_gate_rows = [{"kind": w["kind"], "severity": w["severity"],
                             "detail": str(w.get("detail", ""))[:400]} for w in _batch_hits]
        _batch_gate_counts: dict = {}
        for w in _batch_hits:
            _batch_gate_counts[w["kind"]] = _batch_gate_counts.get(w["kind"], 0) + 1
        # Any gate-severity hit blocks.
        from .gates import batch_gate_blockers
        _fp_gate = batch_gate_blockers(_batch_hits)
        if _fp_gate:
            print(f"[haenv] blocked by the batch-level emission gate: {[w['kind'] for w in _fp_gate]}")
            # Persist the full hit list before returning.
            batch_mod.register(job, cfg, len(job.cases), len(built), a.cmd,
                               case_digests=digests, generator=gen_tag,
                               gate_report={"blocked_by_batch_gate": [w["kind"] for w in _fp_gate],
                                            "blocked_detail": [w["detail"] for w in _fp_gate],
                                            "batch_gates": _batch_gate_rows,
                                            "batch_gate_counts": _batch_gate_counts})
            print(f"[haenv]    the batch is persisted and marked blocked_by_batch_gate -- "
                  f"`run`/`report` will refuse it (add --override-batch-gate explicitly to force it)")
            return 4  # 4 = batch-level gate failure
        # ---- discriminative and single-feature gates: per spec, not per case (warn-level) ----
        _seen_layer: dict = {"ran": False, "counts": {}, "rows": [], "error": None}
        try:
            from .gates import check_rival_discriminable, check_single_finding_solvable
            from .overlay import condition_registry, rivals_for
            from .registry import condition_findings_for_case
            _prof_all = condition_findings_for_case()
            _fw: list[dict] = []
            for _sid, _spec in sorted(condition_registry(include_draft=True).items()):
                _rv = rivals_for(_sid, _spec) or []
                if not _rv:
                    continue
                _fw += check_rival_discriminable(_sid, _rv, _prof_all.get(_sid) or {})
                _fw += check_single_finding_solvable(_sid, _rv)
            for _w in _fw:
                print(f"[gates] findings-layer warning (warn) {_w.get('kind')}: "
                      f"{_w.get('spec_id')} {_w.get('detail', '')}")
            if _fw:
                print(f"[gates] findings layer: {len(_fw)} item(s) total -- warn-level, does "
                      f"not block emission; see `check_rival_discriminable` / "
                      f"`check_single_finding_solvable`'s docstrings in gates.py for when this "
                      f"would be escalated")
            _seen_layer["ran"] = True
            for _w in _fw:
                _k = str(_w.get("kind"))
                _seen_layer["counts"][_k] = _seen_layer["counts"].get(_k, 0) + 1
            _seen_layer["rows"] = [{"kind": str(_w.get("kind")),
                                    "spec_id": str(_w.get("spec_id") or ""),
                                    "severity": str(_w.get("severity") or "warn")}
                                   for _w in _fw]
        except Exception as _e:                              # noqa: BLE001
            # Reported so that "no warning" is not read as "no problem".
            _seen_layer["error"] = f"{type(_e).__name__}: {str(_e)[:200]}"
            print(f"[gates] findings layer could not run ({type(_e).__name__}: {_e}) -- "
                  f"do not read this as the findings layer having no problems")
        rows = []
        for cid, r in built.items():
            spx, _ = build_instance(r, int(r.prediction_context["prediction_time_T"]))
            jg = ((r.adjudication or {}).get("ddx") or {}).get("join_gold")
            rows.append((cid, case_features(spx), (r.gold_drivers or [None])[0],
                         r.outcome_label, jg))  # the join gold is scanned too
        # ---- labels this task type scores; unscored labels are exempt from the shortcut gate ----
        from .gates import check_outcome_derivable
        _specs = {c.case_id: c for c in job.cases}
        _oc_na = [any(w["kind"] == "outcome_rule_not_applicable"
                      for w in check_outcome_derivable(r, _specs.get(cid)))
                  for cid, r in built.items()]
        _graded = {"driver", "outcome", "join_gold"}
        # The information-insufficient tier does not score join type.
        _all_insuff = bool(built) and all(
            bool(((getattr(_specs.get(cid), "latent", None) or {}).get("ddx_insufficient")))
            for cid in built)
        if _all_insuff:
            _graded.discard("join_gold")
            print(f"[gates] A5 scored-label set: {sorted(_graded)} -- join_gold removed, "
                  f"because all {len(built)} case(s) are in the information-insufficient "
                  f"tier (this tier does not score join type; hits are still reported, marked "
                  f"ungraded)")
        if _oc_na and all(_oc_na):
            _graded.discard("outcome")
            print(f"[gates] A5 scored-label set: {sorted(_graded)} -- outcome removed, "
                  f"because all {len(_oc_na)} case(s) carry outcome_rule_not_applicable "
                  f"(hits are still reported, marked ungraded)")
        # Unscannable: the gold is constant, so the dimension could not be scanned.
        n_hit = {"clinical": 0, "nonclinical": 0, "ungraded": 0, "unscannable": 0}
        _ev_types = {str((r.prediction_context or {}).get("target_event_type") or "")
                     for r in built.values()}
        _ev_type = next(iter(_ev_types)) if len(_ev_types) == 1 else None
        _a5_unpowered_strata = 0
        _a5_kinds: dict = {}
        for w in check_shortcut(rows, graded_targets=_graded, event_type=_ev_type):
            n_hit[w.get("class", "nonclinical")] = n_hit.get(w.get("class", "nonclinical"), 0) + 1
            _kc = _a5_kinds.setdefault(str(w.get("kind")), {})
            _kcls = str(w.get("class", "nonclinical"))
            _kc[_kcls] = _kc.get(_kcls, 0) + 1
            if w.get("powered") is False:
                _a5_unpowered_strata += 1
            print(f"[gates] batch warning {w['kind']}: {w['detail']}")
        if sum(n_hit.values()):
            # Hits on a registered evidence stream are intended inferences and do not block; nonclinical
            # hits refuse emission.
            print(f"[gates] A5v2: clinical {n_hit['clinical']} (the question is meant to be "
                  f"inferable this way, not blocked)"
                  f" · ungraded {n_hit.get('ungraded', 0)} (this task type does not score this "
                  f"label, not blocked but recorded)"
                  f" · chance {n_hit.get('chance', 0)} (not corrected for multiple "
                  f"comparisons, not a verdict; the `[...significance not reached]` prefix "
                  f"in each detail is a classification, not a conclusion)"
                  f" · unscannable {n_hit.get('unscannable', 0)} (this dimension could "
                  f"not be scanned, do not read it as \"no shortcut\")"
                  f" · nonclinical {n_hit['nonclinical']} (a defect, emission refused)")
            _acc = sum(n_hit.values())
            if _acc != sum(n_hit.get(k, 0) for k in
                           ("clinical", "ungraded", "chance", "unscannable", "nonclinical")):
                # A hit class this summary does not know about.
                print(f"[gates] the A5v2 summary line does not cover all {_acc} hit(s) "
                      f"-- an unregistered tier appeared: "
                      f"{sorted(set(n_hit) - {'clinical', 'ungraded', 'chance', 'unscannable', 'nonclinical'})}")

        # ---- per-case gate output, collected from the audits and persisted (launched and blocked
        # separately) ----
        _gr: dict = {"per_case_kinds": {}, "counts": {},
                     "per_case_kinds_blocked": {}, "counts_blocked": {}}
        for _a in (audits or []):
            if not isinstance(_a, dict):
                continue
            _kinds = [str(w).split(":", 1)[0] for w in (_a.get("gate_warnings") or [])]
            if not _kinds:
                continue
            _cid = str(_a.get("case_id", "?"))
            _pk, _ck = (("per_case_kinds", "counts") if _a.get("emitted")
                        else ("per_case_kinds_blocked", "counts_blocked"))
            _gr[_pk][_cid] = _kinds
            for _k in _kinds:
                _gr[_ck][_k] = _gr[_ck].get(_k, 0) + 1
        # How many gate kinds reported anything on this batch.
        _gr["n_gate_kinds_reporting"] = len(set(_gr["counts"]) | set(_gr["counts_blocked"]))
        from .gates import blocked_counts as _bcnt, blocked_reasons as _breas
        _blocked = _breas(audits)
        _gr["blocked_kinds"] = _blocked  # per case: why it was not emitted
        _gr["n_blocked"] = len(_blocked)
        # `block_reason_counts`: why cases were not emitted; `warn_counts_on_blocked`: warnings on
        # blocked cases.
        _gr["block_reason_counts"] = _bcnt(audits)
        _gr["blocked_counts"] = _gr["block_reason_counts"]  # alias, kept for compatibility
        _gr["warn_counts_on_blocked"] = _gr["counts_blocked"]
        # The discard ledger, so the discard-and-reinject loop is visible.
        from .gates import dropped_items as _drit, dropped_rates as _drra
        _gr["dropped_items"] = _drit(audits)
        _gr["dropped_rates"] = _drra(audits)  # per item; a rate of 1.00 means a structural drop
        _hi = [k for k, v in _gr["dropped_rates"].items() if (v.get("rate") or 0) >= 0.5]
        if _hi:
            print(f"[gates] {len(_hi)} item kind(s) were dropped in over half of the "
                  f"emitted cases: "
                  + " · ".join(f"{k} {_gr['dropped_rates'][k]['n']}/"
                               f"{_gr['dropped_rates'][k]['of']}" for k in _hi)
                  + " -- dropping lets validation pass, but the pack is missing this content")
        if _blocked:
            print(f"[gates] {len(_blocked)} case(s) not emitted, reasons recorded in "
                  f"batch.json: {_gr['blocked_counts']}")
        _gr["a5_shortcut"] = dict(n_hit)
        _gr["a5_shortcut_kinds"] = _a5_kinds
        _gr["batch_gates"] = _batch_gate_rows
        _gr["batch_gate_counts"] = _batch_gate_counts
        _gr["seen_layer"] = _seen_layer  # `ran` separates 'did not run' from 'zero hits'
        _gr["batch_gate_level_kinds"] = sorted({w["kind"] for w in _batch_hits
                                                if w["severity"] == "gate"})
        # A nonclinical shortcut hit refuses emission; an unpowered stratum is only recorded.
        _n_scanned = len(rows)
        _a5_underpowered = _a5_unpowered_strata > 0
        if _a5_underpowered:
            print(f"[gates] A5 {_a5_unpowered_strata} tier(s) are unpowered "
                  f"({_n_scanned} case(s) in this batch) -- even a perfect split in those tiers "
                  f"would not survive multiple-comparisons correction, so they are recorded as "
                  f"unpowered, not read as \"no shortcut\". There are only two ways to gain "
                  f"power: add cases, or narrow the feature family.")
        _gr["a5_blocked"] = bool(n_hit.get("nonclinical"))
        # Single batch-level failure marker; `a5_blocked` is kept for compatibility.
        if _gr["a5_blocked"]:
            _gr.setdefault("blocked_by_batch_gate", []).append("a5_nonclinical_shortcut")
        _gr["a5_underpowered"] = _a5_underpowered
        _gr["a5_unpowered_strata"] = _a5_unpowered_strata
        _gr["a5_n_scanned"] = _n_scanned
        _gr["note"] = ("per-case gate kinds are persisted. All per-case gates "
                       "are collected in full from `audits['gate_warnings']` (counts / "
                       "counts_blocked); the three batch-level branches (check_batch, "
                       "hardwired_horizons, footprint/coupling) are in `batch_gates`; the "
                       "findings layer's checks are in `seen_layer`; A5 by kind is in "
                       "`a5_shortcut_kinds` (the class buckets stay in `a5_shortcut`).")
        if _gr["counts"]:
            print(f"[gates] per-case gate hits recorded in batch.json: {_gr['counts']}")

        # ---- idempotence gate: compare against the previous batch of the same job ----
        from .store import check_idempotent
        prev = batch_mod.previous_digests(job)
        if prev:
            diff = check_idempotent(built, prev[1])
            # Drift fails only when the generation code, knobs, job file and generator are unchanged,
            # all cases were rebuilt and no model call bypassed the cache; otherwise it is reported.
            _prev_meta_p = job.cases_file.parent.parent / prev[0] / "batch.json"
            try:
                _prev_meta = json.loads(_prev_meta_p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                _prev_meta = {}
            _same, _why_not = batch_mod.same_generation_inputs(_prev_meta, job, cfg, gen_tag)
            _strict = (_same and not a.regen and not a.cases
                       and (dispatch is None or dispatch.n_calls == 0))
            if diff and _strict:
                _gr["idempotency_drift"] = {
                    "previous_batch": prev[0], "n_drifted": len(diff), "n_built": len(built),
                    "cases": [d["case"] for d in diff]}
                _gr.setdefault("blocked_by_batch_gate", []).append("idempotency_drift")
            elif diff:
                print(f"[haenv]    idempotency: report only, the generation inputs differ "
                      f"from batch {prev[0]} ({_why_not or 'regen, a case subset or live model calls'})")
            if diff:
                # Whether the question text drifted or only the truth side (the latter only needs a re-score).
                from .store import drift_scope
                _prev_dir = job.cases_file.parent.parent / prev[0]
                _sc = drift_scope(built, _prev_dir / job.cases_file.name)
                _tag = {"question": "the question text changed too -> persisted "
                                    "responses are void, must re-run",
                        "truth_only": "🟡 the question text is byte-identical, only the "
                                      "truth side changed -> responses are still valid, just "
                                      "`recompute_judges --full` to re-score",
                        "none": "(the fingerprint changed but neither the question text nor "
                                "the truth side did -- check for a serialization difference)",
                        "undecidable": f"cannot be determined ({_sc.get('why')}) -- treat as the worst case"}
                print(f"[haenv] idempotency gate: {len(diff)}/{len(built)} case(s) have "
                      f"drifted relative to batch {prev[0]}"
                      f" ({', '.join(d['case'] for d in diff[:4])}{'…' if len(diff) > 4 else ''})")
                print(f"[haenv]    drift scope: {_tag.get(_sc['scope'], _sc['scope'])}"
                      f" (question changed in {len(_sc.get('question_changed') or [])} case(s) / "
                      f"only truth changed in {len(_sc.get('truth_only_changed') or [])} case(s))")
            else:
                print(f"[haenv] idempotency gate: identical to batch {prev[0]} for every case ✓ ({len(built)} case(s))")
        # Cohort-level glucose relations (R1 ceiling, R8 co-direction) on the emitted cases; warn-level,
        # `None` when the batch is too small.
        from . import relations as _rel_mod
        _a1c_m, _fbg_m, _da, _df = [], [], [], []
        for _raw in built.values():
            _ld = getattr(_raw, "longitudinal_data", None) or {}
            _a = [q["value"] for q in _ld.get("HbA1c") or [] if isinstance(q.get("value"), (int, float))]
            _f = [q["value"] for q in _ld.get("fasting_glucose") or [] if isinstance(q.get("value"), (int, float))]
            if len(_a) >= 2 and len(_f) >= 2:
                _a1c_m.append(sum(_a) / len(_a)); _fbg_m.append(sum(_f) / len(_f))
                _da.append(_a[-1] - _a[0]); _df.append(_f[-1] - _f[0])
        _rc = {c.relation: {"ok": c.ok, "detail": c.detail}
               for c in (_rel_mod.cohort_a1c_fbg_ceiling(_a1c_m, _fbg_m),
                         _rel_mod.cohort_glucose_scales_codirectional(_da, _df))}
        _gr["relations_cohort"] = _rc
        for _k, _v in _rc.items():
            print(f"[haenv] {_k}: {'✓' if _v['ok'] else ('⚪ not measured' if _v['ok'] is None else '✗')} "
                  f"{_v['detail']}")
        batch_mod.register(job, cfg, len(job.cases), len(built), a.cmd,
                           case_digests=digests, generator=gen_tag, gate_report=_gr)
        print(f"[haenv] case bodies saved: {len(built)} case(s) -> {_rel(job.cases_file)}"
              f" ({job.cases_file.stat().st_size / 1024:.0f} KB · per-case sha256 recorded in batch.json)")
        if _gr["a5_blocked"]:
            print(f"[haenv] emission refused: A5 nonclinical shortcut hits "
                  f"{n_hit['nonclinical']} -- this batch's questions can be answered without "
                  f"reasoning. The batch is persisted and marked a5_blocked; fix the questions "
                  f"and re-run.")
            return 4
        if _gr.get("idempotency_drift"):
            _d = _gr["idempotency_drift"]
            print(f"[haenv] refused: {_d['n_drifted']} cases differ from batch "
                  f"{_d['previous_batch']}, which was built from the same generation code, "
                  f"knobs, job file and generator. Generation is not reproducible; look for "
                  f"shared mutable state before blaming the model (a case built alone and in "
                  f"the batch should match). The batch is on disk, marked idempotency_drift.")
            return 4

    if a.cmd == "build":
        # One line per case; the full record is `audit.jsonl`.
        for x in audits:
            if x.get("emitted"):
                what = (f"emitted · solver-visible signals {x.get('solver_visible_signals', '—')}"
                        f" · evidence {x.get('solver_visible_evidence', '—')}"
                        f" · noise {len(x.get('noise_applied') or [])}")
            else:
                what = "blocked · " + (", ".join(x.get("post_noise_conflicts") or [])
                                       or str(x.get("premise_error") or "see audit"))
            print(f"   {x.get('case_id')}: {what}")
        _aud = job.cases_file.parent / AUDIT_FILENAME if not a.limit else None
        if _aud is not None and _aud.is_file():
            print(f"[haenv] full per-case audit: {_rel(_aud)}")
        return 0

    # ---- generate and verify only, with no model calls ----
    if a.cmd == "verify":
        from .verify_report import write_verify_report
        reports = [x["_verify_report"] for x in audits if x.get("_verify_report")]
        md, jsonl = write_verify_report(job, audits, reports)
        n_bad = sum(r.get("n_bad_items", 0) for r in reports)
        n_leak = sum(0 if r["solver_text"]["ok"] else 1 for r in reports)
        print(f"[haenv] per-item verification: {sum(r.get('n_items', 0) for r in reports)} item(s) · "
              f"failed {n_bad} · text leaked in {n_leak} case(s)")
        print(f"[haenv] verify report -> {md}")
        print(f"[haenv] per-item ledger -> {jsonl}")
        return 0 if (n_bad == 0 and n_leak == 0) else 3

    return _eval_and_report(a, job, cfg, built, audits, run_eval, write_report)


def _fine_vintage_gate(job, rows, *, refuse: bool) -> bool:
    """Fine-fingerprint gate: when the scoring stamp on disk is not current, did any part
    these rows actually used change?

    If none did, resuming is allowed (recompute before publishing). If some did, or the
    batch recorded no parts, the resume is refused so one `eval.jsonl` never mixes two
    scoring versions. Returns ``False`` to refuse.
    """
    from .anchor import judging_fp_cached, judging_vintages, parts_compatible
    if not rows:
        return True
    cur = judging_fp_cached()
    stale = {k: n for k, n in judging_vintages(rows).items() if k != cur}
    if not stale:
        print(f"[haenv] scoring vintage: all {len(rows)} row(s) on disk carry today's stamp `{cur}` ✓ (fine fingerprint not needed)")
        return True
    print(f"[haenv] coarse fingerprint is not today's vintage: today `{cur}` · on disk {stale}")
    meta_p = job.results_dir / "batch.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.is_file() else {}
    before = meta.get("judge_parts")
    if not isinstance(before, dict) or not before:
        ok, changed = False, ["<batch.json has no `judge_parts`>"]
        print("[haenv]    fine fingerprint cannot be determined: this batch's "
              "`batch.json` did not record `judge_parts` (an older batch, from before "
              "per-part stamping). \"this part was not recorded last time\" and \"this part "
              "did not change\" must be kept apart -- the former is unknown, and unknown "
              "goes to recompute, not to a pass.")
    else:
        ok, changed = parts_compatible(rows, before)
    if ok:
        print(f"[haenv]    🟡 fine fingerprint: not one part these rows used has changed "
              f"({len(before)} part(s) recorded at the time) => recompute, not re-run. "
              f"Persisted responses are still valid; the coarse fingerprint's staleness here "
              f"is over-invalidation.")
        print("[haenv]    proceeding; before publishing the board run "
              "`uv run python tools/recompute_judges.py --full` once to bring the old rows up "
              "to today's stamp, or `assert_publishable` will refuse for mixed versions.")
        return True
    print(f"[haenv]    {len(changed)} used part(s) changed: "
          f"{changed[:6]}{'…' if len(changed) > 6 else ''}")
    if not refuse:
        print("[haenv]    (the report is still rendered -- same rule as `assert_publishable`'s "
              "\"the rendering layer does not refuse\"; but this batch's numbers cannot be "
              "published, the board-publishing gate will refuse them.)")
        return True
    print("[haenv] refusing to resume: continuing would append two scoring versions "
          "into the same `eval.jsonl`, which the board-publishing gate would later reject. "
          "Two options:\n"
          "        · For a clean batch of new readings: add `--fresh` (opens a new "
          "timestamped directory, the old batch is left untouched)\n"
          "        · To keep the responses you already paid for: first run "
          "`uv run python tools/recompute_judges.py --full` (recomputes the old rows to "
          "today's vintage), then resume")
    return False


def _eval_and_report(a, job, cfg, built, audits, run_eval, write_report) -> int:
    """Evaluate (incremental JSONL, resumable), then report. ``built`` is freshly generated
    or read back from the batch.
    """
    if a.cmd == "run":
        # ---- fine-fingerprint gate before resuming ----
        if not a.fresh and job.results_file.exists():
            from .evaluate import load_rows as _load_rows_pre
            if not _fine_vintage_gate(job, _load_rows_pre(job.results_file), refuse=True):
                return 5
        # ---- false-premise probes: polarity assigned deterministically per case ----
        from . import evaluate as _ev_mod
        _pm = _ev_mod.assign_premises(built)
        if _pm:
            _nf = sum(1 for v in _pm.values() if v.get("polarity") == "false")
            print(f"[haenv] Q-side false-premise probes: {len(_pm)}/{len(built)} case(s) carry "
                  f"a premise (false {_nf} · true {len(_pm) - _nf}) -- a true premise is a "
                  f"false-alarm control; raising a conflict on it is a false positive")
        # ---- no-op probes, then computable phrasings rotated across three classes; one
        # call, shared with `tools/recompute_judges.py` (the quant probe avoids the noop
        # stream, and `trend` is built on the window each case is answered on) ----
        _nm, _qm = _ev_mod.assign_qside_probes(
            built, answer_t=_ev_mod.answer_windows(job, built, job.results_file.parent))
        if _nm:
            _na = sum(1 for v in _nm.values() if not v.get("truth_present"))
            print(f"[haenv] Q-side no-op probes: {len(_nm)}/{len(built)} case(s) "
                  f"(asking about something absent {_na} · asking about something present "
                  f"{len(_nm) - _na}) -- claiming insufficient data on a signal that is "
                  f"present is a false positive")
        # Gold list for the oracle stub only; never rendered.
        _om = _ev_mod.assign_oracle_gold(built)
        _ow = _ev_mod.assign_oracle_warranted(built)
        print(f"[haenv] oracle gold list: {len(_om)}/{len(built)} case(s) · "
              f"oracle-reviewed gold {len(_ow)}/{len(built)} case(s) -- read only by the "
              f"oracle stub, never enters any question (an upper/lower bound row, not a "
              f"solution)")
        if _qm:
            import collections as _co
            _kc = dict(_co.Counter(v["kind"] for v in _qm.values()))
            print(f"[haenv] Q-side data-check questions: {len(_qm)}/{len(built)} case(s) · "
                  f"distribution across the three classes {_kc} -- truth is computed by code, "
                  f"the answer is an enum/integer => anchor-exempt")
        from .settings import append_record
        Path(job.results_dir).mkdir(parents=True, exist_ok=True)
        append_record(job.results_dir, cfg, config_layers())
        if a.offline:
            rows = run_eval(job, cfg, built, resume=not a.fresh)
        else:
            rows = run_eval(job, cfg, built, resume=not a.fresh,
                            budget_ledger=a.judge_budget_ledger, budget_usd=a.judge_budget_usd)
    else:  # report: read the existing JSONL
        from .evaluate import load_rows
        p = job.results_file
        if not p.exists():
            print(f"[haenv] no results file {p}; run `run` first."); return 2
        rows = load_rows(p)
        _fine_vintage_gate(job, rows, refuse=False)

    semantic = None
    if a.cmd == "run":
        from .semantic_pipeline import default_after_run
        if a.limit and not a.offline:
            # `--limit` saves no cases.jsonl, so the batch cannot be prepared for judging.
            semantic = _limited_run_semantic(rows)
        else:
            semantic = default_after_run(job.results_file.parent, cfg,
                                         offline=bool(a.offline),
                                         budget_ledger=getattr(a, "judge_budget_ledger", None),
                                         budget_usd=getattr(a, "judge_budget_usd", None))
        print(f"[haenv] semantic judging: {semantic['status']}")
    if not a.offline:
        from .semantic_report import view_for_batch
        from .semantic_rubric import load_policy
        _semantic_dir = (a.semantic_run if a.semantic_run is not None else
                         job.results_file.parent / "semantic" / load_policy()["version"])
        if a.semantic_run is None and not (_semantic_dir / "manifest.json").is_file():
            _semantic_dir = None
        rows = view_for_batch(rows, job.results_file.parent, _semantic_dir)
        if semantic is None:
            semantic = _report_semantic_state(rows, _semantic_dir)
            print(f"[haenv] semantic judging: {semantic['status']} · real-model cells {semantic['cells']} "
                  f"· {semantic['states']}")
    out = write_report(job, rows, built, audits, cfg)
    if semantic is not None:
        import json as _semantic_json
        _state_file = Path(out).parent / "semantic-status.json"
        _state_file.write_text(_semantic_json.dumps(semantic, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    _catalog_gate(job, out)
    _noise_floor_gate(job, rows, out)
    print(f"[haenv] report -> {out}")
    print(f"[haenv] details -> {job.results_file}")
    if _semantic_exit_code(semantic):
        print("[haenv] semantic results are incomplete; legacy code scores are not a substitute")
    return _semantic_exit_code(semantic)


_NO_ANSWER = ("missing_response", "empty_response", "unparseable_response")


def _report_semantic_state(rows: list[dict], semantic_dir) -> dict:
    """Semantic state of the rows `report` shows.

    Offline reference solvers are never judged, so a batch without a real-model row
    has nothing to judge: `not_applicable`. Otherwise the rule `default_after_run`
    applies after `run`: `completed` only when every real-model cell is resolved.
    Pending, unresolved (no 2+1 majority) and unanswered cells are counted apart;
    each of them leaves the semantic scores a partial measurement of the batch.
    """
    from collections import Counter
    from .baselines import BASELINE_NAMES
    live = [r for r in rows if r["solver"] not in BASELINE_NAMES]
    states = Counter(r["semantic"]["status"] for r in live)
    status = ("not_applicable" if not live else
              "completed" if set(states) == {"resolved"} else "incomplete")
    return {"status": status, "output": str(semantic_dir) if semantic_dir is not None else None,
            "cells": len(live), "states": dict(sorted(states.items())),
            "pending": states["pending"], "unresolved": states["unresolved"],
            "no_answer": sum(states[s] for s in _NO_ANSWER)}


def _limited_run_semantic(rows: list[dict]) -> dict:
    """Semantic state of an online `run --limit N`: judging is skipped (nothing is saved for the
    sampled cases), and the real-model cells are reported as unjudged."""
    from .baselines import BASELINE_NAMES
    live = [r for r in rows if r["solver"] not in BASELINE_NAMES]
    return {"status": "not_run_limited" if live else "not_applicable",
            "reason": "limited run: cases.jsonl not saved, semantic judging skipped",
            "cells": len(live), "semantic_scores_available": False}


def _semantic_exit_code(semantic: dict | None) -> int:
    """6 when a real-model batch's semantic judgment is incomplete (including a limited run whose
    judging was skipped); 0 otherwise."""
    return 6 if semantic is not None and semantic["status"] in ("incomplete", "not_run_limited") else 0


def _floor_is_for_this_batch(job) -> bool:
    """Whether the newest reliability artefact was measured on this batch's questions
    (unreadable counts as no).
    """
    import json as _j
    from pathlib import Path as _P
    d = getattr(job, "results_dir", None)
    if d is None:
        return False
    cands = sorted(_P(d).parent.glob("reliability-*.json"), reverse=True)
    if not cands:
        return False
    try:
        rep = _j.loads(cands[0].read_text(encoding="utf-8"))
    except Exception:                                                # noqa: BLE001
        return False
    return str(rep.get("prompt_source_batch") or "") == str(getattr(job, "batch", ""))


def _gate_unreadable(out, what: str, err: Exception) -> None:
    """Print that a gate could not be evaluated and record it in `GATE-UNREADABLE.md`
    next to the report, so it is not mistaken for a pass.
    """
    print(f"[haenv] {what} could not be read ({type(err).__name__}), this gate did not "
          f"return a verdict this time -- a gate that cannot be read reliably must not be "
          f"treated as a pass")
    try:
        from pathlib import Path as _P
        note = _P(out).parent / "GATE-UNREADABLE.md"
        prev = note.read_text(encoding="utf-8") if note.is_file() else \
            "# A gate did not return a verdict this time\n\nThis is not a \"pass\". Listed below.\n"
        note.write_text(prev + f"\n* **{what}** -- `{type(err).__name__}: "
                               f"{str(err)[:160]}`\n", encoding="utf-8")
        print(f"[haenv] note -> {note}")
    except OSError as e:
        print(f"[haenv] could not write the note ({e})")


def _noise_floor_gate(job, rows, out) -> None:
    """Mark a report whose batch has no noise floor.

    Without repeated runs, differences between adjacent ranks cannot be read. The report
    is still written, and `NOISE-FLOOR-ABSENT.md` is placed next to it. Offline stub
    batches are exempt. Lives here rather than in the report module, which is in the
    frozen scoring segment; this gate changes no score.
    """
    try:
        from .analytics import noise_floor_for
        from .report import real_solver_pool
        pool, _ = real_solver_pool(rows)
        if not pool:  # stub-only batch
            return
        nz = noise_floor_for(job)
    except Exception as e:                                           # noqa: BLE001
        # Persist the error too, so it is not read as a pass.
        _gate_unreadable(out, "noise floor", e)
        return
    # The floor must come from this batch's questions.
    if nz and not _floor_is_for_this_batch(job):
        print(f"[haenv] the noise floor on disk is not this batch's -- it belongs to "
              f"another batch's questions and does not count for this one. Rank-adjacent "
              f"differences still cannot be read.")
        nz = {}
    if nz:
        print(f"[haenv] ✅ noise floor: {len(nz)} dimension(s) passed (k>=3, this batch's questions)")
        return
    print(f"[haenv] this batch has no noise floor (k=1) -- differences between "
          f"rank-adjacent models cannot be read, only the broad picture can. To read "
          f"differences: `.venv/bin/python tools/reliability_passk.py "
          f"{getattr(job, 'path', None) or job.job_id} --from-batch {job.batch} --k 3`")
    try:
        from pathlib import Path
        note = Path(out).parent / "NOISE-FLOOR-ABSENT.md"
        note.write_text(
            f"# this batch has no noise floor (k=1)\n\n"
            f"pack `{job.job_id}` · batch `{job.batch}` · {len(pool)} real model(s).\n\n"
            f"**Differences between rank-adjacent models cannot be read**, only the broad "
            f"picture can: with a single pass per model there is nothing to compare a rank "
            f"gap against, and adjacent ranks are false precision "
            f"(see the header of `haenv/ranking.py`).\n\n"
            f"To add a floor:\n\n```\n.venv/bin/python tools/reliability_passk.py "
            f"<job.yaml> --from-batch {job.batch} --k 3\n```\n\n"
            f"`k<3` does not count: fewer than three repeats underestimates the noise floor.\n\n"
            f"gate `haenv/cli.py:_noise_floor_gate`.\n",
            encoding="utf-8")
        print(f"[haenv] note -> {note}")
    except OSError as e:
        print(f"[haenv] could not write the note ({e})")


def _catalog_gate(job, out) -> None:
    """Warn when rendering a pack marked not publishable in the catalogue, and write
    `NOT-FOR-RELEASE.md` next to the report. Packs absent from the catalogue are not flagged.
    Lives here rather than in the report module (frozen scoring segment).
    """
    job_id = str(getattr(job, "job_id", job))
    # Unreviewed conditions are reported for every pack.
    try:
        import json as _j

        from .catalog import review_pending_in
        _cf = getattr(job, "cases_file", None)
        if _cf is not None and _cf.is_file():
            _rows = [_j.loads(x) for x in _cf.open(encoding="utf-8") if x.strip()]
            _pend = review_pending_in(_rows)
            _n = sum(1 for r in _rows
                     if __import__("haenv.catalog", fromlist=["x"]).case_spec_id(r) in set(_pend))
            if _pend:
                print(f"[haenv] this batch contains {len(_pend)} clinically unreviewed "
                      f"condition(s) · {_n}/{len(_rows)} case(s): {_pend} -- this sentence "
                      f"must be quoted alongside any reading from this batch")
    except Exception as e:                                           # noqa: BLE001
        _gate_unreadable(out, "unreviewed-condition tally", e)
    try:
        from .catalog import is_publishable, pack_entry
        ok = is_publishable(job_id)
    except Exception as e:                                           # noqa: BLE001
        # Persist the error too.
        _gate_unreadable(out, "pack catalogue", e)
        return
    if ok is not False:
        return
    why = str((pack_entry(job_id) or {}).get("why") or "").strip()
    print(f"[haenv] readings from `{job_id}` must not go into any external materials. {why}")
    try:
        from pathlib import Path
        note = Path(out).parent / "NOT-FOR-RELEASE.md"
        note.write_text(
            f"# readings from this directory must not go external\n\npack `{job_id}`.\n\n{why}\n\n"
            f"basis: `publishable: false` in `registry/pack_catalog.yaml`; "
            f"gate `haenv/cli.py:_catalog_gate`.\n",
            encoding="utf-8")
        print(f"[haenv] note -> {note}")
    except OSError as e:
        print(f"[haenv] could not write the note ({e})")


if __name__ == "__main__":
    raise SystemExit(main())
