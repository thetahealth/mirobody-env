"""Parse command options without executing a job or loading a provider."""
from __future__ import annotations

import argparse
from pathlib import Path


def parser_for(maintenance: set[str], twin_ops: list[str]) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="haenv", description="Health agent environment: generate, evaluate, and report in one pipeline")
    ap.add_argument("cmd", choices=["run", "build", "verify", "report", "twin"]
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
    ap.add_argument("--no-semantic", action="store_true", help="skip semantic judging; report code scores only")
    ap.add_argument("--judge-budget-usd", help="cap for a new ledger; omit to reuse the pinned ledger cap")
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
                         "Deterministic generation runs on processes: 0 = job `gen_workers`, else min(16, cores); 1 = serial")
    ap.add_argument("--regen", action="store_true",
                    help="ignore the question-generation LLM cache and force fresh model calls (default: hits cases/_llm_cache/)")
    ap.add_argument("--workers", type=int, default=0,
                    help="cells in flight per model (0 = use `config.eval.workers`; 1 = serial). Each model's "
                         "pool is min(this value, `backend_limits[backend]`, its cell count), and the models "
                         "on one backend share that backend's cap; rows are identical to a serial run apart "
                         "from timing fields. Under `--pooling elastic` only 1 (serial) matters: pools are "
                         "sized by `config.eval.elastic_max`")
    ap.add_argument("--geometry", choices=["query", "single"], default=None,
                    help="how a pack case (P2, P3, P4, M2) is put to a model (default `config.query.geometry`, "
                         "query): `query` = the record is read through tools, `single` = the whole record in "
                         "one prompt (smoke runs); `--geometry X` is `--set query.geometry=X`")
    ap.add_argument("--pooling", choices=["model", "backend", "elastic"], default=None,
                    help="scheduling (default `config.eval.pooling`, elastic): `elastic` = per-model pools "
                         "sized to the backend cap (at most `config.eval.elastic_max`) with the backend "
                         "slot going to the model that needs the longest to finish; `model` = one fixed "
                         "pool per model, longest expected cell first; `backend` = one pool per backend "
                         "in task order")
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
    # ---- twin records (`haenv twin`, see haenv/world/generation/twins.py) ----
    ap.add_argument("--op", help="`twin` only: the one-fact change, one of " + ", ".join(twin_ops))
    ap.add_argument("--from-batch", type=Path, metavar="DIR",
                    help="`twin` only: the built batch (directory or batch prefix) whose records are twinned")
    ap.add_argument("--out", type=Path, metavar="DIR", help="`twin` only: where the twin and original jobs are written")
    ap.add_argument("--items", metavar="ID,ID",
                    help="`twin` only: the items to twin (each must be eligible; default: every eligible item)")
    ap.add_argument("--job-id", metavar="ID", help="`twin` only: job id of the twin job (default <job>-twin-<op>)")
    return ap
