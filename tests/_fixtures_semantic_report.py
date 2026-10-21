"""Primary reports use semantic values, never a proxy or imputed judge result."""

from copy import deepcopy

from haenv.scoring.semantic.report import overlay_rows

from haenv.scoring.semantic.report import report_section

from haenv.scoring.analysis.ranking import rank_ddx

from haenv.scoring.quantities import resolve

import json

import pytest

def rows():
    return [{"case": f"case-{i}", "solver": model, "geometry": "single",
             "gold_kind": "ddx:unified", "overall": "SCORED", "gates": [],
             "judging_sha16": "legacy", "dx_hit": 1., "noop_ok": 1.,
             "tests_recall": 1., "wk_tests_recall_last": 1., "tests_precision": 1.,
             "quant_ok": 1.}
            for model in ("gpt-6-luna", "deepseek-v4-pro", "minimax-m3") for i in range(20)]

def results(raw):
    return {(r["case"], r["solver"]): {"status": "resolved", "summary": {
        "metrics": {"dx_hit": 0., "noop_ok": 1., "tests_recall": .5,
                    "tests_precision": 1., "tests_f1": 2 / 3,
                    "disc_recall": None, "action_consistency": 1.},
        "not_applicable": {"disc_recall": "no rival"},
        "coverage": {key: {"expected": 1, "resolved": 1} for key in
                     ("dx_hit", "noop_ok", "tests_recall", "tests_precision", "action_consistency")},
        "version": "semantic-v1"}} for r in raw}

def saved_run(tmp_path):
    from dataclasses import asdict
    from haenv.scoring.semantic.pipeline import digest
    from haenv.scoring.semantic.judge import JudgeTask
    from haenv.scoring.semantic.judge import JudgeReply
    from haenv.scoring.semantic.judge import evaluate_consensus
    from haenv.scoring.semantic.rubric import summarize_verdicts
    batch = tmp_path / "source";batch.mkdir()
    original = batch / "responses.jsonl";original.write_text('original answer\n')
    out = tmp_path / "semantic";out.mkdir();(out / "results").mkdir()
    task = JudgeTask("test", ("dx",), "prompt", "A", "test", "v1")
    rubric = {"groups": {"dx_hit": ["dx"]}, "not_applicable": {}, "version": "v1"}
    source = {"batch": str(batch), "case": "case", "solver": "model"}
    record = {"key": "one", "source": source, "status": "ready", "task": asdict(task),
              "task_sha256": task.fingerprint, "rubric": rubric}
    tasks = out / "tasks.jsonl";tasks.write_text(json.dumps(record) + "\n")
    manifest = {"sealed": True, "run_id": "test", "code": {"judging_sha16": "semantic"},
                "tasks_sha256": digest(tasks),
                "sources": [{"batch": str(batch), "files": {"responses.jsonl": digest(original)}}]}
    (out / "manifest.json").write_text(json.dumps(manifest))
    raw = json.dumps({"verdicts": {"dx": {"label": "yes", "reason": "A", "evidence": ["A"]}}})
    consensus = evaluate_consensus(task, lambda request: JudgeReply(raw, False, None))
    (out / "samples").mkdir()
    (out / "samples/one.jsonl").write_text(
        "".join(json.dumps(sample) + "\n" for sample in consensus["samples"]))
    result = {"source": source, "consensus": consensus,
              "summary": summarize_verdicts(consensus["verdicts"], rubric)}
    path = out / "results/one.json";path.write_text(json.dumps(result))
    return out, batch, path


__all__ = [name for name in globals() if not name.startswith("__")]
