"""Adaptive semantic judging must count independent votes, not cache hits."""

import json

import pytest

from haenv.scoring.semantic.judge import JudgeReply

from haenv.scoring.semantic.judge import JudgeTask

from haenv.scoring.semantic.judge import evaluate_consensus

def task():
    return JudgeTask(run_id="run-a", item_ids=("dx", "reason"),
                     prompt="Judge the anonymous answer under fixed rubric v1.",
                     answer_text="Possible A. Evidence E1 supports A.",
                     judge_model="test-judge", rubric_version="v1")

def reply(dx="yes", reason="yes"):
    return json.dumps({"verdicts": {k: {"label": v, "reason": "rubric applied",
                                        "evidence": ["Possible A."]}
                                   for k, v in (("dx", dx), ("reason", reason))}})

def dispatcher(outputs):
    calls = []
    def call(request):
        calls.append(request)
        value = outputs[len(calls) - 1]
        if isinstance(value, Exception):
            raise value
        if isinstance(value, JudgeReply):
            return value
        return JudgeReply(raw=value, cached=False, usage=None)
    return calls, call


__all__ = [name for name in globals() if not name.startswith("__")]
