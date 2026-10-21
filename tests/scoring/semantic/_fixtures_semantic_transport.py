"""The semantic transport binds fresh calls to verified pricing and the shared cap."""

from decimal import Decimal

import pytest

from haenv.scoring.semantic.budget import BudgetExceeded

from haenv.scoring.semantic.budget import BudgetLedger

from haenv.scoring.semantic.judge import JudgeRequest

from haenv.scoring.semantic.transport import PricedJudge

from haenv.scoring.semantic.transport import PriceSchedule

def metadata():
    return {"id": "openai/gpt-6-luna", "context_length": 100000,
            "supported_parameters": ["reasoning", "max_tokens", "response_format", "structured_outputs"],
            "pricing": {"prompt": "0.0000001", "completion": "0.0000005",
                        "overrides": [{"min_prompt_tokens": 272000,
                                       "prompt": "0.0000002", "completion": "0.00000075"}]}}

class Solver:
    model = "openai/gpt-6-luna"
    reasoning_effort = "high"
    max_tokens = 12000
    pool = None

    def __init__(self, usage):
        self.usage = usage
        self.calls = 0

    def _post(self, prompt):
        self.calls += 1
        return {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": self.usage}


__all__ = [name for name in globals() if not name.startswith("__")]
