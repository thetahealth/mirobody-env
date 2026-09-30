"""llm_rubric.py — LLM judge that answers atomic rubric points with booleans
only; rubric points live in `registry/rubric_points.yaml`.

Verdicts are `bool`/`None` (`None` = could not be judged, never folded into
False); aggregation is plain code in `aggregate()`. Process fields
such as `trace` never reach the judge's input. The dimension is
always `diagnostic`, so judge changes stay recomputable without model calls.
"""
from __future__ import annotations

from .yamlcache import load_yaml as _cached_yaml

import functools
import hashlib
import json
import pathlib
from typing import Any

from haenv import data_root as _data_root
ROOT = _data_root()
REGISTRY = ROOT / "registry" / "rubric_points.yaml"

#: Always reports only, never feeds the score; hardcoded, not read from the registry.
ROLE = "diagnostic"

#: The LLM judge dimension's name in the scoring registry. Defined once here,
#: checked against by `scoring.load_profile`.
DIM = "rubric_binary"

#: Allowlist of answer fields the judge may see (a denylist miss would leak a
#: process record).
ALLOWED_INPUT_FIELDS = ("join_type", "join_reason", "action", "tests_to_order",
                        "referral_specialty", "cited_evidence")


class RubricError(ValueError):
    """The registry is malformed, a verdict has an invalid shape, or someone
    tried to feed a process record to the judge."""


@functools.lru_cache(maxsize=1)
def _doc() -> dict[str, Any]:
    import yaml
    if not REGISTRY.is_file():
        raise RubricError(f"{REGISTRY} is missing — the rubric points have no source; there is no fallback to an empty table")
    d = _cached_yaml(REGISTRY) or {}
    pts = d.get("points")
    if not isinstance(pts, dict) or not pts:
        raise RubricError(f"{REGISTRY}: `points:` is missing or empty")
    for name, spec in pts.items():
        for f in ("question", "answer_true_means", "gold_ref", "why", "polarity"):
            if not str((spec or {}).get(f) or "").strip():
                raise RubricError(f"{REGISTRY}: rubric point {name} is missing `{f}:`")
        if spec["polarity"] not in ("higher_is_better", "lower_is_better"):
            raise RubricError(
                f"{REGISTRY}: {name}.polarity={spec['polarity']!r} is not a controlled value — "
                "getting the direction wrong flips the sign of the whole dimension, invisibly in the readout")
    j = d.get("judge") or {}
    if not j.get("model"):
        raise RubricError(f"{REGISTRY}: `judge.model` is missing — the judge model must be registered")
    # The judge's route must be declared in this table (see `judge_route()`).
    for f in ("backend", "vendor"):
        if not str(j.get(f) or "").strip():
            raise RubricError(
                f"{REGISTRY}: `judge.{f}` is missing — the judge's complete route must live "
                "in this table, because only this table is inside `make_freeze.JUDGING` "
                "(config.yaml is not); a route outside the freeze segment could change without "
                "`judging_sha16` moving")
    if int(j.get("repeats") or 0) < 3:
        raise RubricError(f"{REGISTRY}: `judge.repeats` must be at least 3 — the pass^k (k=3) noise floor depends on it")
    return d


def judge_route() -> dict[str, Any]:
    """The judge's route (backend / vendor / model / max_tokens), read only from
    the registry: the registry is inside the judging freeze segment and
    `config.yaml` is not, so a route change moves `judging_sha16`.
    """
    j = judge_spec()
    return {"backend": str(j["backend"]), "vendor": str(j["vendor"]),
            "model": str(j["model"]), "max_tokens": int(j.get("max_tokens") or 4000)}


def route_tag() -> str:
    """Route label carried by every verdict row; rows on different routes are not comparable."""
    r = judge_route()
    return f"{r['model']}@{r['backend']}"


def points() -> dict[str, dict[str, Any]]:
    """All rubric points."""
    return dict(_doc()["points"])


def judge_spec() -> dict[str, Any]:
    """The judge model's registration (model / backend / temperature / repeat count)."""
    return dict(_doc()["judge"])


def forbidden_input_fields() -> tuple[str, ...]:
    return tuple(_doc().get("forbidden_input_fields") or ())


# ──────────────────────────────────────────────── every rubric point must trace to gold

#: The two valid prefixes for `gold_ref`. `case:` points to a path inside the
#: case, `kernel:` points to a kernel constant.
_GOLD_PREFIXES = ("case:", "kernel:")


def _resolve_case_path(case: dict, path: str):
    cur: Any = case
    for seg in path.split("."):
        if not isinstance(cur, dict) or seg not in cur:
            return None, False
        cur = cur[seg]
    return cur, True


def assert_gold_refs_resolve(case: dict, kernel_names=()) -> None:
    """Raise unless every `gold_ref` resolves (`case:` path in this case, `kernel:` constant)."""
    bad = []
    for name, spec in points().items():
        ref = str(spec["gold_ref"])
        if not ref.startswith(_GOLD_PREFIXES):
            bad.append(f"{name}: gold_ref={ref!r} has an unrecognized prefix (only {_GOLD_PREFIXES} allowed)")
            continue
        kind, _, path = ref.partition(":")
        if kind == "case":
            _v, ok = _resolve_case_path(case, path)
            if not ok:
                bad.append(f"{name}: case path `{path}` does not exist in this case")
        else:
            if path not in set(kernel_names):
                bad.append(f"{name}: kernel constant `{path}` does not exist")
    if bad:
        raise RubricError("rubric points that do not trace to gold (§30):\n  " + "\n  ".join(bad))


def assert_role_is_diagnostic(scoring_profile: dict, dim_name: str) -> None:
    """Raise unless the scoring profile gives this dimension role `diagnostic`."""
    role = ((scoring_profile.get("metrics") or {}).get(dim_name) or {}).get("role")
    if role != ROLE:
        raise RubricError(
            f"`{dim_name}` has role={role!r} in the scoring profile, but the LLM judge dimension "
            f"only allows {ROLE!r}: scoring it would break `recompute_judges` "
            "(a judge change would require calling the model again on recompute)")


# ──────────────────────────────────────────────── boundary: process records never reach the input

def build_prompt(case: dict, answer: dict) -> tuple[str, str]:
    """`(prompt, prompt_sha256)` — the entire judge input. Answer fields come from
    `ALLOWED_INPUT_FIELDS`; `differential` / `drivers` / `forecast` are excluded
    so the judge cannot copy the answer.
    """
    ans = {k: answer.get(k) for k in ALLOWED_INPUT_FIELDS if answer.get(k) is not None}
    leaked = set(ans) & set(forbidden_input_fields())
    if leaked:
        raise RubricError(f"a forbidden field leaked into the judge's input: {sorted(leaked)} (§35)")

    ddx = ((case.get("adjudication") or {}).get("ddx") or {})
    gold = {"join_gold": ddx.get("join_gold"), "tests": ddx.get("tests"),
            "urgency": ddx.get("urgency")}

    qs = [f"{i+1}. [{n}] {s['question'].strip()}\n   回 true 表示:{s['answer_true_means'].strip()}"
          for i, (n, s) in enumerate(sorted(points().items()))]
    body = (
        "你在核对一份临床推理作答。**只回二值,不要打分、不要给档位。**\n"
        "对下面每一个问题回 true / false;**判不了就回 null**(不要猜,不要折成 false)。\n\n"
        "## 金标(供对照)\n" + json.dumps(gold, ensure_ascii=False, indent=1) +
        "\n\n## 待核对的作答(已剔除过程记录与结论字段)\n"
        + json.dumps(ans, ensure_ascii=False, indent=1) +
        "\n\n## 逐条回答\n" + "\n".join(qs) +
        "\n\n## 输出格式(严格 JSON,不要别的)\n"
        '{"<得分点名>": {"value": true|false|null, "why": "<一句话理由>"}, ...}\n'
        "⚠️ `why` 是给人审阅的,**不参与计分**;不要为了让理由好看而改 value。\n"
    )
    return body, hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


# ──────────────────────────────────────────────── verdicts are binary only

def parse_verdict(raw: str) -> dict[str, bool | None]:
    """Judge reply → `{rubric point: bool|None}`; a non-boolean value or an unknown
    key raises, and an unmentioned point is `None`.
    """
    txt = str(raw or "").strip()
    if txt.startswith("```"):
        txt = txt.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        d = json.loads(txt)
    except Exception as e:                                       # noqa: BLE001
        raise RubricError(f"judge reply is not JSON: {type(e).__name__} · first 120 chars: {txt[:120]!r}")
    known = set(points())
    out: dict[str, bool | None] = {}
    for k, v in (d or {}).items():
        if k not in known:
            raise RubricError(f"judge answered a rubric point not on the list: {k!r} (list: {sorted(known)})")
        val = v.get("value") if isinstance(v, dict) else v
        if val is None:
            out[k] = None
        elif isinstance(val, bool):
            out[k] = val
        else:
            raise RubricError(
                f"{k}'s verdict is {val!r} ({type(val).__name__}) — "
                "§30 only allows binary values; a continuous score or tier is always a fail")
    for k in known - set(out):
        out[k] = None                       # judge didn't mention it = unjudged, not False
    return out


def oriented(name: str, value: bool | None) -> bool | None:
    """Orient a binary value by the point's `polarity` (`lower_is_better` takes the complement)."""
    if value is None:
        return None
    return (not value) if points()[name]["polarity"] == "lower_is_better" else value


def aggregate(verdicts: dict[str, bool | None]) -> dict[str, Any]:
    """Per-cell aggregation `{n_true, n_judged, n_unjudged, rate}`; `rate` is over
    judged points only.
    """
    ok = {k: oriented(k, v) for k, v in verdicts.items()}
    judged = [v for v in ok.values() if v is not None]
    return {"n_true": sum(judged), "n_judged": len(judged),
            "n_unjudged": len(ok) - len(judged),
            "rate": (sum(judged) / len(judged)) if judged else None}
