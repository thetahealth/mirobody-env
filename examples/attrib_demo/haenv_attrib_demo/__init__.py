"""Example plugin: wires an "event-effect attribution" question type into haenv without
changing haenv code.

Four registrations, made by `judges()`: a gold-kind rule (`gold_kinds`), a judge, a mount
cell (`mount_table.mount`; a judge without one never runs), and the gold block, probe and
framing (`external_gold`). The judge is mounted on `single` only: an attribution question is
answered once. `examples/check_demos.py` runs this package against a control job without
the plugin.

SYNTHETIC, for evaluation only, not medical advice.
"""
from __future__ import annotations

import sys

NAME = "attrib_event_driver"
KIND = "attrib_event_driver"
KEYS = ("attrib_hit", "attrib_answer_kind", "attrib_gold_kind", "attrib_n_candidates")

ANSWER_KINDS = ("named", "abstained", "invalid_id", "unparseable", "missing_key")


# -- extraction: the kernel's extractor --------------------------------------------------------
def _kernel_extract_json():
    try:
        from haenv import kernel_path as _kp
        p = _kp()
        if p and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    except Exception:                       # noqa: BLE001 -- keep trying even if the kernel path can't be found
        pass
    from solver import _extract_json        # type: ignore[import-not-found]
    return _extract_json


def judge_attribution(item: dict, gold: dict, response_text: str) -> dict:
    """Judge one attribution question (pure function, no model calls).

    | kind | meaning | `attrib_hit` |
    |---|---|---|
    | `named`       | named an event that is in the candidates | 1/0 |
    | `abstained`   | declared no event is driving it | 1/0 |
    | `invalid_id`  | named an id that is not in the candidates | 0 |
    | `unparseable` | JSON couldn't be extracted | None |
    | `missing_key` | JSON present but missing `driver_event_id` | None |

    `invalid_id` is a wrong answer (0); `unparseable`/`missing_key` are pipeline failures
    (None, not measured), as with haenv's `ABORT(unparseable)`.
    """
    gold_kind = gold.get("gold_kind")
    cands = [e.get("event_id") for e in (item.get("candidate_events") or [])
             if isinstance(e, dict)]
    base = {"attrib_gold_kind": gold_kind, "attrib_n_candidates": len(cands)}

    obj = _kernel_extract_json()(response_text or "")
    if not isinstance(obj, dict) or not obj:
        return {**base, "attrib_answer_kind": "unparseable", "attrib_hit": None}
    if "driver_event_id" not in obj:
        return {**base, "attrib_answer_kind": "missing_key", "attrib_hit": None}

    said = obj["driver_event_id"]
    if said is None:
        return {**base, "attrib_answer_kind": "abstained",
                "attrib_hit": 1 if gold_kind == "no_event" else 0}
    if said not in cands:
        return {**base, "attrib_answer_kind": "invalid_id", "attrib_hit": 0}
    return {**base, "attrib_answer_kind": "named",
            "attrib_hit": 1 if (gold_kind == "event"
                                and said == gold.get("gold_event_id")) else 0}


# -- registration into haenv -----------------------------------------------------------------

def _attrib_payload(vp):
    """The attribution question's `(item, gold)` from a VerifierPayload, or `None` when the case
    has no attribution block (never an empty dict, which would read as "0 candidates").
    """
    adj = getattr(vp, "adjudication", None) or {}
    blk = adj.get("attrib") if isinstance(adj, dict) else None
    if not isinstance(blk, dict):
        return None
    item, gold = blk.get("item"), blk.get("gold")
    if not isinstance(item, dict) or not isinstance(gold, dict):
        return None
    return item, gold


def _derive_kind(vp) -> str | None:
    """Kind rule: `attrib_event_driver` if the case carries an attribution gold block, else no match."""
    return KIND if _attrib_payload(vp) is not None else None


def _judge(out, vp, ctx):
    got = _attrib_payload(vp)
    if got is None:
        return {k: None for k in KEYS}
    item, gold = got
    if isinstance(out, str):
        text = out
    else:
        text = str(getattr(out, "_raw_text", "") or "")
        if not text and isinstance(out, dict):
            text = __import__("json").dumps(out, ensure_ascii=False)
    return judge_attribution(item, gold, text)


def _ensure_registered():
    from haenv import gold_kinds as GK
    from haenv import mount_table as MT
    from haenv.judges import Judge, register_judge

    if NAME not in {r.name for r in GK.RULES}:
        GK.register_kind_rule(
            GK.KindRule(NAME, _derive_kind,
                        why="带归因金标(`adjudication.attrib`)的例子走这条。",
                        kinds=(KIND,)),
            source="haenv_attrib_demo",
            # `_rule_not_ddx` matches every case without ddx ground truth, so this rule must come
            # before it; it matches only cases with an `adjudication.attrib` block.
            before="not_ddx")

    if NAME not in MT.MOUNT:
        MT.mount(NAME, {"single": "out"}, why_not={
            "gated": "门控是一次作答 + 若干轮采买;归因题不采买,预算轴在它上面无对象",
            "slices": "切片是同一病人多时点各自独立作答;归因题只问 T 当天那一个格子,"
                      "切成多片等于把同一道题问 N 遍",
            "multi": "多轮考的是「错了之后能不能在正确时点纠正」;归因题是一次性指认,"
                     "没有可纠正的后续轮次",
        })

    return Judge(name=NAME, fn=_judge, kinds=(KIND,), category="deterministic")


# -- gold standard: job.yaml `latent.attrib_payload` -> `adjudication.attrib` --------------------
#
# `run_judges(SINGLE, out, vp)` passes no `ctx`, so external gold reaches the judge through
# `vp`: `external_gold` resolves `latent.attrib_payload` into `adjudication.attrib` (see
# `haenv/external_gold.py`).

LATENT_KEY = "attrib_payload"

#: Shape of `latent.attrib_payload`: a reference into a question pool, not the question itself.
#:
#:     latent:
#:       attrib_payload: {pool: derived/attrib-pool, item_id: ATTR-5300-...-day_30}
#:
#: `pool` is a basename relative to the repo root; the loader reads `<pool>.items.jsonl` and
#: `<pool>.gold.jsonl`.


def _load_from_pool(ref: dict):
    """`(item, gold)` from the question pool by reference. Raises if not found, as
    `job.load_plugins` does.
    """
    import json
    import pathlib

    pool, iid = ref.get("pool"), ref.get("item_id")
    if not pool or not iid:
        raise ValueError(f"latent.{LATENT_KEY} needs keys `pool` and `item_id`, got {sorted(ref)}")
    from haenv import __file__ as _hf
    root = pathlib.Path(_hf).resolve().parent.parent
    ip, gp = root / f"{pool}.items.jsonl", root / f"{pool}.gold.jsonl"
    if not ip.is_file() or not gp.is_file():
        raise FileNotFoundError(f"item pool not found: {ip} / {gp}; "
                                f"`latent.{LATENT_KEY}.pool` must point to existing "
                                f"`<pool>.items.jsonl` and `<pool>.gold.jsonl`")
    item = next((json.loads(x) for x in ip.read_text(encoding="utf-8").splitlines()
                 if x.strip() and json.loads(x).get("item_id") == iid), None)
    # The gold rows carry the canary strings under `canary.FIELD`; drop them before
    # the row becomes `adjudication.attrib`.
    from haenv import canary as _canary
    gold = next((_canary.strip_row(json.loads(x)) for x in gp.read_text(encoding="utf-8").splitlines()
                 if x.strip() and json.loads(x).get("item_id") == iid), None)
    if item is None or gold is None:
        raise KeyError(f"item pool {pool} has no item_id={iid!r}")
    return item, gold


def _gold_block(spec: dict):
    """`external_gold` block factory: spec -> `adjudication.attrib`, or `None` if the case declares
    no `attrib_payload`.
    """
    from haenv import external_gold as EG
    ref = (spec.get(EG.SLOT) or {}).get(LATENT_KEY)
    if not isinstance(ref, dict):
        return None
    item, gold = _load_from_pool(ref)
    return {"item": item, "gold": gold}


def _ensure_gold_block():
    from haenv import external_gold as EG
    if "attrib" in EG.BLOCKS:
        return
    EG.register_gold_block(
        "attrib", _gold_block, latent_keys=(LATENT_KEY,),
        latent_classes={LATENT_KEY: "gold"},    # carries gold_event_id: verifier-side
        source="haenv_attrib_demo",
        probe_fn=_probe_block,
        why="「事件效应归因」题的金标(题面 + gold_event_id)。"
            "块名 `attrib` 与 `_attrib_payload(vp)` 读的键一致 —— 两处必须同名,"
            "否则登记了也取不到,而那与没登记在产物上一样。")


# -- prompt side --------------------------------------------------------------------------------
# The framing below asks for the `driver_event_id` the judge reads.

FRAMING = "ATTRIB_PROMPT"

#: Rendered with `.format()`: literal braces are written `{{` / `}}`.
ATTRIB_PROMPT = """你是"事件效应归因"求解器。只依据下面 index-time T 之前(pre-T)的数据作答。

数据(JSON):
{payload}

`prediction_context.attrib` 里给了本题:
  · `question`         —— 要回答的问题
  · `indicator`        —— 出现偏离的那个指标
  · `series`           —— 该指标 ≤T 的观测序列
  · `candidate_events` —— 截至 T 已发生的事件(只有 id / 名称 / 类型 / 描述 / 起始日 / 时长)

判断该指标在 T 当日相对其基线的偏离,**主要**由哪一个候选事件驱动。
若认为没有任何事件在驱动(噪声或惯性主导),如实声明 —— 那一档是有正解的,不是弃权。

只输出一个 JSON 对象,无多余文字、无 markdown 围栏:
  {{"driver_event_id": "<candidate_events 里的某个 event_id>"}}
或
  {{"driver_event_id": null}}
"""


def _probe_block(spec: dict):
    """Prompt side: the pool item is handed to the solver as-is (items were filtered at export and
    never contain `gold`).
    """
    from haenv import external_gold as EG
    ref = (spec.get(EG.SLOT) or {}).get(LATENT_KEY)
    if not isinstance(ref, dict):
        return None
    item, _gold = _load_from_pool(ref)
    return item


#: This question type's probe. It ships with the package, not in the repo's `probes/`, which is
#: validated as a whole and would reject a probe that needs a plugin.
PROBE = {
    "probe_id": "attrib.direct",
    "mode": "default",
    "difficulty_probe": "P0",
    "framing_ref": FRAMING,
    "framing_sha256": "531b43dff559a4e8",
    "answer_space": "enum",      # the answer is one of the event_ids in candidate_events, or null
    "hint_level": 0,
    "budget": {"max_rounds": 1},
}


def _ensure_probe():
    from haenv import external_gold as EG
    if PROBE["probe_id"] not in EG.PROBES:
        EG.register_probe(PROBE, source="haenv_attrib_demo")


def _ensure_framing():
    from haenv import external_gold as EG
    if FRAMING in EG.FRAMINGS:
        return
    EG.register_framing(
        FRAMING, ATTRIB_PROMPT, source="haenv_attrib_demo",
        why="归因题的问法。自带的 PROMPT 问的是「预警」,与本题型不是一个问题;"
            "没有它,题面侧就只能沿用 early_warning 的问法,而判据要的 `driver_event_id` "
            "永远不会出现。")


def judges():
    """Entry point for `job.yaml`'s `plugins:`: registers framing, probe, gold block and mount,
    then returns the judges. A missing part shows up only as `None` readings.
    """
    _ensure_framing()
    _ensure_probe()
    _ensure_gold_block()
    return [_ensure_registered()]
