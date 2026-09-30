"""`disc_recall` must match item by item, not by searching the concatenated full text.

## Invariant

`judge_discriminative_tool` caps the **item count** with
`tests[:_n_req + TESTS_CAP_MARGIN]`. Matching against the **entire
concatenated text** produced by `"\n".join(tests)` would let a single item
that strings together every discriminator name into one long string drive
`disc_recall` to 1.000, and the item-count cap would never trim it.

`disc_recall` is the highest-separation scoring dimension on the main
leaderboard. `workup` (`tests_recall`) has the same shape: cap the item count
first, then judge item by item.

## Enforcement surface

`OneLongTestSolver` (stub name `onelong`) exercises the invariant: matching item
by item keeps its score low. Without it, "item-by-item matching" is a claim
that is never actually exercised.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
sys.path.insert(0, str((ROOT / _cfg["kernel_path"]).resolve()))

from haenv.judges import judge_discriminative_tool              # noqa: E402
from haenv.registry import load_findings, load_rivals  # noqa: E402


def _spec_with_most_rivals():
    rv = load_rivals()
    if not rv:
        pytest.skip("`rivals.yaml` 为空 —— 这条测试没有对象")
    return max(rv, key=lambda k: len(rv[k])), rv


def _disc_names(spec, rv):
    fx = load_findings()
    out = []
    for r in rv[spec]:
        fid = (r.get("discriminator_finding") or {}).get("finding")
        f = fx.get(fid) or {}
        nm = f.get("name_cn") or f.get("name_en") or fid
        if nm and str(nm) not in out:
            out.append(str(nm))
    return out


class _VP:
    def __init__(self, spec, tests):
        self.adjudication = {"ddx": {"spec_id": spec, "tests": tests,
                                     "diagnosis": "X"}}
        self.gold_drivers = []


class _Out:
    def __init__(self, tests):
        self._raw = {"tests_to_order": list(tests)}
        self.forecast = {}
        self.drivers = []
        self.action = {}
        self.data_quality = {}
        self.cited_evidence = []


def test_one_long_string_cannot_max_out_recall():
    """Stringing every discriminator into a single test must not max out the score."""
    spec, rv = _spec_with_most_rivals()
    names = _disc_names(spec, rv)
    if len(names) < 2:
        pytest.skip(f"{spec} 只有 {len(names)} 个判别项 —— 长串攻击没有对象")
    onelong = "、".join(names)
    r = judge_discriminative_tool(_Out([onelong]), _VP(spec, ["某项必需检查"]), {})
    assert r.get("disc_n_rivals"), r
    assert r["disc_recall"] < 1.0, (
        f"一条 {len(onelong)} 字的长串把 disc_recall 刷到 {r['disc_recall']} ⇒ 上限没拦住")


def test_per_item_answer_still_scores():
    """Positive control: answering item by item, honestly, must still score high.

    Without this test, the test above could be passed by a judge that always
    scores 0 -- which would also reject a legitimate answer.
    """
    spec, rv = _spec_with_most_rivals()
    names = _disc_names(spec, rv)
    if len(names) < 2:
        pytest.skip("判别项不足 2 个")
    r = judge_discriminative_tool(_Out(list(names)), _VP(spec, list(names)), {})
    assert r["disc_recall"] >= 0.5, f"逐项开出来却只拿到 {r['disc_recall']}:{r}"


def test_onelong_stub_is_registered():
    """The `onelong` stub must be in `BASELINE_NAMES` -- otherwise it never enters the degenerate-ceiling denominator."""
    from haenv.baselines import BASELINE_NAMES, OneLongTestSolver
    assert "onelong" in BASELINE_NAMES, BASELINE_NAMES
    assert OneLongTestSolver.name == "onelong"


def test_oracle_disc_names_is_not_in_the_prompt_path():
    """`ORACLE_DISC_NAMES` holds gold-standard values => it must never enter the prompt-rendering path.

    Same discipline as `ORACLE_GOLD_TESTS`: it is reachable in-process, and
    only one line of code away from being rendered into the prompt.
    """
    import ast
    src = (ROOT / "haenv").glob("*.py")
    bad = []
    for f in src:
        if f.name in ("evaluate.py", "baselines.py"):
            continue                                   # the registry and the oracle stub itself -- legitimate
        t = ast.parse(f.read_text(encoding="utf-8"))
        for n in ast.walk(t):
            if isinstance(n, ast.Name) and n.id == "ORACLE_DISC_NAMES":
                bad.append(f.name)
    assert not bad, f"金标判别项名被这些模块引用了:{sorted(set(bad))}"
