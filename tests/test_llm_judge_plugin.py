"""Contract tests for the judge-side LLM plugin (`examples/llm_judge_demo`) -- entirely offline, never calls a real model.

The dispatcher is always fake (an injected lambda, or a real `Dispatcher`
with `subprocess.run` swapped out), so this file runs to completion on any
machine, with no keys at all, and costs nothing.

## Every rule is paired

If only the "it blocks" assertion exists, an implementation that always
blocks would also pass. So every positive assertion below is paired with a
`!` counter-case proving it is not vacuously true:

| Rule | Positive | `!` counter-case |
|---|---|---|
| Runs only once mounted | `test_factory_mounts_and_is_selected` | `test_register_without_mount_never_runs` |
| Absent unless loaded | `test_absent_before_load` | `test_present_after_load` |
| A cache hit is not paid for twice | `test_cache_hit_pays_once` | `test_cache_misses_when_reason_changes` |
| Unresolvable never collapses to 0 | `test_no_dispatch_is_none_not_zero` | `test_real_zero_is_reachable` |
| Return shape | `test_row_keys_are_exactly_declared` | `test_manifest_without_plugin` |
"""
from __future__ import annotations

import importlib.metadata as _md
import pathlib
import sys

import pytest
import yaml

# Same bootstrap as `test_judge_registry.py`: repo root + kernel (the L0 substrate is not part of this repo)
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
sys.path.insert(0, str((ROOT / _cfg["kernel_path"]).resolve()))
#: The plugin package can be tested without being installed: its directory
#: is added straight to the import path. The entry-point path has its own
#: separate test (`test_entry_point_group_resolves`) -- "the module can be
#: imported" and "the entry point resolves" are two different things, and
#: testing them together would let the former mask the latter.
EXAMPLE_DIR = ROOT / "examples" / "llm_judge_demo"
sys.path.insert(0, str(EXAMPLE_DIR))

import haenv_llm_judge_demo as P                    # noqa: E402
from haenv import judges as J                       # noqa: E402
from haenv import mount_table as MT                 # noqa: E402

#: A spec_id with near-miss rivals registered in this repo's `registry/rivals.yaml` (JD-PCOS has 2).
SPEC_WITH_RIVALS = "JD-PCOS"
#: A spec_id that exists in this repo's case pack but has no rivals registered (JD-17 in `joint_dx-ddx.job.yaml`).
SPEC_WITHOUT_RIVALS = "JD-BENIGN2"


# ---------------------------------------------------------------- fixtures
@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    """Clean registry, clean mount table, clean switch. Everything is restored afterward.

    Missing any half would let tests see each other's mounted judges or
    injected dispatchers, so "it's mounted" could just mean the previous
    test mounted it -- and this whole file would test nothing.
    """
    monkeypatch.delenv(P.ENV_SWITCH, raising=False)
    P.set_dispatch(None)
    J.reset_judges()
    _mount_before = {k: dict(v) for k, v in MT.MOUNT.items()}
    _why_before = dict(MT.WHY_NOT)
    yield
    P.set_dispatch(None)
    J.reset_judges()
    MT.MOUNT.clear(); MT.MOUNT.update(_mount_before)
    MT.WHY_NOT.clear(); MT.WHY_NOT.update(_why_before)


class _VP:
    """Minimal verifier payload: carries only the segments `wq.gold_of` needs to go through `W.adjudication.ddx.*`."""

    def __init__(self, spec_id=SPEC_WITH_RIVALS, diagnosis="多囊卵巢综合征(PCOS)"):
        self.case_id = "T-LLMJ-01"
        self.outcome_label = "event_occurred"
        self.adjudication = {"ddx": {"spec_id": spec_id, "diagnosis": diagnosis,
                                     "aliases": ["多囊", "pcos"], "join_gold": "unified"}}


class _OUT:
    """Minimal SolverOutput: the judge only reads `_raw["differential"]` (via `tracks._differential`)."""

    def __init__(self, differential):
        self._raw = {"differential": list(differential)}


def _out_two_claims(reason_a="17-OHP 正常,不支持 21-羟化酶缺陷",
                    reason_b="临床表现不支持"):
    """Two candidates that claim to have ruled out a registered near-miss rival -- the object this file judges."""
    return _OUT([
        {"diagnosis": "多囊卵巢综合征", "rank": 1},
        {"diagnosis": "非经典型先天性肾上腺皮质增生", "rank": 2, "ruled_out_by": reason_a},
        {"diagnosis": "高泌乳素血症", "rank": 3, "ruled_out_by": reason_b},
    ])


def _load_plugin() -> list:
    """Loads through the factory (the same callable `load_judge_plugins` calls)."""
    added = []
    for j in P.judges():
        J.register_judge(j, source="pkg:haenv_llm_judge_demo")
        added.append(j.name)
    return added


# =============================================================== 0. positive control for existence
def test_fixture_surface_is_not_empty():
    """Positive control for scan coverage. Every "found / not found" assertion below is built on these three things; if they were empty, the whole file would degenerate into a string of vacuously true `not in` checks. Assert a lower bound, not just `is not None`."""
    from haenv.overlay import rivals_for
    rivals = rivals_for(SPEC_WITH_RIVALS, {})
    assert len(rivals) >= 2, f"{SPEC_WITH_RIVALS} 在 registry/rivals.yaml 里应有 ≥2 条近名"
    assert all(str(r.get("discriminator") or "").strip() for r in rivals), \
        "每条近名都必须有 discriminator(装载期硬约束),否则本判据的提示词是空的"
    assert not rivals_for(SPEC_WITHOUT_RIVALS, {}), \
        f"{SPEC_WITHOUT_RIVALS} 被当作『没登记近名』的对照,它不该有近名"
    assert len(J._CORE) >= 20, "自带判据表被读空了 —— 引导没接上,后面的对比全无意义"


# =============================================================== 1. load / mount
def test_absent_before_load():
    """Not loaded => the judge does not exist, is absent from the mount table, and `applicable` never selects it."""
    assert P.NAME not in [j.name for j in J.JUDGES]
    assert P.NAME not in MT.MOUNT
    assert P.NAME not in J.registry_manifest()["external"]
    picked = [j.name for j in J.applicable("single", "ddx:unified", _VP())]
    assert P.NAME not in picked


def test_present_after_load():
    """`!` counter-case for `test_absent_before_load`: after loading, all four flip.

    The previous test alone would let an implementation that never loads
    anything also pass.
    """
    assert _load_plugin() == [P.NAME]
    assert P.NAME in [j.name for j in J.JUDGES]
    assert MT.subject_of(P.NAME, "single") == MT.OUT
    assert J.registry_manifest()["external"][P.NAME] == "pkg:haenv_llm_judge_demo"
    assert P.NAME in [j.name for j in J.applicable("single", "ddx:unified", _VP())]


def test_factory_mounts_and_is_selected():
    """Both halves done => selected on single-shot/gated, not on slices (the slices cell has a reason, it's not a hole)."""
    _load_plugin()
    assert MT.subject_of(P.NAME, "single") == MT.OUT
    assert MT.subject_of(P.NAME, "gated") == MT.OUT
    assert MT.subject_of(P.NAME, "slices") == MT.NONE
    assert MT.reason_for(P.NAME, "slices"), "不挂的格必须有理由,否则它是叉乘里的一个洞"
    assert MT.reason_for(P.NAME, "multi"), "multi 由表里既有的 `*@multi` 通配理由兜底"
    assert (P.NAME, "slices") not in MT.holes() and (P.NAME, "multi") not in MT.holes()
    vp = _VP()
    assert P.NAME in [j.name for j in J.applicable("single", "ddx:unified", vp)]
    assert P.NAME in [j.name for j in J.applicable("gated", "ddx:unified", vp)]
    assert P.NAME not in [j.name for j in J.applicable("slices", "ddx:unified", vp)]


def test_register_without_mount_never_runs():
    """`!` register without mount => every geometry is NONE => registered but never runs.

    This proves that `mount` in the previous test is load-bearing, not
    decorative -- "not mounted" and "not registered" look identical in the
    output, so it has to be pinned down separately.
    """
    J.register_judge(J.Judge(P.NAME, ("ddx:unified",), P.judge_rival_discriminator),
                     source="pkg:test-no-mount")
    assert P.NAME in [j.name for j in J.JUDGES]          # registered
    for g in MT.GEOMETRIES:
        assert MT.subject_of(P.NAME, g) == MT.NONE       # but mounted on no geometry at all
    assert P.NAME not in [j.name for j in J.applicable("single", "ddx:unified", _VP())]


def test_factory_is_idempotent():
    """`load_job` is not guaranteed to be called only once per `haenv run` => a second call must return an empty list."""
    _load_plugin()
    assert P.judges() == []
    assert [j.name for j in J.JUDGES].count(P.NAME) == 1


def test_kinds_and_when_align_with_dx_rival():
    """`kinds` / `when` align with `dx_rival` -- otherwise the two dimensions' denominators would disagree when read side by side."""
    _load_plugin()
    mine = [j for j in J.JUDGES if j.name == P.NAME][0]
    core = [j for j in J._CORE if j.name == "dx_rival"][0]
    assert tuple(mine.kinds) == tuple(core.kinds)
    assert mine.when is not None and core.when is not None
    # not mounted on the early-warning kind (`!`: mounted on diagnosis cases)
    assert P.NAME not in [j.name for j in J.applicable("single", "forecast", _VP())]
    assert P.NAME in [j.name for j in J.applicable("single", "ddx:unified", _VP())]
    # the `when` precondition doesn't hold on a case with no registered rivals (`!`: it holds when registered)
    assert P.NAME not in [j.name for j in J.applicable(
        "single", "ddx:unified", _VP(spec_id=SPEC_WITHOUT_RIVALS))]


def test_name_and_keys_do_not_collide_with_core():
    """A key-name collision would be silently overwritten by `run_judges`'s `out.update()`, which looks identical to "not mounted."""
    assert P.NAME not in {j.name for j in J._CORE}
    assert all(k.startswith("llmj_disc_") for k in P.KEYS)
    assert len(set(P.KEYS)) == len(P.KEYS)


# =============================================================== 2. offline: must never silently score 0
def test_no_dispatch_is_none_not_zero():
    """Switch off => reports unresolvable explicitly; `rate` is `None`, not 0.0.

    And `n_claims` reports how many candidates there were to judge -- this
    is the line between "genuinely none were correctly excluded" and
    "nothing was even scanned." Without it, `rate=None` reads like "this
    case had nothing to judge."
    """
    row = P.judge_rival_discriminator(_out_two_claims(), _VP())
    assert row["llmj_disc_status"] == "no_dispatch"
    assert row["llmj_disc_rate"] is None
    assert row["llmj_disc_n_judged"] is None
    assert row["llmj_disc_n_discriminative"] is None
    assert row["llmj_disc_n_claims"] == 2, "可判对象数 —— 扫描面非空的正对照"
    assert row["llmj_disc_n_rivals"] == 2
    assert row["llmj_disc_via"].startswith("off:")


def test_real_zero_is_reachable():
    """`!` 0.0 must be reachable.

    Without this test, the `rate is None` assertion could be satisfied by
    an implementation that always returns None -- which is
    indistinguishable from a broken judge. This test gives it a real
    dispatcher and lets it produce a genuine 0.0.
    """
    P.set_dispatch(lambda prompt: "GENERIC\n只是一句空话,没指向任何可分辨的依据")
    row = P.judge_rival_discriminator(_out_two_claims(), _VP())
    assert row["llmj_disc_status"] == "ok"
    assert row["llmj_disc_n_judged"] == 2
    assert row["llmj_disc_n_discriminative"] == 0
    assert row["llmj_disc_rate"] == 0.0          # <- a genuine 0, distinct from the None in the previous test
    assert row["llmj_disc_via"] == "override"


def test_scoring_is_not_constant():
    """`!` The judge is neither always 0 nor always 1: the same two claims, different verdicts => different readings."""
    seq = iter(["DISCRIMINATIVE\n触及 17-OHP 这条轴", "GENERIC\n空话"])
    P.set_dispatch(lambda prompt: next(seq))
    row = P.judge_rival_discriminator(_out_two_claims(), _VP())
    assert row["llmj_disc_rate"] == 0.5 and row["llmj_disc_n_discriminative"] == 1
    P.set_dispatch(lambda prompt: "DISCRIMINATIVE\n都触及了")
    row2 = P.judge_rival_discriminator(_out_two_claims(), _VP())
    assert row2["llmj_disc_rate"] == 1.0


def test_dispatch_failure_is_not_zero():
    """The dispatcher raising (no key / timeout / empty response -- `haenv.llm` raises `RuntimeError` uniformly once retries are exhausted) => `dispatch_failed`, `rate` still `None`."""
    def _boom(prompt):
        raise RuntimeError("LLM 派发失败(fake):没有 key")
    P.set_dispatch(_boom)
    row = P.judge_rival_discriminator(_out_two_claims(), _VP())
    assert row["llmj_disc_status"] == "dispatch_failed"
    assert row["llmj_disc_rate"] is None and row["llmj_disc_n_judged"] is None
    assert row["llmj_disc_n_claims"] == 2
    assert [v["verdict"] for v in row["llmj_disc_verdicts"]] == [None, None]


def test_partial_failure_still_scores_the_rest():
    """`!` One failure doesn't take the other one down -- otherwise `dispatch_failed` would become a gate that always blocks."""
    seq = iter([RuntimeError("超时"), "DISCRIMINATIVE\n触及鉴别轴"])

    def _flaky(prompt):
        v = next(seq)
        if isinstance(v, Exception):
            raise v
        return v
    P.set_dispatch(_flaky)
    row = P.judge_rival_discriminator(_out_two_claims(), _VP())
    assert row["llmj_disc_status"] == "ok"
    assert row["llmj_disc_n_judged"] == 1          # the denominator only counts the one that got a verdict
    assert row["llmj_disc_rate"] == 1.0


def test_unparsable_response_is_not_zero():
    """The model answers but the verdict word can't be parsed => `unparsed`, `rate` is `None`. Never falls back to some verdict."""
    P.set_dispatch(lambda prompt: "我觉得这条排除写得还行吧")
    row = P.judge_rival_discriminator(_out_two_claims(), _VP())
    assert row["llmj_disc_status"] == "unparsed"
    assert row["llmj_disc_rate"] is None
    assert all(v["verdict"] is None for v in row["llmj_disc_verdicts"])


@pytest.mark.parametrize("resp,want", [
    ("DISCRIMINATIVE\n理由", "DISCRIMINATIVE"),
    ("  **GENERIC**  \n理由", "GENERIC"),
    ("irrelevant\n理由", "IRRELEVANT"),
    ("\n\nGENERIC。\n理由", "GENERIC"),
])
def test_parse_verdict_accepts(resp, want):
    assert P.parse_verdict(resp) == want


@pytest.mark.parametrize("resp", [
    "", None, "不是 DISCRIMINATIVE", "PARTIALLY_DISCRIMINATIVE",
    "理由在前\nDISCRIMINATIVE", "这条排除是 GENERIC 的",
])
def test_parse_verdict_rejects(resp):
    """`!` No substring matching. "This is not DISCRIMINATIVE" also contains that word; if the first line isn't a verdict word, it's treated as an unparsed answer, never guessed."""
    assert P.parse_verdict(resp) is None


def test_no_claim_and_no_rival_are_different_statuses():
    """"no rivals registered" and "registered, but the model claimed to exclude none" are two different things, and are not merged."""
    no_claim = P.judge_rival_discriminator(
        _OUT([{"diagnosis": "多囊卵巢综合征", "rank": 1}]), _VP())
    assert no_claim["llmj_disc_status"] == "no_claim"
    assert no_claim["llmj_disc_n_rivals"] == 2 and no_claim["llmj_disc_n_claims"] == 0
    assert no_claim["llmj_disc_rate"] is None

    no_rival = P.judge_rival_discriminator(
        _out_two_claims(), _VP(spec_id=SPEC_WITHOUT_RIVALS))
    assert no_rival["llmj_disc_status"] == "no_rival"
    assert no_rival["llmj_disc_n_rivals"] == 0
    assert no_rival["llmj_disc_rate"] is None


def test_no_claim_costs_nothing():
    """`!` With nothing to judge, the dispatcher is never called -- otherwise the numerator in the cost model would be wrong."""
    calls = []
    P.set_dispatch(lambda prompt: calls.append(prompt) or "DISCRIMINATIVE\nx")
    P.judge_rival_discriminator(_OUT([{"diagnosis": "多囊卵巢综合征"}]), _VP())
    assert calls == []
    P.judge_rival_discriminator(_out_two_claims(), _VP())
    assert len(calls) == 2, "有可判对象时**必须**派发 —— 证明上一句不是恒不派发"


@pytest.mark.parametrize("val", ["", "off", "0", "false", "no", "OFF"])
def test_env_switch_off_words(monkeypatch, val):
    monkeypatch.setenv(P.ENV_SWITCH, val)
    row = P.judge_rival_discriminator(_out_two_claims(), _VP())
    assert row["llmj_disc_status"] == "no_dispatch"
    assert row["llmj_disc_via"] == f"off:{P.ENV_SWITCH}"


def test_env_switch_bad_model_key_is_reported_not_silently_off(monkeypatch):
    """`!` "turned off" and "misconfigured" must be kept distinct.

    Both are unresolvable, but the former is a configuration choice and
    the latter is an accident -- using the same label would let a
    mistyped model name disguise itself as a normal offline run.
    """
    monkeypatch.setenv(P.ENV_SWITCH, "__no_such_model__")
    row = P.judge_rival_discriminator(_out_two_claims(), _VP())
    assert row["llmj_disc_status"] == "no_dispatch"
    assert row["llmj_disc_via"].startswith("unavailable:"), row["llmj_disc_via"]
    assert row["llmj_disc_rate"] is None


# =============================================================== 3. cache: the same prompt is paid for once
def _fake_dispatcher(tmp_path, monkeypatch, counter):
    """A real `haenv.llm.Dispatcher`, with only `subprocess.run` swapped out.

    Not a hand-rolled fake cache: what this needs to test is exactly this
    repo's on-disk cache (key = sha256(model||argv||prompt)); building a
    fake one would only test itself.
    """
    from haenv import llm as L

    class _Proc:
        returncode, stderr = 0, ""
        stdout = "DISCRIMINATIVE\n触及了那条鉴别轴"

    def _fake_run(*a, **kw):
        counter.append(1)
        return _Proc()
    monkeypatch.setattr(L.subprocess, "run", _fake_run)
    return L.Dispatcher(name="fake-judge", argv=["-b", "fake"], ai="/nonexistent/ai",
                        timeout=5, retries=0, env={}, cache_dir=tmp_path, use_cache=True)


def test_cache_hit_pays_once(tmp_path, monkeypatch):
    """The same prompt is paid for once -- a second Dispatcher hits the on-disk cache, with zero subprocess calls.

    Using a second instance rather than the same one: this forces the hit
    to come from the on-disk cache file. An in-process memory cache would
    also pass a "same instance" version of this test.
    """
    calls: list = []
    d1 = _fake_dispatcher(tmp_path, monkeypatch, calls)
    P.set_dispatch(d1)
    row1 = P.judge_rival_discriminator(_out_two_claims(), _VP())
    assert row1["llmj_disc_status"] == "ok" and row1["llmj_disc_n_judged"] == 2
    assert (d1.n_calls, d1.n_hits) == (2, 0) and len(calls) == 2
    assert len(list(tmp_path.glob("*.json"))) == 2, "两份缓存文件应当落盘"

    d2 = _fake_dispatcher(tmp_path, monkeypatch, calls)      # a new instance, same cache directory
    P.set_dispatch(d2)
    row2 = P.judge_rival_discriminator(_out_two_claims(), _VP())
    assert (d2.n_calls, d2.n_hits) == (0, 2), "全部命中缓存"
    assert len(calls) == 2, "子进程一次都不该再起"
    assert row2 == row1, "命中缓存的读数必须与第一次逐字段相同"


def test_cache_misses_when_reason_changes(tmp_path, monkeypatch):
    """`!` Changing the exclusion reason => must be paid for again.

    Without this test, "zero calls" could be satisfied by an
    implementation that never calls anything -- which looks identical to
    a working cache.
    """
    calls: list = []
    d1 = _fake_dispatcher(tmp_path, monkeypatch, calls)
    P.set_dispatch(d1)
    P.judge_rival_discriminator(_out_two_claims(), _VP())
    assert len(calls) == 2

    d2 = _fake_dispatcher(tmp_path, monkeypatch, calls)
    P.set_dispatch(d2)
    P.judge_rival_discriminator(_out_two_claims(reason_a="换了一句完全不同的排除理由"),
                                _VP())
    assert d2.n_calls == 1 and d2.n_hits == 1, "改了的那条重新调,没改的那条命中"
    assert len(calls) == 3


def test_prompt_carries_no_case_identity():
    """The precondition for the cache to be reusable across cases and across models under test: the prompt contains only those four segments.

    Including a case_id or a clinical history would collapse the cache
    hit rate to near zero, and that section of the cost model would be
    wrong.
    """
    p = P.build_prompt("金标病", "近名病", "鉴别点X", "排除理由Y")
    for seg in ("金标病", "近名病", "鉴别点X", "排除理由Y"):
        assert seg in p
    assert "T-LLMJ-01" not in p and "case_id" not in p
    # `!` changing any segment must change the prompt (otherwise the cache key can't tell two different judgments apart)
    assert P.build_prompt("金标病", "近名病", "鉴别点X", "排除理由Z") != p


# =============================================================== 4. return shape / manifest
def test_row_keys_are_exactly_declared():
    """Every path through the judge returns a key set == `KEYS`, and every status word is in the closed `STATUS` set.

    Keys that vary by branch would make it impossible downstream to tell
    "this column is absent" from "this cell has no value."
    """
    P.set_dispatch(lambda prompt: "DISCRIMINATIVE\nok")
    rows = [
        P.judge_rival_discriminator(_out_two_claims(), _VP()),                      # ok
        P.judge_rival_discriminator(_OUT([]), _VP()),                               # no_claim
        P.judge_rival_discriminator(_out_two_claims(), _VP(SPEC_WITHOUT_RIVALS)),   # no_rival
    ]
    P.set_dispatch(None)
    rows.append(P.judge_rival_discriminator(_out_two_claims(), _VP()))              # no_dispatch
    seen = set()
    for r in rows:
        assert set(r) == set(P.KEYS), f"键集合漂了:{sorted(set(r) ^ set(P.KEYS))}"
        assert r["llmj_disc_status"] in P.STATUS
        seen.add(r["llmj_disc_status"])
    assert seen == {"ok", "no_claim", "no_rival", "no_dispatch"}, \
        "四条分支都要真的走到 —— 否则这条断言只测了其中一条"


def test_row_is_json_serialisable():
    """The row goes into `eval.jsonl`. Anything unserializable stuffed into `verdicts` would only blow up when writing to disk."""
    import json
    P.set_dispatch(lambda prompt: "IRRELEVANT\n举的证分不开这两个病")
    row = P.judge_rival_discriminator(_out_two_claims(), _VP())
    assert json.loads(json.dumps(row, ensure_ascii=False)) == row
    assert all(len(v.get("raw", "")) <= 120 for v in row["llmj_disc_verdicts"]), \
        "行里只留摘要,原文在 cases/_llm_cache/(判据改口径时是重算,不是重买)"


def test_manifest_without_plugin():
    """`!` Without loading, the manifest matches the built-in set field for field."""
    m = J.registry_manifest()
    assert m["n_total"] == m["n_core"] == len(J._CORE)
    assert m["external"] == {} and P.NAME not in m["order"]


def test_manifest_with_plugin():
    """After loading: the external column gains this entry, and the core column must not be polluted."""
    _load_plugin()
    m = J.registry_manifest()
    assert m["n_total"] == m["n_core"] + 1
    assert m["external"] == {P.NAME: "pkg:haenv_llm_judge_demo"}
    assert P.NAME not in m["core"], "外挂判据不许混进 core 那一栏"
    assert m["order"][-1] == P.NAME
    assert m["n_core"] == len(J._CORE)


def test_core_registry_is_untouched():
    """The plugin must not alter a built-in judge -- the judge fingerprint only covers source code, it cannot catch a runtime substitution."""
    _load_plugin()
    with pytest.raises(ValueError, match="built into this repo"):
        J.register_judge(J.Judge("dx_rival", ("*",), P.judge_rival_discriminator),
                         replace=True)
    with pytest.raises(ValueError, match="built into this repo"):
        MT.mount("dx_rival", {"single": MT.OUT})


# =============================================================== 5. entry point and the counter-case job
def test_entry_point_group_resolves():
    """The entry-point path (the one `load_judge_plugins` takes). Three layers of positive control, narrowing at each layer.

    "Not found" and "the mechanism is broken" look the same, so this test
    first proves the scan surface is alive before it says anything about
    this package:

    1. `console_scripts` must be non-empty -- if it's empty,
       `entry_points()` itself is broken, and the test fails;
    2. Can this interpreter find the `haenv` distribution at all? If not,
       this isn't the project's own environment -- `uv run pytest` can end
       up borrowing a pytest found elsewhere on PATH, for which the
       project's editable plugin package is structurally invisible; skip
       in that case, with a note on how to run it instead;
    3. Inside the project's own environment, the already-installed
       `plugin_demo` must be found -- if it can't be found and this
       package also can't be found, the discovery mechanism itself is
       broken, not "this package isn't installed."
    """
    assert list(_md.entry_points(group="console_scripts")), \
        "连 console_scripts 都是空的 —— `entry_points()` 这个机制本身坏了"
    try:
        _md.distribution("haenv")
    except _md.PackageNotFoundError:
        pytest.skip("当前解释器不是项目 env(`uv run pytest` 借的是 PATH 上的 pytest)—— "
                    "editable 外挂包对它结构性不可见。跑这条用它:"
                    "`uv run --with pytest python -m pytest tests/test_llm_judge_plugin.py`")
    control = list(_md.entry_points(group="haenv.judges.demo"))
    assert control, ("entry point 扫描面是空的(连已装的 plugin_demo 都查不到)—— "
                     "这不是『没装本包』,是发现机制坏了")
    mine = list(_md.entry_points(group="haenv.judges.llm_demo"))
    if not mine:
        pytest.skip("本包未安装:uv pip install --no-deps -e examples/llm_judge_demo")
    added = J.load_judge_plugins("haenv.judges.llm_demo")
    assert added == [P.NAME]
    assert MT.subject_of(P.NAME, "single") == MT.OUT, \
        "entry point 只替我们调 register_judge,**不替我们调 mount** —— 工厂必须自己挂"
    assert J.registry_manifest()["external"][P.NAME] == "haenv.judges.llm_demo:rival_discriminator"


def test_demo_jobs_differ_only_by_plugins():
    """Counter-case job.yaml: the two job specs differ only in `plugins:`, with identical prompts otherwise.

    If they differed anywhere else, "this dimension only appears with the
    plugin enabled" would no longer be attributable to the plugin -- and
    this counter-case would prove nothing.
    """
    a = yaml.safe_load((EXAMPLE_DIR / "demo.job.yaml").read_text(encoding="utf-8"))
    b = yaml.safe_load((EXAMPLE_DIR / "demo-noplugin.job.yaml").read_text(encoding="utf-8"))
    assert a["plugins"] == ["haenv.judges.llm_demo"]
    assert "plugins" not in b, "负对照必须**没有** plugins 字段,不是空列表"
    assert a["cases"] == b["cases"] and len(a["cases"]) >= 2
    assert a["task_type"] == b["task_type"] == "joint_dx"
    assert a["job_id"] != b["job_id"], "两张单必须落在不同批次目录,否则会互相覆盖"
    # `!` the case pack must genuinely contain a case with registered rivals, otherwise everything comes back no_rival and the demo demonstrates nothing
    from haenv.overlay import rivals_for
    with_rivals = [c for c in a["cases"]
                   if rivals_for(c["latent"].get("ddx_spec_id"), {})]
    assert len(with_rivals) == len(a["cases"]), \
        "demo 题包每一例都该登记了近名(选 JD-PCOS/JD-CUSH 就是为了这个)"


def test_job_yaml_declares_the_group_that_the_package_publishes():
    """The group name in the job spec must be the same one published in `pyproject.toml`.

    A mismatch makes `load_plugins` just log a warning and move on -- the
    leaderboard still comes out, missing one dimension, which in the
    output looks identical to "this dimension doesn't apply here."
    """
    import re
    pt = (EXAMPLE_DIR / "pyproject.toml").read_text(encoding="utf-8")
    groups = re.findall(r'\[project\.entry-points\."([^"]+)"\]', pt)
    a = yaml.safe_load((EXAMPLE_DIR / "demo.job.yaml").read_text(encoding="utf-8"))
    assert groups == ["haenv.judges.llm_demo"] == a["plugins"]
    assert "haenv.judges.demo" not in groups, \
        "不许与 examples/plugin_demo 共用组名 —— 组名进 job_sha256,要一一对应到一个包"
