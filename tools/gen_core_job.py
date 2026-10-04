"""gen_core_job.py -- one patient list, several ways of asking.

Writes the main pack (`ddx-timeline`) and the tool-track pack (`ddx-workup`) from one
`ddx_case_specs` call. The tool-track pack is a subset of the main pack's cases and differs
from it only by the job-level settings in `CORE_JOBS`, so an effect measured on one carries
back to the other. The generator checks that property before it exits.

Run:
    .venv/bin/python tools/gen_core_job.py

SYNTHETIC data, evaluation use only, not medical advice.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from haenv import kernel_path as _kernel_path       # noqa: E402

if (_kp := _kernel_path()):
    sys.path.insert(0, str(_kp))

from haenv.ddx import ddx_case_specs, strip_inert_gap   # noqa: E402

#: The case_id shape of a base variant (`JD-01` matches, `JD-01v2` doesn't).
_BASE = re.compile(r"^(?!.*v\d+$).+$")

#: The packs that share core, and each one's single job-level difference from the main pack.
#: The self-checks below compare against this table.
CORE_JOBS: dict[str, dict] = {
    "ddx-workup": {"gated": True},
}

#: World knob carried by core, so every pack built from it has the same distractor level.
_DISTRACTOR = "low"


def _carried_noise() -> dict[str, list]:
    """Carry over the armed `noise` from the previous output, keyed by case_id.

    Not reassigned by hash: only cases trialed against the emission gate's preconditions
    are armed, and new case ids stay un-armed. The first existing source wins.
    """
    for rel in ("inputs/ddx-workup.job.yaml",      # the wider pack first: the main pack may carry a subset
                "inputs/ddx-timeline.job.yaml"):
        p = ROOT / rel
        if not p.is_file():
            continue
        doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        got = {str(c["case_id"]): c["latent"]["noise"]
               for c in (doc.get("cases") or []) if (c.get("latent") or {}).get("noise")}
        if got:
            print(f"[core] armed noise 源:{rel}({len(got)} 条)")
            return got
    return {}


_HEAD = """\
# Generated file -- do not edit by hand. Rebuild with the maintainer generator `gen_core_job`, arguments: `{argv}`
#
# Tool-track arm of the benchmark. It differs from the main pack `{main_job}` in one
# job-level setting, {factor}: the patient's history is not handed over, and the agent
# orders (and pays for) each test it wants.
#
# The patients are {scope_text} of the generated set, {n_core} cases ({n_cond} conditions);
# the main pack `{main_job}` is a subset of them, so an effect measured on this arm carries
# back to the main pack.
#
# Differences from the main pack, completely:
#   1. every other `raw` / `latent` field is byte-identical;
#   2. `rhythm_gap` is removed from {n_stripped} cases by `ddx.strip_inert_gap`: this pack
#      is not sliced, so an information gap cannot be delivered.
#
# SYNTHETIC data, evaluation only, not medical advice.
"""


#: The main pack. **From the same `ddx_case_specs` call as core**, with core
#: as its subset of base variants -- this containment relationship is the
#: precondition for "an effect a probe measures can be carried back to the
#: main pack," and it's written here so it doesn't rely on being remembered.
MAIN_JOB = "ddx-timeline"
MAIN_EXTRA = {"findings": True, "slices": "real-rhythm+gap"}

_MAIN_HEAD = """\
# Generated file -- do not edit by hand. Rebuild with the maintainer generator `gen_core_job`, arguments: `{argv}`
#
# Main pack of the benchmark: {n_cond} conditions x {n_var} variants (conditions that need
# no referral get {ind_var}), event density `mixed`, background comorbidities with
# physiological effects.
{subset_note}#
# Distractor level per case (`latent.distractor_level`): {dist_text}.
#
# Tiers, marked per case:
#
# | Tier | Marker | Cases |
# |---|---|---|
# | Long horizon | `index_time_T = 336` | {n_long} |
# | Information gap | `latent.rhythm_gap` | {n_gap} |
# | Insufficient information | `latent.ddx_insufficient` | {n_ins} |
# | Sufficient (the rest) | -- | {n_suff} |
# | Event density dense / sparse | `event_density` | all |
#
# Because the insufficient-information tier sits inside this pack, the pack is mixed:
# the degenerate strategy "always answer that information is insufficient" is penalised
# by over-abstention instead of scoring full marks.
#
# Variants are not independent questions: they share symptom text and gold standard,
# so the case count must not be used as an independent sample size
# (see the docstring of `ddx.ddx_case_specs`).
#
# SYNTHETIC data, evaluation only, not medical advice.
"""


def _core_subset(specs: list[dict], n: int) -> list[dict]:
    """Take `n` cases from all base variants to build core, stratified by condition family,
    with order derived from content (the first `n` after sorting would be all comorbid pairs).
    """
    import collections as _c
    n = n if n > 0 else len(specs)                     # `--core-n 0` = every case in scope
    by = _c.defaultdict(list)
    for c in specs:
        sid = str(c["latent"]["ddx_spec_id"])
        by["HD-COM" if sid.startswith("HD-COM") else
           "HD-IND" if sid.startswith("HD-IND") else
           "HD-UNI" if sid.startswith("HD-UNI") else "JD"].append(c)
    for k in by:
        by[k].sort(key=lambda c: hashlib.sha256(
            str(c["case_id"]).encode()).hexdigest())
    out, i = [], 0
    order = sorted(by)
    while len(out) < n and any(by.values()):          # round-robin, spread proportionally across families
        k = order[i % len(order)]
        if by[k]:
            out.append(by[k].pop(0))
        i += 1
    return sorted(out, key=lambda c: str(c["case_id"]))


def _CANARY() -> str:
    """The canary lines, fetched **live** from `haenv/canary.py` -- never copied a second time."""
    from haenv.canary import block
    return block("# ")


def apply_distractor_level(case: dict, level: str) -> dict:
    """Stamp `latent.distractor_level` on one case. A `high` case also gets its declared
    benign-symptom count raised to the count the injector puts in the ledger
    (`ddx.declare_high_distractor_density`), so `gates.check_event_density` sees declared ==
    injected."""
    from haenv.ddx import declare_high_distractor_density
    case["latent"]["distractor_level"] = level
    if level == "high":
        declare_high_distractor_density(case["latent"])
    return case


def assign_distractor_levels(specs: list[dict], weights: str) -> dict[str, str]:
    """`{case_id: level}` for `weights` like `low:1,high:1`. Cases are ordered by a content
    hash and dealt round-robin over the expanded weight list, so the mix is reproducible and
    independent of list order. A single-level spec (`low:1`) gives every case that level."""
    levels: list[str] = []
    for part in weights.split(","):
        name, w = part.split(":")
        levels += [name.strip()] * int(w)
    order = sorted(specs, key=lambda c: hashlib.sha256(("dist|" + str(c["case_id"])).encode()).hexdigest())
    return {str(c["case_id"]): levels[i % len(levels)] for i, c in enumerate(order)}


def build_main_cases(variants: int, ind_variants: int | None, n_other: int, dist: str,
                     insufficient_frac: float | None) -> list[dict]:
    """The main pack's case list: independent-type conditions (`join_gold == independent`, the
    only ones the referral-free judge applies to) get `ind_variants` variants, every other
    condition `variants` (`n_other` > 0 keeps that many of them, spread over the COM / UNI / JD
    families by hash). Defaults reproduce the historical `variants`-for-all call."""
    from haenv.overlay import condition_registry
    ind_variants = variants if ind_variants is None else ind_variants
    kw = dict(density="mixed", insufficient_frac=insufficient_frac)
    if ind_variants == variants and not n_other:
        allspecs = ddx_case_specs(variants=variants, **kw)
    else:
        reg = condition_registry(include_draft=False)
        ids = list(reg)
        ind = [k for k in ids if reg[k]["join_gold"] == "independent"]
        oth = [k for k in ids if reg[k]["join_gold"] != "independent"]
        if n_other and n_other < len(oth):
            fam = lambda k: "COM" if k.startswith("HD-COM") else "UNI" if k.startswith("HD-UNI") else "JD"
            by: dict[str, list] = {}
            for k in oth:
                by.setdefault(fam(k), []).append(k)
            for f in by:
                by[f].sort(key=lambda k: hashlib.sha256(k.encode()).hexdigest())
            pick, i, fs = [], 0, sorted(by)
            while len(pick) < n_other:
                f = fs[i % len(fs)]
                if by[f]:
                    pick.append(by[f].pop(0))
                i += 1
            oth = [k for k in ids if k in set(pick)]
        allspecs = (ddx_case_specs(only=ind, variants=ind_variants, **kw)
                    + ddx_case_specs(only=oth, variants=variants, **kw))
    lvl = assign_distractor_levels(allspecs, dist)
    noise = _carried_noise()
    out = []
    for c in allspecs:
        c = copy.deepcopy(c)
        apply_distractor_level(c, lvl[str(c["case_id"])])
        if (nz := noise.get(str(c["case_id"]))) is not None:
            c["latent"]["noise"] = copy.deepcopy(nz)
        out.append(c)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=2,
                    help="variants per condition for the main pack (default 2: with the "
                         "condition count at 67, 5 variants would push the main pack past 300)")
    ap.add_argument("--ind-variants", type=int, default=None,
                    help="variants for independent-type conditions (default = --variants)")
    ap.add_argument("--n-other", type=int, default=0,
                    help="keep only this many non-independent conditions (0 = all)")
    ap.add_argument("--dist", default="low:1",
                    help="distractor levels as name:weight,... e.g. low:1,high:1 (default low:1)")
    ap.add_argument("--insufficient-frac", type=float, default=None,
                    help="override the insufficient-tier share (default registry/tiers.yaml)")
    ap.add_argument("--core-n", type=int, default=44,
                    help="cases the probe pack shares with the main pack (default 44; 0 = every "
                         "case in scope)")
    ap.add_argument("--core-scope", choices=("base", "all"), default="base",
                    help="`base`: the probe pack draws from base variants only (`JD-01`, not "
                         "`JD-01v2`); `all`: from every main-pack case. `--core-scope all "
                         "--core-n 0` gives the probe pack the main pack's exact case set")
    ap.add_argument("--case-ids", default=None, metavar="FILE",
                    help="JSON file; the main pack (ddx-timeline) keeps only the case ids listed "
                         "under --case-ids-key. The probe pack (ddx-workup) still carries every "
                         "case, so timeline is a subset of workup")
    ap.add_argument("--case-ids-key", default="timeline_60",
                    help="key inside the --case-ids JSON holding the main pack's case ids "
                         "(default timeline_60)")
    ap.add_argument("--case-ids-scope", choices=("main", "both"), default="main",
                    help="`main`: only ddx-timeline is restricted to --case-ids; `both`: both packs "
                         "carry exactly the listed cases")
    ap.add_argument("--out-dir", default=None, help="write job yaml here (default inputs/)")
    ap.add_argument("--job-suffix", default="", help="appended to both job_ids (candidate runs)")
    a = ap.parse_args()
    out_dir = Path(a.out_dir) if a.out_dir else ROOT / "inputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    main_job = MAIN_JOB + a.job_suffix
    argv = " ".join(sys.argv[1:])
    if a.case_ids:
        # the header must not publish a maintainer-machine path: keep the file name only
        argv = argv.replace(a.case_ids, f"<{Path(a.case_ids).name}>")
    full_cases = build_main_cases(a.variants, a.ind_variants, a.n_other, a.dist,
                                  a.insufficient_frac)
    main_cases = full_cases
    if a.case_ids:
        _raw = json.loads(Path(a.case_ids).read_text(encoding="utf-8"))
        _want = [str(x) for x in (_raw[a.case_ids_key] if isinstance(_raw, dict) else _raw)]
        _have = {str(c["case_id"]) for c in full_cases}
        _miss = sorted(set(_want) - _have)
        if _miss or len(set(_want)) != len(_want) or not _want:
            print(f"[core] 🔴 --case-ids: {len(_miss)} ids not generated by this call "
                  f"({_miss[:4]}), duplicates {len(_want) - len(set(_want))}, listed {len(_want)}. Stop.")
            return 2
        _ws = set(_want)
        main_cases = [c for c in full_cases if str(c["case_id"]) in _ws]
        print(f"[core] main pack restricted to {len(main_cases)} of {len(full_cases)} cases "
              f"({a.case_ids}:{a.case_ids_key})")
        if a.case_ids_scope == "both":
            full_cases = main_cases
            print(f"[core] --case-ids-scope both: the probe pack carries the same {len(main_cases)} cases")
    n_cond = len({str(c["latent"]["ddx_spec_id"]) for c in full_cases})
    n_cond_main = len({str(c["latent"]["ddx_spec_id"]) for c in main_cases})
    pool = [c for c in full_cases if a.core_scope == "all" or _BASE.match(str(c["case_id"]))]
    core = [copy.deepcopy(c) for c in _core_subset(pool, a.core_n)]
    n_armed = sum(1 for c in core if c["latent"].get("noise"))
    noise = _carried_noise()
    print(f"[core] {len(core)} 例({'全部变体' if a.core_scope == 'all' else '基础变体'})· "
          f"noise armed {n_armed} (源 {len(noise)} 条,命中 {n_armed})")
    # The armed count may only go up: `_carried_noise` reads the previous output, so a case
    # core drops loses its arming for good. A drop exits non-zero.
    if len(noise) and n_armed < len(noise):
        print(f"[core] 🔴 **armed 的 noise 从 {len(noise)} 跌到 {n_armed}** —— "
              f"硬门 `acted_on_unverified_signal` 的行使面在衰减,而它只能靠"
              f"上一版产物传下来、没有路径加回来。\n"
              f"        丢掉的 case_id:"
              f"{sorted(set(noise) - {c['case_id'] for c in core if c['latent'].get('noise')})}\n"
              f"        处置:要么把 core 拉回覆盖它们,要么**显式**接受并补登记。停。")
        return 2
    if n_armed == 0:
        print("[core] 🔴 一个 armed 的 noise 都没有 —— 硬门 "
              "`acted_on_unverified_signal` 会**零行使面**。停。")
        return 2

    scope_text = ("every case" if (a.core_scope == "all" and a.core_n == 0) else
                  "a stratified subset of the cases" if a.core_scope == "all" else
                  "base variants, stratified by condition family, ")
    scope_text = scope_text.rstrip(", ")
    _stripped_by_job: dict[str, int] = {}
    job_ids: dict[str, str] = {}
    for job_id, extra in CORE_JOBS.items():
        fac = (" · ".join(f"`{k}: {v}`" for k, v in extra.items())
               or "**没有 `slices` / `findings`**(= 不切片、不给化验台账,"
                  "由 `gated` 几何逐项买)")
        # A pack that is not sliced cannot deliver an information gap, so its cases do not
        # carry `rhythm_gap`; the header states how many were stripped.
        _cases = copy.deepcopy(core)
        _stripped = strip_inert_gap(_cases, extra.get("slices"))
        _stripped_by_job[job_id] = _stripped
        if _stripped:
            print(f"[core]   ↳ `{job_id}` 不切片 ⇒ 摘掉 {_stripped} 例的 "
                  f"`rhythm_gap`(声明了也交付不了)")
        jid = job_id + a.job_suffix
        job_ids[job_id] = jid
        doc = {"job_id": jid, "task_type": "joint_dx", "multiround": False,
               "include_baseline": True, "models": [], "sample_cases": 5,
               "report": f"eval-{jid}.md", **extra,
               "cases": _cases}
        p = out_dir / f"{jid}.job.yaml"
        # The generator writes the canary itself, so a regenerated pack keeps it.
        p.write_text(_CANARY() + _HEAD.format(factor=fac, n_stripped=_stripped, argv=argv,
                                  main_job=main_job, scope_text=scope_text,
                                  n_core=len(core), n_cond=n_cond)
                     + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False),
                     encoding="utf-8")
        print(f"[core] -> {p}  ({len(core)} 例 · {fac})")

    # ---- Main pack: every variant from the same call ----
    # Note: core comes out of the main pack's own case list, so its world knobs
    #    (`distractor_level` / armed `noise` / the declaration a `high` level needs) are
    #    carried together by construction -- "probe pack subset of main pack" is the
    #    property this generator exists to establish.
    doc = {"job_id": main_job, "task_type": "joint_dx", "multiround": False,
           "include_baseline": True, "models": [], "sample_cases": 5,
           "report": f"eval-{main_job}.md", **MAIN_EXTRA, "cases": main_cases}
    pm = out_dir / f"{main_job}.job.yaml"
    # The tier-table numbers in the header are computed from the cases.
    from haenv.evaluate import rhythm_gap_feasible as _feas
    import collections as _col
    _n_long = sum(1 for c in main_cases if _feas(int(c["latent"]["index_time_T"])))
    _n_gap = sum(1 for c in main_cases if c["latent"].get("rhythm_gap"))
    _n_ins = sum(1 for c in main_cases if c["latent"].get("ddx_insufficient"))
    _dist = _col.Counter(c["latent"]["distractor_level"] for c in main_cases)
    pm.write_text(_CANARY() + _MAIN_HEAD.format(
                      n_core=len(core), n_var=a.variants, argv=argv,
                      ind_var=a.variants if a.ind_variants is None else a.ind_variants,
                      dist_text=", ".join(f"{k} {v}" for k, v in sorted(_dist.items())),
                      n_cond=n_cond_main, n_long=_n_long,
                      subset_note=(f"#\n# Restricted to {len(main_cases)} of the {len(full_cases)} generated cases "
                                   f"(`--case-ids`); every one of them is also in the tool-track pack.\n"
                                   if a.case_ids else ""), n_gap=_n_gap, n_ins=_n_ins,
                      n_suff=len(main_cases) - _n_ins)
                  + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False),
                  encoding="utf-8")
    print(f"[core] -> {pm}  ({len(main_cases)} 例 · 主力包)")
    print(f"[core]    档位:长视野 {_n_long} · 缺口 {_n_gap} · 不足 {_n_ins} · "
          f"充分 {len(main_cases) - _n_ins} · 干扰 {dict(_dist)}")
    # Both the insufficient tier and the sufficient tier must be non-empty: with only the
    # first, "always says insufficient data" gets full marks; with only the second,
    # abstention has nothing to check.
    if _n_ins == 0 or _n_ins == len(main_cases):
        print(f"[core] 🔴 不足档 {_n_ins} / {len(main_cases)} —— 主力包不再是混批,"
              f"过度弃权那条罚失去行使面(它是 `mixed` 并进来时唯一要守的性质)。停。")
        return 2

    # Self-check 2: core **must** be a byte-identical subset of the main pack
    # (this is the whole point of the change)
    _mb = {str(c["case_id"]): json.dumps(c, sort_keys=True, ensure_ascii=False)
           for c in main_cases}
    _cb = {str(c["case_id"]): json.dumps(c, sort_keys=True, ensure_ascii=False)
           for c in core}
    if a.case_ids:
        # main pack is the narrower one: main ⊆ core, byte for byte
        _bad = [k for k, v in _mb.items() if _cb.get(k) != v]
        print(f"[core] {'✅' if not _bad else '🔴'} 主力包 ⊆ core 逐字节"
              f"{'成立' if not _bad else f' **不成立**,差 {len(_bad)} 例:{_bad[:4]}'}")
    else:
        _bad = [k for k, v in _cb.items() if _mb.get(k) != v]
        print(f"[core] {'✅' if not _bad else '🔴'} core ⊂ 主力包 逐字节"
              f"{'成立' if not _bad else f' **不成立**,差 {len(_bad)} 例:{_bad[:4]}'}")
    if a.core_scope == "all" and a.core_n == 0 and not a.case_ids:
        _same = set(_cb) == set(_mb)
        print(f"[core] {'✅' if _same else '🔴'} 探针包例集 == 主力包例集({len(_cb)}/{len(_mb)})")
        if not _same:
            return 1

    # ---- Self-check 3: a probe pack's difference from the main pack is
    # **only ever allowed to be the registered one** ----
    #
    # Shared cases are byte-identical apart from `rhythm_gap`, and the number of cases that
    # differ on `rhythm_gap` equals the number stripped.
    _mb_raw = {str(c["case_id"]): c for c in full_cases}
    ok = True
    for job_id in CORE_JOBS:
        d = yaml.safe_load((out_dir / f"{job_ids[job_id]}.job.yaml").read_text(encoding="utf-8"))
        n_gap_diff, n_other_diff, examples = 0, 0, []
        for c in d["cases"]:
            m = _mb_raw.get(str(c["case_id"]))
            if m is None:
                n_other_diff += 1
                examples.append(f"{c['case_id']}(主力包里没有)")
                continue
            a_, b_ = copy.deepcopy(c), copy.deepcopy(m)
            ga = a_["latent"].pop("rhythm_gap", None)
            gb = b_["latent"].pop("rhythm_gap", None)
            if ga != gb:
                n_gap_diff += 1
            if (json.dumps(a_, sort_keys=True, ensure_ascii=False)
                    != json.dumps(b_, sort_keys=True, ensure_ascii=False)):
                n_other_diff += 1
                examples.append(str(c["case_id"]))
        want = _stripped_by_job.get(job_id, 0)
        good = (n_other_diff == 0 and n_gap_diff == want)
        ok = ok and good
        print(f"[core] {'✅' if good else '🔴'} `{job_ids[job_id]}` vs 生成全集:"
              f"除 `rhythm_gap` 外差 {n_other_diff} 例(须 0)· "
              f"`rhythm_gap` 差 {n_gap_diff} 例(须 == 摘掉的 {want})"
              + (f" · 例:{examples[:4]}" if examples else ""))
    return 0 if (ok and not _bad) else 1


if __name__ == "__main__":
    raise SystemExit(main())
