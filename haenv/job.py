"""Input contract: a raw case plus its latent control variables.

A user writes one job file. Each case supplies ``raw`` -- the case facts -- and
``latent`` -- the controls that determine the gold. Everything after that is
automatic: premise checks, conditioned generation, noise derivation, iterative
verification, multi-model evaluation and the report.

Latent controls never reach the solver.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

from .yamlcache import load_yaml as _cached_yaml

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import yaml

log = logging.getLogger("haenv.job")

TASK_TYPES = ("early_warning", "tracking_review", "joint_dx", "hardprob")
OUTCOMES = ("regain", "maintain")
DISTRACTOR_LEVELS = ("none", "low", "high")


# ================================================ latent key registry
#
# Every latent key maps to the gold or premise field that consumes it
# (`EXEMPT` = never enters the gold; None = registered but unread, warns).
# An unregistered key raises.
EXEMPT = "EXEMPT"

LATENT_REGISTRY: dict[str, tuple[str | None, str]] = {
    "outcome":          ("build.HaenvGenerator -> raw.outcome_label", "regain/maintain 决定 event_occurred"),
    "driver":           ("build.HaenvGenerator -> raw.gold_drivers + adjudication.primary_driver", ""),
    "reversal_week":    ("build.HaenvGenerator -> raw.reversal_points[].week", "Track E 迟滞的判据来源"),
    "regain_slope":     ("build._weight_series", "回升斜率"),
    # Regain endpoint in kg; when given, it back-solves the slope so the total
    # gain does not depend on the sampled course length.
    "regain_end_kg":    ("build.HaenvGenerator -> build._weight_series(slope 反解)",
                         "回升终点(kg)。由 (end − nadir) / 回升周数 反解斜率;"
                         "缺省 None ⇒ 沿用 `regain_slope`"),
    "index_time_T":     ("build.build_case(T) -> prediction_context.prediction_time_T", ""),
    "course_end_day":   ("build.course_end_of -> prediction_context.prediction_window "
                         "+ dose/adherence/临床流的地平线",
                         "逐例病程长度;缺省时按 case_id 从 COURSE_END_DOMAIN 采样"),
    # Day of the nadir (default: the index time). Diagnosis cases set it so the
    # weight trend can span months; ignored on cases with a reversal.
    "nadir_day":        ("build._weight_series(desc_end)",
                         "最低点落在第几天;缺省 None ⇒ 沿用 `index_time_T`"),
    "distractor_level": ("build.build_case -> noise.inject_distractors", ""),
    "distractor_cond_axes": ("events.facts_of -> Facts.cond_axes -> "
                             "events._event_ok(准入) + events.event_prior(乘子)",
                             "开哪几根条件轴;取值域见 events.CONDITION_AXIS_SPECS,"
                             "未登记的轴名当场抛(default-deny)"),
    "noise":            ("build.build_case -> noise.inject", ""),
    "event_density":    ("build.premise_spec -> premise.event_density", "events.py 据此定条数与步长"),
    "adherence_low":    ("build.premise_spec -> premise.adherence.trajectory", ""),
    "drug_response":    ("build.premise_spec -> meta.drug_response -> drug_effects.response_for",
                         "个体药效响应系数(HbA1c 与空腹血糖共用);缺省按 case_id 在驱动所属区间内抽,"
                         "显式值必须落在驱动的区间里(不响应 vs 其余),否则载入即抛"),
    "missingness":      ("build.premise_spec -> premise.adherence.missingness_mechanism", ""),
    "target_event":     ("build.premise_spec -> meta.target_event_type", ""),
    "difficulty":       (EXEMPT, "只进 meta.difficulty_class 作记账,不参与任何判定"),
    "rhythm_gap":       ("build.premise_spec -> meta.rhythm_gap -> evaluate.slices_for",
                         "只在 `slices: real-rhythm+gap` 下生效:把该例的某个就诊间隔"
                         "强制拉长到 REAL_RHYTHM_GAP_DAYS(造信息缺口)。"
                         "真实门诊 8.9% 的间隔 >180 天,合成语料默认不含这类长间隔"),
    # Long-horizon tier marker for stratified reading (not a world parameter).
    "long_horizon_tier": ("ddx.long_horizon_for -> latent -> 分层读数(报告/分析侧)",
                          "长视野档(≈1 年视野)的逐例标记。缺口档在它内部抽 ⇒ "
                          "`long_horizon_tier ∧ ¬rhythm_gap` 就是缺口效应的匹配对照臂。"
                          "它不进世界层计算:`index_time_T` 已经承载了那个数"),
    "ddx_spec_id":      ("build.HaenvGenerator -> adjudication.ddx.spec_id",
                         "内核 spec key(JD-PCOS…);不能用作 case_id(会把答案写进题号)"),
    "ddx_diagnosis":    ("build.HaenvGenerator -> adjudication.ddx.diagnosis", "dx_hit 的判据"),
    "ddx_aliases":      ("build.HaenvGenerator -> adjudication.ddx.aliases", "dx_hit 的文本匹配句柄"),
    "ddx_join_gold":    ("build.HaenvGenerator -> adjudication.ddx.join_gold",
                         "unified|comorbidity|independent —— spec §6(c) 跨时间联合归因的正解"),
    "ddx_threads":      ("build.HaenvGenerator -> adjudication.ddx.threads",
                         "comorbidity 的病线声明(overlay.THREADS);缺它判据只能扁平字面匹配 → 假阴性"),
    "ddx_tests":        ("build.HaenvGenerator -> adjudication.ddx.tests", "应索取的检查(动作侧金标)"),
    # Insufficient-information tier: index time before the first symptom; scores
    # abstention, not diagnosis.
    "ddx_insufficient": ("build.HaenvGenerator -> adjudication.ddx.insufficient;"
                         "judges.judge_abstention_calibration / tracks.ddx_hit",
                         "True = 该例信息不足,正解是声明不足而不是压一个诊断上去"),
    "ddx_specialty":    ("build.HaenvGenerator -> adjudication.ddx.specialty", "应转诊科室(动作侧金标)"),
    "ddx_urgency":      ("build.HaenvGenerator -> adjudication.ddx.urgency", "🟢🟡🟠 分级"),
    "ddx_red_flag":     ("build.HaenvGenerator -> adjudication.red_flag_present",
                         "武装 missed_emergency_red_flag 硬门"),
    "ddx_clinician_warranted": ("build.HaenvGenerator -> adjudication.clinician_action_warranted", ""),
    "ddx_outcome_label": ("build.HaenvGenerator -> raw.outcome_label",
                          "内核规格自带,优先于按体重方向推断"),
}


def check_latent_registry(latent: dict, where: str) -> list[str]:
    """Raise on an unregistered key; return warnings for keys with no consumer."""
    warns: list[str] = []
    for k in latent or {}:
        if k not in LATENT_REGISTRY:
            raise ValueError(
                f"{where}: latent key {k!r} is not registered in job.LATENT_REGISTRY -- "
                f"say which ground-truth/premise field it maps to, or mark it EXEMPT and "
                f"explain why it does not enter the ground truth. (An unregistered key would "
                f"silently do nothing: that truth never reaches the question, and scoring "
                f"would run against a placeholder.) If the key comes from a plugin package, "
                f"check that the package is installed.")
        consumer = LATENT_REGISTRY[k][0]
        if consumer is None:
            warns.append(f"{k}: registered but has no consumer ({LATENT_REGISTRY[k][1]})")
    return warns


# ============================================== generation knobs: outcome and driver
#
# Outcome and driver are sampled generation knobs, not extracted facts. These
# functions are their only source; callers must not inline their own defaults.

#: Outcome -> driver. Only these two drivers are realised by the world model.
DRIVER_BY_OUTCOME: dict[str, str] = {
    "regain":   "poor_medication_adherence",
    "maintain": "unknown_or_multifactorial",
}


def outcome_of(case_id: str, latent: dict | None = None) -> str:
    """The declared outcome, otherwise one sampled from ``case_id`` alone."""
    if latent and latent.get("outcome") is not None:
        return str(latent["outcome"])
    from . import rng
    return str(rng.pick(list(OUTCOMES), case_id, "outcome"))


def driver_of(case_id: str, latent: dict | None = None) -> str:
    """The declared driver, otherwise the one `DRIVER_BY_OUTCOME` gives for the
    outcome (the two are coupled in generation)."""
    if latent and latent.get("driver") is not None:
        return str(latent["driver"])
    return DRIVER_BY_OUTCOME[outcome_of(case_id, latent)]


def raw_field(raw: dict, key: str):
    """Read one raw field, falling back to the registered defaults (the only
    place raw defaults are read). An unregistered key raises ``KeyError``.
    """
    v = (raw or {}).get(key)
    if v is not None:
        return v
    if key in CaseSpec.RAW_DEFAULTS:
        return CaseSpec.RAW_DEFAULTS[key]
    if key in CaseSpec.RAW_DERIVED_DEFAULTS:
        return CaseSpec.RAW_DERIVED_DEFAULTS[key](raw or {})
    raise KeyError(f"raw field {key!r} has no registered default; add it to CaseSpec.RAW_DEFAULTS")


@dataclass
class CaseSpec:
    case_id: str
    raw: dict                      # case facts: condition, drug, demographics, devices, weight anchors
    latent: dict                   # latent controls: outcome, driver, reversal, adherence, noise, distractors
    #: The geometry the job runs the case in (`mount_table.claim_geometry`), with what decides
    #: how far the solver sees: `{"name": "multi", "cadence_days": ...}`,
    #: `{"name": "slices", "slices": ...}`, or just the name. Set by `load_job`; `None` reads
    #: as single-shot. The emission gate reads it.
    geometry: dict | None = None

    # ---- raw field defaults: the only copy (read through `raw_field`) ----
    RAW_DEFAULTS: ClassVar[dict] = {
        "disease": "obesity",
        # the premise builder's dose ladder is this drug's titration schedule
        "drug": "tirzepatide",
        "sex": "F",
        "age_range": "45-49",
        "start_weight": 98.0,
        "sampling_days": 1,
        "comorbidities": [],
    }
    #: Defaults computed from other fields.
    RAW_DERIVED_DEFAULTS: ClassVar[dict] = {
        "nadir_weight": lambda raw: float(raw_field(raw, "start_weight")) - 12.0,
    }

    # ---- convenience readers ----
    @property
    def outcome(self) -> str:
        return outcome_of(self.case_id, self.latent)

    @property
    def driver(self) -> str:
        return driver_of(self.case_id, self.latent)

    @property
    def index_time_T(self) -> int:
        """The index time, read from ``latent`` (it is a generation knob)."""
        _stray = self.raw.get("index_time_T")
        if _stray is not None:
            log.warning("[job] %s: `index_time_T` is under `raw` (it belongs under `latent`) -- "
                        "it is a generation knob and has no effect on the raw side", self.case_id)
        return int(self.latent.get("index_time_T", 84))

    @property
    def distractor_level(self) -> str:
        return self.latent.get("distractor_level", "none")

    @property
    def noise(self) -> list[dict]:
        return self.latent.get("noise", []) or []

    @property
    def ddx(self) -> dict:
        """Verifier-only diagnosis gold (lands in ``adjudication``); ``{}`` otherwise."""
        d = {k[4:]: v for k, v in self.latent.items()
             if k.startswith("ddx_") and k not in ("ddx_red_flag", "ddx_clinician_warranted",
                                                   "ddx_outcome_label")}
        return d if d.get("diagnosis") else {}


@dataclass
class Job:
    job_id: str
    task_type: str
    cases: list[CaseSpec]
    models: list[str] = field(default_factory=list)     # empty: use the configured default models
    include_baseline: bool = True                        # offline deterministic reference
    multiround: bool = False                             # tracking review runs multi-round
    cadence_days: int = 28
    sample_cases: int | None = None
    report_path: str | None = None
    root: Path = field(default=Path("."), repr=False)
    path: str | None = None          # path to this job file; provenance records its hash
    batch: str = ""                                      # batch timestamp, YYYYmmdd-HHMMSS
    slices: str | list = ""                              # multi-slice consultation: "auto", an explicit list, or empty for none
    # Question phrasing; empty = the task type's default.
    probe_id: str = ""
    # Allow a retired probe (to reproduce old batches).
    allow_retired: bool = False
    # Gold provenance block written by the extraction tool; empty for a
    # hand-written job (reported as "not recorded").
    provenance: dict = field(default_factory=dict)
    # Render labs into the evidence ledger (off by default).
    findings: bool = False
    # Tool-track geometry: signals are withheld and must be queried.
    gated: bool = False
    # Judge plugin entry-point groups. Only declared groups load (never scanned
    # implicitly), so the set of judges is part of the job hash.
    plugins: list[str] = field(default_factory=list)

    # World plugin groups. They move the world stamp, while judge plugins move
    # the scoring fingerprint. `plugins:` as a bare list means judge groups.
    world_plugins: list[str] = field(default_factory=list)

    # ---- artefact paths: always under a per-batch timestamped directory ----
    @property
    def results_dir(self) -> Path:
        return self.root / f"results/{self.task_type}/{self.job_id}/{self.batch}"

    @property
    def reports_dir(self) -> Path:
        return self.root / f"reports/{self.job_id}/{self.batch}"

    @property
    def results_file(self) -> Path:
        return self.results_dir / "eval.jsonl"

    @property
    def report_file(self) -> Path:
        name = Path(self.report_path).name if self.report_path else f"eval-{self.job_id}.md"
        return self.reports_dir / name

    @property
    def verify_results_file(self) -> Path:
        return self.results_dir / "verify.jsonl"

    @property
    def cases_file(self) -> Path:
        """The emitted cases, including verifier-only gold, one per line. Evaluation
        and reporting read this file rather than rebuilding cases.
        """
        return self.results_dir / "cases.jsonl"

    @property
    def verify_report_file(self) -> Path:
        return self.reports_dir / f"verify-{self.job_id}.md"


def _need(d: dict, key: str, where: str):
    if key not in d:
        raise ValueError(f"{where}: missing required field '{key}'")
    return d[key]


def load_job(path: str | Path, root: Path | None = None) -> Job:
    """Load and validate a job file; validation failures raise."""
    p = Path(path)
    data = _cached_yaml(p)
    if not isinstance(data, dict):
        raise ValueError(f"{p}: a job file must be a YAML mapping")

    job_id = _need(data, "job_id", str(p))
    task_type = data.get("task_type", "early_warning")
    if task_type not in TASK_TYPES:
        raise ValueError(f"{p}: task_type={task_type!r} is not in {TASK_TYPES}")

    raw_cases = _need(data, "cases", str(p))
    if not raw_cases:
        raise ValueError(f"{p}: cases is empty")

    # Judge plugins attach before case validation: they may register latent keys.
    _judge_groups = _plugin_groups(data.get("plugins"))[0]
    _load_judge_groups(_judge_groups)

    cases: list[CaseSpec] = []
    seen: set[str] = set()
    for i, c in enumerate(raw_cases, 1):
        where = f"{p}#cases[{i}]"
        cid = _need(c, "case_id", where)
        if cid in seen:
            raise ValueError(f"{where}: duplicate case_id {cid!r}")
        seen.add(cid)
        raw = c.get("raw", {}) or {}
        lat = c.get("latent", {}) or {}
        if lat.get("outcome", "regain") not in OUTCOMES:
            raise ValueError(f"{where}: latent.outcome must be one of {OUTCOMES}")
        if lat.get("distractor_level", "none") not in DISTRACTOR_LEVELS:
            raise ValueError(f"{where}: latent.distractor_level must be one of {DISTRACTOR_LEVELS}")
        for nz in (lat.get("noise") or []):
            if "class" not in nz or "week" not in nz:
                raise ValueError(f"{where}: each latent.noise entry needs class and week")
        if lat.get("drug_response") is not None:
            from .drug_effects import check_declared_response
            _why = check_declared_response(float(lat["drug_response"]), driver_of(cid, lat))
            if _why:
                raise ValueError(f"{where}: {cid}: {_why}")
        for w in check_latent_registry(lat, where):
            log.warning("[job] %s latent %s", cid, w)
        cases.append(CaseSpec(case_id=cid, raw=raw, latent=lat))

    # Findings and physiology switches are set here, once per job load.
    from . import events as _ev
    _ev.FINDINGS_ENABLED[0] = bool(data.get("findings") or False)
    # Physiology layer: on by default; `physio: false` turns it off.
    _ev.PHYSIO_ENABLED[0] = bool(data.get("physio", True))

    # New ledger scope per job: the same case id can be a different world in
    # another job.
    from . import wq as _wq
    _wq.enter_scope(job_id)

    _pg = _plugin_groups(data.get("plugins"))

    job = Job(
        job_id=job_id, task_type=task_type, cases=cases,
        models=data.get("models") or [],
        include_baseline=bool(data.get("include_baseline", True)),
        multiround=bool(data.get("multiround", task_type == "tracking_review")),
        cadence_days=int(data.get("cadence_days", 28)),
        sample_cases=data.get("sample_cases"),
        slices=data.get("slices") or "",
        probe_id=str(data.get("probe_id") or ""),
        findings=bool(data.get("findings") or False),
        gated=bool(data.get("gated") or False),
        plugins=_pg[0], world_plugins=_pg[1],
        report_path=data.get("report"),
        provenance=dict(data.get("_provenance") or {}),
        root=Path(root) if root else p.resolve().parent.parent,
        path=str(p.resolve()),
    )
    from .mount_table import claim_geometry
    _geo = {"name": claim_geometry(job)}
    if _geo["name"] == "multi":
        _geo["cadence_days"] = job.cadence_days
    elif _geo["name"] == "slices":
        _geo["slices"] = job.slices
    for c in cases:
        c.geometry = dict(_geo)
    check_probe_geometry(job)
    load_plugins(job)
    # World plugins attach before any case is built.
    load_world_plugins(job)
    log.info("[job] %s task=%s cases=%d models=%s multiround=%s",
             job.job_id, job.task_type, len(job.cases), job.models or "(default)", job.multiround)
    return job


def _plugin_groups(raw) -> tuple[list[str], list[str]]:
    """Both spellings of ``plugins:`` mapped to (judge groups, world groups).

    A bare list is judge groups only; a mapping is read per side. Unknown keys
    raise.
    """
    if not raw:
        return [], []
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw], []
    if isinstance(raw, dict):
        unknown = sorted(set(raw) - {"judges", "world"})
        if unknown:
            raise ValueError(
                f"job.yaml `plugins:` has unknown keys {unknown}; only `judges` / `world` are allowed")
        return ([str(x) for x in (raw.get("judges") or [])],
                [str(x) for x in (raw.get("world") or [])])
    raise ValueError(f"job.yaml `plugins:` must be a list or a {{judges,world}} mapping, got {type(raw).__name__}")


def load_world_plugins(job) -> int:
    """Load the world-side plugins a job declares; returns the number of entries
    attached. Must run before any case is built.
    """
    if not getattr(job, "world_plugins", None):
        return 0
    from .world_plugins import load_world_plugins as _load
    from .world_plugins import manifest
    n = 0
    for group in job.world_plugins:
        n += _load(group)
    if n:
        log.info("[job] %d world plugin(s): %s", n, manifest()["entries"])
    else:
        log.warning("[job] world plugins=%s declared, but none registered -- "
                    "misspelled group name, or the package is not installed?", job.world_plugins)
    return n


def load_plugins(job) -> list[str]:
    """Load the judge plugins a job declares; returns the names attached. Load
    failures raise.
    """
    return _load_judge_groups(getattr(job, "plugins", None) or [])


#: Judge groups already attached in this process (a job load reaches
#: `_load_judge_groups` twice; duplicate registration raises).
_LOADED_GROUPS: set[str] = set()


def _load_judge_groups(groups) -> list[str]:
    if not groups:
        return []
    from .judges import load_judge_plugins, registry_manifest
    added: list[str] = []
    for group in groups:
        if group in _LOADED_GROUPS:
            continue
        got = load_judge_plugins(group)
        if not got:
            log.warning("[job] plugins group %r declared, but no judge registered -- "
                        "misspelled group name, or the package is not installed?", group)
        added += got
        _LOADED_GROUPS.add(group)
    if added:
        _m = registry_manifest()
        log.info("[job] %d plugin judge(s): %s (built-in %d · total %d)",
                 len(added), added, _m["n_core"], _m["n_total"])
    return added


#: Probes that require the respondent to submit a process trace.
TRACE_PROBES: frozenset[str] = frozenset({"ddx.trace"})


#: Geometries whose row builder runs the process judges (from the mount registry).
def _geoms_running_process_judges() -> frozenset[str]:
    from .mounting import MOUNTS
    return frozenset(m.geometry for m in MOUNTS if "run_process_judges" in m.judges)


#: Geometry for a job switch, asked of the live resolver.
def _geometry_of_flag(flag: str) -> str:
    from .mount_table import claim_geometry

    class _Probe:
        gated = False
        slices = ""
        multiround = False

    p = _Probe()
    setattr(p, flag, "auto" if flag == "slices" else True)
    return claim_geometry(p, None)


def check_probe_geometry(job: "Job") -> None:
    """Raise if a trace-requiring probe is combined with a geometry that does not
    run the process judges (the trace would be discarded unjudged).
    """
    if str(job.probe_id or "") not in TRACE_PROBES:
        return
    ok = _geoms_running_process_judges()
    bad = [f"{n} (geometry {_geometry_of_flag(n)})"
           for n, on in (("gated", bool(job.gated)),
                         ("slices", bool(job.slices)),
                         ("multiround", bool(job.multiround)))
           if on and _geometry_of_flag(n) not in ok]
    if bad:
        raise ValueError(
            f"[job] {job.job_id}: probe {job.probe_id!r} requires a process trace, "
            f"but this job's geometry is {'+'.join(bad)}, "
            f"and `run_process_judges` is only called for {sorted(ok)}; "
            "other geometries discard the trace the model submits. "
            "Remove that geometry flag, or wire run_process_judges into it first.")
