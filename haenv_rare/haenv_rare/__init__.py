"""haenv 罕见病**编码**判据(`rc_*`)—— 仓外插件包。

设计:`docs/design/2026-09-20-rare-gen-full.md` §4。全部 `category: deterministic`。

读什么:`out._raw`(solver 整份 JSON,`evaluate.py:1175` 挂上)的 `assertions[]` / `diagnosis` / `gene`;
金标:`vp.adjudication["rare"]`(`hpo_gold[]`,按 `idx` 与 Q 侧台账 `real_symptom_evidence_ids` 对齐)。

纪律:映射不到记 0 不猜;不适用记 None 并给 `rc_na_reason`(spec §32);键名一律 `rc_` 前缀。
SYNTHETIC,仅评测用,非医疗建议。
"""
from __future__ import annotations

NAME = "rare_coding"
KEYS = ("rc_n_gold", "rc_n_assert", "rc_coverage", "rc_hpo_strict", "rc_hpo_hier",
        "rc_polarity_ok", "rc_subject_ok", "rc_wrong_rate", "rc_negation_trap",
        "rc_orpha_top1", "rc_hgnc_ok",
        # 层2(VCF/PED 附件)维度(2026-09-21)。金标 = `adjudication.rare.attachments.genome`
        # (与 R5 回读同一份真值);无基因组附件的例(获得性)全部 None。
        "rc_n_variants", "rc_variant_hit", "rc_variant_gt_ok", "rc_variant_inh_ok",
        "rc_gene_from_variant",
        # 层4(DICOM 指针)维度(2026-09-21)。金标 = `attachments.imaging.series[*].sha256`;
        # `rc_signal_phi_free` 读文件头里的患者标识(与 R6 同一读法)并断言不出现在 solver 输出里。
        "rc_signal_index_ok", "rc_signal_phi_free",
        # Case-report (journal-style Markdown) dimension. Gold = `attachments.narrative.spans`
        # (each {hpo_id, polarity, subject, start, end}; hpo_id None marks a life-event noise
        # sentence). Every key is None for a case without a report.
        "rc_narr_recall", "rc_narr_span_ok", "rc_narr_noise_abstain", "rc_narr_polarity_ok",
        # EEG (EDF / EDF+) pointers. Gold = `attachments.eeg.recordings[*]`; None without recordings.
        "rc_eeg_index_ok", "rc_eeg_phi_hidden",
        # A VCF that is not GRCh38 by design (external Exomiser case): refusing it is the right answer.
        "rc_vcf_build_refused",
        "rc_na_reason")

_NA = {k: None for k in KEYS}


def _ont():
    from .gen import Ontology
    return Ontology.get()


def _expected_inheritance(gts: dict) -> str | None:
    """由真值基因型推期望的遗传来源;单样本无家长 ⇒ None(不适用)。"""
    if "father" not in gts or "mother" not in gts:
        return None
    def carries(g):
        return any(a not in ("0", ".") for a in str(g).replace("|", "/").split("/"))
    f, m = carries(gts["father"]), carries(gts["mother"])
    return "biparental" if f and m else "paternal" if f else "maternal" if m else "de_novo"


def _judge_variants(raw: dict, rare: dict, gene_ok: int | None) -> dict:
    gen = (rare.get("attachments") or {}).get("genome") or {}
    tv = gen.get("variant")
    if not tv:
        return {"rc_n_variants": None, "rc_variant_hit": None, "rc_variant_gt_ok": None,
                "rc_variant_inh_ok": None, "rc_gene_from_variant": None}
    vs = [v for v in (raw.get("variants") or []) if isinstance(v, dict)]
    key = (str(tv["chrom"]).replace("chr", ""), int(tv["pos"]), tv["ref"], tv["alt"])
    hit = None
    for v in vs:
        try:
            if (str(v.get("chrom", "")).replace("chr", ""), int(v.get("pos", -1)), v.get("ref"), v.get("alt")) == key:
                hit = v
                break
        except (TypeError, ValueError):
            continue
    gts = gen.get("genotypes") or {}
    exp_inh = _expected_inheritance(gts)
    gt_ok = inh_ok = None
    if hit is not None:
        gt_ok = 1 if str(hit.get("gt") or "").replace("|", "/") == str(gts.get("proband") or "") else 0
        if exp_inh is not None:
            inh_ok = 1 if str(hit.get("inheritance") or "") == exp_inh else 0
    method = str(((raw.get("gene") or {}).get("method")) or "") if isinstance(raw.get("gene"), dict) else ""
    # 基因「由变异得出」要三件事同时成立:基因对、方法声明为 variant…、**且真值变异在列表里**。
    # 只看方法字符串会被「说了 variant 却没列出那条变异」骗过(负对照 2026-09-21 抓到)。
    gfv = None if gene_ok is None else (1 if gene_ok == 1 and method.startswith("variant") and hit is not None else 0)
    return {"rc_n_variants": len(vs), "rc_variant_hit": 1 if hit is not None else 0,
            "rc_variant_gt_ok": gt_ok, "rc_variant_inh_ok": inh_ok, "rc_gene_from_variant": gfv}


def _dicom_identifiers(z) -> set[str]:
    """PatientName / PatientID / PatientBirthDate 的**值**(只在判据进程里,用来断言不在输出里)。

    🔴 用 pydicom 读,不用裸字节扫 tag:TCIA 的 RTSTRUCT/MR 多为 implicit VR,裸扫
    (`rare._dicom_phi` 那种)找不到 VR 字节就返回空 —— 于是「没 PHI」与「没读到」同形,
    判据恒绿。2026-09-21 负对照抓到(把假名 ID 注进输出,判据仍给 1)。
    """
    import io, pathlib, zipfile
    import pydicom
    vals: set[str] = set()
    try:
        with zipfile.ZipFile(pathlib.Path(z)) as zf:
            for n in [x for x in zf.namelist() if x.endswith(".dcm")][:2]:
                ds = pydicom.dcmread(io.BytesIO(zf.read(n)), stop_before_pixels=True)
                for t in ("PatientName", "PatientID", "PatientBirthDate"):
                    v = str(ds.get(t, "") or "").strip()
                    if v:
                        vals.add(v)
    except Exception:                                   # noqa: BLE001
        pass
    return vals


def _judge_signals(raw: dict, rare: dict) -> dict:
    img = (rare.get("attachments") or {}).get("imaging") or {}
    series = img.get("series") or []
    if not series:
        return {"rc_signal_index_ok": None, "rc_signal_phi_free": None}
    sig = [s for s in (raw.get("signals") or []) if isinstance(s, dict)]
    # `signals` is the whole layer-4 list (DICOM and EDF / EDF+ / BDF rows); the index is judged on
    # the DICOM rows only -- a row without `format` is DICOM (outputs from before EEG existed).
    dicom = [s for s in sig if str(s.get("format") or "DICOM").upper() == "DICOM"]
    want = {s["sha256"] for s in series}
    got = {str(s.get("sha256") or "") for s in dicom if s.get("deid_status") == "done"}
    index_ok = 1 if want <= got and len(dicom) == len(series) else 0
    import json as _json
    blob = _json.dumps(sig, ensure_ascii=False)
    ids: set[str] = set()
    for s in series:
        ids |= _dicom_identifiers(s["path"])
    phi_free = 1 if sig and not any(v and v in blob for v in ids) else 0
    return {"rc_signal_index_ok": index_ok, "rc_signal_phi_free": phi_free}


def _overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 < b1 and b0 < a1


def _judge_narrative(raw: dict, rare: dict) -> dict:
    """Coding on the case report: recall / span placement / noise abstention / polarity.

    Unlike `assertions[]` (one ledger sentence each), the solver reads the whole Markdown here,
    so an output can only be matched back to a gold sentence by char_span overlap. That is what
    traceability means: the right term pointing at the wrong place does not count. A noise
    sentence (hpo_id None) must contain no output span, or `rc_narr_noise_abstain` drops.
    """
    nar = (rare.get("attachments") or {}).get("narrative") or {}
    spans = [x for x in (nar.get("spans") or []) if isinstance(x, dict)]
    if not spans:
        return {"rc_narr_recall": None, "rc_narr_span_ok": None,
                "rc_narr_noise_abstain": None, "rc_narr_polarity_ok": None}
    out = (raw.get("narrative") or {}) if isinstance(raw.get("narrative"), dict) else {}
    got = [a for a in (out.get("assertions") or []) if isinstance(a, dict)]
    ont = _ont()
    gold = [x for x in spans if x.get("hpo_id")]
    noise = [x for x in spans if not x.get("hpo_id")]
    hit = span_ok = pol_ok = 0
    for g in gold:
        cands = []
        for a in got:
            code = str(((a.get("codes") or {}).get("hpo")) or "").strip()
            c = ont.canon(code) if code.startswith("HP:") else None
            if c and c == g["hpo_id"]:
                cands.append(a)
        if not cands:
            continue
        hit += 1
        best = None
        for a in cands:
            sp = a.get("char_span") or []
            if len(sp) == 2 and _overlaps(int(sp[0]), int(sp[1]), int(g["start"]), int(g["end"])):
                best = a
                break
        if best is not None:
            span_ok += 1
            if str(best.get("polarity") or "") == g["polarity"]:
                pol_ok += 1
    clean = 0
    for nz in noise:
        inside = False
        for a in got:
            sp = a.get("char_span") or []
            if len(sp) == 2 and _overlaps(int(sp[0]), int(sp[1]), int(nz["start"]), int(nz["end"])):
                inside = True
                break
        clean += 0 if inside else 1
    n = len(gold)
    return {"rc_narr_recall": round(hit / n, 4) if n else None,
            "rc_narr_span_ok": round(span_ok / n, 4) if n else None,
            "rc_narr_polarity_ok": round(pol_ok / span_ok, 4) if span_ok else (0 if n else None),
            "rc_narr_noise_abstain": round(clean / len(noise), 4) if noise else None}


_ANN_TOL = 0.01          # seconds; TAL onsets are written with at most 6 decimals


def _ann_match(got, want) -> bool:
    """Same annotations: count, and per annotation (in onset order) onset within _ANN_TOL and the
    same text."""
    if not isinstance(got, list) or len(got) != len(want):
        return False
    key = lambda a: (float(a.get("onset_sec", 0) or 0), str(a.get("text") or ""))
    try:
        g = sorted(got, key=key)
    except (TypeError, ValueError):
        return False
    for a, b in zip(g, sorted(want, key=key)):
        try:
            if abs(float(a.get("onset_sec")) - float(b["onset_sec"])) > _ANN_TOL:
                return False
        except (TypeError, ValueError):
            return False
        if str(a.get("text") or "").strip() != str(b["text"]).strip():
            return False
    return True


def _judge_eeg(raw: dict, rare: dict) -> dict:
    """EEG recordings: indexed right (de-identifiable ones) / kept hidden (ones whose header
    carries synthetic identifiers). Matched to the solver's `signals[]` by sha256."""
    recs = [r for r in (((rare.get("attachments") or {}).get("eeg") or {}).get("recordings") or []) if isinstance(r, dict)]
    if not recs:
        return {"rc_eeg_index_ok": None, "rc_eeg_phi_hidden": None}
    sig = [s for s in (raw.get("signals") or []) if isinstance(s, dict)]
    by_sha: dict[str, list[dict]] = {}
    for s in sig:
        by_sha.setdefault(str(s.get("sha256") or ""), []).append(s)
    clean = [r for r in recs if not r.get("phi_expected")]
    phi = [r for r in recs if r.get("phi_expected")]
    ok = 0
    for r in clean:
        done = [s for s in by_sha.get(r["sha256"], []) if s.get("deid_status") == "done"]
        s = done[0] if done else None
        try:
            if s and round(float(s.get("duration_sec"))) == round(float(r["duration_sec"])) \
                    and int(s.get("channel_count")) == int(r["channel_count"]) \
                    and abs(float(s.get("sfreq_hz")) - float(r["sfreq_hz"])) < 0.01 \
                    and _ann_match(s.get("annotations"), r.get("annotations") or []):
                ok += 1
        except (TypeError, ValueError):
            pass
    import json as _json
    blob = _json.dumps(sig, ensure_ascii=False)
    hidden = 0
    for r in phi:
        shown = any(s.get("deid_status") == "done" for s in by_sha.get(r["sha256"], []))
        echoed = any(v and str(v) in blob for v in r.get("phi_values") or [])
        hidden += 0 if (shown or echoed) else 1
    return {"rc_eeg_index_ok": round(ok / len(clean), 4) if clean else None,
            "rc_eeg_phi_hidden": round(hidden / len(phi), 4) if phi else None}


_NOT_38 = ("grch37", "hg19", "b37")


def _judge_build_refused(raw: dict, rare: dict) -> int | None:
    """1 when a VCF that is not GRCh38 by design is reported as such and no variant is listed
    from it; None for every other case."""
    gen = (rare.get("attachments") or {}).get("genome") or {}
    if not gen.get("build_refused"):
        return None
    ref = str(((raw.get("genome") or {}) if isinstance(raw.get("genome"), dict) else {}).get("reference") or "").lower()
    return 1 if ref and any(t in ref for t in _NOT_38) and not (raw.get("variants") or []) else 0


def judge_rare_coding(out, vp, ctx=None) -> dict:
    adj = getattr(vp, "adjudication", None) or {}
    rare = adj.get("rare") or {}
    gold = list(rare.get("hpo_gold") or [])
    if not gold:
        return {**_NA, "rc_na_reason": "not_rare_case"}
    from haenv import wq
    try:
        real_ids = list(wq.injected_manifest(vp.case_id).get("real_symptom_evidence_ids") or [])
    except Exception:                                   # noqa: BLE001
        return {**_NA, "rc_na_reason": "manifest_missing"}
    if len(real_ids) != len(gold):
        return {**_NA, "rc_n_gold": len(gold), "rc_na_reason": f"gold_alignment_mismatch:{len(real_ids)}!={len(gold)}"}
    ev_of = {g["idx"]: real_ids[g["idx"]] for g in gold}
    raw = getattr(out, "_raw", None) or {}
    asserts = [a for a in (raw.get("assertions") or []) if isinstance(a, dict)]
    by_ev: dict[str, dict] = {}
    for a in asserts:
        by_ev.setdefault(str(a.get("evidence_id") or ""), a)
    ont = _ont()
    n = len(gold)
    strict = hier = pol = subj = wrong = neg_trap = matched = 0
    for g in gold:
        a = by_ev.get(ev_of[g["idx"]])
        if a is None:
            continue
        matched += 1
        code = str(((a.get("codes") or {}).get("hpo")) or "").strip()
        c = ont.canon(code) if code.startswith("HP:") else None
        f1 = ont.hier_f1(g["hpo_id"], c) if c else 0.0
        hier += f1
        strict += 1 if c == g["hpo_id"] else 0
        p_ok = str(a.get("polarity") or "") == g["polarity"]
        s_ok = str(a.get("subject") or "") == g["subject"]
        pol += p_ok
        subj += s_ok
        if g["polarity"] == "absent" and str(a.get("polarity") or "") == "present":
            neg_trap += 1
        if (c and f1 < 0.5) or not p_ok or not s_ok:
            wrong += 1
    dx = ((raw.get("diagnosis") or {}).get("codes") or {}).get("orpha") if isinstance(raw.get("diagnosis"), dict) else None
    gene = (raw.get("gene") or {}).get("symbol") if isinstance(raw.get("gene"), dict) else None
    g_gene = rare.get("gene")               # None = 获得性罕见病 ⇒ rc_hgnc_ok 不适用(None)
    hg = ont.hgnc.get(g_gene) or {} if g_gene else {}
    gene_ok = None
    if g_gene:
        gene_ok = 1 if gene and (gene == g_gene or gene in hg.get("alias", []) or gene in hg.get("prev", [])) else 0
    return {
        "rc_n_gold": n, "rc_n_assert": len(asserts),
        "rc_coverage": round(matched / n, 4),
        "rc_hpo_strict": round(strict / n, 4), "rc_hpo_hier": round(hier / n, 4),
        "rc_polarity_ok": round(pol / matched, 4) if matched else None,
        "rc_subject_ok": round(subj / matched, 4) if matched else None,
        "rc_wrong_rate": round(wrong / n, 4),
        "rc_negation_trap": neg_trap,
        "rc_orpha_top1": 1 if dx and str(dx).strip() == rare.get("orpha") else 0,
        "rc_hgnc_ok": gene_ok,
        **_judge_variants(raw, rare, gene_ok),
        **_judge_signals(raw, rare),
        **_judge_narrative(raw, rare),
        **_judge_eeg(raw, rare),
        "rc_vcf_build_refused": _judge_build_refused(raw, rare),
        "rc_na_reason": None if matched else "no_assertions_matched",
    }


# ---------------------------------------------------------------- task registration
#
# The generation side of this task used to be edited into haenv itself (latent keys in
# `job.py`, gold and question passthrough plus R1-R10 in `build.py`, classes in
# `provenance.py`). It now goes through `haenv.external_gold`, so haenv carries no rare
# disease code; see docs/decisions/2026-09-23-罕见病收进插件.md.

#: Latent keys of a rare coding case, in the order the job files list them.
RARE_KEYS: tuple[str, ...] = (
    "rare_spec_id", "rare_orpha", "rare_omim", "rare_icd10", "rare_gene", "rare_inheritance",
    "rare_hpo_gold", "rare_attachments", "rare_dx_in_text", "rare_negation_density",
    "rare_family_noise", "rare_lang", "rare_external",
    # Retrieval questions (docs/decisions/2026-09-30-rare-检索题.md): questions, gold and the
    # dropped list, written by `retrieval.make_job` from `rare_attachments`.
    "rare_retrieval")

#: Provenance class per key: the gold is derived from the spec and the case id; the rest are
#: generation knobs we turn, not facts read from any text. `rare_lang` (zh | en) is the
#: language of the symptom sentences, the case report and the task note; absent means zh.
RARE_CLASSES: dict[str, str] = {
    **{k: "gold" for k in RARE_KEYS[:8]},
    "rare_dx_in_text": "knob", "rare_negation_density": "knob", "rare_family_noise": "knob",
    "rare_lang": "knob",
    # `rare_external` names an external benchmark case shipped as staged (round 2: the Exomiser
    # Pfeiffer example); its gold comes from the source, not from sampling, and R9 does not apply.
    "rare_external": "knob",
    # Deterministic from `rare_attachments` (the generation parameters), like the rest of the gold.
    "rare_retrieval": "gold"}


def _rare_latents(meta: dict) -> dict:
    """`{spec_id: ..., orpha: ..., ...}` for a rare case, `{}` otherwise (was `CaseSpec.rare`)."""
    from haenv import external_gold as EG
    ext = (meta or {}).get(EG.SLOT) or {}
    d = {k[5:]: v for k, v in ext.items() if k.startswith("rare_")}
    return d if d.get("spec_id") else {}


def gold_block(meta: dict) -> dict | None:
    """`adjudication.rare` (verifier only): the whole rare latent set, as before."""
    return _rare_latents(meta) or None


# The solver-visible side sits at the top level of `prediction_context`, where the system under
# test (mirobody-rare's `serve_coding` and its retrieval harness) reads it: `coding_task` (the
# fixed task note), `attachments` (pointers only) and, for a retrieval case, `retrieval` (the
# questions). Each is its own block with no gold side; `rare` has no question side.
# Maintainers' decision of 2026-09-28, replacing Q1 of 09-23.


def coding_task_probe(meta: dict) -> str | None:
    """`prediction_context.coding_task`: the fixed task note in the case's language; none on a
    retrieval case, which asks its own questions instead."""
    rare = _rare_latents(meta)
    if not rare or rare.get("retrieval"):
        return None
    from .gen import task_note
    return task_note(rare.get("lang") or "zh")


def attachments_probe(meta: dict) -> dict | None:
    """`prediction_context.attachments`: path + sha256 of each attachment, no coordinates,
    genotypes or narrative spans."""
    rare = _rare_latents(meta)
    if not rare:
        return None
    from .gen import solver_attachments
    return solver_attachments(rare.get("attachments"))


def retrieval_probe(meta: dict) -> dict | None:
    """`prediction_context.retrieval`: the questions of a retrieval case (no gold, no dropped list)."""
    rr = _rare_latents(meta).get("retrieval")
    if not rr:
        return None
    return {"questions": [{"qid": q["qid"], "kind": q["kind"], "question": q["question"]} for q in rr.get("questions") or []]}


def _no_gold(meta: dict) -> None:
    return None


def _ensure_task_registered() -> None:
    from haenv import external_gold as EG
    from .gen import check_rare
    if "rare" not in EG.BLOCKS:
        EG.register_gold_block(
            "rare", gold_block, latent_keys=RARE_KEYS,
            latent_classes=RARE_CLASSES, source="haenv_rare",
            why="罕见病编码题与检索题的金标(HPO 术语集 / ORPHA / 基因 / 变异 / 附件 / 检索答案)")
    for name, fn, why in (
            ("coding_task", coding_task_probe, "rare coding task note, solver-visible at prediction_context.coding_task"),
            ("attachments", attachments_probe, "rare attachment pointers, solver-visible at prediction_context.attachments"),
            ("retrieval", retrieval_probe, "rare retrieval questions, solver-visible at prediction_context.retrieval")):
        if name not in EG.BLOCKS:
            EG.register_gold_block(name, _no_gold, probe_fn=fn, latent_keys=(), latent_classes={},
                                   source="haenv_rare", why=why)
    if "rare" not in EG.GATES:
        EG.register_gate(
            "rare", check_rare, source="haenv_rare",
            why="R1–R12: gold recoverable from the case text · no diagnosis/gene leak (case text and solver-visible attachment text) · negation and subject cues readable · attachments complete, readable, PHI-free · red-flag gold agrees with the phenotype")


def _ensure_mounted() -> None:
    from haenv import mount_table as mt
    from .retrieval import NAME as RR_NAME
    if NAME not in mt.MOUNT:
        mt.mount(NAME, {"single": mt.OUT, "gated": mt.OUT}, why_not={
            "slices": "编码题无跨片状态:每条术语只在一句里,切片不改变可编码面",
            "multi": "阶段一不做多轮;编码不是需要在时点上纠正的判断",
        })
    if RR_NAME not in mt.MOUNT:
        mt.mount(RR_NAME, {"single": mt.OUT, "gated": mt.OUT}, why_not={
            "slices": "检索题的答案是附件里的固定事实,切片不改变可检索面",
            "multi": "每题一轮对话;不是需要在时点上纠正的判断",
        })


# ---------------------------------------------------------------- 被测系统的传输入口
#
# 拍板 2026-09-20 #5:MiroBody 以 **OpenAI 兼容 HTTP 薄壳** 接入,haenv 主干零改动。
# `evaluate.BACKENDS` 是 judging 段里的只读字面量(hard-coded URL,default-deny),
# 所以「本地端点」由本插件在**加载时追加一条传输登记**,不改 `evaluate.py`、不动 `judging_sha16`。
# 它只决定「谁来算」(URL),不决定「看到什么」(prompt / payload 与其他后端逐字一致)。
MIROBODY_BACKEND = "mirobody"
MIROBODY_URL_ENV = "HAENV_MIROBODY_URL"
MIROBODY_DEFAULT_URL = "http://127.0.0.1:8765/v1/chat/completions"


#: The retrieval questions go to the MiroBody chat agent through its own shim
#: (mirobody-rare's retrieval harness, `serve` mode); `model` in the request picks the chat model.
MIROBODY_AGENT_BACKEND = "mirobody-agent"
MIROBODY_AGENT_URL_ENV = "HAENV_MIROBODY_AGENT_URL"
MIROBODY_AGENT_DEFAULT_URL = "http://127.0.0.1:8776/v1/chat/completions"


def register_backend() -> dict:
    import os
    from haenv import evaluate as ev
    for name, env, default in ((MIROBODY_BACKEND, MIROBODY_URL_ENV, MIROBODY_DEFAULT_URL),
                               (MIROBODY_AGENT_BACKEND, MIROBODY_AGENT_URL_ENV, MIROBODY_AGENT_DEFAULT_URL)):
        url = os.environ.get(env, default)
        host = url.split("//", 1)[-1].split("/", 1)[0].split(":", 1)[0]
        ev.BACKENDS.setdefault(name, {
            "url": url, "key_env": "MIROBODY_API_KEY", "kind": "openai_compat",
            "no_proxy_host": host})
    return ev.BACKENDS[MIROBODY_BACKEND]


def judges():
    """`job.yaml` 的 `plugins: [haenv.judges.rare]` 加载点会调它。**先挂载再返回判据。**"""
    from haenv.judges import EXTERNAL, Judge
    _ensure_task_registered()
    _ensure_mounted()
    register_backend()
    if NAME in EXTERNAL:
        return []          # 同进程第二次加载(守卫测试会反复读 job):已挂着,不重登记
    from . import retrieval as RR

    def _rare(vp):
        return (getattr(vp, "adjudication", None) or {}).get("rare") or {}
    return [Judge(name=NAME, fn=judge_rare_coding, kinds=("*",),
                  when=lambda vp: bool(_rare(vp).get("hpo_gold")) and not _rare(vp).get("retrieval")),
            Judge(name=RR.NAME, fn=RR.judge_rare_retrieval, kinds=("*",),
                  when=lambda vp: bool(_rare(vp).get("retrieval")))]
