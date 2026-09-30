"""Journal-style narrative for a rare case (2026-09-22): the same gold terms the ledger
carries, rendered as a case report with the section skeleton mined from the 84 open-access
《罕见病研究》 case reports (体格检查 33/41 · 实验室检查 32/41 · 影像学检查 23/41 · 基因检测
14/41 of the reports with a 临床资料 section; 797 negative-finding sentences, 50 "因…就诊"
chief complaints). Structure and phrasing are taken from the corpus; no patient text is copied.

Every sentence that carries a gold term is recorded as a span into the rendered text
(`{hpo_id, polarity, subject, start, end, section}`); life-event noise sentences are recorded
with `hpo_id: None`, so a judge can ask both "was the term found where it is" and "was the
noise left alone". The diagnosis, its aliases and the gene never appear (gate `rare_narrative_leak`).

Imaging pairing: a disease gets a TCIA collection whose class (LFS ↔ sarcoma) or body region
(neurological disorders ↔ brain MR) matches, and the tier says which; everything else is an
`unrelated_attachment` and the report text says so. SYNTHETIC, evaluation only.
"""
from __future__ import annotations

import hashlib
from typing import Any

from haenv import rng

# spec id -> (collection, tier). Brain MR collections serve the neurological disorders as a
# region match; only Li-Fraumeni gets a class match (sarcoma). Everything else: unrelated.
BRAIN_COLLECTIONS = ("UPENN-GBM", "Vestibular-Schwannoma-SEG")
SARCOMA_COLLECTIONS = ("Soft-tissue-Sarcoma", "TCGA-SARC")
IMAGING_MATCH: dict[str, tuple[str, str]] = {
    "RD-LFS": ("Soft-tissue-Sarcoma", "exact_class"),
    "RD-TSC": ("UPENN-GBM", "region_match"), "RD-NF1": ("Vestibular-Schwannoma-SEG", "region_match"),
    "RD-L2HGA": ("UPENN-GBM", "region_match"), "RD-XALD": ("UPENN-GBM", "region_match"),
    "RD-RETT": ("Vestibular-Schwannoma-SEG", "region_match"), "RD-DRAVET": ("UPENN-GBM", "region_match"),
    "RD-MPS1": ("Vestibular-Schwannoma-SEG", "region_match"), "RD-MPS2": ("UPENN-GBM", "region_match"),
    "RD-NPC": ("Vestibular-Schwannoma-SEG", "region_match"), "RD-ALS": ("UPENN-GBM", "region_match"),
    "RD-DANON": ("Vestibular-Schwannoma-SEG", "region_match"),
}
BODY_PART = {"Soft-tissue-Sarcoma": "四肢软组织", "TCGA-SARC": "躯干及四肢", "UPENN-GBM": "头颅", "Vestibular-Schwannoma-SEG": "头颅"}

# Round 2 (style=2): the report says what an unread scan honestly supports. A brain MR paired by
# region only is "done, reading pending" (the style-1 sentence "no intracranial space-occupying
# lesion" is false for glioma / schwannoma scans); a ReMIND tumour scan paired with a tumour
# predisposition is reported as a space-occupying lesion pending pathology. EEG gets its own
# section, again with the result withheld.
BODY_PART_R2 = {**BODY_PART, "ReMIND": "头颅"}
_IMG_FINDING_R2 = {"exact_class": "见占位性病变，性质待专科阅片及病理",
                   "region_match": "检查已完成，所见待专科阅片",
                   "unrelated_attachment": "为既往其他原因所摄，未见与本次主诉相关的特异性改变"}
_EEG_LINE = "已行脑电图检查（{n}份记录），正式报告待出。"


def pair_imaging(spec_id: str, idx: int, available: list[str]) -> dict:
    """`modalities.imaging` for a spec: the matched collection when it is downloaded, else an
    unrelated one round-robin over the sarcoma collections (then anything available)."""
    coll, tier = IMAGING_MATCH.get(spec_id, (None, "unrelated_attachment"))
    if coll in available:
        return {"collection": coll, "tier": tier}
    pool = [c for c in SARCOMA_COLLECTIONS if c in available] or list(available)
    return {"collection": pool[idx % len(pool)], "tier": "unrelated_attachment"}


# --- templates (phrasing families observed in the corpus; the ledger's own term templates
# are reused for the term-bearing sentences so the coder sees the same surface forms)
_ONSET = ("起病初", "病程中", "此后", "近期", "同期", "随后")
_PRESENT = ("{o}出现{l}", "{o}自述{l}", "{o}{l}较明显", "{o}伴{l}", "{l}逐渐明显")
_ABSENT = ("未见{l}", "{l}阴性", "查体无{l}", "否认{l}", "未及{l}")
_FAMILY = ("其{r}有{l}史", "{r}曾有{l}", "{r}也出现过{l}")
_NOISE = ("整理文件时右手食指被纸张划伤，贴创可贴次日愈合", "家中安装了新的空气净化器", "近日搬家，作息尚在调整",
          "更换了新的手机并重新配置了健康应用", "周末与家人外出郊游一日", "工作调整后通勤时间缩短", "开始记录每日饮水量",
          "订阅了一份新的报纸", "为家中老人购置了血压计", "把书房的灯换成了暖光", "参加了社区的健康讲座", "重新整理了药箱")
_LAB_GENERIC = "血常规、肝肾功能、电解质、凝血功能未见明显异常。"
_IMG_FINDING = {"exact_class": "见软组织肿块影，边界欠清，具体所见待专科阅片",
                "region_match": "颅内未见明确占位性病变，白质信号改变请结合临床",
                "unrelated_attachment": "为既往其他原因所摄，未见与本次主诉相关的特异性改变"}


def _rel(g: dict) -> str:
    return str(g.get("relative") or "家属")


# --- English case report (`lang="en"`, 2026-09-23): the same skeleton and the same span contract,
# section names as English case reports use them. Phrasing families mirror the zh ones one to one.
_ONSET_EN = ("At onset", "During the course of the illness", "Subsequently", "Recently", "At the same time", "Later")
_PRESENT_EN = ("{o}, the patient developed {l}", "{o}, the patient reported {l}", "{o}, {l} became noticeable",
               "{o}, {l} was also present", "{L} gradually became apparent")
_ABSENT_EN = ("No {l} was observed", "Examination showed no {l}", "The patient denies {l}", "No {l} was found",
              "Negative for {l}")
_FAMILY_EN = ("The patient's {r} has a history of {l}", "The patient's {r} previously had {l}",
              "The patient's {r} also experienced {l}")
_NOISE_EN = ("While sorting papers the patient cut the right index finger; it healed a day later under a plaster",
             "A new air purifier was installed at home", "The family moved house recently and routines are still settling",
             "The patient switched to a new phone and set up the health app again", "Spent a weekend day on an outing with family",
             "A change at work shortened the daily commute", "Started keeping a log of daily water intake",
             "Subscribed to a new newspaper", "Bought a blood pressure monitor for an elderly relative",
             "Replaced the study lamp with a warm-light bulb", "Attended a community health talk", "Reorganised the medicine cabinet")
_LAB_GENERIC_EN = "Complete blood count, liver and kidney function, electrolytes and coagulation were unremarkable."
_IMG_FINDING_EN = {"exact_class": "showing a soft-tissue mass with ill-defined margins; formal specialist reading pending",
                   "region_match": "showing no definite intracranial space-occupying lesion; white-matter signal changes to be correlated clinically",
                   "unrelated_attachment": "obtained earlier for an unrelated reason, with no specific change relevant to the current complaint"}
BODY_PART_EN = {"Soft-tissue-Sarcoma": "extremity soft-tissue", "TCGA-SARC": "trunk and extremity", "UPENN-GBM": "head",
                "Vestibular-Schwannoma-SEG": "head"}
BODY_PART_EN_R2 = {**BODY_PART_EN, "ReMIND": "head"}
_IMG_FINDING_EN_R2 = {"exact_class": "showing a space-occupying lesion; specialist reading and pathology pending",
                      "region_match": "completed; specialist reading pending",
                      "unrelated_attachment": "obtained earlier for an unrelated reason, with no specific change relevant to the current complaint"}
_EEG_LINE_EN = "EEG has been recorded ({n} recording{s}); the formal report is pending."


def _render_en(case_id: str, spec: dict, gold: list[dict], profile: dict | None,
               imaging: dict | None, genome: dict | None, eeg: dict | None = None,
               style: int = 1) -> tuple[str, list[dict]]:
    prof = profile or {}
    sex = "man" if str(prof.get("sex", "")).upper() == "M" else "woman"
    age = str(prof.get("age_range") or "adult")
    present = [g for g in gold if g["subject"] == "proband" and g["polarity"] == "present"]
    absent = [g for g in gold if g["subject"] == "proband" and g["polarity"] == "absent"]
    family = [g for g in gold if g["subject"] == "relative"]
    known = [str(k) for k in (prof.get("known_conditions") or [])]
    drug = str(prof.get("drug") or "")
    buf: list[str] = []
    spans: list[dict] = []
    pos = 0

    def emit(s: str, section: str, g: dict | None = None, noise: bool = False) -> None:
        nonlocal pos
        start = pos + (len(s) - len(s.lstrip()))
        buf.append(s)
        pos += len(s)
        core = s.strip()
        if g is not None or noise:
            spans.append({"hpo_id": g["hpo_id"] if g else None, "polarity": g["polarity"] if g else None,
                          "subject": g["subject"] if g else None, "start": start, "end": start + len(core),
                          "section": section, "idx": g["idx"] if g else None})

    def line(s: str) -> None:
        emit(s + "\n", "")

    def sent(s: str, section: str, g: dict | None = None, noise: bool = False) -> None:
        emit(s[:1].upper() + s[1:] + ".", section, g, noise)
        buf.append("\n")
        nonlocal pos
        pos += 1

    line(f"# Case {case_id}")
    line("")
    line("## Chief Complaint")
    who = f"A {sex} aged {age} years" if age != "adult" else f"An adult {sex}"
    if present:
        sent(f"{who} presented with {present[0]['label']}", "Chief Complaint", present[0])
    else:
        line(f"{who}.")
    line("")
    line("## History of Present Illness")
    k_noise = 2 + rng.below(3, case_id, "rare", "narr_noise_n")
    noise = [rng.pick(list(_NOISE_EN), case_id, "rare", "narr_noise", i) for i in range(k_noise)]
    seen = set()
    noise = [n for n in noise if not (n in seen or seen.add(n))]
    rest = present[1:]
    slots = max(1, len(rest))
    for i, g in enumerate(rest):
        tpl = rng.pick(list(_PRESENT_EN), case_id, "rare", "narr_present", g["idx"])
        o = _ONSET_EN[min(i * len(_ONSET_EN) // slots, len(_ONSET_EN) - 1)]
        sent(tpl.format(o=o, l=g["label"], L=g["label"][:1].upper() + g["label"][1:]), "History of Present Illness", g)
        if noise and i % 3 == 1:
            sent(noise.pop(0), "History of Present Illness", noise=True)
    for n in noise:
        sent(n, "History of Present Illness", noise=True)
    line("")
    line("## Past Medical History")
    line((f"History of {', '.join(known)}" + (f", currently treated with {drug}" if drug else "") + ".") if known else "Previously healthy.")
    line("No history of surgery, trauma or blood transfusion.")
    line("")
    line("## Family History")
    if family:
        for g in family:
            tpl = rng.pick(list(_FAMILY_EN), case_id, "rare", "narr_family", g["idx"])
            sent(tpl.format(r=str(g.get("relative") or "relative"), l=g["label"]), "Family History", g)
        line("The parents are not consanguineous.")
    else:
        line("Both parents are healthy and not consanguineous; no similar illness in the family.")
    line("")
    line("## Physical Examination")
    line("Vital signs were stable; the patient was alert and answered questions appropriately.")
    for g in absent:
        tpl = rng.pick(list(_ABSENT_EN), case_id, "rare", "narr_absent", g["idx"])
        sent(tpl.format(l=g["label"]), "Physical Examination", g)
    line("")
    line("## Laboratory Tests")
    line(_LAB_GENERIC_EN)
    line("Specialised tests have been sent out; results are pending.")
    line("")
    line("## Imaging")
    img = imaging or {}
    series = img.get("series") or []
    if series:
        mods = sorted({str(s.get("modality") or "") for s in series} - {""}) or ["imaging"]
        parts, finding = (BODY_PART_EN_R2, _IMG_FINDING_EN_R2) if style == 2 else (BODY_PART_EN, _IMG_FINDING_EN)
        part = parts.get(str(img.get("collection")), "relevant-region")
        line(f"{part[:1].upper() + part[1:]} {'/'.join(mods)} ({len(series)} series), "
             f"{finding.get(str(img.get('tier')), finding['unrelated_attachment'])}.")
    else:
        line("No imaging has been performed yet.")
    recs = (eeg or {}).get("recordings") or []
    if recs:
        line("")
        line("## Electroencephalography")
        line(_EEG_LINE_EN.format(n=len(recs), s="" if len(recs) == 1 else "s"))
    line("")
    line("## Genetic Testing")
    gm = genome or {}
    if gm.get("files"):
        who = "the proband and both parents (trio)" if len(gm["files"]) == 3 else "the proband"
        line(f"Whole-exome sequencing (GRCh38) of {who} has been sent out; results are pending.")
    else:
        line("No genetic testing has been performed.")
    md = "".join(buf)
    for sp in spans:                                     # self-check: every span slices back
        assert md[sp["start"]:sp["end"]] == md[sp["start"]:sp["end"]].strip()
    return md, spans


def render_narrative(case_id: str, spec: dict, gold: list[dict], profile: dict | None,
                     imaging: dict | None, genome: dict | None, lang: str = "zh",
                     eeg: dict | None = None, style: int = 1) -> tuple[str, list[dict]]:
    """→ (markdown, spans). Deterministic in `case_id`. `lang="en"`: the English report (`_render_en`).
    `style=2` (round 2): neutral brain-MR and tumour wording plus an EEG section; style 1 is the
    wording of the packs built before it, kept byte for byte."""
    if style == 2:
        return _render_v2(case_id, spec, gold, profile, imaging, genome, eeg, lang)
    if lang == "en":
        return _render_en(case_id, spec, gold, profile, imaging, genome, eeg=eeg, style=style)
    prof = profile or {}
    sex = "男" if str(prof.get("sex", "")).upper() == "M" else "女"
    age = str(prof.get("age_range") or "成年")
    present = [g for g in gold if g["subject"] == "proband" and g["polarity"] == "present"]
    absent = [g for g in gold if g["subject"] == "proband" and g["polarity"] == "absent"]
    family = [g for g in gold if g["subject"] == "relative"]
    known = [str(k) for k in (prof.get("known_conditions") or [])]
    drug = str(prof.get("drug") or "")
    buf: list[str] = []
    spans: list[dict] = []
    pos = 0

    def emit(s: str, section: str, g: dict | None = None, noise: bool = False) -> None:
        nonlocal pos
        start = pos + (len(s) - len(s.lstrip()))
        buf.append(s)
        pos += len(s)
        core = s.strip()
        if g is not None or noise:
            spans.append({"hpo_id": g["hpo_id"] if g else None, "polarity": g["polarity"] if g else None,
                          "subject": g["subject"] if g else None, "start": start, "end": start + len(core),
                          "section": section, "idx": g["idx"] if g else None})

    def line(s: str) -> None:
        emit(s + "\n", "")

    line(f"# 病例 {case_id}")
    line("")
    # 主诉 —— 因“首个主要表型”就诊(语料 50 例的主诉句式)
    line("## 主诉")
    if present:
        g0 = present[0]
        emit(f"患者{sex}性，{age}岁，因“{g0['label']}”就诊。", "主诉", g0)
        buf.append("\n"); pos += 1
    else:
        line(f"患者{sex}性，{age}岁。")
    line("")
    # 现病史 —— 其余 present 术语按时序铺开,噪声句夹在其中
    line("## 现病史")
    k_noise = 2 + rng.below(3, case_id, "rare", "narr_noise_n")
    noise = [rng.pick(list(_NOISE), case_id, "rare", "narr_noise", i) for i in range(k_noise)]
    seen = set()
    noise = [n for n in noise if not (n in seen or seen.add(n))]
    rest = present[1:]
    slots = max(1, len(rest))
    for i, g in enumerate(rest):
        tpl = rng.pick(list(_PRESENT), case_id, "rare", "narr_present", g["idx"])
        o = _ONSET[min(i * len(_ONSET) // slots, len(_ONSET) - 1)]
        emit(tpl.format(o=o, l=g["label"]) + "。", "现病史", g)
        buf.append("\n"); pos += 1
        if noise and i % 3 == 1:
            emit(noise.pop(0) + "。", "现病史", noise=True)
            buf.append("\n"); pos += 1
    for n in noise:
        emit(n + "。", "现病史", noise=True)
        buf.append("\n"); pos += 1
    line("")
    line("## 既往史")
    line((f"既往{'、'.join(known)}病史" + (f"，{drug}治疗中" if drug else "") + "。") if known else "既往体健。")
    line("否认手术、外伤及输血史。")
    line("")
    line("## 家族史")
    if family:
        for g in family:
            tpl = rng.pick(list(_FAMILY), case_id, "rare", "narr_family", g["idx"])
            emit(tpl.format(r=_rel(g), l=g["label"]) + "。", "家族史", g)
            buf.append("\n"); pos += 1
        line("父母非近亲结婚。")
    else:
        line("父母体健，非近亲结婚，否认家族类似病史。")
    line("")
    # 体格检查 —— 阴性表述(语料 797 句)承载 absent 术语
    line("## 体格检查")
    line("生命体征平稳，神志清楚，对答切题。")
    for g in absent:
        tpl = rng.pick(list(_ABSENT), case_id, "rare", "narr_absent", g["idx"])
        emit(tpl.format(l=g["label"]) + "。", "体格检查", g)
        buf.append("\n"); pos += 1
    line("")
    line("## 实验室检查")
    line(_LAB_GENERIC)
    line("专项检查已送检，结果待回。")
    line("")
    line("## 影像学检查")
    img = imaging or {}
    series = img.get("series") or []
    if series:
        mods = sorted({str(s.get("modality") or "") for s in series} - {""}) or ["影像"]
        parts, finding = (BODY_PART_R2, _IMG_FINDING_R2) if style == 2 else (BODY_PART, _IMG_FINDING)
        part = parts.get(str(img.get("collection")), "相关部位")
        line(f"行{part}{'/'.join(mods)}检查（{len(series)}个序列），{finding.get(str(img.get('tier')), finding['unrelated_attachment'])}。")
    else:
        line("暂未行影像学检查。")
    recs = (eeg or {}).get("recordings") or []
    if recs:
        line("")
        line("## 脑电图检查")
        line(_EEG_LINE.format(n=len(recs)))
    line("")
    line("## 基因检测")
    gm = genome or {}
    if gm.get("files"):
        who = "家系三人" if len(gm["files"]) == 3 else "先证者"
        line(f"已送检{who}全外显子组测序（GRCh38），结果待回。")
    else:
        line("未行基因检测。")
    md = "".join(buf)
    for sp in spans:                                     # self-check: every span slices back
        assert md[sp["start"]:sp["end"]] == md[sp["start"]:sp["end"]].strip()
    return md, spans


def leak_terms(spec: dict) -> list[str]:
    out = [str(spec.get("diagnosis") or "")] + [str(a) for a in (spec.get("aliases") or [])]
    if spec.get("gene"):
        out.append(str(spec["gene"]))
    return [t for t in out if len(t) >= 2]


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ================================================================ style 2 (round 2)
# Measured against the real case sections (haenv-rare-stats, 2026-09-28): style 1 carries 9-10
# headings over ~300 zh / ~1250 en characters, the real sections ~1800 / ~3600 characters under
# 0-6 headings. Style 2 keeps the one-finding-per-sentence span contract and adds only
# scaffolding that carries no finding (course opening and close, allergy and personal history,
# vital signs within normal ranges for the age, general appearance) and folds the four test
# sections into one investigations section, as the corpus does ("辅助检查" / "Investigations").
# The remaining length gap is lab values, imaging descriptions and treatment courses, which in a
# synthetic case would carry findings outside the gold or name the diagnosis.
_V2 = {
    "zh": {"title": "# 病例 {cid}", "cc": "## 主诉", "hpi": "## 现病史", "pmh": "## 既往史及个人史", "fh": "## 家族史",
           "pe": "## 体格检查", "inv": "## 辅助检查",
           "who": "患者{sex}性，{age}岁", "cc_s": "{who}，因“{l}”就诊。", "cc_none": "{who}。",
           "open": ("患者起病隐匿，病程中症状逐渐加重", "患者近年来逐渐出现不适，曾于当地医院就诊，未明确诊断",
                    "患者起病后多次于外院就诊，诊断未明"),
           "close": "为进一步诊治来我院就诊。",
           "pmh_known": "既往{k}病史{d}。", "pmh_none": "既往体健。", "pmh_rest": "否认手术、外伤及输血史，否认食物、药物过敏史。",
           "personal_child": "足月顺产，按计划预防接种。", "personal_adult": "生于原籍，否认疫区居留史，无烟酒嗜好。",
           "fh_close": "父母非近亲结婚。", "fh_none": "父母体健，非近亲结婚，否认家族类似病史。",
           "vitals": "体温{t}℃，脉搏{hr}次/分，呼吸{rr}次/分，血压{sbp}/{dbp} mmHg。",
           "general_child": "神志清楚，精神反应可。", "general_adult": "神志清楚，对答切题，查体合作。",
           "lab": _LAB_GENERIC, "pending": "专项检查已送检，结果待回。",
           "img": "{part}{mods}检查（{n}个序列）{finding}。", "img_none": "暂未行影像学检查。",
           "eeg": "已行脑电图检查（{n}份记录），正式报告待出。",
           "gen": "已送检{who}全外显子组测序（GRCh38），结果待回。", "gen_none": "未行基因检测。",
           "trio": "家系三人", "proband": "先证者", "end": "。"},
    "en": {"title": "# Case {cid}", "cc": "## Chief Complaint", "hpi": "## History of Present Illness",
           "pmh": "## Past and Personal History", "fh": "## Family History", "pe": "## Physical Examination",
           "inv": "## Investigations",
           "who": "{a} {sex} aged {age} years", "cc_s": "{who} presented with {l}.", "cc_none": "{who}.",
           "open": ("The onset was insidious and the symptoms worsened over time",
                    "Over recent years the patient gradually became unwell and was seen locally without a diagnosis",
                    "Since onset the patient had been seen at several other hospitals without a diagnosis"),
           "close": "The patient was referred for further evaluation.",
           "pmh_known": "History of {k}{d}.", "pmh_none": "Previously healthy.",
           "pmh_rest": "No history of surgery, trauma or blood transfusion; no known food or drug allergies.",
           "personal_child": "Born at term by vaginal delivery; immunisations are up to date.",
           "personal_adult": "No travel to endemic areas; does not smoke or drink alcohol.",
           "fh_close": "The parents are not consanguineous.",
           "fh_none": "Both parents are healthy and not consanguineous; no similar illness in the family.",
           "vitals": "Temperature {t} °C, pulse {hr}/min, respiratory rate {rr}/min, blood pressure {sbp}/{dbp} mmHg.",
           "general_child": "The child was alert and responsive.", "general_adult": "The patient was alert, oriented and cooperative.",
           "lab": _LAB_GENERIC_EN, "pending": "Specialised tests have been sent out; results are pending.",
           "img": "{part} {mods} ({n} series), {finding}.", "img_none": "No imaging has been performed yet.",
           "eeg": "EEG has been recorded ({n} recording{s}); the formal report is pending.",
           "gen": "Whole-exome sequencing (GRCh38) of {who} has been sent out; results are pending.",
           "gen_none": "No genetic testing has been performed.",
           "trio": "the proband and both parents (trio)", "proband": "the proband", "end": "."},
}


def _age_lo(age: str) -> int | None:
    try:
        return int(str(age).split("-", 1)[0])
    except ValueError:
        return None


def _render_v2(case_id: str, spec: dict, gold: list[dict], profile: dict | None, imaging: dict | None,
               genome: dict | None, eeg: dict | None, lang: str) -> tuple[str, list[dict]]:
    T = _V2[lang]
    en = lang == "en"
    prof = profile or {}
    male = str(prof.get("sex", "")).upper() == "M"
    age = str(prof.get("age_range") or ("adult" if en else "成年"))
    lo = _age_lo(age)
    child = lo is not None and lo < 18
    present = [g for g in gold if g["subject"] == "proband" and g["polarity"] == "present"]
    absent = [g for g in gold if g["subject"] == "proband" and g["polarity"] == "absent"]
    family = [g for g in gold if g["subject"] == "relative"]
    known = [str(k) for k in (prof.get("known_conditions") or [])]
    drug = str(prof.get("drug") or "")
    buf: list[str] = []
    spans: list[dict] = []
    pos = 0

    def emit(s: str, section: str, g: dict | None = None, noise: bool = False) -> None:
        nonlocal pos
        start = pos + (len(s) - len(s.lstrip()))
        buf.append(s)
        pos += len(s)
        core = s.strip()
        if g is not None or noise:
            spans.append({"hpo_id": g["hpo_id"] if g else None, "polarity": g["polarity"] if g else None,
                          "subject": g["subject"] if g else None, "start": start, "end": start + len(core),
                          "section": section, "idx": g["idx"] if g else None})

    def line(s: str) -> None:
        emit(s + "\n", "")

    def sent(s: str, section: str, g: dict | None = None, noise: bool = False) -> None:
        nonlocal pos
        if en:
            s = s[:1].upper() + s[1:]
        emit(s + ("" if s.endswith((".", "。")) else T["end"]), section, g, noise)
        buf.append("\n")
        pos += 1

    sec = {k: T[k][3:] for k in ("cc", "hpi", "pmh", "fh", "pe", "inv")}
    line(T["title"].format(cid=case_id))
    line("")
    line(T["cc"])
    if en:
        who = T["who"].format(a="A", sex=("boy" if male else "girl") if child else ("man" if male else "woman"), age=age) \
            if age != "adult" else f"An adult {'man' if male else 'woman'}"
    else:
        who = T["who"].format(sex="男" if male else "女", age=age)
    if present:
        sent(T["cc_s"].format(who=who, l=present[0]["label"]), sec["cc"], present[0])
    else:
        line(T["cc_none"].format(who=who))
    line("")
    line(T["hpi"])
    sent(rng.pick(list(T["open"]), case_id, "rare", "narr2_open"), sec["hpi"])
    pool = _NOISE_EN if en else _NOISE
    k_noise = 2 + rng.below(3, case_id, "rare", "narr_noise_n")
    noise = [rng.pick(list(pool), case_id, "rare", "narr_noise", i) for i in range(k_noise)]
    seen: set = set()
    noise = [n for n in noise if not (n in seen or seen.add(n))]
    rest = present[1:]
    slots = max(1, len(rest))
    onset, tpls = (_ONSET_EN, _PRESENT_EN) if en else (_ONSET, _PRESENT)
    for i, g in enumerate(rest):
        tpl = rng.pick(list(tpls), case_id, "rare", "narr_present", g["idx"])
        o = onset[min(i * len(onset) // slots, len(onset) - 1)]
        sent(tpl.format(o=o, l=g["label"], L=g["label"][:1].upper() + g["label"][1:]), sec["hpi"], g)
        if noise and i % 3 == 1:
            sent(noise.pop(0), sec["hpi"], noise=True)
    for n in noise:
        sent(n, sec["hpi"], noise=True)
    line(T["close"])
    line("")
    line(T["pmh"])
    if known:
        line(T["pmh_known"].format(k=("、" if not en else ", ").join(known),
                                   d=(f"，{drug}治疗中" if not en else f", currently treated with {drug}") if drug else ""))
    else:
        line(T["pmh_none"])
    line(T["pmh_rest"])
    line(T["personal_child"] if child else T["personal_adult"])
    line("")
    line(T["fh"])
    if family:
        fam = _FAMILY_EN if en else _FAMILY
        for g in family:
            tpl = rng.pick(list(fam), case_id, "rare", "narr_family", g["idx"])
            sent(tpl.format(r=(str(g.get("relative") or "relative") if en else _rel(g)), l=g["label"]), sec["fh"], g)
        line(T["fh_close"])
    else:
        line(T["fh_none"])
    line("")
    line(T["pe"])
    t = 36.3 + rng.below(7, case_id, "rare", "narr2_t") / 10
    hr = (86 if child else 64) + rng.below(24, case_id, "rare", "narr2_hr")
    rr = (20 if child else 14) + rng.below(5, case_id, "rare", "narr2_rr")
    sbp = (96 if child else 108) + rng.below(18, case_id, "rare", "narr2_sbp")
    dbp = (58 if child else 66) + rng.below(14, case_id, "rare", "narr2_dbp")
    line(T["vitals"].format(t=f"{t:.1f}", hr=hr, rr=rr, sbp=sbp, dbp=dbp))
    line(T["general_child"] if child else T["general_adult"])
    ab = _ABSENT_EN if en else _ABSENT
    for g in absent:
        tpl = rng.pick(list(ab), case_id, "rare", "narr_absent", g["idx"])
        sent(tpl.format(l=g["label"]), sec["pe"], g)
    line("")
    line(T["inv"])
    line(T["lab"])
    line(T["pending"])
    img = imaging or {}
    series = img.get("series") or []
    if series:
        mods = "/".join(sorted({str(s.get("modality") or "") for s in series} - {""}) or (["imaging"] if en else ["影像"]))
        part = (BODY_PART_EN_R2 if en else BODY_PART_R2).get(str(img.get("collection")), "relevant-region" if en else "相关部位")
        finding = (_IMG_FINDING_EN_R2 if en else _IMG_FINDING_R2).get(str(img.get("tier")), "")
        s = T["img"].format(part=part[:1].upper() + part[1:] if en else part, mods=mods, n=len(series),
                            finding=finding if en else f"，{finding}")
        line(s)
    else:
        line(T["img_none"])
    recs = (eeg or {}).get("recordings") or []
    if recs:
        line(T["eeg"].format(n=len(recs), s="" if len(recs) == 1 else "s"))
    gm = genome or {}
    if gm.get("files") and not gm.get("build_refused"):
        line(T["gen"].format(who=T["trio"] if len(gm["files"]) == 3 else T["proband"]))
    elif gm.get("files"):
        line("An outside exome VCF was provided by the family." if en else "家属提供外院外显子组测序VCF文件。")
    else:
        line(T["gen_none"])
    md = "".join(buf)
    for sp in spans:
        assert md[sp["start"]:sp["end"]] == md[sp["start"]:sp["end"]].strip()
    return md, spans
