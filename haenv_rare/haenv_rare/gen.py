"""gen.py(haenv_rare)—— 罕见病**编码**条件的 gold 生成器 + 发射门(经 external_gold 注册,盖 world_sha)。

设计:`docs/design/2026-09-20-rare-gen-full.md` §3;拍板:`docs/decisions/2026-09-20-rare-coding-合成与评测.md`。

## 它做什么(金标先行,再渲染)

    spec(本包 data/conditions_rare.yaml)+ case_id
      ① HPO 术语集 G:phenotype.hpoa 的 ORPHA 行按频率分层抽样(种子 = case_id);
         negated 项来自 rivals 的高频表型;family 项按 inheritance 派生
      ② 变异 V:ClinVar P/LP + ≥2 星(`data/ontology/_cache/clinvar_<gene>.json`,由 tools 预扫)
      ③ 附件:骨架 VCF(按染色体切片)掺入 V + PED + DICOM 指针(sha256 + 路径)
      ④ 渲染:G → `raw.symptoms[]`(确定性模板;LLM 档另议)
      ⑤ `latent.rare_*` → `build.HaenvGenerator` 落到 `adjudication.rare`(verifier-only)

## 🔴 三条纪律

* **只吃 case_id 与规格**,不调模型 —— 金标由此确定性推出(§2a)。
* **罕见病是 overlay**:底下仍是 `demographics.sample_profile` 采出的代谢病人
  (内核 `validate_premise` 要求 `disease ∈ DISEASE_SIGNAL_DOMAIN`)。
* **门在这里,不在 `gates.py`**(那是 judging 段)。R1–R12 的实现在本模块,负对照在
  `haenv_rare/tests/`。

SYNTHETIC,仅评测用,非医疗建议。
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import os
import pathlib
import re
import struct
import zipfile
from functools import lru_cache

from haenv import rng

log = logging.getLogger("haenv.rare")

_HERE = pathlib.Path(__file__).resolve().parent
# Repo root of the haenv checkout (not this package's parent): the default ontology
# and attachment directories are resolved against it, as they were before the task moved into this package.
import haenv as _haenv                                   # noqa: E402
ROOT = pathlib.Path(_haenv.__file__).resolve().parent.parent

# ------------------------------------------------------------------ 路径
# Each data location: the environment variable, else `rare.<key>` in haenv's config
# (`config.yaml` with `config.local.yaml` merged over it), else the default.
def _configured(key: str) -> str | None:
    from haenv.cli import load_cfg
    v = ((load_cfg() or {}).get("rare") or {}).get(key)
    return str(v) if v else None


def ontology_dir() -> pathlib.Path:
    return pathlib.Path(os.environ.get("HAENV_RARE_ONTOLOGY") or _configured("ontology_dir")
                        or (ROOT / "data" / "ontology"))


def rare_data_root() -> pathlib.Path:
    return pathlib.Path(os.environ.get("HAENV_RARE_DATA_ROOT") or _configured("data_root")
                        or "/data/xfs_recovery/data/rare")


def attachments_root() -> pathlib.Path:
    """P0 落点:`derived/rare_attachments/`(gitignore)。P3 再搬进批次目录(拍板 6)。"""
    return pathlib.Path(os.environ.get("HAENV_RARE_ATTACH_ROOT") or _configured("attach_root")
                        or (ROOT / "derived" / "rare_attachments"))


# ------------------------------------------------------------------ 规格
_SPEC_FIELDS = frozenset({
    "clinical_review", "case_id", "diagnosis", "aliases", "orpha", "gene", "inheritance",
    "phenotype_policy", "rivals", "modalities", "weight", "urgency", "red_flag",
    "clinician_warranted", "outcome_label", "tests", "specialty", "sex",
})
#: Optional spec fields (every other field is required).
_SPEC_OPTIONAL = frozenset({"clinical_review", "sex"})
_INHERITANCE = ("AR", "AD", "XLR", "de_novo", "none")
_POLICY_KEYS = ("obligate", "very_frequent", "frequent", "occasional", "negated", "family")


class RareSpecError(ValueError):
    pass


@lru_cache(maxsize=1)
def load_specs() -> dict[str, dict]:
    """装载包内 `data/conditions_rare.yaml`,default-deny。"""
    import copy
    from haenv.yamlcache import load_yaml     # same cached reader as the registry tables
    doc = copy.deepcopy(load_yaml(_HERE / "data" / "conditions_rare.yaml")) or {}
    out: dict[str, dict] = {}
    ont = Ontology.get()
    for sid, spec in (doc.get("conditions") or {}).items():
        bad = set(spec) - _SPEC_FIELDS
        if bad:
            raise RareSpecError(f"{sid}: 未登记字段 {sorted(bad)}")
        miss = _SPEC_FIELDS - set(spec) - _SPEC_OPTIONAL
        if spec.get("sex") not in (None, "F", "M"):
            raise RareSpecError(f"{sid}: sex {spec.get('sex')!r} must be F, M or absent")
        if miss:
            raise RareSpecError(f"{sid}: 缺字段 {sorted(miss)}")
        if str(spec["orpha"]) not in ont.orpha:
            raise RareSpecError(f"{sid}: orpha {spec['orpha']!r} 不在 en_product1.xml")
        if spec["gene"] is None:
            if spec["inheritance"] != "none" or ((spec.get("modalities") or {}).get("genome") not in (None, "none")):
                raise RareSpecError(f"{sid}: gene 为空只允许 inheritance=none 且 genome=none(获得性罕见病)")
        elif spec["gene"] not in ont.hgnc:
            raise RareSpecError(f"{sid}: gene {spec['gene']!r} 不是 HGNC approved symbol")
        if spec["inheritance"] not in _INHERITANCE:
            raise RareSpecError(f"{sid}: inheritance {spec['inheritance']!r} ∉ {_INHERITANCE}")
        if not re.fullmatch(r"JD-\d+", str(spec["case_id"])):
            raise RareSpecError(f"{sid}: case_id 须为 JD-nn(与 case_ids.yaml 同域)")
        pol = spec["phenotype_policy"]
        if set(pol) != set(_POLICY_KEYS):
            raise RareSpecError(f"{sid}: phenotype_policy 键须恰为 {_POLICY_KEYS}")
        for r in spec["rivals"]:
            if str(r) not in ont.orpha:
                raise RareSpecError(f"{sid}: rival orpha {r!r} 不在 en_product1.xml")
        if bool(spec["red_flag"]) and str(spec["urgency"]) != "🔴":
            raise RareSpecError(f"{sid}: red_flag=true 而 urgency={spec['urgency']!r} —— haenv 铁律由 "
                                f"red_flag_present 推出 🔴,不一致的金标会让每一格 ABORT(iron_law)")
        if not ont.hpoa.get("ORPHA:" + str(spec["orpha"])):
            raise RareSpecError(f"{sid}: phenotype.hpoa 无 ORPHA:{spec['orpha']} 的表型注释")
        out[sid] = dict(spec)
    if len({s["case_id"] for s in out.values()}) != len(out):
        raise RareSpecError("case_id 重复")
    return out


@lru_cache(maxsize=1)
def red_flag_hpo() -> frozenset[str]:
    """Terms of `data/conditions_rare.yaml:red_flag_hpo`; a term missing from hp.obo or obsolete is refused."""
    from haenv.yamlcache import load_yaml
    doc = load_yaml(_HERE / "data" / "conditions_rare.yaml") or {}
    ont = Ontology.get()
    terms = frozenset(str(h) for h in (doc.get("red_flag_hpo") or {}))
    bad = sorted(h for h in terms if h not in ont.name or h in ont.obsolete)
    if bad:
        raise RareSpecError(f"red_flag_hpo: terms missing from hp.obo or obsolete: {bad}")
    return terms


def case_triage(spec: dict, gold: list[dict]) -> tuple[bool, str]:
    """Case-level (red_flag, urgency): (True, 🔴) when the condition-level `red_flag` is true or the
    proband presents a `red_flag_hpo` term or one of its descendants; else (False, condition urgency)."""
    rf_terms = red_flag_hpo()
    ont = Ontology.get()
    hit = any(({g["hpo_id"]} | ont.ancestors(g["hpo_id"])) & rf_terms for g in gold
              if g.get("polarity") == "present" and g.get("subject") == "proband")
    rf = bool(spec["red_flag"]) or hit
    return rf, ("🔴" if rf else str(spec["urgency"]))


# ------------------------------------------------------------------ 本体
FREQ = {"HP:0040280": "obligate", "HP:0040281": "very_frequent", "HP:0040282": "frequent",
        "HP:0040283": "occasional", "HP:0040284": "very_rare", "HP:0040285": "excluded"}
_ROOTS = {"HP:0000001", "HP:0000118"}


class Ontology:
    """hp.obo + babelon zh + phenotype.hpoa(ORPHA 行)+ product1 + HGNC。进程内缓存一份。"""
    _inst = None

    @classmethod
    def get(cls) -> "Ontology":
        if cls._inst is None:
            cls._inst = cls(ontology_dir())
        return cls._inst

    def __init__(self, d: pathlib.Path):
        self.dir = d
        self.name: dict[str, str] = {}
        self.parents: dict[str, list[str]] = {}
        self.obsolete: set[str] = set()
        self.replaced: dict[str, str] = {}
        self.syn: dict[str, list[str]] = {}
        cur = None
        for line in (d / "hp.obo").open(encoding="utf-8"):
            line = line.rstrip("\n")
            if line == "[Term]":
                cur = None
            elif line.startswith("id: HP:"):
                cur = line[4:]
                self.parents.setdefault(cur, [])
            elif cur is None:
                continue
            elif line.startswith("name: "):
                self.name[cur] = line[6:]
            elif line.startswith("is_a: "):
                self.parents[cur].append(line[6:].split(" ")[0])
            elif line.startswith("is_obsolete: true"):
                self.obsolete.add(cur)
            elif line.startswith("replaced_by: "):
                self.replaced[cur] = line[13:].strip()
            elif line.startswith("synonym: "):
                m = re.match(r'synonym: "(.*?)"', line)
                if m:
                    self.syn.setdefault(cur, []).append(m.group(1))
        self.zh: dict[str, str] = {}
        p = d / "hp-zh.babelon.tsv"
        if p.exists():
            for line in p.open(encoding="utf-8"):
                f = line.rstrip("\n").split("\t")
                if len(f) >= 6 and f[0] == "en" and f[3] == "rdfs:label":
                    self.zh[f[2]] = f[5]
        self.hpoa: dict[str, list[tuple[str, str, str]]] = {}
        for line in (d / "phenotype.hpoa").open(encoding="utf-8"):
            if line.startswith("#") or line.startswith("database_id"):
                continue
            f = line.rstrip("\n").split("\t")
            if f[0].startswith("ORPHA:"):
                self.hpoa.setdefault(f[0], []).append((f[3], f[7], f[2]))   # hp, freq, qualifier
        import xml.etree.ElementTree as ET
        self.orpha: dict[str, dict] = {}
        for dis in ET.parse(d / "en_product1.xml").getroot().iter("Disorder"):
            code = dis.findtext("OrphaCode")
            refs = dis.findall("ExternalReferenceList/ExternalReference")
            self.orpha[code] = {
                "name": dis.findtext("Name"),
                "syn": [s.text for s in dis.findall("SynonymList/Synonym") if s.text],
                "omim": [r.findtext("Reference") for r in refs if r.findtext("Source") == "OMIM"],
                "icd10": [r.findtext("Reference") for r in refs if r.findtext("Source") == "ICD-10"],
            }
        self.hgnc: dict[str, dict] = {}
        for line in (d / "hgnc_complete_set.txt").open(encoding="utf-8"):
            if line.startswith("hgnc_id"):
                continue
            f = line.rstrip("\n").split("\t")
            if f[5] == "Approved":
                self.hgnc[f[1]] = {"id": f[0], "alias": [a for a in f[7].split("|") if a],
                                   "prev": [a for a in f[9].split("|") if a]}

    def canon(self, hp: str) -> str | None:
        """obsolete → replaced_by;无替代 → None。"""
        if hp in self.obsolete:
            return self.replaced.get(hp)
        return hp if hp in self.parents else None

    def ancestors(self, hp: str) -> set[str]:
        out, stack = set(), [hp]
        while stack:
            x = stack.pop()
            for p in self.parents.get(x, []):
                if p not in out:
                    out.add(p)
                    stack.append(p)
        return out - _ROOTS

    def label(self, hp: str) -> str:
        return self.zh.get(hp) or self.name.get(hp, hp)

    def hier_f1(self, a: str, b: str) -> float:
        """祖先集 F1(含自身,去根)。`ontology.py` ICD 祖先集的同款。"""
        if not a or not b:
            return 0.0
        A = self.ancestors(a) | {a}
        B = self.ancestors(b) | {b}
        A -= _ROOTS
        B -= _ROOTS
        if not A or not B:
            return 0.0
        inter = len(A & B)
        if inter == 0:
            return 0.0
        p, r = inter / len(B), inter / len(A)
        return 2 * p * r / (p + r)


# ------------------------------------------------------------------ ① 术语集
def _pool(ont: Ontology, orpha: str) -> dict[str, list[str]]:
    """按频率分层的 HPO id 池(去 obsolete、去 excluded、按 id 排序 = 确定性)。"""
    out: dict[str, list[str]] = {k: [] for k in FREQ.values()}
    for hp, fq, qual in ont.hpoa.get("ORPHA:" + orpha, []):
        if qual == "NOT":
            continue
        c = ont.canon(hp)
        if not c:
            continue
        k = FREQ.get(fq, "occasional")
        if k == "excluded":
            continue
        if c not in out[k]:
            out[k].append(c)
    for k in out:
        out[k].sort()
    return out


_REL = {"AR": ["姐姐", "弟弟", "妹妹", "哥哥"], "AD": ["父亲", "母亲"], "XLR": ["舅舅"],
        "de_novo": [], "none": []}


def _label_blocked_words(sex: str) -> tuple[str, ...]:
    """标签里**不许出现**的词:haenv 自己的三张词表 + 性别互斥词。从源头 import,不抄第二份。

    · `events.ATTRIBUTIVE_MARKERS`(「复发」「确诊」…):`verify.check_event` 的 `context_non_attributive`
      会把含它们的真症状句判负丢弃 ⇒ 术语进了金标、句子却没进题面 ⇒ R1 必红(P1 实测 HHT「复发性自发鼻衄」);
    · `verify.TRUTH_WORDS_ZH` / `DRIVER_WORDS_ZH`:`no_answer_words`;
    · `gates.SEX_EXCLUSIVE[sex]`:GEN19(男病人的题面里出现「月经」)。
    """
    from haenv.events import ATTRIBUTIVE_MARKERS
    from haenv.verify import DRIVER_WORDS_ZH, TRUTH_WORDS_ZH
    from haenv.gates import SEX_EXCLUSIVE
    return tuple(ATTRIBUTIVE_MARKERS) + tuple(TRUTH_WORDS_ZH) + tuple(DRIVER_WORDS_ZH) + tuple(SEX_EXCLUSIVE.get(sex, ()))


def sample_gold(spec: dict, case_id: str, sex: str = "F") -> list[dict]:
    """G:每条 {idx, hpo_id, label, polarity, subject, freq}。**只吃 case_id、规格与性别。**"""
    ont = Ontology.get()
    pol = spec["phenotype_policy"]
    pool = _pool(ont, str(spec["orpha"]))
    _blocked = _label_blocked_words(sex)
    # 🔴 泄漏保护是构造性的:hpoa 会把**病名本身**当表型注释(ALS 的 HP:0007354 就叫
    # 「肌萎缩侧索硬化」)。凡 zh/en 标签与诊断名/别名互相包含的术语,一律不进池。
    _leak = [t.lower() for t in [spec["diagnosis"], *spec["aliases"]] if t and len(t) >= 3]

    def _leaky(hp: str) -> bool:
        labs = [ont.label(hp).lower(), ont.name.get(hp, "").lower()]
        for a in _leak:
            for l in labs:
                if not l:
                    continue
                if a.isascii():
                    if re.search(r"(?<![a-z])" + re.escape(a) + r"(?![a-z])", l):
                        return True
                elif a in l or l in a:
                    return True
        return False

    seen_label: set[str] = set()
    for k in pool:
        kept = []
        for hp in pool[k]:
            lab = ont.label(hp)
            if _leaky(hp) or lab in seen_label or any(w in lab for w in _blocked):
                continue
            seen_label.add(lab)
            kept.append(hp)
        pool[k] = kept
    present: list[tuple[str, str]] = []

    def take(k: str, rule):
        items = pool[k]
        if rule == "all":
            return items[:6]
        lo, hi = rule
        return rng.subset(items, lo, hi, case_id, "rare", k)

    for k in ("obligate", "very_frequent", "frequent", "occasional"):
        for hp in take(k, pol[k]):
            if hp not in {h for h, _ in present}:
                present.append((hp, k))
    if len(present) < 3:
        raise RareSpecError(f"{spec.get('diagnosis')}: 抽到的 present 术语 <3({len(present)}),规格池太小")
    pres_ids = {h for h, _ in present}
    pres_anc = set().union(*(ont.ancestors(h) for h in pres_ids))
    # negated:rivals 的高频表型,且与 present 无祖先/后代关系(否则「否认」会自相矛盾)
    neg_pool: list[str] = []
    for r in spec["rivals"]:
        rp = _pool(ont, str(r))
        for hp in rp["obligate"] + rp["very_frequent"] + rp["frequent"]:
            if hp in pres_ids or hp in pres_anc or (ont.ancestors(hp) & pres_ids):
                continue
            if _leaky(hp) or any(w in ont.label(hp) for w in _blocked):   # rival 池同样过词表(P1 实测 JD-62 漏在这里)
                continue
            if hp not in neg_pool:
                neg_pool.append(hp)
    neg_pool.sort()
    lo, hi = pol["negated"]
    negated = rng.subset(neg_pool, lo, hi, case_id, "rare", "negated")
    lo, hi = pol["family"]
    fam_n = lo + rng.below(hi - lo + 1, case_id, "rare", "family_n")
    rels = _REL.get(spec["inheritance"], [])
    family = []
    if rels and fam_n:
        for i, hp in enumerate(rng.subset([h for h, _ in present], fam_n, fam_n, case_id, "rare", "family")):
            family.append((hp, rng.pick(rels, case_id, "rare", "rel", i)))
    gold: list[dict] = []
    for hp, k in present:
        gold.append({"hpo_id": hp, "label": ont.label(hp), "label_en": ont.name.get(hp, ""),
                     "polarity": "present", "subject": "proband", "freq": k})
    for hp in negated:
        gold.append({"hpo_id": hp, "label": ont.label(hp), "label_en": ont.name.get(hp, ""),
                     "polarity": "absent", "subject": "proband", "freq": "rival"})
    for hp, rel in family:
        gold.append({"hpo_id": hp, "label": ont.label(hp), "label_en": ont.name.get(hp, ""),
                     "polarity": "present", "subject": "relative", "relative": rel, "freq": "family"})
    # 顺序确定性打散(否则「先 present 后 absent」本身是一条可查表的顺序)
    order = sorted(range(len(gold)), key=lambda i: rng.counter_u32(case_id, "rare", "order", gold[i]["hpo_id"], gold[i]["subject"]))
    gold = [gold[i] for i in order]
    for i, g in enumerate(gold):
        g["idx"] = i
    return gold


# ------------------------------------------------------------------ ④ 渲染(确定性模板)
NEG_CUES = ("未见", "否认", "无", "阴性", "未出现")
_PRESENT_TPL = ("{l}", "出现{l}", "近期{l}", "{l}较明显", "自述{l}")
_ABSENT_TPL = ("未见{l}", "否认{l}", "查体无{l}", "{l}阴性")
_FAMILY_TPL = ("{r}有{l}史", "{r}曾有{l}", "{r}也出现过{l}")


# English package (`--lang en`, 2026-09-23): the gold set is sampled exactly as for zh (selection
# keys on the zh label, so a case's terms do not depend on the language); only the surface form
# changes. `label` becomes the hp.obo name with a lower-case first letter unless it opens with an
# acronym, `relative` an English kin word; the zh originals stay beside them.
EN_REL = {"姐姐": "sister", "妹妹": "sister", "弟弟": "brother", "哥哥": "brother",
          "父亲": "father", "母亲": "mother", "舅舅": "maternal uncle"}
NEG_CUES_EN = ("no ", "denies", "negative", "without", "absent")
_PRESENT_TPL_EN = ("{L}", "Developed {l}", "Recent onset of {l}", "Presents with {l}", "Reports {l}")
_ABSENT_TPL_EN = ("No {l}", "Denies {l}", "No {l} on examination", "Negative for {l}")
_FAMILY_TPL_EN = ("The patient's {r} has a history of {l}", "The patient's {r} previously had {l}",
                  "The patient's {r} also had {l}")


def surface_en(name: str) -> str:
    """hp.obo `name:` as it reads mid-sentence: `Seizure` → `seizure`, `EEG abnormality` kept."""
    if len(name) > 1 and name[0].isupper() and name[1].islower():
        return name[0].lower() + name[1:]
    return name


def gold_in_lang(gold: list[dict], lang: str) -> list[dict]:
    if lang == "zh":
        return gold
    if lang != "en":
        raise RareSpecError(f"unsupported rare_lang {lang!r} (zh | en)")
    out = []
    for g in gold:
        h = {**g, "label_zh": g["label"], "label": surface_en(g.get("label_en") or g["label"])}
        if g.get("relative"):
            h["relative_zh"], h["relative"] = g["relative"], EN_REL.get(g["relative"], "relative")
        out.append(h)
    return out


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def render_symptoms(gold: list[dict], case_id: str, T: int, lang: str = "zh") -> list[dict]:
    """G → `raw.symptoms[]`(day/text/context)。时点均匀铺在 [7, T-7] 内并抖动;context 留空
    (`overlay.gate_context` 是闭词表,未登记即丢,留空最稳)。"""
    n = len(gold)
    lo, hi = 7, max(8, T - 7)
    span = hi - lo
    out = []
    for g in gold:
        i = g["idx"]
        base = lo + (span * (i + 1)) // (n + 1)
        jit = rng.below(7, case_id, "rare", "day", i) - 3
        day = max(lo, min(hi, base + jit))
        l = g["label"]
        fam, ab, pr = (_FAMILY_TPL_EN, _ABSENT_TPL_EN, _PRESENT_TPL_EN) if lang == "en" else (_FAMILY_TPL, _ABSENT_TPL, _PRESENT_TPL)
        if g["subject"] == "relative":
            tpl = rng.pick(fam, case_id, "rare", "tpl", i)
            text = tpl.format(r=g["relative"], l=l)
        elif g["polarity"] == "absent":
            tpl = rng.pick(ab, case_id, "rare", "tpl", i)
            text = tpl.format(l=l)
        else:
            tpl = rng.pick(pr, case_id, "rare", "tpl", i)
            text = tpl.format(l=l, L=_cap(l))
        out.append({"day": int(day), "text": text, "context": ""})
    return out


# ------------------------------------------------------------------ ② 变异 + ③ 附件
def _sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pick_variant(spec: dict, case_id: str) -> dict:
    ont = Ontology.get()
    cache = ontology_dir() / "_cache" / f"clinvar_{spec['gene']}.json"
    if not cache.exists():
        raise RareSpecError(f"{spec['gene']}: 无 ClinVar 候选缓存 {cache}(第一轮题包的逐基因缓存,本包不再重建;第二轮题包由 genome_os.clinvar_candidates 直接从 ClinVar 取候选)")
    cands = json.loads(cache.read_text())
    if not cands:
        raise RareSpecError(f"{spec['gene']}: ClinVar A10 候选为空")
    orpha = str(spec["orpha"])
    omim = set(ont.orpha[orpha]["omim"])
    pref = [c for c in cands if f"Orphanet:{orpha}" in c["disdb"] or any(f"OMIM:{o}" in c["disdb"] for o in omim)]
    pool = sorted(pref or cands, key=lambda c: int(c["vid"]) if str(c["vid"]).isdigit() else 0)
    v = dict(rng.pick(pool, case_id, "rare", "variant"))
    v["disease_linked"] = bool(pref)
    return v


_TRIO = {"AJ": {"proband": ("HG002", "M"), "father": ("HG003", "M"), "mother": ("HG004", "F")},
         "CHS": {"proband": ("HG005", "M"), "father": ("HG006", "M"), "mother": ("HG007", "F")}}
_SINGLE = {"M": "HG002", "F": "HG001"}


def _skeleton_path(sample: str) -> pathlib.Path:
    d = rare_data_root() / "01_vcf" / "giab" / sample
    hits = sorted(d.glob("*_benchmark.vcf.gz"))
    if not hits:
        raise RareSpecError(f"骨架 VCF 缺失:{d}")
    return hits[0]


def _chrom_slice(sample: str, chrom: str) -> pathlib.Path:
    """骨架按染色体切片并缓存(全基因组 4M 行,每例重扫太慢;P3 再做全基因组)。"""
    cache = attachments_root() / "_skeleton_cache" / f"{sample}.{chrom}.vcf.gz"
    if cache.exists():
        return cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    src = _skeleton_path(sample)
    bare = chrom.replace("chr", "")
    want = {chrom, bare, "chr" + bare}          # ClinVar 写 `14`,GIAB 写 `chr14`;两种都收
    with gzip.open(src, "rt") as fi, gzip.open(cache.with_suffix(".tmp.gz"), "wt") as fo:
        for line in fi:
            if line.startswith("#") or line.split("\t", 1)[0] in want:
                fo.write(line)
    cache.with_suffix(".tmp.gz").replace(cache)
    return cache


def _genotypes(inheritance: str, sex: str) -> dict[str, str]:
    if inheritance == "AR":
        return {"proband": "1/1", "father": "0/1", "mother": "0/1"}
    if inheritance == "de_novo":
        return {"proband": "0/1", "father": "0/0", "mother": "0/0"}
    if inheritance == "XLR":
        return {"proband": "1" if sex == "M" else "0/1", "father": "0", "mother": "0/1"}
    return {"proband": "0/1"}


def _multi_slice(sample: str, chroms: list[str]) -> pathlib.Path:
    """几条染色体的骨架切片拼成一份(表头取自第一片;记录按 `chroms` 的顺序)。"""
    if len(chroms) == 1:
        return _chrom_slice(sample, chroms[0])
    key = "_".join(c.replace("chr", "") for c in chroms)
    cache = attachments_root() / "_skeleton_cache" / f"{sample}.{key}.vcf.gz"
    if cache.exists():
        return cache
    with gzip.open(cache.with_suffix(".tmp.gz"), "wt") as fo:
        for i, c in enumerate(chroms):
            with gzip.open(_chrom_slice(sample, c), "rt") as fi:
                for line in fi:
                    if line.startswith("#"):
                        if i == 0:
                            fo.write(line)
                        continue
                    fo.write(line)
    cache.with_suffix(".tmp.gz").replace(cache)
    return cache


def _gz_text_writer(path: pathlib.Path):
    """Text writer for a `.gz` whose bytes depend only on its content: the gzip header's mtime is
    fixed at 0 (gzip.open would stamp the wall clock), so an unchanged attachment keeps its sha256."""
    return io.TextIOWrapper(gzip.GzipFile(filename=str(path), mode="wb", mtime=0), encoding="utf-8")


def spike_vcf(skeleton: pathlib.Path, out: pathlib.Path, spikes: list[tuple[dict, str]], case_id: str = "") -> None:
    """把若干条 ClinVar 变异写进骨架切片。

    🔴 **文件里不留真值标记**(2026-09-21 改)。第一版沿用 Exomiser 的 `SPIKED;RD_CASE;RD_VID`
    INFO 约定 —— 那是给 R5 回读用的,但附件一旦对 solver 可见,这个标记就是写在题面上的答案。
    现在 ID 列写 `.`、INFO 写 `.`,R5 按 (chrom,pos,ref,alt) 坐标回读(坐标只在 latent 里)。
    `case_id` 参数保留只为兼容调用点,不再写进文件。
    """
    chrom_style = ""
    with gzip.open(skeleton, "rt") as fi:
        for line in fi:
            if line.startswith("##contig=<ID=chr"):
                chrom_style = "chr"
                break
            if not line.startswith("##"):
                break
    pending: dict[str, list[tuple[int, str]]] = {}
    for v, gt in spikes:
        bare = v["chrom"].replace("chr", "")
        rec = "\t".join([chrom_style + bare, str(v["pos"]), ".", v["ref"], v["alt"], "900", "PASS", ".", "GT", gt]) + "\n"
        pending.setdefault(bare, []).append((int(v["pos"]), rec))
    for lst in pending.values():
        lst.sort()
    with gzip.open(skeleton, "rt") as fi, _gz_text_writer(out) as fo:
        cur = None
        for line in fi:
            if line.startswith("#"):
                fo.write(line)
                continue
            f = line.split("\t", 2)
            bare = f[0].replace("chr", "")
            if bare != cur:
                if cur is not None:
                    for _, rec in pending.pop(cur, []):
                        fo.write(rec)               # 上一条染色体末尾未写完的
                cur = bare
            lst = pending.get(bare)
            while lst and lst[0][0] <= int(f[1]):
                fo.write(lst.pop(0)[1])
            fo.write(line)
        if cur is not None:
            for _, rec in pending.pop(cur, []):
                fo.write(rec)
        for bare in sorted(pending):                 # 切片里根本没有这条染色体(chrX/Y 不在 benchmark)
            for _, rec in pending[bare]:
                fo.write(rec)


def _decoy_variants(spec: dict, case_id: str, n: int = 2) -> list[dict]:
    """诱饵 P/LP 变异,两类:

    ① **隐性携带**(2 条,其他 AR 规格的基因,先证者与一位家长 0/1):健康人本就携带若干隐性致病等位基因。
    ② **同谱竞争病**(≤1 条,2026-09-21 加):本规格 `rivals` 里也在登记表内、且有 ClinVar 候选缓存的病种的基因,
       先证者 0/1、一位家长 0/1(即由未患病家长遗传而来)。它与真值的区别只剩表型细节与遗传模式的自洽性 ——
       没有它,`rc_gene_from_variant` 1.000 是构造出来的(P5b 的诱饵只是携带态)。
       男性先证者遇到 X 连锁的竞争基因跳过:半合子 P/LP 本身就是强致病证据,那不是诱饵是第二个答案。
    """
    specs = load_specs()
    pool = sorted(sid for sid, sp in specs.items()
                  if sp.get("gene") and sp.get("inheritance") == "AR" and sp["gene"] != spec.get("gene")
                  and (ontology_dir() / "_cache" / f"clinvar_{sp['gene']}.json").exists())
    out = []
    for sid in rng.subset(pool, n, n, case_id, "rare", "decoy"):
        d = specs[sid]
        v = pick_variant(d, f"{case_id}/decoy/{sid}")
        if v["chrom"].replace("chr", "") in ("X", "Y"):
            continue
        out.append({**{k: v[k] for k in ("vid", "chrom", "pos", "ref", "alt", "hgvs", "clnsig", "rev")},
                    "gene": d["gene"], "kind": "carrier",
                    "carrier_parent": rng.pick(["father", "mother"], case_id, "rare", "decoy_parent", sid)})
    by_orpha = {str(sp["orpha"]): sid for sid, sp in specs.items()}
    sex = spec.get("_sex", "F")
    rivals = [by_orpha[str(r)] for r in (spec.get("rivals") or []) if str(r) in by_orpha]
    rivals = [sid for sid in rivals if specs[sid].get("gene") and specs[sid]["gene"] != spec.get("gene")
              and (ontology_dir() / "_cache" / f"clinvar_{specs[sid]['gene']}.json").exists()
              and not (specs[sid]["inheritance"] == "XLR" and sex == "M")]
    if rivals:
        sid = rng.pick(sorted(rivals), case_id, "rare", "decoy_rival")
        d = specs[sid]
        v = pick_variant(d, f"{case_id}/decoy-rival/{sid}")
        out.append({**{k: v[k] for k in ("vid", "chrom", "pos", "ref", "alt", "hgvs", "clnsig", "rev")},
                    "gene": d["gene"], "kind": "rival", "rival_spec": sid,
                    "carrier_parent": "mother" if d["inheritance"] == "XLR" else rng.pick(["father", "mother"], case_id, "rare", "decoy_rival_parent")})
    return out


_USED_DICOM: set[tuple[str, str]] = set()      # (collection, patient) handed out in this process: no two cases share one


PED_HEADER = "#FamilyID\tIndividualID\tPaternalID\tMaternalID\tSex\tPhenotype"


def write_ped(path: pathlib.Path, case_id: str, sex: str, inheritance: str, trio: bool) -> None:
    """Six-column PED with the column header real PEDs carry (1kGP `trios.ped`, GIAB) and nothing
    else: no diagnosis, no inheritance mode, no generator name. The file is solver-visible.
    The father is marked affected only for AD (he transmits); everyone else follows the gold."""
    sexn = {"M": "1", "F": "2"}
    rows = [PED_HEADER]
    if trio:
        rows += [f"{case_id}\t{case_id}-P\t{case_id}-F\t{case_id}-M\t{sexn[sex]}\t2",
                 f"{case_id}\t{case_id}-F\t0\t0\t1\t{'2' if inheritance == 'AD' else '1'}",
                 f"{case_id}\t{case_id}-M\t0\t0\t2\t1"]
    else:
        rows += [f"{case_id}\t{case_id}-P\t0\t0\t{sexn[sex]}\t2"]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _vcf_header_lines(path: pathlib.Path) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" or path.name.endswith(".bgz") else open
    out = []
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            out.append(line)
    return out


def attachment_text_leaks(att: dict | None, terms: list[str]) -> list[str]:
    """`<file>: <term>` for every leak term found in a solver-visible text attachment: the PED
    in full and the header lines of every VCF (records are covered by the spike checks). ASCII
    terms match as whole words, case-insensitive; others as substrings."""
    att = att or {}
    gen = att.get("genome") or {}
    texts: list[tuple[str, str]] = []
    ped = (gen.get("ped") or {}).get("path")
    if ped and pathlib.Path(ped).exists():
        texts.append((pathlib.Path(ped).name, pathlib.Path(ped).read_text(encoding="utf-8", errors="replace")))
    for f in (gen.get("files") or {}).values():
        p = pathlib.Path(f.get("path") or "")
        if p.is_file():
            texts.append((p.name, "".join(_vcf_header_lines(p))))
    hits = []
    for name, text in texts:
        for t in terms:
            if t and _mentions(text, t, word=True):
                hits.append(f"{name}: {t}")
    return hits


def build_attachments(spec: dict, case_id: str, sex: str, gold: list[dict], profile: dict | None = None,
                      lang: str = "zh") -> dict:
    """VCF/PED(落 `derived/rare_attachments/<case>/`)+ DICOM 指针 + 期刊体叙述病历。返回可进 latent 的 dict。"""
    out_dir = attachments_root() / case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    att: dict = {"root": str(out_dir), "genome": None, "imaging": None, "narrative": None}
    gm = (spec.get("modalities") or {}).get("genome")
    if gm and gm != "none":
        v = pick_variant(spec, case_id)
        inh = spec["inheritance"]
        gts = _genotypes(inh, sex)
        if gm["skeleton"] == "trio":
            trio = rng.pick(sorted(_TRIO), case_id, "rare", "trio")
            members = {role: _TRIO[trio][role] for role in gts}
            skel_note = f"GIAB_{trio}_trio"
        else:
            members = {"proband": (_SINGLE[sex], sex)}
            skel_note = f"GIAB_{_SINGLE[sex]}_single"
        decoys = _decoy_variants({**spec, "_sex": sex}, case_id)
        chroms = [v["chrom"]] + sorted({d["chrom"] for d in decoys} - {v["chrom"]}, key=lambda c: (len(c), c))
        files = {}
        for role, (sample, _s) in members.items():
            sl = _multi_slice(sample, chroms)
            dst = out_dir / f"{role}.vcf.gz"
            spikes = [(v, gts[role])]
            for d in decoys:
                d_gt = "0/1" if role == "proband" or role == d["carrier_parent"] else "0/0"
                d.setdefault("genotypes", {})[role] = d_gt
                spikes.append((d, d_gt))
            spike_vcf(sl, dst, spikes, case_id)
            files[role] = {"path": str(dst), "sha256": _sha256(dst), "sample": sample, "gt": gts[role]}
        ped = out_dir / f"{case_id}.ped"
        write_ped(ped, case_id, sex, inh, trio=len(members) == 3)
        att["genome"] = {"tier": gm["tier"], "skeleton": skel_note, "reference": "GRCh38",
                         "region": v["chrom"], "inheritance": inh, "genotypes": gts,
                         "variant": {k: v[k] for k in ("vid", "chrom", "pos", "ref", "alt", "hgvs", "clnsig", "rev", "disease_linked")},
                         "decoys": decoys,
                         "files": files, "ped": {"path": str(ped), "sha256": _sha256(ped)}}
    im = (spec.get("modalities") or {}).get("imaging")
    if im and im != "none":
        coll = im["collection"]
        base = rare_data_root() / "03_dicom" / coll
        pats = sorted(p.name for p in base.iterdir() if p.is_dir()) if base.exists() else []
        if not pats:
            raise RareSpecError(f"影像集合 {coll} 无已下载患者目录:{base}")
        pid = pick_clean_dicom_patient(base, pats, rng.pick(pats, case_id, "rare", "dicom_patient"), used=_USED_DICOM)
        _USED_DICOM.add((coll, pid))
        picked = select_series(base / pid)
        man = json.loads((rare_data_root() / "03_dicom" / "_manifests" / f"{coll}.series.json").read_text())
        lic = sorted({m.get("LicenseName", "") for m in man if m.get("PatientID") == pid})
        att["imaging"] = {"tier": im["tier"], "collection": coll, "tcia_patient_id": pid,
                          "license": lic, "deid_method": "DICOM PS3.15 Annex E (TCIA)",
                          "series": [{"path": str(z), "sha256": _sha256(z), "series_uid": z.stem,
                                      "modality": meta["modality"], "n_instances": meta["n"], "bytes": meta["bytes"]} for z, meta in picked]}
    from .narrative import render_narrative, sha256_text
    md, spans = render_narrative(case_id, spec, gold, {**(profile or {}), "sex": sex}, att["imaging"], att["genome"], lang=lang)
    mdp = out_dir / f"{case_id}.md"
    mdp.write_text(md, encoding="utf-8")
    att["narrative"] = {"path": str(mdp), "sha256": sha256_text(md), "spans": spans,
                        "n_gold_spans": sum(1 for x in spans if x["hpo_id"]), "n_noise": sum(1 for x in spans if not x["hpo_id"])}
    return att


# ------------------------------------------------------------------ solver 可见的任务说明
def solver_attachments(att: dict | None) -> dict | None:
    """真值段 `rare_attachments` → **solver 可见**的指针子集:只有路径 + sha256 + 参考基因组。

    不带 `region`(染色体 = 基因座位)、不带 `variant`/`genotypes`/`decoys`/`skeleton`。
    Goes to `prediction_context.attachments`; the sp/vp separation checks in `tests/guards` scan it.
    """
    if not att:
        return None
    out: dict = {}
    gen = att.get("genome")
    if gen:
        out["genome"] = {"reference": gen.get("reference"),
                         "files": {r: {"path": f["path"], "sha256": f["sha256"]} for r, f in (gen.get("files") or {}).items()},
                         "ped": {"path": gen["ped"]["path"], "sha256": gen["ped"]["sha256"]} if gen.get("ped") else None}
    img = att.get("imaging")
    if img:
        out["imaging"] = {"series": [{"path": x["path"], "sha256": x["sha256"]} for x in img.get("series") or []],
                          "deid_method": img.get("deid_method")}
    nar = att.get("narrative")
    if nar:
        out["narrative"] = {"path": nar["path"], "sha256": nar["sha256"]}     # spans are verifier-only
    eeg = att.get("eeg")
    if eeg:
        # path + sha256 only; the file names are neutral (eeg-<k>.edf), the source name is verifier-only
        out["eeg"] = {"recordings": [{"path": r["path"], "sha256": r["sha256"]} for r in eeg.get("recordings") or []]}
    return out or None


_TASK_NOTE_EN = (
    "This case also carries a **coding task**: code every symptom / finding / family-history sentence in the "
    "evidence_ledger and, **in addition** to the usual fields, output `assertions` (each {evidence_id, text, "
    "subject: proband|relative, polarity: present|absent, codes: {hpo: HP:nnnnnnn}}), `diagnosis.codes.orpha` "
    "(ORPHA:nnn) and `gene.symbol` (HGNC). Put anything you cannot code in `abstained`; do not guess. "
    "If `prediction_context.attachments` gives VCF/PED paths you may read them and also output `variants` "
    "(each {chrom, pos, ref, alt, gt, gene, clnsig, inheritance: de_novo|maternal|paternal|biparental|unknown}); "
    "`gene.method` says whether the gene came from variant evidence (variant…) or from the phenotype alone. "
    "If `attachments.narrative` (a case report in Markdown) is given, code the whole text the same way and output "
    "`narrative.assertions` (each with text, char_span=[start,end], subject, polarity, codes) and "
    "`narrative.abstained`; do not code life-event sentences.")


def task_note(lang: str = "zh") -> str:
    """Goes to `prediction_context.coding_task`. No per-case content and no kernel FORBIDDEN_TOKENS."""
    if lang == "en":
        return _TASK_NOTE_EN
    return ("本例另有一项**编码任务**:请把 evidence_ledger 里每条症状/所见/家族史句子编成标准编码,"
            "在 JSON 里**额外**输出 `assertions`(每条 {evidence_id, text, subject: proband|relative, "
            "polarity: present|absent, codes: {hpo: HP:nnnnnnn}}),以及 `diagnosis.codes.orpha`(ORPHA:nnn)、"
            "`gene.symbol`(HGNC)。编不出的条目放进 `abstained`,不要猜。原有字段照常输出。"
            "若 `prediction_context.attachments` 给了 VCF/PED 文件路径,可读取它们,并额外输出 "
            "`variants`(每条 {chrom, pos, ref, alt, gt, gene, clnsig, inheritance: de_novo|maternal|paternal|biparental|unknown}),"
            "`gene.method` 写明基因是由变异证据(variant…)还是仅由表型推出。"
            "若给了 `attachments.narrative`(叙述病历 Markdown),请另对全文做同样的编码,输出 `narrative.assertions`"
            "(每条含 text、char_span=[start,end]、subject、polarity、codes)与 `narrative.abstained`;生活事件句不要编码。")


# ------------------------------------------------------------------ 发射门 R1–R12
def _texts_leq_T(raw, T: int) -> list[str]:
    return [str(e.get("symptom", "")) + " " + str(e.get("context", ""))
            for e in (raw.evidence_ledger or []) if int(e.get("source_timestamp", 10 ** 9)) <= T]


def _mentions(text: str, term: str, word: bool = False, fold: bool = True) -> bool:
    """Does `text` contain `term`? zh (default): substring, case-folded as before. `word=True`
    (English text): an ASCII term must stand as a whole word — `ALS` is not in "vitals", `GLA` not
    in "glasses" — while a non-ASCII term stays a substring."""
    if not term:
        return False
    if word and term.isascii():
        import re
        return re.search(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", text, re.I if fold else 0) is not None
    return (term.lower() in text.lower()) if fold else (term in text)


def check_rare(raw, cs, sp, T: int) -> list[dict]:
    """例级门(与 `gates.py` 同形:`{kind, severity, detail}`,severity=gate 并入发射门)。"""
    lat = cs.latent or {}
    sid = lat.get("rare_spec_id")
    if not sid:
        return []
    spec = load_specs()[sid]
    gold = lat.get("rare_hpo_gold") or []
    hits: list[dict] = []
    texts = _texts_leq_T(sp if hasattr(sp, "evidence_ledger") else raw, T)
    blob = "\n".join(texts)
    ont = Ontology.get()
    lang = lat.get("rare_lang") or "zh"
    en = lang == "en"
    low = blob.lower()
    # R1 金标推得出:每条 gold 的标签(zh;en 包为英文表面形式,不分大小写)必须出现在某句题面里
    miss = [g["hpo_id"] for g in gold if (g["label"].lower() not in low if en else g["label"] not in blob)]
    if miss:
        hits.append({"kind": "rare_gold_not_recoverable", "severity": "gate",
                     "detail": f"{len(miss)} 条 gold 术语在 ≤T 题面里找不到:{miss[:4]}"})
    # R2 不泄漏:病名/别名/基因/ORPHA 名
    leak_terms = [spec["diagnosis"], *spec["aliases"], spec.get("gene") or "",
                  ont.orpha[str(spec["orpha"])]["name"], f"ORPHA:{spec['orpha']}"]
    _leak_list = [t for t in leak_terms if t]          # `leak_terms` is rebound below (narrative block)
    if not lat.get("rare_dx_in_text"):
        leaked = [t for t in leak_terms if t and _mentions(blob, t, word=en)]
        if leaked:
            hits.append({"kind": "rare_leak", "severity": "gate", "detail": f"题面含真值词 {leaked[:3]}"})
    # R3 否定 / 主体线索可读
    for g in gold:
        if g["polarity"] == "absent" and not (any(c in low for c in NEG_CUES_EN) if en else any(c in blob for c in NEG_CUES)):
            hits.append({"kind": "rare_negation_cue_missing", "severity": "gate", "detail": g["hpo_id"]})
            break
    for g in gold:
        if g["subject"] == "relative" and g.get("relative") and not _mentions(blob, g["relative"], word=en):
            hits.append({"kind": "rare_relative_cue_missing", "severity": "gate", "detail": g["hpo_id"]})
            break
    # R9 rival 可判别:至少一条 negated
    if not lat.get("rare_external") and not any(g["polarity"] == "absent" for g in gold):
        hits.append({"kind": "rare_no_negated_term", "severity": "gate", "detail": "鉴别不可裁"})
    # R11 red-flag gold agrees with the sampled phenotype
    want = case_triage(spec, gold)
    have = (bool(lat.get("ddx_red_flag", False)), str(lat.get("ddx_urgency") or ""))
    if have != want:
        hits.append({"kind": "rare_red_flag_gold_mismatch", "severity": "gate",
                     "detail": f"latent (red_flag, urgency)={have}, phenotype implies {want}"})
    # R4 / R5 / R6 / R10:附件
    att = lat.get("rare_attachments") or {}
    gen = att.get("genome")
    if gen and gen.get("build_refused"):
        hits += _check_refused_genome(gen)
        hits += _check_ped(gen, cs)
        gen = None
    elif gen and gen.get("source") == "rare-os":
        hits += _check_os_genome(gen, proband_sex=(cs.raw or {}).get("sex"))
        hits += _check_ped(gen, cs)
        gen = None
    if gen:
        for role, f in gen["files"].items():
            p = pathlib.Path(f["path"])
            if not p.exists():
                hits.append({"kind": "rare_attachment_missing", "severity": "gate", "detail": str(p)})
                continue
            if _sha256(p) != f["sha256"]:
                hits.append({"kind": "rare_attachment_sha_mismatch", "severity": "gate", "detail": str(p)})
            tv = gen["variant"]
            want = {(tv["chrom"].replace("chr", ""), str(tv["pos"]), tv["ref"], tv["alt"]): f["gt"]}
            for d in gen.get("decoys") or []:
                want[(d["chrom"].replace("chr", ""), str(d["pos"]), d["ref"], d["alt"])] = (d.get("genotypes") or {}).get(role)
            got: dict = {}
            with gzip.open(p, "rt") as fh:
                head, found_gt, bad_ref = [], None, False
                for line in fh:
                    if line.startswith("##"):
                        head.append(line)
                        if "b37" in line or "GRCh37" in line or "hg19" in line:
                            bad_ref = True
                        continue
                    if line.startswith("#"):
                        continue
                    c = line.rstrip("\n").split("\t")
                    k = (c[0].replace("chr", ""), c[1], c[3], c[4])
                    if k in want:
                        got[k] = c[9]
                        if "SPIKED" in c[7] or "RD_VID" in c[7]:
                            hits.append({"kind": "rare_leak", "severity": "gate",
                                         "detail": f"{role}:{p.name} 变异行带真值标记 INFO={c[7]}"})
                        if len(got) == len(want):
                            break
            found_gt = got.get(next(iter(want)))
            for k, g_exp in list(want.items())[1:]:
                if g_exp and got.get(k) != g_exp:
                    hits.append({"kind": "rare_spike_readback_failed", "severity": "gate",
                                 "detail": f"{role}: 诱饵 {k[0]}:{k[1]} 期望 GT {g_exp},读回 {got.get(k)}"})
            if bad_ref or not any("GRCh38" in h or "hg38" in h or "GRCh38" in "".join(head) for h in head):
                hits.append({"kind": "rare_reference_not_grch38", "severity": "gate", "detail": f"{role}:{p.name}"})
            if found_gt != f["gt"]:
                hits.append({"kind": "rare_spike_readback_failed", "severity": "gate",
                             "detail": f"{role}: 期望 GT {f['gt']},读回 {found_gt}"})
        hits += _check_ped(gen, cs)
    # R12 solver-visible text attachments (PED, VCF headers) do not name the answer
    if not lat.get("rare_dx_in_text"):
        _file_leaks = attachment_text_leaks(att, [t for t in leak_terms if t])
        if _file_leaks:
            hits.append({"kind": "rare_attachment_leak", "severity": "gate", "detail": _file_leaks[:4]})
    nar = att.get("narrative")
    if nar:
        from .narrative import leak_terms, sha256_text
        mp = pathlib.Path(nar["path"])
        if not mp.exists():
            hits.append({"kind": "rare_attachment_missing", "severity": "gate", "detail": str(mp)})
        else:
            md = mp.read_text(encoding="utf-8")
            if sha256_text(md) != nar["sha256"]:
                hits.append({"kind": "rare_attachment_sha_mismatch", "severity": "gate", "detail": str(mp)})
            bad = [x for x in nar.get("spans") or [] if md[x["start"]:x["end"]] != md[x["start"]:x["end"]].strip() or x["end"] <= x["start"]]
            if bad:
                hits.append({"kind": "rare_narrative_span_bad", "severity": "gate", "detail": bad[:2]})
            leaks = [t for t in leak_terms(spec) if _mentions(md, t, word=en, fold=en)]
            if leaks:
                hits.append({"kind": "rare_narrative_leak", "severity": "gate", "detail": leaks})
    img = att.get("imaging")
    if img:
        if not all(("CC BY" in l or "Creative Commons Attribution" in l) for l in img.get("license") or ["?"]):
            hits.append({"kind": "rare_license_not_open", "severity": "gate", "detail": img.get("license")})
        for s in img.get("series") or []:
            z = pathlib.Path(s["path"])
            if not z.exists():
                hits.append({"kind": "rare_attachment_missing", "severity": "gate", "detail": str(z)})
                continue
            bad = _dicom_phi(z)
            if bad:
                hits.append({"kind": "rare_dicom_phi", "severity": "gate", "detail": f"{z.name}: {bad}"})
    if att.get("eeg"):
        hits += _check_eeg(att["eeg"], _leak_list)
    return hits


def _proband_row(ped: pathlib.Path) -> list[str] | None:
    """The proband row of a PED: the `<case>-P` individual, else the only affected individual
    without parents (an external PED such as Exomiser's names its own ids)."""
    rows = [l.split("\t") for l in ped.read_text(encoding="utf-8").splitlines() if l and not l.startswith("#")]
    rows = [r for r in rows if len(r) >= 6]
    named = [r for r in rows if r[1].endswith("-P") and r[5] == "2"]
    if named:
        return named[0]
    lone = [r for r in rows if r[5] == "2" and r[2] == "0" and r[3] == "0"]
    return lone[0] if len(lone) == 1 else None


def _check_ped(gen: dict, cs) -> list[dict]:
    ped = pathlib.Path((gen.get("ped") or {}).get("path") or "")
    if not ped.is_file():
        return []
    hits = []
    if (gen.get("ped") or {}).get("sha256") and _sha256(ped) != gen["ped"]["sha256"]:
        hits.append({"kind": "rare_attachment_sha_mismatch", "severity": "gate", "detail": str(ped)})
    _sex = (cs.raw or {}).get("sex")        # no inline default (single source: job.CaseSpec.RAW_DEFAULTS)
    sexn = {"M": "1", "F": "2"}.get(str(_sex or ""))
    prob = _proband_row(ped)
    if sexn is None or not prob or prob[4] != sexn:
        hits.append({"kind": "rare_ped_sex_mismatch", "severity": "gate", "detail": prob})
    return hits


def _check_refused_genome(gen: dict) -> list[dict]:
    """A VCF that is not GRCh38 by design: present, unchanged, and really not GRCh38."""
    hits = []
    for role, f in (gen.get("files") or {}).items():
        p = pathlib.Path(f["path"])
        if not p.is_file():
            hits.append({"kind": "rare_attachment_missing", "severity": "gate", "detail": str(p)})
            continue
        if _sha256(p) != f["sha256"]:
            hits.append({"kind": "rare_attachment_sha_mismatch", "severity": "gate", "detail": str(p)})
        head = "".join(_vcf_header_lines(p))
        if not any(t in head for t in ("assembly=b37", "GRCh37", "hg19")) or "GRCh38" in head:
            hits.append({"kind": "rare_refused_vcf_not_grch37", "severity": "gate", "detail": f"{role}:{p.name}"})
    return hits


def _check_os_genome(gen: dict, proband_sex: str | None = None) -> list[dict]:
    """Round-2 genome: every member file present and unchanged, GRCh38, sample column = PED id,
    the true variant and every decoy read back in EVERY member file with the gold genotype --
    an explicit hom-ref row for a member who does not carry (a missing row reads as "unknown"
    to a solver, not "0/0") --, the spike copies the background style, and it is rare."""
    from . import genome_os as GO
    hits = []
    tv = gen["variant"]
    for role, f in gen["files"].items():
        p = pathlib.Path(f["path"])
        if not p.is_file():
            hits.append({"kind": "rare_attachment_missing", "severity": "gate", "detail": str(p)})
            continue
        if _sha256(p) != f["sha256"]:
            hits.append({"kind": "rare_attachment_sha_mismatch", "severity": "gate", "detail": str(p)})
        head = _vcf_header_lines(p)
        if "##reference=GRCh38\n" not in head:
            hits.append({"kind": "rare_reference_not_grch38", "severity": "gate", "detail": f"{role}:{p.name}"})
        if not head or head[-1].rstrip("\n").split("\t")[9:] != [f["sample"]]:
            hits.append({"kind": "rare_vcf_sample_id", "severity": "gate", "detail": f"{role}: {head[-1:] and head[-1].strip()}"})
        want = [(tv, (gen.get("genotypes") or {}).get(role))] + [(d, (d.get("genotypes") or {}).get(role)) for d in gen.get("decoys") or []]
        for v, gt in want:
            carries = bool(gt) and any(a not in ("0", ".") for a in str(gt).replace("|", "/").split("/"))
            got = GO.readback(p, v["chrom"], v["pos"], v["ref"], v["alt"])
            _n = lambda x: None if x is None else str(x).replace("|", "/")
            # a hom-ref gold reads back as 0/0, or as haploid 0 at a male's non-PAR chrX site
            ok = (_n(got) == _n(gt)) if carries else (got is not None and _n(got) in ("0/0", "0"))
            if not ok:
                hits.append({"kind": "rare_spike_readback_failed", "severity": "gate",
                             "detail": f"{role}: {v['chrom']}:{v['pos']} expected {gt}, read {got}"})
            if got is not None:
                rec = GO.record(p, v["chrom"], v["pos"], v["ref"], v["alt"])
                st = gen.get("style") or {}
                bad = (rec[7] != "." or rec[5] != st.get("qual") or rec[6] != st.get("filter")
                       or (st.get("id") == "colon") != (rec[2].count(":") == 3)
                       or (len(rec[9]) > 1 and ("|" in rec[9]) != bool(st.get("phased"))))
                if bad:
                    hits.append({"kind": "rare_spike_stands_out", "severity": "gate",
                                 "detail": f"{role}: {v['chrom']}:{v['pos']} {rec[2]} {rec[5]} {rec[6]} {rec[7]} {rec[9]} vs {st}"})
    # a male member is haploid at non-PAR chrX / chrY: no heterozygous row there, anywhere in his file
    sexes = {"proband": proband_sex, "father": "M", "mother": "F"}
    for role, f in gen["files"].items():
        p = pathlib.Path(f["path"])
        if sexes.get(role) == "M" and p.is_file():
            het = GO.male_het_rows(p)
            if het:
                hits.append({"kind": "rare_male_het_nonpar", "severity": "gate", "detail": f"{role}: {het[:4]}"})
    zyg = "het" if str((gen.get("genotypes") or {}).get("proband")) in ("0/1", "0|1", "1/0", "1|0") else "hom"
    if not GO.rare_enough(tv.get("gnomad"), zyg):
        hits.append({"kind": "rare_spike_not_rare", "severity": "gate", "detail": tv.get("gnomad")})
    return hits


def _check_eeg(eeg: dict, terms: list[str]) -> list[dict]:
    """Each recording: present, unchanged, the file reads back to the gold metadata, and its
    header follows the de-identification rule -- EDF+ `X X X X` or a single-token code in a plain
    EDF when phi_expected is false; the synthetic identifiers present when it is true."""
    from . import edf as E
    hits = []
    for r in eeg.get("recordings") or []:
        p = pathlib.Path(r["path"])
        if not p.is_file():
            hits.append({"kind": "rare_attachment_missing", "severity": "gate", "detail": str(p)})
            continue
        if _sha256(p) != r["sha256"]:
            hits.append({"kind": "rare_attachment_sha_mismatch", "severity": "gate", "detail": str(p)})
        try:
            meta, h = E.recording_meta(p), E.read_header(p)
        except (ValueError, OSError) as e:
            hits.append({"kind": "rare_eeg_unreadable", "severity": "gate", "detail": f"{p.name}: {e}"})
            continue
        diff = [k for k in ("format", "duration_sec", "channel_count", "sfreq_hz", "annotations") if meta[k] != r.get(k)]
        if diff:
            hits.append({"kind": "rare_eeg_meta_mismatch", "severity": "gate", "detail": f"{p.name}: {diff}"})
        if r.get("phi_expected"):
            if not h.edfplus or not all(str(v) in h.patient for v in r.get("phi_values") or ["?"]):
                hits.append({"kind": "rare_eeg_phi_marker_missing", "severity": "gate", "detail": f"{p.name}: {h.patient!r}"})
        elif (h.edfplus and h.patient != "X X X X") or (not h.edfplus and (not h.patient or " " in h.patient)):
            hits.append({"kind": "rare_eeg_phi", "severity": "gate", "detail": f"{p.name}: patient field {h.patient!r}"})
        if h.edfplus and not h.recording.startswith("Startdate "):
            hits.append({"kind": "rare_eeg_recording_field", "severity": "gate", "detail": f"{p.name}: {h.recording!r}"})
        text = " ".join([h.patient, h.recording] + [a["text"] for a in meta["annotations"]])
        leaked = [t for t in terms if _mentions(text, t, word=True)]
        if leaked:
            hits.append({"kind": "rare_attachment_leak", "severity": "gate", "detail": f"{p.name}: {leaked}"})
    return hits


def pick_clean_dicom_patient(base: pathlib.Path, pats: list[str], first: str, used: set | None = None) -> str:
    """The seeded pick, unless its series fail the same PHI check the launch gate runs
    (`_dicom_phi`: PatientName must be empty or equal PatientID, no birth date) — then the next
    patient in sorted order that passes, wrapping round. TCIA's own de-identification is not
    uniform (Soft-tissue-Sarcoma STS_032 carries PatientName 'STD_032', 2026-09-22), and a
    case must not be built only to be refused at the gate for a data-side quirk."""
    order = pats[pats.index(first):] + pats[:pats.index(first)] if first in pats else pats
    for pid in order:
        if used is not None and (base.name, pid) in used:
            continue
        zips = [z for z, _ in select_series(base / pid)]
        if zips and not any(_dicom_phi(z) for z in zips):
            if pid != first:
                log.warning("[rare] dicom patient %s fails the PHI check, using %s", first, pid)
            return pid
    raise RareSpecError(f"影像集合 {base.name} 无通过 PHI 检查的患者目录")


SERIES_MIN_INSTANCES = 20          # an image series, not an RTSTRUCT/RTPLAN/RTDOSE single object
SERIES_BYTES_CAP = 150 * 1024 * 1024


def _series_meta(z: pathlib.Path) -> dict:
    """Modality / instance count / bytes of one series zip (first .dcm header only)."""
    import io as _io
    n = 0
    mod = ""
    try:
        with zipfile.ZipFile(z) as zf:
            names = [x for x in zf.namelist() if x.endswith(".dcm")]
            n = len(names)
            if names:
                try:
                    import pydicom
                    ds = pydicom.dcmread(_io.BytesIO(zf.read(names[0])), stop_before_pixels=True)
                    mod = str(ds.get("Modality", "") or "")
                except ImportError:
                    mod = ""
    except zipfile.BadZipFile:
        pass
    return {"modality": mod, "n": n, "bytes": z.stat().st_size}


def select_series(pat_dir: pathlib.Path, k: int = 3) -> list[tuple[pathlib.Path, dict]]:
    """Up to `k` series of a patient: CT/MR with ≥ SERIES_MIN_INSTANCES instances first (largest
    first), RT objects (RTSTRUCT/RTPLAN/RTDOSE, one object each) only as filler when there are
    not enough image series. A single series bigger than SERIES_BYTES_CAP is skipped, and the
    running total stays under it; if that leaves nothing, the smallest series is kept, because a
    case with no attachment is worse than one over budget. 2026-09-22: the old "first three zips
    by UID" gave 9/20 cases RTSTRUCT-only attachments and one 501 MB case."""
    metas = [(z, _series_meta(z)) for z in sorted(pat_dir.glob("*.zip"))]
    if not metas:
        return []
    images = sorted((x for x in metas if x[1]["modality"] in ("CT", "MR") and x[1]["n"] >= SERIES_MIN_INSTANCES),
                    key=lambda x: (-x[1]["n"], x[0].name))
    rest = [x for x in metas if x not in images]
    chosen: list[tuple[pathlib.Path, dict]] = []
    total = 0
    for z, m in images + rest:
        if len(chosen) >= k or m["bytes"] > SERIES_BYTES_CAP or total + m["bytes"] > SERIES_BYTES_CAP:
            continue
        chosen.append((z, m))
        total += m["bytes"]
    return chosen or [min(metas, key=lambda x: x[1]["bytes"])]


def _dicom_phi(z: pathlib.Path) -> str | None:
    """抽 3 个 .dcm:PatientName ∈ {空, PatientID}、生日空。

    2026-09-21:先走 pydicom(装了的话)。裸字节扫只认 explicit VR,TCIA 的 implicit VR 文件
    一个 tag 都找不到 ⇒ 返回 None 与「干净」同形。pydicom 不在时退回裸扫并**把这一事实写进返回值**。
    """
    try:
        import io as _io
        import pydicom
    except ImportError:
        pydicom = None
    if pydicom is not None:
        try:
            with zipfile.ZipFile(z) as zf:
                for n in [x for x in zf.namelist() if x.endswith(".dcm")][:3]:
                    ds = pydicom.dcmread(_io.BytesIO(zf.read(n)), stop_before_pixels=True)
                    pn, pid, bd = (str(ds.get(k, "") or "").strip() for k in ("PatientName", "PatientID", "PatientBirthDate"))
                    if pn not in ("", pid):
                        return f"PatientName={pn!r}"
                    if bd:
                        return f"PatientBirthDate={bd!r}"
            return None
        except zipfile.BadZipFile:
            return "bad zip"
    def tag(b: bytes, t: bytes):
        i = b.find(t)
        while i != -1:
            vr = b[i + 4:i + 6]
            if vr in (b"LO", b"PN", b"CS", b"DA", b"SH"):
                ln = struct.unpack("<H", b[i + 6:i + 8])[0]
                return b[i + 8:i + 8 + ln].decode("latin1").strip()
            i = b.find(t, i + 1)
        return None
    try:
        with zipfile.ZipFile(z) as zf:
            dcms = [n for n in zf.namelist() if n.endswith(".dcm")][:3]
            for n in dcms:
                b = zf.read(n)
                pn, pid, bd = tag(b, b"\x10\x00\x10\x00"), tag(b, b"\x10\x00\x20\x00"), tag(b, b"\x10\x00\x30\x00")
                if pn not in (None, "", pid):
                    return f"PatientName={pn!r}"
                if bd:
                    return f"PatientBirthDate={bd!r}"
    except zipfile.BadZipFile:
        return "bad zip"
    return None
