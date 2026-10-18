"""twins.py -- one-fact twin records: the same records as a built batch, with one declared change.

A twin tests a claim about model behaviour: each model answers the original and the twin, and the paired difference
is the effect of the change. The generator writes the record and its gold from the hidden truth, so a twin can be
built through the normal build path and checked against the original field by field.

    haenv twin <job.yaml> --op <op> --from-batch <batch dir> --out <dir> [--items ID,ID] [--job-id ID]

`--from-batch` is a built batch of the job (a batch directory, or a batch prefix whose `-pk1` holds the rows). The
command

1. selects the items the operation applies to (all eligible items of the batch, or those of `--items`, each of which
   must be eligible);
2. writes two jobs under `--out`: `<job id>.job.yaml`, the twins, and `<job id>-orig.job.yaml`, the same items
   unchanged. Both copy the selected cases of the source job verbatim; the twin job records the operation and the
   per-item details under `_provenance.twin`;
3. builds both deterministically (`haenv build --gen deterministic --fresh`). For a job that carries
   `_provenance.twin`, the build applies the operation to every emitted case before the cases are saved, so the
   case gates have seen the full record and the batch gates see the twins. Rebuilding the twin job reproduces the
   twins;
4. compares the rows (`cases.jsonl`: the patient with its hidden truth; `payloads.jsonl`: what the model reads) with
   the source batch: the original build must be byte-identical to it, and each twin must differ, and only at the
   operation's declared fields. It writes `twin_record.json` into the twin batch (counts and sha16, never record text)
   and exits 1 on any failure.

Operations (`OPS`):

  drop-reagent-note        pack ④: the reagent-change note leaves the method strings of the follow-up report
  reword-reagent-note-1/2  pack ④: the same note in another wording (`REWORDED_NOTES`)
  drop-planted-symptoms    pack ①: the patient-reported symptom entries the generator planted as the support of the
                           gold line(s) leave the evidence ledger
  unname-explainer         pack ①: the established diagnoses that explain away a hidden line are no longer named in
                           `user_profile.known_conditions`; another visible explainer may stay (strata
                           `known_plus_line` / `known_only`)

SYNTHETIC data, evaluation use only, not medical advice.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

#: The suffix pack ④'s report renderer appends to a method string for a reagent change (`haenv/p4/render.py`).
REAGENT_NOTE = "，试剂更换新批号（同厂家，批间比对已通过）"
from haenv.p4.render import REAGENT_NOTICE
#: Other wordings of the same fact (a new lot of the same manufacturer's reagent, lot-to-lot check passed), for testing whether an effect
#: of the note depends on how it is worded. Fixed text: a changed wording is a new operation, never an edit.
REWORDED_NOTES = {"reword-reagent-note-1": "（本次起使用同厂家新批号试剂，已完成批间比对）",
                  "reword-reagent-note-2": "；检验科已更换试剂批号（同厂家），批间验证通过"}
P4_NOISE = "analytic_biological_noise"


def sha16(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class TwinOp:
    """`select(job case, cases row, payload row)` -> per-item details (JSON) or None when the op does not apply;
    `apply(raw case, details)` changes the emitted case in place and returns a per-item record (counts, sha16)."""
    name: str
    pack: str
    what: str
    fields: tuple[str, ...]
    select: Callable[[dict, dict, dict], dict | None]
    apply: Callable[[object, dict], dict]

    def declared(self) -> set[str]:
        """The row paths (list indices as *) a twin may differ at, in `cases.jsonl` (/case) and `payloads.jsonl`."""
        return {f"/{top}/{f}" for f in self.fields for top in ("case", "sp", "vp")}


# ------------------------------------------------------------------ drop-reagent-note (pack ④)
def _p4_report(container: dict) -> list:
    return ((container.get("prediction_context") or {}).get("followup_p4") or {}).get("report") or []


def _p4_notices(container: dict) -> list:
    return ((container.get("prediction_context") or {}).get("followup_p4") or {}).get("laboratory_method_record") or []


def _select_note(job_case: dict, row: dict, _pay: dict) -> dict | None:
    c = row["case"]
    adj = (c.get("adjudication") or {}).get("followup_p4") or {}
    pc = (c.get("prediction_context") or {}).get("followup_p4") or {}
    days = (pc.get("previous_draw_day"), pc.get("today"))
    if adj.get("class") != P4_NOISE:
        return None
    found = [r.get("valid_from") for r in _p4_notices(c) if REAGENT_NOTICE in str(r.get("text") or "")]
    if found:
        return {"days_with_note": found}
    if not any(REAGENT_NOTE in str(r.get("method") or "") for r in _p4_report(c) if r.get("day") in days):
        return None
    return {"days_with_note": [r.get("day") for r in _p4_report(c) if REAGENT_NOTE in str(r.get("method") or "")]}


def _note_to(new: str) -> Callable[[object, dict], dict]:
    """Replace the reagent-change note with `new` ("" removes it) in every report row's method string."""
    def apply(raw, _details: dict) -> dict:
        n = 0
        for r in _p4_report({"prediction_context": raw.prediction_context}):
            m = str(r.get("method") or "")
            if REAGENT_NOTE in m:
                r["method"] = m.replace(REAGENT_NOTE, new)
                n += 1
        records = _p4_notices({"prediction_context": raw.prediction_context})
        n_record = 0
        for r in records:
            if REAGENT_NOTICE in str(r.get("text") or ""):
                r["text"] = r["text"].replace(REAGENT_NOTICE, new)
                n_record += 1
        if records:
            return {"report_rows_changed": n, "source_records_changed": n_record}
        return {"report_rows_changed": n}
    return apply


_apply_note = _note_to("")


# ------------------------------------------------------------------ drop-planted-symptoms (pack ①)
def _planted_ids(row: dict) -> list[str]:
    man = (row.get("question") or {}).get("injected_manifest") or {}
    return [str(x) for x in man.get("real_symptom_evidence_ids") or ()]


def _select_symptoms(_job_case: dict, row: dict, _pay: dict) -> dict | None:
    ids = set(_planted_ids(row))
    ledger = row["case"].get("evidence_ledger") or []
    gone = [e for e in ledger if str(e.get("evidence_id")) in ids]
    if not gone:
        return None
    return {"evidence_ids": sorted(ids), "n_removed": len(gone)}


def _apply_symptoms(raw, details: dict) -> dict:
    ids = set(details["evidence_ids"])
    ledger = list(raw.evidence_ledger or [])
    gone = [e for e in ledger if str(e.get("evidence_id")) in ids]
    raw.evidence_ledger = [e for e in ledger if str(e.get("evidence_id")) not in ids]
    if len(gone) != details["n_removed"]:
        raise ValueError(f"{raw.case_id}: {len(gone)} planted entries in the build, {details['n_removed']} selected")
    return {"n_ledger": len(ledger), "n_removed": len(gone), "removed_entries_sha16": sha16(gone),
            "removed_symptoms_sha16": [sha16(e.get("symptom")) for e in gone]}


# ------------------------------------------------------------------ unname-explainer (pack ①)
def classify_explainers(job_case: dict, known_visible: list[str], med_text: str) -> dict | None:
    """The explainers of a case's explained-away line(s): established diagnoses the record names that can be removed
    (an explainer base that is not the primary disease), and the explainers that stay (another gold line, the primary
    disease). None for a case without an explained-away line.

    A row counts only when one of its findings licenses a clause still on the record in the line's symptoms
    (`line.symptoms[].text`, the clue excluded): a row whose covered symptom the clue replaced explains nothing the
    solver reads, and unnaming its explainer would leave the record's explained evidence as it was (W7 follow-up)."""
    from haenv.world.generation.build import EXPLAINER_BASES
    from haenv.m2 import core
    m2 = (job_case.get("latent") or {}).get("m2") or {}
    lines = (m2.get("explain_away") or {}).get("lines") or []
    if not lines:
        return None
    disease = str(job_case["raw"].get("disease") or "")
    comps = list(core.components(str(job_case["latent"].get("ddx_spec_id") or "")))
    rows = {r["id"]: r for r in core.ea_rows()}
    removable, stays, per_line = set(), set(), []
    for ln in lines:
        rm, st = set(), set()
        atoms = [a for x in ln.get("symptoms") or () for a in core.symptom_atoms(str(x.get("text") or ""))]
        for rid in ln.get("rows") or ():
            r = rows[rid]
            if not core.drug_visible(r, med_text):
                continue
            if not any(core.atom_licensed(a, f) for a in atoms for f in r.get("findings") or ()):
                continue
            for k in r.get("explainer_known") or ():
                if k in known_visible:
                    (rm if (k in EXPLAINER_BASES and k != disease) else st).add(k)
            st.update(x for x in core.EA_LINE.get(str(r.get("explainer_line") or ""), ())
                      if x in comps and x != ln["component"])
        per_line.append({"component": ln["component"], "unname": sorted(rm), "stays": sorted(st)})
        removable |= rm
        stays |= st
    if not removable:
        why = "line_only" if not any(s in EXPLAINER_BASES or s == disease for s in stays) else "primary_plus_line"
        return {"include": False, "why": why, "lines": per_line}
    return {"include": True, "unname": sorted(removable), "stays": sorted(stays),
            "stratum": "known_only" if not stays else "known_plus_line", "lines": per_line}


def _select_unname(job_case: dict, _row: dict, pay: dict) -> dict | None:
    from haenv.m2 import core
    sp = pay["sp"]
    kv = list((sp.get("user_profile") or {}).get("known_conditions") or [])
    scene = ((sp.get("prediction_context") or {}).get("m2_frame") or {}).get("scene")
    led = [(str(e.get("symptom")), str(e.get("context") or "")) for e in (sp.get("evidence_ledger") or [])
           if e.get("source_type") == "patient_reported_symptom"]
    r = classify_explainers(job_case, kv, core.visible_medication_text(scene, led))
    if not r or not r["include"]:
        return None
    return {"unname": r["unname"], "stays": r["stays"], "stratum": r["stratum"]}


def _apply_unname(raw, details: dict) -> dict:
    drop = set(details["unname"])
    up = dict(raw.user_profile or {})
    before = list(up.get("known_conditions") or [])
    up["known_conditions"] = [k for k in before if k not in drop]
    raw.user_profile = up
    return {"n_unnamed": len(before) - len(up["known_conditions"])}


OPS: dict[str, TwinOp] = {op.name: op for op in (
    TwinOp("drop-reagent-note", "p4", "the reagent-change note removed from the follow-up report's method strings",
           ("prediction_context/followup_p4/report/*/method", "prediction_context/followup_p4/laboratory_method_record/*/text"), _select_note, _apply_note),
    *(TwinOp(name, "p4", f"the reagent-change note reworded as {text!r}",
             ("prediction_context/followup_p4/report/*/method", "prediction_context/followup_p4/laboratory_method_record/*/text"), _select_note, _note_to(text))
      for name, text in REWORDED_NOTES.items()),
    TwinOp("drop-planted-symptoms", "m2", "the planted patient-reported symptom entries removed from the evidence ledger",
           ("evidence_ledger",), _select_symptoms, _apply_symptoms),
    TwinOp("unname-explainer", "m2", "the explaining established diagnoses no longer named in known_conditions",
           ("user_profile/known_conditions",), _select_unname, _apply_unname),
)}


# ------------------------------------------------------------------ the build hook
def spec_of(job_path) -> dict | None:
    """The twin specification a job carries (`_provenance.twin`), or None."""
    if not job_path:
        return None
    from haenv.resources.yamlcache import load_yaml
    tw = ((load_yaml(Path(job_path)) or {}).get("_provenance") or {}).get("twin")
    if not tw:
        return None
    if tw.get("op") not in OPS:
        raise SystemExit(f"[twin] unknown operation {tw.get('op')!r} in {job_path}")
    return tw


def apply_all(spec: dict, built: dict) -> dict:
    """Apply the job's operation to every emitted case; every case must have details, every detail a case."""
    op = OPS[spec["op"]]
    items = spec.get("items") or {}
    missing = sorted(set(items) - set(built))
    if missing or set(built) - set(items):
        raise SystemExit(f"[twin] cases and twin items differ: not emitted {missing}, "
                         f"no details {sorted(set(built) - set(items))}")
    return {cid: op.apply(built[cid], items[cid]) for cid in built}


# ------------------------------------------------------------------ rows and checks
def batch_files(batch: Path) -> Path:
    """The directory holding a batch's rows: the batch dir itself, or its first round."""
    batch = Path(batch)
    return batch if (batch / "cases.jsonl").is_file() else batch.parent / f"{batch.name}-pk1"


def rows(batch: Path, name: str) -> dict:
    out = {}
    for ln in (batch_files(batch) / name).read_text(encoding="utf-8").splitlines():
        if ln.strip():
            out.setdefault(json.loads(ln)["case_id"], ln)
    return out


def _diff(a, b, path="") -> list[str]:
    if type(a) is not type(b):
        return [path or "/"]
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            out += [f"{path}/{k}"] if k not in a or k not in b else _diff(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [path or "/"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _diff(x, y, f"{path}/{i}")]
    return [] if a == b else [path or "/"]


def changed(a: str, b: str) -> list[str]:
    """Changed field paths of two JSON rows, list indices written as *."""
    return sorted({re.sub(r"/\d+(?=/|$)", "/*", p) for p in _diff(json.loads(a), json.loads(b))})


def compare(op: TwinOp, src: Path, orig: Path, twin: Path, items: list[str]) -> tuple[dict, list[str]]:
    per, bad = {}, []
    for name in ("cases.jsonl", "payloads.jsonl"):
        s, o, t = rows(src, name), rows(orig, name), rows(twin, name)
        for cid in items:
            if o.get(cid) != s.get(cid):
                bad.append(f"{cid} {name}: the original build differs from the source batch")
            if cid not in t:
                bad.append(f"{cid} {name}: no twin row")
                continue
            got = changed(s[cid], t[cid]) if t[cid] != s.get(cid) else []
            per.setdefault(cid, {})[name] = got
            if not got:
                bad.append(f"{cid} {name}: the twin is identical to the original")
            elif set(got) - op.declared():
                bad.append(f"{cid} {name}: undeclared change at {sorted(set(got) - op.declared())}")
    return per, bad


def carry_audit(op: TwinOp, src: Path, builds: dict[Path, str]) -> bool:
    """A pack's publish gate refuses to answer a batch without its audit marker. A twin or original build passes on
    the marker of an audited source batch (its rows are that batch's, changed only where declared and checked):
    the marker names the source cases' sha256 and what the build is. Nothing is written for an unaudited source."""
    from haenv.world.validation.shared_audit import marker_name
    mk = marker_name(op.pack)
    if not (batch_files(src) / mk).is_file():
        return False
    sha = hashlib.sha256((batch_files(src) / "cases.jsonl").read_bytes()).hexdigest()
    for b, what in builds.items():
        (Path(b) / mk).write_text(f"{sha}\n{what} of the audited batch above, written by `haenv twin` after the "
                                  f"field-by-field check (twin_record.json of the twin build)\n", encoding="utf-8")
    return True


# ------------------------------------------------------------------ the command
def _write_job(src: Path, job: dict, cases: list, job_id: str, dest: Path, prov: dict) -> None:
    import yaml
    j = {k: v for k, v in job.items() if k not in ("cases", "_provenance", "pack")}
    j["job_id"] = job_id
    j["report"] = f"eval-{job_id}.md"
    j["_provenance"] = prov
    j["cases"] = cases
    head = "".join(ln for ln in src.read_text(encoding="utf-8").splitlines(True)
                   if ln.startswith(("# haenv-canary", "# canary")))
    head += (f"# Generated by `haenv twin`; do not edit by hand.\n"
             f"# {job_id}: {len(cases)} cases of {src.name}{' (twins)' if 'twin' in prov else ' (unchanged)'}.\n")
    dest.write_text(head + yaml.safe_dump(j, allow_unicode=True, sort_keys=False, width=200), encoding="utf-8")


def _newest_batch(job_path: Path) -> Path:
    """The batch the last `--fresh` build of this job wrote (its newest stamp directory)."""
    from haenv import output_root
    from haenv.app.job import load_job
    base = load_job(str(job_path), root=output_root()).cases_file.parent
    if (base / "cases.jsonl").is_file():
        return base
    stamps = sorted(d for d in base.iterdir() if d.is_dir() and re.fullmatch(r"\d{8}-\d{6}", d.name))
    if not stamps:
        raise SystemExit(f"[twin] no batch under {base}")
    return stamps[-1]


def run(job_path: str, op_name: str, from_batch: str, out: str, items: str | None = None,
        job_id: str | None = None, build=None) -> int:
    """`haenv twin`: select, write both jobs, build both, compare. `build(argv) -> rc` runs `haenv build`."""
    from haenv.resources.yamlcache import load_yaml
    op = OPS[op_name]
    src_job = Path(job_path).resolve()
    src = Path(from_batch).resolve()
    job = copy.deepcopy(load_yaml(src_job))
    by_id = {c["case_id"]: c for c in job["cases"]}
    cases, pays = rows(src, "cases.jsonl"), rows(src, "payloads.jsonl")
    want = [x for x in (items or "").split(",") if x] or [c["case_id"] for c in job["cases"] if c["case_id"] in cases]
    details, skipped = {}, []
    for cid in want:
        if cid not in by_id or cid not in cases or cid not in pays:
            raise SystemExit(f"[twin] {cid}: not in both the job and the batch")
        d = op.select(by_id[cid], json.loads(cases[cid]), json.loads(pays[cid]))
        if d is None:
            if items:
                raise SystemExit(f"[twin] {cid}: {op.name} does not apply to it")
            skipped.append(cid)
        else:
            details[cid] = d
    if not details:
        raise SystemExit(f"[twin] {op.name} applies to none of the {len(want)} item(s)")
    ids = [cid for cid in want if cid in details]
    tid = job_id or f"{job['job_id']}-twin-{op.name}"
    outd = Path(out).resolve()
    outd.mkdir(parents=True, exist_ok=True)
    src_info = {"source_job": src_job.name, "source_job_sha256": hashlib.sha256(src_job.read_bytes()).hexdigest(),
                "source_cases_sha256": hashlib.sha256((batch_files(src) / "cases.jsonl").read_bytes()).hexdigest()}
    sel = [copy.deepcopy(by_id[cid]) for cid in ids]
    tj, oj = outd / f"{tid}.job.yaml", outd / f"{tid}-orig.job.yaml"
    _write_job(src_job, job, sel, tid, tj, {**src_info, "twin": {"op": op.name, "what": op.what, "items": details}})
    _write_job(src_job, job, copy.deepcopy(sel), f"{tid}-orig", oj, {**src_info, "original_of": tid})
    print(f"[twin] {op.name}: {len(ids)} item(s) selected, {len(skipped)} not eligible; jobs -> {tj.name}, {oj.name}")
    for jp in (oj, tj):
        rc = build(["build", str(jp), "--gen", "deterministic", "--fresh"])
        if rc:
            print(f"[twin] build of {jp.name} failed (rc={rc})")
            return rc
    ob, tb = _newest_batch(oj), _newest_batch(tj)
    per, bad = compare(op, src, ob, tb, ids)
    twin_pay = rows(tb, "payloads.jsonl")
    if op.name == "drop-reagent-note" or op.name in REWORDED_NOTES:
        bad += [f"{cid}: the original note is still in the payload" for cid in ids
                if REAGENT_NOTE in twin_pay[cid] or REAGENT_NOTICE in twin_pay[cid]]
    if op.name in REWORDED_NOTES:
        bad += [f"{cid}: the new wording is not in the payload" for cid in ids if REWORDED_NOTES[op.name] not in twin_pay[cid]]
    if op.name == "drop-planted-symptoms":
        for cid in ids:
            led = json.loads(cases[cid])["case"].get("evidence_ledger") or []
            gone = [e for e in led if str(e.get("evidence_id")) in set(details[cid]["evidence_ids"])]
            if any(str(e.get("symptom") or "") and str(e["symptom"]) in twin_pay[cid] for e in gone):
                bad.append(f"{cid}: a removed symptom text is still in the payload")
    rec = {"op": op.name, "what": op.what, "declared_fields": list(op.fields), **src_info,
           "job_id": tid, "original_job_id": f"{tid}-orig", "items": ids, "not_eligible": skipped,
           "per_item": details, "changed_vs_source": per, "problems": bad}
    if not bad and not carry_audit(op, src, {ob: "unchanged original", tb: f"twin ({op.name}: {op.what})"}):
        print(f"[twin] the source batch has no audit marker: run tools/validation/pack_audit.py on the builds before answering them")
    (tb / "twin_record.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[twin] original build {ob.name}, twin build {tb.name}; record -> twin_record.json")
    print("\n".join(f"[twin] {b}" for b in bad) if bad else
          f"[twin] every twin differs from its original only at {', '.join(op.fields)}")
    return 1 if bad else 0

# ------------------------------------------------------------------ four-pack truth families
# This hook runs after build_case. Every changed record is re-rendered, re-golded,
# and checked through the production emission gates before it can be persisted.
FAMILY_BLOCKS = {"p2": "triage_p2", "p3": "meds_p3", "p4": "followup_p4"}
#: the semantic null edit: the first two entries of the record's known-condition list change places
NULL_KIND = "null-swap-conditions"
NULL_PATH = "user_profile/known_conditions"
NULL_MED_KIND = "null-swap-same-day-medications"
NULL_MED_PATH = "prediction_context/followup_p4/medication_record"
# Priority is a declared design choice, independent of trial enumeration order.
NULL_KINDS = (NULL_KIND, NULL_MED_KIND)
NULL_FIELDS = {NULL_KIND: NULL_PATH, NULL_MED_KIND: NULL_MED_PATH}


def same_day_medication_swap(rows):
    """One eligible adjacent transposition: earliest day, then lowest left index.

    Distinct uniquely resolved medicines may exchange places on the same day.
    Adjacency preserves each drug's own chronology, including repeated rows.
    Return (the exact reordered list, the declaration), or None if unavailable.
    """
    from haenv.world.validation.invariants import drugs_in
    eligible = []
    for i, (a, b) in enumerate(zip(rows, rows[1:])):
        if type(a.get("day")) is not int or type(b.get("day")) is not int or a["day"] != b["day"]:
            continue
        left, right = drugs_in(a.get("drug")), drugs_in(b.get("drug"))
        if len(left) == len(right) == 1 and left != right:
            eligible.append((a["day"], i, next(iter(left)), next(iter(right))))
    if not eligible:
        return None
    day, i, left, right = min(eligible)
    after = copy.deepcopy(rows)
    after[i], after[i + 1] = after[i + 1], after[i]
    return after, {"day": day, "indices": [i, i + 1], "drug_ids": [left, right]}


def _raw_dict(raw):
    from dataclasses import asdict
    return asdict(raw)


def _at(data, path):
    parts = path.split("/")
    for p in parts:
        data = data[int(p)] if isinstance(data, list) else data[p]
    return data


def _set_at(data, path, value):
    parent, key = path.rsplit("/", 1)
    node = _at(data, parent)
    node[int(key) if isinstance(node, list) else key] = copy.deepcopy(value)


def _covered(path, fields):
    path = re.sub(r"/\d+(?=/|$)", "/*", path).lstrip("/")
    fields = [re.sub(r"/\d+(?=/|$)", "/*", f) for f in fields]
    return any(path == f or path.startswith(f + "/") for f in fields)


def _m2_recomputed(raw, *, derive_workup=False):
    """Replay the production frame and optional gold-layer history on a copy."""
    from haenv.m2.core import frame_gold
    from haenv.world.history.history_dependency import policy
    from haenv.world.history.history_dependency import rate
    out = copy.deepcopy(raw)
    frame = frame_gold(out.latent_premise["meta"])
    if frame is None:
        out.adjudication.pop("m2_frame", None)
    else:
        out.adjudication["m2_frame"] = frame
        if "diversity" in (raw.adjudication.get("m2_frame") or {}):
            frame["diversity"] = copy.deepcopy(raw.adjudication["m2_frame"]["diversity"])
        if frame.get("frame") == "F2":
            frame["target_drug"] = out.latent_premise["patient_basics"]["regimen"]["drug"]
    P = policy("m2")
    plan = P.plan_of(out)
    if plan is not None and rate("m2", plan) > 0:
        # The production finalizer starts from native diagnostic keys, before
        # it substitutes archived evidence. Never feed its own prior output back.
        out.adjudication.get("ddx", {}).pop("historical_key_signals", None)
        P.finalize(out, str(out.case_id), plan)
    from haenv.m2.emr import finalize as finalize_emr
    finalize_emr(out)
    if "open_management" in out.adjudication:
        from haenv.scoring.management.plan import finalize
        # Ordinary build snapshots the already-finalized workup. Recompute it
        # only for a declared diagnosis intervention, not during source replay.
        finalize(out, str(out.case_id), "m2", derive_workup=derive_workup)
    return out


def family_gold(raw, pack: str) -> dict:
    """Re-derive the scored gold through the production rule reader, never a twin rule."""
    if pack == "m2":
        from haenv.m2.core import f3_truth
        reproduced = _m2_recomputed(raw)
        recomputed = reproduced.adjudication.get("m2_frame")
        gold = copy.deepcopy(raw.adjudication.get("m2_frame"))
        if gold != recomputed:
            raise ValueError("M2 stored frame differs from production recompute")
        if raw.adjudication.get("open_management") != reproduced.adjudication.get("open_management"):
            raise ValueError("M2 stored management facts differ from production recompute")
        if any(raw.prediction_context.get(k) != reproduced.prediction_context.get(k)
               for k in ("archived_laboratory_record", "record_policy")) or (
                raw.adjudication.get("ddx", {}).get("historical_key_signals") !=
                reproduced.adjudication.get("ddx", {}).get("historical_key_signals")):
            raise ValueError("M2 stored history differs from production recompute")
        if gold and gold.get("frame") == "F3":
            gold = {**gold, "rcv_truth": f3_truth(raw.longitudinal_data, gold,
                                                int(raw.prediction_context["prediction_time_T"]))}
        return {"frame": gold, "diagnosis": copy.deepcopy(raw.adjudication.get("ddx"))}
    block = FAMILY_BLOCKS[pack]
    g = raw.adjudication[block]
    if pack == "p2":
        from haenv.p2.rules import assess
        return assess(raw.prediction_context[block])
    if pack == "p3":
        from haenv.p3.gold import g3
        return g3(g)
    from haenv.p4.gold import g4
    recs = {int(r["day"]): r for r in g["records"]}
    return g4(recs[int(g["t0"])], recs[int(g["t1"])], float(g["rcv"]), tuple(g["band"]))


def _scored_gold(g, pack):
    # Readings, states, numerical percentages are still validated in full, but a
    # changed label means the scored categorical answer changed, not just its reason.
    return g.get("set") if pack == "p2" else g if pack == "m2" else g.get("class")


def _family_fields(pack, kind, path, details=None):
    """Downstream dependencies declared before mutation (not inferred from the diff)."""
    if kind in NULL_FIELDS:
        return (NULL_FIELDS[kind],)
    if pack == "m2":
        if kind == M2_CONFIRMATION_KIND:
            from haenv.m2.confirmation import twin_fields
            return twin_fields()
        return (_m2_line_fields((details or {}).get("footprint_streams") or ()) if kind == M2_LINE_KIND
                else tuple(OPS[kind].fields))
    block = FAMILY_BLOCKS[pack]
    gold = f"adjudication/{block}"
    plan = f"latent_premise/meta/{_family_slot()}/{pack}"
    if pack == "p2":
        field = path.split("/")[-1]
        streams = {"temperature_c": ("body_temp",), "heart_rate_bpm": ("heart_rate",),
                   "spo2_pct": ("spo2",), "systolic_bp_mmHg": ("systolic_bp",),
                   "diastolic_bp_mmHg": ("diastolic_bp",), "kg": ("weight",)}.get(field, ())
        bp = ((f"prediction_context/{block}/home_bp_log/*/systolic_mmHg",) if field == "systolic_bp_mmHg"
              else (f"prediction_context/{block}/home_bp_log/*/diastolic_mmHg",) if field == "diastolic_bp_mmHg" else ())
        return (path, *bp, *(f"longitudinal_data/{st}/*/value" for st in streams),
                "evidence_ledger/*/measured_value",
                *(f"{gold}/{f}" for f in ("rule", "set", "floor", "borderline", "readings", "tier", "plan")),
                plan)
    if pack == "p3":
        fields = (path, f"{gold}/g3", f"{gold}/class", f"{gold}/subtype", f"{gold}/plan", plan,
                  "adjudication/open_management")
        if kind == "p3-onset":
            return fields + (f"{gold}/diary/*/cause", f"prediction_context/{block}/symptom_diary",
                             f"prediction_context/{block}/encounter_record")
        if kind == "p3-adherence":
            return fields + (f"{gold}/pill_count", f"{gold}/visits/*/truth", f"{gold}/visits/*/printed",
                             f"{gold}/params/y_T", f"{gold}/params/zone", f"{gold}/comorbidity/uacr",
                             f"prediction_context/{block}/pill_count_today", f"prediction_context/{block}/visits",
                             f"prediction_context/{block}/medication_review",
                             f"prediction_context/{block}/lab_results", f"prediction_context/{block}/home_measurements")
        return fields + (f"{gold}/visits/*/printed", f"prediction_context/{block}/visits/*/reading",
                         f"prediction_context/{block}/lab_results", f"prediction_context/{block}/home_measurements")
    fields = (path, f"{gold}/records/*/printed", f"{gold}/g4", f"{gold}/class", f"{gold}/factor",
              f"{gold}/next_step", f"{gold}/obs_pct", f"{gold}/plan", plan,
              f"prediction_context/{block}/report")
    if kind in ("p4-remove-source", "p4-remove-candidate"):
        return fields + (f"{gold}/cause", f"{gold}/cause_params", f"prediction_context/{block}/diet_record")
    return fields


def _family_slot():
    from haenv.contracts.external_gold import SLOT
    return SLOT


def _reprint(record, table):
    from haenv.world.labworld.observe import make_record
    record.update(make_record(record["day"], record["time"], record["truth"], record["e"],
                              record["perturbations"], int(table["ndigits"]), tuple(table["range"])))


def _mutate_family(raw, d):
    """One bounded primitive edit, then deterministic downstream record updates."""
    pack, kind, path = d["pack"], d["kind"], d["path"]
    before, after = d["before"], d["after"]
    if before == after:
        raise ValueError("empty twin intervention")
    if pack == "m2" and kind == M2_CONFIRMATION_KIND:
        from haenv.m2.confirmation import mutate_twin
        mutate_twin(raw, d)
        return
    if pack == "m2" and kind == M2_LINE_KIND:
        _m2_remove_line(raw, d)
        return
    if pack == "m2":
        if kind not in ("drop-planted-symptoms", "unname-explainer"):
            raise ValueError("unknown M2 intervention")
        details = d["operation_details"]
        if ((kind == "unname-explainer" and len(details.get("unname") or ()) != 1)
                or (kind == "drop-planted-symptoms" and
                    (len(details.get("evidence_ids") or ()) != 1 or details.get("n_removed") != 1))):
            raise ValueError("M2 family must remove a single existing fact")
        OPS[kind].apply(raw, details)
        return
    if kind == NULL_KIND:
        kc = list(raw.user_profile.get("known_conditions") or ())
        if path != NULL_PATH or before != kc or len(kc) < 2 or kc[0] == kc[1] or after != [kc[1], kc[0], *kc[2:]]:
            raise ValueError("null edit must swap the first two known conditions")
        raw.user_profile["known_conditions"] = list(after)
        return
    if kind == NULL_MED_KIND:
        if pack != "p4" or path != NULL_MED_PATH:
            raise ValueError("same-day medication null is restricted to the P4 medication display")
        rows = raw.prediction_context["followup_p4"]["medication_record"]
        expected = same_day_medication_swap(rows)
        # JSON equality binds types and complete row content, not just parsed doses.
        dump = lambda obj: json.dumps(obj, ensure_ascii=False, sort_keys=True)
        if (expected is None or dump(before) != dump(rows) or dump(after) != dump(expected[0])
                or dump(d.get("operation_details")) != dump(expected[1])):
            raise ValueError("null edit must be the declared same-day adjacent distinct-medication swap, with all facts unchanged")
        raw.prediction_context["followup_p4"]["medication_record"] = expected[0]
        return
    block = FAMILY_BLOCKS[pack]
    g = raw.adjudication[block]
    data = {"adjudication": raw.adjudication, "prediction_context": raw.prediction_context}
    if _at(data, path) != before:
        raise ValueError("source fact differs from intervention precondition")
    if kind == "p2-measurement":
        if not re.fullmatch(r"prediction_context/triage_p2/(home_check_today/[a-z_]+|morning_weight/\d+/kg|wearable_daily/\d+/(irregular_rhythm_alerts|night_spo2_min_pct)|cgm_nightly/\d+/min_mmol_L)", path):
            raise ValueError("p2 path outside measurements")
        field = path.split("/")[-1]
        bounds = {"temperature_c": (0.5, 34, 42), "heart_rate_bpm": (5, 40, 180),
                  "respiratory_rate_per_min": (2, 8, 35), "spo2_pct": (2, 80, 100),
                  "systolic_bp_mmHg": (5, 70, 220), "diastolic_bp_mmHg": (5, 40, 130),
                  "kg": (0.3, 30, 200), "irregular_rhythm_alerts": (1, 0, 3),
                  "night_spo2_min_pct": (1, 80, 100), "min_mmol_L": (0.2, 2, 10)}
        if field not in bounds:
            raise ValueError("unsupported p2 measurement")
        step, lo, hi = bounds[field]
        if not lo <= after <= hi or abs(after - before) > step + 1e-8:
            raise ValueError("p2 intervention outside declared bound")
        _set_at(data, path, after)
        from haenv.p2.hooks import measurement_series
        b = raw.prediction_context[block]
        # The final evening BP and today's home check are the same observation.
        if field in ("systolic_bp_mmHg", "diastolic_bp_mmHg"):
            b["home_bp_log"][-1]["systolic_mmHg"] = b["home_check_today"]["systolic_bp_mmHg"]
            b["home_bp_log"][-1]["diastolic_mmHg"] = b["home_check_today"]["diastolic_bp_mmHg"]
        for stream, points in measurement_series(b).items():
            for q in raw.longitudinal_data.get(stream) or ():
                if int(q["ts"]) in points:
                    q["value"] = points[int(q["ts"])]
        from haenv.world.generation.build import realign_ledger_to_streams
        # The production ledger realigner uses final observations, not a second
        # independently generated clinical reading.
        realign_ledger_to_streams(raw.evidence_ledger, raw.longitudinal_data)
        return
    if pack == "p3":
        from haenv.world.labworld.meds import axis
        from haenv.world.labworld.meds import drug
        from haenv.world.labworld.meds import rule
        from haenv.p3.gold import ae_cause
        table = axis(g["axis"])
        if kind == "p3-onset":
            if not re.fullmatch(r"adjudication/meds_p3/diary/\d+/onset", path):
                raise ValueError("unsupported onset path")
            lo = max(0, min(int(r["day"]) for r in g["visits"]))
            if not isinstance(after, int) or abs(after - before) > 730 or not lo <= after <= int(g["T"]):
                raise ValueError("onset outside actual nonnegative observation window")
            entry = _at(data, path.rsplit("/", 1)[0])
            if entry.get("kind") != "drug_symptom":
                raise ValueError("not a registered drug symptom")
            _set_at(data, path, after)
            cause = ae_cause(entry, g["dose_line"], int(g["start"]), drug(g["drug"])["attribution"])
            if cause is None:
                raise ValueError("onset in gray timing band")
            # A resolved episode must not be transformed into an ongoing AE.
            if entry.get("status") == "已缓解" and cause != "predating":
                raise ValueError("resolved predating episode cannot become ongoing AE")
            entry["cause"] = cause
        elif kind == "p3-adherence":
            if path != "adjudication/meds_p3/a_last" or not 0 <= after <= 1 or abs(after - before) > 0.25:
                raise ValueError("adherence outside bounded intervention")
            from haenv.world.labworld.meds import carry_in
            from haenv.world.labworld.meds import coverage_segments
            from haenv.world.labworld.meds import input_steps
            from haenv.world.labworld.meds import truth
            from haenv.world.labworld.meds import response
            _set_at(data, path, after)
            elapsed = int(g["T"]) - int(g["refills"][-1])
            taken = round(after * elapsed)
            if abs(taken / elapsed - after) > 1e-8:
                raise ValueError("adherence not recoverable from integer pill count")
            from haenv.p3.gold import switch_days
            sw = switch_days(g["dose_line"])
            g["pill_count"] = int(round(int(g["supply"]) + carry_in(g["refills"], g["supply"], sw)[-1])) - taken
            seg = coverage_segments(g["refills"], g["a_before"], after, g["supply"], sw)
            steps = input_steps(g["drug"], g["dose_line"], seg)
            for r in g["visits"]:
                if int(r["day"]) < int(g["refills"][-1]):
                    continue  # the new interval cannot alter an earlier observation
                r["truth"] = truth(g["drug"], int(r["day"]), g["params"]["y_u"], g["params"]["r"], steps)
                _reprint(r, table)
            g["params"]["y_T"] = next(r["truth"] for r in g["visits"] if int(r["day"]) == int(g["T"]))
            if g.get("comorbidity"):
                from haenv.p3.realize import _uacr_today
                X = response(steps, int(g["T"]), float(drug(g["drug"])["tau_truth_days"]))
                g["comorbidity"]["uacr"][1] = _uacr_today(g["comorbidity"], drug(g["drug"]), X)
        elif kind == "p3-observation":
            if not re.fullmatch(r"adjudication/meds_p3/visits/\d+/e", path):
                raise ValueError("unsupported observation path")
            sig = float(table["cv_total"])
            if abs(after - before) > 0.25 * sig + 1e-6 or abs(after) > 3 * sig:
                raise ValueError("observation outside bounded noise")
            _set_at(data, path, after)
            _reprint(_at(data, path.rsplit("/", 1)[0]), table)
        else:
            raise ValueError("unknown p3 intervention")
        return
    if pack == "p4":
        from haenv.world.labworld import analyte
        from haenv.world.labworld import cv_total
        from haenv.p4.gold import method_key
        if not re.fullmatch(r"adjudication/followup_p4/records/\d+/(e|perturbations)", path):
            raise ValueError("unsupported p4 record path")
        rec = _at(data, path.rsplit("/", 1)[0])
        if int(rec["day"]) not in (int(g["t0"]), int(g["t1"])):
            raise ValueError("p4 intervention outside compared draws")
        if kind == "p4-observation":
            sig = cv_total(g["analyte"], method_key(rec))
            if path.split("/")[-1] != "e" or abs(after - before) > 0.25 * sig + 1e-6 or abs(after) > 3 * sig:
                raise ValueError("observation outside bounded noise")
        elif kind in ("p4-remove-source", "p4-remove-candidate"):
            if path.split("/")[-1] != "perturbations":
                raise ValueError("unsupported perturbation path")
            removed = [p for p in before if p not in after]
            if len(removed) != 1 or len(before) - len(after) != 1 or any(p not in before for p in after):
                raise ValueError("must remove exactly one existing source, never invent parameters")
            p = removed[0]
            if kind == "p4-remove-source" and (not p.get("acts") or p.get("role") != "cause"):
                raise ValueError("not an existing acting source")
            if kind == "p4-remove-candidate" and (p.get("acts") or p.get("role") not in ("background", "decoy")):
                raise ValueError("candidate must be nonacting")
            if p["factor"] not in ("meat_meal", "delayed_processing", "postprandial_labelled_fasting",
                                    "lab_switch_zero_bias", "hemolysis_note"):
                raise ValueError("source removal would alter measurement method or medication history")
            if kind == "p4-remove-source":
                g["cause"], g["cause_params"] = None, {}
        else:
            raise ValueError("unknown p4 intervention")
        _set_at(data, path, after)
        _reprint(rec, analyte(g["analyte"]))
        return
    raise ValueError(f"unknown family pack {pack}")


def _regold_family(raw, pack):
    if pack == "m2":
        return family_gold(raw, pack)
    block = FAMILY_BLOCKS[pack]
    g = raw.adjudication[block]
    out = family_gold(raw, pack)
    if pack == "p2":
        from haenv.p2.rules import floor_of
        from haenv.p2.rules import table
        tiers = table()["tiers"]
        if tiers[g["tier"]]["rule"] != out["rule"]:
            options = [t for t, cfg in tiers.items() if cfg["rule"] == out["rule"]]
            if len(options) != 1:
                raise ValueError("recomputed rule has no unique metadata tier")
            g["tier"] = options[0]
        g.update(rule=out["rule"], set=out["set"], readings=out["readings"], floor=floor_of(out["set"]),
                 borderline=len(out["set"]) > 1)
        g["plan"].update(tier=g["tier"], scenario=tiers[g["tier"]]["scenario"], floor=g["floor"])
    elif pack == "p3":
        from haenv.p3.gold import STATES
        from haenv.p3.render import render_block
        if out["class"] is None:
            raise ValueError(f"invalid G3: {out['reason']}")
        matches = [s for s, states in STATES.items() if states == set(out["states"])]
        if not matches:
            raise ValueError("G3 active states have no registered subtype")
        sub = g["subtype"] if g["subtype"] in matches else ("top_off" if out["at_max"] and "top_off" in matches else matches[0])
        g.update(g3=out, **{"class": out["class"], "subtype": sub})
        g["plan"].update({"class": out["class"], "subtype": sub})
        if g["params"].get("zone"):
            g["params"]["zone"] = out["control"]
        rendered = render_block(g)
        if g.get("record_format") == "emr":
            from haenv.p3.emr import source_block
            rendered = source_block(rendered, g["current_medications"], g["patient_quote"])
        raw.prediction_context[block] = rendered
    else:
        import math
        from haenv.p4.gold import next_step
        from haenv.p4.gold import pair_rcv
        from haenv.p4.render import report_row
        from haenv.p4.render import diet_rows
        if out["class"] is None:
            raise ValueError(f"invalid G4: {out['reason']}")
        recs = {int(r["day"]): r for r in g["records"]}
        if abs(pair_rcv(g["analyte"], recs[int(g["t0"])], recs[int(g["t1"])]) - g["rcv"]) > 1e-5:
            raise ValueError("source removal changes RCV")
        g.update(g4=out, **{"class": out["class"], "factor": out["factor"],
                           "obs_pct": round(100 * (math.exp(out["dlog_obs"]) - 1), 3)})
        g["next_step"] = next_step(g)[0]
        g["plan"]["class"] = out["class"]
        # The production block captured medication history before the builder
        # pruned the base-world dose stream. It is independent of these edits.
        b = raw.prediction_context[block]
        b["report"] = [report_row(g, r, str(raw.case_id)) for r in sorted(g["records"], key=lambda r: int(r["day"]))]
        b["diet_record"] = [diet_rows(g, recs[int(day)], str(raw.case_id)) for day in (g["t0"], g["t1"])]
    # Class/tier/subtype are dependent descriptions, not a second intervention.
    raw.latent_premise["meta"][_family_slot()][pack] = copy.deepcopy(g["plan"])
    if pack == "p3" and "open_management" in raw.adjudication:
        from haenv.scoring.management.plan import finalize
        finalize(raw, str(raw.case_id), pack)
    return out


def _check_family(raw, pack):
    """The ordinary emission and realism callbacks, on the final solver payload."""
    from haenv_kernel.build import build_instance
    from dataclasses import asdict
    from haenv.app.artifacts.payloads import canonical_T
    from haenv.contracts.external_gold import gates_for
    import haenv.scoring.gates as gates
    T = canonical_T(raw)
    sp, vp = build_instance(raw, T)
    if pack in FAMILY_BLOCKS:
        from importlib import import_module
        from haenv.world.validation.shared_audit import emission_hits
        gate = import_module(f"haenv.{pack}.gate")
        bad = emission_hits(gate.emission(), raw, sp)
        audit = import_module(f"haenv.{pack}.audit").audit()
        item = {"gold": raw.adjudication[FAMILY_BLOCKS[pack]], "sp": asdict(sp), "case": _raw_dict(raw)}
        for name, predicate in audit.realism.items():
            bad += [(name, str(x)) for x in predicate(item)]
        if bad:
            raise ValueError(f"regular {pack} gate/realism: {bad}")
    hits = (gates_for(raw, None, sp, T) + gates.check_ledger_values_traceable(raw, T)
            + gates.check_gold_coverage(sp, raw.gold_drivers, (raw.adjudication or {}).get("ddx")))
    blockers = [h for h in hits if h.get("severity") == "gate"]
    if blockers:
        raise ValueError(f"regular registered gates: {blockers}")
    return asdict(sp), asdict(vp)


def apply_family(raw, details: dict) -> dict:
    """Apply a prepared truth intervention; reject stale sources/gold and undeclared fan-out."""
    d = details
    before = _raw_dict(raw)
    if sha16(before) != d["source_sha16"]:
        raise ValueError("source world differs from prepared family; rebuild and reselect")
    baseline_gold = family_gold(raw, d["pack"])
    if d["pack"] == "m2":
        import haenv.world.wq as wq
        if sha16(wq.injected_manifest(raw.case_id)) != d.get("source_manifest_sha16"):
            raise ValueError("M2 source manifest differs from prepared intervention")
    sp0, vp0 = _check_family(raw, d["pack"])
    _mutate_family(raw, d)
    # A null edit changes no fact: the gold is read, never rewritten (a rewrite would re-round stored values).
    gold = family_gold(raw, d["pack"]) if d["kind"] in NULL_KINDS else _regold_family(raw, d["pack"])
    if d["kind"] in NULL_KINDS and gold != baseline_gold:
        raise ValueError("semantic null changed gold")
    if gold != d["expected_gold"]:
        raise ValueError("expected gold differs from regular recompute")
    if d["pack"] == "m2":
        changing = d["kind"] in (M2_LINE_KIND, M2_CONFIRMATION_KIND)
        if (gold != baseline_gold) != changing:
            raise ValueError("M2 ablation changed gold" if not changing
                             else "M2 line removal left the gold unchanged")
        _check_m2_support(raw.evidence_ledger, d)
        if sha16(wq.injected_manifest(raw.case_id)) != d["source_manifest_sha16"]:
            raise ValueError("M2 source manifest changed during ablation")
    after = _raw_dict(raw)
    fields = tuple(d["declared_fields"])
    allowed = _family_fields(d["pack"], d["kind"], d["path"], d.get("operation_details"))
    if any(f not in allowed for f in fields):
        raise ValueError("undeclared or broadened dependency declaration")
    changes = _diff(before, after)
    bad = [p for p in changes if not _covered(p, fields)]
    if bad:
        raise ValueError(f"undeclared change: {bad}")
    if "trial_sha16" in d and sha16(after) != d["trial_sha16"]:
        raise ValueError("prepared world differs from reproduced intervention")
    sp, vp = _check_family(raw, d["pack"])
    acquired_change = False
    if d["kind"] == M2_CONFIRMATION_KIND:
        from haenv.m2.confirmation import plan_of
        from haenv.m2.confirmation import rules
        from haenv.interaction.gated import synth_menu_target
        p = plan_of(raw)
        test = rules()["profiles"][p["profile"]]["test"]
        now = synth_menu_target(raw, test, int(raw.prediction_context["prediction_time_T"]))
        acquired_change = bool(now) and now != d["operation_details"]["source_result"]
    if not changes or (sp == sp0 and not acquired_change):
        raise ValueError("empty twin: no solver-visible change")
    # Payload isolation must not produce additional changes outside dependencies.
    for label, a, b in (("sp", sp0, sp), ("vp", vp0, vp)):
        bad = [p for p in _diff(a, b) if not _covered(p, fields)]
        if bad:
            raise ValueError(f"undeclared {label} change: {bad}")
    return {"gold_changed": _scored_gold(gold, d["pack"]) != _scored_gold(baseline_gold, d["pack"]),
            "gold_sha16": sha16(gold), "changed_fields": changes, "world_sha16": sha16(after)}


def check_persisted_twin(orig_raw, twin_raw, d) -> list[str]:
    """A persisted twin must equal its declaration re-applied to the persisted original, carry the expected
    gold, and differ from the original only at declared fields."""
    from dataclasses import asdict
    cid, problems = d["case_id"], []
    bad = [p for p in _diff(asdict(orig_raw), asdict(twin_raw)) if not _covered(p, d["declared_fields"])]
    if bad:
        problems.append(f"{cid}: undeclared world change {bad[:5]}")
    try:
        expected = copy.deepcopy(orig_raw)
        apply_family(expected, d)
        if asdict(expected) != asdict(twin_raw):
            problems.append(f"{cid}: rebuilt twin differs from prepared world")
        if family_gold(twin_raw, d["pack"]) != d["expected_gold"]:
            problems.append(f"{cid}: persisted gold differs from expected recompute")
    except ValueError as exc:
        problems.append(f"{cid}: {exc}")
    return problems


def _check_m2_support(ledger, details):
    """Require retained actual observations for every declared diagnostic line."""
    visible = {str(e.get("evidence_id")) for e in ledger}
    support = details.get("support_by_line") or {}
    if not support:
        raise ValueError("M2 support has no traceable diagnostic line")
    for line, ids in support.items():
        if len(visible & set(ids)) < 2:
            raise ValueError(f"M2 line {line}: fewer than two retained source observations")
    if set(details.get("protected_clue_ids") or ()) - visible:
        raise ValueError("M2 removal deletes a line's actual discriminating clue")


def _m2_source_support(raw, row, job_case, pay):
    """Trace line support from source indices/declared clues, never semantic guessing."""
    from haenv.m2 import core
    ddx = raw.adjudication.get("ddx") or {}
    threads = core.threads_of(ddx)
    if not threads:
        raise ValueError("M2 source has no sufficient hidden diagnostic line")
    man = (row.get("question") or {}).get("injected_manifest") or {}
    sp = pay["sp"]
    visible = {str(e.get("evidence_id")): e for e in sp.get("evidence_ledger") or ()}
    real = set(man.get("real_symptom_evidence_ids") or ()) & set(visible)
    comps = list(dict.fromkeys(t["component"] for t in threads))
    if len(comps) != len(threads):
        raise ValueError("M2 multiple lines share a component without explicit line-level symptom ownership")
    ea = ((job_case.get("latent") or {}).get("m2") or {}).get("explain_away") or {}
    clues = list(ea.get("clues") or ()) + [{"component": ln["component"], **ln["clue"]}
                                          for ln in ea.get("lines") or () if ln.get("clue")]
    source_symptoms = (job_case.get("raw") or {}).get("symptoms") or ()
    planning = {v: k for k, v in (man.get("id_map") or {}).items()}
    support = {c: [] for c in comps}
    protected = []
    from haenv.world.observations.events import symptom_lay_every
    ways = symptom_lay_every()
    norm = lambda t: str(t).replace("，", ",").replace("。", ",").strip().rstrip("、,")
    for eid in sorted(real):
        e = visible[eid]
        # Explicit per-line clue provenance and the production wording catalogue
        # identify the actual rendered clue, not an inferred disease name.
        hit = [c for c in clues if c.get("component") in comps and int(c.get("day", -1)) == int(e["source_timestamp"])
               and norm(e.get("symptom")) in {norm(t) for t in [c["text"], *ways.get(c["text"], ())]}]
        if hit:
            protected.append(eid)
            owners = {c["component"] for c in hit}
        elif len(comps) == 1:
            owners = {comps[0]}
        else:
            m = re.search(r"-S(\d+)$", planning.get(eid, ""))
            if not m or not 1 <= int(m[1]) <= len(source_symptoms):
                raise ValueError("M2 per-line observation has no explicit source index")
            text = str(source_symptoms[int(m[1]) - 1].get("text") or "")
            owners = {c for c in comps if core._norm(text) in {core._norm(t) for t in core.component_templates(c)}}
            # Explained-line provenance supplies indices when rewriting changed
            # the source wording; ambiguous templates cannot prove ownership.
            owners.update(ln["component"] for ln in ea.get("lines") or ()
                          if any((s.get("index") is not None and int(s["index"]) == int(m[1]) - 1)
                                 or (s.get("day") is not None and int(s["day"]) == int(e["source_timestamp"]))
                                 for s in ln.get("symptoms") or ()))
        if len(owners) != 1:
            raise ValueError("M2 per-line observation ownership is absent or ambiguous")
        support[next(iter(owners))].append(eid)
    details = {"support_by_line": support, "protected_clue_ids": sorted(set(protected)),
               "source_manifest_sha16": sha16(man)}
    _check_m2_support(list(visible.values()), details)
    return details


# ------------------------------------------------------------------ M2 gold-changing line removal
# A composed comorbidity question is one patient generated from the first component plus the second
# component's symptom text (`overlay.haenv_comorbid_specs`). Removing the second line is therefore a
# truth edit with a closed dependency set: its owned symptom observations leave the ledger, the gold
# becomes the registry spec of the kept component, and the frame gold and F1 scene are recomputed by
# the production functions. The weight course, streams and every other record stay byte-identical.
M2_LINE_KIND = "m2-remove-line"
M2_CONFIRMATION_KIND = "m2-confirmation"


def _m2_line_fields(footprint_streams=()):
    from haenv.contracts.external_gold import SLOT
    return ("evidence_ledger", "adjudication/ddx", "adjudication/m2_frame", "adjudication/m2_workup",
            "adjudication/open_management", "adjudication/red_flag_present",
            "adjudication/clinician_action_warranted", "latent_premise/meta/ddx", "latent_premise/meta/red_flag",
            "latent_premise/meta/clinician_warranted", "latent_premise/meta/outcome_label",
            f"latent_premise/meta/{SLOT}/m2/scene", "prediction_context/m2_frame/scene",
            "prediction_context/m2_frame/patient_quote", "prediction_context/m2_frame/encounter_record",
            "prediction_context/archived_laboratory_record",
            # a removed line's stream footprint: the readings, and their post-T copy in the verifier view
            *(f"{part}/{st}/*/value" for st in footprint_streams for part in ("longitudinal_data", "future_data")))


def m2_ddx_of_spec(spec_id: str, T: int) -> tuple[dict, dict]:
    """(`meta.ddx`, meta flags) a diagnosis job derives from one registry spec (`ddx.ddx_case_specs`
    latent keys through `job.CaseSpec.ddx`). Verified against the source before it is trusted."""
    from haenv.world.clinical.ddx import insufficient_triage
    from haenv.world.overlay import condition_registry
    from haenv.world.overlay import threads_for
    spec = condition_registry(include_draft=True).get(str(spec_id))
    if not spec:
        raise ValueError(f"M2 line removal: {spec_id} is not a registry spec")
    rf, urg = insufficient_triage(spec, [], int(T), False)
    ddx = {"spec_id": str(spec_id), "diagnosis": spec["diagnosis"], "aliases": list(spec.get("aliases") or []),
           "join_gold": spec["join_gold"],
           "threads": [dict(t) for t in (threads_for(spec_id, spec) or ())] or None,
           "tests": spec["tests"], "specialty": spec["specialty"], "urgency": urg}
    flags = {"red_flag": bool(rf), "clinician_warranted": bool(spec.get("clinician_warranted", True)),
             "outcome_label": spec.get("outcome_label")}
    return json.loads(json.dumps(ddx, ensure_ascii=False)), flags


def _m2_line_select(raw, row, job_case, pay, support):
    """The single eligible line removal of a composed comorbidity question, or a rejection."""
    from haenv.m2 import core
    ddx = raw.adjudication.get("ddx") or {}
    comps = core.components(str(ddx.get("spec_id") or ""))
    if ddx.get("join_gold") != "comorbidity" or len(comps) != 2 or comps[0] == comps[1]:
        raise ValueError("M2 line removal needs a composed two-component comorbidity")
    keep, drop = comps
    man = (row.get("question") or {}).get("injected_manifest") or {}
    planning = {v: k for k, v in (man.get("id_map") or {}).items()}
    by_line = support["support_by_line"]
    removed = sorted(by_line.get(drop) or ())
    indices = []
    for eid in removed:
        m = re.search(r"-S(\d+)$", planning.get(eid, ""))
        if not m:
            raise ValueError("M2 line removal: removed observation has no source symptom index")
        indices.append(int(m[1]) - 1)
    feet = _m2_footprints(raw, removed)
    stated = [re.search(r"-S(\d+)$", planning.get(e, "")) for ids in by_line.values() for e in ids]
    n_src = len((job_case.get("raw") or {}).get("symptoms") or ())
    if sorted(int(m[1]) - 1 for m in stated if m) != list(range(n_src)):
        raise ValueError("M2 line removal: a source symptom has no single traceable observation")
    extra = {"footprint_streams": feet, "removed_planning_ids": sorted(planning[e] for e in removed),
             **({"job_latent": copy.deepcopy(job_case["latent"])} if feet else {})}
    return keep, {**extra, "remove_component": drop, "keep_component": keep, "removed_evidence_ids": removed,
                  "removed_symptom_indices": sorted(indices), "job_raw": copy.deepcopy(job_case["raw"]),
                  "support_by_line": {keep: sorted(by_line[keep])},
                  "protected_clue_ids": sorted(set(support["protected_clue_ids"]) & set(by_line[keep])),
                  "dropped_clue_ids": sorted(set(support["protected_clue_ids"]) - set(by_line[keep]))}


def _m2_footprints(raw, removed_ids):
    """Indicator streams in this world on which a removed observation's topic leaves a kernel footprint."""
    import haenv.world.wq as wq
    from haenv.world.physio import kernel as _pk
    from haenv.resources.regpath import registry_path
    kt = _pk.load_physio_registry(registry_path("physio_kernels.yaml"))
    topics = {str(e.get("topic")) for e in wq.injected_manifest(raw.case_id).get("event_schedule") or ()
              if str(e.get("evidence_id")) in set(removed_ids)}
    return sorted({ind for (_t, ind), p in kt.kernels.items()
                   if p.is_emitted and p.generator_source in topics and (raw.longitudinal_data or {}).get(ind)})


_REBUILD_LOCK = __import__("threading").Lock()


def _m2_rebuild(case_id, od, drop):
    """The ordinary `build_case` of this question with the physiology layer's event schedule minus the
    planning IDs in `drop`. The question registry is restored afterwards; the source world is not touched."""
    import haenv.world.observations.events as events
    import haenv.world.wq as wq
    from haenv.world.generation.build import build_case
    from haenv.app.job import CaseSpec
    if "job_latent" not in od:
        raise ValueError("M2 line removal: a footprint removal needs the source job case")
    drop = {str(x) for x in drop}
    cs = CaseSpec(case_id=str(case_id), raw=copy.deepcopy(od["job_raw"]), latent=copy.deepcopy(od["job_latent"]))
    with _REBUILD_LOCK:
        orig = events._apply_physio_if_enabled
        saved = (dict(wq._QINJ), dict(wq._QINJ_META), dict(events.PANEL_ALIGN_PENDING))

        def filtered(r, cid, schedule):
            return orig(r, cid, [e for e in schedule if str(e.get("evidence_id")) not in drop])
        events._apply_physio_if_enabled = filtered
        try:
            built, _ = build_case(cs)
        finally:
            events._apply_physio_if_enabled = orig
            for table, old in zip((wq._QINJ, wq._QINJ_META, events.PANEL_ALIGN_PENDING), saved):
                table.clear()
                table.update(old)
    if built is None:
        raise ValueError("M2 line removal: the question does not rebuild")
    return built


def _m2_footprint_values(raw, od, feet, drop):
    """The footprint streams without the removed line's events. The world must rebuild exactly, and the
    rebuild without those events may differ from it only at values of the footprint streams."""
    from dataclasses import asdict
    src = asdict(raw)
    if asdict(_m2_rebuild(raw.case_id, od, ())) != src:
        raise ValueError("M2 line removal: the rebuild does not reproduce the source world")
    cut = _m2_rebuild(raw.case_id, od, drop)
    # The private rubric snapshot is a derived copy of these same observations.
    # Verify both copies before excluding them from the physical footprint diff.
    from haenv.scoring.management.plan import finalize
    for candidate in (raw, cut):
        if "open_management" in candidate.adjudication:
            check = copy.deepcopy(candidate)
            finalize(check, str(check.case_id), "m2", derive_workup=False)
            if check.adjudication["open_management"] != candidate.adjudication["open_management"]:
                raise ValueError("M2 line removal: stored management facts do not reproduce")
    cut_data = asdict(cut)
    src["adjudication"].pop("open_management", None)
    cut_data["adjudication"].pop("open_management", None)
    own = re.compile(r"longitudinal_data/(%s)/\d+/value" % "|".join(map(re.escape, feet)))
    bad = [p for p in _diff(src, cut_data) if not own.fullmatch(p.lstrip("/"))]
    if bad:
        raise ValueError(f"M2 line removal: the footprint is not separable: {bad[:5]}")
    return {st: copy.deepcopy(cut.longitudinal_data[st]) for st in feet}


def _m2_remove_line(raw, d):
    from haenv.contracts.external_gold import SLOT
    from haenv.m2 import core
    od = d["operation_details"]
    ddx = (raw.adjudication or {}).get("ddx") or {}
    comps = core.components(str(ddx.get("spec_id") or ""))
    if ddx.get("join_gold") != "comorbidity" or len(comps) != 2 or comps[0] == comps[1]:
        raise ValueError("M2 line removal needs a composed two-component comorbidity")
    # The world (weight course, streams) follows the first component; only the second is removable.
    if [d["after"], od.get("remove_component")] != comps or d["before"] != ddx.get("spec_id"):
        raise ValueError("M2 line removal must keep the world-generating first component")
    meta = raw.latent_premise["meta"]
    lat = core.m2_latent(meta)
    # The removed line may own discriminating clues (they leave with its observations), but it must
    # not explain, or be an explained line of, the kept line: that would change what the kept
    # observations mean. Acute-event scenes are drawn per spec and are excluded.
    drop, ea = od.get("remove_component"), lat.get("explain_away") or {}
    if (lat.get("acute_event") or (lat.get("prior") or {}).get("k_ea") == "by_visible_line"
            or any(ln.get("component") != d["after"] or any(drop in (s.get("explainer") or ()) for s in ln.get("symptoms") or ())
                   for ln in ea.get("lines") or ())):
        raise ValueError("M2 line removal: the removed line takes part in an explain-away structure or acute event")
    T = int(raw.prediction_context["prediction_time_T"])
    # The derivation must reproduce the stored pair gold before it may write the kept one.
    src, src_flags = m2_ddx_of_spec(ddx["spec_id"], T)
    if src != json.loads(json.dumps(meta.get("ddx"), ensure_ascii=False)) or any(
            meta.get(k) != v for k, v in src_flags.items()):
        raise ValueError("M2 line removal: registry derivation does not reproduce the stored gold")
    removed = set(od["removed_evidence_ids"])
    # Ownership is re-derived from the source indices, not trusted from the declaration:
    # every observation of the removed line goes, and nothing else.
    from dataclasses import asdict
    from haenv_kernel.build import build_instance as _bi
    import haenv.world.wq as wq
    owned = _m2_source_support(raw, {"question": {"injected_manifest": wq.injected_manifest(raw.case_id)}},
                               {"raw": od["job_raw"], "latent": {"m2": lat}},
                               {"sp": asdict(_bi(raw, T)[0])})["support_by_line"]
    if removed != set(owned.get(od["remove_component"]) or ()) or not owned.get(d["after"]):
        raise ValueError("M2 line removal: declared observations differ from the removed line's actual support")
    led = list(raw.evidence_ledger or [])
    if not removed or len([e for e in led if str(e.get("evidence_id")) in removed]) != len(removed):
        raise ValueError("M2 line removal: declared observations are not all in the ledger")
    feet = _m2_footprints(raw, removed)
    if feet != sorted(od.get("footprint_streams") or ()):
        raise ValueError(f"M2 line removal: declared footprint streams {od.get('footprint_streams')} differ "
                         f"from the removed observations' {feet}")
    planning = {v: k for k, v in (wq.injected_manifest(raw.case_id).get("id_map") or {}).items()}
    if any(e not in planning for e in removed):
        raise ValueError("M2 line removal: a removed observation has no planning ID")
    foot = _m2_footprint_values(raw, od, feet, {planning[e] for e in removed}) if feet else {}
    plan = lat.get("f3")
    if plan and plan.get("arm") == "exceeds" and (plan["stream"], int(plan["sign"])) not in \
            core.exceed_streams(d["after"], [plan["stream"]]):
        raise ValueError("M2 line removal: the F3 change would no longer be explained by the kept line")
    job_raw = od["job_raw"]
    frame = lat.get("frame")
    if frame and core.scene_text(frame, job_raw, {**lat, "index_time_T": T}, raw.case_id) != lat.get("scene"):
        raise ValueError("M2 line removal: the production scene does not reproduce the stored scene")
    if frame:
        # The plan text stays latent; observe synchronizes only the visible frame
        # and medicine table with the realized dose course at T. Reproduce that
        # view on a copy, so a damaged source is rejected rather than repaired.
        source_view = copy.deepcopy(raw)
        source_view.prediction_context["m2_frame"] = core.frame_probe(source_view.latent_premise["meta"])
        core.sync_current_medications(source_view)
        from haenv.m2.emr import finalize as finalize_emr
        # The patient-independent document was moved from the old root field.
        source_view.prediction_context["m2_frame"]["clinic_documents"] = copy.deepcopy(
            (raw.prediction_context.get("m2_frame") or {}).get("clinic_documents") or [])
        finalize_emr(source_view)
        if any(source_view.prediction_context.get(k) != raw.prediction_context.get(k)
               for k in ("m2_frame", "current_medications")):
            raise ValueError("M2 line removal: the observed scene or medication table does not reproduce the stored view")
    new, flags = m2_ddx_of_spec(d["after"], T)
    raw.evidence_ledger = [e for e in led if str(e.get("evidence_id")) not in removed]
    for st, pts in foot.items():
        raw.longitudinal_data[st] = pts
    meta["ddx"] = new
    meta.update(flags)
    if frame:
        kept = {**job_raw, "symptoms": [s for i, s in enumerate(job_raw.get("symptoms") or ())
                                        if i not in set(od["removed_symptom_indices"])]}
        meta[SLOT]["m2"]["scene"] = core.scene_text(frame, kept, {**lat, "index_time_T": T}, raw.case_id)
        raw.prediction_context["m2_frame"] = core.frame_probe(meta)
        core.sync_current_medications(raw)
    adj = raw.adjudication
    adj["red_flag_present"] = flags["red_flag"]
    adj["clinician_action_warranted"] = flags["clinician_warranted"]
    adj["ddx"] = copy.deepcopy(new)
    if "m2_frame" in adj:
        diversity = copy.deepcopy(adj["m2_frame"].get("diversity"))
        adj["m2_frame"] = core.frame_gold(meta)
        if diversity is not None:
            # Existing incidental encounters are observations, not a fresh draw
            # when a diagnostic line is removed in a truth-family intervention.
            adj["m2_frame"]["diversity"] = diversity
    reproduced = _m2_recomputed(raw, derive_workup=True)
    raw.adjudication = reproduced.adjudication
    raw.prediction_context = reproduced.prediction_context
    from haenv_kernel.build import build_instance
    import haenv.world.clinical.qualifiers as qualifiers
    qualifiers.annotate(raw, build_instance(raw, T)[0])
    if "open_management" in raw.adjudication:
        from haenv.scoring.management.plan import finalize
        finalize(raw, str(raw.case_id), "m2", derive_workup=False)


def _primitive_candidates(raw, pack, row=None, job_case=None, pay=None):
    """Finite, declared intervention inventory; no search for invented clinical sources."""
    if pack == "m2":
        if row is None or job_case is None or pay is None:
            raise ValueError("M2 selection needs source job, case and payload rows")
        from haenv.m2.confirmation import plan_of
        from haenv.m2.confirmation import rules
        confirmation = plan_of(raw)
        if confirmation:
            from haenv.interaction.gated import synth_menu_target
            from haenv.contracts.external_gold import SLOT
            test = rules()["profiles"][confirmation["profile"]]["test"]
            point = synth_menu_target(raw, test, int(raw.prediction_context["prediction_time_T"]))
            yield M2_CONFIRMATION_KIND, f"latent_premise/meta/{SLOT}/m2/confirmation/present", not confirmation["present"], {"source_result": point}
            return
        for name in ("unname-explainer", "drop-planted-symptoms"):
            details = OPS[name].select(job_case, row, pay)
            if details:
                if name == "drop-planted-symptoms":
                    # One existing observation is the intervention. Removing all
                    # support is tested but cannot bypass GEN14's evidence floor.
                    for eid in details["evidence_ids"]:
                        if any(str(e.get("evidence_id")) == eid for e in raw.evidence_ledger):
                            yield name, "evidence_ledger", None, {"evidence_ids": [eid], "n_removed": 1}
                else:
                    for condition in details["unname"]:
                        remaining = details["stays"] + [k for k in details["unname"] if k != condition]
                        one = {**details, "unname": [condition], "stays": remaining,
                               "stratum": "known_plus_line" if remaining else "known_only"}
                        yield name, "user_profile/known_conditions", None, one
        try:
            keep, details = _m2_line_select(raw, row, job_case, pay, _m2_source_support(raw, row, job_case, pay))
        except ValueError as exc:
            yield M2_LINE_KIND, "adjudication/ddx/spec_id", None, {"_rejected": str(exc)}
        else:
            yield M2_LINE_KIND, "adjudication/ddx/spec_id", keep, details
        return
    kc = list((raw.user_profile or {}).get("known_conditions") or ())
    if len(kc) >= 2 and kc[0] != kc[1]:
        yield NULL_KIND, NULL_PATH, [kc[1], kc[0], *kc[2:]], None
    if pack == "p4":
        swap = same_day_medication_swap(raw.prediction_context[FAMILY_BLOCKS[pack]]["medication_record"])
        if swap is not None:
            yield NULL_MED_KIND, NULL_MED_PATH, swap[0], swap[1]
    block = FAMILY_BLOCKS[pack]
    g = raw.adjudication[block]
    if pack == "p2":
        b = raw.prediction_context[block]
        for f, step in (("temperature_c", 0.5), ("respiratory_rate_per_min", 2), ("spo2_pct", 2),
                        ("heart_rate_bpm", 5), ("systolic_bp_mmHg", 5), ("diastolic_bp_mmHg", 5)):
            for sign in (-1, 1):
                yield "p2-measurement", f"prediction_context/{block}/home_check_today/{f}", round(b["home_check_today"][f] + sign * step, 1), None
        for name, field, step in (("morning_weight", "kg", 0.3), ("wearable_daily", "irregular_rhythm_alerts", 1),
                                  ("wearable_daily", "night_spo2_min_pct", 1), ("cgm_nightly", "min_mmol_L", 0.2)):
            for i in range(max(0, len(b[name]) - 4), len(b[name])):
                for sign in (-1, 1):
                    yield "p2-measurement", f"prediction_context/{block}/{name}/{i}/{field}", round(b[name][i][field] + sign * step, 1), None
        return
    if pack == "p3":
        from haenv.world.labworld.meds import axis
        from haenv.world.labworld.meds import carry_in
        from haenv.world.labworld.meds import rule
        from haenv.p3.gold import switch_days
        for i, e in enumerate(g["diary"]):
            if e.get("kind") == "drug_symptom":
                starts = (int(e["onset"]) - 1, int(e["onset"]) + 1,
                          int(g["start"]) - int(rule()["start_gray_days"]) - 1,
                          max(int(d) for d, v in g["dose_line"]) + int(rule()["ae_unattributed_min"]) + 1)
                # Each displacement also in the opposite direction, so a same-size partner can be paired with it.
                mirrors = tuple(2 * int(e["onset"]) - day for day in starts)
                for day in dict.fromkeys(starts + mirrors):
                    yield "p3-onset", f"adjudication/{block}/diary/{i}/onset", day, None
        elapsed = int(g["T"]) - int(g["refills"][-1])
        for delta in (-1, 1, -max(1, round(0.2 * elapsed)), max(1, round(0.2 * elapsed))):
            taken = (int(round(int(g["supply"]) + carry_in(g["refills"], g["supply"], switch_days(g["dose_line"]))[-1]))
                     - int(g["pill_count"]) + delta)
            if elapsed and 0 <= taken <= elapsed:
                yield "p3-adherence", f"adjudication/{block}/a_last", taken / elapsed, None
        sig = float(axis(g["axis"])["cv_total"])
        records = g["visits"]
        name = "visits"
    else:
        from haenv.world.labworld import cv_total
        from haenv.p4.gold import method_key
        records = g["records"]
        name = "records"
        for i, r in enumerate(records):
            if int(r["day"]) not in (int(g["t0"]), int(g["t1"])):
                continue
            for j, p in enumerate(r["perturbations"]):
                kind = "p4-remove-source" if p.get("acts") else "p4-remove-candidate"
                yield kind, f"adjudication/{block}/records/{i}/perturbations", r["perturbations"][:j] + r["perturbations"][j + 1:], None
    for i, r in enumerate(records):
        if int(r["day"]) != int(g["T"] if pack == "p3" else g["t1"]):
            continue
        if pack == "p4":
            sig = cv_total(g["analyte"], method_key(r))
        for sign in (-1, 1):
            yield f"{pack}-observation", f"adjudication/{block}/{name}/{i}/e", round(r["e"] + sign * 0.25 * sig, 6), None


def family_candidates(raw, pack: str, *, row=None, job_case=None, pay=None, rejected=None) -> list[dict]:
    """All valid nonempty trials for this actual question; optionally collect exact rejections."""
    before = _raw_dict(raw)
    out = []
    support = {}
    if pack == "m2":
        try:
            from haenv.m2.confirmation import plan_of
            if plan_of(raw):
                import haenv.world.wq as wq
                ids = [str(e["evidence_id"]) for e in raw.evidence_ledger
                       if e.get("source_type") == "patient_reported_symptom"]
                support = {"source_manifest_sha16": sha16(wq.injected_manifest(raw.case_id)),
                           "support_by_line": {"presentation": ids}, "protected_clue_ids": []}
            else:
                support = _m2_source_support(raw, row, job_case, pay)
        except ValueError as exc:
            if rejected is not None:
                rejected.append({"case_id": raw.case_id, "kind": "m2-source-support", "path": "evidence_ledger", "reason": str(exc)})
            return []
    for kind, path, value, extra in _primitive_candidates(raw, pack, row, job_case, pay):
        fact = _at(before, path)
        d = {"case_id": raw.case_id, "pack": pack, "kind": kind, "path": path,
             "before": fact, "after": value, "source_sha16": sha16(before),
             "declared_fields": list(_family_fields(pack, kind, path, extra))}
        if kind == NULL_MED_KIND:
            d["operation_details"] = copy.deepcopy(extra)
        if pack == "m2":
            if kind == M2_CONFIRMATION_KIND:
                d.update(operation_details=extra, **support)
            elif kind == M2_LINE_KIND:
                if "_rejected" in extra:
                    if rejected is not None:
                        rejected.append({"case_id": raw.case_id, "kind": kind, "path": path, "reason": extra["_rejected"]})
                    continue
                d.update(operation_details=extra, after=value, source_manifest_sha16=support["source_manifest_sha16"],
                         support_by_line=extra["support_by_line"], protected_clue_ids=extra["protected_clue_ids"])
            else:
                d.update(operation_details=extra, after="remove-declared-evidence", **support)
        try:
            twin = copy.deepcopy(raw)
            _mutate_family(twin, d)
            d["expected_gold"] = family_gold(twin, pack) if kind in NULL_KINDS else _regold_family(twin, pack)
            record = apply_family(copy.deepcopy(raw), d)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            if rejected is not None:
                rejected.append({"case_id": raw.case_id, "kind": kind, "path": path, "reason": str(exc)})
            continue
        d.update(gold_changed=record["gold_changed"], trial_sha16=sha16(_raw_dict(twin)))
        out.append(d)
    return out


OPS["truth-family"] = TwinOp("truth-family", "families", "one declared truth fact and its deterministic dependencies",
                              (), lambda *_: None, apply_family)
