"""M2 evidence derivability, emitted gold lines, and report-only key-signal reader."""
from collections import Counter, defaultdict
from contextlib import contextmanager
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from haenv.i18n.identity import source
from haenv_kernel.schema import RawCase
from haenv.m2 import core, evidence, premises as production_premises

from .batch import Item, Operator
from .diffs import diff, record_keys
from .registry import premises, check_complete
from .m2_ops import twins as candidates, abnormal

_CONTEXTS = {}


def canon(raw):
    """Canonicalise both languages before any rule or emission check reads them."""
    return RawCase(**source(asdict(raw)))


def stored_sp(sp):
    return SimpleNamespace(case_id=sp.get("case_id"), evidence_ledger=sp.get("evidence_ledger") or [],
                           longitudinal_data=sp.get("longitudinal_data") or {},
                           prediction_context=sp.get("prediction_context") or {}, user_profile=sp.get("user_profile") or {})


def load_batch(batch_dir):
    """Load the job once; fork workers only after this context has been populated."""
    from haenv.app.job import load_job
    from haenv.locations import output_root
    from .batch import load_batch as stored_items
    batch = Path(batch_dir)
    job = load_job(batch.parents[3] / "m2-pack1.job.yaml", root=output_root())
    cases = {cs.case_id: cs for cs in job.cases}
    rows = {r["case_id"]: r for r in map(json.loads, (batch / "cases.jsonl").read_text().splitlines())}
    audits = {r["case_id"]: r for r in map(json.loads, (batch / "audit.jsonl").read_text().splitlines())}
    items = stored_items(batch, _m2_loaded=True)
    for item in items:
        _CONTEXTS[item.case_id] = dict(cs=cases[item.case_id],
            manifest=(rows[item.case_id].get("question") or {}).get("injected_manifest") or {},
            sources=audits[item.case_id].get("symptom_source") or {})
    return [Item(i.case_id, canon(i.raw), i.sp, i.language) for i in items]


@contextmanager
def on_demand_override(override=None):
    """Temporarily wrap the production synthesis hook, restoring it even on error."""
    original = core.m2_synth_on_demand
    def patched(orig, raw, target, T):
        points = original(orig, raw, target, T)
        if not override or not points:
            return points
        out = []
        for point in points:
            edit = override.get(point.get("finding_id")) or override.get(str(target))
            out.append(edit(copy.deepcopy(dict(point))) if edit else point)
        return out
    core.m2_synth_on_demand = patched
    try:
        yield
    finally:
        core.m2_synth_on_demand = original


def line_key(thread):
    """Component plus key ids distinguishes two lines sharing a component."""
    return thread["component"] + "[" + ",".join(sorted(thread["key_signals"])) + "]"


def direction(component, key):
    fid = key.removeprefix("archive:")
    rows = (core._profiles().get(component) or {}).get("findings") or []
    return next((r.get("direction", "high") for r in rows if r.get("id") == fid), "high")


def key_signal_reader(raw, T, *, threads=None, on_demand=None):
    """Components with a complete non-exclusion evidence path on visible purchases.

    Archived keys read the archived observation, never a fresh on-demand result.
    The optional callable makes this helper usable without loading a batch.
    """
    threads = core.threads_of(raw.adjudication.get("ddx") or {}) if threads is None else threads
    read = on_demand or core._on_demand
    out = set()
    for thread in threads:
        def positive(key):
            fid = key.removeprefix("archive:")
            if key.startswith("archive:"):
                return any(abnormal(row.get("observation")) for row in
                           raw.prediction_context.get("archived_laboratory_record") or [] if row.get("finding") == fid)
            declared = direction(thread["component"], key)
            if declared in ("normal", "negative"):
                return True  # exclusion members do not establish the line
            points = read(raw, fid, T)
            return core._reads_direction(points, declared)
        if any(path and all(positive(k) for k in path) for path in evidence.paths_of(thread)):
            out.add(thread["component"])
    return out


def outcome(refusals, insufficient=False):
    """M2-only vocabulary: an emptied report artefact is not a successful refusal."""
    if refusals.get("RAISE"):
        return "error"
    if refusals.get("unbuy"):
        return "invalid"
    if any(refusals.values()):
        return "refused"
    return "undetermined" if insufficient else "same"


def context(raw):
    from haenv.app.evaluate import gated_menu
    raw = canon(raw)
    cid = raw.case_id
    T = int(raw.prediction_context["prediction_time_T"])
    ddx = raw.adjudication.get("ddx") or {}
    loaded = _CONTEXTS.get(cid, {})
    if not loaded:
        import haenv.world.wq as wq
        loaded = dict(manifest=wq.injected_manifest(cid, required=False) or {}, sources={})
    reach = set()
    for menu in gated_menu(raw, T):
        if menu.get("is_test"):
            reach.update(core.m2_menu_findings(menu["target"], core._findings()))
    return dict(cid=cid, raw=raw, T=T, ddx=ddx, th=core.threads_of(ddx), reach=reach,
                frame=(raw.adjudication.get("m2_frame") or {}).get("frame") or "F0", **loaded)


def symptom_owners(c):
    real = set(c["manifest"].get("real_symptom_evidence_ids") or [])
    visible = [e for e in c["raw"].evidence_ledger if e.get("evidence_id") in real
               and int(e.get("source_timestamp", 10**9)) <= c["T"]]
    owners = {}
    for e in visible:
        eid = str(e["evidence_id"])
        if len(c["th"]) == 1:
            owners[eid] = line_key(c["th"][0])
        else:
            comp = core.symptom_owner(c["sources"].get(eid, str(e.get("symptom"))),
                                      [t["component"] for t in c["th"]])
            owners[eid] = next((line_key(t) for t in c["th"] if t["component"] == comp), None)
    return owners


def emitted_roles(c):
    """Instantiate the production gold's evidence and requirement declarations."""
    roles = {}
    owners = symptom_owners(c)
    def add(pid, base, comp, observed=True, key=None, decisive=True):
        role = ("acting" if decisive else "overridden") if observed else "context"
        roles[pid] = dict(premise=base, component=comp, key=key, role=role,
                          observed=observed, cf_decision=None, texts=[])
    for t in c["th"]:
        # The production evidence paths decide what supports a line: a key is decisive only when every
        # path that establishes the line needs it; with an alternative complete path it is overridden.
        paths = [list(path) for path in evidence.paths_of(t) if path]
        for key in t["key_signals"]:
            base = "m2.archive_key" if key.startswith("archive:") else "m2.key_signal"
            observed = key.startswith("archive:") or abnormal(core._on_demand(c["raw"], key, c["T"]))
            add(f"{base}:{t['component']}:{key}", base, t["component"], observed, key,
                decisive=not paths or all(key in path for path in paths))
        lk = line_key(t)
        if lk in owners.values():
            add("m2.line_symptoms:" + lk, "m2.line_symptoms", t["component"])
    if owners:
        add("m2.all_symptoms", "m2.all_symptoms", None)
    for comp, requirement in production_premises.requirements(c["ddx"]):
        kind = {"glycaemic_diagnosis": "glycaemic"}.get(requirement["kind"], requirement["kind"])
        base = "m2.premise." + kind
        add(base + ":" + comp, base, comp)
    return roles


def checks(c, raw):
    """The investigation's production emission battery, returning check ids only."""
    from haenv_kernel.build import build_instance
    from haenv.m2.item_audit import check_menu1, check_ea2
    from haenv.world.validation import gates_outcome as GO
    from haenv.world.validation.gates_case import check_ledger_values_traceable
    from haenv.world.clinical.ddx import check_visible_tier, insufficient_from_visible
    from haenv.contracts.external_gold import gates_for
    from haenv.m2.history import archive_keys
    T, ddx, cs, cid = c["T"], c["ddx"], c["cs"], c["cid"]
    raw = canon(raw)
    sp, _ = build_instance(raw, T)
    h = {}
    h["prem"] = [x["detail"] for x in production_premises.emission(raw, cs, sp, T)]
    h["cov"] = [x["kind"] for x in GO.check_gold_coverage(sp, raw.gold_drivers, ddx)]
    h["vt"] = [x["kind"] for x in check_visible_tier(sp, cs.latent, T)]
    reach = c["reach"] | archive_keys(pc=sp.prediction_context or {}, ddx=ddx)
    h["menu1"] = [str(v) for v in check_menu1([dict(cid=cid,
        threads=[dict(name=t["name"], key_signals=t["key_signals"]) for t in c["th"]],
        reachable=reach, ea_keys=[], vocab=set())])["violations"]]
    eac = dict(cid=cid, frame=c["frame"], ledger_sources=[c["sources"].get(str(e.get("evidence_id")), str(e.get("symptom")))
        for e in sp.evidence_ledger or [] if e.get("source_type") == "patient_reported_symptom"],
        explain_away=(cs.latent.get("m2") or {}).get("explain_away"),
        lines=sorted(core.components(str(ddx.get("spec_id") or ""))) if ddx.get("join_gold") in ("unified", "comorbidity") else [],
        known_visible=list((sp.user_profile or {}).get("known_conditions") or []))
    h["ea2"] = [str(v) for v in check_ea2([eac])["violations"]]
    h["claim"] = list(core.lab_claim_conflicts(raw, T))
    h["gen25"] = [x["kind"] for x in check_ledger_values_traceable(raw, T) if x.get("severity") == "gate"]
    h["ext"] = [x["kind"] for x in gates_for(raw, cs, sp, T) if x["severity"] == "gate" and x["kind"] != "m2_premises_missing"]
    h["unbuy"] = [t for t, _ in core.unbuyable_gold_tests(raw, T)]
    real = set(c["manifest"].get("real_symptom_evidence_ids") or [])
    insufficient = insufficient_from_visible([int(e["source_timestamp"]) for e in sp.evidence_ledger or []
                                            if e.get("evidence_id") in real], T)
    return h, asdict(sp), insufficient


def excluded_nonkeys(c):
    """Track every finding the premise checks actually read, and every stated claim."""
    original = core._on_demand
    read = set()
    def tracked(raw, target, T):
        read.add(str(target))
        return original(raw, target, T)
    core._on_demand = tracked
    try:
        production_premises.emission(c["raw"], c["cs"], stored_sp(c["sp"]), c["T"])
    finally:
        core._on_demand = original
    texts = core.visible_claim_text(c["raw"])
    read.update(fid for fid, (_, pat, neg) in core.LAB_CLAIMS.items() if core._claims_in(texts, pat, neg))
    for fids, _, pat, neg in core.LAB_PANEL_CLAIMS.values():
        if core._claims_in(texts, pat, neg):
            read.update(fids)
    return read


_NAMES = {"b_ondemand": "key_ondemand_normalise", "a_archive_drop": "archive_row_drop",
          "a_archive_norm": "archive_row_negative", "a_outpatient": "outpatient_key_normalise",
          "c_sym_line": "line_symptoms_drop", "c_sym_all": "all_symptoms_drop", "c_sym_one": "neg_symptom_one",
          "d_med_drop_records": "med_drop_records", "d_med_after_onset": "med_after_onset",
          "d_glyc_fpg_today_row": "glyc_fpg_today_row", "d_glyc_post_onset_row": "glyc_post_onset_row",
          "d_glyc_fpg_hook": "glyc_fpg_hook", "d_glyc_hba1c_hook": "glyc_hba1c_hook",
          "d_renal_rows_hook": "renal_rows_hook", "d_renal_below_hook": "renal_below_hook"}


def loses_line_clue(c, raw):
    """Recognize the last discriminating clue using the production EA-2 rule."""
    from haenv.m2.item_audit import check_ea2
    def violations(ledger):
        row = dict(cid=c["cid"], frame=c["frame"],
            ledger_sources=[c["sources"].get(str(e.get("evidence_id")), str(e.get("symptom")))
                            for e in ledger if e.get("source_type") == "patient_reported_symptom"
                            and int(e.get("source_timestamp", 10**9)) <= c["T"]],
            explain_away=(c["cs"].latent.get("m2") or {}).get("explain_away"),
            lines=sorted(core.components(str(c["ddx"].get("spec_id") or "")))
                  if c["ddx"].get("join_gold") in ("unified", "comorbidity") else [],
            known_visible=list((c["sp"].get("user_profile") or {}).get("known_conditions") or []))
        return set(check_ea2([row])["violations"])
    return bool(violations(raw.evidence_ledger) - violations(c["raw"].evidence_ledger))


def operators(c):
    excluded = excluded_nonkeys(c)
    lines = {t["name"]: t for t in c["th"]}
    for kind, meta, raw, override in candidates(c):
        name = _NAMES.get(kind, kind)
        if kind in ("a_archive_drop", "a_archive_norm") and not meta.get("archive_key"):
            continue
        if kind == "neg_ondemand_nonkey" and meta["fid"] in excluded:
            continue
        if kind == "c_sym_one" and (meta["remaining"] < 2 or loses_line_clue(c, raw)):
            continue
        if name.startswith("neg_"):
            target, base = None, None
        elif meta.get("k"):
            base = "m2.archive_key" if meta["archive_key"] else "m2.key_signal"
            target = f"{base}:{meta['component']}:{meta['k']}"
        elif kind == "c_sym_line":
            thread = lines[meta["line"]]
            meta = {**meta, "component": thread["component"]}
            base = "m2.line_symptoms"
            target = base + ":" + line_key(thread)
        elif kind == "c_sym_all":
            base = target = "m2.all_symptoms"
        else:
            base = "m2.premise." + {"glycaemic_diagnosis": "glycaemic"}.get(meta["kind"], meta["kind"])
            target = base + ":" + meta["component"]
        line = line_key(lines[meta["line"]]) if meta.get("line") in lines else None
        suffix = (target or meta.get("fid") or meta.get("eid") or "item")
        # Only identifiers enter a receipt, never translated line names or records.
        public_meta = {k: meta[k] for k in ("component", "k", "fid", "eid", "archive_key", "remaining") if k in meta}
        yield dict(op=name + ":" + suffix, operator=name, target=target, premise=base, raw=raw,
                   override=override, line=line, meta=public_meta)


def visible_blocks(sp):
    return {k: sp.get(k) or {} for k in ("evidence_ledger", "prediction_context", "longitudinal_data", "user_profile")}


def accessible(c, raw, sp):
    blocks = visible_blocks(sp)
    fids = sorted(c.get("on_demand_fids", {k.removeprefix("archive:") for t in c["th"] for k in t["key_signals"]}))
    blocks["on_demand"] = {fid: core._on_demand(raw, fid, c["T"]) for fid in fids}
    return blocks


class Adapter:
    pack = "m2"
    items = staticmethod(load_batch)
    visible_blocks = staticmethod(visible_blocks)
    rule = staticmethod(lambda: SimpleNamespace(sources=()))

    @staticmethod
    def item_roles(raw):
        return emitted_roles(context(raw))

    @staticmethod
    def decisive_facts(raw, sp=None):
        roles = Adapter.item_roles(raw)
        declarations = {p["id"]: p for p in premises("m2")}
        out = defaultdict(list)
        for pid, role in roles.items():
            if role["role"] != "acting":
                continue
            p = declarations[role["premise"]]
            fact = dict(premise=pid, role=role["role"], label=p["label"], visible=p["visible"], observed=role["observed"])
            if sp is not None:
                fact["visible_records"] = record_keys(visible_blocks(sp), p["visible"])
            components = [role["component"]] if role["component"] else sorted({r["component"] for r in roles.values() if r["component"]})
            for comp in components:
                out[comp].append(fact)
        return dict(out)

    @staticmethod
    def item_receipt(raw, sp, *, twins=True):
        from .receipt import _checks
        c = context(raw)
        if "cs" not in c:
            raise ValueError("m2 emission receipts require the job context from load_batch")
        c["sp"] = sp
        raw = c["raw"]
        roles = emitted_roles(c)
        trials = list(operators(c)) if twins else []
        c["on_demand_fids"] = ({k.removeprefix("archive:") for t in c["th"] for k in t["key_signals"]}
                               | {fid for trial in trials for fid in trial["override"] or {}})
        base, rebuilt, _ = checks(c, raw)
        V0 = key_signal_reader(raw, c["T"])
        blocks0 = accessible(c, raw, rebuilt)
        S0 = sorted({t["component"] for t in c["th"]})
        receipt = dict(case_id=c["cid"], pack="m2", decision=S0, S=S0, V=sorted(V0), visible_in_S=V0 <= set(S0),
            roles=roles, twins=[], trace_equal=True, registry_drift=check_complete("m2"),
            render_equal=visible_blocks(rebuilt) == visible_blocks(sp), join=c["ddx"].get("join_gold"),
            baseline_checks={k: len(v) for k, v in base.items() if v}, lines=[])
        ops = {}
        if twins:
            for trial in trials:
                name, target, raw1 = trial["op"], trial["target"], trial["raw"]
                row = dict(case_id=c["cid"], op=name, operator=trial["operator"], target=target,
                    premise=trial["premise"], tier="R", role=roles[target]["role"] if target else "negative",
                    S=S0, outcome_R="error", V=[], outcome_V="error", visible_in_S=False,
                    atoms_changed=[], target_flipped=False, visible=dict(sections=[], records=[], n_leaves=0),
                    trace_equal=None, sha16=None, error=None, line=trial["line"], **trial["meta"])
                ops[name] = Operator(name, target, "R", None)
                try:
                    with on_demand_override(trial["override"]):
                        h, payload, insufficient = checks(c, raw1)
                        new = {k: [v for v in values if v not in base.get(k, [])] for k, values in h.items()}
                        new = {k: values for k, values in new.items() if values}
                        r = outcome(new, insufficient)
                        V1 = key_signal_reader(raw1, c["T"])
                        component = row.get("component") or (roles[target]["component"] if target else None)
                        v = ("flip" if component in V0 and component not in V1 else "same") if target else ("same" if V1 == V0 else "flip")
                        if target == "m2.all_symptoms":
                            v = "flip" if V0 - V1 else "same"
                        # The key-signal reader reads purchases only. Symptom and premise twins are read by
                        # the emission checks themselves, so their visible outcome is the checks' outcome.
                        keyed = bool(target) and roles[target]["premise"] in ("m2.key_signal", "m2.archive_key")
                        if target and not keyed:
                            v = "flip" if r in ("refused", "undetermined") else "same"
                        blocks = accessible(c, raw1, payload)
                        delta = diff(blocks0, blocks)
                        # Key edits must actually alter the result; symptom/premise edits must alter
                        # their declared evidence, not merely trigger an unrelated refusal.
                        effective = bool(delta["n_leaves"]) if target else False
                        if trial["operator"] == "key_ondemand_normalise":
                            fid = trial["meta"]["fid"]
                            effective = blocks0["on_demand"].get(fid) != blocks["on_demand"].get(fid)
                            # An archived key is established by the archived row, not by a fresh purchase:
                            # normalising the purchase does not touch the decisive fact.
                            if str(trial["meta"].get("k") or "").startswith("archive:"):
                                effective = False
                        row.update(outcome_R=r, V=sorted(V1), outcome_V=v,
                            visible_in_S=(r in ("refused", "undetermined")) == (v == "flip") if keyed or not target else True,
                            atoms_changed=[target] if effective else [], target_flipped=effective,
                            visible=delta, trace_equal=True,
                            sha16=hashlib.sha256(json.dumps(blocks, ensure_ascii=False, sort_keys=True, default=str,
                                                separators=(",", ":")).encode()).hexdigest()[:16],
                            refused_by=sorted(new), check_counts={k: len(values) for k, values in new.items()},
                            insufficient=insufficient)
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"[:300]
                receipt["twins"].append(row)
        declarations = []
        by_base = {p["id"]: p for p in premises("m2")}
        for pid, role in roles.items():
            p = {**by_base[role["premise"]], "id": pid}
            p["operators"] = [r["op"] for r in receipt["twins"] if r["target"] == pid]
            declarations.append(p)
        _checks(receipt, ops, declarations)
        receipt["lines"] = line_results(c["th"], receipt["twins"])
        return receipt


def line_results(threads, twins):
    """Keep every emitted line, including lines with only normal exclusion keys."""
    out = []
    for thread in threads:
        lk = line_key(thread)
        rows = [r for r in twins if r["line"] == lk]
        keyrows = [r for r in rows if r["operator"] in ("key_ondemand_normalise", "archive_row_drop", "archive_row_negative", "outpatient_key_normalise")]
        out.append(dict(line=lk, component=thread["component"],
            any_keysignal_op_refused=any(r["outcome_R"] == "refused" for r in keyrows),
            any_keysignal_check_hit=any(r.get("refused_by") for r in keyrows),
            sym_line_refused=any(r["operator"] == "line_symptoms_drop" and r["outcome_R"] == "refused" for r in rows),
            any_refused=any(r["outcome_R"] == "refused" for r in rows),
            V_tested=len(keyrows), V_agreement=sum(r["visible_in_S"] for r in keyrows)))
    return out


def summary_additions(receipts):
    rows = [r for i in receipts for r in i["twins"]]
    table = []
    for op in sorted({r["operator"] for r in rows}):
        groups = ("unified", "comorbidity") if op == "line_symptoms_drop" else (None,)
        for group in groups:
            rr = [r for i in receipts if group is None or i["join"] == group for r in i["twins"] if r["operator"] == op]
            table.append(dict(operator=op, join=group, total=len(rr), outcomes=dict(Counter(r["outcome_R"] for r in rr)),
                              refused_by=dict(Counter(k for r in rr for k in r.get("refused_by", []))),
                              V_flip=sum(r["outcome_V"] == "flip" for r in rr)))
    lines = [l for i in receipts for l in i["lines"]]
    missed = defaultdict(Counter)
    for r in rows:
        if r["operator"] == "key_ondemand_normalise" and r["outcome_R"] == "same":
            missed[r["component"]][r["k"]] += 1
    tested = [r for r in rows if r["tier"] == "R" and r["target_flipped"] and r["role"] in ("acting", "guard")]
    acting = [r for r in tested if r["role"] == "acting"]
    return dict(refusal_table=table, per_line=dict(lines=len(lines),
        **{k: sum(l[k] for l in lines) for k in ("any_keysignal_op_refused", "any_keysignal_check_hit", "sym_line_refused", "any_refused", "V_tested", "V_agreement")}),
        key_normalisations_without_refusal={comp: dict(sorted(keys.items())) for comp, keys in sorted(missed.items())},
        acting_outcomes=dict(Counter(r["outcome_R"] for r in acting)),
        decisive_outcomes=dict(Counter(r["outcome_R"] for r in tested)))
