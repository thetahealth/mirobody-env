"""Rule-emitted roles, twin outcomes, per-item checks, and language joins."""
from collections import Counter, defaultdict
import hashlib
import json

from .batch import InvalidTwin, Item
from .diffs import diff
from .registry import atom_map, check_complete, premise_id, premises

_DECISIVE = ("acting", "guard")
_OUTCOMES = ("flip", "partial", "same", "undetermined", "invalid", "error")


def _observations(pack, events):
    mapping = atom_map(pack)
    seen = defaultdict(lambda: defaultdict(set))
    for _label, text, _line, value, _effective in events:
        pid = premise_id(mapping[text]) if text in mapping else None
        if pid:
            seen[pid][text].add(value)
    return seen


def _values(pack, events):
    seen = _observations(pack, events)
    out = {}
    for row in premises(pack):
        values = {value for vs in seen[row["id"]].values() for value in vs}
        out[row["id"]] = next(iter(values)) if len(values) == 1 else None
    return out


def roles(adapter, raw):
    """Flip only observed premise atoms in the actual production call sequence."""
    item = raw if isinstance(raw, Item) else Item(getattr(raw, "case_id", ""), raw, {})
    if hasattr(adapter, "item_roles"):
        return adapter.item_roles(item.raw)
    args = adapter.rule_args(item)
    def decide(override=None):
        if hasattr(adapter, "decide"):
            return adapter.decide(*args, override=override)
        result, events = adapter.rule().run(*args, override=override)
        return adapter.decision(result), events
    v0, events = decide()
    seen = _observations(adapter.pack, events)
    out = {}
    for row in premises(adapter.pack):
        texts = seen[row["id"]]
        values = {value for vs in texts.values() for value in vs}
        observed = next(iter(values)) if len(values) == 1 else None
        entry = dict(role="unread", observed=observed, cf_decision=None, texts=sorted(texts))
        if any(len(vs) > 1 for vs in texts.values()):
            entry["role"] = "varying"
        elif len(values) > 1:
            entry["role"] = "inconsistent"
        elif values:
            try:
                v1, _ = decide({text: not observed for text in texts})
            except Exception:
                v1 = None
            entry["cf_decision"] = v1
            entry["role"] = ("acting" if observed else "guard") if v1 != v0 else ("overridden" if observed else "context")
        out[row["id"]] = entry
    return out


def _visible_in(v, scored):
    if scored is None:
        return False
    if scored and isinstance(scored[0], (tuple, list)):
        return tuple(v) in {tuple(s) for s in scored}
    return sorted(v) == sorted(scored) if isinstance(v, tuple) else v in scored


def _outcome(scored, original):
    if scored is None:
        return "undetermined"
    if scored == original:
        return "same"
    return "partial" if set(scored) & set(original or []) else "flip"


def _visible_outcome(v, original_v, original_s):
    if original_s and isinstance(original_s[0], (tuple, list)):
        return "same" if v == original_v else "partial" if _visible_in(v, original_s) else "flip"
    if isinstance(v, tuple):
        return _outcome(sorted(v), original_s)
    return "same" if v == original_v else "partial" if _visible_in(v, original_s) else "flip"


def _error(exc):
    return f"{type(exc).__name__}: {exc}"[:300]


def rendered_equal(rebuilt, stored):
    """The re-rendered visible blocks equal the stored payload's, compared in the batch's language: the
    re-render holds language-free messages, the stored payload their rendering (an en payload never equals the
    canonical zh text of a message)."""
    from haenv.i18n import current
    from haenv.i18n.text import MissingText, render
    lang = current()
    try:
        return render(rebuilt, lang) == render(stored, lang)
    except MissingText:
        return False


def item_receipt(adapter, raw, sp, *, twins=True):
    if hasattr(adapter, "item_receipt"):
        return adapter.item_receipt(raw, sp, twins=twins)
    item = Item(getattr(raw, "case_id", ""), raw, sp)
    if hasattr(adapter, "prepare"):
        adapter.prepare(item)
    result, events = adapter.rule().run(*adapter.rule_args(item))
    item_roles = roles(adapter, item)
    S0, V0 = adapter.scored(raw), adapter.visible_answer(sp)
    baseline = adapter.baseline_payload(item) if hasattr(adapter, "baseline_payload") else sp
    blocks0 = adapter.visible_blocks(baseline)
    meta = adapter.metadata(item, result) if hasattr(adapter, "metadata") else {}
    receipt = dict(case_id=item.case_id, pack=adapter.pack, decision=adapter.decision(result), S=S0, V=V0,
        visible_in_S=_visible_in(V0, S0), roles=item_roles, twins=[],
        trace_equal=adapter.rule().instrumented_equals(*adapter.rule_args(item)),
        registry_drift=check_complete(adapter.pack), render_equal=rendered_equal(blocks0, adapter.visible_blocks(sp)), **meta)
    ops = adapter.operators()
    values0 = _values(adapter.pack, events)
    if twins:
        for name, op in ops.items():
            if hasattr(adapter, "set_operator"):
                adapter.set_operator(name)
            block = adapter.block(raw) if hasattr(adapter, "block") else adapter.rule_args(item)[0]
            target = op.target_for(block)
            row = dict(case_id=item.case_id, op=name, target=target, tier=op.tier,
                role=item_roles[target]["role"] if target else "negative", S=[], outcome_R="error", V=None,
                outcome_V="error", visible_in_S=False, atoms_changed=[], target_flipped=False,
                visible=dict(sections=[], records=[], n_leaves=0), trace_equal=None, sha16=None, error=None)
            try:
                twin = op.fn(block)
            except Exception as exc:
                row["outcome_R"] = "invalid" if isinstance(exc, InvalidTwin) else "error"
                row["error"] = _error(exc)
                receipt["twins"].append(row)
                continue
            if twin is None:
                continue
            try:
                scored = adapter.scored(twin)
                row.update(S=scored if scored is not None else [], outcome_R=_outcome(scored, S0))
                if hasattr(adapter, "scored_metadata"):
                    row.update(adapter.scored_metadata(item, twin, row["outcome_R"]))
                twin_item = Item(item.case_id, twin, {})
                twin_args = adapter.rule_args(twin_item)
                _, twin_events = adapter.rule().run(*twin_args)
                values1 = _values(adapter.pack, twin_events)
                changed = sorted(pid for pid in values0 if values0[pid] != values1[pid])
                row.update(atoms_changed=changed, target_flipped=target in changed,
                           trace_equal=adapter.rule().instrumented_equals(*twin_args))
            except Exception as exc:
                row.update(outcome_R="error", error=_error(exc))
            try:
                payload = adapter.twin_payload(item, twin)
                blocks = adapter.visible_blocks(payload)
                row["visible"] = diff(blocks0, blocks)
                row["sha16"] = hashlib.sha256(json.dumps(blocks, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":")).encode()).hexdigest()[:16]
                visible = adapter.visible_answer(payload)
                row.update(V=visible, outcome_V=_visible_outcome(visible, V0, S0),
                    visible_in_S=_visible_in(visible, row["S"]) if row["outcome_R"] != "undetermined" else False)
                if hasattr(adapter, "twin_metadata"):
                    row.update(adapter.twin_metadata(item, twin, payload))
            except Exception as exc:
                row["error"] = _error(exc)
                if isinstance(exc, InvalidTwin):
                    row["outcome_R"] = "invalid"
            receipt["twins"].append(row)
    _checks(receipt, ops, premises(adapter.pack))
    return receipt


def _checks(receipt, ops, declarations):
    rows, roles0 = receipt["twins"], receipt["roles"]
    declarations = {p["id"]: p for p in declarations}
    tested = [r for r in rows if r["tier"] == "R" and r["role"] in _DECISIVE and r["target_flipped"]]
    failures = []
    for row in tested:
        accepted = row["outcome_R"] == "flip" or (receipt.get("pack") == "m2" and row["outcome_R"] == "refused") or (row["outcome_R"] == "undetermined"
                    and declarations[row["target"]].get("undetermined_ok", False))
        if not (accepted and row["outcome_V"] == "flip" and row["visible_in_S"]):
            failures.append({"target": row["target"], "op": row["op"], "outcome_R": row["outcome_R"], "outcome_V": row["outcome_V"]})
    missing = []
    for pid, role in roles0.items():
        if role["role"] not in _DECISIVE or any(r["target"] == pid for r in tested):
            continue
        declared_ops = declarations[pid]["operators"]
        actual = [op for name, op in ops.items() if name in declared_ops]
        if not declared_ops:
            reason = "no_operator"
        elif actual and all(op.tier != "R" for op in actual):
            reason = "tier_w_only"
        elif any(r["target"] == pid and r["tier"] == "R" for r in rows):
            reason = "operator_ineffective"
        else:
            reason = "not_applicable"
        missing.append({"premise": pid, "reason": reason})
    negatives = [r for r in rows if r["role"] == "negative"]
    ineffective = [{"op": r["op"], "target": r["target"]} for r in rows if r["target"] and not r["target_flipped"]]
    unexpected = [{"op": r["op"], "target": r["target"], "outcome_R": r["outcome_R"]} for r in rows
                  if r["tier"] == "R" and r["role"] in ("overridden", "context", "unread") and r["outcome_R"] != "same"]
    vfail = [r["op"] for r in tested if not r["visible"]["n_leaves"] or not r["visible_in_S"]]
    undeclared = [{"op": r["op"], "sections": sorted(set(r["visible"]["sections"]) - set(declarations[r["target"]]["visible"]))}
                  for r in rows if r["target"] and set(r["visible"]["sections"]) - set(declarations[r["target"]]["visible"])]
    state_map = {s: p["id"] for p in declarations.values() for s in p.get("declared_states", [])}
    findings = sorted(s for s in receipt.get("declared_states", [])
                      if roles0.get(state_map.get(s), {}).get("role") not in _DECISIVE)
    receipt.update(N=dict(status="fail" if failures else "pass" if tested else "untested", tested=len(tested),
                          failures=failures, untested=missing),
        I=dict(status="pass" if not unexpected and all(r["outcome_R"] == r["outcome_V"] == "same" for r in negatives) else "fail",
               tested=len(negatives), same=sum(r["outcome_R"] == r["outcome_V"] == "same" for r in negatives)),
        V_check=dict(status="fail" if vfail else "pass", tested=len(tested), failures=vfail,
                     undeclared_sections=undeclared),
        unexpected_change=unexpected, operator_ineffective=ineffective, declared_state_not_decisive=findings,
        fidelity_fail=not receipt["trace_equal"] or bool(receipt["registry_drift"])
            or any(r["trace_equal"] is False for r in rows),
        role_findings=[pid for pid, r in roles0.items() if r["role"] in ("varying", "inconsistent")])


def summarize(receipts):
    rows = [r for item in receipts for r in item["twins"]]
    tested = [r for r in rows if r["tier"] == "R" and r["target_flipped"] and r["role"] in _DECISIVE]
    acting = [r for r in tested if r["role"] == "acting"]
    counts = Counter((r["op"], r["role"], r["outcome_R"], r["outcome_V"]) for r in rows)
    failures = Counter()
    for item in receipts:
        for f in item["N"]["failures"]:
            failures[(f["target"], str(item["decision"]), item.get("drug", item.get("rule", "")))] += 1
    summary = dict(items=len(receipts), twins=len(rows),
        decisive_by_role=dict(Counter(role["role"] for item in receipts for role in item["roles"].values() if role["role"] in _DECISIVE)),
        N={s: sum(item["N"]["status"] == s for item in receipts) for s in ("pass", "fail", "untested")},
        acting_twins_tested=len(acting), acting_outcomes={o: sum(r["outcome_R"] == o for r in acting) for o in _OUTCOMES},
        decisive_twins_tested=len(tested), negatives=dict(tested=sum(i["I"]["tested"] for i in receipts), same=sum(i["I"]["same"] for i in receipts)),
        I_fail=sum(i["I"]["status"] == "fail" for i in receipts),
        V=dict(tested=len(tested), agreement=sum(r["visible_in_S"] for r in tested),
               failures=sum(i["V_check"]["status"] == "fail" for i in receipts),
               all_twins_tested=len(rows), all_twins_agreement=sum(r["visible_in_S"] for r in rows),
               undeclared_sections=sum(len(i["V_check"]["undeclared_sections"]) for i in receipts)),
        unexpected_change=sum(len(i["unexpected_change"]) for i in receipts),
        declaration=dict(items=sum(bool(i["declared_state_not_decisive"]) for i in receipts),
                         states=dict(Counter(s for i in receipts for s in i["declared_state_not_decisive"]))),
        fidelity=dict(failed=sum(i["fidelity_fail"] for i in receipts),
                      registry_drift=[i["registry_drift"] for i in receipts if i["registry_drift"]]),
        render_equal=sum(i["render_equal"] for i in receipts),
        stored_gold_equal=sum(i.get("stored_gold_equal", False) for i in receipts),
        role_findings=sum(len(i["role_findings"]) for i in receipts),
        operator_ineffective=sum(len(i["operator_ineffective"]) for i in receipts),
        twins_by_outcome=[dict(op=k[0], role=k[1], outcome_R=k[2], outcome_V=k[3], count=n) for k, n in sorted(counts.items())],
        failure_classes=[dict(target=k[0], decision=k[1], cell=k[2], count=n) for k, n in sorted(failures.items())])
    if receipts and receipts[0]["pack"] == "m2":
        from .m2 import summary_additions
        summary.update(summary_additions(receipts))
    return summary


def join_languages(zh_receipts, en_receipts):
    """Join by case and operator; compare stable record sets rather than list positions."""
    zh = {i["case_id"]: i for i in zh_receipts}
    en = {i["case_id"]: i for i in en_receipts}
    differences = []
    equal = total = 0
    for cid in sorted(zh.keys() | en.keys()):
        if cid not in zh or cid not in en:
            differences.append(dict(case_id=cid, op=None, fields=["missing_case"]))
            continue
        a = {r["op"]: r for r in zh[cid]["twins"]}
        b = {r["op"]: r for r in en[cid]["twins"]}
        for op in sorted(a.keys() | b.keys()):
            total += 1
            if op not in a or op not in b:
                differences.append(dict(case_id=cid, op=op, fields=["missing_operator"]))
                continue
            fields = [k for k in ("S", "outcome_R", "outcome_V", "role") if a[op][k] != b[op][k]]
            if set(a[op]["visible"]["records"]) != set(b[op]["visible"]["records"]):
                fields.append("visible.records")
            if fields:
                differences.append(dict(case_id=cid, op=op, fields=fields, zh={k: a[op]["visible"]["records"] if k == "visible.records" else a[op][k] for k in fields},
                                        en={k: b[op]["visible"]["records"] if k == "visible.records" else b[op][k] for k in fields}))
            else:
                equal += 1
    return dict(status="fail" if differences else "pass", cases_zh=len(zh), cases_en=len(en),
                rows=total, equal=equal, differing=len(differences), differences=differences)
