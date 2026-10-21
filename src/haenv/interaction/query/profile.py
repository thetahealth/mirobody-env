"""The process profile of one query-mode cell: what was read, whether the decisive evidence was
read, how many lookups read nothing new. Reported next to the pack's score, never in it.

The decisive evidence of a case is declared by its pack adapter (`key_lookups`, verifier side) as
predicates over the rows a tool returned; a piece counts as seen when a tool result before the
answer carried a row it matches. So the profile reads what the model was shown, not what it asked
for.

Columns (`q_*`):
`q_stop` answered / forced_final / unanswered · `q_rounds` · `q_n_calls` · `q_tools` calls per tool ·
`q_ungrounded` calls that found nothing · `q_dup` repeats of an earlier call (same tool, same
arguments) · `q_errors` calls refused (unknown tool, bad arguments, over the call limit) ·
`q_irrelevant` = ungrounded + repeats + refused · `q_result_chars` · `q_key_n` / `q_key_seen_n` /
`q_key_seen` (all seen; None without declared evidence) / `q_key_missed`.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import json
import re

from haenv.interaction.query.fhir import source_of
from haenv.interaction.query.fhir import unshape


def returned_rows(trace: dict):
    """`(tool name, row)` for every row, point, value or profile a tool returned in the cell; each
    row carries `_from`, the stream, section, ledger or purchase it came from."""
    for m in trace.get("messages") or ():
        if m.get("role") != "tool" or not isinstance(m.get("content"), dict):
            continue
        res, name = m["content"], m.get("name")
        where = res.get("section") or res.get("name") or res.get("target")
        for r in res.get("entry") or ():
            if isinstance(r, dict):
                yield name, {**unshape(r), "_from": source_of(r) or where}
        if "value" in res:
            yield name, {"_from": where, "value": res["value"]}
        if res.get("resourceType") == "Patient":
            yield name, {"_from": "profile", **{k: v for k, v in res.items() if k not in ("extension", "found")}}
            for k, v in (res.get("extension") or {}).items():
                yield name, {"_from": k, "value": v}


_ROW_ID = re.compile(r"([A-Za-z_]+)-(\d+)")


def returned_record_ids(trace: dict, sections) -> frozenset[str]:
    """Ids of the record rows the tools returned in the cell, for the citation check
    (`verifier.grade(returned_ids=)`): every `<section>-<index>` resource id in a tool result
    whose section is one of the pack's record `sections` (not the Patient resource, a protocol or
    a series). Read off the tool results the model was shown, so a row counts only if it was
    returned in this answer."""
    names = set(sections or ())
    if not names:
        return frozenset()
    out: set[str] = set()
    for m in trace.get("messages") or ():
        if m.get("role") != "tool":
            continue
        stack = [m.get("content")]
        while stack:
            o = stack.pop()
            if isinstance(o, dict):
                rid = o.get("id")
                hit = _ROW_ID.fullmatch(rid) if isinstance(rid, str) else None
                if hit and hit.group(1) in names:
                    out.add(rid)
                stack.extend(o.values())
            elif isinstance(o, list):
                stack.extend(o)
    return frozenset(out)


def returned_ids_of(solver) -> frozenset[str]:
    """`returned_record_ids` of the cell a query-mode solver just answered (`_last_query`, set by
    the query loop and reset by the row wrapper before each cell); empty outside query mode."""
    last = getattr(solver, "_last_query", None)
    if not last:
        return frozenset()
    trace, store = last
    return returned_record_ids(trace or {}, getattr(store, "sections", None))


def profile(trace: dict, lookups: list) -> dict:
    calls = trace.get("calls") or []
    seen_args: set = set()
    dup = 0
    tools: dict[str, int] = {}
    for c in calls:
        tools[c["name"]] = tools.get(c["name"], 0) + 1
        key = (c["name"], json.dumps(c.get("arguments"), sort_keys=True, ensure_ascii=False, default=str))
        if key in seen_args and not c.get("error"):
            dup += 1
        seen_args.add(key)
    ungrounded = sum(1 for c in calls if c.get("found") is False)
    errors = sum(1 for c in calls if c.get("error"))
    rows = list(returned_rows(trace))
    seen = [lk for lk in lookups if any(lk.seen(name, r) for name, r in rows)]
    missed = [lk.what for lk in lookups if lk not in seen]
    return {"q_stop": trace.get("stop"), "q_rounds": trace.get("rounds_used"), "q_n_calls": len(calls),
            "q_tools": tools, "q_ungrounded": ungrounded, "q_dup": dup, "q_errors": errors,
            "q_irrelevant": ungrounded + dup + errors, "q_protocol_errors": trace.get("protocol_errors"),
            "q_result_chars": sum(int(c.get("result_chars") or 0) for c in calls),
            "q_key_n": len(lookups), "q_key_seen_n": len(seen),
            "q_key_seen": (len(seen) == len(lookups)) if lookups else None, "q_key_missed": missed or None}
