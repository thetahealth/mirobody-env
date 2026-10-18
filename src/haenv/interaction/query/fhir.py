"""FHIR R4-style shapes of query-mode tool results.

A tool result is what a personal-health-record API returns: a `Bundle` (`type: searchset`, `total`,
`entry`) of typed resources, and the patient as a `Patient`. The values are the record's own,
unchanged; a resource adds its type, an id, where it came from (`meta.source`: a stream, a section
of the pack's record or the evidence ledger) and its time:

* a stream point -> `Observation` (`code`, `valueQuantity`; a laboratory draw has `effectiveDay`
  and its draw time, a device reading `effectivePeriod` over its day);
* a laboratory row of the pack's record -> `Observation` (category `laboratory`, `effectiveDay`);
* a medication row -> `MedicationStatement` (a prescription at a visit -> `MedicationRequest`);
* a visit -> `Encounter` (`period`);
* a dated table of wearable or home readings -> `Observation` (`effectivePeriod`);
* a patient's words or a visit note -> `Communication`;
* any other section -> `Basic`.

Days are case days (`effectiveDay`, `effectivePeriod.startDay/endDay`, `period.startDay`): day 0 is
the start of the record, negative days are earlier history, so no calendar date is invented.
`unshape` returns the record's own row, so a reader can rebuild exactly what it was shown.

The REST form of the tools (`rest_form`) is how an agent would ask a FHIR server for the same
records, e.g. `get_labs(name="creatinine", from_day=-365, to_day=0)` is
`GET /Observation?patient=<id>&category=laboratory&code=creatinine&date=ge-365&date=le0&_count=400`.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

#: keys a resource adds to the record's row
ADDED = ("resourceType", "id", "meta", "code", "category", "effectiveDay", "effectivePeriod", "valueQuantity",
         "period")
#: resource type by section kind (the first kind of a section with several)
BY_KIND = {"labs": "Observation", "meds": "MedicationStatement", "notes": "Communication", "series": "Observation",
           "record": "Basic", "profile": "Basic"}
VITALS = ("weight", "systolic_bp", "diastolic_bp", "resting_hr", "heart_rate", "spo2", "body_temp", "skin_temp",
          "respiratory_rate", "hrv")


def _kind0(kind) -> str:
    return kind if isinstance(kind, str) else kind[0]


def stream_head(name: str, *, lab: bool) -> dict:
    """What every Observation of one stream shares, stated once on its Bundle (a stream of daily
    device summaries: each entry's `effectiveDay` is its `effectivePeriod`, that whole day)."""
    return {"entryResourceType": "Observation", "meta": {"source": name}, "code": {"text": name},
            "category": "laboratory" if lab else ("vital-signs" if name in VITALS else "activity"),
            **({} if lab else {"effectivePeriod": "the entry's whole day (daily device summary)"})}


def point(name: str, p: dict, *, lab: bool, full: bool = False) -> dict:
    """A stream point `{ts, value, ...}` as a compact Observation entry of its stream's Bundle, or
    (`full`, in a Bundle that mixes sources) a whole Observation."""
    r = ({"resourceType": "Observation", "meta": {"source": name}, "code": {"text": name},
          "category": "laboratory" if lab else "activity"} if full else {})
    r.update({"effectiveDay": int(p["ts"]), "valueQuantity": {"value": p.get("value")}})
    r.update({k: v for k, v in p.items() if k not in ("ts", "value")})
    return r


def resource(section: str, kind, row: dict, idx: int, *, rtype: str | None = None, day: int | None = None,
             code: str | None = None) -> dict:
    """A row of the pack's record (or of the evidence ledger) as a resource; the row's own fields stay."""
    k0 = _kind0(kind)
    rt = rtype or BY_KIND.get(k0, "Basic")
    r = {"resourceType": rt, "id": str(row.get("evidence_id") or f"{section}-{idx}"), "meta": {"source": section}}
    if rt == "Observation":
        r["category"] = "laboratory" if k0 == "labs" else "activity"
        if code:
            r["code"] = {"text": code}
    if day is not None:
        if rt == "Encounter":
            r["period"] = {"startDay": day, "endDay": day}
        elif k0 == "series":
            r["effectivePeriod"] = {"startDay": day, "endDay": day}
        else:
            r["effectiveDay"] = day
    r.update(row)
    return r


def unshape(r: dict) -> dict:
    """The record's own row (or stream point) of a resource or stream entry."""
    if not isinstance(r, dict):
        return r
    if "valueQuantity" in r and "effectiveDay" in r:
        return {"ts": r["effectiveDay"], "value": r["valueQuantity"].get("value"),
                **{k: v for k, v in r.items() if k not in ADDED}}
    if "resourceType" not in r:
        return r
    return {k: v for k, v in r.items() if k not in ADDED}


def source_of(r: dict):
    return (r.get("meta") or {}).get("source") if isinstance(r, dict) else None


def bundle(entries: list, total: int, truncated: dict | None = None, **head) -> dict:
    out = {"resourceType": "Bundle", "type": "searchset", "total": total, **head, "entry": entries}
    if truncated:
        out["truncated"] = truncated
    return out


def patient(case_id, profile: dict, extensions: dict) -> dict:
    sex = str(profile.get("sex") or "")
    return {"resourceType": "Patient", "id": str(case_id or "patient"),
            **({"gender": {"M": "male", "F": "female"}[sex]} if sex in ("M", "F") else {}),
            **profile, **({"extension": extensions} if extensions else {})}


#: tool -> (method, path, query parameters) of the REST form
REST = {"list_sources": ("GET", "/Patient/{id}/$everything?_summary=count", ()),
        "get_profile": ("GET", "/Patient/{id}", ()),
        "get_series": ("GET", "/Observation", (("patient", "{id}"), ("code", "name"))),
        "get_labs": ("GET", "/Observation", (("patient", "{id}"), ("category", "laboratory"), ("code", "name"))),
        "get_meds": ("GET", "/MedicationStatement", (("patient", "{id}"),)),
        "get_notes": ("GET", "/Communication", (("patient", "{id}"), ("category", "kind"))),
        "get_record": ("GET", "/{section}", (("patient", "{id}"),)),
        "get_protocol": ("GET", "/Basic/$protocols", (("name", "name"), ("topic", "topic"), ("keyword", "keyword"))),
        "get_menu": ("GET", "/ServiceRequest/$menu", (("patient", "{id}"),)),
        "order_test": ("POST", "/ServiceRequest", ())}


def rest_form(tool: str, args: dict, patient_id: str = "<id>", count: int = 400) -> str:
    """The REST request a FHIR server would take for one tool call (documentation and reports)."""
    method, path, params = REST[tool]
    path = path.replace("{id}", patient_id).replace("{section}", str(args.get("section", "Basic")))
    q = []
    for key, src in params:
        if src == "{id}":
            q.append(f"{key}={patient_id}")
        elif src in args and args[src] not in (None, ""):
            q.append(f"{key}={args[src]}")
        elif src not in ("name", "kind"):
            q.append(f"{key}={src}")
    if "from_day" in args and args["from_day"] is not None:
        q.append(f"date=ge{args['from_day']}")
    if "to_day" in args and args["to_day"] is not None:
        q.append(f"date=le{args['to_day']}")
    if method == "GET" and tool not in ("list_sources", "get_profile"):
        q.append(f"_count={count}")
    body = f"  {{\"code\": {{\"text\": \"{args.get('target')}\"}}}}" if method == "POST" else ""
    return f"{method} {path}" + (("?" if "?" not in path else "&") + "&".join(q) if q else "") + body
