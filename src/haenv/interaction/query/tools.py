"""The record store and the common tool set of query mode.

The store is built from the solver payload of `build_instance(raw, T)` and nothing else: it takes
no verifier payload and no raw case, so the gold is not reachable from a tool by construction.
Every tool result also passes `ResultGuard` before the model sees it (a day after T, a kernel
forbidden token, a pack gold word or a gold number stops the cell), as the leakage probe does for
the single-prompt payload.

One tool set for every pack; a pack only says which part of its block is which kind of record
(`adapter.QueryAdapter.sections`):

* `list_sources()`                          -- what this patient has up to today, with day ranges;
* `get_profile()`                           -- the patient's profile (and the pack's background);
* `get_series(name, from_day, to_day)`      -- one monitoring stream, or a dated table of the pack's record;
* `get_labs(name, from_day, to_day)`        -- laboratory results (lab streams and the pack's lab rows);
* `get_meds(from_day, to_day)`              -- medication records;
* `get_notes(from_day, to_day, kind)`       -- the patient's own words and visit notes (evidence ledger
                                               and the pack's note sections);
* `get_record(section, from_day, to_day)`   -- any section of the pack's record by name.

Days are case days (day 0 = the start of this record; negative = earlier history). A window
defaults to everything up to today. One call returns at most `max_points` points or rows; a longer
window returns the latest ones and says how to page back (`earlier_page`). A tool that finds
nothing returns `{"found": false, ...}`; the process profile counts that call as ungrounded.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Callable

from haenv.interaction.query import fhir

#: Default points or rows per call (the configured value is `config.query.max_points`).
MAX_POINTS = 400

#: Keys whose integer value is a day on the case clock; the guard refuses any of them after T,
#: and a row's own day is read from the first one present.
DAY_KEYS = ("day", "ts", "source_timestamp", "draw_day", "onset_day", "today", "today_day",
            "previous_draw_day", "first_day", "last_day", "night_of_day", "effectiveDay", "startDay", "endDay")
ROW_DAY_KEYS = ("day", "ts", "source_timestamp", "draw_day", "onset_day", "night_of_day")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    params: dict                     # JSON-schema object (properties / required)
    fn: Callable[..., dict] = field(compare=False, repr=False)

    def neutral(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.params}


def _obj(props: dict, required: tuple = ()) -> dict:
    return {"type": "object", "properties": props, "required": list(required)}


_DAY = {"type": "integer", "description": "病程日(本记录第 0 天起算;负数为更早的历史)"}
_WIN = {"from_day": {**_DAY, "description": "起始日(含);省略 = 不设下限"},
        "to_day": {**_DAY, "description": "截止日(含);省略 = 到今天为止"}}


def row_day(row) -> int | None:
    if not isinstance(row, dict):
        return None
    for k in ROW_DAY_KEYS:
        v = row.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return int(v)
    return None


def _window(rows: list, lo, hi, T: int) -> list:
    lo = None if lo is None else int(lo)
    hi = T if hi is None else min(int(hi), T)
    out = []
    for r in rows:
        d = row_day(r)
        if d is None or ((lo is None or d >= lo) and d <= hi):
            out.append(r)
    return out


def _span(rows: list) -> dict:
    days = [d for d in map(row_day, rows) if d is not None]
    return {"first_day": min(days), "last_day": max(days)} if days else {}


class RecordStore:
    """Read-only view of one solver payload at time T, through a pack adapter (or none)."""

    def __init__(self, sp, T: int, adapter=None, *, max_points: int = MAX_POINTS):
        if hasattr(sp, "adjudication") or hasattr(sp, "outcome_label"):
            raise TypeError("RecordStore takes the solver payload only, never the verifier side or the raw case")
        self.T, self.max_points, self.adapter = int(T), int(max_points), adapter
        self.case_id = getattr(sp, "case_id", None)
        self.profile = copy.deepcopy(getattr(sp, "user_profile", None) or {})
        self.streams = {k: [p for p in (v or []) if int(p["ts"]) <= self.T]
                        for k, v in (getattr(sp, "longitudinal_data", None) or {}).items()}
        self.notes = [e for e in (getattr(sp, "evidence_ledger", None) or [])
                      if int(e.get("source_timestamp", 0)) <= self.T]
        self.context = copy.deepcopy(getattr(sp, "prediction_context", None) or {})
        self.block = (self.context.pop(adapter.block, None) if adapter is not None and adapter.block else None) or {}
        self.sections = dict(adapter.sections) if adapter is not None else {}
        self.protocols = self.block.pop("protocol_library", [])
        if self.protocols == "本院规程库":
            from haenv.world.protocols.library import catalog
            self.protocols = catalog()
        if adapter is not None:
            # Common dated sources can live beside a pack's scene block. They
            # use the same registered-section retrieval and paging contract.
            self.block.update({k: self.context.pop(k) for k in self.sections
                               if k in self.context and k not in self.block})

    # ---- helpers ----
    def _bundle(self, items: list, **head) -> dict:
        """`items` = [(row, shape)]: the latest `max_points` in day order, each shaped as a FHIR-style
        resource (`fhir.py`), in a searchset Bundle; a longer window says how to page back."""
        items = sorted(items, key=lambda it: (row_day(it[0]) is not None, row_day(it[0]) or 0))
        kept = items[-self.max_points:]
        trunc = None
        if len(items) > self.max_points:
            first = next((row_day(r) for r, _f in kept if row_day(r) is not None), None)
            trunc = {"returned": self.max_points, "total_in_window": len(items), "kept": "latest",
                     **({"earlier_page": {"to_day": first - 1}} if first is not None else {})}
        return {"found": True, **fhir.bundle([shape(r) for r, shape in kept], len(items), trunc, **head)}

    def _stream_items(self, name: str, lo, hi, *, full: bool = False) -> list:
        lab = name in self.lab_streams()
        return [(p, lambda p, n=name: fhir.point(n, p, lab=lab, full=full))
                for p in _window(self.streams[name], lo, hi, self.T)]

    def _section_items(self, s: str, lo, hi) -> list:
        kind = self.sections[s][0]
        rtype = (getattr(self.adapter, "fhir_types", None) or {}).get(s)
        code = self.adapter.lab_label(self.block, s) if (self.adapter is not None and "labs" in (
            {kind} if isinstance(kind, str) else set(kind))) else None
        rows = list(enumerate(self.section_rows(s)))
        keep = {id(r) for r in _window([r for _i, r in rows], lo, hi, self.T)}
        return [(r, lambda r, i=i: fhir.resource(s, kind, r, i, rtype=rtype, day=row_day(r), code=code))
                for i, r in rows if id(r) in keep]

    def _ledger_items(self, kind, lo, hi) -> list:
        rows = [e for e in self.notes if kind in (None, "", str(e.get("source_type")))]
        return [(e, lambda e, i=i: fhir.resource("evidence_ledger", "notes", e, i, day=row_day(e)))
                for i, e in enumerate(_window(rows, lo, hi, self.T))]

    def section_rows(self, name: str) -> list:
        """The rows of one block section; a dict section is one row, a scalar is `{"value": ...}`."""
        v = self.block.get(name)
        if isinstance(v, list):
            return [r if isinstance(r, dict) else {"value": r} for r in v]
        if isinstance(v, dict):
            return [v]
        return [] if v is None else [{"value": v}]

    def _kind(self, kinds) -> list[str]:
        want = {kinds} if isinstance(kinds, str) else set(kinds)
        return [s for s, (k, _d) in self.sections.items()
                if want & ({k} if isinstance(k, str) else set(k)) and s in self.block]

    def lab_streams(self) -> list[str]:
        """Streams that are laboratory draws: the adapter's list, else streams whose points carry
        a draw time."""
        if self.adapter is not None and self.adapter.lab_streams is not None:
            return [s for s in self.adapter.lab_streams if s in self.streams]
        return sorted(k for k, v in self.streams.items() if v and "time" in v[0])

    # ---- common tools ----
    def list_sources(self) -> dict:
        labs = set(self.lab_streams())
        streams = [{"name": k, "n_points": len(v), **_span(v), **({"lab": True} if k in labs else {})}
                   for k, v in sorted(self.streams.items())]
        kinds: dict[str, int] = {}
        for e in self.notes:
            kinds[str(e.get("source_type"))] = kinds.get(str(e.get("source_type")), 0) + 1
        records = []
        for s, (k, desc) in self.sections.items():
            if s in self.block:
                rows = self.section_rows(s)
                records.append({"section": s, "kind": k if isinstance(k, str) else list(k),
                                "n_rows": len(rows), **_span(rows), "about": desc})
        out = {"today": self.T, "profile_fields": sorted(self.profile), "streams": streams,
               "notes": {"n": len(self.notes), "kinds": kinds, **_span(self.notes)}}
        if records:
            out["records"] = records
        if self.protocols:
            out["protocols"] = [{"protocol_id": d["protocol_id"], "name": d["name"], "topics": d["topics"]}
                                for d in self.protocols]
        return out

    def get_protocol(self, name=None, topic=None, keyword=None) -> dict:
        from haenv.world.protocols.library import search
        docs = search(self.protocols, name=name, topic=topic, keyword=keyword)
        return {"found": bool(docs), **fhir.bundle([
            fhir.resource("protocol_library", "record", d, i) for i, d in enumerate(docs)], len(docs))}

    def get_profile(self) -> dict:
        return {"found": True, **fhir.patient(self.case_id, self.profile,
                                              {s: self.block[s] for s in self._kind("profile")})}

    def get_series(self, name: str, from_day: int | None = None, to_day: int | None = None) -> dict:
        name = str(name)
        if name in self.streams:
            return self._bundle(self._stream_items(name, from_day, to_day), name=name,
                                **fhir.stream_head(name, lab=name in self.lab_streams()))
        if name in self._kind("series"):
            return self._bundle(self._section_items(name, from_day, to_day), name=name)
        return {"found": False, "name": name, "available": sorted(self.streams) + self._kind("series")}

    def get_labs(self, name: str | None = None, from_day: int | None = None, to_day: int | None = None) -> dict:
        items = []
        for s in self.lab_streams():
            if name in (None, "", s):
                items += self._stream_items(s, from_day, to_day, full=True)
        for s in self._kind("labs"):
            label = self.adapter.lab_label(self.block, s) if self.adapter is not None else s
            if name in (None, "", s) or (name and name in label):
                items += self._section_items(s, from_day, to_day)
        if not items:
            return {"found": False, "name": name, "available": self.lab_streams() + [
                self.adapter.lab_label(self.block, s) if self.adapter is not None else s for s in self._kind("labs")]}
        return self._bundle(items)

    def get_meds(self, from_day: int | None = None, to_day: int | None = None) -> dict:
        items = [it for s in self._kind("meds") for it in self._section_items(s, from_day, to_day)]
        return self._bundle(items) if items else {"found": False}

    def get_notes(self, from_day: int | None = None, to_day: int | None = None, kind: str | None = None) -> dict:
        items = self._ledger_items(kind, from_day, to_day)
        for s in self._kind("notes"):
            if kind in (None, "", s):
                items += self._section_items(s, from_day, to_day)
        return self._bundle(items) if items else {"found": False, "kinds": sorted(
            {str(e.get("source_type")) for e in self.notes} | set(self._kind("notes")))}

    def get_record(self, section: str, from_day: int | None = None, to_day: int | None = None) -> dict:
        section = str(section)
        if section not in self.sections or section not in self.block:
            return {"found": False, "section": section, "available": [s for s in self.sections if s in self.block]}
        v = self.block[section]
        if not isinstance(v, list) or not all(isinstance(r, dict) for r in v):
            return {"found": True, "section": section, "resourceType": "Basic", "id": section,
                    "meta": {"source": section}, "value": v}
        return self._bundle(self._section_items(section, from_day, to_day), section=section)

    def common_specs(self) -> list[ToolSpec]:
        mp = self.max_points
        out = [
            ToolSpec("list_sources", "列出这位患者截至今天有哪些数据:各监测流与本次记录各部分的条数与起止日、文字记录的类别。",
                     _obj({}), lambda: self.list_sources()),
            ToolSpec("get_profile", "患者基本档案(年龄段、性别、已知疾病、设备等)与既往背景。", _obj({}),
                     lambda: self.get_profile()),
            ToolSpec("get_series", f"读一条监测流(或记录里按日的表)在 [from_day, to_day] 内的逐点数据;一次最多 {mp} 点,"
                                   "超出时返回最近的部分并给出往前翻页的 to_day。",
                     _obj({"name": {"type": "string", "description": "监测流或表的名称,见 list_sources"}, **_WIN},
                          ("name",)), self.get_series),
            ToolSpec("get_labs", f"读化验结果(含采血时刻与随附说明);name 省略 = 全部项目。一次最多 {mp} 条。",
                     _obj({"name": {"type": "string", "description": "化验项目名称(可省略)"}, **_WIN}),
                     self.get_labs),
            ToolSpec("get_meds", f"读用药记录(处方、剂量变化、取药、停用及原因)。一次最多 {mp} 条。",
                     _obj(dict(_WIN)), self.get_meds),
            ToolSpec("get_notes", f"读患者自述、生活事件与就诊记录等文字条目(可按类别与时间窗筛选)。一次最多 {mp} 条。",
                     _obj({**_WIN, "kind": {"type": "string", "description": "类别,见 list_sources 的 notes.kinds 与 records"}}),
                     self.get_notes),
        ]
        if self.sections:
            out.append(ToolSpec("get_record", f"按名称读本次记录的某一部分(名称见 list_sources 的 records);按日的部分可加时间窗,一次最多 {mp} 条。",
                                _obj({"section": {"type": "string", "description": "记录部分的名称"}, **_WIN}, ("section",)),
                                self.get_record))
        if self.protocols:
            out.append(ToolSpec("get_protocol", "按名称、主题或关键词检索本院规程，返回适用条件、排除条件、完整步骤、例外与出处；省略条件返回全部。",
                                _obj({k: {"type": "string"} for k in ("name", "topic", "keyword")}), self.get_protocol))
        return out

    def specs(self) -> list[ToolSpec]:
        return self.common_specs() + (self.adapter.extra_specs(self) if self.adapter is not None else [])


def call_tool(specs: dict[str, ToolSpec], name: str, args) -> tuple[dict, str | None]:
    """Run one call; returns `(result, error_kind)`. Bad name or arguments return an error result
    the model sees (and the trace counts), never an exception."""
    s = specs.get(str(name))
    if s is None:
        return {"error": f"没有名为 {name} 的工具", "tools": sorted(specs)}, "unknown_tool"
    if isinstance(args, str):
        try:
            args = json.loads(args or "{}")
        except ValueError:
            return {"error": "参数不是合法 JSON"}, "bad_arguments"
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return {"error": "参数必须是对象"}, "bad_arguments"
    allowed = set((s.params.get("properties") or {}))
    extra = sorted(set(args) - allowed)
    missing = [k for k in s.params.get("required") or () if k not in args]
    if extra or missing:
        return {"error": "参数不符合工具定义", "unexpected": extra, "missing": missing}, "bad_arguments"
    try:
        return s.fn(**args), None
    except (TypeError, ValueError) as e:
        return {"error": f"参数取值无效: {e}"}, "bad_arguments"


class QueryLeak(RuntimeError):
    """A tool result or the question text failed the result guard; the cell is `ABORT(leak)`."""

    def __init__(self, violations: list[str]):
        super().__init__("; ".join(violations[:3]))
        self.violations = list(violations)


class ResultGuard:
    """What a tool result must never carry: a day after T, a kernel forbidden token (substring, as
    the leakage probe reads), a pack gold word (whole token for ASCII, as the pack's emission gate
    reads), a gold number. `words`/`numbers` come from the verifier side (the pack adapter) and
    are only compared against, never handed to the store. The question text names the answer
    classes by design, so it is checked by `for_prompt()` (kernel tokens and gold numbers only)."""

    def __init__(self, T: int, words=(), numbers=()):
        from haenv_kernel.build import FORBIDDEN_TOKENS
        self.T = int(T)
        self.kernel = tuple(w.lower() for w in FORBIDDEN_TOKENS)
        self.words = tuple(dict.fromkeys(w for w in words if w))
        self.numbers = tuple(n for n in numbers if n)

    def for_prompt(self) -> "ResultGuard":
        return ResultGuard(self.T, (), self.numbers)

    def _days(self, obj, path="") -> list[str]:
        bad = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in DAY_KEYS and isinstance(v, (int, float)) and not isinstance(v, bool) and v > self.T:
                    bad.append(f"future_day:{path}{k}={v}")
                bad += self._days(v, f"{path}{k}.")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                bad += self._days(v, f"{path}{i}.")
        return bad

    def check(self, obj) -> list[str]:
        from haenv.world.validation.shared_audit import number_hits
        from haenv.world.validation.shared_audit import word_hits
        bad = self._days(obj) if not isinstance(obj, str) else []
        text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
        bad += [f"forbidden_token:{w}" for w in self.kernel if w in text.lower()]
        if isinstance(obj, dict) and obj.get("protocols"):
            from haenv.world.protocols.library import catalog
            titles = [{"protocol_id": d["protocol_id"], "name": d["name"], "topics": d["topics"]} for d in catalog()]
            if obj["protocols"] == titles:
                text = json.dumps({k: v for k, v in obj.items() if k != "protocols"}, ensure_ascii=False)
        # A canonical, patient-independent policy legitimately names all actions.
        # Exempt only exact registry documents, never arbitrary text with a policy tag.
        if isinstance(obj, dict) and obj.get("resourceType") == "Bundle":
            from haenv.world.protocols.library import catalog
            docs = catalog() if any(fhir.source_of(e) == "protocol_library" for e in obj.get("entry") or ()) else []
            entries = [e for e in obj.get("entry") or () if not (
                fhir.source_of(e) == "protocol_library" and fhir.unshape(e) in docs)]
            text = json.dumps({**obj, "entry": entries}, ensure_ascii=False)
        bad += [f"gold_word:{w}" for w in sorted(set(word_hits(text, self.words)))]
        bad += [f"gold_number:{n}" for n in number_hits(text, self.numbers)]
        return bad
