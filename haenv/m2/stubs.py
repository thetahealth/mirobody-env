"""Capability-known offline stubs for the M2 zero-cost validation (spec section 8).

All stubs speak the gated protocol (`{"queries": [...]}` then an answer) in round 1 and answer the
revision block in round 2. Oracle-side stubs read the case gold from `GOLD`, filled by
`tools/m2_stub_ladder.py` from the verifier payload (the same arrangement as
`evaluate.ORACLE_GOLD_TESTS` for the v1.0.1 oracle stubs); no stub ever calls a model.

* `LadderStub(p)`  -- every answer atom (each gold line, each test, quantity, review flag,
  abstention, frame field, each purchase, revision decision, each round-2 line) takes the gold
  value with probability p, else a fixed wrong substitute (a registered rival or the distractor's
  diagnosis; a non-gold catalogue test; the negated flag; a decoy purchase). Draws are keyed on
  (case, atom, p).
* `SplitStub("buy")`  -- buys every key signal, lists every gold line in round 1, keeps in round 2.
  `SplitStub("read")` -- buys nothing, lists only the first gold line in round 1, lists every line
  after the push and changes. Tests, quantity, review and abstention are gold in both.
* `RandomStub(seed)` -- every atom uniform over its legal values.
* `ConstStub(kind)` -- question-blind constants (M1 blind optimum list, cheapest-first and
  random-menu buyers, always keep / change / review / no review / abstain / not abstain).
* `PriorStub` (revision r2-1) -- frame- or frame x sex-conditioned constants from pack-level
  frequencies: the reviewer's four frame-blind stubs and the leave-one-out / in-sample frame x sex
  priors (Z-3).
* `FrameBuyStub` (revision r2-3) -- the Z-4 pair (reading what was bought): buy the frame's
  components' key signals and list what the readings indicate, against the same stub without buying.
* `BuyLadderStub` (revision r3-D3) -- the Z-5 pairs (choosing what to buy): `LadderStub(p)` with the
  case's gold-line key signals bought (`D_p`) or the frame's whole candidate set bought (`W_p`).

`SplitStub` (S_buy / S_read) is an implementation self-check: S_read is built to list one line of
two, so their gap is set by construction and no gate reads it (revision r2-3).
"""
from __future__ import annotations

import hashlib
import json
import random

#: case_id -> gold bundle (see `tools/m2_stub_ladder.py:gold_bundle`)
GOLD: dict[str, dict] = {}
#: batch-level constants for the question-blind stubs (most frequent gold values)
BLIND: dict = {}


def _u(*parts) -> float:
    h = hashlib.blake2b("|".join(str(p) for p in parts).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") / float(1 << 64)


def _cid(payload) -> str:
    return str(getattr(payload, "case_id", "") or "")


def _emit(payload, data: dict):
    from ..evaluate import _to_output
    out = _to_output(data, payload)
    out._raw_text = json.dumps(data, ensure_ascii=False)          # noqa: SLF001
    return out


_ACTION_FOR_URGENCY = {"🔴": "A5", "🟠": "A4", "🟡": "A3", "🟢": "A1"}
_URG = ("🔴", "🟠", "🟡", "🟢")
_MED = ("investigate_first", "uptitrate", "keep", "downtitrate")


class _Base:
    """Two-round protocol shared by every stub. Subclasses define `pick(atom, gold, wrong, legal)`
    and may override `purchases(g, ctx)` and `r1_lines(g)`."""

    name = "m2_stub"

    def __init__(self):
        self._bought_done = False

    # ---- policy hooks
    def pick(self, cid, atom, gold, wrong, legal):
        raise NotImplementedError

    def purchases(self, g, ctx) -> list[str]:
        out = []
        for f, tgt in g["key_targets"].items():
            v = self.pick(g["cid"], f"buy:{f}", tgt, g["decoy_for"][f], [tgt, g["decoy_for"][f], None])
            if v:
                out.append(v)
        return out

    # ---- answer pieces
    def _lines_r1(self, g) -> list[bool]:
        return [self.pick(g["cid"], f"dx:{i}", True, False, [True, False]) for i in range(len(g["threads"]))]

    def _dx_entries(self, g, listed: list[bool], tag: str) -> list[dict]:
        ents = []
        wrongs = list(g["wrong_names"])
        if tag == "2" and g.get("distractor_name"):
            wrongs = [g["distractor_name"]] + wrongs        # round 2: the pushed distractor's diagnosis
        for i, t in enumerate(g["threads"]):
            if listed[i]:
                ents.append(t["say"])
            else:
                ents.append(wrongs[i % len(wrongs)] if wrongs else "原因待查")
        if not g["threads"]:
            ents = list(g["wrong_names"][:1]) or ["原因待查"]
        return [{"rank": k + 1, "diagnosis": d, "certainty": "probable",
                 "supporting_evidence": g["ev"][:1], "ruled_out_by": None} for k, d in enumerate(ents)]

    def answer(self, g, listed: list[bool], round_tag: str) -> dict:
        cid = g["cid"]
        join = self.pick(cid, "join", g["join"], g["join_wrong"], ["unified", "comorbidity", "independent"])
        tests = []
        for k, t in enumerate(g["tests"]):
            tests.append(self.pick(cid, f"test:{k}", t, g["test_wrong"][k], [t, g["test_wrong"][k]]))
        review = self.pick(cid, f"review{round_tag}", g["warranted"], not g["warranted"], [True, False])
        suff_gold = "insufficient_data" if g["insufficient"] else "sufficient"
        suff = self.pick(cid, "abst", suff_gold,
                         "sufficient" if g["insufficient"] else "insufficient_data",
                         ["sufficient", "insufficient_data"])
        urg_g = g.get("urgency") or "🟡"
        urg_w = _URG[(_URG.index(urg_g) + 2) % 4] if urg_g in _URG else "🟢"
        urg = self.pick(cid, "urgency", urg_g, urg_w, list(_URG))
        act_cls = _ACTION_FOR_URGENCY.get(urg, "A3")
        data = {"differential": self._dx_entries(g, listed, round_tag), "join_type": join,
                "join_reason": "stub", "join_evidence": g["ev"][:2],
                "tests_to_order": [t for t in tests if t], "referral_specialty": [],
                "forecast": {"risk_category": "indeterminate"},
                "action": {"selected_action_class": act_cls, "specific_action": "按鉴别结果安排检查与复诊",
                           "what_not_to_do": ["不自主改药"], "clinician_review_required": bool(review),
                           "followup_interval": "7d"},
                "data_quality": {"data_sufficiency": suff, "signal_quality": {}},
                "cited_evidence": g["ev"][:2]}
        q = g.get("quant")
        if q is not None:
            data["quant_answer"] = self.pick(cid, "quant", q["truth"], q["wrong"], [q["truth"], q["wrong"]])
        fr = g.get("frame")
        if fr == "F1":
            data["urgency"] = urg
        elif fr == "F2":
            mg = g["med_gold"]
            mw = "uptitrate" if mg == "investigate_first" else "investigate_first"
            data["treatment_plan"] = {"direction": self.pick(cid, "med", mg, mw, list(_MED)), "reason": "stub"}
        elif fr == "F3" and g.get("rcv_gold") is not None:
            data["followup"] = {"indicator": g.get("rcv_stream"),
                                "exceeds_rcv": self.pick(cid, "rcv", g["rcv_gold"], not g["rcv_gold"], [True, False]),
                                "explained_by": "stub", "next_step": "复查"}
        return data

    def revise(self, g, r1: dict, items: list[dict]) -> dict:
        cid = g["cid"]
        l1 = getattr(self, "_l1", [True] * len(g["threads"]))
        cls_w = bool(g["threads"]) and not all(l1)
        gold_dec = "change" if cls_w else "keep"
        dec = self.pick(cid, "decision", gold_dec, "keep" if gold_dec == "change" else "change", ["keep", "change"])
        l2 = [self.pick(cid, f"r2dx:{i}", True, False, [True, False]) for i in range(len(g["threads"]))]
        keys = [it["item_id"] for it in g["push_items"] if it["role"] == "key"]
        did = next((it["item_id"] for it in g["push_items"] if it["role"] == "distractor"), None)
        good_because = keys if cls_w else []
        bad_because = ([did] if did else keys[:1])
        because = self.pick(cid, "because", good_because, bad_because, [good_because, bad_because])
        ans = self.answer(g, l2, "2")
        ans["revision"] = {"decision": dec, "changed_fields": ["differential"] if dec == "change" else [],
                           "because": list(because or [])}
        return ans

    # ---- protocol
    def solve(self, payload):
        cid = _cid(payload)
        g = GOLD[cid]
        rc = getattr(self, "m2_revision_context", None)
        if rc:
            return _emit(payload, self.revise(g, rc.get("round1") or {}, rc.get("items") or []))
        ctx = getattr(self, "gated_context", None) or {}
        if ctx and not self._bought_done:
            self._bought_done = True
            buys = [b for b in self.purchases(g, ctx) if b]
            if buys:
                kinds = {str(i["target"]): str(i["kind"]) for i in (ctx.get("menu") or [])}
                qs = [{"kind": kinds.get(b, "basic_lab"), "target": b} for b in dict.fromkeys(buys)]
                return _emit_queries(payload, qs)
        self._l1 = self.r1_lines(g)
        return _emit(payload, self.answer(g, self._l1, "1"))

    def r1_lines(self, g) -> list[bool]:
        return self._lines_r1(g)


def _emit_queries(payload, qs):
    from ..evaluate import _to_output
    out = _to_output({}, payload)
    out._raw = {}                                                    # noqa: SLF001
    out._raw_text = json.dumps({"queries": qs, "commit": False}, ensure_ascii=False)  # noqa: SLF001
    return out


class LadderStub(_Base):
    def __init__(self, p: float):
        super().__init__()
        self.p = float(p)
        self.name = f"S_p{p:.1f}"

    def pick(self, cid, atom, gold, wrong, legal):
        return gold if _u(cid, atom, f"{self.p:.2f}") < self.p else wrong


class SplitStub(_Base):
    def __init__(self, kind: str):
        super().__init__()
        assert kind in ("buy", "read")
        self.kind = kind
        self.name = f"S_{kind}"

    def pick(self, cid, atom, gold, wrong, legal):
        return gold

    def purchases(self, g, ctx):
        return list(g["key_targets"].values()) if self.kind == "buy" else []

    def r1_lines(self, g):
        n = len(g["threads"])
        return [True] * n if self.kind == "buy" else [i == 0 for i in range(n)]

    def revise(self, g, r1, items):
        n = len(g["threads"])
        keys = [it["item_id"] for it in g["push_items"] if it["role"] == "key"]
        if self.kind == "buy" or n <= 1 and all(self._l1):
            ans = self.answer(g, [True] * n, "2")
            ans["revision"] = {"decision": "keep", "changed_fields": [], "because": []}
            return ans
        # S_read: list the lines whose key signals appear in the push block
        pushed = {it["fid"] for it in items if "fid" in it} or {it["fid"] for it in g["push_items"]
                                                                 if it["item_id"] in {x["item_id"] for x in items}}
        l2 = [any(k in pushed for k in t["key_signals"]) or not t["key_signals"] for t in g["threads"]]
        ans = self.answer(g, l2, "2")
        ans["revision"] = {"decision": "change", "changed_fields": ["differential"], "because": keys}
        return ans


class RandomStub(_Base):
    def __init__(self, seed: int):
        super().__init__()
        self.seed = int(seed)
        self.name = f"S_rand{seed:02d}"

    def pick(self, cid, atom, gold, wrong, legal):
        r = random.Random(f"{cid}|{atom}|{self.seed}")
        return legal[r.randrange(len(legal))]


class ConstStub(_Base):
    KINDS = ("blind_opt", "cheapest_first", "random_menu", "keep", "change",
             "review", "noreview", "abstain", "noabstain")

    def __init__(self, kind: str):
        super().__init__()
        assert kind in self.KINDS, kind
        self.kind = kind
        self.name = f"C_{kind}"

    def pick(self, cid, atom, gold, wrong, legal):
        raise AssertionError("constant stubs do not draw")

    def purchases(self, g, ctx):
        menu = [i for i in (ctx.get("menu") or [])]
        if self.kind == "cheapest_first":
            menu = sorted(menu, key=lambda i: (float(i.get("cost") or 0), str(i["target"])))
            out, spent = [], 0.0
            for i in menu:
                c = float(i.get("cost") or 0)
                if spent + c > float(ctx.get("budget") or 0):
                    break
                out.append(str(i["target"]))
                spent += c
            return out
        if self.kind == "random_menu":
            r = random.Random(f"{g['cid']}|random_menu")
            r.shuffle(menu)
            return [str(i["target"]) for i in menu[:6]]
        return []

    def r1_lines(self, g):
        return []

    def answer(self, g, listed, round_tag):
        b = BLIND
        data = {"differential": [{"rank": k + 1, "diagnosis": d, "certainty": "probable",
                                  "supporting_evidence": g["ev"][:1], "ruled_out_by": None}
                                 for k, d in enumerate(b["dx_list"])],
                "join_type": b["join"], "join_reason": "constant", "join_evidence": g["ev"][:2],
                "tests_to_order": list(b["tests"]), "referral_specialty": [],
                "forecast": {"risk_category": "indeterminate"},
                "action": {"selected_action_class": "A3", "specific_action": "按常规安排检查与复诊",
                           "what_not_to_do": ["不自主改药"],
                           "clinician_review_required": (False if self.kind == "noreview" else True),
                           "followup_interval": "7d"},
                "data_quality": {"data_sufficiency": ("insufficient_data" if self.kind == "abstain"
                                                      else "sufficient"), "signal_quality": {}},
                "cited_evidence": g["ev"][:2]}
        if g.get("quant") is not None:
            data["quant_answer"] = b["quant"].get(g["quant"]["kind"], g["quant"]["wrong"])
        if g.get("frame") == "F1":
            data["urgency"] = b["urgency"]
        elif g.get("frame") == "F2":
            data["treatment_plan"] = {"direction": b["med"], "reason": "constant"}
        elif g.get("frame") == "F3":
            data["followup"] = {"indicator": g.get("rcv_stream"), "exceeds_rcv": b["rcv"],
                                "explained_by": "constant", "next_step": "复查"}
        return data

    def revise(self, g, r1, items):
        ans = self.answer(g, [], "2")
        if self.kind == "change":
            ans["revision"] = {"decision": "change", "changed_fields": ["differential"],
                               "because": [i["item_id"] for i in items[:2]]}
        else:
            ans["revision"] = {"decision": "keep", "changed_fields": [], "because": []}
        return ans


def ladder(ps=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)):
    return [(f"S_p{p:.1f}", (lambda p=p: LadderStub(p))) for p in ps]


def split():
    return [("S_buy", lambda: SplitStub("buy")), ("S_read", lambda: SplitStub("read"))]


def randoms(n=20):
    return [(f"S_rand{s:02d}", (lambda s=s: RandomStub(s))) for s in range(n)]


def consts():
    return [(f"C_{k}", (lambda k=k: ConstStub(k))) for k in ConstStub.KINDS]


# ------------------------------------------------------------------ frame-prior stubs (revision r2-1)
# Question-blind stubs that condition on what the prompt shows before any case content: the frame
# (F0 has no `m2_frame`; F1-F3 carry a scene and a frame ask) and, optionally, the patient's sex.
# Every constant is a pack-level frequency over the OTHER cases' gold (`loo=True`) or over all cases
# of the cell (`loo=False`, the in-sample upper bound); no stub reads the case's own gold.
from collections import Counter as _Counter  # noqa: E402


def _frame(g) -> str:
    return g.get("frame") or "F0"


def _cell_key(g, by_sex: bool):
    return (_frame(g), g.get("sex")) if by_sex else (_frame(g),)


def _cell(g, by_sex: bool, loo: bool) -> list[dict]:
    k = _cell_key(g, by_sex)
    out = [x for x in GOLD.values() if _cell_key(x, by_sex) == k and not (loo and x["cid"] == g["cid"])]
    if not out and by_sex:                         # an empty sex cell falls back to the frame
        return _cell(g, False, loo)
    return out


def _ranked(cnt: _Counter) -> list:
    return [n for n, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], str(kv[0])))]


def prior_table(g, *, by_sex: bool = False, loo: bool = False) -> dict:
    cell = _cell(g, by_sex, loo)
    names = _Counter(t["say"] for x in cell for t in x["threads"])
    comps = _Counter(t.get("component") for x in cell for t in x["threads"] if t.get("component"))
    tests = _Counter(t for x in cell for t in x["tests"])
    joins = _Counter(x["join"] for x in cell if not x["insufficient"])
    w = [bool(x["warranted"]) for x in cell]
    quant: dict = {}
    for x in cell:
        if x.get("quant"):
            quant.setdefault(x["quant"]["kind"], _Counter())[json.dumps(x["quant"]["truth"])] += 1
    return {"dx_list": _ranked(names), "components": _ranked(comps), "tests": _ranked(tests),
            "join": (_ranked(joins) or ["unified"])[0], "warranted_share": (sum(w) / len(w)) if w else None,
            "quant": {k: json.loads(_ranked(c)[0]) for k, c in quant.items()}, "n": len(cell)}


class PriorStub(ConstStub):
    """A frame- (or frame x sex-) conditioned constant. `review`: "above_mean" answers True when the
    cell's warranted share is above the pack's,
    "majority" answers the cell's majority class; None keeps C_blind_opt's True."""

    def __init__(self, name, *, dx_k=5, tests_k=6, by_sex=False, loo=False, review=None,
                 frame_dx=True, frame_tests=True, frame_quant=False):
        _Base.__init__(self)
        self.kind = "blind_opt"
        self.name = name
        self.dx_k, self.tests_k, self.by_sex, self.loo, self.review = dx_k, tests_k, by_sex, loo, review
        self.frame_dx, self.frame_tests, self.frame_quant = frame_dx, frame_tests, frame_quant

    def answer(self, g, listed, round_tag):
        d = super().answer(g, listed, round_tag)
        tab = prior_table(g, by_sex=self.by_sex, loo=self.loo)
        if self.frame_dx:
            d["differential"] = [{"rank": k + 1, "diagnosis": n, "certainty": "probable",
                                  "supporting_evidence": g["ev"][:1], "ruled_out_by": None}
                                 for k, n in enumerate(tab["dx_list"][:self.dx_k])]
            d["join_type"] = tab["join"]
        if self.frame_tests:
            d["tests_to_order"] = list(tab["tests"][:self.tests_k])
        if self.frame_quant and g.get("quant") is not None and g["quant"]["kind"] in tab["quant"]:
            d["quant_answer"] = tab["quant"][g["quant"]["kind"]]
        if self.review is not None and tab["warranted_share"] is not None:
            if self.review == "above_mean":
                allw = [bool(x["warranted"]) for x in GOLD.values()]
                rv = tab["warranted_share"] > sum(allw) / len(allw)
            else:
                rv = tab["warranted_share"] >= 0.5
            d["action"]["clinician_review_required"] = bool(rv)
        return d


def frame_gate_stubs():
    """Z-3 (ii)-(iv) of PREREG r2 A4: the reviewer's four frame-blind stubs, the leave-one-out frame
    and frame x sex priors (gated), and the in-sample frame x sex prior (reported only)."""
    return [
        ("G_frame_dx", lambda: PriorStub("G_frame_dx", frame_tests=False)),
        ("G_frame_tests", lambda: PriorStub("G_frame_tests", frame_dx=False)),
        ("G_frame_review", lambda: PriorStub("G_frame_review", frame_dx=False, frame_tests=False, review="above_mean")),
        ("G_frame_all", lambda: PriorStub("G_frame_all", review="above_mean")),
        ("G_frame_loo", lambda: PriorStub("G_frame_loo", loo=True, review="majority", frame_quant=True)),
        ("G_framesex_loo", lambda: PriorStub("G_framesex_loo", by_sex=True, loo=True, review="majority", frame_quant=True)),
        ("G_framesex_in", lambda: PriorStub("G_framesex_in", by_sex=True, review="majority", frame_quant=True)),
    ]


# ------------------------------------------------------------------ Z-4 buy-and-read pair (revision r2-3)
_FLAG_DIR = {"high": ("H",), "low": ("L",), "positive": ("POS", "阳")}


def _abnormal_in_direction(flag, direction) -> bool:
    f = str(flag or "").strip().upper()
    return bool(f) and any(f.startswith(x) for x in _FLAG_DIR.get(str(direction), ()))


def key_findings_with_direction(component: str) -> list[tuple[str, str]]:
    """(finding id, declared direction) of a component's key signals (`core._key_findings_of`)."""
    from .core import _key_findings_of, _profiles
    prof = {str(p.get("id")): str(p.get("direction")) for p in ((_profiles().get(component) or {}).get("findings") or ())}
    return [(f, prof.get(f, "")) for f in _key_findings_of(component)]


class FrameBuyStub(_Base):
    """Z-4 pair (PREREG r2 B2). Tests, quantity, review and frame atoms are gold for both;
    `buy=False` lists the frame-prior names; `buy=True` buys the key-signal menu items of the frame's
    components in frame-prior order within the budget, then lists the components whose key findings
    came back abnormal in their declared direction (most such findings first), then the frame-prior
    names. Only the returned readings and the registries are read, never the case's gold lines."""

    def __init__(self, buy: bool):
        super().__init__()
        self.buy = bool(buy)
        self.name = "S_frame_buyread" if buy else "S_frame_nobuy"
        self._revealed: dict = {}

    def pick(self, cid, atom, gold, wrong, legal):
        return gold

    def purchases(self, g, ctx):
        if not self.buy:
            return []
        from .core import menu_target_of
        mt = menu_target_of()
        menu = {str(i["target"]): float(i.get("cost") or 0) for i in (ctx.get("menu") or [])}
        budget, spent, out = float(ctx.get("budget") or 0), float(ctx.get("spent") or 0), []
        for comp in prior_table(g)["components"]:
            for fid, _ in key_findings_with_direction(comp):
                t = mt.get(fid)
                if not t or t not in menu or t in out:
                    continue
                if spent + menu[t] > budget:
                    continue
                out.append(t)
                spent += menu[t]
        return out

    def solve(self, payload):
        ld = getattr(payload, "longitudinal_data", None) or {}
        self._revealed = {k: v for k, v in ld.items() if isinstance(v, list)}
        return super().solve(payload)

    def indicated(self, g) -> list[str]:
        tab = prior_table(g)
        abn: dict = {}
        pts = [p for v in self._revealed.values() for p in v if isinstance(p, dict) and p.get("finding_id")]
        for comp in tab["components"]:
            n = 0
            for fid, direction in key_findings_with_direction(comp):
                if any(p["finding_id"] == fid and _abnormal_in_direction(p.get("flag"), direction) for p in pts):
                    n += 1
            if n:
                abn[comp] = n
        order = {c: i for i, c in enumerate(tab["components"])}
        return sorted(abn, key=lambda c: (-abn[c], order[c]))

    def answer(self, g, listed, round_tag):
        d = super().answer(g, [False] * len(g["threads"]), round_tag)
        tab = prior_table(g)
        say = {}
        for x in GOLD.values():
            for t in x["threads"]:
                if t.get("component"):
                    say.setdefault(t["component"], _Counter())[t["say"]] += 1
        names = [(_ranked(say[c]) or [c])[0] for c in (self.indicated(g) if self.buy else []) if c in say]
        names = list(dict.fromkeys(names + tab["dx_list"]))[:6]
        d["differential"] = [{"rank": k + 1, "diagnosis": n, "certainty": "probable",
                              "supporting_evidence": g["ev"][:1], "ruled_out_by": None}
                             for k, n in enumerate(names)]
        return d


def z4_pair():
    return [("S_frame_buyread", lambda: FrameBuyStub(True)), ("S_frame_nobuy", lambda: FrameBuyStub(False))]


class FrameBlindBuyStub(FrameBuyStub):
    """Reading only (revision r2, not a gate): the buy-and-read purchase policy on top of the frame
    prior for every other field -- tests, quantity, review and join from the frame tables, nothing
    from the case's gold. It reads the case only through the results it buys. It measures how much
    of the composite a buyer gets without choosing purchases from the case (the budget covers the
    frame's candidate panels)."""

    def __init__(self):
        super().__init__(True)
        self.name = "G_frame_buyread"

    def answer(self, g, listed, round_tag):
        base = PriorStub("_", frame_quant=True, review="majority", loo=False)
        d = base.answer(g, [], round_tag)
        d["differential"] = FrameBuyStub.answer(self, g, listed, round_tag)["differential"]
        return d


def blind_buyer():
    return [("G_frame_buyread", lambda: FrameBlindBuyStub())]


class BuyLadderStub(LadderStub):
    """Z-5 (revision r3-D3): `LadderStub(p)` atom for atom, except the purchases -- `D_p` buys every
    key-signal menu item of the case's gold lines (chosen from the case), `W_p` buys the frame's
    candidate components' key signals in frame-prior order within the budget (not from the case)."""

    def __init__(self, p: float, wide: bool):
        super().__init__(p)
        self.wide = bool(wide)
        self.name = f"{'W' if wide else 'D'}_p{p:.1f}"

    def purchases(self, g, ctx):
        if self.wide:
            return FrameBuyStub(True).purchases(g, ctx)
        return list(dict.fromkeys(g["key_targets"].values()))


def z5_pairs(ps=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)):
    return [(f"{k}_p{p:.1f}", (lambda p=p, w=(k == "W"): BuyLadderStub(p, w))) for p in ps for k in ("D", "W")]
