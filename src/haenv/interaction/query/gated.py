"""M2 purchases inside the query loop: the bridge between the native tool conversation and the gated
geometry's purchase rounds (`haenv/interaction/gated.py:run_gated`).

`run_gated` stays the loop that owns the purchases: per round it shows the solver the lean payload
plus what was bought, the menu, the budget and what was refused, takes `{"queries": [...]}` or an
answer, and buys through the Gatekeeper at the menu's price and kind, refusing past the budget. In
query mode each of its rounds is one call of `solve_gated_round`, which continues one native
conversation:

* the first round starts the conversation (the pack's question on the query view, the query
  protocol and the purchase protocol `haenv/m2/query.py:PROTOCOL`);
* every later round first returns the results of the purchases the model made at the end of the
  previous round, read off what `run_gated` revealed (`payload.longitudinal_data`), refused
  (`truncated_last_round`) or could not provide (`unavailable_results`);
* the model then reads freely (the common tools over the lean payload and what was bought,
  `get_menu`) until it orders (`order_test`), which ends the round with those purchases as the
  round's queries, or answers, which ends the cell with that answer.

On the last purchase round `order_test` is not offered; the query limits (`config.query`) bound the
turns and calls of the whole cell, as in every pack. Budget, menu, billing, refusal, the tool track
(T1-T4) and the M2 atoms are the gated geometry's, computed by the same code on the same trace.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import hashlib
import json
import time

from haenv.interaction.query.loop import FORCE_FINAL
from haenv.interaction.query.loop import QueryCall
from haenv.interaction.query.loop import QueryTrace
from haenv.interaction.query.loop import Turn
from haenv.interaction.query.loop import _n_rows  # noqa: F401
from haenv.interaction.query.tools import QueryLeak
from haenv.interaction.query.tools import RecordStore
from haenv.interaction.query.tools import ResultGuard
from haenv.interaction.query.tools import ToolSpec
from haenv.interaction.query.tools import _obj
from haenv.interaction.query.tools import call_tool

PURCHASE = "order_test"


def _menu(gc: dict) -> dict:
    items = [{"target": str(i["target"]), "kind": str(i["kind"]), "cost": i.get("cost"),
              "section": "tests" if i.get("is_test") else "signals"} for i in gc.get("menu") or ()]
    budget, spent = float(gc.get("budget") or 0), float(gc.get("spent") or 0)
    return {"budget": budget, "spent": spent, "remaining": round(budget - spent, 2),
            "purchase_round": gc.get("round"), "max_purchase_rounds": gc.get("max_rounds"), "items": items,
            **({"unavailable": dict(gc["unavailable_results"])} if gc.get("unavailable_results") else {})}


def _purchase_result(target: str, store: RecordStore, gc: dict) -> dict:
    """What one purchase returned, read off the round `run_gated` just ran: the bought stream or
    test result (as `get_series` shows it), else unavailable, refused past the budget, or nothing."""
    if store.streams.get(target):
        return {"target": target, "ordered": True, **store.get_series(target)}
    if target in (gc.get("unavailable_results") or {}):
        return {"target": target, "ordered": True, "found": False,
                "result": "环境无法提供可解释结果(不是正常或阴性,不能据此排除疾病)"}
    if target in (gc.get("truncated_last_round") or ()):
        return {"target": target, "ordered": False, "found": False, "result": "超出余额,未执行",
                "remaining": round(float(gc.get("budget") or 0) - float(gc.get("spent") or 0), 2)}
    return {"target": target, "ordered": True, "found": False, "result": "无结果"}


def _specs(store: RecordStore, gc: dict, can_order: bool) -> list[ToolSpec]:
    specs = store.specs() + [ToolSpec("get_menu", "可购买的项目(监测信号与检查)、各自价格、预算、已花费与余额。",
                                      _obj({}), lambda: _menu(gc))]
    if can_order:
        specs.append(ToolSpec(PURCHASE, "购买菜单上的一项(按菜单价格计费;超出余额不执行)。结果在本轮结束后返回。",
                              _obj({"target": {"type": "string", "description": "菜单上的名称"}}, ("target",)),
                              lambda target: {}))
    return specs


def _output(solver, payload, st: dict, text: str, usages: list, finish):
    from haenv.runtime.solvers import _drain_backoff
    from haenv.runtime.solvers import _drain_requests
    from haenv.runtime.solvers import _to_output
    from haenv_kernel.solver import _extract_json
    from haenv.interaction.query.solver import _sum_usage
    try:
        data = _extract_json(text) if text.strip() else {}
    except Exception:                                                 # noqa: BLE001 -- unparseable is a recorded outcome
        data = {}
    out = _to_output(data if isinstance(data, dict) else {}, payload)
    out._raw_text, out._prompt_mode = text, getattr(solver, "prompt_mode", "default")
    # no `_prompt_sha`: a query round is never stored as a reusable gated round (the conversation
    # is resumed from its receipts instead)
    out._probe_id, out._prompt_sha, out._prompt_sha_full = st["pid"], None, st["sha"]
    out._usage, out._finish, out._max_tokens = _sum_usage(usages), finish, getattr(solver, "max_tokens", None)
    out._failed_attempts = list(getattr(solver, "_failed_turns", []) or [])
    solver._failed_turns = []
    out._rate_limit_retries, out._requests = _drain_backoff(solver), _drain_requests(solver)
    out._latency_s, out._reasoning_chars = round(time.time() - st["t0"], 2), None
    return out


def solve_gated_round(solver, payload):
    from haenv.m2.query import ADAPTER
    from haenv.m2.query import PROTOCOL
    from haenv.interaction.query.solver import QUERY_SYSTEM
    from haenv.interaction.query.solver import QueryView
    from haenv.interaction.query.solver import UNANSWERED
    from haenv.interaction.query.solver import protocol_text
    from haenv.interaction.query.solver import render_question
    gc, q = dict(solver.gated_context), solver.query
    T = int(payload.prediction_context["prediction_time_T"])
    store = RecordStore(payload, T, ADAPTER, max_points=q["max_points"])
    guard = getattr(solver, "query_guard", None) or ResultGuard(T)
    st = getattr(solver, "_gq", None)
    solver.on_case(ADAPTER, payload)
    if st is None or int(gc.get("round") or 1) == 1:
        question, pid = render_question(solver, QueryView(payload, ADAPTER, store))
        question += protocol_text(q, PROTOCOL)
        v = guard.for_prompt().check(question) + guard.for_prompt().check(QUERY_SYSTEM)
        if v:
            raise QueryLeak(v)
        st = solver._gq = {"msgs": [{"role": "user", "content": question}], "pid": pid,
                           "sha": hashlib.sha256(question.encode()).hexdigest(), "pending": [], "n_calls": 0,
                           "trace": QueryTrace(system=QUERY_SYSTEM), "answered": None, "t0": time.time()}
        st["trace"].messages = st["msgs"]
        solver._failed_turns = []
    tr: QueryTrace = st["trace"]
    for p in st["pending"]:
        res = _purchase_result(p["target"], store, gc)
        bad = guard.check(res)
        if bad:
            raise QueryLeak(bad)
        body = json.dumps(res, ensure_ascii=False)
        tr.calls.append(QueryCall(p["round"], PURCHASE, {"target": p["target"]}, None, res.get("found"), len(body),
                                  _n_rows(res)))
        st["msgs"].append({"role": "tool", "tool_call_id": p["id"], "name": PURCHASE, "content": res})
    st["pending"] = []
    if st["answered"] is not None:
        return _output(solver, payload, st, st["answered"], [], None)
    usages, finish = [], None
    can_order = int(gc.get("round") or 1) < int(gc.get("max_rounds") or 1)
    while True:
        final = tr.rounds_used >= q["max_rounds"] or st["n_calls"] >= q["max_calls"]
        specs = _specs(store, gc, can_order and not final)
        if final:
            st["msgs"].append({"role": "user", "content": FORCE_FINAL})
        turn: Turn = solver.chat(QUERY_SYSTEM, st["msgs"], specs, "none" if final else "auto")
        tr.rounds_used += 1
        tr.usage.append(turn.usage)
        tr.finish.append(turn.finish)
        usages.append(turn.usage)
        finish = turn.finish
        calls = list(turn.tool_calls or ())
        st["msgs"].append({"role": "assistant", "content": turn.content or "",
                           **({"tool_calls": [{"id": c.get("id") or f"g{tr.rounds_used}.{i}", "name": str(c.get("name")),
                                               "arguments": c.get("arguments")} for i, c in enumerate(calls)]}
                              if calls else {}),
                           **({"native": turn.native} if turn.native else {})})
        if final and calls:
            tr.protocol_errors += 1
            tr.stop, st["answered"] = "unanswered", UNANSWERED
            break
        if not calls:
            tr.stop, st["answered"] = ("forced_final" if final else "answered"), (turn.content or "")
            break
        by_name = {s.name: s for s in specs}
        for i, c in enumerate(calls):
            cid = c.get("id") or f"g{tr.rounds_used}.{i}"
            args = c.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args or "{}")
                except ValueError:
                    pass
            if (str(c.get("name")) == PURCHASE and PURCHASE in by_name and st["n_calls"] < q["max_calls"]
                    and isinstance(args, dict) and isinstance(args.get("target"), str) and set(args) == {"target"}):
                st["n_calls"] += 1
                st["pending"].append({"id": cid, "target": args["target"], "round": tr.rounds_used})
                continue
            if st["n_calls"] >= q["max_calls"]:
                res, err = {"error": "调用次数已用完"}, "over_budget"
            else:
                res, err = call_tool(by_name, c.get("name"), args)
                st["n_calls"] += 1
                if err:
                    tr.protocol_errors += 1
            bad = guard.check(res)
            if bad:
                raise QueryLeak(bad)
            body = json.dumps(res, ensure_ascii=False)
            tr.calls.append(QueryCall(tr.rounds_used, str(c.get("name")), c.get("arguments"), err,
                                      res.get("found") if isinstance(res, dict) else None, len(body), _n_rows(res)))
            st["msgs"].append({"role": "tool", "tool_call_id": cid, "name": str(c.get("name")), "content": res})
        if st["pending"]:
            menu_kind = {str(i["target"]): str(i["kind"]) for i in gc.get("menu") or ()}
            text = json.dumps({"queries": [{"kind": menu_kind.get(p["target"], "test"), "target": p["target"]}
                                           for p in st["pending"]], "commit": False}, ensure_ascii=False)
            solver._last_query = ({**tr.as_dict(), "pack": "m2", "T": T, "limits": dict(q)}, store)
            return _output(solver, payload, st, text, usages, finish)
    solver._last_query = ({**tr.as_dict(), "pack": "m2", "T": T, "limits": dict(q)}, store)
    return _output(solver, payload, st, st["answered"], usages, finish)
