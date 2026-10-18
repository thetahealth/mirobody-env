"""Offline per-cell request/token estimates using frozen exact-judge price metadata."""
from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal
import json
from pathlib import Path

from haenv.scoring.management.pipeline import _outside_sources
from haenv.scoring.management.pipeline import _tasks
from haenv.scoring.management.pipeline import cell_answer
from haenv.scoring.management.pipeline import read_jsonl
from haenv.scoring.management.pipeline import validate
from haenv.scoring.management.pipeline import write_jsonl
from haenv.scoring.management.rubric import canonical
from haenv.scoring.management.rubric import sha256
from haenv.scoring.semantic.parallel import atomic_json
from haenv.runtime.pricing import PriceSchedule


def estimate(out, destination, price_metadata):
    import tiktoken  # optional offline estimator dependency; never guess if unavailable
    from haenv.runtime.native_accounting import NativePrices
    from haenv.scoring.management.judge import empty_answer
    from haenv.scoring.management.judge import request_for
    out, destination, price_metadata = map(lambda p: Path(p).resolve(), (out, destination, price_metadata))
    manifest = validate(out)
    _outside_sources(destination, [Path(s["batch"]) for s in manifest["sources"]])
    if destination == out:
        raise ValueError("Estimate output must be a separate directory")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Estimate output must be new")
    metadata = json.loads(price_metadata.read_text())
    judge = manifest["policy"]["judge"]
    if judge["backend"] == "google":
        schedule = NativePrices(**metadata["native_tariff"])
        if schedule.model != judge["model_id"] or schedule.backend != judge["backend"]:
            raise ValueError("Native price metadata does not describe the sealed judge")
        input_rate = Decimal(schedule.input_per_million) * Decimal(schedule.usd_per_currency_bound) / 1000000
        output_rate = Decimal(schedule.output_per_million) * Decimal(schedule.usd_per_currency_bound) / 1000000
        rates = {"prompt": str(input_rate), "completion": str(output_rate), "request": "0"}
    else:
        schedule = PriceSchedule.from_metadata(metadata, judge["model_id"])
        input_rate, output_rate = schedule.input_rate, schedule.output_rate
        rates = metadata["pricing"]
    # This repository has not registered an OpenAI native tariff. Record that fact
    # instead of applying a Google/Qwen price or calling an endpoint rate 'native'.
    try:
        native = asdict(NativePrices.verified(judge["model_id"], judge["backend"], require_current=False))
    except ValueError:
        native = None
    tokenizer_name = manifest["policy"]["estimate"]["tokenizer"]
    encoding = tiktoken.get_encoding(tokenizer_name)
    policy = manifest["policy"]
    max_attempts = 1 + policy["judge"].get("empty_transport_retries", 0)
    snapshots = {r["key"]: r["snapshot"] for r in read_jsonl(out / "snapshots.jsonl")}
    cells = read_jsonl(out / "cells.jsonl")
    detail = []
    for cell in cells:
        for mode in ("per_item", "per_case"):
            record = {"key": cell["key"], "source": cell["source"], "mode": mode,
                      "rubric_items": len(cell["rubric"]["items"]), "status": cell["status"],
                      "requests": 0, "input_tokens_point": 0, "input_tokens_upper": 0,
                      "output_tokens_point": 0, "output_tokens_upper": 0,
                      "usd_point": "0", "usd_upper": "0", "item_requests": []}
            if cell["status"] == "ready" and not empty_answer(cell_answer(cell)):
                config = {**manifest, "mode": mode}
                point_cost, upper_cost = Decimal(0), Decimal(0)
                for task, items in _tasks(cell, snapshots[cell["snapshot_key"]], config):
                    req = request_for(task)
                    schema = json.dumps(req.response_format, ensure_ascii=False)
                    text = task.prompt + schema
                    # Fixed chat envelope allowance, labelled approximate rather than provider-measured.
                    inp = len(encoding.encode(text, disallowed_special=())) + 16
                    outp = policy["estimate"]["reasoning_tokens_per_request"] + policy["estimate"]["output_tokens_per_item"] * len(items)
                    cap = policy["judge"]["max_tokens"]
                    if outp > cap:
                        raise ValueError("Point output assumption exceeds the registered request cap")
                    bound_in = len(text.encode()) + 2048
                    point = (inp * Decimal(rates["prompt"])
                             + outp * Decimal(rates["completion"])
                             + Decimal(rates.get("request", "0")))
                    upper = schedule.upper_bound(text, cap)
                    record["requests"] += 1
                    for key, value in (("input_tokens_point", inp), ("input_tokens_upper", bound_in),
                                       ("output_tokens_point", outp), ("output_tokens_upper", cap)):
                        record[key] += value
                    point_cost += point; upper_cost += upper
                    record["item_requests"].append({"items": list(task.item_ids), "input_tokens_point": inp,
                                                     "input_tokens_upper": bound_in, "output_tokens_point": outp,
                                                     "output_tokens_upper": cap, "usd_point": str(point), "usd_upper": str(upper)})
                record.update(usd_point=str(point_cost), usd_upper=str(upper_cost))
            # Keep the single-attempt point estimate explicit. The retry envelope
            # assumes every request fails empty/in transport once, not invalid JSON.
            for base, retry_key in (("requests", "requests_with_retry_upper"),
                                    ("input_tokens_upper", "input_tokens_with_retry_upper"),
                                    ("output_tokens_upper", "output_tokens_with_retry_upper")):
                record[retry_key] = record[base] * max_attempts
            record["usd_with_retry_upper"] = str(Decimal(record["usd_upper"]) * max_attempts)
            detail.append(record)
    totals = {}
    for mode in ("per_item", "per_case"):
        totals[mode] = {}
        for group in ("answers", "controls", "all"):
            selected = [r for r in detail if r["mode"] == mode and
                        (group == "all" or ("control" in r["source"]) == (group == "controls"))]
            totals[mode][group] = {"cells": len(selected),
                **{key: sum(r[key] for r in selected) for key in ("requests", "input_tokens_point", "input_tokens_upper", "output_tokens_point", "output_tokens_upper", "requests_with_retry_upper", "input_tokens_with_retry_upper", "output_tokens_with_retry_upper")},
                **{key: str(sum((Decimal(r[key]) for r in selected), Decimal(0))) for key in ("usd_point", "usd_upper", "usd_with_retry_upper")}}
    report = {"run_id": manifest["run_id"], "rubric_version": manifest["rubric_version"],
              "code": manifest["code"], "native_tariff": native,
              "native_tariff_status": "available" if native else "unregistered_gpt6_luna_openai_in_native_accounting",
              "priced_route": judge["backend"], "price_metadata_sha256": sha256(metadata),
              "price_metadata_path": str(price_metadata), "judge": policy["judge"],
              "point_rates_per_million_usd": {"input": str(Decimal(rates["prompt"]) * 1000000),
                                              "output": str(Decimal(rates["completion"]) * 1000000)},
              "upper_rates_per_million_usd": {"input": str(input_rate * 1000000), "output": str(output_rate * 1000000)},
              "assumptions": {"input_tokenizer_proxy": tokenizer_name, "chat_envelope_tokens": 16,
                              "reasoning_tokens_per_request": policy["estimate"]["reasoning_tokens_per_request"],
                              "output_tokens_per_item": policy["estimate"]["output_tokens_per_item"],
                              "input_upper": "UTF-8 bytes including response schema + 2048",
                              "output_upper": policy["judge"]["max_tokens"], "cache_discount": False,
                              "samples_per_request": 1, "empty_transport_retries": max_attempts - 1,
                              "point_and_usd_upper_attempts": 1, "retry_upper_attempts": max_attempts,
                              "not_provider_measured": True, "empty_and_uncollected_requests": 0}, "totals": totals}
    destination.mkdir(parents=True)
    write_jsonl(destination / "estimate_cells.jsonl", detail)
    atomic_json(destination / "estimate.json", report)
    return report
