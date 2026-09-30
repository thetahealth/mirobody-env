"""Availability references measured on the actual solver-visible time window."""
from __future__ import annotations

from copy import deepcopy
import json
import math


def parse_reference(prompt: str) -> tuple[dict, dict]:
    from .judge_compaction import unpack_reference
    try:
        reference = json.loads(prompt.split("REFERENCE:\n", 1)[1]
                               .split("\n\nATOMIC CRITERIA:\n", 1)[0])
        criteria = json.loads(prompt.split("ATOMIC CRITERIA:\n", 1)[1]
                              .split("\n\nEVIDENCE POINTERS", 1)[0]
                              .split("\n\nBEGIN UNTRUSTED", 1)[0])
    except (IndexError, ValueError) as exc:
        raise ValueError("Sealed reference format is not understood") from exc
    return unpack_reference(reference), criteria


def visible_probe(reference: dict) -> dict | None:
    probe = reference.get("noop")
    if probe is None:
        return None
    target, window = probe.get("target"), probe.get("window")
    if (not isinstance(target, str) or not target or not isinstance(window, list)
            or len(window) != 2 or any(type(t) not in (int, float) or not math.isfinite(t) for t in window)
            or window[0] > window[1] or type(probe.get("truth_present")) is not bool):
        raise ValueError("Invalid recorded availability probe")
    visible = (reference.get("visible_case") or {}).get("longitudinal_data")
    if not isinstance(visible, dict):
        raise ValueError("Actual delivered longitudinal payload is unavailable")
    points = visible.get(target, [])
    if not isinstance(points, list):
        raise ValueError("Delivered target is not a series")
    times = []
    for p in points:
        if (not isinstance(p, dict) or type(p.get("ts")) not in (int, float)
                or not math.isfinite(p["ts"])):
            raise ValueError("Delivered target contains an invalid timestamp")
        times.append(p["ts"])
    count = sum(window[0] <= t <= window[1] for t in times)
    return {"target": target, "window": window, "source_truth_present": probe["truth_present"],
            "target_delivered": target in visible, "n_visible_in_window": count,
            "visible_truth_present": count > 0, "mismatch": (count > 0) != probe["truth_present"]}


def correct_reference(reference: dict) -> tuple[dict, dict | None]:
    """Do not change unaffected reference bytes or manufacture missing inputs."""
    check = visible_probe(reference)
    if check is None or not check["mismatch"]:
        return reference, None
    fixed = deepcopy(reference)
    fixed["noop"]["truth_present"] = check["visible_truth_present"]
    fixed["noop"]["truth_basis"] = "actual_delivered_window"
    return fixed, check
