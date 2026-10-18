"""Lossless tabular representation of repeated time-series keys for judge prompts."""
from __future__ import annotations

from copy import deepcopy

FORMAT = "haenv-column-rows-v1"
FORMAT_INSTRUCTION = (
    "Time-series objects tagged haenv-column-rows-v1 are exact tables: columns gives the keys, "
    "and every row holds their values in that order. Expand each row by zipping columns and values. "
    "All original points, timestamps, nulls and units are preserved; this is not a summary."
)


def pack_reference(reference: dict) -> dict:
    result = deepcopy(reference)
    series = result.get("visible_case", {}).get("longitudinal_data", {})
    for key, points in series.items():
        if not isinstance(points, list) or len(points) < 2 or not all(isinstance(p, dict) for p in points):
            continue
        columns = sorted(points[0])
        if not columns or any(set(p) != set(columns) for p in points):
            continue
        series[key] = {"format": FORMAT, "columns": columns,
                       "rows": [[point[c] for c in columns] for point in points]}
    return result


def unpack_reference(reference: dict) -> dict:
    result = deepcopy(reference)
    series = result.get("visible_case", {}).get("longitudinal_data", {})
    for key, table in series.items():
        if not isinstance(table, dict) or table.get("format") != FORMAT:
            continue
        columns, rows = table["columns"], table["rows"]
        if len(set(columns)) != len(columns) or any(len(row) != len(columns) for row in rows):
            raise ValueError("Malformed lossless time-series table")
        series[key] = [dict(zip(columns, row)) for row in rows]
    return result
