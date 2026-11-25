"""JSON leaf differences using stable record keys, including duplicate occurrences."""
from collections import Counter

_KEYS = ("id", "evidence_id", "record_id", "day", "night_of_day", "draw_day", "date", "drug", "name")
_MISSING = object()


def _keyed(rows):
    counts = Counter()
    out = {}
    for index, row in enumerate(rows):
        field = next((k for k in _KEYS if isinstance(row, dict) and k in row), None)
        key = f"{field}={row[field]}" if field else str(index)
        counts[key] += 1
        suffix = f"#{counts[key]}" if counts[key] > 1 else ""
        out[f"[{key}{suffix}]"] = row
    return out


def leaf_paths(a, b, parts=()):
    if a is _MISSING or b is _MISSING:
        value = b if a is _MISSING else a
        if isinstance(value, dict) and value:
            return [p for key in sorted(value) for p in leaf_paths(
                a.get(key, _MISSING) if isinstance(a, dict) else _MISSING,
                b.get(key, _MISSING) if isinstance(b, dict) else _MISSING, (*parts, str(key))) ]
        if isinstance(value, list) and value:
            aa, bb = _keyed(a) if isinstance(a, list) else {}, _keyed(b) if isinstance(b, list) else {}
            return [p for key in sorted(aa.keys() | bb.keys())
                    for p in leaf_paths(aa.get(key, _MISSING), bb.get(key, _MISSING), (*parts, key))]
        return [parts]
    # Provenance-bearing messages are str subclasses; JSON observes their value.
    if isinstance(a, str) and isinstance(b, str):
        return [] if a == b else [parts]
    if type(a) is not type(b):
        return [parts]
    if isinstance(a, dict):
        return [p for key in sorted(a.keys() | b.keys())
                for p in leaf_paths(a.get(key, _MISSING), b.get(key, _MISSING), (*parts, str(key)))]
    if isinstance(a, list):
        aa, bb = _keyed(a), _keyed(b)
        return [p for key in sorted(aa.keys() | bb.keys())
                for p in leaf_paths(aa.get(key, _MISSING), bb.get(key, _MISSING), (*parts, key))]
    return [] if a == b else [parts]


def _path(parts):
    return "/".join(parts)


def diff(a, b):
    paths = leaf_paths(a, b)
    return {"sections": sorted({p[0] for p in paths if p}),
            "records": sorted({_path(p[:2]) for p in paths}), "n_leaves": len(paths)}


def record_keys(blocks, sections):
    """Only section and record identifiers, with no record contents."""
    return sorted({p for p in diff({}, {k: v for k, v in blocks.items() if k in sections})["records"]})
