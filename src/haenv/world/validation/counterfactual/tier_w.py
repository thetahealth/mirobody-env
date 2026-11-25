"""Deterministic, report-only sample through the existing world twin machinery."""
from collections import Counter
import hashlib
import time

from .receipt import _DECISIVE, roles
from .registry import premises


def sample(case_ids, pack, k=20):
    if k < 0:
        raise ValueError("sample size must be nonnegative")
    return sorted(case_ids, key=lambda cid: (hashlib.sha256(f"cf-tier-w|{pack}|{cid}".encode()).digest(), cid))[:k]


def _mapped(pack, path):
    matches = [(len(prefix), p["id"]) for p in premises(pack) for prefix in p["reads"]
               if path == prefix or path.startswith(prefix + "/")]
    longest = max((length for length, _ in matches), default=-1)
    return sorted({pid for length, pid in matches if length == longest})


def run(adapter, items, k=20):
    if adapter.pack == "m2":
        return {"status": "not_run", "reason": "needs job rows and world rebuild"}
    from haenv.world.generation.twins import family_candidates
    start = time.monotonic()
    items = list(items)
    selected = sample([i.case_id for i in items], adapter.pack, k)
    by_id = {i.case_id: i for i in items}
    rows, rejections = [], Counter()
    preserving = Counter()
    for cid in selected:
        item = by_id[cid]
        item_roles = roles(adapter, item)
        rejected = []
        trials = family_candidates(item.raw, adapter.pack, rejected=rejected)
        rejections.update(str(r["reason"])[:120] for r in rejected)
        for trial in trials:
            mapped = _mapped(adapter.pack, trial["path"])
            decisive = [pid for pid in mapped if item_roles[pid]["role"] in _DECISIVE]
            changing = bool(trial["gold_changed"])
            if not changing:
                preserving.update(mapped or ["unmapped"])
            rows.append(dict(case_id=cid, kind=trial["kind"], path=trial["path"], gold_changed=changing,
                             premises=mapped, agreement="disagree" if changing and not decisive else "agree"))
    return dict(status="report_only", sample=selected, items=len(selected), trials=len(rows),
                gold_changed=sum(r["gold_changed"] for r in rows), gold_preserving=sum(not r["gold_changed"] for r in rows),
                disagree=sum(r["agreement"] == "disagree" for r in rows), preserving_by_premise=dict(preserving),
                rejected=sum(rejections.values()), rejections=dict(sorted(rejections.items())),
                runtime_seconds=round(time.monotonic() - start, 3), rows=rows)
