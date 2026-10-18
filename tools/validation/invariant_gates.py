"""invariant_gates.py -- the whole-pool invariant gates (`haenv/world/validation/invariants.py`) on one pack's emission pool
and its published subset. Zero model calls. Exit 0 whatever the counts; with `--fail-closed`, exit 1 when a gate
of `invariants.FAIL_CLOSED` reports a violation (`tools/generation/make_pack.py` calls `report` after its audit).

  PYTHONPATH=<repo>:<repo>/plugins python tools/validation/invariant_gates.py --make-pack <make_pack.json> --out <dir>
      [--purchases <m2 purchases.jsonl>]          # M2: reuse a purchase dump instead of re-running it
      [--jobs N] [--fail-closed]
  PYTHONPATH=... python tools/validation/invariant_gates.py --pack p2 --batch <pool batch> [--pack-batch <batch>] --out <dir>

M2's purchasable results come from the production gated loop with a scripted full-menu purchase
(`haenv/world/validation/invariants_purchase.py`); without `--purchases` they are dumped into `<out>/m2_purchases.jsonl`.
Writes `<out>/invariants-<pack>.json` and `.md`.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

# Direct file execution and imports select the same marked checkout.
import sys as _tool_sys
from pathlib import Path as _ToolPath
for _tool_root in _ToolPath(__file__).resolve().parents:
    if (_tool_root / "pyproject.toml").is_file() and (_tool_root / "tools").is_dir():
        _tool_sys.path.insert(0, str(_tool_root))
        break
else:
    raise FileNotFoundError(f"Cannot locate repository root from {__file__}")
from tools._bootstrap import bootstrap as _tool_bootstrap, repository_root as _repository_root
_tool_bootstrap(__file__)

import argparse
import json
import logging
import pathlib
import sys

ROOT = _repository_root(__file__)
sys.path.insert(0, str(ROOT))

import haenv.world.validation.invariants as INV       # noqa: E402


def _ids(batch: pathlib.Path) -> set[str]:
    return {json.loads(line)["case_id"] for line in open(batch / "cases.jsonl", encoding="utf-8")}


def md(pack: str, rep: dict) -> str:
    lines = [f"# 不变式门 · {pack}", "", "| 门 | 类 | 池违例 | 包违例 | 例子 |", "|---|---|---|---|---|"]
    for gid, r in rep.items():
        if r["scope"] == "pack":
            ex = "; ".join(r["pack_detail"][:5] or r["pool_detail"][:5])
        else:
            ex = "; ".join(f"{e['case_id']}: {e['violations'][0]}" for e in r["examples"][:3])
        lines.append(f"| {gid} | {r['cls']} | {r['pool']} | {r['pack']} | {ex.replace('|', '/')[:300]} |")
    return "\n".join(lines) + "\n"


def report(pack: str, batch, out, published: set[str] | None = None, job=None, purchases=None,
           pack_batch=None, jobs: int = 1) -> dict:
    """Run every gate on the pool `batch` (and its `published` ids); write `invariants-<pack>.json` / `.md`
    under `out`; return the document. M2 purchases are dumped into `out` unless `purchases` is given."""
    out, batch = pathlib.Path(out), pathlib.Path(batch)
    out.mkdir(parents=True, exist_ok=True)
    buy = purchases
    if pack == "m2" and not buy:
        import haenv.world.validation.invariants_purchase as P
        buy = out / "m2_purchases.jsonl"
        P.dump_parallel(pathlib.Path(job), batch, buy, jobs=max(1, int(jobs or 1)))
    items = INV.load(batch, pack, buy)
    # A source renderer may expose the same clinical facts in a new schema.
    # The optional adapter rebuilds the old invariant inputs from visible facts
    # alone, retaining the original source rows for retrieval/deletion checks.
    from haenv.interaction.query.adapter import ADAPTERS
    from haenv.interaction.query.adapter import load_pack_adapters
    load_pack_adapters()
    adapter = ADAPTERS.get(pack)
    audit_block = getattr(adapter, "audit_block", None)
    if audit_block is not None:
        for it in items:
            pc = it["sp"].get("prediction_context") or {}
            it["sp"] = {**it["sp"], "prediction_context": {
                **pc, adapter.block: audit_block(pc.get(adapter.block) or {})}}
    # The immutable rubric snapshot duplicates dated records; temporal deletion
    # probes must enumerate the original case sources, not this private cache.
    snapshots = 0
    for it in items:
        adj = it["case"].get("adjudication") or {}
        if "open_management" in adj:
            it["case"] = {**it["case"], "adjudication": {
                k: v for k, v in adj.items() if k != "open_management"}}
            snapshots += 1
    if published is not None and (not published or not published <= {it["case_id"] for it in items}):
        raise ValueError("published IDs must be a nonempty subset of the invariant pool")
    rep = INV.run(items, pack, published=published, jobs=jobs)
    pub_items = [it for it in items if published is None or it["case_id"] in published]
    doc = {"pack": pack, "batch": str(batch), "pack_batch": str(pack_batch) if pack_batch else None,
           "purchases": str(buy) if buy else None, "n_pool": len(items), "n_pack": len(pub_items), "gates": rep,
           "branches_pool": INV.r2_table(items), "branches_pack": INV.r2_table(pub_items),
           "fail_closed": list(INV.FAIL_CLOSED), "refusals": INV.refusals(rep),
           "private_rubric_snapshots_omitted_from_gate_view": snapshots}
    (out / f"invariants-{pack}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / f"invariants-{pack}.md").write_text(md(pack, rep), encoding="utf-8")
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--make-pack", default=None, help="make_pack.json of a make_pack run (pool, pack, job)")
    ap.add_argument("--pack", default=None)
    ap.add_argument("--batch", default=None)
    ap.add_argument("--pack-batch", default=None)
    ap.add_argument("--job", default=None)
    ap.add_argument("--purchases", default=None)
    ap.add_argument("--jobs", type=int, default=1, help="M2 purchase worker processes")
    ap.add_argument("--fail-closed", action="store_true", help="exit 1 when a FAIL_CLOSED gate reports a violation")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    logging.disable(logging.CRITICAL)
    if a.make_pack:
        mp = json.loads(pathlib.Path(a.make_pack).read_text(encoding="utf-8"))
        a.pack, a.batch, a.pack_batch, a.job = mp["pack"], mp["batch"], mp.get("pack_batch"), mp.get("job")
    pub = _ids(pathlib.Path(a.pack_batch)) if a.pack_batch else None
    doc = report(a.pack, a.batch, a.out, pub, a.job, a.purchases, a.pack_batch, a.jobs)
    for gid, r in doc["gates"].items():
        print(f"{gid:4s} {r['cls']} pool={r['pool']:4d} pack={r['pack']:4d}{'  [fail-closed]' if gid in INV.FAIL_CLOSED else ''}")
    for x in doc["refusals"]:
        print(f"[invariants] refused: {x}")
    return 1 if (a.fail_closed and doc["refusals"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
