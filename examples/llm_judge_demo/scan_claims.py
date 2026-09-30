"""scan_claims.py -- how many judge requests this LLM judge would make on a batch
already on disk (offline, zero model calls).

    uv run python examples/llm_judge_demo/scan_claims.py results/joint_dx/<job>/<batch>

Offline runs never exercise the judge (every row is `no_claim`), so this counts
the (row x claimed-excluded registered near-name) pairs in existing model
responses and the requests left after cache dedup. Read-only.

SYNTHETIC, for evaluation only, not medical advice.
"""
from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _bootstrap() -> None:
    """Add the kernel to the import path, using the same `config.yaml` key as `haenv.cli`."""
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    sys.path.insert(0, str((ROOT / cfg["kernel_path"]).resolve()))


def scan(batch_dir: Path) -> dict:
    """Count this batch's judgeable objects, rebuilding `(out, vp)` through the
    pipeline's own loaders, kernel JSON extractor and `evaluate._to_output`.
    """
    import haenv_llm_judge_demo as P
    from haenv.evaluate import _to_output
    from haenv.payloads import load_payloads, payload_source
    from haenv.store import load_cases

    resp_p, cases_p = batch_dir / "responses.jsonl", batch_dir / "cases.jsonl"
    for p in (resp_p, cases_p):
        if not p.is_file():
            raise SystemExit(f"[scan] missing {p}")

    src, pl_p = payload_source(batch_dir)
    vps, sps = {}, {}
    if src == "disk":
        for cid, (sp, vp) in load_payloads(pl_p).items():
            vps[cid], sps[cid] = vp, sp
    print(f"[scan] payload source: {src}")
    if src != "disk":
        from build import build_instance                    # kernel
        for cid, rawc in load_cases(cases_p).items():
            T = int(rawc.prediction_context["prediction_time_T"])
            sps[cid], vps[cid] = build_instance(rawc, T)
    else:
        load_cases(cases_p)          # register the Q-side injection ledger back into `wq` (needed for the judge's full visible-EV set)

    from solver import _extract_json                        # kernel (the same parser as the pipeline)

    n_rows = n_with_rivals = n_with_claim = n_claims = n_unparsed = 0
    prompts: set[str] = set()
    by_solver: collections.Counter = collections.Counter()
    for line in resp_p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        cid = rec.get("case")
        vp, sp = vps.get(cid), sps.get(cid)
        if vp is None:
            continue
        n_rows += 1
        rivals = P.rivals_of(vp)
        if not rivals:
            continue
        n_with_rivals += 1
        raw = rec.get("raw") or {}
        if isinstance(raw, str):
            try:
                raw = _extract_json(raw)
            except (ValueError, TypeError, json.JSONDecodeError):
                n_unparsed += 1                             # an unparseable response is counted, not treated as an empty object
                continue
        if not raw:
            n_unparsed += 1
            continue
        out = _to_output(raw, sp)
        claims = P.claims_of(out, rivals)
        if not claims:
            continue
        n_with_claim += 1
        n_claims += len(claims)
        by_solver[rec.get("solver")] += len(claims)
        gold = P.gold_name(vp)
        for c in claims:
            prompts.add(hashlib.sha256(P.build_prompt(
                gold, c["rival"], c["discriminator"], c["reason"]).encode()).hexdigest())
    return {"batch": str(batch_dir), "n_rows": n_rows, "n_with_rivals": n_with_rivals,
            "n_unparsed": n_unparsed, "n_with_claim": n_with_claim, "n_claims": n_claims,
            "n_distinct_prompts": len(prompts), "by_solver": dict(by_solver)}


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    _bootstrap()
    r = scan(Path(argv[0]))
    print(json.dumps(r, ensure_ascii=False, indent=2))
    # When the count is 0, distinguish "this batch has nothing to judge" from "nothing was scanned".
    if r["n_rows"] == 0:
        print("[scan] no rows scanned -- case_id in responses doesn't match cases; "
              "this means the scan surface is empty, not that there's nothing to judge.")
        return 1
    if r["n_claims"] == 0:
        print("[scan] rows were scanned, but none carry a candidate that claims to have "
              "ruled out a registered near-name -- this judge is structurally inactive on "
              "this batch (expected for offline stub batches).")
    else:
        print(f"[scan] cost: {r['n_claims']} judgeable objects -> "
              f"{r['n_distinct_prompts']} requests after cache dedup "
              f"(saves {r['n_claims'] - r['n_distinct_prompts']}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
