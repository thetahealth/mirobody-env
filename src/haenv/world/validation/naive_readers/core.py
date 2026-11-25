"""Deterministic offline baseline, tree and decisive-evidence measurement.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
import json
import os
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class Item:
    case_id: str
    case: dict
    sp: dict
    gold: dict
    in_published: bool = False
    composition: str | None = None


@dataclass(frozen=True)
class Reader:
    name: str
    kind: str
    fn: Callable[[dict], Any]
    note: str


#: Slack above a reader's structural floor: the largest single-item swing of a 70-item stratum is
#: 1/70 = 0.0143, so 0.05 admits three items of sampling noise and nothing more.
FLOOR_MARGIN = 0.05

COMMON_READERS = ("full", "today_only", "latest_row_only", "summary_sentence_only", "trust_system_score",
                  "checklist_only", "latest_dispense_pillbox", "printed_value_only", "keyword_only", "anomaly")
#: Per-reader structural-floor gate. Not in `TARGETS`: only a pack whose readers expose a single-wrong
#: premise can be judged by it (see the world-1.5 difficulty design note).
T2F = {"statement": "Every naive/stub reader at most its structural floor plus the margin", "threshold": None,
       "source": "world-1.5 difficulty design note: per-reader structural floor"}
TARGETS = {
    "T1": {"statement": "Full reader accuracy equals 100%", "threshold": 1.0,
           "source": "C1 answerability (1.4 question design, C1); difficulty design section 1"},
    "T2": {"statement": "Every naive/stub reader and both oracle constants at most 60%", "threshold": 0.60,
           "source": "Difficulty design section M0 (design target)"},
    "T3": {"statement": "Naive depth-3 tree accuracy at most 65%", "threshold": 0.65,
           "source": "Difficulty design section M0 (design target)"},
    "T4": {"statement": "At least 60% of items have two decisive facts", "threshold": 0.60,
           "source": "Difficulty design section M0 (design target)"},
    "T5": {"statement": "At least 50% have decisive evidence outside today", "threshold": 0.50,
           "source": "Difficulty design section M0 (design target)"},
}


def one_thread():
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"


def load_batch(path, pack, published_ids=frozenset()):
    cases = {}
    with (Path(path) / "cases.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            cid = row["case_id"]
            assert cid not in cases, f"duplicate case: {cid}"
            cases[cid] = row.get("case", row)
    payloads = {}
    with (Path(path) / "payloads.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            assert row["case_id"] not in payloads, f"duplicate payload: {row['case_id']}"
            payloads[row["case_id"]] = row["sp"]
    assert cases.keys() <= payloads.keys(), "cases missing solver payloads"
    return [Item(cid, case, payloads[cid], pack.gold(case), cid in published_ids,
                 pack.gold(case).get("composition")) for cid, case in cases.items()]


def stratum_indices(pool, published):
    n = len(pool)
    strata = {"pool": list(range(n)), "published": list(range(n, n + len(published)))}
    values = sorted({item.composition for item in published if item.composition is not None})
    for value in values:
        strata[f"published/{value}"] = [n + i for i, item in enumerate(published) if item.composition == value]
    return strata


def record_answer(pack, answer, gold, error=None):
    return {"answer": answer, "ok": False if error else bool(pack.ok(answer, gold)), "error": error}


def initialize_worker(pack_name, state):
    one_thread()
    from . import adapter
    pack = adapter(pack_name)
    if pack_name == "p2":
        pack.KNOWN_VOCAB = state
    elif pack_name == "m2":
        pack.CODEBOOK, pack.ALIASES = state["codebook"], state["aliases"]


def measure_item(item, pack_name, mode):
    from . import adapter
    from .decisive import AutoSource, select
    pack = adapter(pack_name)
    row = {"case_id": item.case_id, "in_published": item.in_published, "composition": item.composition,
           "gold_label": pack.gold_label(item.gold), "gold_labels": sorted(pack.constant_answers(item.gold)),
           "readers": {}, "trees": {}, "constants": {}}
    if hasattr(pack, "flip_capable"):
        # The premise each naive reader reads wrong, per item: the structural-floor gate reads these.
        row["flip"] = {name: bool(pack.flip_capable(item.sp, item.gold, name)) for name in pack.FLOOR_READERS}
    for reader in pack.READERS:
        try:
            answer, error = reader.fn(item.sp), None
        except Exception as exc:
            answer, error = None, f"{type(exc).__name__}: {exc}"
        row["readers"][reader.name] = record_answer(pack, answer, item.gold, error)
    source = select(pack, mode)
    if isinstance(source, AutoSource):
        source = source.for_item(item)
    row["decisive_source"] = source.name
    try:
        facts = source.read(item, row["readers"].get("full", {}).get("answer"))
        row["facts"] = None if facts is None else [asdict(fact) for fact in facts]
        row["fact_error"] = None
    except Exception as exc:
        row["facts"] = None
        row["fact_error"] = f"{type(exc).__name__}: {exc}"
    facts = row["facts"]
    active = [f for f in facts or [] if f["per_fact"] and f["outcome"] in {"flip", "undetermined"}]
    row.update(n_decisive=len(active) if facts is not None else None,
               n_decisive_outside_today=sum(f["outside_today"] is True for f in active) if facts is not None else None,
               decisive=sorted(f["fact"] for f in active),
               n_flip=sum(f["outcome"] == "flip" for f in active),
               n_undetermined=sum(f["outcome"] == "undetermined" for f in active))
    if pack_name == "p2" and source.name == "knockout":
        row["note_groups"] = sorted(f["fact"] for f in facts or [] if f["fact"] in pack.FACTS and f["outcome"] == "flip")
    if hasattr(pack, "declared"):
        row.update(pack.declared(item.gold))
    try:
        naive, noise = pack.features(item.sp)
        row["features"] = {"naive": naive, "naive+history_noise": {**naive, **noise}}
        row["features"].update(getattr(pack, "extra_features", lambda _: {})(item.sp))
        row["feature_error"] = None
    except Exception as exc:
        row["features"] = {}
        row["feature_error"] = f"{type(exc).__name__}: {exc}"
    try:
        row["group"] = pack.group_key(item.sp)
        row.update(getattr(pack, "metadata", lambda _: {})(item.sp))
    except Exception as exc:
        row["group"] = None
        row["metadata_error"] = f"{type(exc).__name__}: {exc}"
    return row


def constant_result(items, pack, grouped=False):
    groups = defaultdict(list)
    for item in items:
        key = pack.group_key(item.sp) if grouped else "all"
        if grouped and key is None:
            return {"kind": "constant", "n": 0, "n_ok": 0, "rate": None, "note": "Per-group constant is not applicable."}
        groups[key].append(item)
    choices, n_ok = {}, 0
    for key, group in sorted(groups.items()):
        counts = Counter(label for item in group for label in pack.constant_answers(item.gold))
        label, count = min(counts.items(), key=lambda pair: (-pair[1], pair[0])) if counts else ("independent", 0)
        choices[key] = label
        n_ok += count
    n = len(items)
    return {"kind": "constant", "n": n, "n_ok": n_ok, "rate": n_ok / n if n else None,
            "n_error": 0, "errors": [], "choices": choices,
            "note": "Oracle per-stratum upper bound, selected using gold acceptance labels."}


def tree_columns(pool, published, rows, pack):
    import numpy as np
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.model_selection import KFold
    from sklearn.tree import DecisionTreeClassifier
    n = len(pool)
    items = pool + published
    configs = {}
    feature_sets = sorted({key for row in rows for key in row["features"]} | {"naive", "naive+history_noise"})
    published_ids = {item.case_id for item in published}
    for feature_set in feature_sets:
        valid = [i for i in range(n) if feature_set in rows[i]["features"]]
        vectorizer = DictVectorizer(sort=True)
        if valid:
            vectorizer.fit([rows[i]["features"][feature_set] for i in valid])
        for depth in (3, 5):
            for protocol in ("cv5", "pool_minus_published"):
                name = f"tree:{feature_set}:d{depth}:{protocol}"
                configs[name] = {"feature_set": feature_set, "depth": depth, "protocol": protocol,
                                 "min_samples_leaf": 5, "random_state": 0, "feature_keys": []}
                target = list(range(len(items))) if protocol == "cv5" else list(range(n, len(items)))
                for i in target:
                    rows[i]["trees"][name] = record_answer(pack, None, items[i].gold, "Tree prediction unavailable")
                if protocol == "cv5":
                    if n < 5:
                        continue
                    splits = []
                    for train, test in KFold(5, shuffle=True, random_state=0).split(np.arange(n)):
                        held_ids = {pool[i].case_id for i in test}
                        published_test = [n + i for i, item in enumerate(published) if item.case_id in held_ids]
                        splits.append((list(train), list(test) + published_test))
                else:
                    splits = [([i for i in range(n) if pool[i].case_id not in published_ids], list(range(n, len(items))))]
                for train, test in splits:
                    train = [i for i in train if i in valid]
                    test = [i for i in test if feature_set in rows[i]["features"]]
                    if not train or not test:
                        continue
                    x_train = vectorizer.transform([rows[i]["features"][feature_set] for i in train])
                    if x_train.shape[1] == 0:
                        continue
                    model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=5, random_state=0)
                    model.fit(x_train, [pack.gold_label(items[i].gold) for i in train])
                    prediction = model.predict(vectorizer.transform([rows[i]["features"][feature_set] for i in test]))
                    configs[name]["feature_keys"] = sorted(set(configs[name]["feature_keys"]) | set(vectorizer.get_feature_names_out()))
                    for i, label in zip(test, prediction):
                        rows[i]["trees"][name] = record_answer(pack, pack.answer_from_label(str(label)), items[i].gold)
    return configs


def aggregate_answers(records, kind, note):
    n = len(records)
    n_ok = sum(record["ok"] for record in records)
    return {"kind": kind, "n": n, "n_ok": n_ok, "rate": n_ok / n if n else None,
            "n_error": sum(record.get("error") is not None for record in records),
            "errors": list(dict.fromkeys(record["error"] for record in records if record.get("error")))[:3], "note": note}


def decisive_summary(rows):
    available = [r for r in rows if r["n_decisive"] is not None]
    n = len(available)
    decisive = lambda r: [f for f in r["facts"] if f["per_fact"] and f["outcome"] in {"flip", "undetermined"}]
    hist = Counter(str(min(r["n_decisive"], 3)) if r["n_decisive"] < 3 else "3+" for r in available)
    result = {"n": n, "n_unavailable": len(rows) - n, "histogram": {k: hist[k] for k in ("0", "1", "2", "3+")},
              "share_ge2": sum(r["n_decisive"] >= 2 for r in available) / n if n else None,
              "share_outside_today": sum(r["n_decisive_outside_today"] >= 1 for r in available) / n if n else None,
              "by_fact": dict(Counter(f["fact"] for r in available for f in decisive(r))),
              "by_set": dict(Counter("+".join(r["decisive"]) or "none" for r in available)),
              "n_flip": sum(r["n_flip"] for r in available), "n_undetermined": sum(r["n_undetermined"] for r in available),
              "n_error": sum(f["outcome"] == "error" for r in rows for f in r["facts"] or []),
              "n_source_error": sum(r["fact_error"] is not None for r in rows),
              "source_counts": dict(Counter(r["decisive_source"] for r in rows)),
              "available_source_counts": dict(Counter(r["decisive_source"] for r in available)),
              "n_outside_today_known": sum(r["decisive_source"] == "knockout" or
                                           any(f["outside_today"] is not None for f in r["facts"] if f["per_fact"])
                                           for r in available)}
    if any("note_groups" in r for r in rows):
        note = Counter(str(len(r["note_groups"])) for r in rows if "note_groups" in r)
        result["note_groups"] = {"histogram": dict(sorted(note.items())),
                                 "by_set": dict(Counter("+".join(r["note_groups"]) or "none" for r in rows if "note_groups" in r))}
    if any("declared_key_signals" in r for r in rows):
        result["declared"] = {"note": "declared, not knockout tested",
                              "key_signals_histogram": dict(Counter(str(r["declared_key_signals"]) for r in rows)),
                              "history_dependency_active": sum(r["history_dependency_active"] for r in rows)}
    return result


def evaluate_targets(stratum, name):
    readers, constants, decisive = stratum["readers"], stratum["constants"], stratum["decisive"]
    result = {}
    def add(key, values, predicate):
        values = {k: v for k, v in values.items() if v is not None}
        failures = sorted(k for k, value in values.items() if not predicate(value))
        result[key] = {**TARGETS[key], "status": ("fail" if failures else "pass") if values else "n/a",
                       "values": values, "failures": failures}
    add("T1", {k: r["rate"] for k, r in readers.items() if r["kind"] == "full"}, lambda v: v == 1.0)
    add("T2", {**{k: r["rate"] for k, r in readers.items() if r["kind"] in {"naive", "stub"}},
               **{k: r["rate"] for k, r in constants.items()}}, lambda v: v <= 0.60)
    floors = stratum.get("floors")
    if floors:
        # A reader is judged against 1 - (share of published core items whose gold its single wrong
        # premise can move), plus a margin; a reader with no single-wrong premise keeps the flat line.
        margin = FLOOR_MARGIN
        # A reader is allowed its structural floor plus the margin, but nothing above the flat line
        # unless its own structure justifies it: the limit never falls below T2's 0.60.
        flat = TARGETS["T2"]["threshold"]
        limits = {k: (None if v["floor"] is None else round(min(1.0, max(flat, v["floor"] + margin)), 4))
                  for k, v in floors.items() if isinstance(v, dict)}
        values = {k: readers[k]["rate"] for k in limits
                  if k in readers and readers[k]["kind"] in {"naive", "stub"}}
        failures = sorted(k for k, v in values.items() if limits[k] is not None and v > limits[k] + 1e-12)
        result["T2F"] = {**T2F, "status": ("fail" if failures else "pass") if values else "n/a",
                         "values": values, "limits": limits, "failures": failures,
                         "floors": {k: v["floor"] for k, v in floors.items() if isinstance(v, dict)},
                         "capable": {k: v["capable"] for k, v in floors.items() if isinstance(v, dict)},
                         "n": floors.get("_n")}
    protocol = "cv5" if name == "pool" else "pool_minus_published"
    add("T3", {k: r["rate"] for k, r in stratum["trees"].items() if k == f"tree:naive:d3:{protocol}"}, lambda v: v <= 0.65)
    add("T4", {"share_ge2": decisive["share_ge2"]}, lambda v: v >= 0.60)
    outside = decisive["share_outside_today"]
    if decisive.get("n_outside_today_known", 1) == 0:
        outside = None
    add("T5", {"share_outside_today": outside}, lambda v: v >= 0.50)
    return result


def gate_exit(report, gate=False):
    if not gate:
        return 0
    targets = report["strata"][report["target_stratum"]]["targets"]
    floors = targets.get("T2F", {}).get("limits", {})
    flat_failures = [name for name in targets["T2"]["failures"] if floors.get(name) is None]
    other_failures = any(t["status"] == "fail" for key, t in targets.items() if key != "T2")
    return int(bool(flat_failures) or other_failures)


def run(pack_name, pool_path, published_path, lang, workers=32, mode="auto", limit=None):
    one_thread()
    if not 1 <= workers <= 32:
        raise ValueError("workers must be between 1 and 32")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    from . import adapter
    pack = adapter(pack_name)
    published = load_batch(published_path, pack)
    published_ids = {item.case_id for item in published}
    pool = load_batch(pool_path, pack, published_ids)
    assert published_ids <= {item.case_id for item in pool}, "published ids must be a subset of pool ids"
    state = getattr(pack, "configure", lambda _: None)(pool)
    if limit is not None:
        # Keep membership valid while reserving holdout training rows in a smoke run.
        pool = pool[:limit]
        ids = {item.case_id for item in pool}
        if not ids & published_ids and published:
            pool[-1] = next(item for item in load_batch(pool_path, pack, published_ids) if item.case_id == published[0].case_id)
            ids = {item.case_id for item in pool}
        published = [item for item in published if item.case_id in ids][:limit]
    published = [Item(i.case_id, i.case, i.sp, i.gold, True, i.composition) for i in published]
    items = pool + published
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker, initargs=(pack_name, state)) as executor:
        rows = list(executor.map(partial(measure_item, pack_name=pack_name, mode=mode), items, chunksize=4))
    for i, row in enumerate(rows):
        row["origin"] = "pool" if i < len(pool) else "published"
    trees = tree_columns(pool, published, rows, pack)
    blind = None
    if hasattr(pack, "blind_lists"):
        predictions, lists = pack.blind_lists(pool, published)
        blind = lists
        for item, row in zip(items, rows):
            answer = predictions.get(item.case_id)
            row["readers"]["blind_list"] = record_answer(pack, answer, item.gold,
                                                         None if answer is not None else "Fewer than five pool items")
    report = {"pack": pack_name, "lang": lang,
              "inputs": {"pool": str(pool_path), "published": str(published_path), "n_pool": len(pool), "n_published": len(published)},
              "composition": dict(Counter(i.composition or "unflagged" for i in published)),
              "not_applicable": pack.NOT_APPLICABLE, "tree_configs": trees,
              "fact_note": pack.FACT_NOTE, "strata": {}}
    if state is not None:
        report["pool_vocabulary"] = state
    if blind is not None:
        report["blind_training_lists"] = blind
    for name, indices in stratum_indices(pool, published).items():
        subset = [rows[i] for i in indices]
        subitems = [items[i] for i in indices]
        reader_table = {r.name: aggregate_answers([row["readers"][r.name] for row in subset], r.kind, r.note) for r in pack.READERS}
        if blind is not None:
            reader_table["blind_list"] = aggregate_answers([r["readers"]["blind_list"] for r in subset], "stub", "Five-fold question-blind greedy diagnosis list; production line cap.")
        constants = {key: constant_result(subitems, pack, grouped) for key, grouped in
                     (("best_constant", False), ("per_group_constant", True))}
        for key, constant in constants.items():
            if constant["rate"] is not None:
                for i in indices:
                    group = rows[i]["group"] if key == "per_group_constant" else "all"
                    label = constant["choices"][group]
                    rows[i]["constants"].setdefault(name, {})[key] = record_answer(pack, pack.answer_from_label(label), items[i].gold)
        tree_table = {key: aggregate_answers([r["trees"][key] for r in subset if key in r["trees"]], "tree", "Pool-trained deterministic depth-limited decision tree.") for key in trees if any(key in r["trees"] for r in subset)}
        stratum = {"n": len(indices), "readers": reader_table, "constants": constants,
                   "trees": tree_table, "decisive": decisive_summary(subset)}
        if hasattr(pack, "floors"):
            stratum["floors"] = pack.floors(subset)
        if pack_name == "p2":
            stratum["receipt_kinds"] = dict(Counter(r.get("receipt_kind") or "unavailable" for r in subset))
        stratum["targets"] = evaluate_targets(stratum, name)
        report["strata"][name] = stratum
    flagged = any(i.composition is not None for i in published)
    report["target_stratum"] = "published/core" if flagged else "published"
    report["target_fallback"] = not flagged
    if flagged and "published/core" not in report["strata"]:
        empty = {"n": 0, "readers": {}, "constants": {}, "trees": {}, "decisive": decisive_summary([])}
        empty["targets"] = evaluate_targets(empty, "published/core")
        report["strata"]["published/core"] = empty
    report["decisive_source_counts"] = {name: stratum["decisive"]["source_counts"] for name, stratum in report["strata"].items()}
    return report, rows


def markdown(report):
    def score(table, key):
        r = table.get(key)
        return "n/a" if not r or r["rate"] is None else f"{r['n_ok']}/{r['n']}"
    strata = report["strata"]
    lines = [f"# {report['pack']} ({report['lang']}) naive-reader measurement", "",
             "| Reader | Kind | Pool | Published |", "|---|---|---:|---:|"]
    for section in ("readers", "constants", "trees"):
        a, b = strata["pool"][section], strata["published"][section]
        for key in sorted(set(a) | set(b)):
            lines.append(f"| {key} | {(a.get(key) or b[key])['kind']} | {score(a, key)} | {score(b, key)} |")
    lines += ["", "Constants are oracle upper bounds selected separately in each stratum.", report["fact_note"], "",
              f"Gate stratum: {report['target_stratum']}; unflagged fallback: {report['target_fallback']}."]
    for name in ("pool", "published"):
        d = strata[name]["decisive"]
        lines.append(f"{name}: decisive histogram {json.dumps(d['histogram'], sort_keys=True)}; available {d['n']}; unavailable {d['n_unavailable']}.")
        lines.append(f"{name}: sources {json.dumps(d['source_counts'], sort_keys=True)}; flip {d['n_flip']}; undetermined {d['n_undetermined']}.")
        if "note_groups" in d:
            lines.append(f"{name}: note_groups {json.dumps(d['note_groups']['histogram'], sort_keys=True)}.")
        if "receipt_kinds" in strata[name]:
            lines.append(f"{name}: receipts {json.dumps(strata[name]['receipt_kinds'], sort_keys=True)}.")
    lines += ["", "| Target | Pool | Gate stratum | Source |", "|---|---|---|---|"]
    for key, target in TARGETS.items():
        a = strata["pool"]["targets"][key]
        b = strata[report["target_stratum"]]["targets"][key]
        suffix = "; failing: " + ", ".join(b["failures"]) if b["failures"] else ""
        lines.append(f"| {key} | {a['status']} | {b['status']}{suffix} | {target['source']} |")
    gate = strata[report["target_stratum"]]["targets"].get("T2F")
    if gate and gate["values"]:
        lines += ["", f"| Reader | Wrong premise can move | Floor | Limit (floor + margin, not below 0.60) | Rate | Status |",
                  "|---|---|---|---|---|---|"]
        for k in sorted(gate["values"], key=lambda k: -gate["values"][k]):
            status = "fail" if k in gate["failures"] else "pass"
            lines.append(f"| {k} | {gate['capable'][k]}/{gate['n']} | {gate['floors'][k]:.3f} "
                         f"| {gate['limits'][k]:.3f} | {gate['values'][k]:.3f} | {status} |")
    lines += ["", "Reader notes: " + " ".join(f"{k}: {r['note']}" for k, r in strata["pool"]["readers"].items()),
              "Not applicable: " + "; ".join(f"{k}: {v}" for k, v in sorted(report["not_applicable"].items()))]
    assert len(lines) <= 100
    return "\n".join(lines) + "\n"


def write_report(report, rows, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{report['pack']}-{report['lang']}"
    (out / f"{stem}.json").write_text(json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / f"{stem}.items.jsonl").write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (out / f"{stem}.md").write_text(markdown(report), encoding="utf-8")
