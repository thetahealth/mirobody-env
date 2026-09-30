"""text_to_job.py -- **a batch of case-description text -> a job.yaml**.

The front-end command line for haenv's target shape. Each of the three field
classes follows its own path (see the header of `haenv/provenance.py`):

    A patient facts -> LLM extraction     B gold adjudication -> four paths (see --gold)     C generation knobs -> deterministic sampling

Usage:
    uv run python tools/text_to_job.py --in raw_cases/joint_dx/*.md \\
        --job-id my-job --task-type joint_dx --gold absent --out my-job.yaml

    # gold transcribed from the text (only recorded when the text explicitly states the diagnosis)
    ... --gold text
    # gold adjudicated by an LLM -- Warning: violates section 2a literally, **flagged field by field** in the artifact
    ... --gold llm
    # gold filled in from a clinician annotation file (JSON: {case_id: {field: value}})
    ... --gold clinician --annotations doc.json --annotator reviewer-1

`--cut-at` truncates the input at the given heading (default `## Outcome
and Disposition`) -- our own raw-case markdown files **contain
verifier-only sections**, and feeding the whole thing in would show the
answer to the extractor. Real case text doesn't have this problem, so use
`--cut-at ""` for it.

Note: **the artifact carries a `_provenance` block.** It records where each
gold field came from, and this **must** propagate all the way to the report:
"how many cases in this batch had their gold decided by the model itself"
should be a visible number, not something that has to be excavated.

SYNTHETIC data, evaluation use only, not medical advice.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="+", required=True,
                    help="病例描述文本(.md/.txt),可多个")
    ap.add_argument("--out", required=True, help="产出的 job.yaml 路径")
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--task-type", default="joint_dx")
    ap.add_argument("--gold", required=True,
                    choices=["absent", "text", "llm", "clinician"],
                    help="🔴 金标从哪来 —— **必填,不给默认**。它决定这批题的可信度,"
                         "不该由一个默认值替人决定")
    ap.add_argument("--model", default="gemini-3.1-pro")
    ap.add_argument("--annotations", help="--gold clinician 用:JSON 标注文件")
    ap.add_argument("--annotator", help="--gold clinician 用:标注者 id")
    ap.add_argument("--input-kind", choices=["prospective", "retrospective"],
                    default="prospective",
                    help="喂进来的是哪种病例文本。`prospective`(默认,严档)= 入组时的主诉+基线,"
                         "**里面没有结局**;`retrospective` = 完整回顾式病例报告,逐字写着结局 "
                         "⇒ 放开结局标签的转录(20 例实测:提取率 95% / quote 逐字过 95% / "
                         "准确率 19/20,多数类基线 70%、p=0.0076)。"
                         "默认取严档:误声明成回顾式会让抽取器在没有结局的文本里硬找结局,"
                         "并盖上 `source_text` 戳 —— 那是伪造溯源")
    ap.add_argument("--cut-at", default="## 转归与结局",
                    help="在这个小标题处截断输入(防 verifier-only 段落被抽取器看到);"
                         "传空串表示不截断")
    ap.add_argument("--dry", action="store_true", help="只打印,不写文件")
    a = ap.parse_args(argv)

    import yaml

    from haenv.cli import _bootstrap, load_cfg
    cfg = load_cfg()
    _bootstrap(cfg)
    from haenv.extract import ExtractionError, text_to_case

    ann_all = {}
    if a.gold == "clinician":
        if not a.annotations or not a.annotator:
            print("🔴 --gold clinician 必须同时给 --annotations 与 --annotator "
                  "(没有「是谁」的人类锚不是锚)")
            return 2
        ann_all = json.loads(pathlib.Path(a.annotations).read_text(encoding="utf-8"))

    files = [pathlib.Path(f) for f in a.inputs]
    cases, prov, bad = [], {}, []
    for i, f in enumerate(files, 1):
        text = f.read_text(encoding="utf-8")
        if a.cut_at:
            lines = text.splitlines()
            for j, ln in enumerate(lines):
                if ln.startswith(a.cut_at):
                    text = "\n".join(lines[:j])
                    break
        cid = f.stem.upper().replace("RC-", "").replace("_", "-")
        try:
            ec = text_to_case(text, cid, gold_path=a.gold, model=a.model,
                              annotations=ann_all.get(cid), annotator=a.annotator,
                              cfg=cfg, input_kind=a.input_kind)
        except ExtractionError as e:
            bad.append((cid, str(e)[:90]))
            print(f"  [{i}/{len(files)}] {cid}: 🔴 {str(e)[:90]}")
            continue
        cases.append(ec.to_case_dict())
        prov[cid] = ec.ledger.summary()
        _n_llm = len(prov[cid]["gold_llm_fields"])
        print(f"  [{i}/{len(files)}] {cid}: raw {len(ec.raw)} 项 · "
              f"latent {len(ec.latent)} 项 · 金标来源 "
              f"{prov[cid]['gold_provenance_counts']}"
              + (f" 🔴 LLM 判定 {_n_llm} 个字段" if _n_llm else ""))
        for w in ec.warnings:
            print(f"        ⚠️ {w[:130]}")

    if not cases:
        print("🔴 一个 case 都没产出 —— **不写空 job**(空集不是合规)")
        return 2

    # Note: batch-level aggregation: making "where did gold come from" a
    # **visible number**
    #
    # **The source list is read from `provenance.PROVENANCES`, never copied
    # a second time here.** This used to hardcode the four
    # `("source_text","clinician","llm","absent")` -- after the `derived`
    # tier was added on 2026-09-05, the identically-shaped hardcoded list in
    # `cli.py` broke immediately (the per-case count had it, the batch
    # aggregate didn't => 6 cases x 14 fields = 84, but only **77** got
    # summed). That time **only `cli.py` got fixed**, and this entry point
    # wasn't caught until an audit found it on 2026-09-06 -- "the same
    # mistake exists in two copies, fixing one is the same as fixing none"
    # for the Nth time in this repo.
    # Using `.get(p, 0)`: a new source tier won't cause a KeyError here.
    from haenv.provenance import PROVENANCES as _PROVS
    agg = {p: sum(v["gold_provenance_counts"].get(p, 0) for v in prov.values())
           for p in _PROVS}
    _n_fields = sum(sum(v["gold_provenance_counts"].values()) for v in prov.values())
    if sum(agg.values()) != _n_fields:
        # An aggregate silently swallowing fields wouldn't show up in the
        # total => reconcile it here and shout immediately if it doesn't match.
        print(f"🔴 汇总 {sum(agg.values())} ≠ 逐例合计 {_n_fields} —— "
              f"有来源档没被 PROVENANCES 覆盖,**别信这一行的数**")
    n_llm_cases = sum(1 for v in prov.values() if v["gold_llm_fields"])
    print(f"\n=== {len(cases)} 例产出 · 失败 {len(bad)} 例 ===")
    print(f"金标字段来源合计: {agg}")
    print(f"🔴 有 LLM 判定金标的 case: {n_llm_cases}/{len(cases)}"
          + ("  —— 这些题的金标不是转录也不是医生给的,**报告里必须单列**"
             if n_llm_cases else ""))

    job = {
        "job_id": a.job_id, "task_type": a.task_type, "findings": True,
        "multiround": False, "include_baseline": True, "models": [],
        "sample_cases": min(6, len(cases)),
        "report": f"eval-{a.job_id}.md",
        # The trail travels with the job -- it isn't a comment, it's data
        "_provenance": {
            "gold_path": a.gold, "extractor_model": a.model,
            "annotator": a.annotator,
            # Note: **same fields as** the provenance block in
            # `haenv/cli.py` (2026-09-06). The `source_text` stamp on
            # `ddx_outcome_label` is only valid under the `retrospective`
            # tier; `cut_at` determines whether the extractor ever saw the
            # verifier-only answer key/value table -- without truncation,
            # that 95% is a cheating upper bound. Both are preconditions for
            # "can this stamp be trusted."
            "input_kind": a.input_kind, "cut_at": a.cut_at or None,
            "gold_field_source_totals": agg,
            "cases_with_llm_gold": n_llm_cases,
            "per_case": prov,
            "note": ("🔴 `llm` 一栏 > 0 表示这些题的金标由模型判定(§2a 的字面例外,"
                     "经授权后逐字段标注才可进入)。**不要**与 source_text/clinician 的混进"
                     "同一个平均数。"),
        },
        "cases": cases,
    }
    out = pathlib.Path(a.out)
    if a.dry:
        print(f"\n(--dry)将写入 {out}:{len(cases)} 例")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "# 由 tools/text_to_job.py 从病例描述文本生成 —— **勿手改**。\n"
        f"# 重建:uv run python tools/text_to_job.py --in <文本> --job-id {a.job_id} "
        f"--gold {a.gold} --out {a.out}\n"
        "# `_provenance` 记着每个金标字段从哪来;它必须一路传到报告。\n"
        "# SYNTHETIC,仅评测用,非医疗建议。\n"
        + yaml.safe_dump(job, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    print(f"\n✅ 写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
