"""Render the gold-provenance section of a report.

`report.py` (judging segment) calls only `one_line` and `render`, so wording changes here do
not move the judging fingerprint. The `llm` source is always reported on its own line, never
merged with transcribed or clinician gold.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

#: Source -> label. Order is report order (strongest first).
SOURCE_LABEL: tuple[tuple[str, str], ...] = (
    ("source_text", "transcribed from the source text (the LLM only acts as a parser; the quote must be found verbatim in the source)"),
    ("clinician", "clinician-annotated (a human anchor; must carry an annotator id)"),
    ("llm", "**LLM-adjudicated** -- an authorized exception to source-text/clinician gold"),
    ("absent", "dropped (the field has no value ⇒ the criteria resting on it are **unjudgeable**, not scored 0)"),
)


def render(prov: dict | None, n_cases: int) -> list[str]:
    """Markdown lines for the section. Empty `prov` renders as "not recorded", never as "no LLM gold"."""
    L: list[str] = ["### Gold provenance\n"]
    if not prov:
        L.append(
            "> **This batch has no gold-provenance record.** This job was not produced by "
            "extracting gold from case text (a hand-written job.yaml has no "
            "`_provenance` block).\n"
            ">\n"
            "> **\"Not recorded\" does not mean \"recorded as having none\"** -- do not take this to "
            "mean none of this batch's gold was model-adjudicated. To get that number, go through the "
            "text-extraction path.\n")
        return L

    tot = dict(prov.get("gold_field_source_totals") or {})
    n_llm = int(prov.get("cases_with_llm_gold") or 0)
    L.append(f"> Source: `--gold {prov.get('gold_path', '?')}` · "
             f"extractor model `{prov.get('extractor_model', '—')}`"
             + (f" · annotator `{prov['annotator']}`" if prov.get("annotator") else "")
             + "\n")
    L.append("| source | gold fields | meaning |")
    L.append("|---|---:|---|")
    for k, label in SOURCE_LABEL:
        L.append(f"| `{k}` | {tot.get(k, 0)} | {label} |")
    L.append("")

    if n_llm:
        L.append(f"> **{n_llm}/{n_cases} cases had gold decided by the model.** "
                 "These items' answers are **not** transcribed and **not** clinician-given -- "
                 "their readings must **not** be blended into the same average as the other two paths.\n")
    else:
        L.append(f"> ✅ **0/{n_cases} cases had gold decided by the model** -- "
                 "this batch's gold is entirely source-text transcription, clinician annotation, "
                 "or explicitly dropped.\n")

    # Dropped gold fields make their judges unjudgeable, which is not the same as scoring 0.
    per = prov.get("per_case") or {}
    _absent_cases = [c for c, v in per.items() if (v or {}).get("gold_absent_fields")]
    if _absent_cases:
        L.append(f"> {len(_absent_cases)}/{n_cases} cases have gold fields that were **dropped** ⇒ "
                 "the criteria resting on those fields are recorded as \"unjudgeable\", **never scored 0** "
                 "(scoring 0 would penalize a model that got it right).\n")
    _untracked = [c for c, v in per.items() if (v or {}).get("gold_untracked_fields")]
    if _untracked:
        L.append(f"> {len(_untracked)} cases have gold fields with **no provenance record** -- "
                 "neither absent nor holding a value. This is a gap in record-keeping, not in the data.\n")
    return L


def one_line(prov: dict | None, n_cases: int) -> str:
    if not prov:
        return "Gold source: **not recorded** (hand-written job)"
    n_llm = int(prov.get("cases_with_llm_gold") or 0)
    return (f"Gold source: `{prov.get('gold_path', '?')}` · "
            + (f"**{n_llm}/{n_cases} cases model-adjudicated**" if n_llm
               else f"0/{n_cases} cases model-adjudicated"))
