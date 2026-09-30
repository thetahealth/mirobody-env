"""Where the solver sees the rare coding task: `prediction_context.coding_task` (the task note)
and `prediction_context.attachments` (pointers), at the top level, where mirobody-rare
`serve_coding` reads them and where the rare-code-bench branch put them. Nothing solver-visible
under `prediction_context.rare`; the gold stays in `adjudication.rare`."""
import glob
import pathlib

import yaml

import haenv_rare as HR
from haenv import external_gold as EG

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _meta(case: dict) -> dict:
    lat = case["latent"]
    return {EG.SLOT: {k: lat[k] for k in HR.RARE_KEYS if k in lat}}


def _first_case(job: str) -> dict:
    return yaml.safe_load((ROOT / "inputs" / f"{job}.job.yaml").read_text(encoding="utf-8"))["cases"][0]


def test_registration_puts_the_task_and_the_pointers_at_the_top():
    HR._ensure_task_registered()
    gold_fn, probe_fn, keys, _, _ = EG.BLOCKS["rare"]
    assert probe_fn is None and "rare_lang" in keys
    for name in ("coding_task", "attachments"):
        g, p, k, _, _ = EG.BLOCKS[name]
        assert p is not None and k == ()
    case = _first_case("rare_coding-p3")
    got = EG.probe_blocks_for(_meta(case))
    assert set(got) == {"coding_task", "attachments"}, sorted(got)
    assert got["coding_task"] == HR.coding_task_probe(_meta(case))
    assert set(got["attachments"]) == {"genome", "imaging", "narrative"}
    assert "rare" not in got
    gold = EG.blocks_for(_meta(case))
    assert set(gold) == {"rare"} and gold["rare"]["hpo_gold"] == case["latent"]["rare_hpo_gold"]


def test_pointers_only_no_gold_on_the_question_side():
    att = HR.attachments_probe(_meta(_first_case("rare_coding-p5-en-llm")))
    flat = repr(att)
    for secret in ("variant", "genotypes", "decoys", "spans", "region", "hpo_id"):
        assert secret not in flat, secret
    assert att["narrative"].keys() == {"path", "sha256"}


def test_the_task_note_follows_the_pack_language():
    assert HR.coding_task_probe(_meta(_first_case("rare_coding-p4-en"))).startswith("This case also carries")
    assert HR.coding_task_probe(_meta(_first_case("rare_coding-p3"))).startswith("本例另有一项")


def test_a_non_rare_case_gets_nothing():
    assert HR.coding_task_probe({}) is None and HR.attachments_probe({}) is None and HR.gold_block({}) is None


def test_every_shipped_rare_case_gets_the_task_and_its_pointers():
    n = n_att = 0
    for p in sorted(glob.glob(str(ROOT / "inputs" / "rare_coding-*.job.yaml"))):
        for c in yaml.safe_load(pathlib.Path(p).read_text(encoding="utf-8"))["cases"]:
            m = _meta(c)
            assert isinstance(HR.coding_task_probe(m), str), (p, c["case_id"])
            has = any((c["latent"].get("rare_attachments") or {}).get(k) for k in ("genome", "imaging", "narrative"))
            assert bool(HR.attachments_probe(m)) == has, (p, c["case_id"])
            n += 1
            n_att += has
    # p0..p9; p1's four acquired cases carry no attachment (p0..p5 were 146 / 142 before p6..p9)
    assert (n, n_att) == (306, 302)


def test_a_retrieval_case_gets_its_questions_and_pointers_and_no_task_note():
    HR._ensure_task_registered()
    cases = yaml.safe_load((ROOT / "inputs" / "rare_retrieval-p6p7.job.yaml").read_text(encoding="utf-8"))["cases"]
    for c in cases:
        got = EG.probe_blocks_for(_meta(c))
        assert set(got) == {"retrieval", "attachments"}, (c["case_id"], sorted(got))
        rr = c["latent"]["rare_retrieval"]
        assert [q["qid"] for q in got["retrieval"]["questions"]] == [q["qid"] for q in rr["questions"]]
        assert "gold" not in repr(got["retrieval"]) and "dropped" not in repr(got["retrieval"])
    assert len(cases) == 60
