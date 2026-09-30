"""The coding task note tells the solver where its attachment pointers are. Since the pointers sit
at the top of the payload (`prediction_context.attachments`, maintainers' decision of 2026-09-28, replacing Q1 of
2026-09-23), the note must name that path; a note pointing at `prediction_context.rare.attachments`
sends a reader to a key that no longer exists."""
import pytest

import haenv_rare as R
from haenv import external_gold as EG
from haenv_rare.gen import task_note


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_note_names_the_path_the_pointers_use(lang):
    meta = {EG.SLOT: {"rare_spec_id": "RD-LFS", "rare_lang": lang,
                      "rare_attachments": {"root": "/x", "narrative": {"path": "/x/a.md", "sha256": "0" * 64}}}}
    R._ensure_task_registered()
    got = EG.probe_blocks_for(meta)
    assert "attachments" in got and "rare" not in got     # the pointers are really at the top here
    note = task_note(lang)
    assert "prediction_context.attachments" in note
    assert "prediction_context.rare.attachments" not in note   # negative control: the stale path
