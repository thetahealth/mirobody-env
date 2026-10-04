"""Every link in the documentation must resolve inside the tree it ships in.

The scan is `tools/doc_links.py`, which also backs the maintainers' release gate. This file
ships with the public tree, so a broken link there -- including a README link written as
an absolute GitHub URL, which no other check reads -- turns the public test run red.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("doc_links", ROOT / "tools/doc_links.py")
DL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DL)


def test_every_link_resolves():
    r = DL.scan()
    assert r["total"] > 50, f"only {r['total']} links scanned: the scan has lost its surface"
    assert r["broken"] == []


def _tree(tmp_path, readme):
    (tmp_path / "docs").mkdir(exist_ok=True)
    (tmp_path / "docs" / "DATA_CARD.md").write_text("card\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(readme, encoding="utf-8")
    return DL.scan(tmp_path)


def test_a_broken_local_link_is_caught(tmp_path):
    ok = _tree(tmp_path, "[card](docs/DATA_CARD.md)\n")
    bad = _tree(tmp_path, "[card](docs/DATA_CARD_GONE.md)\n")
    assert (ok["total"], ok["broken"]) == (1, [])
    assert bad["broken"] == [("README.md", "docs/DATA_CARD_GONE.md")]


def test_a_broken_repository_url_is_caught_whatever_the_ref(tmp_path):
    base = f"https://github.com/{DL.REPO}/blob"
    raw = f"https://raw.githubusercontent.com/{DL.REPO}"
    ok = _tree(tmp_path, f"[a]({base}/main/docs/DATA_CARD.md#x)\n"
                         f'<img src="{raw}/v1.1.1/docs/DATA_CARD.md">\n')
    bad = _tree(tmp_path, f"[a]({base}/v9.9.9/docs/GONE.md)\n")
    assert (ok["total"], ok["broken"]) == (2, [])
    assert bad["broken"] == [("README.md", "docs/GONE.md")]


def test_other_sites_and_code_are_not_links(tmp_path):
    r = _tree(tmp_path, "[x](https://example.org/a.md) `[y](nope.md)`\n"
                        "```\n[z](nope.md)\n```\n[w](#anchor)\n")
    assert (r["total"], r["broken"]) == (0, [])
