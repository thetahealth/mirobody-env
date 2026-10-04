"""Which links in the Markdown documents do not resolve inside this tree.

    python tools/doc_links.py            # print broken links, exit 1 if any

Two kinds of link are checked:

* **local links** -- `[text](path)` relative to the document, in every Markdown
  file of the tree;
* **links to this repository on GitHub** -- `github.com/<repo>/blob|tree/<ref>/<path>`
  and `raw.githubusercontent.com/<repo>/<ref>/<path>`. The READMEs use these so the
  links also work on the PyPI page, and a scanner that skips every `http` link never
  looks at them. The ref is ignored: the README writes `main`, a release pins a tag,
  and either way the path has to exist in the tree being released.

The same scan serves `tests/test_link_targets.py` (which ships, so the public tree checks
itself) and the maintainers' release gate; there is one way to read a link here.

SYNTHETIC, evaluation use only, not medical advice.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = "thetahealth/mirobody-env"

#: Superseded records keep the links of the layout they were written in, on purpose:
#: repointing them would rewrite history. Counted and reported, never read as clean.
ARCHIVE = "docs/archive/"

#: Directories that are never part of the documentation, when the tree is not a git
#: checkout and the file list comes from the filesystem.
_SKIP_DIRS = {".git", ".venv", "node_modules", "results", "reports", "_scratch",
              "_llm_cache", "dist", "build", "__pycache__"}

# Strip fenced blocks *before* inline spans: a ``` fence is three backticks, and an
# inline-span pattern run first pairs two of them and shifts every pairing after it.
_FENCE = re.compile(r"^```.*?^```", re.M | re.S)
_CODE_SPAN = re.compile(r"`[^`\n]*`")
_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)[^)]*\)")
_IMG = re.compile(r'<img src="([^"]+)"')
_HREF = re.compile(r'<a href="([^"]+)"')


def _repo_url(repo: str) -> re.Pattern:
    r = re.escape(repo)
    return re.compile(rf"^https://(?:github\.com/{r}/(?:blob|tree)/[^/]+|"
                      rf"raw\.githubusercontent\.com/{r}/[^/]+)/([^?#]*)")


def markdown_files(root: pathlib.Path = ROOT) -> list[str]:
    """Tracked Markdown files in a git checkout; every Markdown file otherwise."""
    if (root / ".git").exists():
        p = subprocess.run(["git", "-c", "core.quotePath=false", "ls-files", "*.md"],
                           cwd=root, capture_output=True, text=True)
        if p.returncode == 0:
            return sorted(x for x in p.stdout.splitlines() if x)
    return sorted(f.relative_to(root).as_posix() for f in root.rglob("*.md")
                  if not _SKIP_DIRS & set(f.relative_to(root).parts[:-1]))


def scan(root: pathlib.Path = ROOT, files: list[str] | None = None,
         repo: str = REPO) -> dict:
    """`{"broken": [(doc, target)], "archived": n, "total": n}` over `files` (default: all)."""
    url = _repo_url(repo)
    broken, archived, total = [], 0, 0
    for rel in (markdown_files(root) if files is None else files):
        md = root / rel
        text = md.read_text(encoding="utf-8", errors="ignore")
        text = _CODE_SPAN.sub(" ", _FENCE.sub(" ", text))
        for raw in _LINK.findall(text) + _IMG.findall(text) + _HREF.findall(text):
            raw = raw.strip()
            m = url.match(raw)
            if m:
                target, resolved = m.group(1).rstrip("/"), root / m.group(1)
            elif raw.startswith(("http:", "https:", "mailto:", "#")):
                continue
            else:
                target = raw.split("#")[0].split("?")[0]
                if not target:
                    continue
                resolved = md.parent / target
            total += 1
            if resolved.exists():
                continue
            if rel.startswith(ARCHIVE):
                archived += 1
            else:
                broken.append((rel, target))
    return {"broken": broken, "archived": archived, "total": total}


def main() -> int:
    r = scan()
    for doc, target in r["broken"]:
        print(f"{doc} -> {target}")
    tail = f" ({r['archived']} more in {ARCHIVE}, excluded by design)" if r["archived"] else ""
    print(f"{len(r['broken'])} broken of {r['total']} links{tail}")
    return 1 if r["broken"] else 0


if __name__ == "__main__":
    sys.exit(main())
