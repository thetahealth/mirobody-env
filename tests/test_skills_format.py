"""The agent skills under `skills/` follow the Agent Skills format (https://agentskills.io/specification).

Each `SKILL.md` opens with YAML frontmatter that a strict YAML parser reads, holding only the
fields the format defines, with `name` equal to the skill's directory. A skill that fails this is
rejected by the format's reference validator (`skills-ref validate`), and an agent tool that
parses the frontmatter strictly does not load it. An unquoted description containing ": " is
the common way to fail: YAML reads it as a nested mapping.

This is an ordinary unit test (reads the skill files only).

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import pathlib
import re

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
SKILLS = sorted(ROOT.glob("skills/*/SKILL.md"))
FIELDS = {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def problems(text: str, dirname: str) -> list[str]:
    """What the format refuses in one SKILL.md; empty when it conforms."""
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    if not m:
        return ["no YAML frontmatter"]
    try:
        meta = yaml.safe_load(m.group(1))
    except yaml.YAMLError as e:
        return [f"frontmatter is not valid YAML: {str(e).splitlines()[0]}"]
    if not isinstance(meta, dict):
        return ["frontmatter is not a mapping"]
    out = []
    if extra := sorted(set(meta) - FIELDS):
        out.append(f"fields outside the format: {extra}")
    name = meta.get("name")
    if not (isinstance(name, str) and len(name) <= 64 and NAME.fullmatch(name)):
        out.append(f"name {name!r} is not 1-64 lowercase letters, digits and single hyphens")
    elif name != dirname:
        out.append(f"name {name!r} differs from the directory {dirname!r}")
    desc = meta.get("description")
    if not (isinstance(desc, str) and desc.strip() and len(desc) <= 1024):
        out.append("description is not a non-empty string of at most 1024 characters")
    comp = meta.get("compatibility")
    if comp is not None and not (isinstance(comp, str) and 1 <= len(comp) <= 500):
        out.append("compatibility is not a string of 1-500 characters")
    md = meta.get("metadata", {})
    if not (isinstance(md, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in md.items())):
        out.append("metadata does not map strings to strings")
    return out


def test_the_repository_has_skills():
    assert SKILLS, "no skills/*/SKILL.md"


@pytest.mark.parametrize("path", SKILLS, ids=lambda p: p.parent.name)
def test_skill_follows_the_format(path):
    assert problems(path.read_text(encoding="utf-8"), path.parent.name) == []


def test_a_conforming_frontmatter_passes():
    front = 'name: demo\ndescription: >-\n  Run a pack: offline or billed.\nmetadata: {owner: "me"}\n'
    assert problems(f"---\n{front}---\n# body\n", "demo") == []


@pytest.mark.parametrize("front, reason", [
    ("name: demo\ndescription: Run a pack: offline or billed.\n", "not valid YAML"),
    ('name: demo\ndescription: Run a pack.\nargument-hint: "[job]"\n', "fields outside the format"),
    ("name: other\ndescription: Run a pack.\n", "differs from the directory"),
    ("name: Demo\ndescription: Run a pack.\n", "lowercase"),
    ("name: demo--x\ndescription: Run a pack.\n", "single hyphens"),
    ('name: demo\ndescription: "' + "x" * 1025 + '"\n', "at most 1024"),
    ("name: demo\ndescription: Run a pack.\nmetadata: {version: 1}\n", "strings to strings"),
], ids=["colon", "extra-field", "name-dir", "uppercase", "double-hyphen", "long", "metadata"])
def test_the_check_refuses_what_the_format_refuses(front, reason):
    found = problems(f"---\n{front}---\n# body\n", "demo")
    assert any(reason in p for p in found), found
