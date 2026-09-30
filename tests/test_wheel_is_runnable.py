"""The wheel runs after it's installed -- a contract for the `pip install haenv` path.

## What this guards against

A non-editable install that does not run:

    pip install <wheel> && haenv build ...
    -> FileNotFoundError: <site-packages>/config.yaml

`ROOT = Path(__file__).parent.parent` is the repo root inside the source
tree and `site-packages` itself inside an installed wheel. Two halves keep
an installed wheel runnable: the read side is unified into
`haenv.data_root()` (which picks a tier based on whether `config.yaml` is
present there), and the packaging side force-includes resources into
`haenv/_data/`. This file pins down that both halves are in place.

## Why this doesn't actually install a wheel

A real install needs `uv build` plus building a venv, roughly 30 seconds,
and depends on `uv` being on PATH -- that would make the default tier on a
clean clone slower and more fragile. So this file pins down the
declaration surface (which resources are promised to be packaged) and the
resolution surface (`data_root`'s tiering logic); the end-to-end check is
left to manual pre-release verification, with the command in
`docs/REPRODUCE.md`.

The file also catches "packaged, but the path is wrong". A
`haenv._WHEEL_KERNEL` pointing at a different depth than the `force-include`
destination makes all four of `haenv build/verify/run/report` exit 1
with `内核路径不存在: None` and makes `import haenv.report` raise
`ModuleNotFoundError: No module named 'build'` -- every resource is
packaged, and the constant that finds them is wrong.

`test_wheel_kernel_constant_matches_the_packaging_destination` covers it:
it derives the destination from `force-include` instead of restating it,
so the constant and the packaging cannot drift apart, and the slow
`test_wheel_kernel_is_importable_where_the_constant_points` walks the same
path in a real wheel.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import ast
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"


def _force_include() -> dict[str, str]:
    """The `[tool.hatch.build.targets.wheel.force-include]` section, parsed into {source: destination}."""
    txt = PYPROJECT.read_text(encoding="utf-8")
    body = txt.split("[tool.hatch.build.targets.wheel.force-include]", 1)
    assert len(body) == 2, "pyproject 里没有 force-include 节 —— 资源一件都不会进轮子"
    out: dict[str, str] = {}
    for line in body[1].splitlines():
        line = line.strip()
        if line.startswith("["):            # the next section starts
            break
        m = re.match(r'"([^"]+)"\s*=\s*"([^"]+)"', line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _kernel_force_include() -> tuple[pathlib.Path, pathlib.PurePosixPath]:
    """The one `force-include` entry that carries the L0 kernel, as (source dir in the repo, destination in the wheel).

    Found by asking `haenv` where it looks -- the destination is whatever
    `_WHEEL_KERNEL`'s first path component under the package is -- rather
    than by matching the literal string `core`. Naming the source here
    would make this helper agree with the constant by construction, which
    is the one thing it must not do.
    """
    import haenv

    pkg = pathlib.Path(haenv.__file__).resolve().parent
    top = haenv._WHEEL_KERNEL.relative_to(pkg).parts[0]      # e.g. "_kernel"
    want = f"{pkg.name}/{top}"
    hits = {s: d for s, d in _force_include().items() if d.rstrip("/") == want}
    assert len(hits) == 1, (
        f"`force-include` 里应当恰有一条把内核装到 {want!r},实际 {sorted(hits)} —— "
        f"要么内核没打进轮子,要么 `_WHEEL_KERNEL` 指的根本不是打包落点")
    src, dst = next(iter(hits.items()))
    return (ROOT / src).resolve(), pathlib.PurePosixPath(dst.rstrip("/"))


def _kernel_modules_haenv_imports() -> dict[str, pathlib.Path]:
    """Kernel top-level modules that `haenv` imports, as {module name: file in the source tree}.

    Read with the AST, not by text: the kernel imports are flagged in
    `haenv/` with a trailing `# kernel` comment, and scanning for that
    comment would let a comment impersonate an import (and would miss any
    import whose author forgot the marker). Only absolute, single-segment
    imports count -- `from .build import ...` is haenv's own module, and
    the two are spelled almost identically.
    """
    src, _ = _kernel_force_include()
    by_name = {p.stem: p for p in src.rglob("*.py")}
    found: dict[str, pathlib.Path] = {}
    for p in sorted((ROOT / "haenv").rglob("*.py")):
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(n, ast.ImportFrom):
                if n.level == 0 and n.module in by_name:
                    found[n.module] = by_name[n.module]
            elif isinstance(n, ast.Import):
                for a in n.names:
                    if a.name in by_name:
                        found[a.name] = by_name[a.name]
    return found


def test_wheel_kernel_constant_matches_the_packaging_destination():
    """`haenv._WHEEL_KERNEL` must be the directory the kernel modules actually land in, derived from `force-include`.

    ## What this catches

    A constant reading `haenv/_kernel/metabolic_harness` while
    `force-include` mapped `core` -> `haenv/_kernel`, whose modules are
    already top level: the wheel contains `haenv/_kernel/build.py` and
    the extra level never exists. On a clean venv install of such a wheel:

        haenv build | verify | run | report   -> exit 1,
            "[haenv] 内核路径不存在: None"
        python -c "import haenv.report"       -> ModuleNotFoundError: No module named 'build'

    The whole defect class is "every file is packaged, and the path that
    finds them is wrong," which every other test in this file is blind to
    by construction: they compare `force-include` against the code that
    *reads* resources, never against the constant that *locates* them.

    ## Why the destination is derived, not written down

    Restating `"_kernel"` here would turn this into a copy of the value it
    is judging, and the copy would have been updated in the same commit
    that broke the constant -- or not at all. The expected path is instead
    computed from two independent facts: where `force-include` puts the
    source directory, and where inside that directory each module `haenv`
    imports actually sits.
    """
    import haenv

    src, dst = _kernel_force_include()
    mods = _kernel_modules_haenv_imports()
    assert mods, "`haenv/` 里一个内核模块 import 都没扫到 —— 扫描面塌了,不是通过"

    # `force-include` maps the *contents* of `src` onto `dst`, so a module at
    # `src/<rel>/<name>.py` lands at `dst/<rel>/<name>.py`.
    dirs = {p.parent.relative_to(src).as_posix() for p in mods.values()}
    assert len(dirs) == 1, (
        f"内核模块散在多层目录里 {sorted(dirs)},`sys.path` 上挂一个目录挂不全它们:"
        f"{sorted(mods)}")
    rel = dirs.pop()

    pkg = pathlib.Path(haenv.__file__).resolve().parent
    parts = dst.parts + ((rel,) if rel != "." else ())
    expect = pkg.parent.joinpath(*parts)
    assert haenv._WHEEL_KERNEL == expect, (
        f"🔴 `_WHEEL_KERNEL` = {haenv._WHEEL_KERNEL}\n"
        f"   而 force-include 把内核装到 {expect}\n"
        f"⇒ 轮子装完 `haenv build/verify/run/report` 全部 exit 1(内核路径不存在),"
        f"`import haenv.report` 报 No module named '{sorted(mods)[0]}'。"
        f"别改这条测试来让红变绿 —— 改 `haenv/__init__.py` 的常量。")


@pytest.mark.slow
def test_wheel_kernel_is_importable_where_the_constant_points():
    """The end-to-end half: in a real wheel, the directory `_WHEEL_KERNEL` names must exist and its modules must import.

    The test above is a derivation; this one is the artefact. It builds
    the wheel, unpacks it, and imports the kernel modules from exactly the
    path `_WHEEL_KERNEL` would resolve to for that layout -- the step that
    turns red on a wrong constant with no reasoning required.

    The import runs in a subprocess: the kernel's top-level names include
    `build`, which is also a PyPI package CI installs, so importing it
    into this session would shadow it for every test that follows.
    """
    import shutil
    import subprocess
    import sys
    import tempfile
    import zipfile

    import haenv

    if not shutil.which("uv"):
        pytest.skip("uv 不在 PATH 上,建不了轮子")
    pkg = pathlib.Path(haenv.__file__).resolve().parent
    rel = haenv._WHEEL_KERNEL.relative_to(pkg.parent)        # e.g. "haenv/_kernel"
    mods = sorted(_kernel_modules_haenv_imports())
    assert mods, "`haenv/` 里一个内核模块 import 都没扫到 —— 扫描面塌了,不是通过"

    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run(["uv", "build", "--wheel", "-o", td],
                           cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, f"uv build 失败:\n{r.stdout}{r.stderr}"
        whl = sorted(pathlib.Path(td).glob("*.whl"))[-1]
        out = pathlib.Path(td) / "unpacked"
        with zipfile.ZipFile(whl) as z:
            names = set(z.namelist())
            missing = [m for m in mods if f"{rel.as_posix()}/{m}.py" not in names]
            assert not missing, (
                f"🔴 轮子里 {rel.as_posix()}/ 下没有 {missing} —— "
                f"`_WHEEL_KERNEL` 指的位置装完不存在")
            z.extractall(out)
        kdir = out / rel
        assert kdir.is_dir(), f"🔴 解包后 {kdir} 不是目录"
        probe = ("import sys; sys.path.insert(0, %r); " % str(kdir)
                 + "; ".join(f"import {m}" for m in mods))
        p = subprocess.run([sys.executable, "-c", probe],
                           capture_output=True, text=True)
        assert p.returncode == 0, (
            f"🔴 从 `_WHEEL_KERNEL` 的落点 import 内核失败:\n{p.stdout}{p.stderr}")


def test_data_root_is_the_single_entry():
    """`haenv/` must have no in-place repo-root computation left -- that's exactly the root cause of `pip install` not running.

    The check is "does this take `__file__` and go up two levels as the
    resource root," not a literal string -- scanning by literal text would
    grant immunity to a differently-spelled version of the same thing
    (scanning only for `ROOT /` misses 16 occurrences).
    """
    pat = re.compile(r"(?:Path|_P|_Path|pathlib\.Path|_pathlib\.Path)"
                     r"\(__file__\)\.resolve\(\)\.parent\.parent")
    bad = []
    for p in sorted((ROOT / "haenv").rglob("*.py")):
        if p.name == "__init__.py":         # `data_root()` itself lives here
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if pat.search(line):
                bad.append(f"{p.relative_to(ROOT)}:{i}")
    assert not bad, ("这些地方还在就地算仓根,轮子安装下它们指向 site-packages:\n  "
                     + "\n  ".join(bad) + "\n⇒ 改用 `from haenv import data_root as _dr`")


def test_every_runtime_tools_module_is_shipped():
    """Every `tools/` module `haenv/` imports at runtime must be in force-include.

    Scanned live, not read off a list -- a hand-written list would
    silently go stale when another module is imported, and the symptom is a
    `FileNotFoundError` partway through a run after install (packaging only
    `make_freeze` leaves out `tools/attribute_failures.py`).
    """
    want: set[str] = set()
    pat_path = re.compile(r'"tools"\s*/\s*"([a-z_]+\.py)"')
    pat_imp = re.compile(r'from\s+tools\.([a-z_]+)|import\s+tools\.([a-z_]+)')
    for p in (ROOT / "haenv").rglob("*.py"):
        s = p.read_text(encoding="utf-8")
        want |= {m for m in pat_path.findall(s)}
        for a, b in pat_imp.findall(s):
            want.add(f"{a or b}.py")
    shipped = {k.split("/")[-1] for k in _force_include() if k.startswith("tools/")}
    missing = sorted(want - shipped)
    assert not missing, (f"这些 `tools/` 模块 haenv 运行时会 import,但没打进轮子:{missing}\n"
                         f"⇒ 在 pyproject 的 force-include 里加一行")


def test_resources_the_code_reads_are_shipped():
    """Every top-level resource read under `data_root()` must be in force-include.

    The list comes from scanning live for `_dr() / "<name>"`, not from a copy.
    """
    pat = re.compile(r'_dr\(\)\s*/\s*"([a-zA-Z_]+)"')
    want = set()
    for p in (ROOT / "haenv").rglob("*.py"):
        want |= set(pat.findall(p.read_text(encoding="utf-8")))
    want |= {"config.yaml"}                  # `__init__` and `cli` go through ROOT / "config.yaml"
    shipped = set()
    for src, dst in _force_include().items():
        shipped.add(src.split("/")[0])
        shipped.add(src)
    missing = sorted(w for w in want if w not in shipped and f"{w}.yaml" not in shipped)
    assert not missing, (f"代码会从资源根读这些,但它们没打进轮子:{missing}")


def test_data_root_honours_explicit_override(tmp_path, monkeypatch):
    """Positive control: the path named by `HAENV_DATA_ROOT` must be used verbatim (even if it doesn't exist).

    Shaped identically to `kernel_path()` -- a path the user explicitly
    named should appear in the error message; silently substituting
    something else is harder to debug.
    """
    import importlib

    import haenv
    monkeypatch.setenv("HAENV_DATA_ROOT", str(tmp_path / "nowhere"))
    importlib.reload(haenv)
    try:
        assert haenv.data_root() == (tmp_path / "nowhere").resolve()
    finally:
        monkeypatch.delenv("HAENV_DATA_ROOT", raising=False)
        importlib.reload(haenv)


def test_data_root_falls_back_by_content_not_by_path_shape(tmp_path, monkeypatch):
    """Negative control -- the tiering rule must be "does `config.yaml` exist there," not what the path looks like.

    Guessing from path shape (e.g. checking for `site-packages`) would
    silently take the wrong branch the moment the venv layout changes, and
    the symptom of taking the wrong branch is "reads an empty config" --
    harder to debug than a crash.
    """
    import importlib

    import haenv
    monkeypatch.delenv("HAENV_DATA_ROOT", raising=False)
    importlib.reload(haenv)
    got = haenv.data_root()
    assert (got / "config.yaml").is_file(), \
        f"data_root() 指到 {got},而那里没有 config.yaml —— 分档判据坏了"


@pytest.mark.slow
def test_wheel_actually_contains_the_resources():
    """The end-to-end half: actually builds a wheel, and checks every resource is present.

    More expensive than the tests above (roughly 10 seconds), but it's the
    only one that can catch "the declaration is correct but packaging
    didn't actually take effect." Skips if `uv` isn't on PATH -- the skip
    is printed, never collapsed into a pass.
    """
    import shutil
    import subprocess
    import tempfile
    import zipfile

    if not shutil.which("uv"):
        pytest.skip("uv 不在 PATH 上,建不了轮子")
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run(["uv", "build", "--wheel", "-o", td],
                           cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, f"uv build 失败:\n{r.stdout}{r.stderr}"
        whl = sorted(pathlib.Path(td).glob("*.whl"))[-1]
        names = set(zipfile.ZipFile(whl).namelist())
        for dst in _force_include().values():
            assert any(n == dst or n.startswith(dst.rstrip("/") + "/") for n in names), \
                f"force-include 声明了 {dst},而轮子里没有它"


def test_output_root_never_writes_into_the_package(tmp_path, monkeypatch):
    """An output artifact must never be written into the package directory -- the read root and the write root are two separate entry points.

    ## What this catches

    Running `haenv build` in a work directory outside the repo after
    `pip install <wheel>`, with the write root inside the package, the
    marker method counts:

        site-packages/haenv/_data/results/early_warning/ew-demo/<batch>/...   <- 3 files
        the work directory                                                    <- 0 files

    Three consequences: (1) a read-only, system-level, or containerized
    install fails outright; (2) `pip uninstall haenv` deletes the output
    artifacts along with the package; (3) what got written was a case
    pack with the gold standard included -- scattering the gold standard
    into `site-packages`.

    An acceptance check of "four lines, exit 0" says nothing about where
    the output landed; this test checks that directly.

    ## When this should turn red

    If someone wires `Job.root` back to `data_root()` (or `ROOT`). At that
    point, an install would start writing into the package again.
    """
    import haenv

    # (1) source tree (in-repo): read root == write root, behavior unchanged byte for byte
    assert haenv.output_root() == haenv.data_root(), \
        "仓内两者必须相等 —— 不相等说明这次改动动了源码树上的行为"

    # (2) wheel layout: `data_root()` lands inside the package => `output_root()` must leave the package directory
    pkg = tmp_path / "site-packages" / "haenv"
    (pkg / "_data").mkdir(parents=True)
    (pkg / "_data" / "config.yaml").write_text("models: {}\n", encoding="utf-8")
    monkeypatch.setattr(haenv, "_WHEEL_DATA", pkg / "_data")
    monkeypatch.setattr(haenv, "data_root", lambda: pkg / "_data")
    monkeypatch.delenv("HAENV_OUTPUT_ROOT", raising=False)
    monkeypatch.delenv("HAENV_DATA_ROOT", raising=False)
    work = tmp_path / "workdir"
    work.mkdir()
    monkeypatch.chdir(work)
    got = haenv.output_root()
    assert got == work, f"wheel 布局下产物根必须是 CWD,实际 {got}"
    assert pkg not in got.parents and got != pkg, "🔴 产物根落在包目录里"

    # (3) an explicit override wins (both variables are honored; `HAENV_OUTPUT_ROOT` is more specific)
    monkeypatch.setenv("HAENV_DATA_ROOT", str(tmp_path / "ws1"))
    assert haenv.output_root() == (tmp_path / "ws1").resolve()
    monkeypatch.setenv("HAENV_OUTPUT_ROOT", str(tmp_path / "ws2"))
    assert haenv.output_root() == (tmp_path / "ws2").resolve(), \
        "两个都设时 HAENV_OUTPUT_ROOT 必须赢(它更specific)"


def test_cli_passes_output_root_not_data_root():
    """The contract has to propagate to the call site -- having `output_root()` is useless unless `cli` actually uses it.

    This test uses the AST to check exactly what `cli.main`'s
    `load_job(..., root=...)` passes: passing `ROOT` (the resource read
    root) fails. A text scan would be fooled by a comment, and this very
    comment happens to contain the word `ROOT`.
    """
    import ast
    src = (pathlib.Path(__file__).resolve().parent.parent / "haenv" / "cli.py").read_text("utf-8")
    calls = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "load_job"]
    assert calls, "`cli.py` 里找不到 `load_job(...)` 调用 —— 扫描面塌了,不是通过"
    for c in calls:
        kw = {k.arg: k.value for k in c.keywords}
        assert "root" in kw, "`load_job` 必须显式传 root(默认值是 CWD,那是另一个语义)"
        expr = ast.unparse(kw["root"])
        assert "ROOT" not in expr, (
            f"🔴 `load_job(root={expr})` 传的是资源读根 —— 装成 wheel 之后产物会进 site-packages")


#: Calls allowed to accept the resource read root `ROOT` -- each one must
#: state its reason here.
#: Anything outside this allowlist fails by default: this judge's default
#: answer is "not allowed," not "not checked."
_ROOT_OK: dict[str, str] = {
    # looks for the repo root's `BOARD_FREEZE` marker file -- that's a resource shipped with the repo, not an output artifact.
    "refusal": "board_freeze.refusal() 读仓根的 BOARD_FREEZE 标记,是资源不是产物",
}


def test_cli_never_hands_the_read_root_to_something_that_writes():
    """Any call in `cli.main` that passes `ROOT` out must register its reason in `_ROOT_OK`.

    ## Why default-deny

    A check that only filters on `func.id == "load_job"` scans by syntactic
    shape and grants immunity to any other shape: `batch_mod.resolve(ROOT,
    ...)` (an attribute call, not a `Name`) and `make_dispatcher(cfg, m,
    ROOT)` (a positional argument, not a keyword) both slip past it. With a
    read root that differs from the write root, handing `ROOT` to a writer
    means, under a non-editable install:

      - `haenv report` looks for the batch directory in the read root while
        it lives under the write root (exit 2);
      - every `run` opens a fresh batch and discards the one `build` wrote;
      - the case-generation cache lands in
        `site-packages/haenv/_data/cases/_llm_cache/`.

    The judge is therefore default-deny plus per-call registration: code
    that passes `ROOT` has to state what it reads.
    """
    import ast
    src = (pathlib.Path(__file__).resolve().parent.parent / "haenv" / "cli.py").read_text("utf-8")
    tree = ast.parse(src)

    # a non-empty scan surface is a precondition: if the name `ROOT` were ever renamed, the loop below would become vacuously true.
    assert any(isinstance(n, ast.Name) and n.id == "ROOT" for n in ast.walk(tree)), \
        "`cli.py` 里找不到 `ROOT` —— 扫描面塌了,不是通过"

    bad = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        if not any(isinstance(a, ast.Name) and a.id == "ROOT"
                   for a in list(n.args) + [k.value for k in n.keywords]):
            continue
        f = n.func
        name = getattr(f, "id", None) or (f.attr if isinstance(f, ast.Attribute) else "?")
        if name not in _ROOT_OK:
            bad.append(f"{name}(...) @ cli.py:{n.lineno}")
    assert not bad, (
        "🔴 下面的调用收下了资源**读根** `ROOT`,而它没在 `_ROOT_OK` 里登记:\n  "
        + "\n  ".join(bad)
        + "\n⇒ 它若要**写**,改传 `OUT`(output_root);它若只**读**随仓分发的资源,"
          "把理由写进 `_ROOT_OK`。别改这条测试来让红变绿。")
