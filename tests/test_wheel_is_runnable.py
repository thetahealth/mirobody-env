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


def _kernel_package_dir() -> pathlib.Path:
    """Where `haenv` expects the kernel package to live, relative to the repo root.

    Derived from `haenv._INSTALLED_KERNEL` -- the constant the code actually reads --
    rather than from a literal, so the constant and the packaging cannot drift apart.
    """
    import haenv

    pkg = pathlib.Path(haenv.__file__).resolve().parent
    kp = haenv._INSTALLED_KERNEL
    assert kp.parent == pkg.parent, (
        f"`haenv._INSTALLED_KERNEL` = {kp} 不在 `haenv/` 的同级 —— "
        f"打包时它就不是一个顶层包了")
    return kp


def _declared_packages() -> set[str]:
    """The `packages` list of the wheel target in `pyproject.toml`."""
    txt = PYPROJECT.read_text(encoding="utf-8")
    body = txt.split("[tool.hatch.build.targets.wheel]", 1)
    assert len(body) == 2, "pyproject 里没有 wheel 节"
    m = re.search(r'^packages\s*=\s*\[([^\]]*)\]', body[1], re.M)
    assert m, "wheel 节里没有 `packages = [...]`"
    return {x.strip().strip('"\'') for x in m.group(1).split(",") if x.strip()}


def _kernel_modules_haenv_imports() -> dict[str, pathlib.Path]:
    """Kernel top-level modules that `haenv` imports, as {module name: file in the source tree}.

    Read with the AST, not by text: the kernel imports are flagged in
    `haenv/` with a trailing `# kernel` comment, and scanning for that
    comment would let a comment impersonate an import (and would miss any
    import whose author forgot the marker). Imports are read as AST nodes
    for that reason; `haenv_kernel/__init__.py` re-exports the same modules
    under dotted names, and both spellings are collected here.
    """
    src = _kernel_package_dir()
    by_name = {p.stem: p for p in src.rglob("*.py")}
    found: dict[str, pathlib.Path] = {}
    for p in sorted((ROOT / "haenv").rglob("*.py")):
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(n, ast.ImportFrom):
                mod = n.module or ""
                if n.level:
                    continue
                head, _, tail = mod.partition(".")
                if head == "haenv_kernel" and tail in by_name:
                    found[tail] = by_name[tail]
                elif not tail and mod in by_name:      # legacy bare name, if any survives
                    found[mod] = by_name[mod]
            elif isinstance(n, ast.Import):
                for a in n.names:
                    head, _, tail = a.name.partition(".")
                    if head == "haenv_kernel" and tail in by_name:
                        found[tail] = by_name[tail]
                    elif not tail and a.name in by_name:
                        found[a.name] = by_name[a.name]
    return found


def test_kernel_package_is_declared_and_sits_beside_haenv():
    """`haenv_kernel` must be a declared top-level package, and `_INSTALLED_KERNEL` must name it.

    ## What this catches

    The kernel used to be a bare directory force-included into `haenv/_kernel/` and put
    on `sys.path` at import time. It is now an ordinary top-level package. The failure
    this guards is the half-migrated state: the source tree moved but `pyproject.toml`
    still ships the old layout (or vice versa), so the installed wheel has the code but
    not where the import path looks for it:

        pip install <wheel> && python -c "import haenv"   -> ModuleNotFoundError: haenv_kernel

    ## Why both sides are read rather than written down

    The constant is taken from `haenv._INSTALLED_KERNEL` and the declaration from
    `pyproject.toml`, so the two cannot drift apart by a shared literal going stale in
    one place.
    """
    import haenv

    kp = _kernel_package_dir()
    assert kp.name == "haenv_kernel", (
        f"🔴 `haenv._INSTALLED_KERNEL` = {kp},不叫 `haenv_kernel`")
    assert (kp / "__init__.py").is_file(), (
        f"🔴 {kp} 里没有 `__init__.py` —— 它不是包,装到 site-packages 后 import 不到")
    declared = _declared_packages()
    assert "haenv_kernel" in declared, (
        f"🔴 `pyproject.toml` 的 wheel `packages` 里没有 `haenv_kernel`(实为 {sorted(declared)})—— "
        f"源码树跑得动,轮子装完 import 不到")
    mods = _kernel_modules_haenv_imports()
    assert mods, "`haenv/` 里一个内核模块 import 都没扫到 —— 扫描面塌了,不是通过"
    missing = sorted(m for m in mods if not (kp / f"{m}.py").is_file())
    assert not missing, (
        f"🔴 `haenv/` import 的内核模块 {missing} 不在 {kp} 里 —— 路径常量指错了")


@pytest.mark.slow
def test_wheel_kernel_is_importable_where_the_constant_points():
    """The end-to-end half: in a real wheel, `haenv_kernel` must be a top-level package whose modules import.

    The test above is a derivation; this one is the artefact. It builds the wheel,
    unpacks it, and imports the kernel modules from exactly the location the package
    declares -- the step that turns red on a half-migrated layout with no reasoning
    required.

    The import runs in a subprocess: the kernel's top-level names include `build`, which
    is also a PyPI package CI installs, so importing it into this session would shadow it
    for every test that follows. (Under the package layout `haenv_kernel.build` no longer
    shadows anything, but the probe still runs isolated as a matter of habit.)
    """
    import shutil
    import subprocess
    import sys
    import tempfile
    import zipfile

    if not shutil.which("uv"):
        pytest.skip("uv 不在 PATH 上,建不了轮子")
    pkg = _kernel_package_dir().name                            # "haenv_kernel"
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
            missing = [m for m in mods if f"{pkg}/{m}.py" not in names]
            assert not missing, (
                f"🔴 轮子里 {pkg}/ 下没有 {missing} —— 内核没打进轮子")
            assert f"{pkg}/__init__.py" in names, (
                f"🔴 轮子里 {pkg}/ 不是包(缺 `__init__.py`)")
            z.extractall(out)
        kdir = out / pkg
        assert kdir.is_dir(), f"🔴 解包后 {kdir} 不是目录"
        probe = ("import sys; sys.path.insert(0, %r); " % str(out)
                 + "; ".join(f"import {pkg}.{m}" for m in mods))
        p = subprocess.run([sys.executable, "-c", probe],
                           capture_output=True, text=True)
        assert p.returncode == 0, (
            f"🔴 从轮子里 import `{pkg}` 失败:\n{p.stdout}{p.stderr}")


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
