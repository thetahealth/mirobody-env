"""Language-independent text: `Msg`, lexicon phrases (`L`), templates (`T`), joins (`J`), `render`.

Generation code never writes final text. It writes `Msg` values: a `str` whose value is the
canonical text (the zh rendering, which is what generation code has always seen) and which
remembers what it renders from. At the end of a build `render(obj, lang)` replaces every `Msg`
with its text in `lang`.

* `L(msgid)` -- a phrase of the lexicon. A registry phrase is its own message id (the generator
  draws, orders and hashes registry phrases, so the id is the original string); code phrases use
  the same rule. zh renders the id itself unless `registry/i18n/zh/lexicon/` overrides it; every
  other language needs an entry in `registry/i18n/<lang>/lexicon/`.
* `T(key, **slots)` -- a composed text from `registry/i18n/<lang>/templates/*.yaml`
  (`str.format` syntax; a number keeps the format spec of its slot, so `5.0` stays `5.0`).
* `J(key, items)` -- items joined with the separator `key` of each language.

`render` refuses (`MissingText`) a string with Chinese text that is not a `Msg` when it renders a
language other than zh: such a string bypassed the rendering layer, and printing it would mix
languages.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
from functools import lru_cache
from pathlib import Path

CJK = re.compile(r"[㐀-鿿豈-﫿]")
#: Chinese punctuation and full-width forms (must not survive in English text).
CJK_PUNCT = re.compile(r"[　-〿！-／：-＠［-｀｛-･]")
CANONICAL = "zh"


class MissingText(KeyError):
    """A text that cannot be rendered in the requested language."""

    def __init__(self, what: str, lang: str, path: str = ""):
        super().__init__(what)
        self.what, self.lang, self.path = what, lang, path

    def __str__(self) -> str:
        where = f" at {self.path}" if self.path else ""
        return f"no {self.lang} text for {self.what!r}{where}"


_TRACE = __import__("os").environ.get("HAENV_I18N_TRACE")


def _trace(m, out) -> None:
    """`HAENV_I18N_TRACE=<file>`: log where a phrase was turned into new plain text."""
    if not _TRACE or not CJK.search(out) or len(out) < 2:
        return
    import traceback
    stack = [f"{f.filename.rsplit('/src/', 1)[-1]}:{f.lineno}:{f.name}"
             for f in traceback.extract_stack()[-6:-2]]
    with open(_TRACE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"from": str.__str__(m), "to": out[:120], "at": stack}, ensure_ascii=False) + "\n")


def has_cjk(s) -> bool:
    return isinstance(s, str) and bool(CJK.search(s))


class Msg(str):
    """Canonical text plus what it renders from: `kind` (`lex` | `tpl` | `join`), `key`, `slots`."""

    __slots__ = ("kind", "key", "slots")

    def __new__(cls, value: str, kind: str, key: str, slots=None):
        o = str.__new__(cls, value)
        o.kind, o.key, o.slots = kind, key, slots if slots is not None else {}
        return o

    def __reduce__(self):
        return (Msg, (str.__str__(self), self.kind, self.key, self.slots))

    # `str(m)` and a bare `f"{m}"` keep the phrase: generation code coerces text with them freely.
    def __str__(self):
        return self

    # A transform that changes nothing keeps the phrase; one that changes the text returns a plain
    # `str` (a new text the rendering layer does not know: wrap it with `rewrap`).
    def strip(self, chars=None):
        return self._same(str.strip(self, chars))

    def lstrip(self, chars=None):
        return self._same(str.lstrip(self, chars))

    def rstrip(self, chars=None):
        return self._same(str.rstrip(self, chars))

    def replace(self, old, new, count=-1):
        return self._same(str.replace(self, old, new, count))

    def _same(self, out):
        if out == self:
            return self
        _trace(self, out)
        return out

    def __format__(self, spec):
        return self if not spec else str.__format__(str.__str__(self), spec)

    def __deepcopy__(self, memo):
        return self                    # immutable once made

    def __copy__(self):
        return self

    def __repr__(self) -> str:
        # Messages stand in for the plain strings they replaced, and gate and audit messages
        # format containers of them with repr; keep that output identical to a plain string.
        return str.__repr__(str.__str__(self))

    def describe(self) -> str:
        """Debugging view: the message kind, its id and its canonical text."""
        return f"Msg({self.kind}:{self.key!r}={str.__str__(self)!r})"


if _TRACE:                                                      # debugging aid only
    def _traced(name):
        base = getattr(str, name)

        def f(self, *a, **k):
            out = base(self, *a, **k)
            for o in (out if isinstance(out, list) else [out]):
                if isinstance(o, str) and not isinstance(o, Msg) and o != self:
                    _trace(self, o)
            return out
        return f
    for _n in ("__add__", "__radd__", "__getitem__", "split", "rsplit", "partition", "__mod__", "format",
               "upper", "lower", "casefold", "removeprefix", "removesuffix", "translate"):
        if hasattr(str, _n):
            setattr(Msg, _n, _traced(_n))


def _yaml_representers() -> None:
    """YAML writers print a `Msg` as the plain string it is."""
    try:
        import yaml
    except ImportError:                                        # pragma: no cover
        return
    for dumper in (yaml.SafeDumper, yaml.Dumper):
        dumper.add_representer(Msg, lambda d, v: d.represent_str(str.__str__(v)))
    # Job generators use the optional libyaml dumpers too. They keep separate
    # representer tables, so registering the Python dumpers does not cover them.
    for name in ("CSafeDumper", "CDumper"):
        dumper = getattr(yaml, name, None)
        if dumper is not None:
            dumper.add_representer(Msg, lambda d, v: d.represent_str(str.__str__(v)))


_yaml_representers()


# --------------------------------------------------------------------------- assets
def i18n_dir(lang: str) -> Path:
    from haenv.locations import resource_path
    return resource_path(f"registry/i18n/{lang}")


def _yaml(p: Path) -> dict:
    from haenv.resources.yamlcache import load_yaml
    return load_yaml(p, wrap_text=False) or {}


@lru_cache(maxsize=None)
def templates(lang: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted((i18n_dir(lang) / "templates").glob("*.yaml")):
        for k, v in (_yaml(p) or {}).items():
            if k in out:
                raise ValueError(f"{p.name}: template {k!r} defined twice")
            out[str(k)] = str(v)
    return out


@lru_cache(maxsize=None)
def lexicon(lang: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted((i18n_dir(lang) / "lexicon").glob("*.yaml")):
        for k, v in (_yaml(p).get("entries") or {}).items():
            k, v = str(k), str(v)
            if k in out and out[k] != v:
                raise ValueError(f"{p.name}: {k!r} maps to {out[k]!r} and to {v!r}")
            out[k] = v
    return out


@lru_cache(maxsize=None)
def joiners(lang: str) -> dict[str, str]:
    p = i18n_dir(lang) / "joiners.yaml"
    return {str(k): str(v) for k, v in (_yaml(p) if p.is_file() else {}).items()}


def asset_files(lang: str) -> list[Path]:
    d = i18n_dir(lang)
    return sorted(p for p in d.rglob("*") if p.is_file()) if d.is_dir() else []


def render_fingerprint(lang: str) -> str:
    """sha256/16 over the rendering code and the language's assets (semantic, like the segments)."""
    from haenv.provenance import declarations
    from haenv.provenance.anchor import _fingerprint_of
    base = _fingerprint_of(declarations.RENDER)
    h = hashlib.sha256(f"{lang}|{base}".encode())
    root = i18n_dir(lang)
    for p in asset_files(lang):
        h.update(p.relative_to(root).as_posix().encode())
        from haenv.scoring.semantic.source import semantic_bytes
        h.update(semantic_bytes(p))
    return h.hexdigest()[:16]


render_sha16 = render_fingerprint


def judge_fingerprints(lang: str) -> dict:
    """Language-neutral scoring code plus this language's rubric and prompt assets."""
    from haenv.provenance import declarations
    from haenv.provenance.anchor import _fingerprint_of
    from haenv.scoring.semantic.source import semantic_bytes
    assets = hashlib.sha256(lang.encode())
    root = i18n_dir(lang)
    for p in asset_files(lang):
        if p.name.startswith('judge'):
            assets.update(p.relative_to(root).as_posix().encode())
            assets.update(semantic_bytes(p))
    return {field: {lang: hashlib.sha256(
        f'{_fingerprint_of(segment)}|{assets.hexdigest()}'.encode()).hexdigest()[:16]}
        for field, segment in (
            ('management_rubric_sha16', declarations.MANAGEMENT_JUDGING),
            ('semantic_policy_sha16', declarations.JUDGING))}


def reset() -> None:
    """Drop cached assets (tests and catalog tools re-read edited files)."""
    templates.cache_clear()
    lexicon.cache_clear()
    joiners.cache_clear()


# --------------------------------------------------------------------------- constructors
def L(msgid) -> Msg:
    """A lexicon phrase. `msgid` is a code id (`ns.name`, its zh text in the zh lexicon) or a
    registry phrase (its own canonical text)."""
    if isinstance(msgid, Msg):
        return msgid
    mid = str(msgid)
    return Msg(lexicon(CANONICAL).get(mid, mid), "lex", mid)


def text_of(key: str, lang: str | None = None, **slots) -> str:
    """Run-time text (prompts, menus, tool replies) in `lang` (default: the current language):
    template `key` filled with `slots`, or the lexicon entry `key` when there are no slots."""
    import haenv.i18n as _i18n
    lang = lang or _i18n.current()
    tpl = templates(lang).get(key)
    if tpl is not None:
        return tpl.format(**{k: render(v, lang) for k, v in slots.items()})
    if not slots:
        return render_msg(L(key), lang)
    raise MissingText(f"template {key}", lang)


def is_text(value, msgid: str) -> bool:
    """Whether `value` is the phrase `msgid`: by key for a `Msg`, else by its text in any
    language (a persisted record is plain text). Replaces comparisons with literal text."""
    if isinstance(value, Msg):
        return value.kind == "lex" and value.key == msgid
    if not isinstance(value, str):
        return False
    import haenv.i18n as _i18n
    for lang in _i18n.LANGUAGES:
        try:
            if value == render_msg(L(msgid), lang):
                return True
        except MissingText:
            continue
    return False


def T(key: str, **slots) -> Msg:
    """A composed text: template `key` filled with `slots` (canonical value: the zh rendering)."""
    tpl = templates(CANONICAL).get(key)
    if tpl is None:
        raise MissingText(key, CANONICAL)
    return Msg(tpl.format(**slots), "tpl", key, slots)


def J(key: str, items) -> Msg:
    """`items` joined with the language's separator `key`."""
    items = list(items)
    sep = joiners(CANONICAL).get(key)
    if sep is None:
        raise MissingText(f"joiner {key}", CANONICAL)
    return Msg(sep.join(str.__str__(x) if isinstance(x, str) else str(x) for x in items), "join", key, {"items": items})


def rewrap(orig, new):
    """The result of a text transform of `orig`: `orig` itself when nothing changed, else `new`
    as a phrase of its own (its canonical text is its message id)."""
    if new == orig:
        return orig
    return L(new) if has_cjk(new) else new


def numsub(orig, new, pairs):
    """`orig` with some of its numbers replaced (`pairs` of number strings, in order): `orig` when
    unchanged, else a phrase that renders `orig` in each language and replaces the same numbers."""
    if new == orig:
        return orig
    if isinstance(orig, Msg) and pairs:
        return Msg(new, "numsub", orig.key, {"base": orig, "subs": [list(p) for p in pairs]})
    return rewrap(orig, new)


def lex_ids(texts) -> list[Msg]:
    return [L(t) for t in texts]


def wrap(doc):
    """`doc` (parsed YAML or JSON) with every string that has Chinese text made a lexicon
    phrase, keys included. Registry and job text enters the world through here."""
    if isinstance(doc, Msg):
        return doc
    if isinstance(doc, str):
        return L(doc) if CJK.search(doc) else doc
    if isinstance(doc, dict):
        return {wrap(k): wrap(v) for k, v in doc.items()}
    if isinstance(doc, list):
        return [wrap(v) for v in doc]
    return doc


# --------------------------------------------------------------------------- rendering
_LOCK = threading.Lock()


def render_msg(m: Msg, lang: str, path: str = "") -> str:
    if m.kind == "lex":
        if lang == CANONICAL:
            return lexicon(lang).get(m.key, m.key)
        got = lexicon(lang).get(m.key)
        if got is None:
            if has_cjk(m.key) or m.key in lexicon(CANONICAL):
                raise MissingText(m.key, lang, path)
            return m.key                 # a registry phrase without Chinese text renders as itself
        return got
    if m.kind == "tpl":
        tpl = templates(lang).get(m.key)
        if tpl is None:
            raise MissingText(f"template {m.key}", lang, path)
        return tpl.format(**{k: render(v, lang, f"{path}{{{k}}}") for k, v in m.slots.items()})
    if m.kind == "numsub":
        out = render_msg(m.slots["base"], lang, path)
        for old, new in m.slots["subs"]:
            out, n = re.subn(rf"(?<![0-9.]){re.escape(str(old))}(?![0-9])", str(new), out, count=1)
            if not n:
                raise MissingText(f"number {old} in {out!r}", lang, path)
        return out
    if m.kind == "join":
        sep = joiners(lang).get(m.key)
        if sep is None:
            raise MissingText(f"joiner {m.key}", lang, path)
        return sep.join(str(render(x, lang, path)) for x in m.slots["items"])
    raise ValueError(f"unknown Msg kind {m.kind!r}")


def render(obj, lang: str, path: str = ""):
    """`obj` with every `Msg` rendered in `lang` (dicts, lists and tuples are rebuilt)."""
    if isinstance(obj, Msg):
        return render_msg(obj, lang, path)
    if isinstance(obj, str):
        if lang != CANONICAL and has_cjk(obj):
            raise MissingText(obj, lang, path)
        return obj
    if isinstance(obj, dict):
        return {(render(k, lang, path) if isinstance(k, Msg) else k): render(v, lang, f"{path}/{k}")
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [render(v, lang, f"{path}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, tuple):
        return tuple(render(v, lang, f"{path}[{i}]") for i, v in enumerate(obj))
    return obj


def plain(obj):
    """`obj` with every `Msg` turned into a plain `str` of its canonical value."""
    if isinstance(obj, Msg):
        return str.__str__(obj)
    if isinstance(obj, dict):
        return {plain(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [plain(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(plain(v) for v in obj)
    return obj


def missing(obj, lang: str, path: str = "") -> list[MissingText]:
    """Every text of `obj` that cannot be rendered in `lang` (a coverage tool)."""
    out: list[MissingText] = []

    def walk(o, p):
        if isinstance(o, Msg):
            try:
                render_msg(o, lang, p)
            except MissingText as e:
                out.append(e)
        elif isinstance(o, str):
            if lang != CANONICAL and has_cjk(o):
                out.append(MissingText(o, lang, p))
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{p}/{k}")
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                walk(v, f"{p}[{i}]")
    walk(obj, path)
    return out
