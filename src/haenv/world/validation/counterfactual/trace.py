"""Trace condition leaves on an isolated runtime copy of a production rule."""
from __future__ import annotations

import ast
from contextlib import contextmanager
from contextvars import ContextVar
import inspect
import textwrap

_ACTIVE = ContextVar("counterfactual_run", default=None)


@contextmanager
def recording(override=None):
    """Record several instrumented calls under one context-local override map."""
    events = []
    token = _ACTIVE.set((override or {}, events))
    try:
        yield events
    finally:
        _ACTIVE.reset(token)


def _tree(func):
    if func.__code__.co_freevars:
        raise ValueError("rule functions with free variables are unsupported")
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != func.__name__:
        raise ValueError("expected exactly one matching FunctionDef")
    return tree


def _leaves(node):
    if isinstance(node, ast.BoolOp):
        for value in node.values:
            yield from _leaves(value)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        yield from _leaves(node.operand)
    else:
        yield node


def atoms_of(func):
    """Return original leaf texts and their dedented source line numbers."""
    atoms = {}
    for node in ast.walk(_tree(func)):
        if isinstance(node, (ast.If, ast.IfExp, ast.While)):
            for leaf in _leaves(node.test):
                atoms.setdefault(ast.unparse(leaf), []).append(leaf.lineno)
    return {text: sorted(lines) for text, lines in atoms.items()}


class _Instrument(ast.NodeTransformer):
    def _test(self, node):
        if isinstance(node, ast.BoolOp):
            node.values = [self._test(v) for v in node.values]
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            node.operand = self._test(node.operand)
        else:
            node = ast.copy_location(ast.Call(func=ast.Name(id="__cf_atom__", ctx=ast.Load()),
                args=[ast.Constant(ast.unparse(node)), ast.Constant(node.lineno), node], keywords=[]), node)
        return node

    def _branch(self, node):
        # Wrap before visiting children so text always describes the original leaf.
        node.test = self._test(node.test)
        return self.generic_visit(node)

    visit_If = _branch
    visit_IfExp = _branch
    visit_While = _branch


class RuleTrace:
    """Preserve rule semantics, with context-local recording and text-keyed overrides."""
    def __init__(self, func, *, label, replace_globals=None):
        self.func, self.label = func, label
        self.sources = tuple(source for replacement in (replace_globals or {}).values()
                             for source in getattr(replacement, "__cf_sources__", ())) + (func,)

        def atom(text, lineno, value):
            state = _ACTIVE.get()
            if state is None:
                return value
            override, events = state
            eff = override[text] if text in override else value
            events.append((label, text, lineno, bool(value), bool(eff)))
            return eff if text in override else value

        tree = ast.fix_missing_locations(_Instrument().visit(_tree(func)))
        ns = dict(func.__globals__)
        ns.update(replace_globals or {})
        ns["__cf_atom__"] = atom
        exec(compile(tree, filename=f"<counterfactual:{label}>", mode="exec"), ns)
        self.instrumented = ns[func.__name__]
        self.instrumented.__cf_sources__ = self.sources

    def run(self, *args, override=None, **kwargs):
        with recording(override) as events:
            return self.instrumented(*args, **kwargs), events

    def instrumented_equals(self, *args, **kwargs):
        return self.run(*args, **kwargs)[0] == self.func(*args, **kwargs)
