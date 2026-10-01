"""`ast.unparse` with the output of Python 3.11 and later, on Python 3.10.

The judging, world and solving fingerprints hash source normalised by `ast.unparse`. Python 3.11
changed how it writes some code (bare tuples where the grammar allows them, `lambda:` without a
space, the f-string quote fallback), and 3.10's Unicode 13.0 database makes `repr` escape the
characters Unicode 14.0 added. Imported on 3.10 only, so that unchanged files fingerprint alike
on every supported version.
"""
from __future__ import annotations

import ast
import re
from ast import _ALL_QUOTES, _Precedence, _Unparser

#: Code points Unicode 14.0 assigned as printable characters.
_UNICODE_14 = (
    "61D 870-88E 898-89F 8B5 8C8-8D2 C3C C5D CDD 170D 1715 171F 180F 1AC1-1ACE 1B4C 1B7D-1B7E "
    "1DFA 20C0 2C2F 2C5F 2E53-2E5D 9FFD-9FFF A7C0-A7C1 A7D0-A7D1 A7D3 A7D5-A7D9 A7F2-A7F4 FBC2 "
    "FD40-FD4F FDCF FDFE-FDFF 10570-1057A 1057C-1058A 1058C-10592 10594-10595 10597-105A1 "
    "105A3-105B1 105B3-105B9 105BB-105BC 10780-10785 10787-107B0 107B2-107BA 10F70-10F89 "
    "11070-11075 110C2 116B9 11740-11746 11AB0-11ABF 12F90-12FF2 16A70-16ABE 16AC0-16AC9 "
    "1AFF0-1AFF3 1AFF5-1AFFB 1AFFD-1AFFE 1B11F-1B122 1CF00-1CF2D 1CF30-1CF46 1CF50-1CFC3 "
    "1D1E9-1D1EA 1DF00-1DF1E 1E290-1E2AE 1E7E0-1E7E6 1E7E8-1E7EB 1E7ED-1E7EE 1E7F0-1E7FE "
    "1F6DD-1F6DF 1F7F0 1F979 1F9CC 1FA7B-1FA7C 1FAA9-1FAAC 1FAB7-1FABA 1FAC3-1FAC5 1FAD7-1FAD9 "
    "1FAE0-1FAE7 1FAF0-1FAF6 2A6DE-2A6DF 2B735-2B738")
UNICODE_14_PRINTABLE = frozenset(
    c for part in _UNICODE_14.split() for a, _, b in [part.partition("-")]
    for c in range(int(a, 16), int(b or a, 16) + 1))
#: A `\u` or `\U` escape, not an escaped backslash followed by text.
_ESCAPE = re.compile(r"(?<!\\)((?:\\\\)*)\\(?:u([0-9a-f]{4})|U([0-9a-f]{8}))")


def unescape_unicode_14(text: str) -> str:
    """`text` (a repr or an escaped literal) with Unicode 14.0 characters written as themselves."""
    def put(m):
        cp = int(m.group(2) or m.group(3), 16)
        return m.group(1) + chr(cp) if cp in UNICODE_14_PRINTABLE else m.group(0)
    return _ESCAPE.sub(put, text)


class Unparser(_Unparser):
    def visit_NamedExpr(self, node):
        with self.delimit("(", ")"):
            self.set_precedence(_Precedence.ATOM, node.target, node.value)
            self.traverse(node.target)
            self.write(" := ")
            self.traverse(node.value)

    def visit_Assign(self, node):
        self.set_precedence(_Precedence.TUPLE, *node.targets)
        super().visit_Assign(node)

    def _for_helper(self, fill, node):
        self.set_precedence(_Precedence.TUPLE, node.target)
        super()._for_helper(fill, node)

    def visit_Tuple(self, node):
        with self.delimit_if("(", ")", not node.elts
                             or self.get_precedence(node) > _Precedence.TUPLE):
            self.items_view(self.traverse, node.elts)

    def visit_Subscript(self, node):
        self.set_precedence(_Precedence.ATOM, node.value)
        self.traverse(node.value)
        with self.delimit("[", "]"):
            if isinstance(node.slice, ast.Tuple) and node.slice.elts:
                self.items_view(self.traverse, node.slice.elts)
            else:
                self.traverse(node.slice)

    def visit_Lambda(self, node):
        with self.require_parens(_Precedence.TEST, node):
            self.write("lambda")
            outer, self._source = self._source, []
            self.traverse(node.args)
            args, self._source = self._source, outer
            if args:
                self.write(" " + "".join(args))
            self.write(": ")
            self.set_precedence(_Precedence.TEST, node.body)
            self.traverse(node.body)

    def visit_JoinedStr(self, node):
        if self._avoid_backslashes:
            return super().visit_JoinedStr(node)
        self.write("f")
        parts = []
        for value in node.values:
            getattr(self, "_fstring_" + type(value).__name__)(value, self.buffer_writer)
            parts.append((self.buffer, isinstance(value, ast.Constant)))
        out, quotes = [], list(_ALL_QUOTES)
        for value, is_constant in parts:
            value, narrowed = self._str_literal_helper(
                value, quote_types=quotes, escape_special_whitespace=is_constant)
            if set(narrowed).isdisjoint(quotes):
                quotes = ["'''"]
                out = [unescape_unicode_14(repr('"' + v))[2:-1] for v, _ in parts]
                break
            out.append(value)
            quotes = narrowed
        self.write(f"{quotes[0]}{''.join(out)}{quotes[0]}")

    def _write_constant(self, value):
        if isinstance(value, str) and not self._avoid_backslashes:
            self.write(unescape_unicode_14(repr(value)))
        else:
            super()._write_constant(value)

    def _str_literal_helper(self, string, **kwargs):
        text, quotes = super()._str_literal_helper(string, **kwargs)
        return unescape_unicode_14(text), quotes


def unparse(tree) -> str:
    return Unparser().visit(tree)
