"""Find what a change to a Python file actually changes.

Walks the old and new ASTs together and reports the smallest differing
subtrees. Each one is classified:

* runtime: inside a function or lambda body. Only code that runs it can
  notice, so its old line span is looked up in the per-test line coverage.
* import: anywhere else (class bodies, module level, def headers,
  decorators, default values). Its effect is measured by fingerprinting the
  patched module state at import time.

Comment-only and formatting-only edits produce no differences at all.
"""

from __future__ import annotations

import ast
import difflib
from dataclasses import dataclass, field

FUNC = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_POS = ("lineno", "col_offset", "end_lineno", "end_col_offset")


@dataclass
class PyChange:
    syntax_error: bool = False
    semantic: bool = False
    runtime_lines: set = field(default_factory=set)  # old line numbers; negative = function entry
    import_time: bool = False
    import_lines: set = field(default_factory=set)  # old lines of import-time diffs
    def_changes: list = field(default_factory=list)  # (qualname, old_first, old_last) of defs whose header changed
    toplevel_names: set = field(default_factory=set)  # names bound by changed module/class level statements
    class_quals: set = field(default_factory=set)  # qualnames of classes whose bodies changed at import level


def _dump(n) -> str:
    return ast.dump(n, include_attributes=False)


def _stmt_lines(n) -> set:
    return set(range(n.lineno, (n.end_lineno or n.lineno) + 1))


class _Walker:
    def __init__(self, change: PyChange):
        self.c = change

    # ctx: tuple of ancestor nodes in the OLD tree (outermost first)
    def node(self, a, b, ctx, stmt):
        if type(a) is not type(b):
            self.report(a, ctx, stmt, None)
            return
        if isinstance(a, ast.AST):
            if _dump(a) == _dump(b):
                return
            if isinstance(a, ast.stmt):
                stmt = a
            if isinstance(a, FUNC):
                # header vs body
                hdr_a = [getattr(a, f, None) for f in ("args", "decorator_list", "returns", "name", "type_params")]
                hdr_b = [getattr(b, f, None) for f in ("args", "decorator_list", "returns", "name", "type_params")]
                hdr_same = all(_dump_any(x) == _dump_any(y) for x, y in zip(hdr_a, hdr_b))
                if not hdr_same:
                    self.report_def(a, ctx)
                body_a = a.body if isinstance(a.body, list) else [a.body]
                body_b = b.body if isinstance(b.body, list) else [b.body]
                self.seq(body_a, body_b, ctx + (a,), stmt, a)
                return
            for f in a._fields:
                va, vb = getattr(a, f, None), getattr(b, f, None)
                if isinstance(va, list) and isinstance(vb, list):
                    self.seq(va, vb, ctx + (a,), stmt, a)
                elif isinstance(va, ast.AST) or isinstance(vb, ast.AST):
                    if va is None or vb is None:
                        self.report(a, ctx, stmt, None)
                    else:
                        self.node(va, vb, ctx + (a,), stmt)
                elif va != vb:
                    self.report(a, ctx, stmt, None)
            return
        if a != b:
            self.report(stmt, ctx, stmt, None)

    def seq(self, la, lb, ctx, stmt, parent):
        da = [_dump_any(x) for x in la]
        db = [_dump_any(x) for x in lb]
        if da == db:
            return
        sm = difflib.SequenceMatcher(a=da, b=db, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            if tag == "replace" and (i2 - i1) == (j2 - j1):
                for x, y in zip(la[i1:i2], lb[j1:j2]):
                    if isinstance(x, ast.AST) and isinstance(y, ast.AST):
                        self.node(x, y, ctx, stmt)
                    elif x != y:
                        self.report(parent, ctx[:-1], stmt, None)
                continue
            if tag in ("replace", "delete"):
                for x in la[i1:i2]:
                    if isinstance(x, ast.AST):
                        self.report(x, ctx, stmt if not isinstance(x, ast.stmt) else x, None)
                    else:
                        self.report(parent, ctx[:-1], stmt, None)
            if tag in ("replace", "insert"):
                # inserted elements: blame the neighbours in the old sequence
                neighbours = []
                if i1 - 1 >= 0:
                    neighbours.append(la[i1 - 1])
                if i2 < len(la):
                    neighbours.append(la[i2])
                if tag == "replace":
                    neighbours.extend(la[i1:i2])
                if not neighbours or not all(isinstance(n, ast.AST) for n in neighbours):
                    self.report(parent, ctx[:-1], stmt, None)
                else:
                    for n in neighbours:
                        self.report(n, ctx, stmt if not isinstance(n, ast.stmt) else n, None, insertion=True)

    def _func_ctx(self, ctx):
        """Innermost function/lambda whose BODY contains this position."""
        for i in range(len(ctx) - 1, -1, -1):
            if isinstance(ctx[i], FUNC):
                return ctx[i]
        return None

    def report(self, n, ctx, stmt, _unused, insertion=False):
        fn = self._func_ctx(ctx)
        if isinstance(n, FUNC) and fn is None and not insertion:
            # a whole def replaced/removed at class or module level
            self.report_def(n, ctx)
            lines = {-n.lineno} | _stmt_lines(n)
            self.c.runtime_lines |= lines
            self.c.import_time = True
            self._bind(n, ctx)
            return
        if fn is not None:
            lines = set()
            if hasattr(n, "lineno"):
                lines |= set(range(n.lineno, (getattr(n, "end_lineno", None) or n.lineno) + 1))
            if stmt is not None and hasattr(stmt, "lineno") and stmt is not fn:
                lines.add(stmt.lineno)
                if not isinstance(stmt, (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.AsyncFor, ast.AsyncWith, ast.Match, ast.TryStar)):
                    lines |= _stmt_lines(stmt)
            if insertion or not lines:
                lines.add(-fn.lineno)
            if isinstance(n, FUNC) and n is not fn:
                lines.add(-n.lineno)
            self.c.runtime_lines |= lines
            return
        # import-time difference
        self.c.import_time = True
        if hasattr(n, "lineno"):
            self.c.import_lines |= set(range(n.lineno, (getattr(n, "end_lineno", None) or n.lineno) + 1))
        if isinstance(n, ast.ClassDef):
            self.c.class_quals.add(_qual(ctx, n.name))
        self._bind(n if isinstance(n, ast.stmt) else stmt, ctx)

    def report_def(self, fn, ctx):
        if isinstance(fn, ast.Lambda):
            self.c.runtime_lines |= {fn.lineno, -fn.lineno}
            return
        q = _qual(ctx, fn.name)
        self.c.def_changes.append((q, fn.lineno, fn.end_lineno))
        self.c.runtime_lines.add(-fn.lineno)
        self.c.runtime_lines |= _stmt_lines(fn)
        self.c.import_time = True
        self.c.import_lines.add(fn.lineno)
        self._bind(fn, ctx)

    def _bind(self, stmt, ctx):
        if stmt is None:
            return
        cls = [c for c in ctx if isinstance(c, ast.ClassDef)]
        if cls:
            self.c.class_quals.add(_qual(ctx[: ctx.index(cls[-1])], cls[-1].name))
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            self.c.toplevel_names.add(stmt.name)
        for t in ast.walk(stmt) if isinstance(stmt, ast.stmt) else ():
            if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store):
                self.c.toplevel_names.add(t.id)
            elif isinstance(t, (ast.Import, ast.ImportFrom)):
                for al in t.names:
                    self.c.toplevel_names.add((al.asname or al.name).split(".")[0])


def _dump_any(x) -> str:
    if isinstance(x, ast.AST):
        return _dump(x)
    if isinstance(x, list):
        return "[" + ",".join(_dump_any(y) for y in x) + "]"
    return repr(x)


def _qual(ctx, name) -> str:
    parts = [c.name for c in ctx if isinstance(c, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
    return ".".join(parts + [name])


def analyze(old_src: str | None, new_src: str | None) -> PyChange:
    c = PyChange()
    try:
        a = ast.parse(old_src or "")
    except SyntaxError:
        a = None
    try:
        b = ast.parse(new_src or "") if new_src is not None else None
    except SyntaxError:
        c.syntax_error = True
        c.semantic = True
        c.import_time = True
        return c
    if new_src is None:  # file deleted
        c.semantic = True
        c.import_time = True
        return c
    if a is None:
        c.semantic = True
        c.import_time = True
        return c
    if _dump(a) == _dump(b):
        # also compare the encoding-sensitive bits: the module docstring etc. are in the AST already
        return c
    c.semantic = True
    _Walker(c).seq(a.body, b.body, (a,), None, a)
    return c
