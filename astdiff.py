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
    # import-time differences other than a def's header (args, decorators ...)
    other_import: bool = False
    import_lines: set = field(default_factory=set)  # old lines of import-time diffs
    def_changes: list = field(default_factory=list)  # (qualname, old_first, old_last) of defs whose header changed
    toplevel_names: set = field(default_factory=set)  # names bound by changed module/class level statements
    class_quals: set = field(default_factory=set)  # qualnames of classes whose bodies changed at import level
    # one entry per runtime difference: (old lines, first line of enclosing function, its last line)
    runtime_groups: list = field(default_factory=list)
    # runtime differences that only swap the operator of a one-operator
    # comparison: (site "line:col:end_line:end_col", old op, new op, old lines, index in runtime_groups)
    swaps: list = field(default_factory=list)
    plain_lines: set = field(default_factory=set)  # runtime_lines from every other kind of difference


def _first(fn) -> int:
    """co_firstlineno of a function: decorated defs start at their first decorator."""
    decs = getattr(fn, "decorator_list", None) or []
    return min([fn.lineno] + [d.lineno for d in decs])


_COMPOUND = (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.AsyncFor, ast.AsyncWith, ast.Match, ast.TryStar)


def _dump(n) -> str:
    return ast.dump(n, include_attributes=False)


def _stmt_lines(n) -> set:
    return set(range(n.lineno, (n.end_lineno or n.lineno) + 1))


_BIG = 40  # lines: above this, compare by recursing instead of dumping


def _span(x):
    if not isinstance(x, ast.AST) or not hasattr(x, "lineno"):
        return None
    first = x.lineno
    decs = getattr(x, "decorator_list", None)
    if decs:
        first = min([first] + [d.lineno for d in decs])
    return first, (getattr(x, "end_lineno", None) or x.lineno)


class _Walker:
    def __init__(self, change: PyChange, old_lines=None, new_lines=None):
        self.c = change
        self.ol = old_lines
        self.nl = new_lines
        self._swap = None

    def _key(self, x, lines):
        """Source text of an element (whole lines). Equal text means an equal
        subtree, so only elements whose text differs need a closer look."""
        sp = _span(x)
        if sp is not None and lines is not None:
            return "\n".join(lines[sp[0] - 1 : sp[1]])
        return _dump_any(x)

    def node(self, a, b, ctx, stmt):
        if type(a) is not type(b):
            cmp = ctx[-1] if ctx else None
            if (isinstance(a, ast.cmpop) and isinstance(b, ast.cmpop) and isinstance(cmp, ast.Compare)
                    and len(cmp.ops) == 1 and hasattr(cmp, "end_col_offset")):
                key = f"{cmp.lineno}:{cmp.col_offset}:{cmp.end_lineno}:{cmp.end_col_offset}"
                self._swap = (key, type(a).__name__, type(b).__name__)
                try:
                    self.report(a, ctx, stmt, None)
                finally:
                    self._swap = None
                return
            self.report(a, ctx, stmt, None)
            return
        if isinstance(a, ast.AST):
            sp = _span(a)
            if (sp is None or sp[1] - sp[0] <= _BIG) and _dump(a) == _dump(b):
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
                elif va != vb or type(va) is not type(vb):  # True == 1 == 1.0
                    self.report(a, ctx, stmt, None)
            return
        if a != b:
            self.report(stmt, ctx, stmt, None)

    def seq(self, la, lb, ctx, stmt, parent):
        # A statement that is only a constant (a docstring, a bare string or
        # `...`) compiles to nothing; at most it sets __doc__, which nothing in
        # sqlglot or its tests reads at runtime. Compare without them.
        la = [x for x in la if not _noop(x)]
        lb = [x for x in lb if not _noop(x)]
        if not la and not lb:
            return
        da = [self._key(x, self.ol) for x in la]
        db = [self._key(x, self.nl) for x in lb]
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
            lines = {-_first(n)} | _stmt_lines(n)
            self.c.runtime_lines |= lines
            self.c.plain_lines |= lines
            self.c.import_time = True
            self.c.other_import = True
            self._bind(n, ctx)
            return
        if fn is not None:
            lines = set()
            if hasattr(n, "lineno"):
                lines |= set(range(n.lineno, (getattr(n, "end_lineno", None) or n.lineno) + 1))
            else:
                # operators (Eq, And, Add, ...) carry no position: use the
                # innermost enclosing node that does
                for anc in reversed(ctx):
                    if hasattr(anc, "lineno") and not isinstance(anc, FUNC):
                        lines |= set(range(anc.lineno, (getattr(anc, "end_lineno", None) or anc.lineno) + 1))
                        break
            if stmt is not None and hasattr(stmt, "lineno") and stmt is not fn:
                lines.add(stmt.lineno)
                if isinstance(stmt, _COMPOUND):
                    # the whole header expression: `if (` alone fires no line event
                    for f in ("test", "iter", "target", "subject"):
                        e = getattr(stmt, f, None)
                        if isinstance(e, ast.AST) and hasattr(e, "lineno"):
                            lines |= set(range(e.lineno, (e.end_lineno or e.lineno) + 1))
                    for it in getattr(stmt, "items", None) or []:
                        e = getattr(it, "context_expr", None)
                        if e is not None:
                            lines |= set(range(e.lineno, (e.end_lineno or e.lineno) + 1))
                else:
                    lines |= _stmt_lines(stmt)
            if insertion or not lines:
                lines.add(-_first(fn))
            if isinstance(n, FUNC) and n is not fn:
                lines.add(-_first(n))
            self.c.runtime_lines |= lines
            fn_first = _first(fn)
            fn_last = getattr(fn, "end_lineno", None) or fn.lineno
            if self._swap is not None and not insertion and fn_first not in (-l for l in lines):
                key, old_op, new_op = self._swap
                self.c.swaps.append((key, old_op, new_op, frozenset(lines), len(self.c.runtime_groups)))
            else:
                self.c.plain_lines |= lines
            self.c.runtime_groups.append((frozenset(lines), fn_first, fn_last))
            return
        # import-time difference
        self.c.import_time = True
        self.c.other_import = True
        if hasattr(n, "lineno"):
            self.c.import_lines |= set(range(n.lineno, (getattr(n, "end_lineno", None) or n.lineno) + 1))
        if isinstance(n, ast.ClassDef):
            self.c.class_quals.add(_qual(ctx, n.name))
        self._bind(n if isinstance(n, ast.stmt) else stmt, ctx)

    def report_def(self, fn, ctx):
        if isinstance(fn, ast.Lambda):
            self.c.runtime_lines |= {fn.lineno, -fn.lineno}
            self.c.plain_lines |= {fn.lineno, -fn.lineno}
            return
        q = _qual(ctx, fn.name)
        self.c.def_changes.append((q, _first(fn), fn.end_lineno))
        self.c.runtime_lines.add(-_first(fn))
        self.c.runtime_lines |= _stmt_lines(fn)
        self.c.plain_lines.add(-_first(fn))
        self.c.plain_lines |= _stmt_lines(fn)
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


def _future_annotations(tree) -> bool:
    return any(isinstance(s, ast.ImportFrom) and s.module == "__future__" and any(a.name == "annotations" for a in s.names)
               for s in tree.body)


class _Annotations(ast.NodeTransformer):
    """Blank annotations that never run. Under `from __future__ import
    annotations` a def's argument and return annotations become strings in
    its __annotations__, which nothing in sqlglot or its tests reads; an
    annotated assignment inside a function never evaluates its annotation
    (CPython compiles it away). Class- and module-level annotations stay:
    dataclasses, NamedTuple and TypedDict read those."""

    def __init__(self, defs: bool):
        self.defs = defs
        self.depth = 0

    def visit_FunctionDef(self, n):
        if self.defs:
            n.returns = None
            a = n.args
            for x in a.posonlyargs + a.args + a.kwonlyargs + [y for y in (a.vararg, a.kwarg) if y is not None]:
                x.annotation = None
        self.depth += 1
        self.generic_visit(n)
        self.depth -= 1
        return n

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, n):
        saved, self.depth = self.depth, 0  # a class body is its own (class) scope
        self.generic_visit(n)
        self.depth = saved
        return n

    def visit_AnnAssign(self, n):
        if self.depth > 0:
            n.annotation = ast.Constant(value=None)
        self.generic_visit(n)
        return n


def _strip_annotations(tree, defs: bool) -> None:
    """What _Annotations(defs).visit(tree) does, walking statement lists only:
    defs, classes and annotated assignments are statements, and no
    expression holds a statement."""

    def walk(stmts, depth):
        for n in stmts:
            t = type(n)
            if t is ast.FunctionDef or t is ast.AsyncFunctionDef:
                if defs:
                    n.returns = None
                    a = n.args
                    for x in a.posonlyargs + a.args + a.kwonlyargs + [y for y in (a.vararg, a.kwarg) if y is not None]:
                        x.annotation = None
                walk(n.body, depth + 1)
                continue
            if t is ast.ClassDef:
                walk(n.body, 0)  # a class body is its own (class) scope
                continue
            if t is ast.AnnAssign:
                if depth > 0:
                    n.annotation = ast.Constant(value=None)
                continue
            for f in ("body", "orelse", "finalbody"):
                sub = getattr(n, f, None)
                if sub:
                    walk(sub, depth)
            for h in getattr(n, "handlers", None) or ():
                walk(h.body, depth)
            for c in getattr(n, "cases", None) or ():
                walk(c.body, depth)

    walk(tree.body, 0)


def _noop(x) -> bool:
    return isinstance(x, ast.Expr) and isinstance(x.value, ast.Constant)


def _dump_any(x) -> str:
    if isinstance(x, ast.AST):
        return _dump(x)
    if isinstance(x, list):
        return "[" + ",".join(_dump_any(y) for y in x) + "]"
    return repr(x)


def _qual(ctx, name) -> str:
    parts = [c.name for c in ctx if isinstance(c, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
    return ".".join(parts + [name])


def analyze(old_src: str | None, new_src: str | None, old_changed=None, new_changed=None) -> PyChange:
    """old_changed/new_changed are accepted for compatibility and unused."""
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
        c.other_import = True
        return c
    if new_src is None:  # file deleted
        c.semantic = True
        c.import_time = True
        c.other_import = True
        return c
    if a is None:
        c.semantic = True
        c.import_time = True
        c.other_import = True
        return c
    import re

    cookie = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*([-\w.]+)")
    ha = [l for l in (old_src or "").splitlines()[:2] if cookie.match(l)]
    hb = [l for l in (new_src or "").splitlines()[:2] if cookie.match(l)]
    if ha != hb:
        c.semantic = True
        c.import_time = True
        c.other_import = True
    if old_src == new_src:
        # also compare the encoding-sensitive bits: the module docstring etc. are in the AST already
        return c
    defs = _future_annotations(a) and _future_annotations(b)
    _strip_annotations(a, defs)
    _strip_annotations(b, defs)
    _Walker(c, (old_src or "").split("\n"), (new_src or "").split("\n")).seq(a.body, b.body, (a,), None, a)
    c.semantic = bool(c.runtime_lines or c.import_time or c.def_changes or c.import_lines)
    return c
