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
import copy
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
    # qualnames (among def_changes) of undecorated defs whose headers differ
    # only in the values of constant defaults
    default_defs: set = field(default_factory=set)
    toplevel_names: set = field(default_factory=set)  # names bound by changed module/class level statements
    class_quals: set = field(default_factory=set)  # qualnames of classes whose bodies changed at import level
    # one entry per runtime difference: (old lines, first line of enclosing function, its last line)
    runtime_groups: list = field(default_factory=list)
    # runtime differences that only swap the operator of a one-operator
    # comparison: (site "line:col:end_line:end_col", old op, new op, old lines, index in runtime_groups)
    swaps: list = field(default_factory=list)
    # runtime differences that only swap one `and`/`or` of a chain:
    # (chain site "line:col:end_line:end_col", operator index, "And"/"Or", old lines, index in runtime_groups)
    bswaps: list = field(default_factory=list)
    plain_lines: set = field(default_factory=set)  # runtime_lines from every other kind of difference
    # module-level names renamed everywhere in the file, (old, new); the
    # caller must check that no other file mentions either
    global_renames: list = field(default_factory=list)


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


class _FlatBool(ast.NodeTransformer):
    """Nested chains of the same operator read as one: `a and (b and c)`
    evaluates exactly like `a and b and c` (same operands, same order, same
    value)."""

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        vals = []
        for v in node.values:
            if isinstance(v, ast.BoolOp) and type(v.op) is type(node.op):
                vals.extend(v.values)
            else:
                vals.append(v)
        node.values = vals
        return node


def _flat_dump(e) -> str:
    return _dump(_FlatBool().visit(copy.deepcopy(e)))


def _flipped(n: ast.BoolOp, i: int) -> ast.expr:
    """Chain n with its i-th operator swapped, as Python parses that source:
    `and` binds tighter than `or`."""
    vals = n.values
    if isinstance(n.op, ast.And):
        left, right = vals[: i + 1], vals[i + 1 :]
        lft = left[0] if len(left) == 1 else ast.BoolOp(op=ast.And(), values=left)
        rgt = right[0] if len(right) == 1 else ast.BoolOp(op=ast.And(), values=right)
        return ast.BoolOp(op=ast.Or(), values=[lft, rgt])
    rest = vals[:i] + [ast.BoolOp(op=ast.And(), values=[vals[i], vals[i + 1]])] + vals[i + 2 :]
    return rest[0] if len(rest) == 1 else ast.BoolOp(op=ast.Or(), values=rest)


def _bool_flip(a: ast.BoolOp, b: ast.BoolOp):
    """(chain, i) when b is a with only the i-th `and`/`or` of one chain
    inside a swapped (compared with same-operator chains flattened), else None."""
    target = _flat_dump(b)
    for n in ast.walk(a):
        if not isinstance(n, ast.BoolOp) or not hasattr(n, "end_col_offset"):
            continue
        for i in range(len(n.values) - 1):
            memo: dict = {}
            root = copy.deepcopy(a, memo)
            nc = memo[id(n)]
            new = _flipped(nc, i)
            if root is nc:
                root = new
            else:
                for p in ast.walk(root):
                    for f, v in ast.iter_fields(p):
                        if v is nc:
                            setattr(p, f, new)
                        elif isinstance(v, list):
                            for j, x in enumerate(v):
                                if x is nc:
                                    v[j] = new
            if _flat_dump(root) == target:
                return n, i
    return None


class _Walker:
    def __init__(self, change: PyChange, old_lines=None, new_lines=None, old_src=None, new_src=None, alpha: bool = True):
        self.c = change
        self.alpha = alpha
        self.ol = old_lines
        self.nl = new_lines
        self._swap = None
        self._bswap = None
        self._src = (old_src, new_src)
        self._code: list = [None, None]
        self.ta = self.tb = frozenset()  # names bound to the typing module (old, new)

    def _same_code(self, fa, fb) -> bool:
        """The compiled functions match instruction for instruction, with the
        same constants, global and attribute names and parameters: only the
        names of locals may differ."""
        for i in (0, 1):
            if self._code[i] is None:
                try:
                    self._code[i] = compile(self._src[i], "<m>", "exec", dont_inherit=True)
                except Exception:  # noqa: BLE001
                    self._code[i] = False
        if not self._code[0] or not self._code[1]:
            return False
        ca = _find_code(self._code[0], fa)
        cb = _find_code(self._code[1], fb)
        return ca is not None and cb is not None and _code_equal(ca, cb)

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
            if isinstance(a, ast.BoolOp) and isinstance(b, ast.BoolOp) and self._swap is None and self._bswap is None:
                fl = _bool_flip(a, b)
                if fl is not None:
                    chain, i = fl
                    key = f"{chain.lineno}:{chain.col_offset}:{chain.end_lineno}:{chain.end_col_offset}"
                    self._bswap = (key, i, type(chain.op).__name__)
                    try:
                        self.report(a, ctx, stmt, None)
                    finally:
                        self._bswap = None
                    return
            if isinstance(a, FUNC):
                # header vs body
                hdr_a = [getattr(a, f, None) for f in ("args", "decorator_list", "returns", "name", "type_params")]
                hdr_b = [getattr(b, f, None) for f in ("args", "decorator_list", "returns", "name", "type_params")]
                hdr_same = all(_dump_any(x) == _dump_any(y) for x, y in zip(hdr_a, hdr_b))
                if hdr_same and self.alpha and _alpha_equal(a, b) and self._same_code(a, b):
                    return  # the same code with its local variables renamed
                if not hdr_same:
                    self.report_def(a, ctx)
                    if _defaults_only(a, b):
                        self.c.default_defs.add(_qual(ctx, a.name))
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
        if isinstance(parent, (ast.Module, ast.ClassDef)) and self._func_ctx(ctx) is None:
            # A def that the next statement rebinds (an overload stub before
            # its implementation) is never seen: compare without those.
            ka, kb = _dead_defs(la, self.ta), _dead_defs(lb, self.tb)
            if ka or kb:
                if [_dump(la[i]) for i in sorted(ka)] != [_dump(lb[j]) for j in sorted(kb)]:
                    # their headers still run while the module imports: the
                    # snapshot child imports the patched module, so a new
                    # import error is still seen
                    self.c.import_time = True
                la = [x for i, x in enumerate(la) if i not in ka]
                lb = [x for j, x in enumerate(lb) if j not in kb]
        if not la and not lb:
            return
        da = [self._key(x, self.ol) for x in la]
        db = [self._key(x, self.nl) for x in lb]
        if da == db:
            return
        if isinstance(parent, (ast.Module, ast.ClassDef)) and _only_moved_defs(la, lb, da, db):
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
            if isinstance(fn, ast.Lambda):
                # a lambda's body runs only when it is called, in its own
                # frame; the statement around it (a dict of many lambdas, a
                # call that takes one) runs its other lines without it
                b = fn.body
                lines |= set(range(b.lineno, (getattr(b, "end_lineno", None) or b.lineno) + 1))
            elif stmt is not None and hasattr(stmt, "lineno") and stmt is not fn:
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
            elif self._bswap is not None and not insertion and fn_first not in (-l for l in lines):
                key, i, op = self._bswap
                self.c.bswaps.append((key, i, op, frozenset(lines), len(self.c.runtime_groups)))
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


# ------------------------------------------------------------ renamed locals
# A function whose new body is the old one with some of its local variables
# consistently renamed behaves the same: Python resolves a name by where it is
# bound, and every binding and use of a renamed name moves with it. Only plain
# variable names may differ (never parameters, attributes, keywords, imports,
# except/match captures or def names), each renamed name must resolve to a
# binding inside the function at every use (never a global or builtin), on
# both sides, and the function must not reach its own frame or namespace.
_FRAME_NAMES = {"locals", "vars", "eval", "exec", "dir", "globals", "compile", "breakpoint", "__import__"}
_FRAME_ATTRS = {"_getframe", "currentframe", "f_locals", "f_back", "f_globals", "f_code", "stack", "trace",
                "getouterframes", "getinnerframes", "getframeinfo", "co_varnames", "co_names", "__code__"}
_COMPS = (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)


class _Unsupported(Exception):
    pass


def _shape(x, y, m: dict, rev: dict) -> bool:
    """Equal trees except for ast.Name ids, which must correspond one to one."""
    if type(x) is not type(y):
        return False
    if isinstance(x, ast.AST):
        for f in x._fields:
            vx, vy = getattr(x, f, None), getattr(y, f, None)
            if (f == "id" and isinstance(x, ast.Name)) or (f == "name" and isinstance(x, ast.ExceptHandler) and vx and vy):
                if m.setdefault(vx, vy) != vy or rev.setdefault(vy, vx) != vx:
                    return False
            elif not _shape(vx, vy, m, rev):
                return False
        return True
    if isinstance(x, list):
        return len(x) == len(y) and all(_shape(p, q, m, rev) for p, q in zip(x, y))
    return x == y


class _Scope:
    __slots__ = ("parent", "comp", "bound", "params", "uses")

    def __init__(self, parent, comp=False):
        self.parent = parent
        self.comp = comp
        self.bound: set = set()
        self.params: set = set()
        self.uses: list = []


def _params(a: ast.arguments) -> set:
    out = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
    for x in (a.vararg, a.kwarg):
        if x is not None:
            out.add(x.arg)
    return out


def _scopes(fn) -> list:
    """The scopes inside fn (fn's own first), with the names each binds and uses."""
    root = _Scope(None)
    root.params = _params(fn.args)
    root.bound |= root.params
    out = [root]

    def new(parent, comp=False):
        sc = _Scope(parent, comp)
        out.append(sc)
        return sc

    def visit(n, sc):
        if isinstance(n, (ast.ClassDef, ast.Global, ast.Nonlocal)):
            raise _Unsupported
        if isinstance(n, ast.Name):
            if n.id in _FRAME_NAMES:
                raise _Unsupported
            if not isinstance(n.ctx, ast.Load):
                sc.bound.add(n.id)
            sc.uses.append(n.id)
            return
        if isinstance(n, ast.Attribute) and n.attr in _FRAME_ATTRS:
            raise _Unsupported
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            if not isinstance(n, ast.Lambda):
                sc.bound.add(n.name)
                sc.uses.append(n.name)
                for d in n.decorator_list:
                    visit(d, sc)
                if n.returns is not None:
                    visit(n.returns, sc)
            a = n.args
            for d in a.defaults + [d for d in a.kw_defaults if d is not None]:
                visit(d, sc)
            for x in a.posonlyargs + a.args + a.kwonlyargs + [y for y in (a.vararg, a.kwarg) if y is not None]:
                if x.annotation is not None:
                    visit(x.annotation, sc)
            child = new(sc)
            child.params = _params(a)
            child.bound |= child.params
            for x in n.body if isinstance(n.body, list) else [n.body]:
                visit(x, child)
            return
        if isinstance(n, _COMPS):
            gens = n.generators
            visit(gens[0].iter, sc)  # the first iterable runs in the enclosing scope
            child = new(sc, comp=True)
            for i, g in enumerate(gens):
                if i:
                    visit(g.iter, child)
                visit(g.target, child)
                for c in g.ifs:
                    visit(c, child)
            for x in ((n.key, n.value) if isinstance(n, ast.DictComp) else (n.elt,)):
                visit(x, child)
            return
        if isinstance(n, ast.NamedExpr):
            t = sc
            while t.comp:  # binds in the nearest enclosing function scope
                t = t.parent
            t.bound.add(n.target.id)
            t.uses.append(n.target.id)
            visit(n.value, sc)
            return
        if isinstance(n, ast.ExceptHandler) and n.name:
            sc.bound.add(n.name)
            sc.uses.append(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                nm = al.asname or al.name.split(".")[0]
                sc.bound.add(nm)
                sc.uses.append(nm)
        elif isinstance(n, (ast.MatchAs, ast.MatchStar)) and n.name:
            sc.bound.add(n.name)
            sc.uses.append(n.name)
        elif isinstance(n, ast.MatchMapping) and n.rest:
            sc.bound.add(n.rest)
            sc.uses.append(n.rest)
        for c in ast.iter_child_nodes(n):
            visit(c, sc)

    for x in fn.body if isinstance(fn.body, list) else [fn.body]:
        visit(x, root)
    return out


def _all_local(fn, names: set) -> bool:
    """Every use of each name resolves to a non-parameter binding inside fn."""
    try:
        scopes = _scopes(fn)
    except _Unsupported:
        return False
    for sc in scopes:
        for nm in sc.uses:
            if nm not in names:
                continue
            t = sc
            while t is not None and nm not in t.bound:
                t = t.parent
            if t is None or nm in t.params:
                return False
    return True


def _alpha_equal(fa, fb) -> bool:
    m: dict = {}
    rev: dict = {}
    try:
        if not _shape(fa, fb, m, rev):
            return False
    except RecursionError:
        return False
    old = {x for x, y in m.items() if x != y}
    if not old:
        return False
    return _all_local(fa, old) and _all_local(fb, {m[x] for x in old})


def _find_code(module_code, fn):
    """The code object compiled for def/lambda node fn, or None (or None if ambiguous)."""
    name = "<lambda>" if isinstance(fn, ast.Lambda) else fn.name
    first = fn.lineno if isinstance(fn, ast.Lambda) else _first(fn)
    found = []
    stack = [module_code]
    while stack:
        co = stack.pop()
        for k in co.co_consts:
            if isinstance(k, type(module_code)):
                if k.co_name == name and k.co_firstlineno == first:
                    found.append(k)
                stack.append(k)
    return found[0] if len(found) == 1 else None


def _code_equal(ca, cb) -> bool:
    if (ca.co_code != cb.co_code or ca.co_names != cb.co_names or ca.co_flags != cb.co_flags
            or ca.co_argcount != cb.co_argcount or ca.co_posonlyargcount != cb.co_posonlyargcount
            or ca.co_kwonlyargcount != cb.co_kwonlyargcount or ca.co_nlocals != cb.co_nlocals
            or ca.co_stacksize != cb.co_stacksize or ca.co_exceptiontable != cb.co_exceptiontable
            or ca.co_name != cb.co_name or ca.co_qualname != cb.co_qualname
            or len(ca.co_varnames) != len(cb.co_varnames) or len(ca.co_cellvars) != len(cb.co_cellvars)
            or len(ca.co_freevars) != len(cb.co_freevars)):
        return False
    if _FRAME_NAMES.intersection(ca.co_names) or _FRAME_ATTRS.intersection(ca.co_names):
        return False
    npar = ca.co_argcount + ca.co_kwonlyargcount + bool(ca.co_flags & 0x04) + bool(ca.co_flags & 0x08)
    if ca.co_varnames[:npar] != cb.co_varnames[:npar]:
        return False
    if len(ca.co_consts) != len(cb.co_consts):
        return False
    for x, y in zip(ca.co_consts, cb.co_consts):
        if isinstance(x, type(ca)):
            if not isinstance(y, type(ca)) or not _code_equal(x, y):
                return False
        elif type(x) is not type(y) or repr(x) != repr(y):
            return False
    return True


# --------------------------------------------------------------- moved defs
def _movable(x) -> bool:
    """A def whose definition evaluates nothing (no decorators, constant
    defaults, no evaluated annotations): defining it earlier or later among
    other such defs binds the same function to the same name."""
    if not isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) or x.decorator_list or getattr(x, "type_params", None):
        return False
    a = x.args
    if any(not isinstance(d, ast.Constant) for d in a.defaults + [d for d in a.kw_defaults if d is not None]):
        return False
    every = a.posonlyargs + a.args + a.kwonlyargs + [y for y in (a.vararg, a.kwarg) if y is not None]
    return x.returns is None and all(y.annotation is None for y in every)


def _only_moved_defs(la: list, lb: list, da: list, db: list) -> bool:
    """A module or class body whose only change is the order of movable defs
    between the same other statements (each name defined once per run). Only
    the order of names in the namespace differs, which sqlglot never reads
    (it looks names up, and lists members through sorted inspect.getmembers)."""
    if len(la) != len(lb):
        return False

    def runs(stmts, keys):
        fixed, out, cur = [], [], []
        for x, k in zip(stmts, keys):
            if _movable(x):
                cur.append((k, x.name))
            else:
                out.append(cur)
                cur = []
                fixed.append(k)
        out.append(cur)
        return fixed, out

    fa, ra = runs(la, da)
    fb, rb = runs(lb, db)
    if fa != fb or len(ra) != len(rb):
        return False
    moved = False
    for x, y in zip(ra, rb):
        if x == y:
            continue
        names = [n for _, n in x]
        if sorted(x) != sorted(y) or len(names) != len(set(names)):
            return False
        moved = True
    return moved


# ----------------------------------------------------------- reordered imports
_PINNED_ROOTS = {"sqlglot", "tests", "__future__"}


def _plain_import(s):
    """Names an import of a module outside sqlglot and the tests binds, else None."""
    if isinstance(s, ast.Import):
        if any(al.name.split(".")[0] in _PINNED_ROOTS for al in s.names):
            return None
        return [al.asname or al.name.split(".")[0] for al in s.names]
    if isinstance(s, ast.ImportFrom):
        if s.level or not s.module or s.module.split(".")[0] in _PINNED_ROOTS:
            return None
        if any(al.name == "*" for al in s.names):
            return None
        return [al.asname or al.name for al in s.names]
    return None


def _permuted_imports(ba: list, bb: list) -> set:
    """Indexes of a contiguous run of module-level imports that the change only
    reorders (outside sqlglot, each name bound once): the module ends up with
    the same bindings to the same already-importable modules. Empty if none."""
    if len(ba) != len(bb):
        return set()
    # only imports are dumped (a module's class bodies are large); any other
    # statement that differs is the walker's to report as usual
    imp = (ast.Import, ast.ImportFrom)
    diff = [i for i in range(len(ba)) if (isinstance(ba[i], imp) or isinstance(bb[i], imp)) and _dump(ba[i]) != _dump(bb[i])]
    if not diff:
        return set()
    run = range(diff[0], diff[-1] + 1)
    names: list = []
    for i in run:
        na, nb = _plain_import(ba[i]), _plain_import(bb[i])
        if na is None or nb is None:
            return set()
        names += na
    if len(names) != len(set(names)) or sorted(_dump(ba[i]) for i in run) != sorted(_dump(bb[i]) for i in run):
        return set()
    return set(run)


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


def _literal(x) -> bool:
    """A constant, or a signed number literal: evaluating it does nothing else."""
    if isinstance(x, ast.UnaryOp) and isinstance(x.op, (ast.USub, ast.UAdd)):
        x = x.operand
        return isinstance(x, ast.Constant) and type(x.value) in (int, float, complex)
    return isinstance(x, ast.Constant)


def _defaults_only(a, b) -> bool:
    """Two undecorated defs whose headers differ only in the values of
    constant defaults (same parameters, kinds, annotations and return)."""
    if not isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef)) or type(a) is not type(b):
        return False
    if a.decorator_list or b.decorator_list:
        return False
    for f in ("returns", "name", "type_params"):
        if _dump_any(getattr(a, f, None)) != _dump_any(getattr(b, f, None)):
            return False
    aa, ba = a.args, b.args
    if len(aa.defaults) != len(ba.defaults) or len(aa.kw_defaults) != len(ba.kw_defaults):
        return False
    for x, y in zip(list(aa.defaults) + list(aa.kw_defaults), list(ba.defaults) + list(ba.kw_defaults)):
        if (x is None) != (y is None):
            return False
        if x is None or (_literal(x) and _literal(y)):
            continue
        if _dump(x) != _dump(y):
            return False

    def rest(args):
        c = copy.copy(args)
        c.defaults = []
        c.kw_defaults = []
        return _dump(c)

    return rest(aa) == rest(ba)


def _typing_names(tree) -> frozenset:
    """Names a module binds to the typing module (`import typing [as t]`)."""
    out = set()
    for s in tree.body:
        if isinstance(s, ast.Import):
            for al in s.names:
                if al.name == "typing":
                    out.add(al.asname or "typing")
    return frozenset(out)


def _loads(nodes, name: str) -> bool:
    return any(isinstance(x, ast.Name) and x.id == name and not isinstance(x.ctx, ast.Store)
               for n in nodes if n is not None for x in ast.walk(n))


def _header(fn) -> list:
    a = fn.args
    params = a.posonlyargs + a.args + a.kwonlyargs + [x for x in (a.vararg, a.kwarg) if x is not None]
    return list(fn.decorator_list) + list(a.defaults) + list(a.kw_defaults) + [p.annotation for p in params] + [fn.returns]


def _dead_defs(stmts: list, typing_names) -> set:
    """Indices of defs in a module or class body that the very next statement
    rebinds with another def of the same name, which does not read the name
    while its header runs. Such a binding is never seen (no metaclass in
    sqlglot defines __prepare__). Building it must do nothing else: no
    decorator other than typing.overload (which only records the stub in a
    registry nothing reads), literal defaults only, no annotation left to
    evaluate (they were blanked under `from __future__ import annotations`)."""
    out = set()
    for i in range(len(stmts) - 1):
        a, b = stmts[i], stmts[i + 1]
        if not (isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef)) and isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))):
            continue
        if a.name != b.name or getattr(a, "type_params", None):
            continue
        if not all(isinstance(d, ast.Attribute) and d.attr == "overload" and isinstance(d.value, ast.Name)
                   and d.value.id in typing_names for d in a.decorator_list):
            continue
        h = _header(a)
        if any(x is not None and not _literal(x) for x in h[len(a.decorator_list):]):
            continue
        if _loads(_header(b), b.name):
            continue
        out.add(i)
    return out


def _dump_any(x) -> str:
    if isinstance(x, ast.AST):
        return _dump(x)
    if isinstance(x, list):
        return "[" + ",".join(_dump_any(y) for y in x) + "]"
    return repr(x)


def _qual(ctx, name) -> str:
    parts = [c.name for c in ctx if isinstance(c, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
    return ".".join(parts + [name])


# ------------------------------------------------ renamed module-level names
# A function or constant bound once at module level and renamed, with every
# use in the file renamed too, changes nothing the program can observe as
# long as (a) no other file refers to either name (the caller checks), (b)
# nothing reads it by a string (getattr, __all__ ...), and (c) for a
# function, nothing sees the function object itself, only calls it: its
# __name__, __qualname__ and code name are the only things that change.


def _binders(tree) -> dict:
    """Module-level names bound exactly once, by an undecorated def or a
    plain assignment to that one name: name -> statement. (Not annotated
    assignments: the module's __annotations__ is keyed by the name.)"""
    seen: dict = {}
    for st in tree.body:
        n = None
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
            n = st.name if not st.decorator_list else None
            if n is None:
                seen[st.name] = None
        elif isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            n = st.targets[0].id
        if n is not None:
            seen[n] = st if n not in seen else None
    return {k: v for k, v in seen.items() if v is not None}


def _word_in(text: str, name: str) -> bool:
    import re

    return re.search(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])", text) is not None


def _mentioned(tree, name: str, binder=None, calls_only: bool = False) -> bool:
    """Whether tree refers to name other than through binder (its binding
    statement) and loads of it (only as the callee of a call when
    calls_only). Strings count, except statements that are only a constant
    (docstrings)."""
    parent: dict = {}
    noop_consts = set()
    for p in ast.walk(tree):
        for ch in ast.iter_child_nodes(p):
            parent[ch] = p
        if isinstance(p, ast.Expr) and isinstance(p.value, ast.Constant):
            noop_consts.add(id(p.value))
    for x in ast.walk(tree):
        if x is binder:
            if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                    isinstance(y, ast.ClassDef) for y in ast.walk(x)):
                return True  # a class defined inside would change its __qualname__
            continue
        if isinstance(x, ast.Name):
            if x.id != name:
                continue
            p = parent.get(x)
            if binder is None:
                return True
            if p is binder:
                continue  # the binding target itself
            if not isinstance(x.ctx, ast.Load):
                return True
            if calls_only and not (isinstance(p, ast.Call) and p.func is x):
                return True
            continue
        if isinstance(x, ast.Constant):
            if isinstance(x.value, str) and id(x) not in noop_consts and _word_in(x.value, name):
                return True
            continue
        for f in ("name", "attr", "arg", "asname", "rest", "id"):
            v = getattr(x, f, None)
            if isinstance(v, str) and (v == name or (f == "name" and isinstance(x, ast.alias) and _word_in(v, name))):
                return True
        for f in ("names", "kwd_attrs"):
            v = getattr(x, f, None)
            if isinstance(v, list) and any(isinstance(y, str) and y == name for y in v):
                return True
    return False


def _frame_free(fn) -> bool:
    for x in ast.walk(fn):
        if isinstance(x, ast.Name) and x.id in _FRAME_NAMES:
            return False
        if isinstance(x, ast.Attribute) and x.attr in _FRAME_ATTRS:
            return False
    return True


class _Rename(ast.NodeTransformer):
    def __init__(self, old: str, new: str, binder):
        self.old, self.new, self.binder = old, new, binder

    def visit_Name(self, n):
        if n.id == self.old:
            n.id = self.new
        return n

    def generic_visit(self, n):
        if n is self.binder and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            n.name = self.new
        return super().generic_visit(n)


def _global_renames(a, b) -> list:
    """Pairs (old, new) of module-level names renamed consistently in the
    whole file. Rewrites b to use the old names, so the rest of the diff sees
    only the other changes."""
    ba, bb = _binders(a), _binders(b)
    gone = [n for n in ba if n not in bb and not any(isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and s.name == n for s in b.body)]
    new = [n for n in bb if n not in ba and not any(isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and s.name == n for s in a.body)]
    if not gone or len(gone) != len(new):
        return []
    out = []
    for o, n in zip(gone, new):
        sa, sb = ba[o], bb[n]
        func = isinstance(sa, (ast.FunctionDef, ast.AsyncFunctionDef))
        if func != isinstance(sb, (ast.FunctionDef, ast.AsyncFunctionDef)) or type(sa) is not type(sb):
            continue
        if not (_frame_free(a) and _frame_free(b)):  # globals(), vars() ... see the module's names
            return out
        if (_mentioned(a, o, sa, func) or _mentioned(b, n, sb, func)
                or _mentioned(a, n) or _mentioned(b, o)):
            continue
        _Rename(n, o, sb).visit(b)
        out.append((o, n))
    return out


# ------------------------------------------------------------ renamed methods
# A private method renamed together with every call in its file changes
# nothing observable when: both names are plain private names (not dunder or
# sunder hooks, not *_sql, which the generator dispatches on by name, not the
# namedtuple-style names other code duck-types on); every use is the def of
# a method or an attribute load on self, cls or super(); every class that
# defines it has only sqlglot bases (so no outside framework calls it by
# name); and (the caller checks) no other file mentions either name.

_DUCK_PRIVATE = {"_asdict", "_replace", "_make", "_fields", "_field_defaults", "_source"}
_METHOD_DECOS = {"property", "staticmethod", "classmethod"}


def _private_name(n: str) -> bool:
    return (n.isidentifier() and n.startswith("_") and not n.startswith("__") and not n.endswith("_")
            and not n.endswith("_sql") and n not in _DUCK_PRIVATE)


def _internal_roots(tree) -> set:
    """Names that refer to sqlglot code in this module: its classes and what
    it imports from sqlglot (or relatively)."""
    out = {"object"}
    for x in ast.walk(tree):
        if isinstance(x, ast.ClassDef):
            out.add(x.name)
        elif isinstance(x, ast.ImportFrom) and (x.level or (x.module or "").split(".")[0] == "sqlglot"):
            out.update(a.asname or a.name for a in x.names)
        elif isinstance(x, ast.Import):
            for a in x.names:
                if a.name.split(".")[0] == "sqlglot":
                    out.add(a.asname or a.name.split(".")[0])
    return out


def _internal_class(c, roots: set) -> bool:
    def ok(e):
        if isinstance(e, ast.Subscript):  # t.Generic[E], Generic[...]
            d = _dotted(e.value)
            return d in ("t.Generic", "typing.Generic", "Generic")
        d = _dotted(e)
        return d is not None and d.split(".")[0] in roots
    return all(ok(b) for b in c.bases) and all(k.arg == "metaclass" and ok(k.value) for k in c.keywords)


def _method_defs(tree) -> dict:
    """name -> [(class qualname, ordinal among that class's defs, def, class)]"""
    out: dict = {}

    def walk(body, qual):
        for st in body:
            if isinstance(st, ast.ClassDef):
                q = qual + st.name
                k = 0
                for x in st.body:
                    if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        out.setdefault(x.name, []).append((q, k, x, st))
                        k += 1
                walk(st.body, q + ".")
            elif isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                walk(st.body, qual + st.name + ".<locals>.")
            else:
                for f in ("body", "orelse", "finalbody"):
                    walk(getattr(st, f, None) or [], qual)
                for h in getattr(st, "handlers", None) or ():
                    walk(h.body, qual)

    walk(tree.body, "")
    return out


def _method_uses_ok(tree, name: str, defs: list, roots: set) -> bool:
    if not defs:
        return False
    def_ids = {id(d) for _, _, d, _ in defs}
    for _, _, d, c in defs:
        if any(not (isinstance(x, ast.Name) and x.id in _METHOD_DECOS) for x in d.decorator_list):
            return False
        if not _internal_class(c, roots):
            return False
    noop_consts = {id(p.value) for p in ast.walk(tree) if isinstance(p, ast.Expr) and isinstance(p.value, ast.Constant)}
    for x in ast.walk(tree):
        if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x.name == name:
            if id(x) not in def_ids:
                return False
            continue
        if isinstance(x, ast.Attribute) and x.attr == name:
            v = x.value
            if not isinstance(x.ctx, ast.Load):
                return False
            if not ((isinstance(v, ast.Name) and v.id in ("self", "cls"))
                    or (isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "super" and not v.args)):
                return False
            continue
        if isinstance(x, ast.Constant):
            if isinstance(x.value, str) and id(x) not in noop_consts and _word_in(x.value, name):
                return False
            continue
        if isinstance(x, ast.Name) and x.id == name:
            return False
        for f in ("name", "arg", "asname", "rest"):
            v = getattr(x, f, None)
            if isinstance(v, str) and (v == name or (isinstance(x, ast.alias) and _word_in(v, name))):
                return False
        for f in ("names", "kwd_attrs"):
            v = getattr(x, f, None)
            if isinstance(v, list) and any(isinstance(y, str) and y == name for y in v):
                return False
    return True


class _AttrRename(ast.NodeTransformer):
    def __init__(self, old: str, new: str):
        self.old, self.new = old, new

    def visit_Attribute(self, n):
        if n.attr == self.old:
            n.attr = self.new
        return self.generic_visit(n)

    def visit_FunctionDef(self, n):
        if n.name == self.old:
            n.name = self.new
        return self.generic_visit(n)

    visit_AsyncFunctionDef = visit_FunctionDef


def _method_renames(a, b) -> list:
    """Pairs (old, new) of private methods renamed consistently in the file
    (matched by class and position). Rewrites b to use the old names."""
    ma, mb = _method_defs(a), _method_defs(b)
    pos_b = {(q, k): n for n, ds in mb.items() for q, k, _, _ in ds}
    pairs: dict = {}
    for o, ds in ma.items():
        if o in mb or not _private_name(o):
            continue
        for q, k, _, _ in ds:
            n = pos_b.get((q, k))
            if n is None or n in ma or not _private_name(n):
                pairs[o] = None
                break
            if pairs.setdefault(o, n) != n:
                pairs[o] = None
                break
    targets = [n for n in pairs.values() if n is not None]
    if not targets:
        return []
    ra, rb = _internal_roots(a), _internal_roots(b)
    out = []
    for o, n in pairs.items():
        if n is None or targets.count(n) != 1:
            continue
        if len(ma[o]) != len(mb.get(n, ())):
            continue
        if not (_method_uses_ok(a, o, ma[o], ra) and _method_uses_ok(b, n, mb[n], rb)):
            continue
        if _mentioned(a, n) or _mentioned(b, o):
            continue
        _AttrRename(n, o).visit(b)
        out.append((o, n))
    return out


def analyze(old_src: str | None, new_src: str | None, old_changed=None, new_changed=None, renames: bool = True) -> PyChange:
    """old_changed/new_changed are accepted for compatibility and unused.
    renames=False: no rename counts as no change (for code a test inspects:
    renaming changes co_names / co_varnames)."""
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
    if renames:
        c.global_renames = _global_renames(a, b) + _method_renames(a, b)
    skip = _permuted_imports(a.body, b.body)
    ba = [x for i, x in enumerate(a.body) if i not in skip]
    bb = [x for i, x in enumerate(b.body) if i not in skip]
    w = _Walker(c, (old_src or "").split("\n"), (new_src or "").split("\n"), old_src, new_src, renames)
    w.ta, w.tb = _typing_names(a), _typing_names(b)
    w.seq(ba, bb, (a,), None, a)
    c.semantic = bool(c.runtime_lines or c.import_time or c.def_changes or c.import_lines)
    return c


# ------------------------------------------- import-time side effects of tests
# pytest imports every test module (and what they import) while it collects,
# before any test runs, in every process that runs tests. So a test module
# whose import changes shared state (patches a sqlglot class, edits a
# registry, sets an environment variable ...) can break tests in any other
# file. side_effects() lists the statements of a test-side module that run at
# import and might do that: everything except imports, docstrings, plain
# defs and test classes, and names bound to values computed without calls
# (or with a few pure builtins). Comparing the lists of the old and new
# versions tells whether a patch adds, removes or changes any of them.

_SAFE_CALLS = {
    "os.path.join", "os.path.dirname", "os.path.abspath", "os.path.realpath", "os.path.normpath",
    "os.path.basename", "os.path.splitext", "os.path.isdir", "os.path.isfile", "os.path.exists",
    "os.environ.get", "os.getenv", "re.compile", "frozenset", "set", "list", "dict", "tuple", "str",
    "int", "float", "bool", "bytes", "len", "sorted", "reversed", "range", "enumerate", "zip", "min",
    "max", "sum", "any", "all", "isinstance", "issubclass", "repr", "string_to_bool", "is_compiled",
}
_SAFE_DECOS = {
    "classmethod", "staticmethod", "property",
    "unittest.skip", "unittest.skipIf", "unittest.skipUnless", "unittest.expectedFailure",
    "skip", "skipIf", "skipUnless", "expectedFailure",
    "mock.patch", "mock.patch.object", "mock.patch.dict", "mock.patch.multiple",
    "unittest.mock.patch", "unittest.mock.patch.object", "unittest.mock.patch.dict",
    "unittest.mock.patch.multiple", "patch", "patch.object", "patch.dict", "patch.multiple",
    "pytest.mark.skip", "pytest.mark.skipif",  # (not xfail or timeout: pytest's verdict would differ from a plain run)
}
_TEST_BASES = {"unittest.TestCase", "TestCase", "object"}


def _dotted(n) -> str | None:
    parts = []
    while isinstance(n, ast.Attribute):
        parts.append(n.attr)
        n = n.value
    if not isinstance(n, ast.Name):
        return None
    parts.append(n.id)
    return ".".join(reversed(parts))


def _pure(n) -> bool:
    if n is None:
        return True
    for x in ast.walk(n):
        if isinstance(x, (ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr)):
            return False
        if isinstance(x, ast.Call) and _dotted(x.func) not in _SAFE_CALLS:
            return False
    return True


def _deco_ok(d) -> bool:
    if isinstance(d, ast.Call):
        return _dotted(d.func) in _SAFE_DECOS and all(_pure(a) for a in d.args) and all(_pure(k.value) for k in d.keywords)
    return _dotted(d) in _SAFE_DECOS


def _names_only(t) -> bool:
    if isinstance(t, ast.Name):
        return True
    if isinstance(t, ast.Starred):
        return _names_only(t.value)
    if isinstance(t, (ast.Tuple, ast.List)):
        return all(_names_only(e) for e in t.elts)
    return False


def _main_guard(s) -> bool:
    t = s.test
    return (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id == "__name__"
            and len(t.ops) == 1 and isinstance(t.ops[0], ast.Eq)
            and isinstance(t.comparators[0], ast.Constant) and t.comparators[0].value == "__main__")


def _se_body(body, ctx: str, ok_bases: set, out: set) -> None:
    for s in body:
        if isinstance(s, (ast.Import, ast.Pass)):
            continue
        if isinstance(s, ast.ImportFrom):
            if s.level or (s.module or "").split(".")[0] == "tests":
                ok_bases.update(a.asname or a.name for a in s.names)  # test helpers (Validator ...)
            continue
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant):
            continue
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = s.args
            hdr = [*a.defaults, *a.kw_defaults, s.returns,
                   *[x.annotation for x in (*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg) if x is not None]]
            if not all(_deco_ok(d) for d in s.decorator_list) or not all(_pure(x) for x in hdr):
                out.add(ctx + "def " + s.name + " " + _dump(ast.Tuple(elts=[*s.decorator_list, *[x for x in hdr if x is not None]], ctx=ast.Load())))
            continue
        if isinstance(s, ast.ClassDef):
            if (all(_deco_ok(d) for d in s.decorator_list) and not s.keywords
                    and all(_dotted(b) in _TEST_BASES or _dotted(b) in ok_bases for b in s.bases)):
                ok_bases.add(s.name)
            else:  # a metaclass (Dialect, Expression ...) may register it globally
                out.add(ctx + "class " + s.name + " " + _dump(ast.Tuple(elts=[*s.decorator_list, *s.bases, *[k.value for k in s.keywords]], ctx=ast.Load())))
            _se_body(s.body, ctx + s.name + ".", set(ok_bases), out)
            continue
        if isinstance(s, (ast.Assign, ast.AnnAssign)):
            tg = s.targets if isinstance(s, ast.Assign) else [s.target]
            if all(_names_only(t) for t in tg) and _pure(s.value) and (isinstance(s, ast.Assign) or _pure(s.annotation)):
                continue
        if isinstance(s, ast.If) and _main_guard(s):
            continue
        out.add(ctx + _dump(s))


def side_effects(src: str | None) -> set:
    """Signatures of the statements of a test-side module that run when it is
    imported and may change state other tests share. A module that does not
    parse never runs, so it has none."""
    if src is None:
        return set()
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return set()
    out: set = set()
    _se_body(tree.body, "", set(), out)
    return out
