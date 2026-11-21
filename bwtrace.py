"""And/or-swap profile (build time only).

Another frequent kind of edit swaps one `and` for `or` (or the reverse) in a
chain of boolean operators. Such an edit can change a test's outcome only if,
somewhere while that test runs, the swapped chain gives a different result
or evaluates something the original did not (or skips something it did).

This tracer imports sqlglot with every `and`/`or` chain inside a function
rewritten into a call that evaluates the operands exactly as Python would,
then works out, for each single swap, what the swapped chain would have done
with the same operand values:

  - `a and b and c` with its i-th `and` swapped reads (a and .. ) or (.. c);
  - `a or b or c` with its i-th `or` swapped reads a or (b and c) ...

An operand the swapped chain would evaluate but the original did not is
evaluated here only through a guarded copy of its code that refuses anything
that could change state (a call other than isinstance/len/bool/type/
callable/hasattr, arithmetic on non-builtin values, a property whose getter
stores something or that is cached, a module's or class's __getattr__,
...). An operand the original evaluated but the swap would skip must pass the
same guard. Where the chain's value is only tested for truth (an if or while
test, `not`, bool(), assert, a comprehension filter, or a chain inside such
a chain), equal truth suffices; elsewhere the swapped value must be the same
object. Anything that cannot be decided counts as a difference.

Each test runs in its own forked child; the child reports, per chain, which
swaps ever made a difference. The selector charges a swap only to those
tests.

usage: python bwtrace.py SQLGLOT_ROOT tests.json OUT_DIR [JOBS]
"""

from __future__ import annotations

import ast
import builtins
import copy
import functools
import importlib.machinery
import json
import operator
import os
import resource
import sys
import time
import types
import unittest

ROOT = os.path.realpath(sys.argv[1])
TESTS_JSON = sys.argv[2]
OUT = sys.argv[3]
JOBS = int(sys.argv[4]) if len(sys.argv) > 4 else os.cpu_count() or 2
ONLY = set(os.environ.get("TRACE_ONLY", "").split(",")) - {""}
PREFIX = ROOT + "/"

sys.dont_write_bytecode = True
sys.path.insert(0, ROOT)
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
os.environ.setdefault("SKIP_INTEGRATION", "0")

SITES: list = []  # sid -> (file, "line:col:eline:ecol", "And"/"Or", n operands, truth-only)
INFO: list = []  # sid -> (is_and, n, truth_only, variant trees)
FULL: list = []

_FORBIDDEN_NAMES = {"super", "__class__", "locals", "vars", "globals", "eval", "exec", "dir"}


def _variants(is_and: bool, n: int) -> list:
    out = []
    for i in range(n - 1):
        if is_and:
            left = 0 if i == 0 else ("and", list(range(i + 1)))
            right = n - 1 if i == n - 2 else ("and", list(range(i + 1, n)))
            out.append(("or", [left, right]))
        else:
            kids = list(range(i)) + [("and", [i, i + 1])] + list(range(i + 2, n))
            out.append(("or", kids) if len(kids) > 1 else kids[0])
    return out


def _register(rel: str, node: ast.BoolOp, truth_only: bool) -> int:
    is_and = isinstance(node.op, ast.And)
    n = len(node.values)
    sid = len(SITES)
    SITES.append((rel, f"{node.lineno}:{node.col_offset}:{node.end_lineno}:{node.end_col_offset}", "And" if is_and else "Or", n, truth_only))
    INFO.append((is_and, n, truth_only, _variants(is_and, n)))
    FULL.append((1 << (n - 1)) - 1)
    return sid


def _lam(body: ast.expr) -> ast.Lambda:
    args = ast.arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    return ast.Lambda(args=args, body=body)


def _call(name: str, args: list) -> ast.Call:
    return ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=args, keywords=[])


class _Impure(Exception):
    pass


class _Guard(ast.NodeTransformer):
    """A copy of an operand that evaluates it without changing any state, or
    raises: every attribute read, subscript, call, comparison, arithmetic or
    `not` goes through a checking helper. Unknown node kinds -> _Impure."""

    ok = True

    def generic_visit(self, node):
        if not isinstance(node, (ast.Name, ast.Constant, ast.Load, ast.Tuple, ast.List, ast.Set, ast.BoolOp, ast.IfExp,
                                 ast.And, ast.Or, ast.expr_context)):
            self.ok = False
        return super().generic_visit(node)

    def visit_Name(self, node):
        if not isinstance(node.ctx, ast.Load) or node.id in _FORBIDDEN_NAMES:
            self.ok = False
        return node

    def visit_Attribute(self, node):
        if not isinstance(node.ctx, ast.Load) or node.attr.startswith("__") and not node.attr.endswith("__"):
            self.ok = False  # (a class-private name would need mangling)
            return node
        return _call("__brA__", [self.visit(node.value), ast.Constant(node.attr)])

    def visit_Subscript(self, node):
        if not isinstance(node.ctx, ast.Load) or isinstance(node.slice, ast.Slice):
            self.ok = False
            return node
        return _call("__brS__", [self.visit(node.value), self.visit(node.slice)])

    def visit_BinOp(self, node):
        return _call("__brB__", [ast.Constant(type(node.op).__name__), self.visit(node.left), self.visit(node.right)])

    def visit_UnaryOp(self, node):
        return _call("__brU__", [ast.Constant(type(node.op).__name__), self.visit(node.operand)])

    def visit_Compare(self, node):
        ops = ast.Tuple([ast.Constant(type(o).__name__) for o in node.ops], ast.Load())
        thunks = ast.Tuple([_lam(self.visit(x)) for x in [node.left] + list(node.comparators)], ast.Load())
        return _call("__brK__", [ops, thunks])

    def visit_Call(self, node):
        f = node.func
        if (isinstance(f, ast.Attribute) and isinstance(f.ctx, ast.Load) and not node.keywords
                and not (f.attr.startswith("__") and not f.attr.endswith("__"))
                and not any(isinstance(a, ast.Starred) for a in node.args)):
            return _call("__brM__", [self.visit(f.value), ast.Constant(f.attr)] + [self.visit(a) for a in node.args])
        if (not isinstance(node.func, ast.Name) or node.keywords
                or node.func.id not in ("isinstance", "len", "bool", "type", "callable", "hasattr")
                or any(isinstance(a, ast.Starred) for a in node.args)):
            self.ok = False
            return node
        return _call("__brC__", [node.func] + [self.visit(a) for a in node.args])


def _guarded(expr: ast.expr):
    g = _Guard()
    out = g.visit(copy.deepcopy(expr))
    return out if g.ok else None


def _uses_forbidden(node: ast.AST) -> bool:
    for x in ast.walk(node):
        if isinstance(x, (ast.NamedExpr, ast.Yield, ast.YieldFrom, ast.Await)):
            return True
        if isinstance(x, ast.Name) and x.id in _FORBIDDEN_NAMES:
            return True
    return False


class _T(ast.NodeTransformer):
    def __init__(self, rel):
        self.rel = rel
        self.parents: dict = {}

    def visit_Module(self, node):
        for p in ast.walk(node):
            for c in ast.iter_child_nodes(p):
                self.parents[c] = p
        return self.generic_visit(node)

    def generic_visit(self, node):
        # leave annotations alone (as wmtrace.py)
        for field, old in ast.iter_fields(node):
            if field in ("annotation", "returns"):
                continue
            if isinstance(old, list):
                new = []
                for v in old:
                    if isinstance(v, ast.AST):
                        v = self.visit(v)
                        if v is None:
                            continue
                        if not isinstance(v, ast.AST):
                            new.extend(v)
                            continue
                    new.append(v)
                old[:] = new
            elif isinstance(old, ast.AST):
                nv = self.visit(old)
                if nv is None:
                    delattr(node, field)
                else:
                    setattr(node, field, nv)
        return node

    def _in_function(self, node) -> bool:
        """Inside a function's or lambda's body (decorators, defaults and
        annotations are evaluated in the enclosing scope)."""
        child, p = node, self.parents.get(node)
        while p is not None:
            if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)) and child in p.body:
                return True
            if isinstance(p, ast.Lambda) and child is p.body:
                return True
            if isinstance(p, (ast.ClassDef, ast.Module)):
                return False
            child, p = p, self.parents.get(p)
        return False

    def _truth_only(self, node) -> bool:
        p = self.parents.get(node)
        if isinstance(p, (ast.If, ast.While, ast.IfExp, ast.Assert)) and p.test is node:
            return True
        if isinstance(p, ast.comprehension) and node in p.ifs:
            return True
        if isinstance(p, ast.UnaryOp) and isinstance(p.op, ast.Not):
            return True
        if (isinstance(p, ast.Call) and isinstance(p.func, ast.Name) and p.func.id == "bool"
                and len(p.args) == 1 and not p.keywords and p.args[0] is node):
            return True
        if isinstance(p, ast.BoolOp):
            return self._truth_only(p)
        return False

    def visit_BoolOp(self, node):
        if not self._in_function(node) or _uses_forbidden(node):
            return self.generic_visit(node)
        truth_only = self._truth_only(node)
        guarded = [_guarded(v) for v in node.values]
        sid = _register(self.rel, node, truth_only)
        self.generic_visit(node)  # nested chains inside the operands
        call = _call("__brb__", [
            ast.Constant(sid),
            ast.Tuple([_lam(v) for v in node.values], ast.Load()),
            ast.Tuple([_lam(g) if g is not None else ast.Constant(None) for g in guarded], ast.Load()),
        ])
        return ast.copy_location(call, node)


class _Loader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        path = self.get_filename(fullname)
        return self.source_to_code(self.get_data(path), path)

    def source_to_code(self, data, path, *, _optimize=-1):
        tree = ast.parse(data, filename=path)
        tree = _T(path[len(PREFIX):]).visit(tree)
        ast.fix_missing_locations(tree)
        return compile(tree, path, "exec", dont_inherit=True)


class _Finder:
    @classmethod
    def find_spec(cls, name, path=None, target=None):
        if name != "sqlglot" and not name.startswith("sqlglot."):
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is not None and spec.origin and spec.origin.endswith(".py") and os.path.realpath(spec.origin).startswith(PREFIX):
            spec.loader = _Loader(name, spec.origin)
        return spec

    @classmethod
    def invalidate_caches(cls):
        pass


sys.meta_path.insert(0, _Finder)

# ------------------------------------------------------------- guard helpers
_BUILTIN_SCALARS = (int, float, complex, bool, str, bytes, type(None))
_BUILTIN_CONTAINERS = (list, tuple, dict, set, frozenset, str, bytes, range,
                       type({}.keys()), type({}.values()), type({}.items()))
_MUTATORS = {"set", "append", "appendleft", "pop", "popleft", "popitem", "update", "setdefault", "add", "replace",
             "extend", "insert", "clear", "remove", "discard", "sort", "reverse", "__setitem__", "__delitem__",
             "setattr", "delattr", "add_comments", "pop_comments", "replace_children", "set_meta"}
_PROP_OK: dict = {}


def _getter_pure(fget) -> bool:
    """A property getter counts as pure when its own code stores nothing and
    calls no mutating method (checked once per getter, from its source)."""
    r = _PROP_OK.get(fget)
    if r is not None:
        return r
    r = False
    try:
        import inspect
        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(fget)))
        r = True
        for x in ast.walk(tree):
            if isinstance(x, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
                tg = x.targets if isinstance(x, (ast.Assign, ast.Delete)) else [x.target]
                if any(not isinstance(t, ast.Name) for t in tg):
                    r = False
            elif isinstance(x, (ast.Global, ast.Nonlocal, ast.NamedExpr, ast.Yield, ast.YieldFrom)):
                r = False
            elif isinstance(x, ast.Call):
                f = x.func
                nm = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
                if nm in _MUTATORS:
                    r = False
    except Exception:  # noqa: BLE001
        r = False
    _PROP_OK[fget] = r
    return r


def _static(klasses, name):
    for k in klasses:
        d = k.__dict__
        if name in d:
            return True, d[name]
    return False, None


def __brA__(obj, name):  # noqa: N807
    if isinstance(obj, types.ModuleType):
        d = obj.__dict__
        if name in d:
            return d[name]
        raise _Impure  # a module __getattr__ (lazy imports) would run
    tp = type(obj)
    if tp.__getattribute__ is not object.__getattribute__ and not (isinstance(obj, type) and tp.__getattribute__ is type.__getattribute__):
        raise _Impure
    if isinstance(obj, type):
        found, attr = _static(obj.__mro__, name)
        if found:
            if isinstance(attr, (types.FunctionType, classmethod, staticmethod, property, functools.cached_property)) or not hasattr(type(attr), "__get__"):
                return getattr(obj, name)
            raise _Impure
        found, attr = _static(tp.__mro__, name)
        if found:
            if isinstance(attr, property):
                if attr.fget is None or not _getter_pure(attr.fget):
                    raise _Impure
                return getattr(obj, name)
            if isinstance(attr, (types.FunctionType, classmethod, staticmethod)) or not hasattr(type(attr), "__get__"):
                return getattr(obj, name)
            raise _Impure
        if _static(tp.__mro__, "__getattr__")[0]:
            raise _Impure
        return getattr(obj, name)
    found, attr = _static(tp.__mro__, name)
    inst = getattr(obj, "__dict__", None)
    if found:
        at = type(attr)
        if isinstance(attr, property):
            if attr.fget is None or not _getter_pure(attr.fget):
                raise _Impure
            return getattr(obj, name)
        if at.__name__ in ("member_descriptor", "getset_descriptor"):
            return getattr(obj, name)
        if hasattr(at, "__set__") or hasattr(at, "__delete__"):
            raise _Impure  # some other data descriptor
        if inst is not None and name in inst:
            return inst[name]
        if isinstance(attr, functools.cached_property):
            raise _Impure  # computing it now would cache a value
        if isinstance(attr, (types.FunctionType, classmethod, staticmethod, types.BuiltinFunctionType,
                             types.MethodDescriptorType, types.WrapperDescriptorType)) or not hasattr(at, "__get__"):
            return getattr(obj, name)
        raise _Impure
    if inst is not None and name in inst:
        return inst[name]
    if _static(tp.__mro__, "__getattr__")[0]:
        raise _Impure
    return getattr(obj, name)


def __brS__(obj, key):  # noqa: N807
    if type(obj) not in _BUILTIN_CONTAINERS:
        raise _Impure
    if type(key) not in _BUILTIN_SCALARS and type(key) is not tuple and type(obj) is not list and type(obj) is not tuple:
        raise _Impure  # hashing an arbitrary key may run its __hash__
    return obj[key]


_BINOPS = {"Add": operator.add, "Sub": operator.sub, "Mult": operator.mul, "Div": operator.truediv,
           "FloorDiv": operator.floordiv, "Mod": operator.mod, "Pow": operator.pow, "LShift": operator.lshift,
           "RShift": operator.rshift, "BitOr": operator.or_, "BitXor": operator.xor, "BitAnd": operator.and_}


def __brB__(op, a, b):  # noqa: N807
    if type(a) not in (int, float, complex, bool) or type(b) not in (int, float, complex, bool):
        if not (op in ("Add", "Mult") and type(a) in (str, bytes, tuple) and type(b) in (str, bytes, tuple, int)):
            raise _Impure
        if type(a) is tuple or type(b) is tuple:
            raise _Impure
    if op == "Pow" and type(b) is int and abs(b) > 64:
        raise _Impure
    if op in ("LShift", "Mult") and type(b) is int and b > 4096:
        raise _Impure
    return _BINOPS[op](a, b)


def _truth(x) -> bool:
    tp = type(x)
    if tp in _BUILTIN_SCALARS or tp in _BUILTIN_CONTAINERS:
        return bool(x)
    for k in tp.__mro__:
        if k is object:
            break
        for nm in ("__bool__", "__len__"):
            f = k.__dict__.get(nm)
            if f is not None:
                if not isinstance(f, types.FunctionType) or not _getter_pure(f):
                    raise _Impure
                return bool(x)
    return bool(x)


_STR_PURE = {"startswith", "endswith", "upper", "lower", "strip", "lstrip", "rstrip", "isdigit", "isalpha",
             "isalnum", "isidentifier", "isupper", "islower", "isspace", "isnumeric", "isdecimal", "casefold",
             "find", "rfind", "count", "split", "rsplit", "replace", "partition", "rpartition", "title",
             "capitalize", "swapcase", "zfill", "removeprefix", "removesuffix", "join"}


def __brM__(obj, name, *args):  # noqa: N807
    """A method call that cannot change anything: dict.get with a plain key,
    a str method on plain arguments, or sqlglot's Expression.text."""
    tp = type(obj)
    if tp is dict and name == "get" and 1 <= len(args) <= 2:
        if type(args[0]) not in _BUILTIN_SCALARS and type(args[0]) is not tuple:
            raise _Impure
        return obj.get(*args)
    if tp is str and name in _STR_PURE:
        for a in args:
            if type(a) not in _BUILTIN_SCALARS and not (type(a) in (tuple, list) and all(type(x) is str for x in a)):
                raise _Impure
        return getattr(obj, name)(*args)
    if name == "text" and len(args) == 1 and type(args[0]) is str:
        found, f = _static(tp.__mro__, "text")
        if (found and isinstance(f, types.FunctionType) and f.__module__.startswith("sqlglot.")
                and f.__qualname__.endswith(("Expression.text", "Expr.text")) and _getter_pure(f)):
            return f(obj, args[0])
    raise _Impure


def __brU__(op, x):  # noqa: N807
    if op == "Not":
        return not _truth(x)
    if type(x) not in (int, float, complex, bool):
        raise _Impure
    return {"USub": operator.neg, "UAdd": operator.pos, "Invert": operator.invert}[op](x)


_CMPS = {"Eq": operator.eq, "NotEq": operator.ne, "Lt": operator.lt, "LtE": operator.le, "Gt": operator.gt,
         "GtE": operator.ge, "Is": operator.is_, "IsNot": operator.is_not}


def __brK__(ops, thunks):  # noqa: N807
    left = thunks[0]()
    res = True
    for op, th in zip(ops, thunks[1:]):
        right = th()
        if op in ("In", "NotIn"):
            if type(right) not in _BUILTIN_CONTAINERS:
                raise _Impure
            if type(right) not in (list, tuple, str, bytes) and type(left) not in _BUILTIN_SCALARS and type(left) is not tuple:
                raise _Impure
            res = (left in right) if op == "In" else (left not in right)
        else:
            res = _CMPS[op](left, right)
        if op not in ("Is", "IsNot", "In", "NotIn") and type(res) is not bool:
            raise _Impure  # a rich comparison that builds something
        if not res:
            return res
        left = right
    return res


def __brC__(f, *args):  # noqa: N807
    if f is isinstance:
        return isinstance(*args)
    if f is type and len(args) == 1:
        return type(args[0])
    if f is callable:
        return callable(*args)
    if f is len:
        if len(args) != 1 or type(args[0]) not in _BUILTIN_CONTAINERS:
            raise _Impure
        return len(args[0])
    if f is bool:
        return _truth(args[0]) if args else False
    if f is hasattr and len(args) == 2 and type(args[1]) is str:
        try:
            __brA__(args[0], args[1])
            return True
        except AttributeError:
            return False
    raise _Impure


# --------------------------------------------------------- the chain itself
MASK: dict[int, int] = {}
CNT: dict[int, int] = {}
ACC: dict[int, float] = {}
_pc = time.perf_counter
SLOW = 200e-6
_BAD = object()


class _Unknown(Exception):
    pass


def _ev(t, get, used):
    if type(t) is int:
        used.add(t)
        return get(t)
    op, kids = t
    last = len(kids) - 1
    v = None
    for j, c in enumerate(kids):
        v = _ev(c, get, used)
        if j == last:
            break
        if op == "and":
            if not _truth(v):
                break
        elif _truth(v):
            break
    return v


def __brb__(sid, th, gth):  # noqa: N807
    is_and, n, truth_only, variants = INFO[sid]
    m = MASK.get(sid, 0)
    full = FULL[sid]
    vals = []
    last = n - 1
    try:
        for k in range(n):
            v = th[k]()
            vals.append(v)
            if k == last:
                break
            if is_and:
                if not v:
                    break
            elif v:
                break
    except BaseException:
        if m != full:
            MASK[sid] = full
        raise
    res = vals[-1]
    if m == full:
        return res
    c = CNT.get(sid, 0)
    if c < 32:
        t0 = _pc()
    ne = len(vals)
    cache: dict = {}

    def get(k):
        if k < ne:
            return vals[k]
        r = cache.get(k, None)
        if r is None:
            g = gth[k]
            if g is None:
                r = _BAD
            else:
                try:
                    r = (g(),)
                except BaseException:  # noqa: BLE001
                    r = _BAD
            cache[k] = r
        if r is _BAD:
            raise _Unknown
        return r[0]

    def pure_again(k):
        r = cache.get(-1 - k)
        if r is None:
            g = gth[k]
            r = False
            if g is not None:
                try:
                    g()
                    r = True
                except BaseException:  # noqa: BLE001
                    r = False
            cache[-1 - k] = r
        return r

    try:
        orig_truth = _truth(res) if truth_only else None
    except BaseException:  # noqa: BLE001
        MASK[sid] = full
        return res
    for i in range(n - 1):
        bit = 1 << i
        if m & bit:
            continue
        same = False
        try:
            used: set = set()
            r2 = _ev(variants[i], get, used)
            ok = True
            for k in range(ne):
                if k not in used and not pure_again(k):
                    ok = False  # the swap would skip something the original ran
                    break
            if ok:
                same = (_truth(r2) == orig_truth) if truth_only else (r2 is res)
        except BaseException:  # noqa: BLE001
            same = False
        if not same:
            m |= bit
    if c < 32:
        ACC[sid] = ACC.get(sid, 0.0) + (_pc() - t0)
        CNT[sid] = c + 1
        if c == 31 and ACC[sid] / 32 > SLOW:
            m = full  # too costly to keep checking: treat every swap as visible
    MASK[sid] = m
    return res


for _n, _f in (("__brb__", __brb__), ("__brM__", __brM__), ("__brA__", __brA__), ("__brS__", __brS__), ("__brB__", __brB__),
               ("__brU__", __brU__), ("__brK__", __brK__), ("__brC__", __brC__)):
    setattr(builtins, _n, _f)

# ---------------------------------------------------------------- the run
import fingerprint  # noqa: E402

mods, errors = fingerprint.import_all("sqlglot")
if errors:
    print("import errors", errors, file=sys.stderr)
import_mask = dict(MASK)

tests = json.load(open(TESTS_JSON))["tests"]
by_module: dict[str, list[dict]] = {}
for t in tests:
    by_module.setdefault(t["file"], []).append(t)

module_masks: dict[str, dict] = {}
loader = unittest.TestLoader()
loaded: dict[int, unittest.TestSuite] = {}
for f in sorted(by_module):
    modname = f[:-3].replace("/", ".")
    MASK.clear()
    try:
        __import__(modname)
    except Exception as e:  # noqa: BLE001
        print("cannot import", modname, e, file=sys.stderr)
    module_masks[f] = {str(k): v for k, v in MASK.items() if v}
    for t in by_module[f]:
        parts = t["id"].split("::")
        name = modname + "." + ".".join(parts[1:])
        try:
            loaded[t["number"]] = unittest.TestSuite([loader.loadTestsFromName(name)])
        except Exception as e:  # noqa: BLE001
            print("cannot load", name, e, file=sys.stderr)

import inline  # noqa: E402

inline.install()


def run_child(num: int) -> None:
    MASK.clear()
    CNT.clear()
    ACC.clear()
    spawned = []
    import subprocess as _sp
    import multiprocessing.process as _mpp

    _orig_popen_init = _sp.Popen.__init__
    _orig_start = _mpp.BaseProcess.start

    def _popen_init(self, *a, **k):
        spawned.append("popen")
        return _orig_popen_init(self, *a, **k)

    def _start(self, *a, **k):
        spawned.append("process")
        return _orig_start(self, *a, **k)

    _sp.Popen.__init__ = _popen_init
    _mpp.BaseProcess.start = _start
    res = unittest.TestResult()
    c0 = time.process_time()
    loaded[num].run(res)
    cpu = time.process_time() - c0
    rec = {
        "n": num,
        "failed": bool(res.failures or res.errors or res.unexpectedSuccesses),
        "msgs": [tb[-400:] for _, tb in (res.failures + res.errors)][:2],
        "cpu": round(cpu, 3),
        "mask": {str(k): v for k, v in MASK.items() if v},
        "seen": sorted(MASK),
        "spawned": spawned,
    }
    tmp = os.path.join(OUT, f".{num}.json")
    with open(tmp, "w") as fh:
        json.dump(rec, fh)
    os.replace(tmp, os.path.join(OUT, f"{num}.json"))


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "_sites.json"), "w") as fh:
        json.dump({"sites": SITES, "import_mask": {str(k): v for k, v in import_mask.items() if v},
                   "module_masks": module_masks}, fh)
    todo = [n for n in sorted(loaded) if not os.path.exists(os.path.join(OUT, f"{n}.json"))]
    if ONLY:
        todo = [n for n in todo if str(n) in ONLY]
    secs = {t["number"]: t["seconds"] for t in tests}
    todo.sort(key=lambda n: -secs.get(n, 0))
    running: dict[int, int] = {}
    i = 0
    t0 = time.time()
    while i < len(todo) or running:
        while i < len(todo) and len(running) < JOBS:
            n = todo[i]
            i += 1
            pid = os.fork()
            if pid == 0:
                try:
                    lim = int(60 + 30 * secs.get(n, 1))
                    resource.setrlimit(resource.RLIMIT_CPU, (lim, lim))
                    run_child(n)
                    os._exit(0)
                except BaseException as e:  # noqa: BLE001
                    try:
                        print("child error", n, repr(e), file=sys.stderr)
                    finally:
                        os._exit(3)
            running[pid] = n
        pid, status = os.wait()
        n = running.pop(pid, None)
        if n is None:
            continue
        if status != 0:
            print("test", n, "child exit", status, file=sys.stderr)
        done = i - len(running)
        if done % 100 == 0:
            print(f"{done}/{len(todo)} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    print("done", round(time.time() - t0, 1), file=sys.stderr)


if __name__ == "__main__":
    main()
