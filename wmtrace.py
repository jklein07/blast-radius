"""Comparison-swap profile (build time only).

A frequent kind of edit swaps the operator of a comparison: `is` for `==`,
`<` for `<=`, and so on. Such an edit can change a test's outcome only if,
somewhere while that test runs, the new operator gives a different result
than the old one on the same operands. Until that first difference the
patched program runs exactly like the original.

This tracer imports sqlglot with every single-operator comparison rewritten
into a call that computes the original result and, alongside, the results
of the alternative operators. Each test runs in its own forked child; the
child reports, per comparison site, which alternative operators ever gave a
different result (or raised differently). The selector then charges a swap
only to the tests in which it would actually change a value.

usage: python wmtrace.py SQLGLOT_ROOT tests.json OUT_DIR [JOBS]
"""

from __future__ import annotations

import ast
import builtins
import importlib.machinery
import json
import operator
import os
import resource
import sys
import time
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

# operator name -> (function, bit)
OPS = {
    "Is": (operator.is_, 1),
    "IsNot": (operator.is_not, 2),
    "Eq": (operator.eq, 4),
    "NotEq": (operator.ne, 8),
    "Lt": (operator.lt, 16),
    "LtE": (operator.le, 32),
    "Gt": (operator.gt, 64),
    "GtE": (operator.ge, 128),
}
FAMILY_EQ = ("Is", "IsNot", "Eq", "NotEq")
FAMILY_ORD = ("Lt", "LtE", "Gt", "GtE", "Eq", "NotEq")

SITES: list[tuple[str, str, str]] = []  # sid -> (file, "line:col:eline:ecol", op)
OPF: list = []
ALTS: list = []
FULL: list = []


def _register(rel: str, node: ast.Compare) -> int:
    op = type(node.ops[0]).__name__
    fam = FAMILY_ORD if op in ("Lt", "LtE", "Gt", "GtE") else FAMILY_EQ
    alts = [(OPS[o][1], OPS[o][0]) for o in fam if o != op]
    sid = len(SITES)
    SITES.append((rel, f"{node.lineno}:{node.col_offset}:{node.end_lineno}:{node.end_col_offset}", op))
    OPF.append(OPS[op][0])
    ALTS.append(alts)
    full = 0
    for b, _ in alts:
        full |= b
    FULL.append(full)
    return sid


class _T(ast.NodeTransformer):
    def __init__(self, rel):
        self.rel = rel

    def generic_visit(self, node):
        # leave annotations alone: they are strings at runtime here, and a
        # rewritten one would read differently to typing.get_type_hints
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

    def visit_Compare(self, node):
        if len(node.ops) != 1 or type(node.ops[0]).__name__ not in OPS:
            return self.generic_visit(node)
        sid = _register(self.rel, node)
        self.generic_visit(node)
        call = ast.Call(
            func=ast.Name(id="__brc__", ctx=ast.Load()),
            args=[ast.Constant(sid), node.left, node.comparators[0]],
            keywords=[],
        )
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

# per-process state (reset in each child)
MASK: dict[int, int] = {}
CNT: dict[int, int] = {}
ACC: dict[int, float] = {}
_pc = time.perf_counter
SLOW = 40e-6  # average seconds per evaluation of the alternatives that counts as too slow to keep checking


def __brc__(sid, l, r):  # noqa: N807
    m = MASK.get(sid, 0)
    full = FULL[sid]
    try:
        res = OPF[sid](l, r)
    except Exception as e:
        if m != full:
            for bit, alt in ALTS[sid]:
                if m & bit:
                    continue
                try:
                    alt(l, r)
                    m |= bit
                except Exception as e2:  # noqa: BLE001
                    if type(e2) is not type(e):
                        m |= bit
            MASK[sid] = m
        raise
    if m == full:
        return res
    c = CNT.get(sid, 0)
    if c < 32:
        t0 = _pc()
    for bit, alt in ALTS[sid]:
        if m & bit:
            continue
        try:
            v = alt(l, r)
        except Exception:  # noqa: BLE001
            m |= bit
            continue
        if v is not res:
            m |= bit
    if c < 32:
        ACC[sid] = ACC.get(sid, 0.0) + (_pc() - t0)
        CNT[sid] = c + 1
        if c == 31 and ACC[sid] / 32 > SLOW:
            m = full  # too costly to keep checking: treat every swap as visible
    MASK[sid] = m
    return res


builtins.__brc__ = __brc__

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
        "spawned": spawned,
    }
    tmp = os.path.join(OUT, f".{num}.json")
    with open(tmp, "w") as fh:
        json.dump(rec, fh)
    os.replace(tmp, os.path.join(OUT, f"{num}.json"))


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "_sites.json"), "w") as fh:
        json.dump({"sites": SITES, "import_mask": {str(k): v for k, v in import_mask.items() if v}, "module_masks": module_masks}, fh)
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
