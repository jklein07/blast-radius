"""Record what tpcfp.py needs for the TPC executor tests (build time only).

Per test: the front-end artifact fingerprints of every query (computed in a
fresh process per query and again in one process in order; queries whose
fingerprints differ between the two count as unstable), the CPU each costs,
and the lines the test executes outside its front end: setUpClass, the
Python executor, result pickling, DuckDB and comparison glue. A change that
touches any of those lines cannot be judged from fingerprints.

usage: python ftrace.py SQLGLOT_ROOT OUT.json
"""

from __future__ import annotations

import json
import os
import sys
import time
import unittest

ROOT = os.path.realpath(sys.argv[1])
OUT = sys.argv[2]
PREFIX = ROOT + "/"
sys.path.insert(0, ROOT)
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
os.environ.setdefault("SKIP_INTEGRATION", "0")

import fingerprint  # noqa: E402

fingerprint.import_all("sqlglot")
import tests.test_executor as te  # noqa: E402
import inline  # noqa: E402
import tpcfp  # noqa: E402

inline.install()
mon = sys.monitoring
TOOL = 3
EV = mon.events.LINE | mon.events.PY_START


def in_child(fn):
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            res = fn()
        except BaseException as e:  # noqa: BLE001
            res = {"error": repr(e)[:300]}
        data = json.dumps(res).encode()
        while data:
            k = os.write(w, data)
            data = data[k:]
        os._exit(0)
    os.close(w)
    buf = b""
    while True:
        ch = os.read(r, 1 << 16)
        if not ch:
            break
        buf += ch
    os.close(r)
    os.waitpid(pid, 0)
    return json.loads(buf)


def exec_lines(tid):
    """Lines the full test runs with its front-end calls switched off, and the
    names of the sqlglot class attributes read meanwhile."""
    import enum

    lines: dict = {}
    depth = [0]
    attrs: set = set()
    lk: set = set()
    live = [False]
    keys: dict = {}

    class _TA:  # records reads of a wrapped class attribute while the execution side runs
        __slots__ = ("v", "k")

        def __init__(self, v, k):
            self.v = v
            self.k = k

        def __get__(self, obj, typ=None):
            if live[0] and depth[0] == 0:
                attrs.add(self.k)
                t = typ if typ is not None else type(obj)
                lk.add((keys.get(id(t)) or f"{getattr(t, '__module__', '?')}:{getattr(t, '__qualname__', '?')}") + "|" + self.k)
            return self.v

    for _key, c in fingerprint.iter_classes(sorted(sys.modules), "sqlglot"):
        keys[id(c)] = _key
        if isinstance(c, enum.EnumMeta):
            continue
        for name, v in list(c.__dict__.items()):
            if name.startswith("__") and name.endswith("__"):
                continue
            if hasattr(type(v), "__get__") and not isinstance(v, type):
                continue
            try:
                setattr(c, name, _TA(v, name))
            except Exception:  # noqa: BLE001
                pass

    def on_line(code, line):
        f = code.co_filename
        if f.startswith(PREFIX):
            lines.setdefault(f[len(PREFIX):], set()).add(line)
        return mon.DISABLE

    def on_start(code, offset):
        f = code.co_filename
        if f.startswith(PREFIX):
            lines.setdefault(f[len(PREFIX):], set()).add(-code.co_firstlineno)
        return mon.DISABLE

    def off(fn):
        def w(*a, **k):
            depth[0] += 1
            if depth[0] == 1:
                mon.set_events(TOOL, 0)
            try:
                return fn(*a, **k)
            finally:
                depth[0] -= 1
                if depth[0] == 0:
                    mon.set_events(TOOL, EV)

        return w

    import sqlglot.executor as sx

    te.parse_one = off(te.parse_one)
    te.transpile = off(te.transpile)
    sx.optimize = off(sx.optimize)
    sx.Plan = off(sx.Plan)
    name = tid.split("::")
    suite = unittest.TestLoader().loadTestsFromName("tests.test_executor." + ".".join(name[1:]))
    mon.use_tool_id(TOOL, "ftrace")
    mon.register_callback(TOOL, mon.events.LINE, on_line)
    mon.register_callback(TOOL, mon.events.PY_START, on_start)
    mon.set_events(TOOL, EV)
    res = unittest.TestResult()
    live[0] = True
    try:
        suite.run(res)
    finally:
        live[0] = False
        mon.set_events(TOOL, 0)
    # every identifier (or identifier-like string) on the lines the execution side ran
    import io
    import tokenize

    names: set = set()
    for f, ls in lines.items():
        try:
            src = open(os.path.join(ROOT, f), encoding="utf-8").read().split("\n")
        except OSError:
            continue
        for ln in ls:
            if ln <= 0 or ln > len(src):
                continue
            try:
                for tok in tokenize.generate_tokens(io.StringIO(src[ln - 1] + "\n").readline):
                    if tok.type == tokenize.NAME:
                        names.add(tok.string)
                    elif tok.type == tokenize.STRING:
                        names.update(w for w in __import__("re").findall(r"[A-Za-z_][A-Za-z0-9_]*", tok.string))
            except (tokenize.TokenError, IndentationError, SyntaxError):
                names.update(__import__("re").findall(r"[A-Za-z_][A-Za-z0-9_]*", src[ln - 1]))
    return {"failed": bool(res.failures or res.errors), "lines": {f: sorted(v) for f, v in lines.items()},
            "attrs": sorted(attrs), "names": sorted(names), "lk": sorted(lk)}


def main():
    out = {}
    for tid, (attr, kind) in tpcfp.TESTS.items():
        sqls = tpcfp.queries(attr)
        n = len(sqls)
        fresh = {}
        for i in range(n):
            fresh[str(i)] = in_child(lambda i=i: tpcfp.artifacts(tid, i))

        def seq():
            arts, cpu = {}, {}
            for i in range(n):
                c0 = time.process_time()
                arts[str(i)] = tpcfp.artifacts(tid, i, sqls)
                cpu[str(i)] = round(time.process_time() - c0, 4)
            # again, warm: what a run over several queries costs per query
            for i in range(n):
                c0 = time.process_time()
                tpcfp.artifacts(tid, i, sqls)
                cpu[str(i)] = round(min(cpu[str(i)], time.process_time() - c0), 4)
            return {"arts": arts, "cpu": cpu}

        s = in_child(seq)
        unstable = sorted(int(i) for i in fresh if fresh[i] != s["arts"].get(i))
        e = in_child(lambda: exec_lines(tid))
        if e.get("failed") or "error" in e:
            raise SystemExit(f"{tid}: test run failed while tracing: {e.get('error')}")
        out[tid] = {"attr": attr, "nq": n, "arts": fresh, "unstable": unstable, "cpu": s["cpu"], "exec": e["lines"],
                    "exec_attrs": e["attrs"], "exec_names": e["names"], "exec_lk": e["lk"]}
        print(tid.split("::")[-1], "queries", n, "unstable", unstable, "exec lines", sum(len(v) for v in e["lines"].values()),
              "cpu", round(sum(s["cpu"].values()), 2), "exec attrs", len(e["attrs"]), "names", len(e["names"]), file=sys.stderr, flush=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh)


if __name__ == "__main__":
    main()
