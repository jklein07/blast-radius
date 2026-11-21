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
    """Lines the full test runs with its front-end calls switched off."""
    lines: dict = {}
    depth = [0]

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
    try:
        suite.run(res)
    finally:
        mon.set_events(TOOL, 0)
    return {"failed": bool(res.failures or res.errors), "lines": {f: sorted(v) for f, v in lines.items()}}


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
        out[tid] = {"attr": attr, "nq": n, "arts": fresh, "unstable": unstable, "cpu": s["cpu"], "exec": e["lines"]}
        print(tid.split("::")[-1], "queries", n, "unstable", unstable, "exec lines", sum(len(v) for v in e["lines"].values()),
              "cpu", round(sum(s["cpu"].values()), 2), file=sys.stderr, flush=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh)


if __name__ == "__main__":
    main()
