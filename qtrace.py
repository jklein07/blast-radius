"""Per-query coverage for the TPC tests (build time only).

test_execute_tpcds, test_execute_tpch and test_optimized_tpch loop over the
queries of a fixture file that setUpClass loads into a class attribute. Most
changes reach only a few of those queries, so the selector can run the test on
just those queries instead of all of them. This script records, for each
query, which lines it executes when the test runs on that query alone, plus
the lines setUpClass executes (a change there affects every query).

Each query runs in its own forked child: setUpClass first (not traced), then
the test method with the class attribute cut down to that one query.

usage: python qtrace.py SQLGLOT_ROOT OUT.json [JOBS]
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time
import unittest

ROOT = os.path.realpath(sys.argv[1])
OUT = sys.argv[2]
JOBS = int(sys.argv[3]) if len(sys.argv) > 3 else os.cpu_count() or 2
PREFIX = ROOT + "/"

sys.path.insert(0, ROOT)
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
os.environ.setdefault("SKIP_INTEGRATION", "0")

import fingerprint  # noqa: E402

fingerprint.import_all("sqlglot")
import tests.test_executor as te  # noqa: E402
import inline  # noqa: E402

inline.install()

# test id -> class attribute holding its queries
SPLIT = {
    "tests/test_executor.py::TestExecutor::test_execute_tpcds": "tpcds_sqls",
    "tests/test_executor.py::TestExecutor::test_execute_tpch": "tpch_sqls",
    "tests/test_executor.py::TestExecutor::test_optimized_tpch": "tpch_sqls",
}
FIXTURE = {"tpcds_sqls": "optimizer/tpc-ds/tpc-ds.sql", "tpch_sqls": "optimizer/tpc-h/tpc-h.sql"}
mon = sys.monitoring
TOOL = 3


def traced(fn):
    """Run fn() with line coverage on; return (lines by file, cpu, error)."""
    lines: dict[str, set] = {}

    def on_line(code, line):
        f = code.co_filename
        if f.startswith(PREFIX):
            s = lines.get(f)
            if s is None:
                s = lines[f] = set()
            s.add(line)
        return mon.DISABLE

    def on_start(code, offset):
        f = code.co_filename
        if f.startswith(PREFIX):
            s = lines.get(f)
            if s is None:
                s = lines[f] = set()
            s.add(-code.co_firstlineno)
        return mon.DISABLE

    mon.use_tool_id(TOOL, "qtrace")
    mon.register_callback(TOOL, mon.events.LINE, on_line)
    mon.register_callback(TOOL, mon.events.PY_START, on_start)
    mon.set_events(TOOL, mon.events.LINE | mon.events.PY_START)
    mon.restart_events()
    c0 = time.process_time()
    err = None
    try:
        fn()
    except BaseException as e:  # noqa: BLE001
        err = repr(e)[:300]
    finally:
        mon.set_events(TOOL, 0)
    cpu = time.process_time() - c0
    return {f[len(PREFIX):]: sorted(s) for f, s in lines.items()}, cpu, err


def child(job, path):
    kind = job[0]
    rec: dict = {"job": list(job)}
    cls = te.TestExecutor
    if kind == "setup":
        lines, cpu, err = traced(cls.setUpClass)
        rec.update(lines=lines, cpu=cpu, err=err)
    else:
        _, tid, i = job
        attr = SPLIT[tid]
        cls.setUpClass()
        orig = getattr(cls, attr)
        setattr(cls, attr, [orig[i]])
        method = tid.split("::")[-1]
        res = unittest.TestResult()

        def go():
            cls(method).run(res)

        lines, cpu, err = traced(go)
        failed = bool(res.failures or res.errors)
        rec.update(lines=lines, cpu=cpu, err=err, failed=failed,
                   msgs=[tb[-400:] for _, tb in res.failures + res.errors][:2])
        # untraced and warm: the test with no queries (fixed cost), then with
        # this query alone; the difference is what the query adds to a run
        for key, sub in (("cpu_fixed", []), ("cpu_plain", [orig[i]])):
            setattr(cls, attr, sub)
            cls.cache = {}
            res2 = unittest.TestResult()
            c0 = time.process_time()
            cls(method).run(res2)
            rec[key] = time.process_time() - c0
    with open(path + ".tmp", "w") as f:
        json.dump(rec, f)
    os.replace(path + ".tmp", path)


def main():
    from tests.helpers import load_sql_fixture_pairs

    counts = {a: len(list(load_sql_fixture_pairs(fx))) for a, fx in FIXTURE.items()}
    jobs = [("setup",)]
    for tid, attr in SPLIT.items():
        jobs += [("q", tid, i) for i in range(min(counts[attr], int(os.environ.get("QT_MAX", "100000"))))]
    tmpdir = OUT + ".d"
    os.makedirs(tmpdir, exist_ok=True)
    running = {}
    k = 0
    t0 = time.time()
    paths = {}
    for j in jobs:
        paths[j] = os.path.join(tmpdir, "_".join(str(x).replace("/", "-").replace(":", "-") for x in j) + ".json")
    todo = [j for j in jobs if not os.path.exists(paths[j])]
    while k < len(todo) or running:
        while k < len(todo) and len(running) < JOBS:
            j = todo[k]
            k += 1
            pid = os.fork()
            if pid == 0:
                try:
                    resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
                    child(j, paths[j])
                    os._exit(0)
                except BaseException as e:  # noqa: BLE001
                    print("child error", j, repr(e), file=sys.stderr)
                    os._exit(3)
            running[pid] = j
        pid, st = os.wait()
        j = running.pop(pid, None)
        if j is not None and st != 0:
            print("job", j, "exit", st, file=sys.stderr)
        if k % 20 == 0:
            print(f"{k}/{len(todo)} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    out: dict = {"setup": None, "tests": {}}
    for j in jobs:
        if not os.path.exists(paths[j]):
            raise SystemExit(f"missing {j}")
        rec = json.load(open(paths[j]))
        if j[0] == "setup":
            out["setup"] = rec["lines"]
            out["setup_cpu"] = rec["cpu"]
        else:
            if rec.get("failed") or rec.get("err"):
                print("query failed on pristine", j, rec.get("msgs"), rec.get("err"), file=sys.stderr)
            t = out["tests"].setdefault(j[1], {"attr": SPLIT[j[1]], "q": []})
            t["q"].append({"lines": rec["lines"], "cold": round(rec["cpu"], 4), "warm": round(rec["cpu_plain"], 4),
                           "fixed": round(rec["cpu_fixed"], 4), "failed": bool(rec.get("failed") or rec.get("err"))})
    with open(OUT, "w") as f:
        json.dump(out, f)
    print("done", time.time() - t0, file=sys.stderr)


if __name__ == "__main__":
    main()
