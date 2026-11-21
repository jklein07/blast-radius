"""Add lean plan fingerprints to maps/fe.json.gz (map build time only).

For each TPC test that runs queries in sqlglot's Python executor, run every
executed query the way the test's worker does (tests.test_executor.mp_execute
over the class set-up's tables) and record which of Expression's _type,
_meta and comments slots the execution side (PythonExecutor.execute) reads.
A read inside Expression.__deepcopy__ only copies the slot into the copy,
whose own reads are recorded in turn, so it does not count. Queries are run
under several hash seeds and the reads are united.

For a query whose execution never reads some of those slots, the map gets
lean[str(i)] = [those slots, fingerprint of the query's plan without them],
computed in a fresh process and again in one process in order (a query whose
two values differ gets no entry). tpcfp._same uses it: when everything else
the front end hands over matches the pinned commit's, and the plan matches
without slots the execution side never reads, the execution side (untouched,
or the test would not be judged by fingerprints) runs exactly as it did.

usage: python ftrace_lean.py SQLGLOT_ROOT maps/fe.json.gz
"""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys

ROOT = os.path.realpath(sys.argv[1])
MAP = sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = ("0", "1", "2")


def reads(tid: str) -> dict:
    """str(i) -> slots read while executing query i (this process's seed)."""
    import pickle

    from sqlglot import parse_one
    from sqlglot.executor.python import PythonExecutor
    from sqlglot.expressions.core import Expression

    import tests.helpers as helpers
    import tests.test_executor as te
    import tpcfp

    on = [False]
    seen: set = set()

    class Tr:
        def __init__(self, name, orig):
            self.name, self.orig = name, orig

        def __get__(self, obj, typ=None):
            if obj is None:
                return self
            if on[0]:
                f = sys._getframe(1).f_code
                if not (f.co_name == "__deepcopy__" and f.co_filename.endswith("sqlglot/expressions/core.py")):
                    seen.add(self.name)
            return self.orig.__get__(obj, typ)

        def __set__(self, obj, v):
            self.orig.__set__(obj, v)

        def __delete__(self, obj):
            self.orig.__delete__(obj)

    for f in tpcfp.LEAN_FIELDS:
        setattr(Expression, f, Tr(f, Expression.__dict__[f]))
    run = PythonExecutor.execute

    def traced(self, plan):
        on[0] = True
        try:
            return run(self, plan)
        finally:
            on[0] = False

    PythonExecutor.execute = traced
    te.TestExecutor.setUpClass()
    attr, _ = tpcfp.TESTS[tid]
    tables = te.TestExecutor.tpcds_tables if attr == "tpcds_sqls" else te.TestExecutor.tpch_tables
    schema = getattr(helpers, tpcfp.SCHEMAS[attr])
    te.initializer(schema, tables)
    out = {}
    for i, (meta, sql, _) in enumerate(tpcfp.queries(attr)):
        if not meta.get("execute"):
            continue
        seen.clear()
        e = pickle.loads(pickle.dumps(parse_one(sql), protocol=pickle.HIGHEST_PROTOCOL))
        te.mp_execute(e, meta)
        out[str(i)] = sorted(seen)
    return out


def lean_fps(tid: str, skips: dict, fresh_i=None) -> dict:
    import tpcfp

    attr, _ = tpcfp.TESTS[tid]
    sqls = tpcfp.queries(attr)
    out = {}
    for i, skip in skips.items():
        if fresh_i is not None and i != fresh_i:
            continue
        arts, plan = tpcfp._artifacts(tid, int(i), sqls)
        out[i] = [arts, tpcfp.fp(plan, tuple(skip)) if plan is not None else None]
    return out


def child(mode: str, tid: str, seed: str, arg: str = "") -> dict:
    env = dict(os.environ, PYTHONHASHSEED=seed, SKIP_INTEGRATION="0", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", ROOT, mode, tid, arg],
                       env=env, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise SystemExit(f"{mode} {tid}: {r.stderr[-800:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def main() -> None:
    import tpcfp

    with gzip.open(MAP, "rt") as fh:
        fe = json.load(fh)
    for tid, (attr, kind) in tpcfp.TESTS.items():
        if kind != "mp" or tid not in fe:
            continue
        read: dict = {}
        for seed in SEEDS:
            for i, r in child("reads", tid, seed).items():
                read.setdefault(i, set()).update(r)
        skips = {i: sorted(set(tpcfp.LEAN_FIELDS) - r) for i, r in read.items()}
        skips = {i: s for i, s in skips.items() if s}
        seq = child("lean", tid, "0", json.dumps(skips))
        lean = {}
        for i, s in skips.items():
            fr = child("lean1", tid, "0", json.dumps({i: s}))
            if fr.get(i) != seq.get(i) or seq[i][0] != fe[tid]["arts"].get(i) or seq[i][1] is None:
                continue  # unstable, or not what the map's artifacts were made from
            lean[i] = [s, seq[i][1]]
        fe[tid]["lean"] = lean
        print(tid.split("::")[-1], "executed", len(read), "lean", len(lean),
              {f: sum(1 for v in lean.values() if f in v[0]) for f in tpcfp.LEAN_FIELDS}, file=sys.stderr, flush=True)
    with gzip.open(MAP, "wt") as fh:
        json.dump(fe, fh, separators=(",", ":"))


if __name__ == "__main__":
    if sys.argv[1] == "--child":
        _, _, root, mode, tid, arg = sys.argv
        sys.path.insert(0, root)
        sys.path.insert(1, HERE)
        os.chdir(root)
        if mode == "reads":
            res = reads(tid)
        elif mode == "lean":
            res = lean_fps(tid, json.loads(arg))
        else:
            sk = json.loads(arg)
            res = lean_fps(tid, sk, next(iter(sk)))
        print(json.dumps(res))
    else:
        sys.path.insert(0, ROOT)
        sys.path.insert(1, HERE)
        main()
