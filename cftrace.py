"""Per-pair coverage for the fixture-driven optimizer tests (build time only).

For each test in tests/test_optimizer.py that loops over fixture pairs
(cfsel.py), record which lines each pair executes, the lines the test runs
outside its pairs (always run, whichever pairs are kept), the lines
setUpClass runs, and CPU costs. Pairs are traced twice:

  solo  the test with that pair alone (what a verification run of a few
        pairs executes, cold caches included);
  ctx   the whole test, each line attributed to the pair being processed
        when it first ran in that pair's turn (state left by earlier pairs
        included).

A pair's turn spans its loop iteration and, in check_file, the comparison
after `future.result()` for its future.

usage: python cftrace.py SQLGLOT_ROOT OUTDIR [JOBS]
"""

from __future__ import annotations

import concurrent.futures as cf
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
import tests.test_optimizer as to  # noqa: E402
import inline  # noqa: E402
import cfsel  # noqa: E402

inline.install()
cfsel.install(to)
CLS = to.TestOptimizer
mon = sys.monitoring
TOOL = 3

cur = [None]  # the pair whose turn it is, (load index, pair index), or None
TIMING: dict = {}  # untraced ctx runs: pair -> CPU seconds of its turns
_last = [0.0]
OUTSIDE: dict = {}
INSIDE: dict = {}  # (k, i) -> {file: set(lines)}


def switch(to_):
    if cur[0] != to_:
        if TIMING.get("on"):
            t = time.perf_counter()  # the CPU clock may tick in 4 ms steps
            TIMING[cur[0]] = TIMING.get(cur[0], 0.0) + t - _last[0]
            _last[0] = t
            cur[0] = to_
            return
        cur[0] = to_
        mon.restart_events()


def _rec(f, ln):
    if cur[0] is None:
        d = OUTSIDE
    else:
        d = INSIDE.get(cur[0])
        if d is None:
            d = INSIDE[cur[0]] = {}
    s = d.get(f)
    if s is None:
        s = d[f] = set()
    s.add(ln)


def on_line(code, line):
    f = code.co_filename
    if f.startswith(PREFIX):
        _rec(f[len(PREFIX):], line)
    return mon.DISABLE


def on_start(code, offset):
    f = code.co_filename
    if f.startswith(PREFIX):
        _rec(f[len(PREFIX):], -code.co_firstlineno)
    return mon.DISABLE


# a check_file pair's comparison runs after its future's result() call
_submit = inline.InlineExecutor.submit


def _tagged_submit(self, fn, /, *a, **k):
    fut = _submit(self, fn, *a, **k)
    fut._cf_tag = cur[0]
    return fut


_result = cf.Future.result


def _switching_result(self, timeout=None):
    tag = getattr(self, "_cf_tag", None)
    if tag is not None:
        switch(tag)
    return _result(self, timeout)


_check_file = CLS.check_file


def _check_file_wrapper(self, *a, **k):
    try:
        return _check_file(self, *a, **k)
    finally:
        switch(None)


def hooks_on():
    inline.InlineExecutor.submit = _tagged_submit
    cf.Future.result = _switching_result
    CLS.check_file = _check_file_wrapper
    cfsel.HOOKS[:] = [lambda k, i: switch((k, i)), lambda: switch(None)]


def hooks_off():
    inline.InlineExecutor.submit = _submit
    cf.Future.result = _result
    CLS.check_file = _check_file
    cfsel.HOOKS[:] = []
    cur[0] = None


def run(method):
    res = unittest.TestResult()
    CLS(method).run(res)
    return bool(res.failures or res.errors), [tb[-600:] for _, tb in res.failures + res.errors][:2]


def traced(fn):
    OUTSIDE.clear()
    INSIDE.clear()
    cur[0] = None
    mon.use_tool_id(TOOL, "cftrace")
    mon.register_callback(TOOL, mon.events.LINE, on_line)
    mon.register_callback(TOOL, mon.events.PY_START, on_start)
    mon.set_events(TOOL, mon.events.LINE | mon.events.PY_START)
    mon.restart_events()
    c0 = time.process_time()
    try:
        out = fn()
    finally:
        mon.set_events(TOOL, 0)
        mon.free_tool_id(TOOL)
    return out, time.process_time() - c0


def _lines(d):
    return {f: sorted(s) for f, s in d.items()}


def keep_for(loads, k=None, i=None):
    return {kk: (fn, n, {i} if kk == k else set()) for kk, (fn, n) in enumerate(loads)}


def child(job, path, loads):
    kind = job[0]
    rec: dict = {"job": list(job)}
    if kind == "setup":
        _, cpu = traced(CLS.setUpClass)
        rec.update(lines=_lines(OUTSIDE), cpu=cpu)
    elif kind == "disc":
        CLS.setUpClass()
        cfsel.RECORD[0] = True
        names = unittest.TestLoader().getTestCaseNames(CLS)
        fails = {}
        for m in names:
            failed, msgs = run(m)
            if failed:
                fails[m] = msgs
        cfsel.RECORD[0] = False
        rec.update(loads=cfsel.LOADS, fails=fails)
    else:
        m = job[1]
        CLS.setUpClass()
        if kind == "fixed":
            cfsel.KEEP[m] = keep_for(loads[m])
        elif kind == "ctx":
            cfsel.KEEP[m] = {}  # listed, no load restricted: every pair, turns attributed
        else:
            cfsel.KEEP[m] = keep_for(loads[m], job[2], job[3])
        hooks_on()
        (failed, msgs), cpu = traced(lambda: run(m))
        hooks_off()
        rec.update(failed=failed, msgs=msgs, cold=cpu, outside=_lines(OUTSIDE),
                   inside=[[k, i, _lines(d)] for (k, i), d in INSIDE.items()])
        # untraced and warm: what the run costs once caches are filled
        warm = []
        pc: dict = {}
        reps = 3 if kind != "q" else 1
        for _ in range(reps):
            if kind == "ctx":  # and what each pair's turns cost in it
                TIMING.clear()
                hooks_on()
                TIMING["on"] = True
            c0 = time.process_time()
            w0 = time.perf_counter()
            _last[0] = w0
            run(m)
            cpu = time.process_time() - c0
            warm.append(cpu)
            if kind == "ctx":
                wall = time.perf_counter() - w0
                TIMING.pop("on")
                hooks_off()
                f = cpu / wall if wall > 0 else 1.0  # wall shares of the run's CPU
                for key, v in TIMING.items():
                    if key is not None:
                        pc[key] = pc.get(key, 0.0) + v * f / reps
        rec["warm"] = min(warm)
        if kind == "ctx":
            rec["pc"] = [[k, i, round(v, 6)] for (k, i), v in pc.items()]
    with open(path + ".tmp", "w") as f:
        json.dump(rec, f)
    os.replace(path + ".tmp", path)


def run_jobs(jobs, tmpdir, loads):
    paths = {j: os.path.join(tmpdir, "_".join(str(x) for x in j) + ".json") for j in jobs}
    todo = [j for j in jobs if not os.path.exists(paths[j])]
    running = {}
    k = 0
    t0 = time.time()
    while k < len(todo) or running:
        while k < len(todo) and len(running) < JOBS:
            j = todo[k]
            k += 1
            pid = os.fork()
            if pid == 0:
                try:
                    resource.setrlimit(resource.RLIMIT_CPU, (600, 600))
                    child(j, paths[j], loads)
                    os._exit(0)
                except BaseException as e:  # noqa: BLE001
                    print("child error", j, repr(e)[:300], file=sys.stderr, flush=True)
                    os._exit(3)
            running[pid] = j
        pid, st = os.wait()
        j = running.pop(pid, None)
        if j is not None and st != 0:
            print("job", j, "exit", st, file=sys.stderr, flush=True)
        if k % 200 == 0:
            print(f"{k}/{len(todo)} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    return paths


def main():
    tmpdir = OUT
    os.makedirs(tmpdir, exist_ok=True)
    p = run_jobs([("setup",), ("disc",)], tmpdir, {})
    disc = json.load(open(p[("disc",)]))
    loads = {m: [tuple(x) for x in v] for m, v in disc["loads"].items()}
    if disc["fails"]:
        print("tests failing on the pinned commit (left out):", sorted(disc["fails"]), file=sys.stderr)
    loads = {m: v for m, v in loads.items() if m not in disc["fails"] and sum(n for _, n in v) > 0}
    only = os.environ.get("CF_ONLY")
    if only:
        loads = {m: v for m, v in loads.items() if m in only.split(",")}
    jobs = []
    for m, v in sorted(loads.items()):
        jobs += [("fixed", m), ("ctx", m)]
        jobs += [("q", m, k, i) for k, (_, n) in enumerate(v) for i in range(n)]
    print(len(loads), "tests", len(jobs), "jobs", file=sys.stderr, flush=True)
    run_jobs(jobs, tmpdir, loads)
    with open(os.path.join(tmpdir, "index.json"), "w") as f:
        json.dump({"loads": loads, "jobs": jobs}, f)
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
