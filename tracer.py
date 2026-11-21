"""Per-test dependency tracer (build time only).

Runs inside the image at build time, against the pristine sqlglot checkout.
The parent imports everything once, wraps every sqlglot class attribute that
holds data in a recording descriptor, then forks one child per test. Each
child records which lines of sqlglot/tests it executed (sys.monitoring),
which (class, attribute) pairs it read and which repository files it opened,
runs the test with unittest, and writes one JSON record.

usage: python tracer.py SQLGLOT_ROOT tests.json OUT_DIR [JOBS]
"""

from __future__ import annotations

import builtins
import enum
import io
import json
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
CALIB = os.environ.get("TRACE_CALIB") == "1"  # measure plain CPU only

sys.path.insert(0, ROOT)
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
os.environ.setdefault("SKIP_INTEGRATION", "0")

import fingerprint  # noqa: E402

PREFIX = ROOT + "/"

# ---------------------------------------------------------------- file reads
_reads: set[str] = set()
_orig_open = builtins.open
_orig_listdir = os.listdir
_orig_scandir = os.scandir


def _note(path) -> None:
    try:
        if isinstance(path, int):
            return
        p = os.fspath(path)
        if isinstance(p, bytes):
            p = p.decode()
        p = os.path.realpath(p)
        if p.startswith(PREFIX):
            _reads.add(p[len(PREFIX):])
    except Exception:
        pass


def _open(file, *a, **k):
    _note(file)
    return _orig_open(file, *a, **k)


def _listdir(path="."):
    _note(path)
    return _orig_listdir(path)


def _scandir(path="."):
    _note(path)
    return _orig_scandir(path)


builtins.open = _open
io.open = _open
os.listdir = _listdir
os.scandir = _scandir

# ------------------------------------------------------------- import phase
_imp_lines: dict[str, set[int]] = {}


def _imp_line(code, line):
    fn = code.co_filename
    if fn.startswith(PREFIX):
        _imp_lines.setdefault(fn[len(PREFIX):], set()).add(line)
    return sys.monitoring.DISABLE


sys.monitoring.use_tool_id(4, "brimport")
sys.monitoring.register_callback(4, sys.monitoring.events.LINE, _imp_line)
sys.monitoring.set_events(4, sys.monitoring.events.LINE)
mods, errors = fingerprint.import_all("sqlglot")
if errors:
    print("import errors", errors, file=sys.stderr)

tests = json.load(_orig_open(TESTS_JSON))["tests"]
by_module: dict[str, list[dict]] = {}
for t in tests:
    by_module.setdefault(t["file"], []).append(t)

module_reads: dict[str, list[str]] = {}
loader = unittest.TestLoader()
loaded: dict[int, unittest.TestSuite] = {}
for f in sorted(by_module):
    modname = f[:-3].replace("/", ".")
    _reads.clear()
    try:
        __import__(modname)
    except Exception as e:  # noqa: BLE001
        print("cannot import", modname, e, file=sys.stderr)
    module_reads[f] = sorted(_reads)
    for t in by_module[f]:
        parts = t["id"].split("::")
        name = modname + "." + ".".join(parts[1:])
        try:
            loaded[t["number"]] = unittest.TestSuite([loader.loadTestsFromName(name)])
        except Exception as e:  # noqa: BLE001
            print("cannot load", name, e, file=sys.stderr)

sys.monitoring.set_events(4, 0)
sys.monitoring.free_tool_id(4)

import inline  # noqa: E402

inline.install()

# ------------------------------------------------- attribute-read descriptors
_active = False
_lk: set = set()


class _TA:
    __slots__ = ("v", "k")

    def __init__(self, v, k):
        self.v = v
        self.k = k

    def __get__(self, obj, typ=None):
        if _active:
            _lk.add((typ if typ is not None else type(obj), self.k))
        return self.v


wrapped = 0
class_keys: dict[int, str] = {}
# list first: wrapping replaces a class's nested classes (dialect Tokenizers ...)
# in its __dict__, and a lazy walk would then never reach them
for key, c in list(fingerprint.iter_classes(sorted(sys.modules), "sqlglot")):
    class_keys[id(c)] = key
    if CALIB or isinstance(c, enum.EnumMeta):
        continue
    for name, v in list(c.__dict__.items()):
        if name.startswith("__") and name.endswith("__"):
            continue
        if hasattr(type(v), "__get__") and not isinstance(v, type):
            continue
        try:
            setattr(c, name, _TA(v, name))
            wrapped += 1
        except Exception:
            pass
print(f"wrapped {wrapped} class attributes", file=sys.stderr)


def _cls_key(c) -> str:
    k = class_keys.get(id(c))
    if k is None:
        k = f"{getattr(c, '__module__', '?')}:{getattr(c, '__qualname__', '?')}"
    return k


# ------------------------------------------------------------------ children
mon = sys.monitoring
TOOL = 3


def run_child(num: int) -> None:
    global _active
    lines: dict[str, set[int]] = {}

    def on_line(code, line):
        fn = code.co_filename
        if fn.startswith(PREFIX):
            s = lines.get(fn)
            if s is None:
                s = lines[fn] = set()
            s.add(line)
        return mon.DISABLE

    def on_start(code, offset):
        fn = code.co_filename
        if fn.startswith(PREFIX):
            s = lines.get(fn)
            if s is None:
                s = lines[fn] = set()
            s.add(-code.co_firstlineno)  # negative: function entered
        return mon.DISABLE

    _reads.clear()
    _lk.clear()
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
    if CALIB:
        res = unittest.TestResult()
        c0 = time.process_time()
        loaded[num].run(res)
        rec = {"n": num, "failed": bool(res.failures or res.errors), "cpu": round(time.process_time() - c0, 4), "spawned": spawned}
        tmp = os.path.join(OUT, f".{num}.json")
        with _orig_open(tmp, "w") as f:
            json.dump(rec, f)
        os.replace(tmp, os.path.join(OUT, f"{num}.json"))
        return
    mon.use_tool_id(TOOL, "brtrace")
    mon.register_callback(TOOL, mon.events.LINE, on_line)
    mon.register_callback(TOOL, mon.events.PY_START, on_start)
    mon.set_events(TOOL, mon.events.LINE | mon.events.PY_START)
    res = unittest.TestResult()
    c0 = time.process_time()
    w0 = time.time()
    _active = True
    try:
        loaded[num].run(res)
    finally:
        _active = False
        mon.set_events(TOOL, 0)
    cpu = time.process_time() - c0
    wall = time.time() - w0
    failed = bool(res.failures or res.errors or res.unexpectedSuccesses)
    msgs = [tb[-600:] for _, tb in (res.failures + res.errors)][:3]
    rec = {
        "n": num,
        "failed": failed,
        "skipped": len(res.skipped),
        "msgs": msgs,
        "cpu": round(cpu, 4),
        "wall": round(wall, 4),
        "lines": {fn[len(PREFIX):]: sorted(s) for fn, s in lines.items()},
        "lk": sorted({f"{_cls_key(c)}|{a}" for c, a in _lk}),
        "reads": sorted(_reads),
        "spawned": spawned,
    }
    tmp = os.path.join(OUT, f".{num}.json")
    with _orig_open(tmp, "w") as f:
        json.dump(rec, f)
    os.replace(tmp, os.path.join(OUT, f"{num}.json"))


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    with _orig_open(os.path.join(OUT, "_module_reads.json"), "w") as f:
        json.dump(module_reads, f)
    with _orig_open(os.path.join(OUT, "_import_lines.json"), "w") as f:
        json.dump({k: sorted(v) for k, v in _imp_lines.items()}, f)
    todo = [n for n in sorted(loaded) if not os.path.exists(os.path.join(OUT, f"{n}.json"))]
    if ONLY:
        todo = [n for n in todo if str(n) in ONLY]
    # longest first so the big tests do not trail at the end
    secs = {t["number"]: t["seconds"] for t in tests}
    todo.sort(key=lambda n: -secs.get(n, 0))
    running: dict[int, tuple[int, float]] = {}
    i = 0
    t0 = time.time()
    while i < len(todo) or running:
        while i < len(todo) and len(running) < JOBS:
            n = todo[i]
            i += 1
            pid = os.fork()
            if pid == 0:
                try:
                    resource.setrlimit(resource.RLIMIT_CPU, (900, 900))
                    run_child(n)
                    os._exit(0)
                except BaseException as e:  # noqa: BLE001
                    try:
                        print("child error", n, repr(e), file=sys.stderr)
                    finally:
                        os._exit(3)
            running[pid] = (n, time.time())
        pid, status = os.wait()
        if pid not in running:
            continue
        n, _ = running.pop(pid)
        if status != 0:
            print("test", n, "child exit", status, file=sys.stderr)
        done = len(todo) - (len(todo) - i) - len(running)
        if done % 100 == 0:
            print(f"{done}/{len(todo)} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
