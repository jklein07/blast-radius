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
import gc
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


# ------------------------------------------ dict-valued attributes, by key
# A class attribute holding a plain dict is swapped for a _RD copy that
# records which keys a test looks up (hits and misses alike). Anything that
# depends on more than single keys (iteration, len, ==, copies, merges,
# mutation, pickling) marks the whole dict as read. Overriding __iter__ also
# keeps CPython's merge fast path ({**d}, dict(d), d2.update(d)) off it, so
# those go through keys() and count as whole reads.
_kl: dict = {}  # tag index -> keys looked up in this test
_kla: set = set()  # tag indexes read as a whole in this test
KL_TAGS: list = []
_DBG = os.environ.get("TRACE_KL_DEBUG")
_DBG_HITS: list = []  # tag index -> ["class key|attr", ...] (one dict can sit in several classes)


class _Len:  # pseudo-key: the dict's size was read (len(), truth tests)
    __slots__ = ()


_LEN = _Len()


def _klk(t, k):
    s = _kl.get(t)
    if s is None:
        s = _kl[t] = set()
    try:
        s.add(k)
    except TypeError:
        if _DBG and t not in _kla:
            import traceback
            _DBG_HITS.append("UNHASHABLE " + repr(type(k)) + " " + str(KL_TAGS[t][:2]) + "\n" + "".join(traceback.format_stack(limit=7)[:-1]) + "\n")
        _kla.add(t)


class _RD(dict):
    __slots__ = ("_t",)

    def __getitem__(self, k):
        if _active:
            _klk(self._t, k)
        return dict.__getitem__(self, k)

    def get(self, k, d=None):
        if _active:
            _klk(self._t, k)
        return dict.get(self, k, d)

    def __contains__(self, k):
        if _active:
            _klk(self._t, k)
        return dict.__contains__(self, k)

    def __len__(self):  # also truth tests: depends only on the size
        if _active:
            _klk(self._t, _LEN)
        return dict.__len__(self)

    def __reduce_ex__(self, proto):
        if _active:
            if _DBG and self._t not in _kla:
                import traceback
                _DBG_HITS.append("REDUCE " + str(KL_TAGS[self._t][:2]) + "\n" + "".join(traceback.format_stack(limit=9)[:-1]) + "\n")
            _kla.add(self._t)
        return (dict, (dict.copy(self),))

    __reduce__ = lambda self: self.__reduce_ex__(2)  # noqa: E731


def _bulk(name):
    base = getattr(dict, name)

    def f(self, *a, **k):
        if _active:
            if _DBG and self._t not in _kla:
                import traceback
                _DBG_HITS.append("BULK " + name + " " + str(KL_TAGS[self._t][:2]) + "\n" + "".join(traceback.format_stack(limit=7)[:-1]) + "\n")
            _kla.add(self._t)
        return base(self, *a, **k)

    f.__name__ = name
    return f


for _n in ("__iter__", "keys", "values", "items", "__repr__", "copy", "__eq__", "__ne__", "__or__",
           "__ror__", "__ior__", "__reversed__", "setdefault", "pop", "popitem", "update", "__setitem__",
           "__delitem__", "clear", "__sizeof__"):
    setattr(_RD, _n, _bulk(_n))


def _foreign_refs(v, owners_ns: list) -> bool:
    """True when anything but the owning class namespaces refers to v: reads
    through that reference would bypass the recording copy."""
    import types as _types

    vid = id(v)
    for r in gc.get_referrers(v):
        if isinstance(r, _types.FrameType) or r is _dict_owners:
            continue
        if isinstance(r, dict) and any(r is ns or (id(r.get(n)) == vid and r.keys() == ns.keys()) for ns, n in owners_ns):
            continue
        return True
    return False


wrapped = 0
class_keys: dict[int, str] = {}
# list first: wrapping replaces a class's nested classes (dialect Tokenizers ...)
# in its __dict__, and a lazy walk would then never reach them
_all_classes = list(fingerprint.iter_classes(sorted(sys.modules), "sqlglot"))
_dict_owners: dict = {}  # id(dict) -> [(class, key, attr)]
_rd_for: dict = {}  # id(original dict) -> its _RD
_keep: list = []


def _build_rd() -> None:
    # (a function, so no module-level variable holds a dict while referrers are checked)
    for key, c in _all_classes:
        if CALIB or isinstance(c, enum.EnumMeta):
            continue
        for name in list(c.__dict__):
            if not (name.startswith("__") and name.endswith("__")) and type(c.__dict__[name]) is dict:
                _dict_owners.setdefault(id(c.__dict__[name]), []).append((c, key, name))
    gc.collect()
    for vid, os_ in _dict_owners.items():
        c0, _k0, n0 = os_[0]
        if _foreign_refs(c0.__dict__[n0], [(c.__dict__, n) for c, _k, n in os_]):
            continue
        _keep.append(c0.__dict__[n0])  # keep the original alive: ids in _rd_for stay unique
        rd = _RD(c0.__dict__[n0])
        rd._t = len(KL_TAGS)
        KL_TAGS.append(sorted({f"{k}|{n}" for _c, k, n in os_}))
        _rd_for[vid] = rd


_build_rd()
for key, c in _all_classes:
    class_keys[id(c)] = key
    if CALIB or isinstance(c, enum.EnumMeta):
        continue
    for name, v in list(c.__dict__.items()):
        if name.startswith("__") and name.endswith("__"):
            continue
        if hasattr(type(v), "__get__") and not isinstance(v, type):
            continue
        try:
            setattr(c, name, _TA(_rd_for.get(id(v), v) if type(v) is dict else v, name))
            wrapped += 1
        except Exception:
            pass
print(f"wrapped {wrapped} class attributes, {len(KL_TAGS)} dicts by key of {len(_dict_owners)}", file=sys.stderr)


def _dispatch_by_key() -> None:
    """Generator.__init__ builds (once per class, inside a test) a dispatch
    table: a copy of cls.TRANSFORMS plus the class's *_sql methods. That copy
    would read TRANSFORMS as a whole in every test that generates SQL. An
    entry of the table differs between two versions only when TRANSFORMS
    differs at that key, so lookups in the table are recorded as lookups of
    the same key in TRANSFORMS instead."""
    if CALIB:
        return
    import sqlglot.generator as g

    orig = g._build_dispatch

    def build(cls):
        global _active
        tr = cls.TRANSFORMS  # (records the attribute read)
        was = _active
        _active = False
        try:
            d = orig(cls)
        finally:
            _active = was
        if isinstance(tr, _RD):
            rd = _RD(d)
            rd._t = tr._t
            return rd
        return d

    g._build_dispatch = build


_dispatch_by_key()


def _cls_key(c) -> str:
    k = class_keys.get(id(c))
    if k is None:
        k = f"{getattr(c, '__module__', '?')}:{getattr(c, '__qualname__', '?')}"
    return k


def _lk_keys(c, a: str) -> list[str]:
    """A read through a class the import-time walk never saw (defined in a
    test, or built while one runs) also counts as a read through each of its
    sqlglot ancestors: the selector only knows sqlglot's own class tree."""
    out = [f"{_cls_key(c)}|{a}"]
    if id(c) not in class_keys:
        for b in getattr(c, "__mro__", ())[1:]:
            kb = class_keys.get(id(b))
            if kb is not None:
                out.append(f"{kb}|{a}")
    return out


def _kl_out() -> dict:
    out: dict = {}
    for t, keys in _kl.items():
        if t in _kla:
            continue
        ids = sorted({fingerprint.LEN_KEY if k is _LEN else fingerprint.key_id(k) for k in keys})
        for tag in KL_TAGS[t]:
            out[tag] = ids
    return out


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
    _kl.clear()
    _kla.clear()
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
        "lk": sorted({k for c, a in _lk for k in _lk_keys(c, a)}),
        "reads": sorted(_reads),
        "spawned": spawned,
        "kl": _kl_out(),
        "kla": sorted({tag for t in _kla for tag in KL_TAGS[t]}),
        **({"kldbg": _DBG_HITS} if _DBG else {}),
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
    with _orig_open(os.path.join(OUT, "_kl_tags.json"), "w") as f:
        json.dump(sorted({tag for tags in KL_TAGS for tag in tags}), f)  # dicts recorded by key
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
