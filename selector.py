"""Blast Radius selector: pick the sqlglot tests a patch can break.

Pipeline for every commit:

1. Parse the diff. For each changed Python file, diff the old and new ASTs
   (astdiff.py). Comment and formatting edits change nothing.
2. Changes inside function or lambda bodies map to tests through per-test line
   coverage of the pinned commit (maps/, built by tracer.py).
3. Changes that act at import time (class bodies, module constants, def
   headers) are measured: a child process imports the patched sqlglot,
   fingerprints every class attribute and module global, and compares with
   the pristine fingerprint. Each changed (class, attribute) maps to the tests
   that read it through that class or a subclass; each changed global maps
   to the tests that execute a line referring to it.
4. Data files (SQL fixtures and so on) map to the tests that opened them.
5. Candidates that are cheap to run are run on the patched tree, in a forked
   child, and kept only if they fail. Expensive ones are kept unverified when
   that is cheaper. The final set is packed onto 1-8 workers to minimise
   busiest-worker time + 2 s per worker.

Everything is decided from the pinned commit's own code and tests: there is
no table from patches or kinds of edit to failing tests.
"""

from __future__ import annotations

import gzip
import json
import os
import signal
import sys
import time

APP = os.path.dirname(os.path.abspath(__file__))
PRISTINE = os.environ.get("BR_SQLGLOT", "/opt/sqlglot")
DATA = os.environ.get("BR_DATA", "/data")
MAPS = os.environ.get("BR_MAPS", os.path.join(APP, "maps", "maps.json.gz"))
SNAP = os.environ.get("BR_SNAP", os.path.join(APP, "snap.json"))
TMP = os.environ.get("BR_TMP", "/tmp")
WORKER_MS = 2000
KAPPA = float(os.environ.get("BR_KAPPA", "1.15"))  # CPU seconds per recorded test second
CHILD_OVH = float(os.environ.get("BR_OVH", "0.30"))  # CPU seconds to start a verification child
SNAP_OVH = float(os.environ.get("BR_SNAP_OVH", "1.2"))
VERIFY = os.environ.get("BR_VERIFY", "1") == "1"
# prior chance that a candidate test actually fails, by why it is a candidate
P_DATA = float(os.environ.get("BR_P_DATA", "0.9"))
P_CODE = float(os.environ.get("BR_P_CODE", "0.35"))
P_ATTR = float(os.environ.get("BR_P_ATTR", "0.35"))
P_TEST = float(os.environ.get("BR_P_TEST", "0.5"))
P_LOW = 0.05
DEBUG = os.environ.get("BR_DEBUG", "0") == "1"

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.path.insert(0, APP)
import astdiff  # noqa: E402
import patchapply  # noqa: E402

T0 = time.time()


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# --------------------------------------------------------------------- inputs
TESTS = json.load(open(os.path.join(DATA, "tests.json")))["tests"]
NUM_BY_ID = {t["id"]: t["number"] for t in TESTS}
MS = {t["number"]: round(t["seconds"] * 1000) for t in TESTS}
SEC = {t["number"]: t["seconds"] for t in TESTS}
FILE_OF = {t["number"]: t["file"] for t in TESTS}
ALL = [t["number"] for t in TESTS]
BY_FILE: dict[str, list[int]] = {}
for t in TESTS:
    BY_FILE.setdefault(t["file"], []).append(t["number"])

_BIN = MAPS[: -len(".json.gz")] + ".bin" if MAPS.endswith(".json.gz") else MAPS + ".bin"
if os.path.exists(_BIN):
    import marshal

    with open(_BIN, "rb") as fh:
        M = marshal.load(fh)
else:
    with gzip.open(MAPS, "rt") as fh:
        M = json.load(fh)
_ids = M["tests"]  # map-local index -> test id
_local2num = [NUM_BY_ID.get(i) for i in _ids]
_known = set(x for x in _local2num if x is not None)
UNKNOWN = [n for n in ALL if n not in _known]
_IDENTITY = all(_local2num[i] == i for i in range(len(_local2num)))


import array as _array


def _u16(b):
    if isinstance(b, (bytes, bytearray)):
        a = _array.array("H")
        a.frombytes(b)
        return a.tolist()
    return b


def _i32(b):
    if isinstance(b, (bytes, bytearray)):
        a = _array.array("i")
        a.frombytes(b)
        return a.tolist()
    return b


class _Sets:
    """Test sets by id, decoded and translated to this run's test numbers on first use."""

    def __init__(self, raw):
        self.raw = raw
        self.cache: dict = {}

    def __getitem__(self, i):
        v = self.cache.get(i)
        if v is None:
            xs = _u16(self.raw[i])
            if not _IDENTITY:
                xs = [_local2num[x] for x in xs if _local2num[x] is not None]
            v = self.cache[i] = xs
        return v


class _Cov:
    """file -> {line: set id}, decoded per file on first use."""

    def __init__(self, raw):
        self.raw = raw
        self.cache: dict = {}

    def get(self, f, default=None):
        v = self.cache.get(f)
        if v is None:
            r = self.raw.get(f)
            if r is None:
                return default
            if isinstance(r, dict):
                v = {int(k): x for k, x in r.items()}
            else:
                xs = _i32(r)
                v = dict(zip(xs[0::2], xs[1::2]))
            self.cache[f] = v
        return v

    def __contains__(self, f):
        return f in self.raw

    def __getitem__(self, f):
        return self.get(f)


class _Names:
    def __init__(self, raw, files):
        self.raw = raw
        self.files = files

    def refs(self, mod, name):
        r = self.raw.get(mod, {}).get(name)
        if r is None:
            return []
        if isinstance(r, list):
            return r
        xs = _i32(r)
        return [(self.files[a], b) for a, b in zip(xs[0::2], xs[1::2])]


SETS = _Sets(M["sets"])
COV = _Cov(M["cov"])  # file -> {line: setid}
LK: dict = M["lk"]  # "classkey|attr" -> setid
CLASSUSE: dict = M["classuse"]  # classkey -> setid
READS: dict = M["reads"]  # path -> setid
DIRREADS: dict = M.get("dirreads", {})
MODREADS: dict = M["module_reads"]  # test file -> [paths read while importing]
MODLINES: dict = M.get("module_lines", {})  # test file -> {sqlglot file: [lines run while importing]}
IMPORT_LINES = M["import_lines"]
NAMES = _Names(M["names"], M.get("name_files", []))  # module -> name -> [(file, line), ...]
TEST_IMPORTERS: dict = M["test_importers"]  # test module file -> [test files importing it, transitively]
NOVERIFY = {_local2num[i] for i in M["noverify"] if _local2num[i] is not None}
VCPU = {}
for _i, _c in M.get("vcpu", {}).items():
    _n = _local2num[int(_i)]
    if _n is not None:
        VCPU[_n] = _c
SPAWNERS = {_local2num[i] for i in M.get("spawners", []) if _local2num[i] is not None}
HEAVY_FILES = {"tests/test_optimizer.py", "tests/test_executor.py"}


def _cls(n: int) -> str:
    parts = TESTS_BY_NUM_ID[n].split("::")
    return parts[0] + "::" + (parts[1] if len(parts) > 2 else "")


TESTS_BY_NUM_ID = {t["number"]: t["id"] for t in TESTS}
CLASS_SETUP: dict[str, float] = {}
for _n, _c in VCPU.items():
    _k = _cls(_n)
    CLASS_SETUP[_k] = min(CLASS_SETUP.get(_k, _c), _c)


def vcost(n: int) -> float:
    """Estimated marginal CPU seconds to verify test n (class setup excluded)."""
    c = VCPU.get(n)
    if c is None:
        return KAPPA * SEC[n]
    return max(0.0, c - CLASS_SETUP.get(_cls(n), 0.0)) + 0.002


def setup_cost(n: int) -> float:
    return CLASS_SETUP.get(_cls(n), 0.0) if n in VCPU else 0.0
LK_BY_ATTR: dict[str, list[str]] = {}
for k in LK:
    cls, _, attr = k.rpartition("|")
    LK_BY_ATTR.setdefault(attr, []).append(cls)
PYTEST_CONFIG = {"setup.cfg", "pyproject.toml", "pytest.ini", "tox.ini", "conftest.py", "tests/conftest.py"}


def cov_tests(path: str, lines) -> set:
    out: set = set()
    fm = COV.get(path)
    if not fm:
        return out
    for ln in lines:
        sid = fm.get(ln)
        if sid is not None:
            out.update(SETS[sid])
    # tests whose module import (not the test itself) ran these lines
    for tf, files in MODLINES.items():
        ls = files.get(path)
        if ls and any(l in ls for l in lines):
            out |= set(BY_FILE.get(tf, []))
    return out


def func_span_tests(path: str, first: int, last: int) -> set:
    out: set = set()
    fm = COV.get(path)
    if not fm:
        return out
    for ln, sid in fm.items():
        li = int(ln)
        if first <= abs(li) <= last:
            out.update(SETS[sid])
    return out


# ----------------------------------------------------------------- scheduling
def schedule(nums) -> list[list[int]]:
    nums = sorted(set(nums), key=lambda n: (-MS[n], n))
    if not nums:
        return []
    best = None
    for w in range(1, 9):
        if w > len(nums):
            break
        loads = [0] * w
        bins: list[list[int]] = [[] for _ in range(w)]
        for n in nums:
            i = loads.index(min(loads))
            bins[i].append(n)
            loads[i] += MS[n]
        cost = max(loads) + WORKER_MS * w
        if best is None or cost < best[0]:
            best = (cost, [b for b in bins if b])
    return best[1]


def sched_cost_ms(nums) -> int:
    b = schedule(nums)
    if not b:
        return 0
    return max(sum(MS[n] for n in x) for x in b) + WORKER_MS * len(b)


def choose_verify(cands: dict, already_forked: bool) -> tuple[list[int], list[int]]:
    """Split candidates into (verify, include) minimising expected cost.

    Verifying test i costs its CPU; it removes i from the schedule only if it
    passes (probability 1 - p_i). Expected schedule cost is approximated as
    P(all verified pass) * S(include) + P(some fail) * S(include + verified).
    """
    if not VERIFY:
        return [], sorted(cands)
    fixed = sorted(n for n in cands if n in NOVERIFY)
    free = [n for n in cands if n not in NOVERIFY]
    if not free:
        return [], fixed
    ovh = 0.0 if already_forked else CHILD_OVH

    def key(n):
        return ((vcost(n) + setup_cost(n)) / max(1e-3, 1.0 - cands[n]), n)

    free.sort(key=key)
    m = len(free)
    ks = sorted({0, m} | {int(m * f) for f in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)} | set(range(min(m, 40))))
    prefix = [0.0]
    pnone = [1.0]
    seen_cls: set = set()
    for n in free:
        extra = 0.0
        k_ = _cls(n)
        if k_ not in seen_cls:
            seen_cls.add(k_)
            extra = setup_cost(n)
        prefix.append(prefix[-1] + vcost(n) + extra)
        pnone.append(pnone[-1] * (1.0 - cands[n]))
    best = None
    for k in ks:
        if k == 0:
            cost = sched_cost_ms(fixed + free) / 1000.0
        else:
            s_u = sched_cost_ms(fixed + free[k:]) / 1000.0
            s_all = sched_cost_ms(fixed + free) / 1000.0
            cost = ovh + prefix[k] + pnone[k] * s_u + (1.0 - pnone[k]) * s_all
        if best is None or cost < best[0] - 1e-9:
            best = (cost, k)
    k = best[1]
    return free[:k], fixed + free[k:]


# ------------------------------------------------------------------- analysis
class Cands(dict):
    """test number -> prior probability that it fails."""

    def add(self, nums, p: float) -> None:
        for n in nums:
            if self.get(n, -1.0) < p:
                self[n] = p


class Plan:
    def __init__(self, cid: str):
        self.cid = cid
        self.cands = Cands()
        self.need_snapshot = False
        self.all_tests = False  # include everything, unverified
        self.files: dict[str, str | None] = {}  # rel path -> new text (None = deleted)
        self.py_changes: dict[str, astdiff.PyChange] = {}
        self.notes: list[str] = []


def read_pristine(rel: str) -> str | None:
    p = os.path.join(PRISTINE, rel)
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8", errors="surrogateescape", newline="") as fh:
        return fh.read()


def module_of(rel: str) -> str:
    m = rel[:-3].replace("/", ".")
    if m.endswith(".__init__"):
        m = m[: -len(".__init__")]
    return m


def analyze(cid: str, text: str) -> Plan:
    plan = Plan(cid)
    try:
        fps = patchapply.parse(text)
    except Exception as e:  # noqa: BLE001
        plan.all_tests = True
        plan.notes.append(f"unparseable patch {e}")
        return plan
    for fp in fps:
        old_rel = fp.old_path
        new_rel = fp.new_path
        rels = {r for r in (old_rel, new_rel) if r}
        if fp.is_binary:
            for r in rels:
                plan.cands.add(tests_reading(r, True), P_DATA)
            continue
        try:
            old = read_pristine(old_rel) if old_rel and not fp.is_new else None
            new = patchapply.apply_to_text(old, fp) if not fp.is_deleted else None
        except Exception as e:  # noqa: BLE001
            plan.all_tests = True
            plan.notes.append(f"cannot apply {fp.path}: {e}")
            continue
        if old_rel and new_rel and old_rel != new_rel:
            plan.files[old_rel] = None
        if new_rel:
            plan.files[new_rel] = new
        elif old_rel:
            plan.files[old_rel] = None
        for r in rels:
            if os.path.basename(r) in ("conftest.py",) or r in PYTEST_CONFIG:
                if r in ("setup.cfg", "pyproject.toml", "tox.ini") and old is not None and new is not None and "pytest" not in (old + new).lower():
                    continue
                plan.all_tests = True
                plan.notes.append(f"pytest config {r}")
        path = new_rel or old_rel
        listing = fp.is_new or fp.is_deleted or (old_rel != new_rel)
        for r in rels:  # tests that open the file itself (source readers too)
            plan.cands.add(tests_reading(r, listing), P_CODE if r.endswith(".py") else P_DATA)
        if path.endswith(".py") and (path.startswith("sqlglot/") or path.startswith("tests/")):
            ch = astdiff.analyze(old, new)
            if old_rel != new_rel and old_rel and new_rel:
                ch.semantic = True
                ch.import_time = True
            plan.py_changes[path] = ch
            if not ch.semantic:
                continue
            if path.startswith("sqlglot/"):
                sqlglot_change(plan, old_rel or path, ch)
            else:
                test_change(plan, old_rel or path, ch)
        else:
            # data file (already added above as a read file)
            for r in rels:
                if r.endswith(".py") and r.startswith("sqlglot/"):
                    plan.need_snapshot = True
    return plan


def tests_reading(rel: str, listing: bool = False) -> set:
    out: set = set()
    sid = READS.get(rel)
    if sid is not None:
        out.update(SETS[sid])
    for tf, paths in MODREADS.items():
        if rel in paths:
            out |= set(BY_FILE.get(tf, []))
            for imp in TEST_IMPORTERS.get(tf, []):
                out |= set(BY_FILE.get(imp, []))
    if listing:  # the set of files in a folder changes only on add/remove/rename
        d = os.path.dirname(rel)
        sid = DIRREADS.get(d)
        if sid is not None:
            out.update(SETS[sid])
    return out


def sqlglot_change(plan: Plan, rel: str, ch: astdiff.PyChange) -> None:
    plan.cands.add(SPAWNERS, P_LOW)
    if ch.syntax_error:
        plan.need_snapshot = True  # the import error decides
        return
    if ch.runtime_lines:
        plan.cands.add(cov_tests(rel, ch.runtime_lines), P_CODE)
        # functions that also run while sqlglot is imported change import-time state
        imp = set(IMPORT_LINES.get(rel, ()))
        if any(l in imp for l in ch.runtime_lines if l > 0):
            plan.need_snapshot = True
    for q, first, last in ch.def_changes:
        plan.cands.add(func_span_tests(rel, first, last), P_CODE)
    if ch.import_time:
        plan.need_snapshot = True


def test_change(plan: Plan, rel: str, ch: astdiff.PyChange) -> None:
    if ch.syntax_error or ch.import_time:
        plan.cands.add(BY_FILE.get(rel, []), P_TEST)
        for imp in TEST_IMPORTERS.get(rel, []):
            plan.cands.add(BY_FILE.get(imp, []), P_TEST)
        if rel not in BY_FILE and not TEST_IMPORTERS.get(rel):
            # a helper module no test file is known to import: be safe
            plan.cands.add(tests_reading(rel), P_TEST)
    if ch.runtime_lines:
        plan.cands.add(cov_tests(rel, ch.runtime_lines), P_TEST)
        for q, first, last in ch.def_changes:
            plan.cands.add(func_span_tests(rel, first, last), P_TEST)


# ---------------------------------------------------- snapshot-diff candidates
_SNAP = None


def pristine_snapshot():
    global _SNAP
    if _SNAP is None:
        with open(SNAP) as fh:
            _SNAP = json.load(fh)
    return _SNAP


def snapshot_candidates(pristine: dict, patched: dict, diff: dict, plan: Plan) -> tuple[set, bool]:
    """Tests affected by changed import-time state. Returns (tests, everything)."""
    out: set = set()
    if diff["errors"]:
        new_err = {m for m, _ in diff["errors"] if m in patched["errors"]}
        if new_err:
            return set(ALL), True
    # subclass index over both versions
    children: dict[str, set] = {}
    for snap in (pristine, patched):
        for k, mro in snap["bases"].items():
            for b in mro:
                children.setdefault(b, set()).add(k)

    def subtree(k: str) -> set:
        return {k} | children.get(k, set())

    for key, attr in diff["attrs"]:
        sub = subtree(key)
        if attr == "*":
            for x in sub:
                sid = CLASSUSE.get(x)
                if sid is not None:
                    out.update(SETS[sid])
            continue
        hit = False
        for x in LK_BY_ATTR.get(attr, ()):
            if x in sub:
                out.update(SETS[LK[f"{x}|{attr}"]])
                hit = True
        # methods (unwrapped) and attributes no traced test ever read: fall back
        # to every test that used the class family, restricted to tests that ran
        # code of a same-named function anywhere in that family's sources.
        meth = method_lines(key, attr, pristine, patched)
        if meth is not None:
            out |= meth
        elif not hit:
            pass
    for mod, name in diff["globals"]:
        out |= global_ref_tests(mod, name)
    return out, False


def method_lines(key: str, attr: str, pristine: dict, patched: dict):
    """Tests that ran a function named attr defined on key or its ancestors."""
    mro = [key] + pristine["bases"].get(key, []) + patched["bases"].get(key, [])
    out: set = set()
    found = False
    for k in mro:
        mod, _, qual = k.partition(":")
        rel = mod.replace(".", "/") + ".py"
        if rel not in COV:
            rel = mod.replace(".", "/") + "/__init__.py"
        spans = M["defs"].get(rel, {}).get(f"{qual}.{attr}")
        if spans:
            found = True
            out |= func_span_tests(rel, spans[0], spans[1])
    if not found:
        return None
    # restrict to tests that touched the class family at all
    fam: set = set()
    for x in [key]:
        sid = CLASSUSE.get(x)
        if sid is not None:
            fam.update(SETS[sid])
    return out


def global_ref_tests(mod: str, name: str) -> set:
    out: set = set()
    if name == "*":
        rel = mod.replace(".", "/") + ".py"
        for ln, sid_ in COV.get(rel, {}).items():
            out.update(SETS[sid_])
        return out
    for f, ln in NAMES.refs(mod, name):
        out |= cov_tests(f, [ln])
    rel = mod.replace(".", "/") + ".py"
    spans = M["defs"].get(rel, {}).get(name)
    if spans:
        out |= func_span_tests(rel, spans[0], spans[1])
    return out


# -------------------------------------------------------------- child process
def build_farm(cid: str, files: dict) -> str:
    """A tree that looks like the patched checkout: symlinks to the pristine
    files, real files only where the patch changed something."""
    root = os.path.join(TMP, "w", cid)
    os.makedirs(root, exist_ok=True)
    real_dirs = {""}
    for rel in files:
        parts = rel.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            real_dirs.add("/".join(parts[:i]))
    for d in sorted(real_dirs, key=len):
        src = os.path.join(PRISTINE, d) if d else PRISTINE
        dst = os.path.join(root, d) if d else root
        os.makedirs(dst, exist_ok=True)
        if not os.path.isdir(src):
            continue
        for name in os.listdir(src):
            sub = f"{d}/{name}" if d else name
            if sub in real_dirs or sub in files:
                continue
            target = os.path.join(dst, name)
            if not os.path.lexists(target):
                os.symlink(os.path.join(src, name), target)
    for rel, text in files.items():
        dst = os.path.join(root, rel)
        if os.path.lexists(dst):
            os.remove(dst)
        if text is None:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8", errors="surrogateescape", newline="") as fh:
            fh.write(text)
    return root


class _Timeout(BaseException):
    pass


def _on_alarm(*_):
    raise _Timeout()


def run_tests(nums: list[int]) -> tuple[set, set]:
    """Run tests in this process. Returns (failing, unsure)."""
    import unittest

    failing: set = set()
    unsure: set = set()
    loader = unittest.TestLoader()
    id2num = {}
    by_class: dict = {}
    for n in nums:
        tid = next(t["id"] for t in [TESTS_BY_NUM[n]])
        parts = tid.split("::")
        mod = FILE_OF[n][:-3].replace("/", ".")
        uid = mod + "." + ".".join(parts[1:])
        id2num[uid] = n
        by_class.setdefault((mod, parts[1] if len(parts) > 2 else ""), []).append((n, uid))
    limits = {uid: 3.0 * SEC[n] + 1.5 for uid, n in id2num.items()}

    class R(unittest.TestResult):
        def startTest(self, test):
            super().startTest(test)
            signal.setitimer(signal.ITIMER_REAL, limits.get(test.id(), 5.0))

        def stopTest(self, test):
            signal.setitimer(signal.ITIMER_REAL, 0)
            super().stopTest(test)

    signal.signal(signal.SIGALRM, _on_alarm)
    suites = []
    for (mod, cls), items in by_class.items():
        suite = unittest.TestSuite()
        for n, uid in items:
            try:
                suite.addTest(loader.loadTestsFromName(uid))
            except BaseException:  # noqa: BLE001
                failing.add(n)
        suites.append((items, suite))
    import inline

    inline.install()
    for items, suite in suites:
        res = R()
        try:
            suite.run(res)
        except BaseException:  # noqa: BLE001  (timeout in a fixture, SystemExit, ...)
            signal.setitimer(signal.ITIMER_REAL, 0)
            failing |= {n for n, _ in items}
            continue
        for test, _tb in res.failures + res.errors + [(x, "") for x in res.unexpectedSuccesses]:
            tc = getattr(test, "test_case", None)
            tid = (tc or test).id()
            if tid in id2num:
                failing.add(id2num[tid])
            else:
                failing |= {n for n, _ in items}
    return failing, unsure


TESTS_BY_NUM = {t["number"]: t for t in TESTS}


def child_main(plan: Plan, wfd: int) -> None:
    """Runs in a forked child. Writes one JSON object to wfd."""
    result: dict = {"cid": plan.cid}
    os.dup2(2, 1)  # tests may print: never let a child write to our stdout
    try:
        root = build_farm(plan.cid, plan.files)
        os.chdir(root)
        sys.path.insert(0, root)
        os.environ["SKIP_INTEGRATION"] = "0"
        sys.dont_write_bytecode = True
        cands = Cands(plan.cands)
        everything = False
        if plan.need_snapshot:
            import fingerprint

            pristine = pristine_snapshot()
            patched = fingerprint.snapshot("sqlglot")
            d = fingerprint.diff(pristine, patched)
            extra, everything = snapshot_candidates(pristine, patched, d, plan)
            cands.add(extra, P_ATTR)
            result["diff"] = {"attrs": d["attrs"][:50], "globals": d["globals"][:50], "errors": d["errors"][:5], "n": len(extra)}
        if everything:
            result["select"] = sorted(ALL)
            result["verified"] = 0
        else:
            verify, include = choose_verify(cands, True)
            failing, _ = run_tests(verify) if verify else (set(), set())
            result["select"] = sorted(set(include) | failing)
            result["verified"] = len(verify)
            result["failing_verified"] = sorted(failing)
            result["cands"] = len(cands)
    except BaseException as e:  # noqa: BLE001
        import traceback

        result["error"] = traceback.format_exc()[-1500:]
    import resource as _r

    _u = _r.getrusage(_r.RUSAGE_SELF)
    _c = _r.getrusage(_r.RUSAGE_CHILDREN)
    result["cpu"] = round(_u.ru_utime + _u.ru_stime + _c.ru_utime + _c.ru_stime, 3)
    data = (json.dumps(result) + "\n").encode()
    while data:
        k = os.write(wfd, data)
        data = data[k:]
    os.close(wfd)


# ------------------------------------------------------------------------ main
def main() -> None:
    commits = sorted(p[:-6] for p in os.listdir(os.path.join(DATA, "commits")) if p.endswith(".patch"))
    n = len(commits)
    budget = float(os.environ.get("BR_BUDGET", 0)) or (150.0 if n <= 45 else 235.0) * max(1.0, n / (40 if n <= 45 else 60))
    plans: dict[str, Plan] = {}
    for cid in commits:
        with open(os.path.join(DATA, "commits", cid + ".patch"), encoding="utf-8", errors="surrogateescape", newline="") as fh:
            text = fh.read()
        try:
            plans[cid] = analyze(cid, text)
        except Exception as e:  # noqa: BLE001
            p = Plan(cid)
            p.all_tests = True
            p.notes.append(f"analyze error {e!r}")
            plans[cid] = p
    _ru = __import__("resource").getrusage(0)
    log(f"phase analyze cpu {_ru.ru_utime + _ru.ru_stime:.2f}")
    selections: dict[str, list[int]] = {}
    jobs = []
    for cid, p in plans.items():
        if UNKNOWN and (p.cands or p.need_snapshot):
            p.cands.add(UNKNOWN, P_LOW)
        if p.all_tests:
            selections[cid] = list(ALL)
        elif p.need_snapshot:
            jobs.append(cid)
        elif not p.cands:
            selections[cid] = []
        else:
            verify, include = choose_verify(p.cands, False)
            if not verify:
                selections[cid] = include
            else:
                jobs.append(cid)
    if any(plans[c].need_snapshot for c in jobs):
        pristine_snapshot()
    if any(plans[c].need_snapshot or any(FILE_OF[x] in HEAVY_FILES for x in plans[c].cands) for c in jobs):
        try:
            import numpy  # noqa: F401
            import pandas  # noqa: F401
            import pandas.testing  # noqa: F401
            import duckdb  # noqa: F401
        except Exception:  # noqa: BLE001
            pass
    _ru = __import__("resource").getrusage(0)
    log(f"phase preimport cpu {_ru.ru_utime + _ru.ru_stime:.2f}")
    # cheap first so a slow child cannot starve the rest
    jobs.sort(key=lambda c: (plans[c].need_snapshot, sum(SEC[x] for x in plans[c].cands)))
    running: dict[int, tuple[str, int, float]] = {}
    outputs: dict[str, dict] = {}
    max_par = int(os.environ.get("BR_PAR", "2"))
    queue = list(jobs)
    while queue or running:
        while queue and len(running) < max_par and time.time() - T0 < budget:
            cid = queue.pop(0)
            r, w = os.pipe()
            pid = os.fork()
            if pid == 0:
                os.close(r)
                try:
                    child_main(plans[cid], w)
                finally:
                    os._exit(0)
            os.close(w)
            running[pid] = (cid, r, time.time())
        if not running:
            break
        pid, _status = os.waitpid(-1, 0)
        if pid not in running:
            continue
        cid, r, st = running.pop(pid)
        buf = b""
        while True:
            chunk = os.read(r, 65536)
            if not chunk:
                break
            buf += chunk
        os.close(r)
        try:
            outputs[cid] = json.loads(buf.decode() or "{}")
        except Exception:  # noqa: BLE001
            outputs[cid] = {"error": "bad child output"}
        outputs[cid]["secs"] = round(time.time() - st, 2)
    for cid in jobs:
        o = outputs.get(cid)
        p = plans[cid]
        if o and "select" in o and "error" not in o:
            selections[cid] = o["select"]
        else:
            # no answer from the child (error or out of time): fall back safely
            selections[cid] = list(ALL) if p.need_snapshot else sorted(p.cands)
            log(cid, "fallback", (o or {}).get("error", "no output")[-400:])
    for cid in commits:
        sel = selections.get(cid, list(ALL))
        print(json.dumps({"commit": cid, "workers": schedule(sel)}), flush=True)
    if DEBUG:
        for cid in commits:
            p = plans[cid]
            o = outputs.get(cid, {})
            log(json.dumps({"cid": cid, "cands": len(p.cands), "snap": p.need_snapshot, "all": p.all_tests, "notes": p.notes,
                            "sel": len(selections[cid]), "cost": sched_cost_ms(selections[cid]) / 1000,
                            **{k: v for k, v in o.items() if k not in ("select",)}}))
    import resource

    ru = resource.getrusage(resource.RUSAGE_SELF)
    rc = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = ru.ru_utime + ru.ru_stime + rc.ru_utime + rc.ru_stime
    total = sum(sched_cost_ms(selections[c]) for c in commits) / 1000
    log(f"done {len(commits)} commits in {time.time() - T0:.1f}s cpu {cpu:.1f}s simulated(no misses) {total:.1f}s")


if __name__ == "__main__":
    main()
