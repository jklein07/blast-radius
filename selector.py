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

import gc
import gzip
import itertools
import json
import os
import select
import signal
import sys
import time
import unittest  # noqa: F401

# sqlglot decides at import time whether to run its strict argument checks
# (UNITTEST = "unittest" in sys.modules or "pytest" in sys.modules). Tests
# always run with them on, so every sqlglot import here must see unittest
# loaded first, or a verification child would pass tests that really fail.

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
P_CODE = float(os.environ.get("BR_P_CODE", "0.2"))
P_ATTR = float(os.environ.get("BR_P_ATTR", "0.2"))
P_TEST = float(os.environ.get("BR_P_TEST", "0.5"))
P_LOW = 0.05
DEBUG = os.environ.get("BR_DEBUG", "0") == "1"
PRELOAD = os.environ.get("BR_PRELOAD", "1") == "1"

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.path.insert(0, APP)
import astdiff  # noqa: E402
import patchapply  # noqa: E402

T0 = time.time()
PROF: list = []  # (stage, user, sys) CPU in a verification child, debug runs only


def _ph(stage: str) -> None:
    if DEBUG:
        t = os.times()
        PROF.append((stage, round(t.user, 3), round(t.system, 3)))


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
# tests that read a sqlglot module attribute (module.name, from-imports), by
# "module|name" (re-exports included) and by module (tracer.py TRACE_MODATTR);
# for the modules recorded there NAMES holds bare-name references only
MODATTR: dict = M.get("modattr", {})
MODATTR_MOD: dict = M.get("modattr_mod", {})
TEST_IMPORTERS: dict = M["test_importers"]  # test module file -> [test files importing it, transitively]
NOVERIFY = {_local2num[i] for i in M["noverify"] if _local2num[i] is not None}
VCPU = {}
VSCALE = float(os.environ.get("BR_VSCALE", "1.0"))  # scorer CPU per calibrated CPU second
for _i, _c in M.get("vcpu_alt" if os.environ.get("BR_CALIB") == "alt" and M.get("vcpu_alt") else "vcpu", {}).items():
    _n = _local2num[int(_i)]
    if _n is not None:
        VCPU[_n] = _c * VSCALE
SPAWNERS = {_local2num[i] for i in M.get("spawners", []) if _local2num[i] is not None}
LAZY_NUM = next((t["number"] for t in TESTS if t["id"] == "tests/dialects/test_dialect.py::TestDialect::test_lazy_load"), None)
# dict-valued class attributes the tracer recorded by key (tracer.py _RD)
KLTAGS = set(M.get("kltags", ()))
KL: dict = M.get("kl", {})  # "class|attr" -> {key id: setid of tests that looked it up}
KLA: dict = M.get("kla", {})  # "class|attr" -> setid of tests that read it as a whole
HEAVY_FILES = {"tests/test_optimizer.py", "tests/test_executor.py"}
MOD_IMPORTS: dict = M.get("mod_imports", {})
_IMPORTERS: dict = {}
for _m, _deps in MOD_IMPORTS.items():
    for _d in _deps:
        _IMPORTERS.setdefault(_d, set()).add(_m)


def importer_closure(changed: set) -> set:
    seen = set(changed)
    stack = list(changed)
    while stack:
        m = stack.pop()
        for imp in _IMPORTERS.get(m, ()):
            if imp not in seen:
                seen.add(imp)
                stack.append(imp)
    return seen


def affected_modules(changed: set):
    """Modules whose import-time state a change to `changed` can alter, or None
    when unsure (then everything is fingerprinted)."""
    if not MOD_IMPORTS or not changed:
        return None
    if any(m not in MOD_IMPORTS for m in changed):
        return None
    seen = set(changed)
    stack = list(changed)
    while stack:
        m = stack.pop()
        for imp in _IMPORTERS.get(m, ()):
            if imp not in seen:
                seen.add(imp)
                stack.append(imp)
    # import-time code of an affected module can mutate what it imports
    out = set(seen)
    for m in seen:
        out.update(MOD_IMPORTS.get(m, ()))
    return out


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


# ------------------------------------------------- query-level verification
# Tests that loop over the queries of a fixture which setUpClass loads into a
# class attribute. maps/split.json.gz (qtrace.py) records which queries run
# each line, so a change reaching only some queries is verified on those alone.
SPLIT_IDS = (
    "tests/test_executor.py::TestExecutor::test_execute_tpcds",
    "tests/test_executor.py::TestExecutor::test_execute_tpch",
    "tests/test_executor.py::TestExecutor::test_optimized_tpch",
)
SPLIT_NUMS = {NUM_BY_ID[t]: t for t in SPLIT_IDS if t in NUM_BY_ID}
_SPLIT = None
_REG = None


def reg_tests(rel: str, ch) -> set:
    """Tests that run a changed line only when the dialect registry is empty.
    tests/test_dialect_entry_points.py clears Dialect._classes in setUp and
    tearDown, so a test pytest runs after it in the same process looks its
    dialects up again through _try_load. maps/regcov.json.gz (tracer.py with
    TRACE_CLEARREG=1, minus the plain trace) lists, per line, the tests that
    run it only then. A child runs a test from a full registry, so these are
    kept, not verified."""
    global _REG
    if _REG is None:
        try:
            with gzip.open(os.path.join(os.path.dirname(MAPS), "regcov.json.gz"), "rt") as fh:
                _REG = json.load(fh)
        except OSError:
            _REG = {}
    fm = _REG.get(rel)
    if not fm:
        return set()
    lines = set(ch.runtime_lines)
    spans = [(first, last) for _q, first, last in ch.def_changes]
    out: set = set()
    for ln, nums in fm.items():
        li = int(ln)
        if li in lines or any(first <= abs(li) <= last for first, last in spans):
            out.update(nums)
    return out


def split_maps() -> dict:
    global _SPLIT
    if _SPLIT is None:
        try:
            with gzip.open(os.path.join(os.path.dirname(MAPS), "split.json.gz"), "rt") as fh:
                _SPLIT = json.load(fh)
        except Exception:  # noqa: BLE001
            _SPLIT = {}
    return _SPLIT


_FE = None


def fe_maps() -> dict:
    global _FE
    if _FE is None:
        try:
            with gzip.open(os.path.join(os.path.dirname(MAPS), "fe.json.gz"), "rt") as fh:
                _FE = json.load(fh)
            for v in _FE.values():
                v["_exec"] = {f: set(ls) for f, ls in v["exec"].items()}
        except Exception:  # noqa: BLE001
            _FE = {}
    return _FE


def changed_hits(n: int, plan) -> dict | None:
    """{sqlglot file: lines in test n's coverage that the changes reach}, or
    None when a change reaches it in a way no line map covers (import time,
    a deleted or unparsable module): then it must run whole."""
    tfile = FILE_OF[n]
    out: dict = {}
    for rel, ch in plan.py_changes.items():
        if not rel.startswith("sqlglot/") or not ch.semantic:
            continue
        old_src = read_pristine(rel)
        if old_src is None:
            continue  # a new module: reaches tests only through imports (snapshot)
        if plan.files.get(rel, "") is None or ch.syntax_error:
            return None
        fm = COV.get(rel) or {}
        ml = (MODLINES.get(tfile) or {}).get(rel)
        if ml and any(l in ml for l in ch.runtime_lines):
            return None
        hit = set()
        for l in ch.runtime_lines:
            sid = fm.get(l)
            if sid is not None and n in SETS[sid]:
                hit.add(l)
        spans = [(first, last) for _q, first, last in ch.def_changes]
        exe = None
        for lines, first, last in ch.runtime_groups:
            if any(l in fm for l in lines):
                continue
            if exe is None:
                exe = exec_lines(rel, old_src)
            if exe is not None and any(l > 0 and l in exe for l in lines):
                continue
            if -first in fm:
                if n in SETS[fm[-first]]:
                    hit.add(-first)
            else:
                spans.append((first, last))
        for first, last in spans:
            for ln, sid in fm.items():
                if first <= abs(ln) <= last and n in SETS[sid]:
                    hit.add(ln)
        if hit:
            out[rel] = hit
    return out


def split_queries(n: int, plan) -> tuple | None:
    """(attr, query indexes, nq, CPU cost, mode) when test n can be verified
    on only the queries that execute a changed line; None when it must run
    whole. mode "fe": every changed line it runs belongs to its front end
    (parse, optimize, plan, transpile), so comparing those artifacts with the
    pinned commit's decides it (tpcfp.py); mode "run": run those queries."""
    if n in CF_NUMS:
        return cf_pairs(n, plan)
    tid = SPLIT_NUMS.get(n)
    if tid is None or n in plan.cands.other:
        return None
    sm_all = split_maps()
    sm = sm_all.get("tests", {}).get(tid)
    if not sm:
        return None
    setup = sm_all.get("setup", {})
    fe = fe_maps().get(tid)
    fe_ok = fe is not None
    hits = changed_hits(n, plan)
    if hits is None:
        return None
    q = 0
    for rel, hit in hits.items():
        su = set(setup.get(rel, ()))
        lm = sm["lines"].get(rel, {})
        ex = fe["_exec"].get(rel, ()) if fe_ok else ()
        if fe_ok and any(l in ex for l in hit):
            fe_ok = False  # the executor, data loading or comparison side runs it
        for l in hit:
            if l in su:
                return None  # setUpClass runs it: every query depends on it
            m = lm.get(str(l))
            if m is None:
                return None  # the whole test runs it but no single query does
            q |= int(m, 16)
    nq = sm["nq"]
    if q == 0:
        return None
    idx = [i for i in range(nq) if q >> i & 1]
    if fe_ok and fe.get("nq") == nq and not (set(idx) & set(fe.get("unstable", ()))):
        cost = 0.3 + sum(fe["cpu"].get(str(i), 0.05) for i in idx)
        return sm["attr"], idx, nq, round(cost, 3), "fe"
    if q == (1 << nq) - 1:
        return None
    cost = sm["fixed"] + sum(sm["warm"][i] for i in idx) + max(sm["cold"][i] - sm["warm"][i] for i in idx)
    return sm["attr"], idx, nq, round(cost, 3), "run"


# Fixture-pair verification of the optimizer tests that check a rule against
# a fixture file (check_file, test_annotate_types, test_annotate_funcs).
# maps/cf.json.gz (cftrace.py) records which pairs run each line and which
# lines run outside the pairs; cfsel.py cuts a run down to chosen pairs.
CF_PREFIX = "tests/test_optimizer.py::TestOptimizer::"
CF_NUMS = {t["number"]: t["id"][len(CF_PREFIX):] for t in TESTS if t["id"].startswith(CF_PREFIX)}
if os.environ.get("BR_NOCF") == "1":
    CF_NUMS = {}
_CF = None


def cf_maps() -> dict:
    global _CF
    if _CF is None:
        try:
            with gzip.open(os.path.join(os.path.dirname(MAPS), "cf.json.gz"), "rt") as fh:
                _CF = json.load(fh)
        except Exception:  # noqa: BLE001
            _CF = {}
    return _CF


def cf_pairs(n: int, plan) -> tuple | None:
    """("@cf", pair indexes, pair count, CPU cost, "run") when test n can be
    verified on only the fixture pairs that execute a changed line (none,
    when only its own code outside the pairs does); None to run it whole."""
    m = CF_NUMS.get(n)
    if m is None or n in plan.cands.other:
        return None
    cm = cf_maps()
    t = cm.get("tests", {}).get(m)
    if not t:
        return None
    hits = changed_hits(n, plan)
    if not hits:
        return None
    setup = cm.get("setup", {})
    masks = t["masks"]
    q = 0
    for rel, hit in hits.items():
        su = set(setup.get(rel, ()))
        lm = t["lines"].get(rel, {})
        outside = set(t["outside"].get(rel, ()))
        for l in hit:
            if l in su:
                return None  # setUpClass runs it
            j = lm.get(str(l))
            if j is None:
                if l in outside:
                    continue  # the test's own code runs it, whichever pairs run
                return None  # the test runs it, but no traced run did
            q |= int(masks[j], 16)
    nq = t["nq"]
    if q == (1 << nq) - 1:
        return None
    idx = [i for i in range(nq) if q >> i & 1]
    pc, cx = t["pc"], t["cx"]
    est = t["fixed"] + (t["c1"] + sum(pc[i] for i in idx) + max(cx[i] for i in idx) if idx else 0.0)
    # in the calibrated units of the whole test's verification cost
    full = vcost(n)
    cost = full * min(1.0, est / t["full"]) if t["full"] > 0 else full
    return "@cf", idx, nq, round(cost, 4), "run"


def split_plan(plan, cands=None) -> dict:
    cands = plan.cands if cands is None else cands
    out = {}
    for n in list(SPLIT_NUMS) + sorted(n for n in cands if n in CF_NUMS):
        if n in cands and n not in cands.other:
            r = split_queries(n, plan)
            if r is not None:
                out[n] = r
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


def choose_verify(cands: dict, already_forked: bool, split: dict | None = None) -> tuple[list[int], list[int]]:
    """Split candidates into (verify, include) minimising expected cost.

    Verifying test i costs its CPU; it removes i from the schedule only if it
    passes (probability 1 - p_i). Expected schedule cost is approximated as
    P(all verified pass) * S(include) + P(some fail) * S(include + verified).
    """
    if not VERIFY:
        return [], sorted(cands)
    split = split or {}
    fixed = sorted(n for n in cands if n in NOVERIFY and n not in split)
    free = [n for n in cands if n not in NOVERIFY or n in split]
    if not free:
        return [], fixed
    ovh = 0.0 if already_forked else CHILD_OVH

    def vc(n):
        return split[n][3] if n in split else vcost(n)

    def sc(n):
        return 0.0 if n in split and split[n][4] == "fe" else setup_cost(n)

    def key(n):
        return ((vc(n) + sc(n)) / max(1e-3, 1.0 - cands[n]), n)

    free.sort(key=key)
    m = len(free)
    ks = sorted({0, m} | {int(m * f) for f in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)} | set(range(min(m, 40))))
    prefix = [0.0]
    pnone = [1.0]
    seen_cls: set = set()
    for n in free:
        extra = 0.0
        k_ = _cls(n)
        if sc(n) and k_ not in seen_cls:
            seen_cls.add(k_)
            extra = setup_cost(n)
        prefix.append(prefix[-1] + vc(n) + extra)
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
    choice = (free[:k], fixed + free[k:])
    # The query-looping TPC tests dominate a schedule they are in, but come
    # last in the cheap-first order above. Also price verifying them always,
    # plus a cheap-first prefix of the rest.
    B = [n for n in free if n in split and n in SPLIT_NUMS]
    if B:
        nb = [n for n in free if n not in B]
        m2 = len(nb)
        pre = [0.0]
        pn = [1.0]
        seen2: set = set()
        for n in nb:
            extra = 0.0
            k_ = _cls(n)
            if sc(n) and k_ not in seen2:
                seen2.add(k_)
                extra = setup_cost(n)
            pre.append(pre[-1] + vc(n) + extra)
            pn.append(pn[-1] * (1.0 - cands[n]))
        # verify any subset V of them; each fails on its own, and the schedule
        # then holds exactly the ones of V that failed (plus those not verified)
        subsets = [c for r in range(1, len(B) + 1) for c in itertools.combinations(B, r)]
        for k2 in sorted({0, 1, 2, 3, 5, 8, 12, 20, 30, m2 // 2, m2} & set(range(m2 + 1))):
            U = fixed + nb[k2:]
            P = nb[:k2]
            memo: dict = {}

            def s(extra: tuple, with_p: bool) -> float:
                if (extra, with_p) not in memo:
                    memo[extra, with_p] = sched_cost_ms(U + (P if with_p else []) + list(extra)) / 1000.0
                return memo[extra, with_p]

            for V in subsets:
                rest = tuple(n for n in B if n not in V)
                vv = sum(vc(n) for n in V)
                seen_v: set = set()
                for n in V:
                    if sc(n) and _cls(n) not in seen_v:
                        seen_v.add(_cls(n))
                        vv += setup_cost(n)
                expect = 0.0
                for mask in range(1 << len(V)):
                    pf = 1.0
                    failed = list(rest)
                    for i, n in enumerate(V):
                        if mask >> i & 1:
                            pf *= cands[n]
                            failed.append(n)
                        else:
                            pf *= 1.0 - cands[n]
                    key_ = tuple(sorted(failed))
                    expect += pf * (pn[k2] * s(key_, False) + ((1 - pn[k2]) * s(key_, True) if k2 else 0.0))
                cost = ovh + pre[k2] + vv + expect
                if cost < best[0] - 1e-9:
                    best = (cost, k2)
                    choice = (P + list(V), U + list(rest))
    return choice


# ------------------------------------------------------------------- analysis
class Cands(dict):
    """test number -> prior probability that it fails.

    `other` holds the tests added for any reason except executing a changed
    sqlglot line (rt=True); only those can be split into their queries."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        src = a[0] if a else None
        self.other = set(getattr(src, "other", ()))

    def add(self, nums, p: float, rt: bool = False) -> None:
        for n in nums:
            if not rt:
                self.other.add(n)
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
        self.split: dict = {}  # test -> (attr, queries, nq, cost) for query-level verification


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


_NOT_RUN = {"posts", "pdoc", "docs", "benchmarks", ".github"}  # top-level folders no test reads or imports


def _mentions(word: str, texts, exclude=(), runtime_only: bool = False) -> bool:
    """Whether word occurs as a whole word in any of texts or in any file of
    the pristine checkout (but the paths in exclude; with runtime_only, only
    files a test run can import or read: tests read files under sqlglot/ and
    tests/ only, and import nothing from docs or posts)."""
    import re

    pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(word) + r"(?![A-Za-z0-9_])")
    if any(t is not None and pat.search(t) for t in texts):
        return True
    w = word.encode()
    bpat = re.compile(pat.pattern.encode())
    skip = {os.path.join(PRISTINE, r) for r in exclude}
    for dp, dns, fns in os.walk(PRISTINE):
        dns[:] = [d for d in dns if d not in (".git", "__pycache__") and not (runtime_only and dp == PRISTINE and d in _NOT_RUN)]
        for fn in fns:
            if runtime_only and dp == PRISTINE and fn.endswith((".md", ".png")):
                continue
            if os.path.join(dp, fn) in skip:
                continue
            try:
                with open(os.path.join(dp, fn), "rb") as fh:
                    data = fh.read()
            except OSError:
                return True
            if w in data and bpat.search(data):
                return True
    return False


_INTROSPECTED = None


def introspected(rel: str) -> bool:
    """Whether rel lies in a folder that a test lists and whose test file reads
    code objects (test_typing compares the annotators of the sqlglot/typing
    modules by co_code, co_names and co_varnames): there, renaming a name
    can change a test's outcome."""
    global _INTROSPECTED
    if _INTROSPECTED is None:
        _INTROSPECTED = set()
        for d, sid in DIRREADS.items():
            for f in {FILE_OF[n] for n in SETS[sid] if n in FILE_OF}:
                src = read_pristine(f) or ""
                if "__code__" in src or "co_code" in src or "co_names" in src or "co_varnames" in src:
                    _INTROSPECTED.add(d)
    d = os.path.dirname(rel)
    while True:
        if d in _INTROSPECTED:
            return True
        if not d:
            return False
        d = os.path.dirname(d)


def patched_texts(fps) -> dict:
    """New text of every file the patch leaves (None when unreadable)."""
    out = {}
    for fp in fps:
        if fp.is_binary or fp.is_deleted or not fp.new_path:
            continue
        try:
            old = read_pristine(fp.old_path) if fp.old_path and not fp.is_new else None
            out[fp.new_path] = patchapply.apply_to_text(old, fp)
        except Exception:  # noqa: BLE001
            out[fp.new_path] = None
    return out


def unreferenced_new_modules(fps) -> set:
    """New sqlglot modules nothing can import: Python imports a module only
    by its name, and every dynamic import in sqlglot and its tests builds that
    name from text that appears somewhere (a dialect string, an attribute
    name, a fixed table), or lists a folder (the tests that list folders are
    known). So a new module whose name appears in no other file, old or
    patched, and that no test can see by listing a folder, never runs."""
    new = [fp for fp in fps if fp.is_new and not fp.is_binary and (fp.new_path or "").startswith("sqlglot/")
           and fp.new_path.endswith(".py")]
    if not new:
        return set()
    texts = {}
    for fp in fps:
        if fp.is_binary or fp.is_deleted or not fp.new_path:
            continue
        try:
            old = read_pristine(fp.old_path) if fp.old_path and not fp.is_new else None
            texts[fp.new_path] = patchapply.apply_to_text(old, fp)
        except Exception:  # noqa: BLE001
            return set()
    out = set()
    for fp in new:
        rel = fp.new_path
        parts = rel[:-3].split("/")
        name = parts[-2] if parts[-1] == "__init__" else parts[-1]
        d, listed = os.path.dirname(rel), False
        while True:
            listed = listed or d in DIRREADS
            if not d:
                break
            d = os.path.dirname(d)
        b = parts[-1]
        if (listed or not name.isidentifier() or not os.path.isdir(os.path.join(PRISTINE, os.path.dirname(rel)))
                or b.startswith("test") or b.endswith("_test") or b == "conftest"):  # pytest may collect those
            continue
        if not _mentions(name, [t for r, t in texts.items() if r != rel]):
            out.add(rel)
    return out


def analyze(cid: str, text: str) -> Plan:
    plan = Plan(cid)
    try:
        fps = patchapply.parse(text)
    except Exception as e:  # noqa: BLE001
        plan.all_tests = True
        plan.notes.append(f"unparseable patch {e}")
        return plan
    skip_new = unreferenced_new_modules(fps)
    for fp in fps:
        old_rel = fp.old_path
        new_rel = fp.new_path
        if new_rel in skip_new:
            plan.notes.append(f"new module nothing imports {new_rel}")
            continue
        rels = {r for r in (old_rel, new_rel) if r}
        if fp.is_binary:
            for r in rels:
                plan.cands.add(tests_reading(r, True), 1.0)  # not applied in the farm: never verify
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
        if any(test_side(r, fp.is_new) for r in rels) and astdiff.side_effects(old) != astdiff.side_effects(new):
            # pytest imports it while collecting, before any test runs
            plan.all_tests = True
            plan.notes.append(f"test module side effect {path}")
        listing = fp.is_new or fp.is_deleted or (old_rel != new_rel)
        for r in rels:  # tests that open the file itself (source readers too)
            plan.cands.add(tests_reading(r, listing), P_CODE if r.endswith(".py") else P_DATA)
        if path.endswith(".py") and (path.startswith("sqlglot/") or path.startswith("tests/")):
            oc, nc = fp.changed_lines() if (old is not None and new is not None and not fp.is_new) else (None, None)
            ch = astdiff.analyze(old, new, oc, nc, renames=not introspected(path))
            if ch.global_renames and not fp.is_new and old_rel == new_rel and path.startswith("sqlglot/"):
                texts = [t for r, t in patched_texts(fps).items() if r != path]
                if any(_mentions(w, texts, (path,), True) for pair in ch.global_renames for w in pair):
                    ch = astdiff.analyze(old, new, oc, nc, renames=False)
                else:
                    plan.notes.append(f"renamed {ch.global_renames} in {path}")
            elif ch.global_renames:
                ch = astdiff.analyze(old, new, oc, nc, renames=False)
            if old_rel != new_rel and old_rel and new_rel:
                ch.semantic = True
                ch.import_time = True
            plan.py_changes[path] = ch
            if not ch.semantic:
                continue
            if path.startswith("sqlglot/"):
                sqlglot_change(plan, old_rel or path, ch, old)
            else:
                test_change(plan, old_rel or path, ch, old)
        elif path.endswith(".py") and any(r in COV for r in rels):
            # code run by tests through runpy/subprocess paths the farm cannot
            # reproduce (it resolves symlinks back to pristine): include, never verify
            for r in rels:
                fm = COV.get(r) or {}
                for ln, sid_ in fm.items():
                    plan.cands.add(SETS[sid_], 1.0)
        else:
            # data file (already added above as a read file)
            for r in rels:
                if r.endswith(".py") and r.startswith("sqlglot/"):
                    plan.need_snapshot = True
    # The spawners (test_lazy_load) run `import sqlglot` in a fresh
    # interpreter and check what it registered. Only code that runs while
    # sqlglot imports can change that, and every such change needs the
    # import-time snapshot (module level, import statements, functions run
    # at import time, syntax errors, deleted, new or renamed modules).
    if plan.need_snapshot and any(r.startswith("sqlglot/") for r in plan.files):
        plan.cands.add(SPAWNERS, P_LOW)
    return plan


def test_side(rel: str, is_new: bool) -> bool:
    """A Python file under tests/ that pytest imports while collecting: a test
    module, a package __init__, a module some test file imports, or a new
    file (another changed file may import it)."""
    if not (rel.startswith("tests/") and rel.endswith(".py")):
        return False
    b = os.path.basename(rel)
    return (is_new or rel in BY_FILE or bool(TEST_IMPORTERS.get(rel)) or b == "__init__.py"
            or b.startswith("test_") or b.endswith("_test.py"))


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
        # (and so does every enclosing folder's, when it creates or empties a subfolder)
        d = os.path.dirname(rel)
        while True:
            sid = DIRREADS.get(d)
            if sid is not None:
                out.update(SETS[sid])
            if not d:
                break
            d = os.path.dirname(d)
    return out


_EXEC_CACHE: dict = {}


def _pristine_code(rel: str, src: str | None):
    """The module code the image compiled for the pinned file (its
    __pycache__ entry), when src is exactly that file's text and the entry
    is current: the same code compile() would build, without compiling."""
    try:
        import importlib.util
        import marshal

        if not src or read_pristine(rel) != src:
            return None
        path = os.path.join(PRISTINE, rel)
        with open(importlib.util.cache_from_source(path), "rb") as fh:
            data = fh.read()
        if data[:4] != importlib.util.MAGIC_NUMBER or int.from_bytes(data[4:8], "little") != 0:
            return None
        st = os.stat(path)
        if int.from_bytes(data[8:12], "little") != int(st.st_mtime) & 0xFFFFFFFF or int.from_bytes(data[12:16], "little") != st.st_size & 0xFFFFFFFF:
            return None
        return marshal.loads(data[16:])
    except Exception:  # noqa: BLE001
        return None


def exec_lines(rel: str, src: str | None) -> set:
    """Lines of the old source that carry bytecode (can fire a line event)."""
    v = _EXEC_CACHE.get(rel)
    if v is None:
        v = set()
        try:
            co = _pristine_code(rel, src)
            stack = [co if co is not None else compile(src or "", rel, "exec", dont_inherit=True)]
            while stack:
                co = stack.pop()
                for _s, _e, ln in co.co_lines():
                    if ln is not None:
                        v.add(ln)
                stack.extend(c for c in co.co_consts if hasattr(c, "co_lines"))
        except Exception:  # noqa: BLE001
            v = None
        _EXEC_CACHE[rel] = v
    return v


_WM = None


def wm_maps() -> dict:
    global _WM
    if _WM is None:
        try:
            with gzip.open(os.path.join(os.path.dirname(MAPS), "wm.json.gz"), "rt") as fh:
                _WM = json.load(fh)
            nums = [NUM_BY_ID.get(i) for i in _WM["tests"]]
            _WM["_nums"] = nums
            _WM["_incomplete"] = {nums[i] for i in _WM["incomplete"] if nums[i] is not None}
            # tests of this run the profile never saw count as incomplete
            _WM["_incomplete"] |= set(ALL) - {n for n in nums if n is not None}
        except Exception:  # noqa: BLE001
            _WM = {}
    return _WM


def _cov_only(rel: str, lines) -> set:
    out: set = set()
    fm = COV.get(rel) or {}
    for ln in lines:
        sid = fm.get(ln)
        if sid is not None:
            out.update(SETS[sid])
    return out


_BW = None


def bw_maps() -> dict:
    global _BW
    if _BW is None:
        try:
            with gzip.open(os.path.join(os.path.dirname(MAPS), "bw.json.gz"), "rt") as fh:
                _BW = json.load(fh)
            nums = [NUM_BY_ID.get(i) for i in _BW["tests"]]
            _BW["_nums"] = nums
            _BW["_incomplete"] = {nums[i] for i in _BW["incomplete"] if nums[i] is not None}
            _BW["_incomplete"] |= set(ALL) - {n for n in nums if n is not None}
        except Exception:  # noqa: BLE001
            _BW = {}
    return _BW


def bw_tests(rel: str, key: str, i: int, op: str, lines) -> set | None:
    """Tests in which swapping the i-th operator of this and/or chain ever
    changes its value (or what it evaluates), per bwtrace.py. None: no
    profile for the chain."""
    W = bw_maps()
    if not W:
        return None
    site = W["sites"].get(rel, {}).get(key)
    if site is None or site[0] != op or not 0 <= i < site[1] - 1:
        return None
    for d in W["modules"].values():  # the swap shows while a test module imports
        if i in d.get(rel, {}).get(key, ()):
            return None
    out: set = set()
    sid = site[3].get(str(i))
    if sid is not None:
        nums = W["_nums"]
        out.update(nums[j] for j in W["sets"][sid] if nums[j] is not None)
    out |= _cov_only(rel, lines) & W["_incomplete"]
    return out


WM_FAMILY_EQ = ("Is", "IsNot", "Eq", "NotEq")  # as in wmtrace.py
WM_FAMILY_ORD = ("Lt", "LtE", "Gt", "GtE", "Eq", "NotEq")


def wm_tests(rel: str, key: str, old_op: str, new_op: str, lines) -> set | None:
    """Tests in which swapping this comparison's operator for new_op changes
    a result at least once (wmtrace.py). None: no profile for the site."""
    W = wm_maps()
    if not W:
        return None
    site = W["sites"].get(rel, {}).get(key)
    if site is None or site[0] != old_op:
        return None
    # wmtrace.py only evaluates alternatives within the operator's family; a
    # swap to anything else (== -> <=, != -> in, ...) was never measured
    fam = WM_FAMILY_ORD if old_op in ("Lt", "LtE", "Gt", "GtE") else WM_FAMILY_EQ
    if old_op not in WM_FAMILY_ORD + WM_FAMILY_EQ or new_op not in fam or new_op == old_op:
        return None
    for d in W["modules"].values():  # the swap shows while a test module imports
        if new_op in d.get(rel, {}).get(key, ()):
            return None
    out: set = set()
    sid = site[1].get(new_op)
    if sid is not None:
        nums = W["_nums"]
        out.update(nums[i] for i in W["sets"][sid] if nums[i] is not None)
    out |= _cov_only(rel, lines) & W["_incomplete"]
    return out


def runtime_tests(rel: str, ch: astdiff.PyChange, old_src) -> set:
    """Tests that execute a runtime difference. When no line of a difference
    can fire a line event (a bare `if (` line, `nonlocal`, ...), blame its
    whole function instead. An operator swap (a comparison's, or one `and`/`or`
    of a chain) counts only for the tests in which it would give a different
    result."""
    swaps = getattr(ch, "swaps", None) or []
    bswaps = getattr(ch, "bswaps", None) or []
    swap_groups = set()
    out: set = set()
    for key, old_op, new_op, lines, gi in swaps:
        t = wm_tests(rel, key, old_op, new_op, lines)
        if t is None:
            out |= cov_tests(rel, lines)
        else:
            out |= t
            swap_groups.add(gi)
    for key, i, op, lines, gi in bswaps:
        t = bw_tests(rel, key, i, op, lines)
        if t is None:
            out |= cov_tests(rel, lines)
        else:
            out |= t
            swap_groups.add(gi)
    out |= cov_tests(rel, ch.plain_lines if swaps or bswaps else ch.runtime_lines)
    fm = COV.get(rel) or {}
    for gi, (lines, first, last) in enumerate(ch.runtime_groups):
        if gi in swap_groups:
            continue
        if any(l in fm for l in lines):
            continue
        exe = exec_lines(rel, old_src)
        if exe is not None and any(l > 0 and l in exe for l in lines):
            continue  # executable, and no test ever ran it
        t = cov_tests(rel, [-first])
        if not t:
            t = func_span_tests(rel, first, last)
        out |= t
    return out


def sqlglot_change(plan: Plan, rel: str, ch: astdiff.PyChange, old_src=None) -> None:
    if ch.syntax_error:
        plan.need_snapshot = True  # the import error decides
        return
    if ch.runtime_lines:
        plan.cands.add(runtime_tests(rel, ch, old_src), P_CODE, rt=True)
        # functions that also run while sqlglot is imported change import-time state
        imp = set(IMPORT_LINES.get(rel, ()))
        if any(l in imp for l in ch.runtime_lines if l > 0):
            plan.need_snapshot = True
    for q, first, last in ch.def_changes:
        plan.cands.add(func_span_tests(rel, first, last), P_CODE, rt=True)
    plan.cands.add(reg_tests(rel, ch), 1.0)  # kept unverified
    if ch.import_time:
        plan.need_snapshot = True


def test_change(plan: Plan, rel: str, ch: astdiff.PyChange, old_src=None) -> None:
    if rel.endswith("__init__.py"):
        d = os.path.dirname(rel) + "/"
        plan.cands.add([n for f, ns in BY_FILE.items() if f.startswith(d) for n in ns], P_TEST)
    if ch.syntax_error or ch.import_time:
        plan.cands.add(BY_FILE.get(rel, []), P_TEST)
        for imp in TEST_IMPORTERS.get(rel, []):
            plan.cands.add(BY_FILE.get(imp, []), P_TEST)
        if rel not in BY_FILE and not TEST_IMPORTERS.get(rel):
            # a helper module no test file is known to import: be safe
            plan.cands.add(tests_reading(rel), P_TEST)
    if ch.runtime_lines:
        plan.cands.add(runtime_tests(rel, ch, old_src), P_TEST)
        for q, first, last in ch.def_changes:
            plan.cands.add(func_span_tests(rel, first, last), P_TEST)
    plan.cands.add(reg_tests(rel, ch), 1.0)  # kept unverified


# ---------------------------------------------------- snapshot-diff candidates
_SNAP = None


def pristine_snapshot():
    global _SNAP
    if _SNAP is None:
        with open(SNAP) as fh:
            _SNAP = json.load(fh)
    return _SNAP


def class_subtree(pristine: dict, patched: dict):
    """k -> k and its subclasses, each under every name a trace may have used
    for it: the snapshot key and module:qualname (they differ for classes a
    metaclass builds on the fly, and base lists use module:qualname)."""
    alias: dict = {}
    for snap in (pristine, patched):
        alias.update((snap or {}).get("alias") or {})
    children: dict[str, set] = {}
    for snap in (pristine, patched):
        for k, mro in ((snap or {}).get("bases") or {}).items():
            for b in mro:
                children.setdefault(b, set()).add(k)

    def subtree(k: str) -> set:
        names = {k}
        if k in alias:
            names.add(alias[k])
        out = set(names)
        for nm in names:
            out |= children.get(nm, set())
        out |= {alias[x] for x in list(out) if x in alias}
        return out

    return subtree


SNAPDK = os.environ.get("BR_SNAPDK", os.path.splitext(SNAP)[0] + "dk.json")
_DK: dict | None = None


def pristine_dk() -> dict:
    global _DK
    if _DK is None:
        try:
            with open(SNAPDK) as fh:
                _DK = json.load(fh)
        except (OSError, ValueError):
            _DK = {}
    return _DK


def dict_key_tests(key: str, attr: str, live: dict) -> set | None:
    """Tests a change to the dict held in class attribute key.attr can reach:
    those that looked up a key whose entry changed (or the size, if it
    changed), and those that read the dict as a whole. None when that cannot
    be decided by key."""
    import fingerprint

    tag = f"{key}|{attr}"
    old = pristine_dk().get(tag)
    c = live.get(key)
    if tag not in KLTAGS or old is None or c is None:
        return None
    new = fingerprint.key_fps(c.__dict__.get(attr))
    if new is None:
        return None
    changed = [k for k in old.keys() | new.keys() if old.get(k) != new.get(k)]
    if not changed:
        return None  # a difference the entries do not show (order ...)
    out: set = set()
    sid = KLA.get(tag)
    if sid is not None:
        out.update(SETS[sid])
    d = KL.get(tag, {})
    for k in changed:
        sid = d.get(k)
        if sid is not None:
            out.update(SETS[sid])
    return out


def snapshot_candidates(pristine: dict, patched: dict, diff: dict, plan: Plan) -> tuple[set, bool]:
    """Tests affected by changed import-time state. Returns (tests, everything)."""
    out: set = set()
    new_err = {m for m, _ in diff["errors"] if m in patched["errors"]} if diff["errors"] else set()
    if new_err and ("sqlglot" in new_err or not PRECISE_ERR):
        return set(ALL), True
    subtree = class_subtree(pristine, patched)
    if new_err:
        # modules that no longer import, and every submodule of a package that
        # does not: like deleted modules, whatever uses them fails (the root
        # package still imports, or snapshot() itself raised and the child
        # falls back to everything)
        plan.import_errors = set(new_err)
        gone = set(new_err) | {m for m in pristine["globals"] if any(m.startswith(f + ".") for f in new_err)}
        for m in gone:
            out |= global_ref_tests(m, "*")
            out |= import_site_tests(m, True)
            # names a package loads from it lazily (sqlglot.dialects.Snowflake):
            # every attribute access of them
            for pkg, names in LAZY_ATTRS.items():
                for n, (tm, ta) in names.items():
                    if tm == m:
                        out |= global_ref_tests(m, ta)
                        for f, ln in NAMES.refs(pkg, n):
                            out |= cov_tests(f, [ln])
                        for k in (f"{pkg}|{n}", f"{pkg}|__dict__"):
                            sid = MODATTR.get(k)
                            if sid is not None:
                                out.update(SETS[sid])
            # tests that list its folder may import every module in it
            sid = DIRREADS.get(os.path.dirname(_mod_rel(m)))
            if sid is not None:
                out.update(SETS[sid])
            # every test that read anything through one of its classes
            for key in pristine["classes"]:
                if key.partition(":")[0] == m:
                    for x in subtree(key):
                        sid = CLASSUSE.get(x)
                        if sid is not None:
                            out.update(SETS[sid])
        # and measured, not inferred: import every test module from the patched
        # tree, as pytest's collection does before any test runs. A test file
        # that no longer imports fails whole, whatever chain broke it.
        import importlib

        for tf in sorted(BY_FILE):
            try:
                importlib.import_module(tf[:-3].replace("/", "."))
            except _Timeout:
                raise
            except BaseException:  # noqa: BLE001
                out.update(BY_FILE[tf])

    live = None
    flows: list = []
    dd = default_defs(plan)
    for key, attr in diff["attrs"]:
        if attr == "__doc__":
            continue  # docstrings: nothing reads them at runtime
        if dd:
            if live is None:
                import fingerprint

                live = dict(fingerprint.iter_classes(sorted(sys.modules), "sqlglot"))
            c = live.get(key)
            if c is not None and default_only_value(c.__dict__.get(attr), dd):
                # a method whose header changed only in constant defaults:
                # only a call can see that, every call runs its body (the
                # parent charged those tests from coverage), and nothing reads
                # defaults by introspection; readers through the class family
                # are added all the same
                sub = subtree(key)
                for x in LK_BY_ATTR.get(attr, ()):
                    if x in sub:
                        out.update(SETS[LK[f"{x}|{attr}"]])
                continue
        if fresh_member(key, attr, plan, pristine):
            continue  # a new member under a name nothing mentions
        t = dispatch_member_tests(key, attr, plan, pristine, subtree(key))
        if t is not None:
            out |= t  # a new X_sql: tests that generated an X node
            continue
        if f"{key}|{attr}" in KLTAGS:
            if live is None:
                import fingerprint

                live = dict(fingerprint.iter_classes(sorted(sys.modules), "sqlglot"))
            t = dict_key_tests(key, attr, live)
            if t is not None:
                out |= t
                continue
        sub = subtree(key)
        if attr == "*":
            for x in sub:
                sid = CLASSUSE.get(x)
                if sid is not None:
                    out.update(SETS[sid])
            continue
        base = attr_reader_tests(key, attr, sub, pristine, patched)
        if live is None:
            import fingerprint

            live = dict(fingerprint.iter_classes(sorted(sys.modules), "sqlglot"))
        t = core_flow_tests(key, attr, plan, live)
        if t is not None:
            # a tokenizer setting: the tests that ran the core's reads of it,
            # or those that read it through the class (both cover it)
            flows.append((t, base))
            continue
        out |= base
    if flows:
        out |= cheapest_cover(out, flows)
    for mod, name in diff["globals"]:
        if name == "__doc__":
            continue
        if dd and mod in sys.modules and default_only_value(vars(sys.modules[mod]).get(name), dd):
            continue  # a function (or a name bound to it) whose defaults alone changed: as above
        if mod == "sqlglot.expressions" and name == "EXPR_CLASSES" and expr_registry_grew(pristine, diff, plan):
            sid = MODATTR.get(f"{mod}|__dict__")
            if sid is not None:
                out.update(SETS[sid])
            continue  # new classes no generator can dispatch to
        out |= global_ref_tests(mod, name)
        if name == "*":  # the module is gone (or new)
            out |= import_site_tests(mod, True)
        else:
            gone = name in ((pristine.get("globals") or {}).get(mod) or {}) and name not in ((patched.get("globals") or {}).get(mod) or {})
            out |= import_site_tests(f"{mod}|{name}", gone)
            out |= import_site_tests(f"{mod}.{name}", gone)
    # import-time edits the fingerprints could not see: fall back to everything
    # that refers to the names those statements bind
    seen_mods = {k.partition(":")[0] for k, _ in diff["attrs"]} | {m for m, _ in diff["globals"]}
    for rel, ch in plan.py_changes.items():
        if not (rel.startswith("sqlglot/") and ch.semantic and ch.import_time) or ch.syntax_error:
            continue
        mod = module_of(rel)
        if seen_mods & importer_closure({mod}):
            continue
        if ch.other_import:
            # tests chosen by name, not by a measured difference (a def header
            # alone is not: what runs it is known from coverage)
            plan.unseen = True
        for name in ch.toplevel_names:
            out |= global_ref_tests(mod, name)
        for q in ch.class_quals:
            key = f"{mod}:{q}"
            sub = subtree(key)
            names = ch.toplevel_names or set()
            hit = False
            for name in names:
                for x in LK_BY_ATTR.get(name, ()):
                    if x in sub:
                        out.update(SETS[LK[f"{x}|{name}"]])
                        hit = True
                meth = method_lines(key, name, pristine, patched)
                if meth is not None:
                    out |= meth
                    hit = True
            if not hit:
                out |= family_tests(key, sub)
    return out, False


def default_defs(plan) -> set:
    """(module, qualname) of the undecorated sqlglot defs whose headers changed
    only in the values of constant defaults (astdiff), outside folders whose
    tests read code objects."""
    out = set()
    for rel, ch in plan.py_changes.items():
        if rel.startswith("sqlglot/") and rel.endswith(".py") and not introspected(rel):
            for q in getattr(ch, "default_defs", None) or ():
                out.add((module_of(rel), q))
    return out


def default_only_value(v, dd: set) -> bool:
    import types

    return isinstance(v, types.FunctionType) and (getattr(v, "__module__", None), getattr(v, "__qualname__", None)) in dd


_LIVE: dict = {}


def _mention_sites(word: str) -> list:
    """(path, line number, line) of every whole-word occurrence of word in the
    files of the pristine checkout a test run can import or read."""
    import re

    pat = re.compile(rb"(?<![A-Za-z0-9_])" + re.escape(word.encode()) + rb"(?![A-Za-z0-9_])")
    out = []
    for dp, dns, fns in os.walk(PRISTINE):
        dns[:] = [d for d in dns if d not in (".git", "__pycache__") and not (dp == PRISTINE and d in _NOT_RUN)]
        for fn in fns:
            if dp == PRISTINE and fn.endswith((".md", ".png")):
                continue
            path = os.path.join(dp, fn)
            try:
                with open(path, "rb") as fh:
                    data = fh.read()
            except OSError:
                return [(None, 0, b"")]
            if word.encode() not in data:
                continue
            for i, ln in enumerate(data.split(b"\n"), 1):
                if pat.search(ln):
                    out.append((os.path.relpath(path, PRISTINE), i, ln))
    return out


def expr_registry_grew(pristine: dict, diff: dict, plan: Plan) -> bool:
    """Whether the change to sqlglot.expressions.EXPR_CLASSES (expression key
    -> class) is only new keys of new classes, which no test can feel through
    it. Its one reader is generator._build_dispatch; new keys only add
    dispatch entries keyed by the new classes, consulted only to generate a
    node of a new class. A test gets such a node only through another changed
    lookup (the parser's FUNCTIONS, FUNCTION_BY_NAME, ALL_FUNCTIONS, the new
    module names, changed code), each charged on its own. Old classes keep
    their keys, and none is displaced by a new class with the same key."""
    if "sqlglot/generator.py" in plan.files or "sqlglot/expressions/__init__.py" in plan.files:
        return False
    if any(a == "key" for _, a in diff["attrs"]):
        return False
    span = M["defs"].get("sqlglot/generator.py", {}).get("_build_dispatch")
    if not span:
        return False
    for rel, ln, text in _mention_sites("EXPR_CLASSES"):
        if rel == "sqlglot/generator.py" and span[0] <= ln <= span[1]:
            continue
        if rel == "sqlglot/expressions/__init__.py" and text.startswith(b"EXPR_CLASSES"):
            continue
        return False
    if any(t and "EXPR_CLASSES" in t for t in plan.files.values()):
        return False
    try:
        from sqlglot import exp
    except Exception:  # noqa: BLE001
        return False
    if not _LIVE:
        import fingerprint

        _LIVE.update(dict(fingerprint.iter_classes(sorted(sys.modules), "sqlglot")))
    pcl = pristine.get("classes") or {}
    rev = {id(v): k for k, v in _LIVE.items()}
    reg = getattr(exp, "EXPR_CLASSES", None)
    if not isinstance(reg, dict):
        return False
    new = {}
    for k, c in reg.items():
        fk = rev.get(id(c))
        if fk is None:
            return False
        if fk not in pcl:
            new[k] = c
    if not new:
        return False
    try:
        for fk, c in _LIVE.items():
            if fk in pcl and isinstance(c, type) and issubclass(c, exp.Expr) and getattr(c, "key", None) in new:
                return False  # an old class's key now maps to a new class
        for fk in pcl:
            if fk.startswith("sqlglot.expressions") and fk not in _LIVE:
                return False  # a class is gone: its key may be too
    except Exception:  # noqa: BLE001
        return False
    return True


# ------------------------------------------------- tokenizer settings flow
# Tokenizer._init_core hands every tokenizer setting to a new TokenizerCore
# each time a tokenizer is built, so every test that tokenizes reads every
# setting through the class, and the attribute-read trace cannot tell which
# tests use which. What a setting does is decided where the core reads the
# field it was handed (`self.byte_string_escapes` in a byte-string branch,
# say). A setting whose only by-name reads are that hand-off, class bodies
# and class-creation hooks (whose results land on classes, which the
# snapshot compares) is charged to the tests that ran the core's reads of
# its field, the hook lines they ran, and the test files that name it.
_TOK_REL = "sqlglot/tokens.py"
_CORE_REL = "sqlglot/tokenizer_core.py"
_MENTIONS: dict = {}
_FLOW: dict = {}


def _mentions_cached(word: str) -> list:
    if word not in _MENTIONS:
        _MENTIONS[word] = _mention_sites(word)
    return _MENTIONS[word]


def _flow_kinds(src: str, word: str, lines) -> dict | None:
    """line -> kind for each given line of src that mentions word:
    ("transfer", field) for the keyword argument field=self.word (or
    bool(self.word)) of the TokenizerCore(...) call in Tokenizer._init_core,
    "hook" in a class-creation hook (__init_subclass__, a metaclass's
    __new__/__init__), "class" in a class body, "comment". None when any
    line is anything else (code that runs at test time, module level, a def
    header, a class created inside a function)."""
    import ast

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    text = src.split("\n")
    spans: list = []

    def walk(body, chain):
        for st in body:
            spans.append((st.lineno, st.end_lineno or st.lineno, st, chain))
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                walk(st.body, chain + (st,))
                continue
            for f in ("body", "orelse", "finalbody"):
                walk(getattr(st, f, None) or [], chain)
            for h in getattr(st, "handlers", None) or []:
                walk(h.body, chain)
            for c in getattr(st, "cases", None) or []:
                walk(c.body, chain)

    walk(tree.body, ())
    FN = (ast.FunctionDef, ast.AsyncFunctionDef)
    out: dict = {}
    for ln in lines:
        if 0 < ln <= len(text) and text[ln - 1].strip().startswith("#"):
            out[ln] = "comment"
            continue
        inner = None
        for first, last, st, chain in spans:
            if first <= ln <= last and (inner is None or (first >= inner[0] and last <= inner[1])):
                inner = (first, last, st, chain)
        if inner is None:
            return None
        _f, _l, st, chain = inner
        if isinstance(st, (*FN, ast.ClassDef)):
            return None  # a header: decorators, defaults, bases
        if not chain:
            return None  # module level
        # the statement's own lines (a compound statement: its header)
        hdr = st.end_lineno or st.lineno
        if isinstance(st, (ast.If, ast.While)):
            hdr = st.test.end_lineno or st.lineno
        elif isinstance(st, (ast.For, ast.AsyncFor)):
            hdr = max(st.iter.end_lineno or st.lineno, st.target.end_lineno or st.lineno)
        elif isinstance(st, (ast.With, ast.AsyncWith)):
            hdr = max((it.context_expr.end_lineno or st.lineno) for it in st.items)
        elif isinstance(st, ast.Match):
            hdr = st.subject.end_lineno or st.lineno
        elif isinstance(st, (ast.Try, getattr(ast, "TryStar", ast.Try))):
            hdr = st.lineno
        read = ("read", st.lineno, hdr)
        if isinstance(chain[-1], ast.ClassDef):
            if any(isinstance(c, FN) for c in chain):
                out[ln] = read  # a class body run when its function runs
                continue
            out[ln] = "class"
            continue
        fn = chain[-1]
        owner = chain[-2] if len(chain) >= 2 else None
        if not isinstance(owner, ast.ClassDef) or any(isinstance(c, FN) for c in chain[:-1]):
            out[ln] = read  # code that runs when its function is called
            continue
        if fn.name == "__init_subclass__" or (
            fn.name in ("__new__", "__init__") and any(isinstance(b, ast.Name) and b.id == "type" for b in owner.bases)
        ):
            out[ln] = "hook"
            continue
        if fn.name == "_init_core" and owner.name == "Tokenizer" and len(chain) == 2:
            field = None
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "TokenizerCore":
                    for kw in node.keywords:
                        v = kw.value
                        if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "bool" and len(v.args) == 1 and not v.keywords:
                            v = v.args[0]
                        if (
                            kw.arg
                            and isinstance(v, ast.Attribute)
                            and isinstance(v.value, ast.Name)
                            and v.value.id == "self"
                            and v.attr == word
                            and kw.value.lineno <= ln <= (kw.value.end_lineno or kw.value.lineno)
                        ):
                            field = kw.arg
            if field:
                out[ln] = ("transfer", field)
                continue
        out[ln] = read  # a method that runs at test time
    return out


def _core_field_uses(src: str, field: str) -> set | None:
    """Lines of TokenizerCore that load self.<field>; None when the field is
    written any other way than one plain store `self.field = field` in
    __init__, read off anything but self, or named as a string outside
    __slots__. (Locals, parameters and keyword arguments that share its name
    hold only what such a load gave them.)"""
    import ast

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    cores = [c for c in tree.body if isinstance(c, ast.ClassDef) and c.name == "TokenizerCore"]
    if len(cores) != 1:
        return None
    core = cores[0]
    uses: set = set()
    stores = 0

    def visit(node, where):
        nonlocal stores
        if isinstance(node, ast.Attribute) and node.attr == field:
            if not (where in ("method", "init", "nested") and isinstance(node.value, ast.Name) and node.value.id == "self"):
                return False
            if isinstance(node.ctx, ast.Load):
                uses.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
            elif not (isinstance(node.ctx, ast.Store) and where == "init"):
                return False
        elif isinstance(node, ast.Constant) and node.value == field:
            if where != "slots":
                return False
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == field:
            return False
        elif isinstance(node, ast.alias) and field in (node.name, node.asname):
            return False
        elif isinstance(node, (ast.Global, ast.Nonlocal)) and field in node.names:
            return False
        if isinstance(node, (ast.AugAssign, ast.AnnAssign)) and isinstance(node.target, ast.Attribute) and node.target.attr == field:
            return False
        if isinstance(node, ast.Assign) and where == "init":
            t = node.targets
            if any(isinstance(x, ast.Attribute) and x.attr == field for x in t):
                if not (
                    len(t) == 1
                    and isinstance(t[0], ast.Attribute)
                    and isinstance(t[0].value, ast.Name)
                    and t[0].value.id == "self"
                    and isinstance(node.value, ast.Name)
                    and node.value.id == field
                ):
                    return False
                stores += 1
        for ch in ast.iter_child_nodes(node):
            w = where
            if node is core:
                if isinstance(ch, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "__slots__" for x in ch.targets):
                    w = "slots"
                elif isinstance(ch, ast.FunctionDef):
                    w = "init" if ch.name == "__init__" else "method"
                else:
                    w = "core"
            elif isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)) and where in ("init", "method"):
                w = "nested"  # a closure or nested class could keep the value
            if visit(ch, w) is False:
                return False
        return True

    if visit(tree, "module") is False or stores != 1:
        return None
    return uses


def _reads_member(src, word: str) -> bool:
    """Whether src may read a member called word: an attribute access
    `.word`, or the string "word" (getattr, __slots__ walks). True when it
    cannot be parsed."""
    import ast

    if src is None:
        return True
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == word:
            return True
        if isinstance(node, ast.Constant) and node.value == word:
            return True
    return False


def core_flow_tests(key: str, attr: str, plan: Plan, live: dict) -> set | None:
    """Tests a change to the tokenizer setting key.attr can reach, by where
    its value flows (see above); None when the rule does not apply."""
    try:
        tok = sys.modules.get("sqlglot.tokens")
        c = live.get(key)
        if tok is None or c is None or not (isinstance(c, type) and issubclass(c, tok.Tokenizer)):
            return None
        if attr.startswith("__") or _CORE_REL in plan.files:
            return None
        ck = (plan.cid, attr)
        if ck in _FLOW:
            return _FLOW[ck]
        _FLOW[ck] = None
        out: set = set()
        fields: set = set()
        hook_lines: dict = {}
        sites = _mentions_cached(attr)
        by_rel: dict = {}
        for rel, ln, _t in sites:
            if rel is None:
                return None
            by_rel.setdefault(rel, []).append(ln)
        for rel, lines in by_rel.items():
            if rel.startswith("tests/"):
                out.update(BY_FILE.get(rel, ()))
                for imp in TEST_IMPORTERS.get(rel, []):
                    out.update(BY_FILE.get(imp, ()))
                continue
            if not (rel.startswith("sqlglot/") and rel.endswith(".py")):
                return None
            src = read_pristine(rel)
            kinds = _flow_kinds(src, attr, lines) if src is not None else None
            if kinds is None:
                return None
            imp_rel = set(IMPORT_LINES.get(rel, ()))
            for ln, k in kinds.items():
                if isinstance(k, tuple) and k[0] == "transfer":
                    if rel != _TOK_REL:
                        return None
                    fields.add(k[1])
                elif isinstance(k, tuple):  # ("read", first, last): runs at test time
                    span = range(k[1], k[2] + 1)
                    if any(x in imp_rel for x in span):
                        return None  # also while sqlglot imports: what it built may persist
                    hook_lines.setdefault(rel, []).extend(span)
                elif k == "hook":
                    hook_lines.setdefault(rel, []).append(ln)
        import re

        def lines_of(text, word):
            pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(word) + r"(?![A-Za-z0-9_])")
            return [i for i, s in enumerate(text.split("\n"), 1) if pat.search(s)]

        # the patched files may not add other kinds of reads either
        for rel, text in plan.files.items():
            if text is None or not rel.endswith(".py") or rel.startswith("tests/"):
                continue
            ls = lines_of(text, attr)
            if not ls:
                continue
            if not rel.startswith("sqlglot/"):
                return None
            kinds = _flow_kinds(text, attr, ls)
            if kinds is None:
                return None
            for k in kinds.values():
                if isinstance(k, tuple) and k[0] == "transfer":
                    if rel != _TOK_REL:
                        return None
                    fields.add(k[1])
        core_src = read_pristine(_CORE_REL)
        if core_src is None:
            return None
        imp = set(IMPORT_LINES.get(_CORE_REL, ()))
        for field in fields:
            uses = _core_field_uses(core_src, field)
            if uses is None or uses & imp:
                return None  # read some other way, or while sqlglot imports
            # nothing else may read the field off a core: no `.field` access
            # and no "field" string anywhere else, old or patched
            for rel in {r for r, _l, _t in _mentions_cached(field)}:
                if rel is None:
                    return None
                if rel == _CORE_REL or not rel.endswith(".py"):
                    continue
                if _reads_member(read_pristine(rel), field):
                    return None
            for rel, text in plan.files.items():
                if text is not None and rel.endswith(".py") and lines_of(text, field) and _reads_member(text, field):
                    return None
            out |= cov_tests(_CORE_REL, sorted(uses))
        for rel, ls in hook_lines.items():
            out |= cov_tests(rel, ls)
        _FLOW[ck] = out
        return out
    except Exception:  # noqa: BLE001
        return None


def cheapest_cover(have: set, pairs: list) -> set:
    """One set from each (a, b) pair, chosen so that the union with have
    costs the fewest recorded test seconds."""
    import itertools

    def cost(sel):
        u = set().union(*sel) - have
        return sum(SEC.get(n, 0) for n in u)

    if len(pairs) > 8:
        opts = [[a for a, _ in pairs], [b for _, b in pairs]]
    else:
        opts = (list(c) for c in itertools.product(*pairs))
    best = min(opts, key=cost)
    return set().union(*best)


def attr_reader_tests(key: str, attr: str, sub, pristine: dict, patched: dict) -> set:
    """Tests that read class attribute key.attr through the class family sub."""
    out: set = set()
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
    elif not hit and (attr.startswith("_") or attr not in LK_BY_ATTR):
        # (private attributes are mostly derived caches a metaclass or a
        # lazily built object reads before any test runs, so the trace may
        # not have seen their readers)
        # reads the descriptors cannot record (dunders such as __slots__,
        # enum members, descriptors): every user of the class family
        out |= family_tests(key, sub)
    return out


def fresh_member(key: str, attr: str, plan: Plan, pristine: dict) -> bool:
    """A class member the patch adds under a name that appears in no
    pristine file. Only the patch's own new text can name it (its definition,
    and calls from changed code, which are charged as such), and on a class whose whole
    MRO is sqlglot code (no Enum, dataclass, NamedTuple or Protocol, no
    foreign metaclass) nothing enumerates members generically, except the
    generator's dispatch on public *_sql methods (dispatch_member_tests)."""
    if attr.endswith("_sql"):
        return False
    return _fresh_class(key, attr, plan, pristine) is not None


def dispatch_member_tests(key: str, attr: str, plan: Plan, pristine: dict, sub: set) -> set | None:
    """Tests a new public generator method X_sql can reach, when nothing
    calls it by name (every mention is a def header) and no ancestor or
    mixin had it. The generator finds such a
    method only by name: its dispatch table (built from dir(cls), keyed by
    the class exp.EXPR_CLASSES[X]) and lookups of node.key + "_sql". Before
    the patch, a node of that class with no handler went to
    function_fallback_sql (it must be a Func), which reads the class's
    arg_types; every key-based lookup reads its key. Both reads are recorded
    through the node's class, so only tests that read something through a
    class keyed X, and used the generator family, can change. None when the
    rule does not apply."""
    if not attr.endswith("_sql") or attr.startswith("_"):
        return None
    cls = _member_class(key, attr, pristine)
    if cls is None:
        return None
    # no ancestor had it, and nothing names it but def headers (other
    # generators' own X_sql): it is found only by the lookups below
    if any(attr in k.__dict__ for k in cls.__mro__[1:]) or not _only_defs(attr, plan.files.values()):
        return None
    # and every subclass now resolves it to this method or its own override
    # (not to a mixin's, which had been its handler before)
    stack, seen = list(cls.__subclasses__()), set()
    while stack:
        sc = stack.pop()
        if sc in seen:
            continue
        seen.add(sc)
        stack.extend(sc.__subclasses__())
        owner = next((k for k in sc.__mro__ if attr in k.__dict__), None)
        if owner is not cls and not (isinstance(owner, type) and issubclass(owner, cls)):
            return None
    try:
        from sqlglot import exp
        from sqlglot.generator import Generator
    except Exception:  # noqa: BLE001
        return None
    if not issubclass(cls, Generator):
        return None
    x = attr[:-4]
    targets: list = []
    c = getattr(exp, "EXPR_CLASSES", {}).get(x)
    if c is not None:
        if not (isinstance(c, type) and issubclass(c, exp.Func)):
            return None
        targets.append(c)
    try:
        for v in _LIVE.values():
            if isinstance(v, type) and issubclass(v, exp.Expr) and getattr(v, "key", None) == x and v not in targets:
                targets.append(v)
    except Exception:  # noqa: BLE001
        return None
    rev = {id(v): k for k, v in _LIVE.items()}
    out: set = set()
    for v in targets:
        k = rev.get(id(v))
        if k is None:
            return None
        sid = CLASSUSE.get(k)
        if sid is not None:
            out.update(SETS[sid])
    return out & family_tests(key, sub) if out else out


def _fresh_class(key: str, attr: str, plan: Plan, pristine: dict):
    """The live class, when attr is a member the patch adds to it under a
    name found in no pristine file, on a class whose MRO is all sqlglot code;
    else None. Every other mention of the name is then new text in a changed
    file (a call from a changed function, say), and the change there is
    charged on its own."""
    cls = _member_class(key, attr, pristine)
    if cls is None:
        return None
    return None if _mentions(attr, []) else cls


def _only_defs(word: str, texts) -> bool:
    """Whether every whole-word occurrence of word, in texts and in the files
    of the pristine checkout a test run can import or read, is a def header."""
    import re

    pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(word) + r"(?![A-Za-z0-9_])")
    dpat = re.compile(r"(?<![A-Za-z0-9_])def[ \t]+" + re.escape(word) + r"[ \t]*\(")
    for t in texts:
        if t is not None and len(pat.findall(t)) != len(dpat.findall(t)):
            return False
    w = word.encode()
    bpat = re.compile(pat.pattern.encode())
    bdpat = re.compile(dpat.pattern.encode())
    for dp, dns, fns in os.walk(PRISTINE):
        dns[:] = [d for d in dns if d not in (".git", "__pycache__") and not (dp == PRISTINE and d in _NOT_RUN)]
        for fn in fns:
            if dp == PRISTINE and fn.endswith((".md", ".png")):
                continue
            try:
                with open(os.path.join(dp, fn), "rb") as fh:
                    data = fh.read()
            except OSError:
                return False
            if w in data and len(bpat.findall(data)) != len(bdpat.findall(data)):
                return False
    return True


def _member_class(key: str, attr: str, pristine: dict):
    """The live class, when attr is a member the patch adds to a class whose
    MRO is all sqlglot code (no Enum, dataclass, NamedTuple or Protocol, no
    foreign metaclass); else None."""
    if attr.startswith("__") or (attr.startswith("_") and attr.endswith("_")) or not attr.isidentifier():
        return None
    pc = (pristine.get("classes") or {}).get(key)
    if pc is None or attr in pc:
        return None
    if not _LIVE:
        import fingerprint

        _LIVE.update(dict(fingerprint.iter_classes(sorted(sys.modules), "sqlglot")))
    cls = _LIVE.get(key)
    if not isinstance(cls, type):
        return None
    import dataclasses
    import typing

    if dataclasses.is_dataclass(cls):
        return None
    meta = type(cls)
    if meta is not type and not (getattr(meta, "__module__", "") or "").startswith("sqlglot"):
        return None
    for k in cls.__mro__:
        if k is object or k is typing.Generic:
            continue
        if not (getattr(k, "__module__", "") or "").startswith("sqlglot"):
            return None
    return cls


def family_tests(key: str, sub: set) -> set:
    """Every test that read anything through the class or a subclass; for
    classes nobody reads through (enums), everything referring to the class."""
    out: set = set()
    for x in sub:
        sid = CLASSUSE.get(x)
        if sid is not None:
            out.update(SETS[sid])
    if not out:
        mod, _, qual = key.partition(":")
        out |= global_ref_tests(mod, qual.split(".")[0])
    return out


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


IMPORT_SITES: dict = M.get("import_sites", {})
LAZY_ATTRS: dict = M.get("lazy_attrs", {})  # package -> name -> [module, attr] (PEP 562 __getattr__)
PRECISE_ERR = os.environ.get("BR_PRECISE_ERR", "1") == "1"  # 0: any new import error selects everything


def import_site_tests(key: str, gone: bool) -> set:
    """Tests that run an import statement of key ("mod" or "mod|name") inside
    a function, and (when the module or name is gone) every test of a test
    file importing it at module level: that file no longer imports."""
    out: set = set()
    for f, a, b, top in IMPORT_SITES.get(key, ()):
        if not top:
            out |= cov_tests(f, list(range(a, b + 1)))
        elif gone and f.startswith("tests/"):
            if f.endswith("__init__.py"):  # every test module of the package imports it
                d = os.path.dirname(f) + "/"
                out.update(n for tf, ns in BY_FILE.items() if tf.startswith(d) for n in ns)
            out.update(BY_FILE.get(f, ()))
            for imp in TEST_IMPORTERS.get(f, []):
                out.update(BY_FILE.get(imp, ()))
    return out


def global_ref_tests(mod: str, name: str) -> set:
    out: set = set()
    if name == "*":
        rel = mod.replace(".", "/") + ".py"
        for ln, sid_ in COV.get(rel, {}).items():
            out.update(SETS[sid_])
        sid_ = MODATTR_MOD.get(mod)
        if sid_ is not None:
            out.update(SETS[sid_])
        return out
    for f, ln in NAMES.refs(mod, name):
        out |= cov_tests(f, [ln])
    for k in (f"{mod}|{name}", f"{mod}|__dict__"):
        sid_ = MODATTR.get(k)
        if sid_ is not None:
            out.update(SETS[sid_])
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
    if os.path.lexists(root):
        # a leftover tree may hold a symlink where this patch needs a real
        # directory; writing through it would change the pristine checkout
        import shutil

        shutil.rmtree(root, ignore_errors=True)
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
    real_root = os.path.realpath(root) + os.sep
    for rel, text in files.items():
        dst = os.path.join(root, rel)
        if not os.path.realpath(os.path.dirname(dst)).startswith(real_root.rstrip(os.sep)):
            raise RuntimeError(f"farm path {rel} leaves the farm")
        if os.path.lexists(dst):
            os.remove(dst)
        if text is None:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8", errors="surrogateescape", newline="") as fh:
            fh.write(text)
    return root


def _core_modules() -> frozenset:
    try:
        with open(os.path.join(APP, "maps", "core.json")) as fh:
            return frozenset(json.load(fh))
    except Exception:  # noqa: BLE001
        return frozenset()


# the sqlglot modules `import sqlglot` runs at the pinned commit
CORE = _core_modules()


def _is_sqlglot(name: str) -> bool:
    return name == "sqlglot" or name.startswith("sqlglot.")


def preload_ok(plan) -> bool:
    """May this commit's child start from the parent's pristine `import
    sqlglot`? Only when the patch leaves alone every module that import runs
    and every file it looks for: it may change existing sqlglot modules outside
    that set and existing test-side files (not a package `__init__` or a
    conftest), and it adds, deletes and renames nothing. A fresh import in the
    child would then run exactly the code the parent ran."""
    if not PRELOAD or not CORE or not plan.files:
        return False
    for rel, text in plan.files.items():
        if text is None or not os.path.isfile(os.path.join(PRISTINE, rel)):
            return False
        if rel.startswith("sqlglot/"):
            if not rel.endswith(".py") or module_of(rel) in CORE:
                return False
        elif not rel.startswith("tests/") or os.path.basename(rel) in ("__init__.py", "conftest.py"):
            return False
    return True


def preload_core() -> bool:
    """Import pristine sqlglot in the parent (unittest is already loaded, as
    in a child). True when it loaded exactly CORE."""
    before = set(sys.modules)
    sys.path.insert(0, PRISTINE)
    try:
        import sqlglot  # noqa: F401
        ok = True
    except _ParentTimeout:
        raise
    except BaseException:  # noqa: BLE001
        ok = False
    finally:
        if sys.path and sys.path[0] == PRISTINE:
            del sys.path[0]
    loaded = {m for m in set(sys.modules) - before if _is_sqlglot(m)}
    return ok and loaded == CORE


def purge_sqlglot() -> None:
    for name in [m for m in sys.modules if _is_sqlglot(m)]:
        del sys.modules[name]


def rebase_core(root: str) -> bool:
    """Make the preloaded modules look imported from the farm, as a fresh
    import there would have them: their files, and above all the package
    paths that submodules (the patched ones among them) are found through."""
    old = os.path.join(PRISTINE, "")
    new = os.path.join(root, "")

    def fix(p):
        if p is None:
            return None
        if not isinstance(p, str) or not p.startswith(old):
            raise ValueError(p)
        return new + p[len(old):]

    try:
        mods = [(n, m) for n, m in sys.modules.items() if _is_sqlglot(n)]
        if {n for n, _ in mods} != CORE:
            return False
        plan = []
        for _n, m in mods:
            spec = m.__spec__
            path = getattr(m, "__path__", None)
            if path is not None and (not isinstance(path, list) or spec.submodule_search_locations is not path):
                return False
            plan.append((m, spec, fix(m.__file__), fix(getattr(m, "__cached__", None)), fix(spec.origin),
                         fix(spec.cached), [fix(p) for p in path] if path is not None else None))
    except Exception:  # noqa: BLE001
        return False
    for m, spec, file_, cached, origin, scached, path in plan:
        m.__file__ = file_
        m.__cached__ = cached
        spec.origin = origin
        spec.cached = scached
        if path is not None:
            m.__path__[:] = path  # the same list as spec.submodule_search_locations
        if hasattr(m.__loader__, "path"):
            m.__loader__.path = origin
    return True


class _Timeout(BaseException):
    pass


TIMED_OUT = [False]  # debug: some verification timed out
SETUP_LIMIT = float(os.environ.get("BR_SETUP_LIMIT", "15"))  # class/module set-up and tear-down


def _on_alarm(*_):
    TIMED_OUT[0] = True
    raise _Timeout()


# CPU seconds a child may spend importing the patched code (sqlglot for the
# snapshot, test modules before verification). Normal imports take a fraction
# of a second; a patch that loops forever where modules run code at import
# (a parser loop reached from a module-level parse) would otherwise hold the
# child until the parent kills it. That patch makes pytest's collection hang
# too, so the answer is every test.
PREP_CPU = float(os.environ.get("BR_PREP_CPU", "8"))  # test modules
SNAP_CPU = float(os.environ.get("BR_SNAP_CPU", "5"))  # sqlglot, for the snapshot (worst seen: 1.1 s)
HUNG = [False]


def _on_prof(*_):
    HUNG[0] = True
    raise _Timeout()


def _prep_limit(sec: float) -> None:
    """CPU-time limit (ITIMER_PROF, apart from the tests' wall-clock alarms); 0 cancels."""
    if sec > 0:
        signal.signal(signal.SIGPROF, _on_prof)
    signal.setitimer(signal.ITIMER_PROF, sec)


def _apply_split(split: dict, tpc: bool = True) -> None:
    """Cut the query lists setUpClass loads down to the queries to verify,
    and the optimizer tests' fixture loads down to the pairs to verify."""
    by_cls: dict = {}
    cf = {n: v for n, v in split.items() if v[0] == "@cf"}
    if cf:
        mod = sys.modules.get("tests.test_optimizer")
        t = cf_maps().get("tests", {})
        if mod is not None and hasattr(mod, "load_sql_fixture_pairs"):
            import cfsel

            cfsel.install(mod)
            for n, (_a, idx, nq, _c, _m) in cf.items():
                ent = t.get(CF_NUMS[n])
                if ent is None or ent["nq"] != nq:
                    continue  # runs whole
                keep: dict = {}
                off = 0
                for k, (fn, cnt) in enumerate(ent["loads"]):
                    keep[k] = (fn, cnt, {i - off for i in idx if off <= i < off + cnt})
                    off += cnt
                cfsel.KEEP[CF_NUMS[n]] = keep
    if not tpc:
        return
    for n, (attr, idx, nq, _c, _m) in split.items():
        if attr == "@cf":
            continue
        parts = TESTS_BY_NUM[n]["id"].split("::")
        mod = sys.modules.get(FILE_OF[n][:-3].replace("/", "."))
        cls = getattr(mod, parts[1], None) if mod is not None and len(parts) > 2 else None
        if cls is None:
            continue  # runs whole
        ent = by_cls.setdefault(cls, {}).setdefault(attr, [nq, set()])
        ent[1].update(idx)
    for cls, keep in by_cls.items():
        orig = cls.__dict__.get("setUpClass")
        if not isinstance(orig, classmethod):
            continue

        def su(c, _f0=orig.__func__, _keep=keep):
            _f0(c)
            for attr, (nq, idx) in _keep.items():
                seq = getattr(c, attr, None)
                if isinstance(seq, list) and len(seq) == nq:
                    setattr(c, attr, [seq[i] for i in sorted(idx)])

        cls.setUpClass = classmethod(su)


def run_tests(nums: list[int], split: dict | None = None) -> tuple[set, set]:
    """Run tests in this process. Returns (failing, unsure). Tests in `split`
    run on the listed queries only."""
    import unittest

    failing: set = set()
    unsure: set = set()
    loader = unittest.TestLoader()
    id2num = {}
    by_class: dict = {}
    for n in sorted(nums):  # collection order, as pytest runs them
        tid = next(t["id"] for t in [TESTS_BY_NUM[n]])
        parts = tid.split("::")
        mod = FILE_OF[n][:-3].replace("/", ".")
        uid = mod + "." + ".".join(parts[1:])
        id2num[uid] = n
        by_class.setdefault((mod, parts[1] if len(parts) > 2 else ""), []).append((n, uid))
    limits = {uid: 3.0 * SEC[n] + 1.5 for uid, n in id2num.items()}

    class R(unittest.TestResult):
        # A test has failed once one of its subtests fails, so the rest of
        # it can only cost time: with failfast, subTest() ends the test at
        # its first failing subtest. stop() is a no-op so that failfast
        # never ends the suite: the class's other tests still run.
        def __init__(self):
            super().__init__()
            self.failfast = True

        def stop(self):
            pass

        def startTest(self, test):
            super().startTest(test)
            signal.setitimer(signal.ITIMER_REAL, limits.get(test.id(), 5.0))

        def stopTest(self, test):
            # the next class's set-up (or this one's tear-down) runs before the next startTest
            signal.setitimer(signal.ITIMER_REAL, SETUP_LIMIT)
            super().stopTest(test)

    signal.signal(signal.SIGALRM, _on_alarm)
    suites = []
    _prep_limit(PREP_CPU)
    for (mod, cls), items in by_class.items():
        suite = unittest.TestSuite()
        for n, uid in items:
            try:
                suite.addTest(loader.loadTestsFromName(uid))
            except _Timeout:
                raise
            except BaseException:  # noqa: BLE001
                failing.add(n)
        suites.append((items, suite))
    _prep_limit(0)
    if HUNG[0]:
        raise _Timeout()
    _ph("load")
    import inline

    inline.install()
    if split:
        # (a query-looping TPC test that must run whole shares the class
        # attribute, so cut no TPC query list in that case)
        _apply_split(split, tpc=not any(n in SPLIT_NUMS and n not in split for n in nums))
    for items, suite in suites:
        res = R()
        try:
            signal.setitimer(signal.ITIMER_REAL, SETUP_LIMIT)
            suite.run(res)
            signal.setitimer(signal.ITIMER_REAL, 0)
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


def _mod_rel(mod: str) -> str:
    rel = mod.replace(".", "/")
    return rel + "/__init__.py" if os.path.isfile(os.path.join(PRISTINE, rel, "__init__.py")) else rel + ".py"


_DEFS: dict = {}


def _def_span(rel: str, qual: str):
    """(first, last) line of the function `qual` (dotted path) in the
    pristine source of rel, or None if it is not a def there."""
    k = (rel, qual)
    if k not in _DEFS:
        import ast

        span = None
        try:
            body = ast.parse(read_pristine(rel) or "").body
            node = None
            for part in qual.split("."):
                node = next((x for x in body if isinstance(x, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and x.name == part), None)
                if node is None:
                    break
                body = node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                span = (astdiff._first(node), node.end_lineno)
        except Exception:  # noqa: BLE001
            span = None
        _DEFS[k] = span
    return _DEFS[k]


def fe_import_ok(n: int, d: dict, plan, pristine=None, patched=None) -> bool:
    """Can the fingerprint check also vouch for test n under this import-time
    diff? Only if nothing that changed is used by the test's execution side:
    changed functions never run there, changed values are never read there."""
    if getattr(plan, "unseen", False) or d.get("errors"):
        return False
    fe = fe_maps().get(SPLIT_NUMS.get(n, ""))
    if not fe or "exec_names" not in fe:
        return False
    ex = fe["_exec"]
    xa = set(fe.get("exec_attrs", ()))
    xn = set(fe.get("exec_names", ()))
    lkby = fe.get("_lkby")
    if lkby is None and "exec_lk" in fe:
        lkby = fe["_lkby"] = {}
        for k in fe["exec_lk"]:
            c, _, a = k.rpartition("|")
            lkby.setdefault(a, set()).add(c)
    subtree = class_subtree(pristine, patched)

    def runs_there(rel, span):
        return any(span[0] <= abs(l) <= span[1] for l in ex.get(rel, ()))

    for key, attr in d["attrs"]:
        if attr == "__doc__":
            continue
        if attr == "*" or attr.startswith("__"):
            return False
        mod, _, qual = key.partition(":")
        rel = _mod_rel(mod)
        span = _def_span(rel, f"{qual}.{attr}")
        if span is not None:
            if runs_there(rel, span):
                return False
            continue
        if lkby is not None and pristine is not None:
            # which classes the execution side read this attribute through
            readers = lkby.get(attr, set())
            if readers & subtree(key):
                return False
            if not readers and attr in xn:
                return False  # named there but never read as a recorded class attribute
            continue
        if attr in xa or attr in xn:
            return False
    gmods = {m for m, _ in d["globals"]}
    for mod, name in d["globals"]:
        if name == "__doc__":
            continue
        spans = [(r, sp) for r, sp in ((_mod_rel(m), _def_span(_mod_rel(m), name)) for m in gmods) if sp is not None]
        if spans:  # a function (possibly imported under the same name elsewhere)
            if any(runs_there(r, sp) for r, sp in spans):
                return False
            continue
        if name in xn:
            return False
    return True


def fe_check(vfe: dict) -> set:
    """Tests whose front-end artifacts differ from the pinned commit's on
    one of their queries (or cannot be computed): those count as failing."""
    failing = set()
    try:
        import tpcfp
    except Exception:  # noqa: BLE001
        return set(vfe)
    for n, (attr, idx, nq, _c, _m) in vfe.items():
        tid = SPLIT_NUMS[n]
        fe = fe_maps().get(tid) or {}
        try:
            if tpcfp.differing(tid, idx, fe.get("arts", {})):
                failing.add(n)
        except BaseException:  # noqa: BLE001
            failing.add(n)
    return failing


FE_RUN_MAX = int(os.environ.get("BR_FE_RUN_MAX", "8"))


def fe_check_queries(vfe: dict) -> tuple[set, dict]:
    """Like fe_check, but when only a few queries differ, plan to run the test
    on just those. Queries with equal artifacts behave as at the pinned commit
    (the execution side is untouched: that is what made the test eligible),
    so the test passes exactly when it passes on the differing ones.
    Returns (failing, {test: split entry to run})."""
    failing: set = set()
    runs: dict = {}
    try:
        import tpcfp
    except Exception:  # noqa: BLE001
        return set(vfe), runs
    signal.signal(signal.SIGALRM, _on_alarm)
    for n, (attr, idx, nq, c, _m) in vfe.items():
        tid = SPLIT_NUMS[n]
        fe = fe_maps().get(tid) or {}
        try:
            # a front end that hangs counts as differing
            signal.setitimer(signal.ITIMER_REAL, 5.0 + 4.0 * c)
            D = tpcfp.differing_upto(tid, idx, fe.get("arts", {}), FE_RUN_MAX)
            signal.setitimer(signal.ITIMER_REAL, 0)
        except BaseException:  # noqa: BLE001
            signal.setitimer(signal.ITIMER_REAL, 0)
            D = None
        if D == []:
            continue
        sm = split_maps().get("tests", {}).get(tid)
        if D is None or not sm or sm.get("nq") != nq or sm.get("attr") != attr:
            failing.add(n)
            continue
        cost = sm["fixed"] + sum(sm["warm"][i] for i in D) + max(sm["cold"][i] - sm["warm"][i] for i in D)
        if cost * KAPPA > 0.5 * MS[n] / 1000:
            failing.add(n)
            continue
        runs[n] = (attr, sorted(D), nq, round(cost, 3), "run")
    return failing, runs


CHILD_MEM = int(float(os.environ.get("BR_CHILD_MEM_MB", "1300")) * (1 << 20))
CHILD_VM = int(float(os.environ.get("BR_CHILD_VM_MB", "3000")) * (1 << 20))
_VM0 = [0]


def _status_kb(field: str):
    with open("/proc/self/status") as fh:
        return next((int(l.split()[1]) for l in fh if l.startswith(field + ":")), None)


def cap_memory() -> None:
    """A patch can make a test allocate without end (a loop that keeps
    growing a list), faster than its timer fires; two children run at once
    in a 4 GB container. A thread checks this child's resident memory every
    10 ms and kills the child (and anything it started) once it holds
    CHILD_MEM bytes more than at the fork: with no answer from it, the
    parent keeps every candidate. Address space is capped too, far above
    that, for a single allocation too big to wait for (libraries such as
    duckdb reserve address space they never touch, so resident memory is
    the real measure)."""
    try:
        import resource
        import threading

        page = os.sysconf("SC_PAGE_SIZE")
        fd = os.open("/proc/self/statm", os.O_RDONLY)

        def rss() -> int:
            return int(os.pread(fd, 128, 0).split()[1]) * page

        limit = rss() + CHILD_MEM
        base = _status_kb("VmData")
        if base is not None:
            _VM0[0] = base
            cap = base * 1024 + CHILD_VM
            soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
            if hard != resource.RLIM_INFINITY:
                cap = min(cap, hard)
            resource.setrlimit(resource.RLIMIT_DATA, (cap, hard))

        def watch() -> None:
            while True:
                time.sleep(0.01)
                try:
                    if rss() > limit:
                        if os.getpgid(0) == os.getpid():  # its own group (never the parent's)
                            os.killpg(os.getpid(), signal.SIGKILL)
                        os.kill(os.getpid(), signal.SIGKILL)
                except Exception:  # noqa: BLE001
                    return

        threading.Thread(target=watch, daemon=True).start()
    except Exception:  # noqa: BLE001  (no /proc: nothing to do)
        pass


def child_main(plan: Plan, wfd: int) -> None:
    """Runs in a forked child. Writes one JSON object to wfd."""
    result: dict = {"cid": plan.cid}
    os.dup2(2, 1)  # tests may print: never let a child write to our stdout
    cap_memory()
    # importing sqlglot and fingerprinting it leave almost no cyclic garbage:
    # no collector passes until something runs sqlglot (tests, TPC checks)
    gc.disable()
    try:
        _ph("start")
        root = build_farm(plan.cid, plan.files)
        pre = bool(getattr(plan, "preloaded", False)) and rebase_core(root)
        if not pre:
            purge_sqlglot()  # import it fresh from the farm
        if DEBUG:
            result["pre"] = pre
        _ph("farm")
        _prep_limit(SNAP_CPU)
        os.chdir(root)
        sys.path.insert(0, root)
        os.environ["SKIP_INTEGRATION"] = "0"
        sys.dont_write_bytecode = True
        cands = Cands(plan.cands)
        everything = False
        lazy = None
        if plan.need_snapshot and LAZY_NUM in cands and FILE_OF[LAZY_NUM] not in plan.files:
            # test_lazy_load runs `import sqlglot; assert len(sqlglot.Dialect._classes) == 1`
            # in a fresh interpreter from the checkout. Nothing here has imported
            # sqlglot yet, so doing that import first answers it without the subprocess.
            try:
                import sqlglot as _sg

                lazy = len(_sg.Dialect._classes) == 1
            except _Timeout:
                raise
            except BaseException:  # noqa: BLE001
                lazy = False
        if plan.need_snapshot:
            import fingerprint

            pristine = pristine_snapshot()
            changed_mods = {module_of(r) for r in plan.files if r.startswith("sqlglot/") and r.endswith(".py")}
            only = affected_modules(changed_mods)
            patched = fingerprint.snapshot("sqlglot", only)
            d = fingerprint.diff(pristine, patched, only)
            result["only"] = len(only) if only is not None else -1
            extra, everything = snapshot_candidates(pristine, patched, d, plan)
            cands.add(extra, P_ATTR)
            _ph("snap")
            result["diff"] = {"attrs": d["attrs"][:50], "globals": d["globals"][:50], "errors": d["errors"][:5], "n": len(extra)}
        _prep_limit(0)
        if HUNG[0]:
            raise _Timeout()  # (in case a handler above swallowed it)
        gc.enable()
        if everything:
            result["select"] = sorted(ALL)
            result["verified"] = 0
        else:
            split = {n: v for n, v in plan.split.items() if n not in cands.other}
            if plan.need_snapshot and not everything:
                # TPC tests reached through import-time state: still decidable
                # from front-end fingerprints when nothing changed is used by
                # their execution side (all queries are checked)
                for n in SPLIT_NUMS:
                    if getattr(plan, "import_errors", None):
                        break  # the execution side may not import: run TPC tests for real
                    if n not in cands or n in plan.cands.other:
                        continue
                    if n in plan.cands and (plan.split.get(n) or (None,) * 5)[4] != "fe":
                        continue
                    if not fe_import_ok(n, d, plan, pristine, patched):
                        continue
                    sm = split_maps().get("tests", {}).get(SPLIT_NUMS[n])
                    fe = fe_maps().get(SPLIT_NUMS[n])
                    if not sm or fe.get("nq") != sm["nq"] or fe.get("unstable"):
                        continue
                    nq = sm["nq"]
                    split[n] = (sm["attr"], list(range(nq)), nq, round(0.3 + sum(fe["cpu"].values()), 3), "fe")
            if getattr(plan, "import_errors", None):
                split = {n: v for n, v in split.items() if v[4] != "fe"}
            verify, include = choose_verify(cands, True, split)
            vsplit = {n: v for n, v in split.items() if n in verify}
            vfe = {n: v for n, v in vsplit.items() if v[4] == "fe"}
            vrun = {n: v for n, v in vsplit.items() if v[4] != "fe"}
            rest = [n for n in verify if n not in vfe and not (n == LAZY_NUM and lazy is not None)]
            fe_failing, fe_runs = fe_check_queries(vfe) if vfe else (set(), {})
            rest += sorted(fe_runs)
            vrun = {**vrun, **fe_runs}
            failing, _ = run_tests(rest, vrun) if rest else (set(), set())
            failing |= fe_failing
            if lazy is False and LAZY_NUM in verify:
                failing.add(LAZY_NUM)
            if vsplit:
                result["split"] = {str(n): [len(v[1]), v[3], v[4], n in failing] + ([len(fe_runs[n][1])] if n in fe_runs else []) for n, v in vsplit.items()}
            result["select"] = sorted(set(include) | failing)
            result["verified"] = len(verify)
            if DEBUG:
                result["vlist"] = sorted(verify)
                if TIMED_OUT[0]:
                    result["timed_out"] = True
            result["failing_verified"] = sorted(failing)
            result["cands"] = len(cands)
    except BaseException as e:  # noqa: BLE001
        import traceback

        _prep_limit(0)
        if HUNG[0]:
            result["select"] = sorted(ALL)
            result["verified"] = 0
            result["hung"] = True
        else:
            result["error"] = traceback.format_exc()[-1500:]
    import resource as _r

    _u = _r.getrusage(_r.RUSAGE_SELF)
    _c = _r.getrusage(_r.RUSAGE_CHILDREN)
    result["cpu"] = round(_u.ru_utime + _u.ru_stime + _c.ru_utime + _c.ru_stime, 3)
    if DEBUG:
        _ph("end")
        result["prof"] = PROF
        result["maxrss_mb"] = round(_u.ru_maxrss / 1024)
        try:
            result["vmdata_mb"] = round((_status_kb("VmData") - _VM0[0]) / 1024)
        except Exception:  # noqa: BLE001
            pass
    data = (json.dumps(result) + "\n").encode()
    while data:
        k = os.write(wfd, data)
        data = data[k:]
    os.close(wfd)


# ------------------------------------------------------------------------ main
class _ParentTimeout(BaseException):
    pass


def _parent_alarm(signum, frame):
    raise _ParentTimeout()


ANALYZE_LIMIT = float(os.environ.get("BR_ANALYZE_LIMIT", "20"))


def main() -> None:
    """Whatever goes wrong in this process (an error, a patch that keeps the
    analysis busy, the batch running out of time), every commit still gets
    exactly one line on stdout: the selection made so far, or a safe one."""
    st: dict = {"commits": None, "plans": {}, "selections": {}, "outputs": {}, "printed": set(), "running": {}}
    signal.signal(signal.SIGALRM, _parent_alarm)
    try:
        _main(st)
    except BaseException as e:  # noqa: BLE001
        signal.setitimer(signal.ITIMER_REAL, 0)
        log("parent error, printing safe selections:", repr(e)[:300])
        for cpid in list(st["running"]):
            for kill in (os.killpg, os.kill):
                try:
                    kill(cpid, signal.SIGKILL)
                except OSError:
                    pass
        commits = st["commits"]
        if commits is None:
            commits = sorted(p[:-6] for p in os.listdir(os.path.join(DATA, "commits")) if p.endswith(".patch"))
        for cid in commits:
            if cid in st["printed"]:
                continue
            sel = st["selections"].get(cid)
            o = st["outputs"].get(cid)
            p = st["plans"].get(cid)
            if sel is None and o and "select" in o and "error" not in o:
                sel = o["select"]
            if sel is None:
                sel = sorted(p.cands) if p is not None and not p.need_snapshot and not p.all_tests else list(ALL)
            try:
                workers = schedule(sel)
            except BaseException:  # noqa: BLE001
                u = sorted(set(sel) & set(ALL))
                workers = [w for w in (u[i::8] for i in range(8)) if w]
            sys.stdout.write(json.dumps({"commit": cid, "workers": workers}) + "\n")
            sys.stdout.flush()
            st["printed"].add(cid)


def _main(st: dict) -> None:
    commits = sorted(p[:-6] for p in os.listdir(os.path.join(DATA, "commits")) if p.endswith(".patch"))
    st["commits"] = commits
    n = len(commits)
    # wall-clock limits: 3 minutes for practice (40 commits), 4.5 for the final
    # (about 60); only local benchmarks far larger than either scale them up
    scale = max(1.0, n / (40 if n <= 45 else 60)) if n > 120 else 1.0
    budget = float(os.environ.get("BR_BUDGET", 0)) or (150.0 if n <= 45 else 235.0) * scale
    plans: dict[str, Plan] = st["plans"]
    hard = float(os.environ.get("BR_HARD", 0)) or (168.0 if n <= 45 else 258.0) * scale
    for cid in commits:
        with open(os.path.join(DATA, "commits", cid + ".patch"), encoding="utf-8", errors="surrogateescape", newline="") as fh:
            text = fh.read()
        # a patch that keeps the analysis busy gets every test, not the batch's time
        left = hard - 20.0 - (time.time() - T0)
        try:
            if left < 0.5:
                raise _ParentTimeout()
            signal.setitimer(signal.ITIMER_REAL, min(ANALYZE_LIMIT, left))
            p = analyze(cid, text)
            signal.setitimer(signal.ITIMER_REAL, 0)
            plans[cid] = p
        except _ParentTimeout:
            signal.setitimer(signal.ITIMER_REAL, 0)
            p = Plan(cid)
            p.all_tests = True
            p.notes.append("analyze timeout")
            plans[cid] = p
        except Exception as e:  # noqa: BLE001
            signal.setitimer(signal.ITIMER_REAL, 0)
            p = Plan(cid)
            p.all_tests = True
            p.notes.append(f"analyze error {e!r}")
            plans[cid] = p
    _ru = __import__("resource").getrusage(0)
    log(f"phase analyze cpu {_ru.ru_utime + _ru.ru_stime:.2f}")
    selections: dict[str, list[int]] = st["selections"]
    jobs = []
    # past this the batch is out of time: main() prints what is known
    signal.setitimer(signal.ITIMER_REAL, max(0.1, hard + 6.0 - (time.time() - T0)))
    for cid, p in plans.items():
        if UNKNOWN and (p.cands or p.need_snapshot):
            p.cands.add(UNKNOWN, P_LOW)
        try:
            p.split = split_plan(p)
        except Exception as e:  # noqa: BLE001
            p.split = {}
            p.notes.append(f"split error {e!r}")
        if p.all_tests:
            selections[cid] = list(ALL)
        elif p.need_snapshot:
            jobs.append(cid)
        elif not p.cands:
            selections[cid] = []
        else:
            verify, include = choose_verify(p.cands, False, {n: v for n, v in p.split.items()})
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
    # Children whose patch leaves every module `import sqlglot` runs alone can
    # start from one pristine import in the parent instead of each importing
    # it; they are forked after the others, so no other child inherits it.
    # One import in the parent pays only if it saves at least two.
    pre = {c for c in jobs if preload_ok(plans[c])}
    if len(pre) < 2:
        pre = set()
    # cheap first so a slow child cannot starve the rest
    jobs.sort(key=lambda c: (c in pre, plans[c].need_snapshot, sum(SEC[x] for x in plans[c].cands)))
    running: dict[int, tuple[str, int, float]] = st["running"]
    outputs: dict[str, dict] = st["outputs"]
    max_par = int(os.environ.get("BR_PAR", "2"))
    if os.environ.get("BR_GCFREEZE", "1") == "1":
        gc.collect()
        gc.freeze()  # children: no GC pass over the parent's objects (copy-on-write)
    queue = list(jobs)
    # hard wall-clock watchdog: the whole batch must finish inside the limit
    child_limit = float(os.environ.get("BR_CHILD_LIMIT", "45"))
    while queue or running:
        while queue and len(running) < max_par and time.time() - T0 < budget:
            cid = queue.pop(0)
            if cid in pre and "core" not in st:
                st["core"] = preload_core()
                if os.environ.get("BR_GCFREEZE", "1") == "1":
                    gc.freeze()
            plans[cid].preloaded = cid in pre and st.get("core", False)
            r, w = os.pipe()
            pid = os.fork()
            if pid == 0:
                os.close(r)
                try:
                    os.setpgid(0, 0)  # own process group: the watchdog kills it whole
                    child_main(plans[cid], w)
                finally:
                    os._exit(0)
            os.close(w)
            running[pid] = (cid, r, time.time())
        if not running:
            break
        pid, _status = os.waitpid(-1, os.WNOHANG)
        if pid == 0:
            now = time.time()
            for cpid, (ccid, cr, cst) in list(running.items()):
                if now - cst > child_limit or now - T0 > hard:
                    try:
                        os.killpg(cpid, signal.SIGKILL)
                    except OSError:
                        try:
                            os.kill(cpid, signal.SIGKILL)
                        except OSError:
                            pass
                    try:
                        os.waitpid(cpid, 0)
                    except ChildProcessError:
                        pass
                    os.close(cr)
                    running.pop(cpid, None)
                    outputs[ccid] = {"error": f"killed after {now - cst:.1f}s"}
            time.sleep(0.01)
            continue
        if pid not in running:
            continue
        cid, r, t_start = running.pop(pid)
        try:
            os.killpg(pid, signal.SIGKILL)  # whatever the child left running in its group
        except OSError:
            pass
        buf = b""
        # the child has exited, so its answer is already in the pipe; a process
        # it left outside its group may still hold the write end open
        while select.select([r], [], [], 1.0)[0]:
            chunk = os.read(r, 65536)
            if not chunk:
                break
            buf += chunk
        os.close(r)
        try:
            outputs[cid] = json.loads(buf.decode() or "{}")
        except Exception:  # noqa: BLE001
            outputs[cid] = {"error": "bad child output"}
        outputs[cid]["secs"] = round(time.time() - t_start, 2)
    for cid in jobs:
        o = outputs.get(cid)
        p = plans[cid]
        if o and "select" in o and "error" not in o:
            selections[cid] = o["select"]
        else:
            # no answer from the child (error or out of time): fall back safely
            selections[cid] = list(ALL) if p.need_snapshot else sorted(p.cands)
            log(cid, "fallback", (o or {}).get("error", "no output")[-400:])
    lines = [(cid, json.dumps({"commit": cid, "workers": schedule(selections.get(cid, list(ALL)))}) + "\n") for cid in commits]
    signal.setitimer(signal.ITIMER_REAL, 0)
    for cid, line in lines:
        sys.stdout.write(line)  # one write per line: never half a line
        sys.stdout.flush()
        st["printed"].add(cid)
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
