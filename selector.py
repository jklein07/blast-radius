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


def split_queries(n: int, plan) -> tuple | None:
    """(attr, query indexes, nq, CPU cost, mode) when test n can be verified
    on only the queries that execute a changed line; None when it must run
    whole. mode "fe": every changed line it runs belongs to its front end
    (parse, optimize, plan, transpile), so comparing those artifacts with the
    pinned commit's decides it (tpcfp.py); mode "run": run those queries."""
    tid = SPLIT_NUMS.get(n)
    if tid is None or n in plan.cands.other:
        return None
    sm_all = split_maps()
    sm = sm_all.get("tests", {}).get(tid)
    if not sm:
        return None
    setup = sm_all.get("setup", {})
    tfile = FILE_OF[n]
    fe = fe_maps().get(tid)
    fe_ok = fe is not None
    q = 0
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


def split_plan(plan, cands=None) -> dict:
    cands = plan.cands if cands is None else cands
    out = {}
    for n in SPLIT_NUMS:
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
    B = [n for n in free if n in split]
    if B:
        nb = [n for n in free if n not in split]
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
        vb = sum(vc(n) + sc(n) for n in B)
        pb = 1.0
        for n in B:
            pb *= 1.0 - cands[n]
        for k2 in sorted({0, 1, 2, 3, 5, 8, 12, 20, 30, m2 // 2, m2} & set(range(m2 + 1))):
            U = fixed + nb[k2:]
            P = nb[:k2]
            s_u = sched_cost_ms(U) / 1000.0
            s_ub = sched_cost_ms(U + B) / 1000.0
            if k2:
                s_up = sched_cost_ms(U + P) / 1000.0
                s_upb = sched_cost_ms(U + P + B) / 1000.0
            else:
                s_up, s_upb = s_u, s_ub
            cost = ovh + pre[k2] + vb + pb * (pn[k2] * s_u + (1 - pn[k2]) * s_up) + (1 - pb) * (pn[k2] * s_ub + (1 - pn[k2]) * s_upb)
            if cost < best[0] - 1e-9:
                best = (cost, k2)
                choice = (P + B, U)
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
        listing = fp.is_new or fp.is_deleted or (old_rel != new_rel)
        for r in rels:  # tests that open the file itself (source readers too)
            plan.cands.add(tests_reading(r, listing), P_CODE if r.endswith(".py") else P_DATA)
        if path.endswith(".py") and (path.startswith("sqlglot/") or path.startswith("tests/")):
            oc, nc = fp.changed_lines() if (old is not None and new is not None and not fp.is_new) else (None, None)
            ch = astdiff.analyze(old, new, oc, nc)
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


_EXEC_CACHE: dict = {}


def exec_lines(rel: str, src: str | None) -> set:
    """Lines of the old source that carry bytecode (can fire a line event)."""
    v = _EXEC_CACHE.get(rel)
    if v is None:
        v = set()
        try:
            stack = [compile(src or "", rel, "exec", dont_inherit=True)]
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
    whole function instead. An operator swap counts only for the tests in
    which the new operator would give a different result."""
    swaps = getattr(ch, "swaps", None) or []
    swap_groups = set()
    out: set = set()
    for key, old_op, new_op, lines, gi in swaps:
        t = wm_tests(rel, key, old_op, new_op, lines)
        if t is None:
            out |= cov_tests(rel, lines)
        else:
            out |= t
            swap_groups.add(gi)
    out |= cov_tests(rel, ch.plain_lines if swaps else ch.runtime_lines)
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
    plan.cands.add(SPAWNERS, P_LOW)
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
    v = c.__dict__.get(attr)
    if type(v) is not dict:
        return None
    new = fingerprint.dict_key_fps(v)
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
    if diff["errors"]:
        new_err = {m for m, _ in diff["errors"] if m in patched["errors"]}
        if new_err:
            return set(ALL), True
    subtree = class_subtree(pristine, patched)

    live = None
    for key, attr in diff["attrs"]:
        if attr == "__doc__":
            continue  # docstrings: nothing reads them at runtime
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
    for mod, name in diff["globals"]:
        if name == "__doc__":
            continue
        out |= global_ref_tests(mod, name)
    # import-time edits the fingerprints could not see: fall back to everything
    # that refers to the names those statements bind
    seen_mods = {k.partition(":")[0] for k, _ in diff["attrs"]} | {m for m, _ in diff["globals"]}
    for rel, ch in plan.py_changes.items():
        if not (rel.startswith("sqlglot/") and ch.semantic and ch.import_time) or ch.syntax_error:
            continue
        mod = module_of(rel)
        if seen_mods & importer_closure({mod}):
            continue
        plan.unseen = True  # tests chosen by name, not by a measured difference
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


class _Timeout(BaseException):
    pass


def _on_alarm(*_):
    raise _Timeout()


def _apply_split(split: dict) -> None:
    """Cut the query lists setUpClass loads down to the queries to verify."""
    by_cls: dict = {}
    for n, (attr, idx, nq, _c, _m) in split.items():
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
    if split and not any(n in SPLIT_NUMS and n not in split for n in nums):
        # (a query-looping test that must run whole shares the class
        # attribute, so cut nothing in that case)
        _apply_split(split)
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
            changed_mods = {module_of(r) for r in plan.files if r.startswith("sqlglot/") and r.endswith(".py")}
            only = affected_modules(changed_mods)
            patched = fingerprint.snapshot("sqlglot", only)
            d = fingerprint.diff(pristine, patched, only)
            result["only"] = len(only) if only is not None else -1
            extra, everything = snapshot_candidates(pristine, patched, d, plan)
            cands.add(extra, P_ATTR)
            result["diff"] = {"attrs": d["attrs"][:50], "globals": d["globals"][:50], "errors": d["errors"][:5], "n": len(extra)}
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
            verify, include = choose_verify(cands, True, split)
            vsplit = {n: v for n, v in split.items() if n in verify}
            vfe = {n: v for n, v in vsplit.items() if v[4] == "fe"}
            vrun = {n: v for n, v in vsplit.items() if v[4] != "fe"}
            rest = [n for n in verify if n not in vfe]
            failing, _ = run_tests(rest, vrun) if rest else (set(), set())
            if vfe:
                failing |= fe_check(vfe)
            if vsplit:
                result["split"] = {str(n): [len(v[1]), v[3], v[4], n in failing] for n, v in vsplit.items()}
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
    # cheap first so a slow child cannot starve the rest
    jobs.sort(key=lambda c: (plans[c].need_snapshot, sum(SEC[x] for x in plans[c].cands)))
    running: dict[int, tuple[str, int, float]] = {}
    outputs: dict[str, dict] = {}
    max_par = int(os.environ.get("BR_PAR", "2"))
    queue = list(jobs)
    # hard wall-clock watchdog: the whole batch must finish inside the limit
    hard = float(os.environ.get("BR_HARD", 0)) or (168.0 if n <= 45 else 258.0) * max(1.0, n / (40 if n <= 45 else 60))
    child_limit = float(os.environ.get("BR_CHILD_LIMIT", "45"))
    while queue or running:
        while queue and len(running) < max_par and time.time() - T0 < budget:
            cid = queue.pop(0)
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
