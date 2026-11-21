"""Front-end fingerprints for the TPC executor tests.

test_execute_tpcds (and its TPC-H siblings) spends almost all of its time
running each query twice over fixed data: once in sqlglot's Python executor,
once in DuckDB on sqlglot's transpiled SQL. What sqlglot contributes before
that is cheap: parse, pickle round trip, find_tables, optimize, plan,
transpile. When a change only reaches that front end, it can only alter the
outcome by altering what the front end hands over. So the selector
recomputes those artifacts with the patched code and compares them with the
pinned commit's: identical artifacts, with the execution-side code untouched,
mean identical results.

`artifacts()` must mirror the tests exactly; it is shared by the build-time
recorder (ftrace.py) and the selector.
"""

from __future__ import annotations

import enum
import hashlib
import pickle

FIXTURES = {"tpcds_sqls": "optimizer/tpc-ds/tpc-ds.sql", "tpch_sqls": "optimizer/tpc-h/tpc-h.sql"}
SCHEMAS = {"tpcds_sqls": "TPCDS_SCHEMA", "tpch_sqls": "TPCH_SCHEMA"}
# test method -> (query list attribute, kind)
TESTS = {
    "tests/test_executor.py::TestExecutor::test_execute_tpcds": ("tpcds_sqls", "mp"),
    "tests/test_executor.py::TestExecutor::test_execute_tpch": ("tpch_sqls", "mp"),
    "tests/test_executor.py::TestExecutor::test_optimized_tpch": ("tpch_sqls", "opt"),
}
_SKIP_ATTRS = {"dependents", "_dag", "parent", "arg_key", "index"}


def _canon(x, memo: dict, out: list) -> None:
    t = type(x)
    if x is None or t in (bool, int, float, str, bytes, complex):
        out.append(t.__name__[:1] + repr(x))
        return
    if isinstance(x, type):
        out.append("T" + x.__module__ + "." + x.__qualname__)
        return
    if isinstance(x, enum.Enum):
        out.append("E" + t.__qualname__ + "." + str(x.name))
        return
    i = id(x)
    if i in memo:
        out.append("@" + str(memo[i]))
        return
    memo[i] = len(memo)
    if t is list or t is tuple:
        out.append("[" if t is list else "(")
        for v in x:
            _canon(v, memo, out)
        out.append("]")
    elif isinstance(x, dict):
        out.append("{")
        for k, v in sorted(x.items(), key=lambda kv: repr(kv[0])):
            _canon(k, memo, out)
            _canon(v, memo, out)
        out.append("}")
    elif isinstance(x, (set, frozenset)):
        # order-free: each element on its own
        out.append("S" + ",".join(sorted(canon(v) for v in x)))
    elif hasattr(x, "args") and hasattr(x, "arg_types"):  # a sqlglot expression
        out.append("X" + t.__module__ + "." + t.__qualname__)
        for k in sorted(x.args):
            out.append(k)
            _canon(x.args[k], memo, out)
        for a in ("_type", "_meta", "comments"):
            out.append(a)
            _canon(getattr(x, a, None), memo, out)
    else:
        out.append("O" + t.__module__ + "." + t.__qualname__)
        fields = {}
        d = getattr(x, "__dict__", None)
        if d:
            fields.update(d)
        for c in t.__mro__:
            for s in getattr(c, "__slots__", ()) or ():
                if s not in fields and hasattr(x, s):
                    fields[s] = getattr(x, s)
        for k in sorted(fields):
            if k in _SKIP_ATTRS:
                continue
            out.append(k)
            _canon(fields[k], memo, out)


def canon(x) -> str:
    out: list = []
    _canon(x, {}, out)
    return "\x1f".join(out)


def fp(x) -> str:
    return hashlib.sha1(canon(x).encode("utf-8", "surrogatepass")).hexdigest()[:20]


def queries(attr: str) -> list:
    from tests.helpers import load_sql_fixture_pairs

    return list(load_sql_fixture_pairs(FIXTURES[attr]))


def artifacts(tid: str, i: int, sqls=None) -> list:
    """Fingerprints of everything the test's front end produces for query i."""
    import tests.helpers as helpers
    from sqlglot import parse_one, transpile

    attr, kind = TESTS[tid]
    sqls = sqls if sqls is not None else queries(attr)
    meta, sql, expected = sqls[i]
    if kind == "opt":
        # test_optimized_tpch: DuckDB runs both transpiled queries
        return [fp(transpile(sql, write="duckdb")[0]), fp(transpile(expected, write="duckdb")[0])]
    # _mp_execute: parse in the test process, pickle to the worker ...
    e = parse_one(sql)
    e2 = pickle.loads(pickle.dumps(e, protocol=pickle.HIGHEST_PROTOCOL))
    out = [fp(e2)]
    if meta.get("execute"):
        # ... where mp_execute picks the tables find_tables names (by name) ...
        from sqlglot import find_tables
        from sqlglot.executor import optimize, Plan
        from sqlglot.schema import ensure_schema

        out.append(fp(sorted(t.name for t in find_tables(e2))))
        # ... and sqlglot.executor.execute optimizes, renders for its debug
        # log (only whether that raises matters) and plans it
        schema = ensure_schema(getattr(helpers, SCHEMAS[attr]), dialect=None)
        o = optimize(e2, schema, leave_tables_isolated=True, dialect=None)
        o.sql(pretty=True)
        out.append(fp(Plan(o)))
        # DuckDB side (cached_execute)
        out.append(fp(transpile(sql, write="duckdb")[0]))
    return out


def differing(tid: str, idx, expected: dict) -> list:
    """The first query of idx whose artifacts differ from `expected`
    (str(i) -> list), as a one-element list; [] when none does."""
    attr, _ = TESTS[tid]
    sqls = queries(attr)
    for i in idx:  # the first difference decides
        try:
            got = artifacts(tid, i, sqls)
        except Exception:  # noqa: BLE001
            return [i]
        if got != expected.get(str(i)):
            return [i]
    return []


def differing_upto(tid: str, idx, expected: dict, cap: int):
    """Every query of idx whose artifacts differ from `expected`, or None as
    soon as more than `cap` do (or one cannot be computed)."""
    attr, _ = TESTS[tid]
    sqls = queries(attr)
    out = []
    for i in idx:
        try:
            got = artifacts(tid, i, sqls)
        except Exception:  # noqa: BLE001
            return None
        if got != expected.get(str(i)):
            out.append(i)
            if len(out) > cap:
                return None
    return out
