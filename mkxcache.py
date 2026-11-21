"""Build-time: run TestExecutor's real class set-up on the pinned commit,
store the rows it parsed (xcache.bin), and check that xsetup's set-up
rebuilds exactly the same class state from them."""
import marshal
import os
import sys
import unittest  # noqa: F401  (sqlglot's strict checks, as in a test run)

root, dst = sys.argv[1], sys.argv[2]
sys.dont_write_bytecode = True
os.chdir(root)
sys.path.insert(0, root)
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
os.environ["SKIP_INTEGRATION"] = "0"

import tests.test_executor as te  # noqa: E402
import xsetup  # noqa: E402

cls = te.TestExecutor
orig = cls.__dict__["setUpClass"]
cls.setUpClass()


def state(c):
    out = {}
    for kind in ("tpch", "tpcds"):
        tables = getattr(c, kind + "_tables")
        conn = getattr(c, kind + "_conn")
        out[kind] = [
            (name, t.columns, t.column_range, [[(type(v), v) for v in r] for r in t.rows],
             conn.execute(f"SELECT COUNT(*) FROM {name}").fetchall(), conn.execute(f"DESCRIBE {name}").fetchall())
            for name, t in tables.items()
        ]
    out["sqls"] = (c.tpch_sqls, c.tpcds_sqls, c.cache)
    return out


want = state(cls)
data = {"tpch": {k: v.rows for k, v in cls.tpch_tables.items()}, "tpcds": {k: v.rows for k, v in cls.tpcds_tables.items()}}
for d in data.values():
    for rows in d.values():
        for r in rows:
            assert type(r) is tuple and all(type(v) in (int, float, str, type(None)) for v in r), r
with open(dst, "wb") as fh:
    marshal.dump(data, fh)
cls.tearDownClass()
for a in ("tpch_conn", "tpcds_conn", "tpch_tables", "tpcds_tables", "cache", "tpch_sqls", "tpcds_sqls"):
    delattr(cls, a)
xsetup._DATA = None
assert xsetup.install(root, root, dst), "fast set-up not installed"
assert cls.__dict__["setUpClass"] is not orig
cls.setUpClass()
got = state(cls)
assert got == want, "fast set-up differs from the real one"
cls.tearDownClass()
# and with connections no test opened: tear-down must not fail
cls.setUpClass()
assert cls.tpch_conn.__dict__["_conn"] is None
cls.tearDownClass()
print("xcache", os.path.getsize(dst), "bytes,", sum(len(r) for d in data.values() for r in d.values()), "rows: fast set-up identical")
