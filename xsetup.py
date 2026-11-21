"""TestExecutor's class set-up without re-parsing the TPC tables.

`tests/test_executor.py::TestExecutor.setUpClass` reads sixteen gzipped CSV
files (TPC-H and TPC-DS, 2.1 million values) with a pure-Python loop before
any of the class's tests can run: about 1 s of CPU, every time a child
verifies one executor test. The rows it builds depend only on the test
file's own loop, the standard library (csv, gzip, ast.literal_eval) and the
bytes of the CSV files: no sqlglot code runs in that loop. `mkxcache.py`
runs the real set-up once at build time, on the pinned commit, and stores
the rows (`xcache.bin`, marshal).

`install()` swaps in a set-up that does everything the original does, in
the same order, except the parsing: every `Table(columns=..., rows=...)`
call (the patched sqlglot's
`Table`, with the same arguments, so a patch that breaks it still breaks the
set-up), `cache`, and both fixture-pair lists. The two duckdb connections
and their views are opened when a test first uses them (see `_LazyConn`).
It is installed only when
the files that decide the rows are the pinned ones: in the child's tree,
`tests/test_executor.py`, `tests/helpers.py` and every CSV file must be
symlinks to the pinned checkout, and csv, gzip and ast must be the standard
library's. Otherwise the original set-up runs.
"""

from __future__ import annotations

import marshal
import os
import sys

_DATA = None
TEST_REL = "tests/test_executor.py"
DEPS = (TEST_REL, "tests/helpers.py", "tests/__init__.py")
DIRS = {"tpch": "tests/fixtures/optimizer/tpc-h/", "tpcds": "tests/fixtures/optimizer/tpc-ds/"}


def load(path: str):
    global _DATA
    if _DATA is None:
        with open(path, "rb") as fh:
            _DATA = marshal.loads(fh.read())
    return _DATA


def tree_ok(root: str, pristine: str, data: dict) -> bool:
    """In the child's tree: every file the rows depend on is the pinned one."""
    rp = os.path.realpath(pristine)
    rels = list(DEPS)
    for kind, d in DIRS.items():
        rels += [f"{d}{t}.csv.gz" for t in data[kind]]
    for rel in rels:
        if os.path.realpath(os.path.join(root, rel)) != os.path.join(rp, rel):
            return False
    lib = os.path.dirname(os.__file__)
    for name in ("csv", "gzip", "ast"):
        m = sys.modules.get(name)
        if m is None or os.path.dirname(getattr(m, "__file__", "") or "") != lib:
            return False
    return True


class _LazyConn:
    """A duckdb connection with its views, opened at its first use. Creating
    a view makes duckdb read the head of a CSV file: work that involves only
    duckdb and the pinned files, so it gives the same views whenever it
    runs. Only the three TPC tests query these connections; for every other
    test of the class they are never opened."""

    def __init__(self, duckdb):
        self.__dict__["_duckdb"] = duckdb
        self.__dict__["_views"] = []
        self.__dict__["_conn"] = None

    def _open(self):
        conn = self.__dict__["_conn"]
        if conn is None:
            conn = self.__dict__["_duckdb"].connect()
            for sql in self.__dict__["_views"]:
                conn.execute(sql)
            self.__dict__["_conn"] = conn
        return conn

    def close(self):
        conn = self.__dict__["_conn"]
        if conn is not None:
            conn.close()

    def __getattr__(self, name):
        return getattr(self._open(), name)

    def __setattr__(self, name, value):
        setattr(self._open(), name, value)


def fast_setup(te, data: dict):
    """The set-up of tests/test_executor.py at the pinned commit, with the
    rows taken from `data`. Names are read from the test module when it
    runs, as the original does. Each set-up gets its own row lists (Table
    keeps the list it is given, and append/pop change it), as the original's
    fresh parse does."""

    def setUpClass(cls):
        cls.tpch_conn = _LazyConn(te.duckdb)
        cls.tpcds_conn = _LazyConn(te.duckdb)
        cls.tpch_tables = {}
        cls.tpcds_tables = {}

        def setup(conn, directory, table, columns, tables, rows):
            file_name = f"{directory}{table}.csv.gz"

            conn._views.append(
                f"""
                CREATE VIEW {table} AS
                SELECT *
                FROM READ_CSV('{file_name}', delim='|', header=True, columns={columns})
                """
            )

            tables[table] = rows
            tables[table] = te.Table(columns=columns, rows=rows)

        for table, columns in te.TPCH_SCHEMA.items():
            setup(cls.tpch_conn, te.DIR_TPCH, table, columns, cls.tpch_tables, list(data["tpch"][table]))

        for table, columns in te.TPCDS_SCHEMA.items():
            setup(cls.tpcds_conn, te.DIR_TPCDS, table, columns, cls.tpcds_tables, list(data["tpcds"][table]))

        cls.cache = {}
        cls.tpch_sqls = list(te.load_sql_fixture_pairs("optimizer/tpc-h/tpc-h.sql"))
        cls.tpcds_sqls = list(te.load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))

    return setUpClass


def install(root: str, pristine: str, path: str) -> bool:
    """Swap the set-up of an imported tests.test_executor. False: left alone."""
    te = sys.modules.get("tests.test_executor")
    cls = getattr(te, "TestExecutor", None) if te is not None else None
    if cls is None or not isinstance(cls.__dict__.get("setUpClass"), classmethod):
        return False
    if os.path.realpath(getattr(te, "__file__", "") or "") != os.path.join(os.path.realpath(pristine), TEST_REL):
        return False
    try:
        data = load(path)
    except Exception:  # noqa: BLE001
        return False
    if set(data["tpch"]) != set(te.TPCH_SCHEMA) or set(data["tpcds"]) != set(te.TPCDS_SCHEMA):
        return False
    if not tree_ok(root, pristine, data):
        return False
    cls.setUpClass = classmethod(fast_setup(te, data))
    return True
