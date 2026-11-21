# Blast Radius: change-aware test selection for sqlglot

Entry for the RWX Blast Radius track (Builders Buildathon, Nov 2026). For each
patch against sqlglot `eb11aec`, it prints which tests to run and how to split
them across workers.

## How it decides

1. **AST diff, not text diff.** `astdiff.py` walks the old and new syntax trees
   together and reports the smallest subtrees that differ. Comments and
   formatting change nothing, so they select nothing. Three other edits are
   proved to change nothing before they are dropped:
   - *Renamed locals.* A function whose only difference is a consistent,
     one-to-one renaming of its local variables counts as unchanged only if
     scope analysis shows every renamed name is a non-parameter local of
     that function, nothing in it can see its frame (`locals`, `eval`,
     `vars`, ...), and both versions compile to identical bytecode apart
     from the names.
   - *Reordered imports.* A contiguous run of plain stdlib or third-party
     imports, each binding a different name, may be permuted. Imports of
     sqlglot or the tests, relative, star and `__future__` imports, never
     count as reordered.
   - *Moved definitions.* Undecorated functions with constant defaults and no
     annotations may swap places within a run of such definitions between
     identical statements, in a module or class body.
2. **Runtime changes → line coverage.** A difference inside a function or
   lambda body only matters to tests that execute it. `tracer.py` ran every
   test of the pinned commit in its own forked process with `sys.monitoring`
   and recorded the lines each one executed (process-pool work included; see
   `inline.py`).
3. **Import-time changes → measured state diff.** Class bodies, module
   constants, def headers and decorators act at import time, and sqlglot's
   metaclasses derive many structures from them (tries, inverse mappings,
   generator transforms). Instead of guessing, a child imports the patched
   sqlglot and fingerprints every class attribute and module global
   (`fingerprint.py`). It compares them with the pristine fingerprint taken at
   build time. Each changed `(class, attribute)` maps to the tests that read
   that attribute through the class or a subclass. The tracer records those
   reads by wrapping every class attribute in a recording descriptor. Each
   changed global maps to the tests that execute a line referring to it.
4. **Operator swaps → value profile.** When the only difference in a
   comparison is its operator (`is` for `==`, `<` for `<=`, `is not` for `!=`,
   ...), executing the line is not enough to change anything: the new
   operator must also give a different result on the same operands. Until
   that first different result, the patched program runs exactly like the
   original. `wmtrace.py` imports sqlglot with every one-operator comparison
   rewritten to also evaluate the alternative operators. It runs each test in
   its own process and records, per comparison site, which alternatives ever
   gave a different result (or raised differently) in that test. A swap is
   charged only to those tests. Line coverage still applies to tests whose
   instrumented run cannot be trusted, and to sites the profile does not
   cover. The profile is the union of a native run and an amd64 run.
5. **Query-level verification for the TPC tests.** `test_execute_tpcds`
   and its TPC-H siblings loop over a fixture's queries. `qtrace.py` records
   which queries execute each line. When a change reaches only some of them,
   the test can be verified on those queries alone. The optimizer tests that
   check a rule against a fixture file get the same treatment per fixture
   pair (`cftrace.py`, `cfsel.py`).
6. **Data files → file reads.** SQL fixtures and other files map to the tests
   that opened them (or whose module opened them while importing).
7. **Verify cheap candidates.** Candidates that cost little to run are run on
   a symlink copy of the patched tree in a forked child, and kept only if they
   fail. Expensive ones are kept unverified when that is cheaper. The cost
   model uses each test's measured CPU and its class setup. Every verified
   test runs under its own timer (three times its measured cost plus a
   margin; 15 s for class setup); a test that runs out of time is kept,
   because a patch that makes a test hang makes it fail.
8. **Pack.** The final set goes onto 1–8 workers (longest first), choosing
   the worker count that minimises busiest-worker time + 2 s per worker.

When anything is uncertain (unparseable patch, import failure, pytest
configuration change, child error or time budget), it falls back to a safe
superset. One such case is a test-side module (a test file, a package
`__init__`, a helper the tests import, or a new file) whose import-time
statements change in a way that could touch shared state (patching a sqlglot
class, defining a dialect subclass, setting an environment variable) or
change pytest's verdict (an xfail or timeout mark). pytest imports every
test module while collecting, in every worker, so shared state can break
tests in any file. Imports, plain definitions,
test classes, skip and `mock.patch` decorators and call-free constants are
known not to.

No answers are worked out ahead of time: there is no table from patches or
kinds of edit to failing tests. The maps in `maps/` are built only from the
pinned commit (coverage, attribute reads, file reads, imports, name
references).

## Files

| File | Role |
| --- | --- |
| `selector.py` | Runtime entry point |
| `astdiff.py` | Semantic AST diff |
| `fingerprint.py` | Import-time state fingerprints and diff |
| `patchapply.py` | Unified-diff parser and applier (stdlib only) |
| `inline.py` | In-process ProcessPoolExecutor with pickle round trips |
| `tracer.py`, `modlines.py`, `build_maps.py`, `mksnap.py` | Build-time tools that produce `maps/maps.json.gz` and `snap.json` from the pinned commit |
| `wmtrace.py`, `build_wm.py` | Build-time: comparison-swap profile, `maps/wm.json.gz` |
| `qtrace.py`, `build_split.py` | Build-time: per-query coverage of the TPC tests, `maps/split.json.gz` |
| `cftrace.py`, `build_cf.py`, `cfsel.py` | Per-pair coverage of the fixture-driven optimizer tests, `maps/cf.json.gz`; `cfsel.py` runs such a test on chosen pairs |

## Rebuilding the maps

```sh
# inside the image (python:3.13-slim + requirements.lock + sqlglot at eb11aec on PYTHONPATH)
PYTHONHASHSEED=0 python tracer.py /opt/sqlglot tests.json out/trace 6
TRACE_CALIB=1 PYTHONHASHSEED=0 python tracer.py /opt/sqlglot tests.json out/calib 2
python modlines.py /opt/sqlglot tests.json out/modlines.json
CALIB_DIR=out/calib python build_maps.py /opt/sqlglot tests.json out/trace out/modlines.json maps/maps.json.gz
PYTHONHASHSEED=0 python wmtrace.py /opt/sqlglot tests.json out/wm 8
python build_wm.py tests.json out/wm out/calib maps/wm.json.gz
PYTHONHASHSEED=0 python qtrace.py /opt/sqlglot out/qtrace.json 4
python build_split.py out/qtrace.json maps/split.json.gz
PYTHONHASHSEED=0 python cftrace.py /opt/sqlglot out/cftrace 10
python build_cf.py out/cftrace maps/cf.json.gz
```

## Third-party code

None beyond the pinned environment in `requirements.lock` (installed from
PyPI) and sqlglot itself (downloaded from its own repository at `eb11aec`).

## License

MIT
