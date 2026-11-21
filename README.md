# Blast Radius: change-aware test selection for sqlglot

Entry for the RWX Blast Radius track (Builders Buildathon, Nov 2026). For each
patch against sqlglot `eb11aec`, it prints which tests to run and how to split
them across workers.

## How it decides

1. **AST diff, not text diff.** `astdiff.py` walks the old and new syntax trees
   together and reports the smallest subtrees that differ. Comments and
   formatting change nothing, so they select nothing.
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
4. **Data files → file reads.** SQL fixtures and other files map to the tests
   that opened them (or whose module opened them while importing).
5. **Verify cheap candidates.** Candidates that cost little to run are run on
   a symlink copy of the patched tree in a forked child, and kept only if they
   fail. Expensive ones are kept unverified when that is cheaper. The cost
   model uses each test's measured CPU and its class setup.
6. **Pack.** The final set goes onto 1–8 workers (longest first), choosing
   the worker count that minimises busiest-worker time + 2 s per worker.

When anything is uncertain (unparseable patch, import failure, pytest
configuration change, child error or time budget), it falls back to a safe
superset.

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

## Rebuilding the maps

```sh
# inside the image (python:3.13-slim + requirements.lock + sqlglot at eb11aec on PYTHONPATH)
PYTHONHASHSEED=0 python tracer.py /opt/sqlglot tests.json out/trace 6
TRACE_CALIB=1 PYTHONHASHSEED=0 python tracer.py /opt/sqlglot tests.json out/calib 2
python modlines.py /opt/sqlglot tests.json out/modlines.json
CALIB_DIR=out/calib python build_maps.py /opt/sqlglot tests.json out/trace out/modlines.json maps/maps.json.gz
```

## Third-party code

None beyond the pinned environment in `requirements.lock` (installed from
PyPI) and sqlglot itself (downloaded from its own repository at `eb11aec`).

## License

MIT
