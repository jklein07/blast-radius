"""Write maps/regcov.json.gz: per file and line, the tests that run it only
when the dialect registry starts empty.

tests/test_dialect_entry_points.py clears Dialect._classes in setUp and
tearDown, so any test pytest runs after it in the same process meets an
empty registry and looks its dialects up again through _try_load. Trace the
pinned commit twice, the same way, once plainly and once with the registry
cleared before each test (TRACE_CLEARREG=1), and keep what only the second
run executed.

usage:
  PYTHONHASHSEED=0 python tracer.py /opt/sqlglot tests.json out/trace_nat 6
  TRACE_CLEARREG=1 PYTHONHASHSEED=0 python tracer.py /opt/sqlglot tests.json out/trace_cr 6
  python build_regcov.py out/trace_nat out/trace_cr maps/regcov.json.gz
"""

import gzip
import json
import os
import sys

plain, cleared, dst = sys.argv[1:4]
out: dict = {}
for f in sorted(os.listdir(cleared)):
    if not f[0].isdigit() or not f.endswith(".json"):
        continue
    with open(os.path.join(cleared, f)) as fh:
        b = json.load(fh)
    pa = os.path.join(plain, f)
    if not os.path.exists(pa):
        sys.exit(f"{f}: missing from {plain}")
    with open(pa) as fh:
        a = json.load(fh)
    for fn, lines in b["lines"].items():
        have = set(a["lines"].get(fn, ()))
        for ln in lines:
            if ln not in have:
                out.setdefault(fn, {}).setdefault(str(ln), []).append(b["n"])
for fm in out.values():
    for ln in fm:
        fm[ln].sort()
with gzip.open(dst, "wt") as fh:
    json.dump(out, fh, sort_keys=True, separators=(",", ":"))
print(sum(len(t) for fm in out.values() for t in fm.values()), "pairs in", len(out), "files", file=sys.stderr)
