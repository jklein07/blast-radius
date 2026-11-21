"""Build maps/split.json.gz from qtrace.py output.

For each query-looping test: which queries execute each line (a bitmask per
line), each query's CPU cost, and the lines setUpClass runs.

usage: python build_split.py qtrace.json maps/split.json.gz
"""

import gzip
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
d = json.load(open(src))
out = {"setup": d["setup"], "setup_cpu": d["setup_cpu"], "tests": {}}
for tid, t in d["tests"].items():
    qs = t["q"]
    if any(q["failed"] for q in qs):
        raise SystemExit(f"{tid}: a query fails on the pinned commit")
    masks: dict = {}
    for i, q in enumerate(qs):
        for f, lines in q["lines"].items():
            fm = masks.setdefault(f, {})
            for ln in lines:
                fm[ln] = fm.get(ln, 0) | (1 << i)
    fixed = min(q["fixed"] for q in qs)
    out["tests"][tid] = {
        "attr": t["attr"],
        "nq": len(qs),
        "warm": [round(max(0.0, q["warm"] - fixed), 4) for q in qs],
        "cold": [round(max(0.0, q["cold"] - fixed), 4) for q in qs],
        "fixed": round(fixed, 4),
        "lines": {f: {str(ln): format(m, "x") for ln, m in sorted(fm.items())} for f, fm in masks.items()},
    }
with gzip.open(dst, "wt") as fh:
    json.dump(out, fh, separators=(",", ":"))
print({tid: (v["nq"], round(sum(v["warm"]), 2), sum(len(x) for x in v["lines"].values())) for tid, v in out["tests"].items()})
