"""Build maps/wm.json.gz from wmtrace.py output.

For each comparison site: its operator and, per alternative operator, the
set of tests in which that alternative ever gave a different result.
Tests whose instrumented run cannot be trusted (failed, crashed, spawned a
process the instrumentation cannot see into) are listed as incomplete; the
selector falls back to line coverage for them.

Several WM_DIRs (comma-separated) are merged by union: a swap counts for a
test if any run saw it change a result there.

usage: python build_wm.py tests.json WM_DIR[,WM_DIR...] CALIB_DIR maps/wm.json.gz
"""

import gzip
import json
import os
import sys

tests_json, wm_dir, calib_dir, dst = sys.argv[1:5]
tests = json.load(open(tests_json))["tests"]
ids = [t["id"] for t in tests]
num2local = {t["number"]: i for i, t in enumerate(tests)}
BITS = {1: "Is", 2: "IsNot", 4: "Eq", 8: "NotEq", 16: "Lt", 32: "LtE", 64: "Gt", 128: "GtE"}

dirs = wm_dir.split(",")
metas = [json.load(open(os.path.join(d, "_sites.json"))) for d in dirs]
meta = metas[0]
sites = meta["sites"]  # sid -> [file, key, op]
for m in metas[1:]:
    if m["sites"] != sites:
        raise SystemExit("runs instrumented different sites")
    for k, v in m["import_mask"].items():
        meta["import_mask"][k] = meta["import_mask"].get(k, 0) | v
    for f, mm in m["module_masks"].items():
        dst_ = meta["module_masks"].setdefault(f, {})
        for k, v in mm.items():
            dst_[k] = dst_.get(k, 0) | v
diff: dict[int, dict[str, set]] = {}
incomplete = set()
for d in dirs:
    for t in tests:
        li = num2local[t["number"]]
        p = os.path.join(d, f"{t['number']}.json")
        if not os.path.exists(p):
            incomplete.add(li)
            continue
        r = json.load(open(p))
        cp = os.path.join(calib_dir, f"{t['number']}.json")
        base_failed = json.load(open(cp)).get("failed") if os.path.exists(cp) else None
        if r["spawned"] or r["failed"] != bool(base_failed):
            incomplete.add(li)
        for sid, mask in r["mask"].items():
            dd = diff.setdefault(int(sid), {})
            for bit, name in BITS.items():
                if mask & bit:
                    dd.setdefault(name, set()).add(li)

sets: list = []
index: dict = {}


def setid(s) -> int:
    fs = frozenset(s)
    i = index.get(fs)
    if i is None:
        i = index[fs] = len(sets)
        sets.append(sorted(fs))
    return i


out_sites: dict = {}
for sid, (rel, key, op) in enumerate(sites):
    alts = {name: setid(s) for name, s in diff.get(sid, {}).items()}
    out_sites.setdefault(rel, {})[key] = [op, alts]


def masks_by_site(m):
    out: dict = {}
    for sid, mask in m.items():
        rel, key, _op = sites[int(sid)]
        out.setdefault(rel, {})[key] = [BITS[b] for b in BITS if mask & b]
    return out


res = {
    "tests": ids,
    "sets": sets,
    "sites": out_sites,
    "incomplete": sorted(incomplete),
    "import": masks_by_site(meta["import_mask"]),
    "modules": {f: masks_by_site(m) for f, m in meta["module_masks"].items() if m},
}
with gzip.open(dst, "wt") as fh:
    json.dump(res, fh, separators=(",", ":"))
print("sites", len(sites), "with diffs", len(diff), "sets", len(sets), "incomplete", len(incomplete), file=sys.stderr)
