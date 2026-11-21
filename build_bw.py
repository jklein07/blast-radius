"""Build maps/bw.json.gz from bwtrace.py output.

For each `and`/`or` chain: its operator, operand count, whether only its
truth is used, and per single swap (the i-th operator of the chain) the set
of tests in which that swap ever made a difference. Tests whose instrumented
run cannot be trusted (failed differently from the plain run, crashed,
spawned a process) are listed as incomplete; the selector falls back to line
coverage for them.

Several BW_DIRs (comma-separated) are merged by union.

usage: python build_bw.py tests.json BW_DIR[,BW_DIR...] CALIB_DIR maps/bw.json.gz
"""

import gzip
import json
import os
import sys

tests_json, bw_dir, calib_dir, dst = sys.argv[1:5]
tests = json.load(open(tests_json))["tests"]
ids = [t["id"] for t in tests]
num2local = {t["number"]: i for i, t in enumerate(tests)}

dirs = bw_dir.split(",")
metas = [json.load(open(os.path.join(d, "_sites.json"))) for d in dirs]
meta = metas[0]
sites = meta["sites"]  # sid -> [file, key, op, n, truth_only]
for m in metas[1:]:
    if m["sites"] != sites:
        raise SystemExit("runs instrumented different sites")
    for k, v in m["import_mask"].items():
        meta["import_mask"][k] = meta["import_mask"].get(k, 0) | v
    for f, mm in m["module_masks"].items():
        dst_ = meta["module_masks"].setdefault(f, {})
        for k, v in mm.items():
            dst_[k] = dst_.get(k, 0) | v
diff: dict[int, dict[int, set]] = {}
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
            n = sites[int(sid)][3]
            dd = diff.setdefault(int(sid), {})
            for i in range(n - 1):
                if mask >> i & 1:
                    dd.setdefault(i, set()).add(li)

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
for sid, (rel, key, op, n, truth_only) in enumerate(sites):
    flips = {str(i): setid(s) for i, s in diff.get(sid, {}).items()}
    out_sites.setdefault(rel, {})[key] = [op, n, bool(truth_only), flips]


def masks_by_site(m):
    out: dict = {}
    for sid, mask in m.items():
        rel, key, _op, n, _t = sites[int(sid)]
        out.setdefault(rel, {})[key] = [i for i in range(n - 1) if mask >> i & 1]
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
