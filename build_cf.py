"""Build maps/cf.json.gz from cftrace.py output.

For each fixture-looping optimizer test: its fixture loads, which pairs
execute each sqlglot line (the union of the solo and in-context traces, as
interned bitmasks over the test's pairs in load order), the sqlglot lines it
runs outside its pairs, per-pair CPU, and the lines setUpClass runs.

usage: python build_cf.py CFTRACE_DIR maps/cf.json.gz
"""

import gzip
import json
import os
import sys

src, dst = sys.argv[1], sys.argv[2]
idx = json.load(open(os.path.join(src, "index.json")))


def rec(*job):
    with open(os.path.join(src, "_".join(str(x) for x in job) + ".json")) as fh:
        return json.load(fh)


def sg(d):
    return {f: v for f, v in d.items() if f.startswith("sqlglot/")}


out = {"setup": sg(rec("setup")["lines"]), "tests": {}}
for m, loads in sorted(idx["loads"].items()):
    off = [0]
    for _, n in loads:
        off.append(off[-1] + n)
    nq = off[-1]
    fx = rec("fixed", m)
    ctx = rec("ctx", m)
    if fx["failed"] or ctx["failed"]:
        raise SystemExit(f"{m}: fails on the pinned commit while tracing: {fx.get('msgs') or ctx.get('msgs')}")
    masks: dict = {}
    outside: dict = {}

    def add_out(d):
        for f, ls in sg(d).items():
            outside.setdefault(f, set()).update(ls)

    def add_in(k, i, d):
        bit = 1 << (off[k] + i)
        for f, ls in sg(d).items():
            fm = masks.setdefault(f, {})
            for ln in ls:
                fm[ln] = fm.get(ln, 0) | bit

    add_out(fx["outside"])
    add_out(ctx["outside"])
    for k, i, d in ctx["inside"]:
        add_in(k, i, d)
    # per-pair CPU: its turns in the untraced whole-test run; a run of any
    # pairs also pays a one-off overhead (c1, from the solo runs) and the
    # first-time work of its most expensive pair (cx)
    pc = [None] * nq
    for k, i, v in ctx.get("pc", ()):
        pc[off[k] + i] = v
    avg = max(0.0, ctx["warm"] - fx["warm"]) / max(1, nq)
    pc = [round(avg if v is None else v, 6) for v in pc]
    solo = [0.0] * nq
    cx = [0.0] * nq
    for k, (_, n) in enumerate(loads):
        for i in range(n):
            r = rec("q", m, k, i)
            if r["failed"]:
                raise SystemExit(f"{m} pair {k}/{i} fails alone: {r.get('msgs')}")
            add_out(r["outside"])
            for kk, ii, d in r["inside"]:
                add_in(kk, ii, d)
            q = off[k] + i
            solo[q] = max(0.0, r["warm"] - fx["warm"] - pc[q])
            cx[q] = round(max(0.0, (r["cold"] - fx["cold"]) - (r["warm"] - fx["warm"])), 5)
    ids: dict = {}
    mlist: list = []
    lines: dict = {}
    for f, fm in sorted(masks.items()):
        lf = lines[f] = {}
        for ln, mk in sorted(fm.items()):
            h = format(mk, "x")
            j = ids.get(h)
            if j is None:
                j = ids[h] = len(mlist)
                mlist.append(h)
            lf[str(ln)] = j
    c1 = sorted(solo)[len(solo) // 2] if solo else 0.0
    full = fx["warm"] + c1 + sum(pc) + max(cx, default=0.0)
    out["tests"][m] = {
        "loads": loads,
        "nq": nq,
        "fixed": round(fx["warm"], 5),
        "c1": round(c1, 5),
        "pc": pc,
        "cx": cx,
        "full": round(full, 5),
        "masks": mlist,
        "lines": lines,
        "outside": {f: sorted(v) for f, v in sorted(outside.items())},
    }
    print(m, "pairs", nq, "masks", len(mlist), "lines", sum(len(v) for v in lines.values()),
          "outside", sum(len(v) for v in outside.values()), "fixed", round(fx["warm"], 3), "c1", round(c1, 3), "pairs cpu", round(sum(pc), 3), "full", round(full, 3), file=sys.stderr)
with gzip.open(dst, "wt") as fh:
    json.dump(out, fh, separators=(",", ":"))
print("wrote", dst, os.path.getsize(dst), file=sys.stderr)
