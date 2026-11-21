"""Aggregate tracer.py output into maps/maps.json.gz (build-time tool).

usage: python build_maps.py SQLGLOT_ROOT tests.json TRACE_DIR MODLINES_JSON OUT
"""

from __future__ import annotations

import ast
import glob
import gzip
import json
import os
import sys

root, tests_json, trace_dir, modlines_json, out = sys.argv[1:6]
tests = json.load(open(tests_json))["tests"]
ids = [t["id"] for t in tests]
num2local = {t["number"]: i for i, t in enumerate(tests)}

sets: list[list[int]] = []
set_index: dict[frozenset, int] = {}


def sid(s) -> int:
    fs = frozenset(s)
    i = set_index.get(fs)
    if i is None:
        i = set_index[fs] = len(sets)
        sets.append(sorted(fs))
    return i


cov: dict[str, dict[int, set]] = {}
lk: dict[str, set] = {}
classuse: dict[str, set] = {}
reads: dict[str, set] = {}
failed = []
cpu = {}
spawners = []
missing = []
for t in tests:
    p = os.path.join(trace_dir, f"{t['number']}.json")
    if not os.path.exists(p):
        missing.append(t["number"])
        continue
    r = json.load(open(p))
    li = num2local[r["n"]]
    if r["failed"]:
        failed.append(li)
    if r.get("spawned"):
        spawners.append(li)
    cpu[li] = r["cpu"]
    for f, lines in r["lines"].items():
        d = cov.setdefault(f, {})
        for ln in lines:
            d.setdefault(ln, set()).add(li)
    for k in r["lk"]:
        lk.setdefault(k, set()).add(li)
        classuse.setdefault(k.rpartition("|")[0], set()).add(li)
    for path in r["reads"]:
        reads.setdefault(path, set()).add(li)

print("missing traces", missing, "failed under trace", [ids[i] for i in failed], "spawners", [ids[i] for i in spawners], file=sys.stderr)

dirreads = {p: s for p, s in reads.items() if os.path.isdir(os.path.join(root, p))}
filereads = {p: s for p, s in reads.items() if not os.path.isdir(os.path.join(root, p))}

# ---------------------------------------------------------------- static maps
def py_files(sub):
    for d, _, fs in os.walk(os.path.join(root, sub)):
        for f in fs:
            if f.endswith(".py"):
                yield os.path.relpath(os.path.join(d, f), root)


def modname(rel):
    m = rel[:-3].replace("/", ".")
    return m[: -len(".__init__")] if m.endswith(".__init__") else m


files = sorted(list(py_files("sqlglot")) + list(py_files("tests")))
trees = {}
for f in files:
    try:
        trees[f] = ast.parse(open(os.path.join(root, f), encoding="utf-8").read())
    except SyntaxError:
        pass

# defs: qualname -> span
defs: dict[str, dict[str, list[int]]] = {}
for f, tree in trees.items():
    d = defs.setdefault(f, {})

    def walk(body, prefix):
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                q = prefix + n.name
                d[q] = [min([n.lineno] + [x.lineno for x in n.decorator_list]), n.end_lineno]
                walk(n.body, q + ".")
            elif isinstance(n, (ast.If, ast.Try, ast.With)):
                walk(n.body, prefix)
                walk(getattr(n, "orelse", []), prefix)

    walk(tree.body, "")

# module-level names of each sqlglot module
mod_names: dict[str, set] = {}
for f, tree in trees.items():
    if not f.startswith("sqlglot/"):
        continue
    names = set()
    for n in tree.body:
        for t in ast.walk(n) if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else [n]:
            if isinstance(t, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(t.name)
            elif isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store):
                names.add(t.id)
            elif isinstance(t, (ast.Import, ast.ImportFrom)):
                for al in t.names:
                    names.add((al.asname or al.name).split(".")[0])
    mod_names[modname(f)] = names

# references: for module M name G -> lines that load G from M
attr_lines: dict[str, list] = {}
name_lines: dict[str, dict[str, list]] = {}  # file -> name -> lines
from_imports: dict[str, list] = {}  # (module, name) -> [(file, alias)]
star_imports: dict[str, list] = {}  # module -> [file]
for f, tree in trees.items():
    nl = name_lines.setdefault(f, {})
    me = modname(f)
    pkg = me if f.endswith("__init__.py") else me.rpartition(".")[0]
    for t in ast.walk(tree):
        if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Load):
            nl.setdefault(t.id, []).append(t.lineno)
        elif isinstance(t, ast.Attribute):
            attr_lines.setdefault(t.attr, []).append((f, t.lineno))
        elif isinstance(t, ast.ImportFrom):
            base = t.module or ""
            if t.level:
                parts = pkg.split(".")
                up = parts[: len(parts) - (t.level - 1)] if t.level > 1 else parts
                base = ".".join(up + ([t.module] if t.module else []))
            for al in t.names:
                if al.name == "*":
                    star_imports.setdefault(base, []).append(f)
                else:
                    from_imports.setdefault(f"{base}|{al.name}", []).append((f, al.asname or al.name))

names_index: dict[str, dict[str, list]] = {}
for m, names in mod_names.items():
    rel = None
    for cand in (m.replace(".", "/") + ".py", m.replace(".", "/") + "/__init__.py"):
        if cand in trees:
            rel = cand
    out_m = names_index.setdefault(m, {})
    for g in names:
        refs = set()
        if rel:
            for ln in name_lines.get(rel, {}).get(g, []):
                refs.add((rel, ln))
        for f, alias in from_imports.get(f"{m}|{g}", []):
            for ln in name_lines.get(f, {}).get(alias, []):
                refs.add((f, ln))
        for f in star_imports.get(m, []):
            for ln in name_lines.get(f, {}).get(g, []):
                refs.add((f, ln))
            # re-exported through a package: users access pkg.G
        for f, ln in attr_lines.get(g, []):
            refs.add((f, ln))
        out_m[g] = sorted(refs)

# test import graph
test_imports: dict[str, set] = {}
for f, tree in trees.items():
    if not f.startswith("tests/"):
        continue
    deps = set()
    me = modname(f)
    pkg = me.rpartition(".")[0]
    for t in ast.walk(tree):
        mods = []
        if isinstance(t, ast.Import):
            mods = [a.name for a in t.names]
        elif isinstance(t, ast.ImportFrom):
            base = t.module or ""
            if t.level:
                parts = pkg.split(".")
                up = parts[: len(parts) - (t.level - 1)] if t.level > 1 else parts
                base = ".".join(up + ([t.module] if t.module else []))
            mods = [base] + [base + "." + a.name for a in t.names]
        for m in mods:
            if m.startswith("tests"):
                for cand in (m.replace(".", "/") + ".py", m.replace(".", "/") + "/__init__.py"):
                    if cand in trees:
                        deps.add(cand)
    test_imports[f] = deps
importers: dict[str, set] = {}
for f in test_imports:
    # transitive closure
    stack = list(test_imports[f])
    seen = set()
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        stack.extend(test_imports.get(x, ()))
    for x in seen:
        importers.setdefault(x, set()).add(f)

# tests we should not trust a verification run for
noverify = set(failed)
for i, tid in enumerate(ids):
    low = tid.lower()
    if any(w in low for w in ("memory", "leak", "retain", "_mp_", "tpcds", "tpch", "deadlock", "timeout", "perf", "benchmark")):
        noverify.add(i)

vcpu = {}
calib_dir = os.environ.get("CALIB_DIR")
if calib_dir:
    for t in tests:
        p = os.path.join(calib_dir, f"{t['number']}.json")
        if os.path.exists(p):
            r = json.load(open(p))
            vcpu[str(num2local[t["number"]])] = r["cpu"]
            if r.get("spawned"):
                spawners.append(num2local[t["number"]])
import_lines = json.load(open(os.path.join(trace_dir, "_import_lines.json")))
module_reads = json.load(open(os.path.join(trace_dir, "_module_reads.json")))
module_lines = json.load(open(modlines_json)) if os.path.exists(modlines_json) else {}

maps = {
    "tests": ids,
    "sets": sets,
    "cov": {f: {str(ln): sid(s) for ln, s in d.items()} for f, d in cov.items()},
    "lk": {k: sid(s) for k, s in lk.items()},
    "classuse": {k: sid(s) for k, s in classuse.items()},
    "reads": {k: sid(s) for k, s in filereads.items()},
    "dirreads": {k: sid(s) for k, s in dirreads.items()},
    "module_reads": module_reads,
    "module_lines": module_lines,
    "import_lines": import_lines,
    "names": names_index,
    "defs": defs,
    "test_importers": {k: sorted(v) for k, v in importers.items()},
    "noverify": sorted(noverify),
    "cpu": {str(k): v for k, v in cpu.items()},
    "spawners": sorted(spawners),
    "vcpu": vcpu,
}
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
raw = json.dumps(maps, separators=(",", ":"))
with gzip.open(out, "wt", compresslevel=9) as fh:
    fh.write(raw)
print(f"sets {len(sets)} cov files {len(cov)} lk {len(lk)} raw {len(raw)/1e6:.1f}MB gz {os.path.getsize(out)/1e6:.2f}MB "
      f"noverify {len(noverify)}", file=sys.stderr)
