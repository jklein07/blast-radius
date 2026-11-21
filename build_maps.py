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


kl: dict[str, dict[str, set]] = {}  # "class|attr" -> dict key id -> tests that looked it up
kla: dict[str, set] = {}  # "class|attr" -> tests that read the dict as a whole
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
    for tag, kids in r.get("kl", {}).items():
        d = kl.setdefault(tag, {})
        for kid in kids:
            d.setdefault(kid, set()).add(li)
    for tag in r.get("kla", ()):
        kla.setdefault(tag, set()).add(li)

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

# module attribute reads measured by tracer.py (TRACE_MODATTR): for the
# modules it recorded, an attribute access `x.G` somewhere no longer counts as
# a reference to M's global G; the tests that really read module.G do.
ma_dir = os.environ.get("MODATTR_DIR")
ma_import = None
ma_mods: set = set()
if ma_dir:
    ma_missing = [t["number"] for t in tests if not os.path.exists(os.path.join(ma_dir, f"{t['number']}.json"))]
    if ma_missing:
        raise SystemExit(f"module attribute trace incomplete: {len(ma_missing)} tests missing")
    ma_import = json.load(open(os.path.join(ma_dir, "_ma_import.json")))
    ma_mods = set(ma_import["modules"])

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
        if m not in ma_mods:
            for f, ln in attr_lines.get(g, []):
                refs.add((f, ln))
        out_m[g] = sorted(refs)

# every import statement of a sqlglot module or name: "mod" / "mod|name" ->
# [file, first line, last line, 1 if it runs at import time (module or class
# level) else 0]. When a module or a name disappears, the tests that run such
# a statement fail (function-level), or the whole test file does (test module
# level); the snapshot sees only sqlglot's own import-time imports.
import_sites: dict[str, list] = {}
# packages that load names lazily (PEP 562 __getattr__): name -> (module, attr),
# read from the pristine checkout itself
lazy_attrs: dict[str, dict[str, tuple[str, str]]] = {}
_saved_path = list(sys.path)
try:
    import importlib
    import subprocess

    _code = (
        "import json, sys; sys.path.insert(0, sys.argv[1]); import importlib; out = {}\n"
        "d = importlib.import_module('sqlglot.dialects')\n"
        "out['sqlglot.dialects'] = {n: ['sqlglot.dialects.' + m, n] for n, m in getattr(d, 'MODULE_BY_ATTRIBUTE', {}).items()}\n"
        "o = importlib.import_module('sqlglot.optimizer')\n"
        "out['sqlglot.optimizer'] = {n: list(v) for n, v in getattr(o, '_LAZY_ATTRS', {}).items()}\n"
        "print(json.dumps(out))\n"
    )
    _r = subprocess.run([sys.executable, "-c", _code, root], capture_output=True, text=True, timeout=120,
                        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    lazy_attrs = {k: {n: tuple(v) for n, v in d.items()} for k, d in json.loads(_r.stdout.strip().splitlines()[-1]).items()}
except Exception as e:  # noqa: BLE001
    raise SystemExit(f"cannot read the lazy-import maps: {e!r}")
finally:
    sys.path[:] = _saved_path
print("lazy attrs", {k: len(v) for k, v in lazy_attrs.items()}, file=sys.stderr)
for f, tree in trees.items():
    if not (f.startswith("sqlglot/") or f.startswith("tests/")):
        continue
    me = modname(f)
    pkg = me if f.endswith("__init__.py") else me.rpartition(".")[0]

    def _site(key, n, top):
        if key.startswith("sqlglot"):
            import_sites.setdefault(key, []).append([f, n.lineno, n.end_lineno or n.lineno, 1 if top else 0])

    def _visit(node, top):
        for n in ast.iter_child_nodes(node):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                _visit(n, False)
                continue
            if isinstance(n, ast.Import):
                for a in n.names:
                    parts = a.name.split(".")
                    for i in range(1, len(parts) + 1):
                        _site(".".join(parts[:i]), n, top)
            elif isinstance(n, ast.ImportFrom):
                base = n.module or ""
                if n.level:
                    parts = pkg.split(".")
                    up = parts[: len(parts) - (n.level - 1)] if n.level > 1 else parts
                    base = ".".join(up + ([n.module] if n.module else []))
                parts = base.split(".")
                for i in range(1, len(parts) + 1):
                    _site(".".join(parts[:i]), n, top)
                for a in n.names:
                    if a.name != "*":
                        _site(f"{base}|{a.name}", n, top)
                        _site(f"{base}.{a.name}", n, top)  # when the name is a submodule
                        lz = lazy_attrs.get(base, {}).get(a.name)
                        if lz:  # imported through the package's __getattr__
                            lp = lz[0].split(".")
                            for i in range(1, len(lp) + 1):
                                _site(".".join(lp[:i]), n, top)
                            _site(f"{lz[0]}|{lz[1]}", n, top)
            _visit(n, top)

    _visit(tree, True)

# import-time module graph of sqlglot: module -> sqlglot modules it imports
# at module or class level (function-level imports run later, not at import)
def _toplevel_imports(tree, me, is_pkg):
    pkg = me if is_pkg else me.rpartition(".")[0]
    out = set()

    def visit(body):
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(n, ast.Import):
                for a in n.names:
                    out.add(a.name)
            elif isinstance(n, ast.ImportFrom):
                base = n.module or ""
                if n.level:
                    parts = pkg.split(".")
                    up = parts[: len(parts) - (n.level - 1)] if n.level > 1 else parts
                    base = ".".join(up + ([n.module] if n.module else []))
                out.add(base)
                for a in n.names:
                    out.add(base + "." + a.name)
            for f in ("body", "orelse", "finalbody", "handlers"):
                sub = getattr(n, f, None)
                if isinstance(sub, list) and not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    visit([x for x in sub if isinstance(x, ast.AST)])

    visit(tree.body)
    return out


all_mods = {modname(f) for f in trees if f.startswith("sqlglot/")}
mod_imports = {}
for f, tree in trees.items():
    if not f.startswith("sqlglot/"):
        continue
    me = modname(f)
    deps = set()
    for m in _toplevel_imports(tree, me, f.endswith("__init__.py")):
        # a module import also imports its parent packages
        parts = m.split(".")
        for i in range(1, len(parts) + 1):
            cand = ".".join(parts[:i])
            if cand in all_mods and cand != me:
                deps.add(cand)
    mod_imports[me] = sorted(deps)

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

# module attribute readers, by "module|name": the tests that read module.name
# (or from-imported it at run time), and the readers of the same name in every
# module that re-exports it by a from-import or a star import (it is the same
# object there); "module|__dict__" readers (vars(), ...) count for every name
modattr: dict[str, set] = {}
modattr_mod: dict[str, set] = {}
if ma_import is not None:
    files_local: dict[str, list] = {}
    for t in tests:
        files_local.setdefault(t["file"], []).append(num2local[t["number"]])
    raw_ma: dict[str, set] = {}

    def _ma_add(k, users):
        raw_ma.setdefault(k, set()).update(users)
        modattr_mod.setdefault(k.partition("|")[0], set()).update(users)

    for t in tests:
        r = json.load(open(os.path.join(ma_dir, f"{t['number']}.json")))
        li = num2local[r["n"]]
        for k in r["ma"]:
            _ma_add(k, (li,))
    for f, keys in ma_import["by_import"].items():
        users = files_local.get(f, ())
        for k in keys:
            _ma_add(k, users)
    for f, keys in ma_import["by_frame"].items():
        users = set(files_local.get(f, ()))
        for g in importers.get(f, ()):
            users.update(files_local.get(g, ()))
        for k in keys:
            _ma_add(k, users)
    # re-exports: "M|G" -> modules P binding the same G (from M import G / from M import *)
    reexp: dict[str, set] = {}
    for key, sites in from_imports.items():
        m_, _, g_ = key.partition("|")
        for f, alias in sites:
            if f.startswith("sqlglot/") and alias == g_:
                reexp.setdefault(key, set()).add(modname(f))
    star_by: dict[str, set] = {}
    for m_, fs in star_imports.items():
        for f in fs:
            if f.startswith("sqlglot/"):
                star_by.setdefault(m_, set()).add(modname(f))
    names_seen = {k.partition("|")[2] for k in raw_ma}
    for m_ in mod_names:
        for g_ in mod_names[m_] & names_seen:
            todo, seen_m = [m_], set()
            while todo:
                x = todo.pop()
                if x in seen_m:
                    continue
                seen_m.add(x)
                todo.extend(reexp.get(f"{x}|{g_}", ()))
                todo.extend(star_by.get(x, ()))
            users = set()
            for x in seen_m:
                users |= raw_ma.get(f"{x}|{g_}", set())
                users |= raw_ma.get(f"{x}|__dict__", set())
            if users:
                modattr[f"{m_}|{g_}"] = users
    for k, v in raw_ma.items():  # __dict__, and names a module gets lazily (PEP 562) or at run time
        modattr.setdefault(k, v)
    print("module attribute readers:", len(modattr), "keys,", len(modattr_mod), "modules,",
          sum(len(v) for k, v in modattr.items() if k.endswith("|__dict__")), "module __dict__ reads", file=sys.stderr)

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
vcpu_alt = {}  # a second calibration (another machine), selected at run time by BR_CALIB=alt
calib_alt = os.environ.get("CALIB_ALT_DIR")
if calib_alt:
    for t in tests:
        p = os.path.join(calib_alt, f"{t['number']}.json")
        if os.path.exists(p):
            vcpu_alt[str(num2local[t["number"]])] = json.load(open(p))["cpu"]
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
    "import_sites": import_sites,
    "lazy_attrs": {k: {n: list(v) for n, v in d.items()} for k, d in lazy_attrs.items()},
    "test_importers": {k: sorted(v) for k, v in importers.items()},
    "noverify": sorted(noverify),
    "cpu": {str(k): v for k, v in cpu.items()},
    "spawners": sorted(set(spawners)),
    "mod_imports": mod_imports,
    "vcpu": vcpu,
    "vcpu_alt": vcpu_alt,
    # dict-valued class attributes recorded by key (tracer.py _RD)
    "kltags": json.load(open(os.path.join(trace_dir, "_kl_tags.json"))) if os.path.exists(os.path.join(trace_dir, "_kl_tags.json")) else [],
    "kl": {tag: {kid: sid(s) for kid, s in d.items()} for tag, d in kl.items()},
    "kla": {tag: sid(s) for tag, s in kla.items()},
    "modattr": {k: sid(s) for k, s in modattr.items()},
    "modattr_mod": {k: sid(s) for k, s in modattr_mod.items()},
    "modattr_mods": sorted(ma_mods),
}
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
raw = json.dumps(maps, separators=(",", ":"))
with gzip.open(out, "wt", compresslevel=9) as fh:
    fh.write(raw)
print(f"sets {len(sets)} cov files {len(cov)} lk {len(lk)} raw {len(raw)/1e6:.1f}MB gz {os.path.getsize(out)/1e6:.2f}MB "
      f"noverify {len(noverify)}", file=sys.stderr)
