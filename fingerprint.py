"""Deterministic fingerprints of sqlglot's import-time state.

After importing every sqlglot module, each class's own attributes and each
module's globals are reduced to a short hash. Comparing a patched import with
the pristine one shows exactly which (class, attribute) pairs and module
globals a change altered, including anything metaclasses derive from them.
Code objects are deliberately left out: function-body changes are mapped
through line coverage instead.
"""

from __future__ import annotations

import enum
import functools
import collections
import datetime
import decimal
import fractions
import hashlib
import logging
import re
import importlib
import os
import pkgutil
import sys
import types

_SIMPLE = (type(None), bool, int, float, complex, str, bytes)
_VALUE_TYPES = (datetime.date, datetime.time, datetime.timedelta, datetime.tzinfo, decimal.Decimal, fractions.Fraction, range, slice)
_FUNC_TYPES = (types.FunctionType, types.BuiltinFunctionType, types.MethodType, types.BuiltinMethodType)


def _fp(v, depth: int, seen: set) -> str:
    t = type(v)
    if t in _SIMPLE:
        return f"{t.__name__}:{v!r}"
    if isinstance(v, enum.Enum):
        return f"E:{type(v).__module__}.{type(v).__qualname__}.{v.name}"
    if isinstance(v, type):
        return f"C:{getattr(v, '__module__', '?')}.{getattr(v, '__qualname__', '?')}"
    if isinstance(v, types.ModuleType):
        return f"M:{v.__name__}"
    if isinstance(v, (logging.Logger, logging.LoggerAdapter, logging.Handler, logging.Filter)):
        return f"LOG:{type(v).__name__}:{getattr(v, 'name', '')}"
    if t.__module__ in ("threading", "_thread", "weakref", "_weakref", "_weakrefset"):
        return f"T:{t.__qualname__}"
    if isinstance(v, re.Pattern):
        return f"RE:{v.pattern!r}:{v.flags}"
    if isinstance(v, _VALUE_TYPES):
        return f"V:{t.__qualname__}:{v!r}"
    if hasattr(t, "arg_types") and hasattr(v, "args") and t.__module__.startswith("sqlglot"):
        try:  # sqlglot expression trees: depth-independent
            return f"X:{t.__qualname__}:{v!r}"
        except Exception:  # noqa: BLE001
            pass
    if depth <= 0:
        return f"T:{t.__qualname__}"
    vid = id(v)
    if vid in seen:
        return "@cycle"
    seen.add(vid)
    try:
        if isinstance(v, (types.FunctionType,)):
            parts = [f"F:{v.__module__}.{v.__qualname__}"]
            if v.__defaults__:
                parts.append(_fp(v.__defaults__, depth - 1, seen))
            if v.__kwdefaults__:
                parts.append(_fp(v.__kwdefaults__, depth - 1, seen))
            if v.__closure__:
                cells = []
                for c in v.__closure__:
                    try:
                        cells.append(_fp(c.cell_contents, depth - 1, seen))
                    except ValueError:
                        cells.append("<empty>")
                parts.append("[" + ",".join(cells) + "]")
            return "(" + "|".join(parts) + ")"
        if isinstance(v, _FUNC_TYPES):
            self_ = getattr(v, "__self__", None)
            name = getattr(v, "__qualname__", getattr(v, "__name__", "?"))
            mod = getattr(v, "__module__", None)
            s = f"B:{mod}.{name}"
            if self_ is not None and not isinstance(self_, types.ModuleType):
                s += "/" + _fp(self_, depth - 1, seen)
            return s
        if isinstance(v, (staticmethod, classmethod)):
            return f"{t.__name__}(" + _fp(v.__func__, depth - 1, seen) + ")"
        if isinstance(v, property):
            return "P(" + ",".join(_fp(f, depth - 1, seen) for f in (v.fget, v.fset, v.fdel)) + ")"
        if isinstance(v, functools.partial):
            return "Pa(" + _fp(v.func, depth - 1, seen) + _fp(v.args, depth - 1, seen) + _fp(v.keywords, depth - 1, seen) + ")"
        if isinstance(v, dict):
            items = sorted(_fp(k, depth - 1, seen) + "=" + _fp(x, depth - 1, seen) for k, x in v.items())
            return f"D{t.__name__}{{" + ",".join(items) + "}"
        if isinstance(v, (list, tuple)):
            parts = [_fp(x, depth - 1, seen) for x in v]
            if parts and all(isinstance(x, type) for x in v):
                parts.sort()  # often built from a set of classes: order is address-dependent
            return f"L{t.__name__}[" + ",".join(parts) + "]"
        if isinstance(v, (set, frozenset)):
            return f"S{t.__name__}{{" + ",".join(sorted(_fp(x, depth - 1, seen) for x in v)) + "}"
        if isinstance(v, types.MappingProxyType):
            return "MP{" + ",".join(_fp(k, depth - 1, seen) + "=" + _fp(x, depth - 1, seen) for k, x in v.items()) + "}"
        d = getattr(v, "__dict__", None)
        slots = getattr(t, "__slots__", None)
        parts = [f"O:{t.__module__}.{t.__qualname__}"]
        if isinstance(v, collections.defaultdict):
            parts.append(_fp(v.default_factory, depth - 1, seen))
        if not isinstance(d, dict) and not slots:
            try:
                r = repr(v)
                if " at 0x" not in r:
                    parts.append(r[:2000])
            except Exception:  # noqa: BLE001
                pass
        if isinstance(d, dict):
            parts.append(_fp(d, depth - 1, seen))
        if slots:
            if isinstance(slots, str):
                slots = (slots,)
            for s in slots:
                try:
                    parts.append(s + "=" + _fp(getattr(v, s), depth - 1, seen))
                except Exception:
                    pass
        return "(" + "|".join(parts) + ")"
    finally:
        seen.discard(vid)


_ROOTS: list[str] = []


def fp(v, depth: int = 8) -> str:
    try:
        s = _fp(v, depth, set())
        for r in _ROOTS:
            s = s.replace(r, "<ROOT>")
    except Exception as e:  # pragma: no cover - defensive
        s = f"ERR:{type(e).__name__}"
    return hashlib.blake2b(s.encode("utf-8", "surrogatepass"), digest_size=8).hexdigest()


def import_all(root_pkg: str = "sqlglot") -> tuple[list[str], dict[str, str]]:
    """Import every module of the package in a fixed order. Returns
    (module names imported, {module: error})."""
    errors: dict[str, str] = {}
    pkg = importlib.import_module(root_pkg)
    names = sorted(
        info.name
        for info in pkgutil.walk_packages(pkg.__path__, prefix=root_pkg + ".")
        if info.name != root_pkg + ".__main__"
    )
    ok = [root_pkg]
    for name in names:
        try:
            importlib.import_module(name)
            ok.append(name)
        except Exception as e:  # noqa: BLE001
            errors[name] = f"{type(e).__name__}: {e}"[:300]
    return ok, errors


def _ours(v, root_pkg: str) -> bool:
    if not isinstance(v, type):
        return False
    m = v.__dict__.get("__module__") if isinstance(v.__dict__, types.MappingProxyType) else None
    if not isinstance(m, str):
        m = getattr(v, "__module__", "")
    return isinstance(m, str) and (m == root_pkg or m.startswith(root_pkg + "."))


def _resolves(c) -> bool:
    mod = sys.modules.get(getattr(c, "__module__", None) or "")
    if mod is None:
        return False
    obj = mod
    for part in getattr(c, "__qualname__", "").split("."):
        if part == "<locals>":
            return False
        obj = getattr(obj, "__dict__", {}).get(part) if not isinstance(obj, types.ModuleType) else vars(obj).get(part)
        if obj is None:
            return False
    return obj is c


def iter_classes(modules: list[str], root_pkg: str = "sqlglot"):
    """Yield (key, cls) for every class reachable from the given modules,
    including nested and metaclass-generated classes. key is
    'module:Qual.Name' when that path resolves to the class, otherwise the
    path it was found by ('module:Owner.attr')."""
    seen = set()
    stack = []
    for mname in sorted(modules, reverse=True):
        mod = sys.modules.get(mname)
        if mod is None:
            continue
        for name, v in sorted(vars(mod).items(), reverse=True):
            if _ours(v, root_pkg):
                stack.append((v, f"{mname}:{name}"))
    while stack:
        c, path = stack.pop()
        if id(c) in seen:
            continue
        seen.add(id(c))
        key = f"{c.__module__}:{c.__qualname__}" if _resolves(c) else path
        yield key, c
        for name, v in sorted(c.__dict__.items(), reverse=True):
            if _ours(v, root_pkg):
                stack.append((v, f"{key}.{name}"))


def snapshot(root_pkg: str = "sqlglot", only: set | None = None) -> dict:
    """only: if given, fingerprint just the classes and globals of these modules
    (classes are still discovered through every module so keys stay stable)."""
    pkg_dir = None
    try:
        import importlib.util

        spec = importlib.util.find_spec(root_pkg)
        pkg_dir = os.path.dirname(os.path.dirname(spec.origin)) if spec and spec.origin else None
    except Exception:
        pass
    if pkg_dir:
        _ROOTS[:] = [os.path.realpath(pkg_dir), pkg_dir]
    mods, errors = import_all(root_pkg)
    classes: dict[str, dict[str, str]] = {}
    bases: dict[str, list[str]] = {}
    for key, c in iter_classes(sorted(sys.modules), root_pkg):
        if only is not None and key.partition(":")[0] not in only:
            continue
        if key in classes:
            # same qualname defined twice (e.g. dynamic type()); merge conservatively
            key = key + f"#{len([k for k in classes if k.startswith(key)])}"
        attrs = {}
        for name, v in list(c.__dict__.items()):
            if name in ("__dict__", "__weakref__", "__doc__", "__module__", "__qualname__", "__firstlineno__", "__static_attributes__"):
                continue
            attrs[name] = fp(v)
        classes[key] = attrs
        bases[key] = [f"{b.__module__}:{b.__qualname__}" for b in c.__mro__[1:] if b is not object]
    globals_: dict[str, dict[str, str]] = {}
    for mname in sorted(sys.modules):
        if not (mname == root_pkg or mname.startswith(root_pkg + ".")):
            continue
        if only is not None and mname not in only:
            continue
        mod = sys.modules[mname]
        g = {}
        for name, v in list(vars(mod).items()):
            if name.startswith("__") and name.endswith("__"):
                continue
            g[name] = fp(v)
        globals_[mname] = g
    return {"classes": classes, "bases": bases, "globals": globals_, "errors": errors}


def diff(a: dict, b: dict, only: set | None = None) -> dict:
    """Changed (class, attr) pairs and module globals between snapshots a and b,
    optionally restricted to the given modules."""

    def keep(key: str) -> bool:
        return only is None or key.partition(":")[0] in only

    changed_attrs = []
    for key in {k for k in a["classes"] if keep(k)} | {k for k in b["classes"] if keep(k)}:
        ca = a["classes"].get(key)
        cb = b["classes"].get(key)
        if ca is None or cb is None or a["bases"].get(key) != b["bases"].get(key):
            changed_attrs.append((key, "*"))
            continue
        for name in set(ca) | set(cb):
            if ca.get(name) != cb.get(name):
                changed_attrs.append((key, name))
    changed_globals = []
    for m in {k for k in a["globals"] if keep(k)} | {k for k in b["globals"] if keep(k)}:
        ga = a["globals"].get(m)
        gb = b["globals"].get(m)
        if ga is None or gb is None:
            changed_globals.append((m, "*"))
            continue
        for name in set(ga) | set(gb):
            if ga.get(name) != gb.get(name):
                changed_globals.append((m, name))
    err = sorted(set(a["errors"].items()) ^ set(b["errors"].items()))
    return {"attrs": sorted(changed_attrs), "globals": sorted(changed_globals), "errors": err}
