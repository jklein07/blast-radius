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


_LOG_TYPES = (logging.Logger, logging.LoggerAdapter, logging.Handler, logging.Filter)
_THREAD_MODS = ("threading", "_thread", "weakref", "_weakref", "_weakrefset")
_KIND: dict = {}  # type -> which leaf rule applies to its instances (0: none)
_MEMO: dict = {}  # (id, depth) -> (object, string) for subtrees with no cycle marker
_CYC = [0]
_LEAF: dict = {}  # id -> (object, string) for enum members and classes (the object is held, so its id stays unique)


def _kind(t) -> int:
    if t in _SIMPLE:
        k = 1
    elif issubclass(t, enum.Enum):
        k = 2
    elif issubclass(t, type):
        k = 3
    elif issubclass(t, types.ModuleType):
        k = 4
    elif issubclass(t, _LOG_TYPES):
        k = 5
    elif t.__module__ in _THREAD_MODS:
        k = 6
    elif issubclass(t, re.Pattern):
        k = 7
    elif issubclass(t, _VALUE_TYPES):
        k = 8
    elif hasattr(t, "arg_types") and t.__module__.startswith("sqlglot"):
        k = 9
    else:
        k = 0
    _KIND[t] = k
    return k


def _fp(v, depth: int, seen: set) -> str:
    t = type(v)
    k = _KIND.get(t)
    if k is None:
        k = _kind(t)
    if k:  # depth-independent leaves (the rule is decided once per type)
        if k == 1:
            return f"{t.__name__}:{v!r}"
        if k == 2 or k == 3:  # enum members and classes: same string while the snapshot runs
            hit = _LEAF.get(id(v))
            if hit is not None and hit[0] is v:
                return hit[1]
            if k == 2:
                r = f"E:{t.__module__}.{t.__qualname__}.{v.name}"
            else:
                r = f"C:{getattr(v, '__module__', '?')}.{getattr(v, '__qualname__', '?')}"
            _LEAF[id(v)] = (v, r)
            return r
        if k == 4:
            return f"M:{v.__name__}"
        if k == 5:
            return f"LOG:{t.__name__}:{getattr(v, 'name', '')}"
        if k == 6:
            return f"T:{t.__qualname__}"
        if k == 7:
            return f"RE:{v.pattern!r}:{v.flags}"
        if k == 8:
            return f"V:{t.__qualname__}:{v!r}"
        if hasattr(v, "args"):
            try:  # sqlglot expression trees: depth-independent
                return f"X:{t.__qualname__}:{v!r}"
            except Exception:  # noqa: BLE001
                pass
    if depth <= 0:
        return f"T:{t.__qualname__}"
    vid = id(v)
    hit = _MEMO.get((vid, depth))
    if hit is not None and hit[0] is v:
        return hit[1]
    if vid in seen:
        _CYC[0] += 1
        return "@cycle"
    seen.add(vid)
    c0 = _CYC[0]
    r = _expand(v, t, depth, seen, vid)
    if _CYC[0] == c0:
        _MEMO[(vid, depth)] = (v, r)
    return r


def _expand(v, t, depth: int, seen: set, vid: int) -> str:
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
        if len(_LEAF) > 200000:
            _LEAF.clear()
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
        # onerror: a package that raises anything (a SyntaxError below it, not
        # just ImportError) is recorded by the import loop, not raised here
        for info in pkgutil.walk_packages(pkg.__path__, prefix=root_pkg + ".", onerror=lambda _n: None)
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
    alias: dict[str, str] = {}  # key -> module:qualname when they differ (dynamic classes)
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
        mq = f"{getattr(c, '__module__', '?')}:{getattr(c, '__qualname__', '?')}"
        if mq != key:
            alias[key] = mq
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
    return {"classes": classes, "bases": bases, "globals": globals_, "errors": errors, "alias": alias}


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


# ------------------------------------------------------- dict entries by key
def key_id(k) -> str:
    """A process-independent id for a dict key. Keys that compare equal
    across types (True == 1 == 1.0, a str-valued enum member and its string)
    get the same id, because a lookup with either finds the same entry."""
    import fractions as _fr
    import numbers as _nb

    if isinstance(k, str):
        return "s:" + k  # str subclasses (str-valued enums) hash and compare as their text
    if isinstance(k, _nb.Number) and not isinstance(k, enum.Enum) or isinstance(k, (int, float)):
        try:
            if k == k:
                if isinstance(k, complex):
                    k = k.real if k.imag == 0 else k
                if not isinstance(k, complex):
                    return "n:" + str(_fr.Fraction(k))
        except (TypeError, ValueError, OverflowError):
            pass
    if type(k) is tuple:
        return "(" + ",".join(key_id(x) for x in k) + ")"
    return fp(k, 3)


LEN_KEY = "#len"  # pseudo-key for the dict's size (len(), truth tests)


def dict_key_fps(d: dict) -> dict:
    """key id -> fingerprint of the value; entries whose key ids collide are
    combined, so a change to either shows. LEN_KEY holds the size."""
    out: dict = {LEN_KEY: str(len(d))}
    for k, v in d.items():
        kid, vf = key_id(k), fp(v)
        out[kid] = vf if kid not in out else "+".join(sorted(out[kid].split("+") + [vf]))
    return out


def set_key_fps(s) -> dict:
    """element id -> "1" (like dict_key_fps with every value the same);
    LEN_KEY holds the size."""
    out: dict = {LEN_KEY: str(len(s))}
    for e in s:
        out[key_id(e)] = "1"
    return out


def key_fps(v):
    """dict_key_fps / set_key_fps by type, None for anything else."""
    if type(v) is dict:
        return dict_key_fps(v)
    if type(v) in (set, frozenset):
        return {"#type": type(v).__name__, **set_key_fps(v)}
    return None


def dict_snapshot(root_pkg: str = "sqlglot") -> dict:
    """'class key|attr' -> key_fps for every dict-, set- or frozenset-valued
    class attribute."""
    out: dict = {}
    for key, c in list(iter_classes(sorted(sys.modules), root_pkg)):
        if isinstance(c, enum.EnumMeta):
            continue
        for name, v in list(c.__dict__.items()):
            if type(v) in (dict, set, frozenset) and not (name.startswith("__") and name.endswith("__")):
                out[f"{key}|{name}"] = key_fps(v)
    return out
