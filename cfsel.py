"""Run the fixture-driven optimizer tests on chosen fixture pairs only.

Most tests in tests/test_optimizer.py check an optimizer rule against a
fixture file: check_file, test_annotate_types and test_annotate_funcs iterate
enumerate(load_sql_fixture_pairs(path), start=1) and check each pair. A change
usually reaches only a few of those pairs, so the selector can verify such a
test on those pairs alone (cftrace.py records which pairs run each line).

install() replaces the test module's load_sql_fixture_pairs and enumerate
globals. For a test method listed in KEEP, the k-th fixture load it makes
(counted per test case) yields only the kept pairs, and enumerate still numbers
them by their place in the file, so titles (and check_file's `only=`) do not
change. A load whose fixture path or pair count differs from the recorded one
yields every pair, as does any load of a test not listed.
"""

from __future__ import annotations

import builtins
import sys

KEEP: dict = {}  # test method -> {load index: (fixture path, pair count, kept pair indexes)}
LOADS: dict = {}  # filled while RECORD: test method -> [(fixture path, pair count)]
RECORD = [False]  # build time: record every load and keep none of its pairs
HOOKS: list = []  # build time: [enter(load index, pair index), leave()] around each kept pair's iteration
_enum = builtins.enumerate


class Pairs:
    """The pairs of one fixture load; iterating yields the kept ones."""

    __slots__ = ("items", "keep", "k")

    def __init__(self, items: list, keep, k: int):
        self.items = items
        self.keep = keep  # sorted indexes, or None for every pair
        self.k = k

    def _sel(self):
        idx = range(len(self.items)) if self.keep is None else self.keep
        if not HOOKS:
            return ((i, self.items[i]) for i in idx)
        return self._hooked(idx)

    def _hooked(self, idx):
        enter, leave = HOOKS
        for i in idx:
            enter(self.k, i)
            try:
                yield i, self.items[i]
            finally:
                leave()

    def __iter__(self):
        return (x for _, x in self._sel())


def enumerate(it, start=0):  # noqa: A001  (shadows the builtin in the test module only)
    if type(it) is not Pairs:
        return _enum(it, start)
    return ((start + i, x) for i, x in it._sel())


def install(mod) -> None:
    if mod.__dict__.get("_cfsel_installed"):
        return
    orig = mod.load_sql_fixture_pairs
    g = mod.__dict__

    def load_sql_fixture_pairs(filename, *a, **kw):
        seq = orig(filename, *a, **kw)
        f = sys._getframe(1)
        if f.f_globals is not g:
            return seq
        tc = f.f_locals.get("self")
        name = getattr(tc, "_testMethodName", None)
        if name is None or not (RECORD[0] or name in KEEP):
            return seq
        d = tc.__dict__
        k = d.get("_cfsel_k", 0)
        d["_cfsel_k"] = k + 1
        items = list(seq)
        if RECORD[0]:
            LOADS.setdefault(name, []).append((filename, len(items)))
            return Pairs(items, [], k)
        want = KEEP[name].get(k)
        if want is None or want[0] != filename or want[1] != len(items):
            return Pairs(items, None, k)
        return Pairs(items, sorted(i for i in want[2] if 0 <= i < len(items)), k)

    mod.load_sql_fixture_pairs = load_sql_fixture_pairs
    mod.enumerate = enumerate
    mod._cfsel_installed = True
