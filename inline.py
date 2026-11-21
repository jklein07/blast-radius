"""Run process-pool work in the calling process.

sqlglot's optimizer and executor tests hand their work to a spawned
ProcessPoolExecutor. Tracing cannot see into those workers, and starting them
(fresh interpreters importing pandas and duckdb) costs far more CPU than the
work itself. InlineExecutor runs each task immediately in this process.
Arguments, results and exceptions still go through a pickle round trip, as
they would between processes, so code that only runs while pickling
(sqlglot's serde) is exercised the same way.

With LAZY set (the selector's verification runs), a task runs when its
future is first asked for its result, not when it is submitted. The tests submit one task per fixture pair (or TPC query) and
then check the results one by one, each in a subtest; the verification
result stops a test at its first failing subtest, so the pairs after it are
never computed. Tasks still run in the order they were submitted, each one
exactly as before; a test that passes runs all of them. In a real pool the
tasks run in other processes, in any order, so no task can depend on a
check the test makes between two results.
"""

from __future__ import annotations

import concurrent.futures as cf
import pickle
import sys


def _roundtrip(x):
    return pickle.loads(pickle.dumps(x, protocol=pickle.HIGHEST_PROTOCOL))


LAZY = False  # the selector turns it on; the build-time tracers run every task at submit


class LazyFuture(cf.Future):
    """A future whose task runs at the first call that needs its outcome."""

    def __init__(self, thunk):
        super().__init__()
        self._thunk = thunk

    def _force(self):
        thunk, self._thunk = self._thunk, None
        if thunk is not None:
            thunk(self)

    def result(self, timeout=None):
        self._force()
        return super().result(timeout)

    def exception(self, timeout=None):
        self._force()
        return super().exception(timeout)

    def done(self):
        self._force()
        return super().done()

    def add_done_callback(self, fn):
        self._force()
        super().add_done_callback(fn)

    def cancel(self):
        self._force()
        return super().cancel()


def as_completed(fs, timeout=None):
    """concurrent.futures.as_completed for the tests: lazy futures in the
    order they were submitted, each run as it is handed out."""
    fs = list(fs)
    if not all(isinstance(f, LazyFuture) for f in fs):
        yield from cf.as_completed(fs, timeout)
        return
    for f in fs:
        f._force()
        yield f


def wait(fs, timeout=None, return_when=cf.ALL_COMPLETED):
    """concurrent.futures.wait: lazy futures count as finished (asking one
    for anything runs it first)."""
    fs = list(fs)
    if not all(isinstance(f, LazyFuture) for f in fs):
        return cf.wait(fs, timeout, return_when)
    return cf._base.DoneAndNotDoneFutures(set(fs), set())


class InlineExecutor(cf.Executor):
    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **_kw):
        self._initializer = initializer
        self._initargs = initargs
        self._ready = False
        self._broken: BaseException | None = None

    def submit(self, fn, /, *args, **kwargs):
        if LAZY:
            return LazyFuture(lambda f: self._run(f, fn, args, kwargs))
        f: cf.Future = cf.Future()
        self._run(f, fn, args, kwargs)
        return f

    def _run(self, f, fn, args, kwargs):
        try:
            if not self._ready:
                self._ready = True
                if self._initializer is not None:
                    try:
                        self._initializer(*_roundtrip(self._initargs))
                    except BaseException as e:  # noqa: BLE001
                        self._broken = e
            if self._broken is not None:
                raise cf.process.BrokenProcessPool("initializer failed") from self._broken
            fn2, a2, k2 = _roundtrip((fn, args, kwargs))
            r = fn2(*a2, **k2)
            f.set_result(_roundtrip(r))
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:  # noqa: BLE001
            try:
                e = _roundtrip(e)
            except Exception:  # noqa: BLE001
                pass
            f.set_exception(e)

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        return super().map(fn, *iterables, timeout=timeout)

    def shutdown(self, wait=True, *, cancel_futures=False):
        return None


def install() -> None:
    """Swap ProcessPoolExecutor everywhere tests can reach it."""
    import concurrent.futures.process as cfp

    cf.ProcessPoolExecutor = InlineExecutor
    cfp.ProcessPoolExecutor = InlineExecutor
    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if (name.startswith("tests") or name.startswith("sqlglot")) and getattr(mod, "ProcessPoolExecutor", None) is not None:
            try:
                mod.ProcessPoolExecutor = InlineExecutor
                if LAZY and name.startswith("tests"):
                    if getattr(mod, "as_completed", None) is cf.as_completed:
                        mod.as_completed = as_completed
                    if getattr(mod, "wait", None) is cf.wait:
                        mod.wait = wait
            except Exception:  # noqa: BLE001
                pass
