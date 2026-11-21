"""Run process-pool work in the calling process.

sqlglot's optimizer and executor tests hand their work to a spawned
ProcessPoolExecutor. Tracing cannot see into those workers, and starting them
(fresh interpreters importing pandas and duckdb) costs far more CPU than the
work itself. InlineExecutor runs each task immediately in this process.
Arguments, results and exceptions still go through a pickle round trip, as
they would between processes, so code that only runs while pickling
(sqlglot's serde) is exercised the same way.
"""

from __future__ import annotations

import concurrent.futures as cf
import pickle
import sys


def _roundtrip(x):
    return pickle.loads(pickle.dumps(x, protocol=pickle.HIGHEST_PROTOCOL))


class InlineExecutor(cf.Executor):
    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **_kw):
        self._initializer = initializer
        self._initargs = initargs
        self._ready = False
        self._broken: BaseException | None = None

    def submit(self, fn, /, *args, **kwargs):
        f: cf.Future = cf.Future()
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
        return f

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
            except Exception:  # noqa: BLE001
                pass
