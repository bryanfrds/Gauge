# Copyright 2026 bryanfrds (https://github.com/bryanfrds)
# SPDX-License-Identifier: Apache-2.0
"""Run the model in a child process that stops when it has been idle.

The MCP server stays open for a whole chat and is usually idle. Freeing the model
inside the server process doesn't give the memory back: ONNX Runtime keeps what it
allocated (measured: 565 MB in use, 483 MB still held after dropping the session).
A process that exits gives back everything, so the model lives in a child process
that is stopped after `idle_unload` quiet seconds and started again on the next call.
Measured cost of that restart with the ONNX backend: 0.4 to 1 second.
"""

from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import threading
import time

# A server sits open for a whole chat, so by default it frees the model after two
# quiet minutes. GAUGE_IDLE_UNLOAD=0 keeps it loaded in the server process instead.
DEFAULT_IDLE_UNLOAD = 120.0


def env_idle_unload() -> float | None:
    """GAUGE_IDLE_UNLOAD in seconds; unset means the default, 0 means never.

    Server config, so a bad value is a RuntimeError rather than a caller error.
    """
    raw = os.environ.get("GAUGE_IDLE_UNLOAD")
    if raw is None or not raw.strip():
        return DEFAULT_IDLE_UNLOAD
    try:
        value = float(raw)
    except ValueError:
        value = -1.0
    if not value >= 0:  # also catches nan
        raise RuntimeError(f"GAUGE_IDLE_UNLOAD must be a number of seconds, 0 or more, got {raw!r}")
    return value or None


def _send(conn, result) -> None:
    try:
        conn.send(result)
    except Exception as e:  # an exception that won't pickle
        conn.send((False, RuntimeError(f"{type(e).__name__}: {e}")))


def _serve(conn, factory: str) -> None:
    """Child process: build the decider, then answer calls until told to stop."""
    # stdout belongs to the MCP protocol in the parent; a stray print from a library
    # here must not land in it. (Anything printed while multiprocessing re-imports
    # the parent's main module, before this line, would still reach it; nothing does.)
    os.dup2(2, 1)
    try:
        module, _, name = factory.partition(":")
        decider = getattr(importlib.import_module(module), name)()
    except Exception as e:
        # A setup problem (bad GAUGE_THRESHOLD, model missing). Report it on every call
        # instead of dying, so the caller sees the real cause, not "stopped unexpectedly".
        setup_error = e
    else:
        setup_error = None
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):  # the parent has gone
            return
        if msg is None:
            return
        if setup_error is not None:
            _send(conn, (False, setup_error))
            continue
        method, args = msg
        try:
            result = (True, getattr(decider, method)(*args))
        except Exception as e:
            result = (False, e)
        _send(conn, result)


class WorkerDecider:
    """The Decider methods the server uses, answered by a child process.

    Calls are serialized, as they are in Decider itself.
    """

    def __init__(self, idle_unload: float, factory: str = "gauge.model:Decider"):
        self.idle_unload = idle_unload
        self._factory = factory
        self._lock = threading.Lock()
        self._proc = None
        self._conn = None
        self._last_used = 0.0
        self._timer: threading.Timer | None = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def _call(self, method: str, *args):
        with self._lock:
            if not self.running:
                self._start_locked()
            try:
                self._conn.send((method, args))
                ok, value = self._conn.recv()
            except (EOFError, OSError):
                self._stop_locked()
                raise RuntimeError(
                    "the gauge model process stopped unexpectedly; the next call starts a new one"
                ) from None
            except Exception as e:  # an error that pickled but won't unpickle here
                raise RuntimeError(f"the gauge model process sent back an unreadable error: "
                                   f"{type(e).__name__}: {e}") from None
            finally:
                self._used_locked()
        if ok:
            return value
        raise value

    def _start_locked(self) -> None:
        self._stop_locked()  # tidy up one that died
        ctx = mp.get_context("spawn")  # a clean interpreter; fork would copy our threads
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=_serve, args=(child, self._factory), daemon=True)
        proc.start()
        child.close()
        self._proc, self._conn = proc, parent

    def _stop_locked(self) -> None:
        if self._conn is not None:
            try:
                self._conn.send(None)
            except OSError:
                pass
            self._conn.close()
        if self._proc is not None:
            self._proc.join(5)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join()
        self._proc = self._conn = None

    def _used_locked(self) -> None:
        self._last_used = time.monotonic()
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(self.idle_unload, self._stop_if_idle)
        self._timer.daemon = True  # never keeps the server alive
        self._timer.start()

    def _stop_if_idle(self) -> None:
        with self._lock:
            if threading.current_thread() is not self._timer:
                return  # a call since this timer started has set a newer one
            if not self.running:
                return
            idle_for = time.monotonic() - self._last_used
            if idle_for >= self.idle_unload:
                self._stop_locked()
                return
            # Woke early. The timer waits on the wall clock but idle time is measured
            # on the monotonic clock, which pauses while a Mac sleeps, so after a sleep
            # the two disagree. Wait out the rest rather than never stopping.
            self._timer = threading.Timer(self.idle_unload - idle_for, self._stop_if_idle)
            self._timer.daemon = True
            self._timer.start()

    def stop(self, wait: float = 10.0) -> None:
        """Stop the model process. If a call is stuck for longer than `wait` seconds,
        kill the process anyway, so the server can still exit."""
        if not self._lock.acquire(timeout=wait):
            proc = self._proc
            if proc is not None:
                proc.kill()  # the stuck call then fails and tidies up
            return
        try:
            if self._timer is not None:
                self._timer.cancel()
            self._stop_locked()
        finally:
            self._lock.release()

    def check(self, text, claim):
        return self._call("check", text, claim)

    def decide(self, text, options):
        return self._call("decide", text, options)

    def check_many(self, texts, claim):
        return self._call("check_many", texts, claim)

    def decide_many(self, texts, options):
        return self._call("decide_many", texts, options)

    def check_statements(self, statements):
        return self._call("check_statements", statements)
