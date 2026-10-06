"""The model's child process: answers calls, stops when idle, and survives a crash."""

from __future__ import annotations

import os
import threading
import time

import pytest

from gauge.model import Decision, InputError
from gauge.worker import DEFAULT_IDLE_UNLOAD, WorkerDecider, env_idle_unload

FAKE = "fake_worker_decider:FakeDecider"


@pytest.fixture
def worker():
    w = WorkerDecider(60, factory=FAKE)
    yield w
    w.stop()


def wait_for(condition, seconds: float = 5.0) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.02)
    return False


def test_nothing_runs_until_the_first_call(worker):
    assert not worker.running


def test_answers_come_from_the_child_process(worker):
    r = worker.check("text", "claim")
    assert isinstance(r, Decision) and r.answer == "true"
    assert r.model == f"pid {worker._proc.pid}" != f"pid {os.getpid()}"


def test_every_method_the_server_uses_reaches_the_child(worker):
    assert worker.decide("t", ["a", "b"]).answer == "a"
    assert [r.answer for r in worker.check_many(["x", "y"], "c")] == ["true", "true"]
    assert len(worker.decide_many(["x"], ["a", "b"])) == 1
    assert worker.check_statements(["short"]) is None


def test_one_process_serves_call_after_call(worker):
    pids = {worker.check("t", "c").model for _ in range(3)}
    assert len(pids) == 1


def test_a_caller_mistake_comes_back_as_an_input_error(worker):
    with pytest.raises(InputError, match="non-empty"):
        worker.check("bad", "claim")
    with pytest.raises(InputError, match="too long"):
        worker.check_statements(["far too long a claim"])
    assert worker.running                    # and the process keeps going


def test_stops_after_the_idle_time_and_starts_again_on_the_next_call():
    w = WorkerDecider(0.3, factory=FAKE)
    try:
        first = w.check("t", "c").model
        assert wait_for(lambda: not w.running)
        assert w.check("t", "c").model != first   # a fresh process
    finally:
        w.stop()


def test_steady_use_keeps_the_same_process():
    w = WorkerDecider(0.5, factory=FAKE)
    try:
        pids = set()
        for _ in range(5):                   # 1s of use, each gap well under 0.5s
            pids.add(w.check("t", "c").model)
            time.sleep(0.2)
        assert len(pids) == 1
    finally:
        w.stop()


def test_a_timer_from_before_the_last_call_leaves_the_process_running(worker):
    worker.check("t", "c")
    current = worker._timer
    worker._stop_if_idle()                   # an old timer firing late
    assert worker.running
    assert worker._timer is current          # and doesn't start a second countdown


def test_a_crashed_process_is_reported_and_replaced(worker):
    with pytest.raises(RuntimeError, match="stopped unexpectedly"):
        worker.check("crash", "c")
    assert worker.check("t", "c").answer == "true"


def test_library_output_in_the_child_never_reaches_stdout(worker, capfd):
    """stdout carries the MCP protocol; a stray print there would corrupt it."""
    worker.check("print", "c")
    out, err = capfd.readouterr()
    assert "library noise" not in out
    assert "library noise" in err


def test_stop_ends_the_process(worker):
    worker.check("t", "c")
    proc = worker._proc
    worker.stop()
    assert not proc.is_alive()
    assert not worker.running


@pytest.mark.parametrize("raw, expected", [
    (None, DEFAULT_IDLE_UNLOAD), ("", DEFAULT_IDLE_UNLOAD), ("30", 30.0), ("0.5", 0.5), ("0", None),
])
def test_idle_unload_setting(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("GAUGE_IDLE_UNLOAD", raising=False)
    else:
        monkeypatch.setenv("GAUGE_IDLE_UNLOAD", raw)
    assert env_idle_unload() == expected


@pytest.mark.parametrize("raw", ["soon", "-5", "nan"])
def test_bad_idle_unload_setting_is_an_error(monkeypatch, raw):
    monkeypatch.setenv("GAUGE_IDLE_UNLOAD", raw)
    with pytest.raises(RuntimeError, match="GAUGE_IDLE_UNLOAD"):
        env_idle_unload()


def test_a_process_killed_between_calls_is_replaced(worker):
    """Say the system killed it while idle: the next call just starts a new one."""
    worker.check("t", "c")
    worker._proc.kill()
    worker._proc.join()
    assert worker.check("t", "c").answer == "true"



def test_a_timer_that_wakes_early_waits_out_the_rest(worker):
    """After a Mac sleep the timer can wake before the idle time has passed on the
    clock idle time is measured with. It must try again, not give up."""
    worker.idle_unload = 0.3
    worker.check("t", "c")
    worker._timer.cancel()
    worker._last_used = time.monotonic()           # "used just now", as seen on waking
    worker._timer = threading.Timer(0, worker._stop_if_idle)
    worker._timer.start()                           # fires early
    assert wait_for(lambda: not worker.running, 3)


def test_a_setup_error_reaches_the_caller():
    """A bad setting should be reported as itself, on every call, not as a crash."""
    w = WorkerDecider(60, factory="fake_worker_decider:BrokenSetup")
    try:
        for _ in range(2):
            with pytest.raises(RuntimeError, match="GAUGE_THRESHOLD"):
                w.check("t", "c")
        assert w.running                            # one process, not one per retry
    finally:
        w.stop()


def test_an_error_that_cannot_be_rebuilt_is_still_reported():
    w = WorkerDecider(60, factory="fake_worker_decider:Slow")
    try:
        with pytest.raises(RuntimeError, match="unreadable error"):
            w.check("odd", "c")
        assert w.check("t", "c").answer == "true"
    finally:
        w.stop()


def test_stop_gives_up_waiting_for_a_stuck_call_and_kills_it():
    w = WorkerDecider(60, factory="fake_worker_decider:Slow")
    w.check("t", "c")
    caller = threading.Thread(target=lambda: pytest.raises(RuntimeError, w.check, "hang", "c"))
    caller.start()
    time.sleep(0.2)                                 # the call is now stuck in the child
    start = time.monotonic()
    w.stop(wait=0.3)
    assert time.monotonic() - start < 2
    caller.join(5)
    assert not caller.is_alive()                    # the stuck call failed and returned
    assert not w.running


@pytest.fixture
def server(monkeypatch):
    """mcp_server.main() with the protocol loop and warm-up thread stubbed out."""
    from gauge import mcp_server

    ran: list = []
    warmed: list = []
    monkeypatch.setattr(mcp_server.mcp, "run", lambda **kw: ran.append(mcp_server.get_decider()))
    monkeypatch.setattr(mcp_server.threading, "Thread", lambda target, daemon: type(
        "T", (), {"start": lambda self: warmed.append(target)})())
    monkeypatch.setattr(mcp_server, "_worker", None)
    return mcp_server, ran, warmed


def test_server_answers_from_a_child_process_by_default(server, monkeypatch):
    mcp_server, ran, warmed = server
    monkeypatch.delenv("GAUGE_IDLE_UNLOAD", raising=False)
    stopped = []
    monkeypatch.setattr(WorkerDecider, "stop", lambda self: stopped.append(self))
    mcp_server.main()
    assert isinstance(ran[0], WorkerDecider)
    assert ran[0].idle_unload == DEFAULT_IDLE_UNLOAD
    assert warmed == []                      # nothing loaded before it's asked
    assert stopped == [ran[0]]               # and the child is stopped on exit


def test_server_with_idle_unload_off_keeps_the_model_in_process(server, monkeypatch):
    mcp_server, ran, warmed = server
    monkeypatch.setenv("GAUGE_IDLE_UNLOAD", "0")
    mcp_server.main()
    assert not isinstance(ran[0], WorkerDecider)
    assert warmed == [mcp_server._warm_up]
