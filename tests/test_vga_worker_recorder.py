"""Tests for :class:`qontinui.vga.worker._RunRecorder`.

Focus: the recorder opens exactly one PostgreSQL connection per run, holds
it across ``start`` / ``append_event`` / ``finish`` (which would otherwise
each open and close their own), commits at every natural boundary so step
events are durable mid-run, and degrades silently if psycopg is unavailable
or the initial connect call raises.

psycopg is imported lazily inside ``__enter__``/methods via
``import psycopg``. To control it without installing a real psycopg, we
inject a fake module into ``sys.modules`` (the same lookup ``import``
hits) for the duration of each test.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from qontinui.vga.worker import _RunRecorder


def _make_connection_mock() -> MagicMock:
    """Build a MagicMock that behaves like a psycopg Connection.

    ``cursor()`` returns a context-manager-capable MagicMock so the
    ``with conn.cursor() as cur:`` pattern works.
    """
    conn = MagicMock(name="psycopg.Connection")

    cursor = MagicMock(name="psycopg.Cursor")
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = cursor

    return conn


@contextmanager
def _install_fake_psycopg(connect: Any):
    """Install a fake ``psycopg`` module in ``sys.modules`` with the given
    ``connect`` callable, restoring any pre-existing module on exit.
    """
    fake = MagicMock(name="psycopg-module")
    fake.connect = connect
    previous = sys.modules.get("psycopg")
    sys.modules["psycopg"] = fake
    try:
        yield fake
    finally:
        if previous is None:
            sys.modules.pop("psycopg", None)
        else:
            sys.modules["psycopg"] = previous


def _make_recorder() -> _RunRecorder:
    return _RunRecorder(
        pg_url="postgresql://fake/db",
        run_id=uuid4(),
        state_machine_id=uuid4(),
        grounding_model="qwen2.5-vl-7b",
    )


def test_single_connection_across_full_run() -> None:
    """Entering the context + start + N append_event + finish must open
    exactly one PG connection, regardless of step count."""
    conn = _make_connection_mock()
    connect = MagicMock(name="psycopg.connect", return_value=conn)

    with _install_fake_psycopg(connect):
        recorder = _make_recorder()
        with recorder:
            recorder.start()
            for i in range(5):
                recorder.append_event({"kind": "vga.step", "i": i})
            recorder.finish("success")

    assert connect.call_count == 1, (
        f"expected exactly one psycopg.connect() call for the whole run, got {connect.call_count}"
    )
    assert conn.close.call_count == 1


def test_connect_failure_disables_recorder_silently() -> None:
    """If ``psycopg.connect`` raises, the recorder flips to disabled and
    every subsequent call — including ``__exit__`` — is a silent no-op.
    No retry connects are attempted."""

    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("pg is down")

    connect = MagicMock(name="psycopg.connect", side_effect=_raise)

    with _install_fake_psycopg(connect):
        recorder = _make_recorder()
        with recorder:
            # None of these should raise or attempt to reconnect.
            recorder.start()
            recorder.append_event({"kind": "vga.step"})
            recorder.append_event({"kind": "vga.step"})
            recorder.finish("success")

    # __enter__ tries once; nothing retries afterwards.
    assert connect.call_count == 1
    assert recorder._disabled is True
    assert recorder._conn is None


def test_exit_closes_even_when_finish_never_called() -> None:
    """If the caller skips ``finish()`` (e.g. crash mid-run), ``__exit__``
    must still close the connection — otherwise we leak a backend session
    per aborted run."""
    conn = _make_connection_mock()
    connect = MagicMock(name="psycopg.connect", return_value=conn)

    with _install_fake_psycopg(connect):
        recorder = _make_recorder()
        with recorder:
            recorder.start()
            recorder.append_event({"kind": "vga.step"})
            # No finish() call — simulate mid-run abort.

    assert conn.close.call_count == 1
    assert recorder._conn is None


def test_each_append_event_commits_for_durability() -> None:
    """Every ``append_event`` must commit so a mid-run crash leaves the
    already-logged steps persisted. With ``start`` + N events + ``finish``
    + one __enter__ search_path commit + one __exit__ final commit, we
    expect ``N + 4`` commits total."""
    conn = _make_connection_mock()
    connect = MagicMock(name="psycopg.connect", return_value=conn)

    events = 5
    with _install_fake_psycopg(connect):
        recorder = _make_recorder()
        with recorder:
            recorder.start()
            for i in range(events):
                recorder.append_event({"kind": "vga.step", "i": i})
            recorder.finish("success")

    # 1 (SET search_path in __enter__) + 1 (start) + N (appends) + 1 (finish)
    # + 1 (idempotent commit in __exit__) = N + 4
    expected_commits = events + 4
    assert conn.commit.call_count == expected_commits, (
        f"expected {expected_commits} commits (enter + start + {events} appends "
        f"+ finish + exit), got {conn.commit.call_count}"
    )


def test_psycopg_missing_disables_recorder() -> None:
    """If the ``psycopg`` module cannot be imported at all, the recorder
    still works as a no-op context manager — production must not crash
    just because the PG history is unavailable."""
    # Force ImportError by making sys.modules["psycopg"] = None, which
    # causes ``import psycopg`` to raise ImportError immediately.
    previous = sys.modules.get("psycopg")
    sys.modules["psycopg"] = None  # type: ignore[assignment]
    try:
        recorder = _make_recorder()
        with recorder:
            recorder.start()
            recorder.append_event({"kind": "vga.step"})
            recorder.finish("success")
        assert recorder._disabled is True
        assert recorder._conn is None
    finally:
        if previous is None:
            sys.modules.pop("psycopg", None)
        else:
            sys.modules["psycopg"] = previous


if __name__ == "__main__":  # pragma: no cover - manual invocation helper
    raise SystemExit(pytest.main([__file__, "-v"]))
