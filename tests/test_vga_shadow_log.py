"""Unit tests for :class:`qontinui.vga.shadow_log.ShadowSampleLogger`.

psycopg is stubbed out with :mod:`unittest.mock` so these tests never
touch a real database. Disk I/O is scoped to ``tmp_path``.

Covered behaviors:

- Private SMs never produce a DB insert or a PNG on disk.
- Two identical samples (same image_sha) only write the PNG once; the
  INSERT is issued once per ``log_sample`` call (the schema does not
  have a UNIQUE index on the dedupe triple — disk-layer dedupe is the
  real content guard).
- When the cached row count exceeds the rate-limit threshold, samples
  are stochastically dropped according to ``sample_rate``.
- Successful writes land a PNG file on disk at
  ``{images_dir}/{sha}.png``.
"""

from __future__ import annotations

import hashlib
import random
import sys
import types
from pathlib import Path
from typing import Any
from unittest import mock
from uuid import uuid4

import pytest

# Ensure ``import psycopg`` inside ShadowSampleLogger resolves to a stub
# module when the real driver isn't installed (e.g. CI, fresh checkouts).
# We only need the module to be importable — every test patches
# ``psycopg.connect`` to the fakes defined below, so no real code from
# psycopg is ever executed.
if "psycopg" not in sys.modules:
    _stub = types.ModuleType("psycopg")

    def _unused_connect(*_args: Any, **_kwargs: Any) -> Any:  # pragma: no cover
        raise RuntimeError(
            "psycopg.connect stub invoked — test forgot to patch psycopg.connect"
        )

    _stub.connect = _unused_connect  # type: ignore[attr-defined]
    sys.modules["psycopg"] = _stub

from qontinui.vga.shadow_log import ShadowSampleLogger  # noqa: E402

# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------


class _FakeCursor:
    """Minimal psycopg cursor double.

    ``count_result`` is what the mocked ``SELECT COUNT(*)`` returns.
    Every execute call is recorded on ``executed`` for later assertions.
    """

    def __init__(self, count_result: int = 0) -> None:
        self._count_result = count_result
        self.executed: list[tuple[str, tuple | None]] = []

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def execute(self, sql: str, params: tuple | None = None) -> None:
        self.executed.append((sql, params))

    def fetchone(self) -> tuple[int] | None:
        # Only the COUNT query uses fetchone in this module.
        return (self._count_result,)


class _FakeConn:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor
        self.committed = 0

    def __enter__(self) -> _FakeConn:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def cursor(self) -> _FakeCursor:
        return self._cursor

    def commit(self) -> None:
        self.committed += 1


class _ConnFactory:
    """Stand-in for ``psycopg.connect`` that hands out cursors in sequence.

    Each call to the factory returns a new _FakeConn whose cursor is the
    next one from ``cursors``. If ``cursors`` runs out, the last one is
    reused (fresh COUNT semantics don't matter for insertion-only tests).
    """

    def __init__(self, cursors: list[_FakeCursor]) -> None:
        self._cursors = cursors
        self.calls: list[str] = []

    def __call__(self, pg_url: str) -> _FakeConn:
        self.calls.append(pg_url)
        cursor = self._cursors[min(len(self.calls) - 1, len(self._cursors) - 1)]
        return _FakeConn(cursor)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _png_bytes(seed: bytes = b"shot-A") -> bytes:
    """Return deterministic 'PNG' bytes for testing sha computation.

    Not a real PNG — the logger only hashes and writes the bytes; it
    never decodes them.
    """
    return b"\x89PNG\r\n\x1a\n" + seed


def _all_inserts(cursor: _FakeCursor) -> list[tuple[str, tuple | None]]:
    return [e for e in cursor.executed if e[0].startswith("INSERT")]


def _all_counts(cursor: _FakeCursor) -> list[tuple[str, tuple | None]]:
    return [e for e in cursor.executed if "COUNT(" in e[0]]


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_private_sample_never_writes(tmp_path: Path) -> None:
    """``private=True`` short-circuits before any disk or DB work."""
    cursor = _FakeCursor(count_result=0)
    factory = _ConnFactory([cursor])

    with mock.patch("psycopg.connect", factory):
        log = ShadowSampleLogger(
            pg_url="postgres://fake",
            images_dir=tmp_path,
        )
        log.log_sample(
            image_png_bytes=_png_bytes(),
            state_machine_id=uuid4(),
            target_process="notepad++.exe",
            prompt="Save button",
            v5_bbox={"x": 10, "y": 20, "w": 30, "h": 40},
            v5_model="qontinui-grounding-v5",
            private=True,
        )

    assert factory.calls == [], "psycopg.connect must not be called for private SMs"
    assert list(tmp_path.iterdir()) == [], "No PNG must land on disk for private SMs"


def test_basic_write_lands_on_disk_and_inserts(tmp_path: Path) -> None:
    """Non-private sample produces both a PNG and an INSERT."""
    cursor = _FakeCursor(count_result=0)
    factory = _ConnFactory([cursor])

    sm_id = uuid4()
    png = _png_bytes()
    expected_sha = hashlib.sha256(png).hexdigest()
    expected_png_path = tmp_path / f"{expected_sha}.png"

    with mock.patch("psycopg.connect", factory):
        log = ShadowSampleLogger(
            pg_url="postgres://fake",
            images_dir=tmp_path,
        )
        log.log_sample(
            image_png_bytes=png,
            state_machine_id=sm_id,
            target_process="notepad++.exe",
            prompt="Save button",
            v5_bbox={"x": 10, "y": 20, "w": 30, "h": 40},
            v5_model="qontinui-grounding-v5",
            private=False,
            confidence=0.75,
        )

    assert expected_png_path.exists(), "PNG must be written on successful sample"
    assert expected_png_path.read_bytes() == png

    inserts = _all_inserts(cursor)
    assert len(inserts) == 1
    _sql, params = inserts[0]
    # (state_machine_id, image_sha, image_path, prompt, target_process,
    #  predicted_bbox(json), model_used, confidence)
    assert params is not None
    assert params[0] == str(sm_id)
    assert params[1] == expected_sha
    assert params[2] == str(expected_png_path)
    assert params[3] == "Save button"
    assert params[4] == "notepad++.exe"
    assert '"x": 10' in params[5]  # JSON serialization
    assert params[6] == "qontinui-grounding-v5"
    assert params[7] == pytest.approx(0.75)


def test_duplicate_sample_writes_png_once(tmp_path: Path) -> None:
    """Two calls with the same PNG contents only write the file once.

    The second call still issues an INSERT — disk-layer dedupe is the
    content guard; the DB-layer dedupe is handled downstream by the
    shadow-eval query's GROUP BY. The important invariant for *disk* is
    that the PNG is written exactly once.
    """
    cursor = _FakeCursor(count_result=0)
    factory = _ConnFactory([cursor])

    sm_id = uuid4()
    png = _png_bytes(b"duplicate")
    expected_sha = hashlib.sha256(png).hexdigest()
    expected_png_path = tmp_path / f"{expected_sha}.png"

    with mock.patch("psycopg.connect", factory):
        log = ShadowSampleLogger(
            pg_url="postgres://fake",
            images_dir=tmp_path,
        )
        for _ in range(2):
            log.log_sample(
                image_png_bytes=png,
                state_machine_id=sm_id,
                target_process="notepad++.exe",
                prompt="Save button",
                v5_bbox={"x": 10, "y": 20, "w": 30, "h": 40},
                v5_model="qontinui-grounding-v5",
                private=False,
            )

    # Only one file on disk, and it has the original bytes.
    png_files = sorted(tmp_path.glob("*.png"))
    assert len(png_files) == 1
    assert png_files[0] == expected_png_path
    assert expected_png_path.read_bytes() == png


def test_rate_limit_drops_samples_when_table_is_full(tmp_path: Path) -> None:
    """Once row count >= threshold, samples are dropped per ``sample_rate``.

    We inject a ``random.Random`` with a fixed seed so the pass/drop
    decisions are deterministic. Threshold is set tiny and sample_rate
    to 0.0 so every decision after the first count must drop.
    """
    # Count query returns 1000 — above threshold of 10.
    cursor = _FakeCursor(count_result=1000)
    factory = _ConnFactory([cursor])

    # sample_rate=0.0 means "never pass the RNG gate" — every sample drops
    # once rate limiting kicks in.
    rng = random.Random(42)
    with mock.patch("psycopg.connect", factory):
        log = ShadowSampleLogger(
            pg_url="postgres://fake",
            images_dir=tmp_path,
            sample_rate=0.0,
            rate_limit_threshold=10,
            rng=rng,
        )
        for i in range(5):
            log.log_sample(
                image_png_bytes=_png_bytes(f"shot-{i}".encode()),
                state_machine_id=uuid4(),
                target_process="notepad++.exe",
                prompt="Save button",
                v5_bbox={"x": i, "y": i, "w": 1, "h": 1},
                v5_model="qontinui-grounding-v5",
                private=False,
            )

    inserts = _all_inserts(cursor)
    # sample_rate=0.0 ⇒ everything is dropped after the count check.
    assert len(inserts) == 0
    assert sorted(tmp_path.glob("*.png")) == []
    # Count was queried at least once (then cached).
    assert len(_all_counts(cursor)) >= 1


def test_rate_limit_with_positive_rate_passes_stochastically(
    tmp_path: Path,
) -> None:
    """With a seeded RNG we can predict the exact pass/drop pattern."""
    cursor = _FakeCursor(count_result=1000)
    factory = _ConnFactory([cursor])

    # Pre-compute the RNG sequence for the same seed so the expected
    # count matches the run's behavior regardless of the RNG's internal
    # version: a single Random(seed) instance drives both the reference
    # sequence and the actual logger.
    seed = 1
    sample_rate = 0.5
    reference = random.Random(seed)
    expected_draws = [reference.random() for _ in range(5)]
    expected_passes = sum(1 for r in expected_draws if r < sample_rate)

    with mock.patch("psycopg.connect", factory):
        log = ShadowSampleLogger(
            pg_url="postgres://fake",
            images_dir=tmp_path,
            sample_rate=sample_rate,
            rate_limit_threshold=10,
            rng=random.Random(seed),
        )
        for i in range(5):
            log.log_sample(
                image_png_bytes=_png_bytes(f"stoch-{i}".encode()),
                state_machine_id=uuid4(),
                target_process="notepad++.exe",
                prompt="Save button",
                v5_bbox={"x": i, "y": i, "w": 1, "h": 1},
                v5_model="qontinui-grounding-v5",
                private=False,
            )

    inserts = _all_inserts(cursor)
    # Must be strictly between 0 and 5 — the whole point of the test is
    # that the gate is probabilistic, not "always pass" or "always drop".
    assert (
        0 < expected_passes < 5
    ), "Test seed produced a degenerate pass count; pick a different seed."
    assert len(inserts) == expected_passes
    # Number of PNGs on disk matches number of passes.
    assert len(list(tmp_path.glob("*.png"))) == expected_passes


def test_below_threshold_no_rate_limiting(tmp_path: Path) -> None:
    """When row count is under threshold, every non-private sample inserts."""
    cursor = _FakeCursor(count_result=5)
    factory = _ConnFactory([cursor])

    with mock.patch("psycopg.connect", factory):
        log = ShadowSampleLogger(
            pg_url="postgres://fake",
            images_dir=tmp_path,
            sample_rate=0.0,  # would block everything if rate limiting active
            rate_limit_threshold=100,
        )
        for i in range(3):
            log.log_sample(
                image_png_bytes=_png_bytes(f"under-{i}".encode()),
                state_machine_id=uuid4(),
                target_process="notepad++.exe",
                prompt="Prompt",
                v5_bbox={"x": i, "y": i, "w": 1, "h": 1},
                v5_model="qontinui-grounding-v5",
                private=False,
            )

    inserts = _all_inserts(cursor)
    assert len(inserts) == 3
    assert len(list(tmp_path.glob("*.png"))) == 3


def test_exception_in_db_insert_is_swallowed(tmp_path: Path) -> None:
    """Any psycopg error must not propagate to the caller."""

    def _boom(pg_url: str) -> _FakeConn:
        raise RuntimeError("connection refused")

    with mock.patch("psycopg.connect", _boom):
        log = ShadowSampleLogger(
            pg_url="postgres://fake",
            images_dir=tmp_path,
        )
        # Must not raise.
        log.log_sample(
            image_png_bytes=_png_bytes(b"boom"),
            state_machine_id=uuid4(),
            target_process="notepad++.exe",
            prompt="Whatever",
            v5_bbox={"x": 0, "y": 0, "w": 1, "h": 1},
            v5_model="qontinui-grounding-v5",
            private=False,
        )
