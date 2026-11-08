"""Seam test: mock-mode find authenticates to the historical data API.

Enters where the population enters -- ``MockFindImplementation._execute_single``,
the per-pattern step of mock-mode ``FindWrapper.find`` -- rather than at the
HTTP client, so it observes that a find in mock mode carries the fleet
credential and reads a refused request as no data instead of crashing.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import numpy as np
import pytest

from qontinui.actions.find import mock_find_implementation as mfi
from qontinui.actions.find.find_options import FindOptions
from qontinui.actions.find.find_result import FindResult
from qontinui.actions.find.mock_find_implementation import MockFindImplementation
from qontinui.mock import historical_data_client as hdc
from qontinui.model.element import Pattern

API_BASE = "https://api.qontinui.io"


def _jwt(marker: str) -> str:
    def seg(obj: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

    payload = {"exp": int(time.time()) + 3600, "user_id": marker}
    return f"{seg({'alg': 'none'})}.{seg(payload)}.sig-{marker}"


DEVICE_TOKEN = _jwt("seam-device-secret")

HISTORICAL_ROW = {
    "id": 1,
    "pattern_id": "seam_pattern",
    "pattern_name": "seam_pattern",
    "action_type": "FIND",
    "active_states": [],
    "success": True,
    "match_count": 1,
    "best_match_score": 0.97,
    "match_x": 111,
    "match_y": 222,
    "match_width": 33,
    "match_height": 44,
    "frame_timestamp_ms": None,
    "has_frame": False,
}


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    for name in ("QONTINUI_WEB_TOKEN", "QONTINUI_API_URL", "QONTINUI_API_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QONTINUI_WEB_URL", API_BASE)
    home = tmp_path / "home"
    (home / ".qontinui").mkdir(parents=True)
    (home / ".qontinui" / "coord-device-jwt").write_text(DEVICE_TOKEN + "\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(hdc.HistoricalDataClient, "_instance", None)
    monkeypatch.setattr(hdc, "_client", None)
    monkeypatch.setattr(mfi, "USE_API_HISTORICAL_DATA", True)
    yield


def _serve(handler: httpx.MockTransport) -> None:
    client = hdc.get_historical_data_client()
    client._client = httpx.Client(base_url=client.api_url, transport=handler)


def _pattern() -> Pattern:
    return Pattern(
        id="seam_pattern",
        name="seam_pattern",
        pixel_data=np.zeros((8, 8, 3), dtype=np.uint8),
        mask=np.ones((8, 8), dtype=np.float32),
    )


def _find() -> FindResult:
    """One mock-mode find for one pattern, entered where FindWrapper enters."""
    return MockFindImplementation._execute_single(
        MockFindImplementation(), _pattern(), FindOptions(), set()
    )


def test_mock_find_sends_the_device_credential_and_uses_the_row() -> None:
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("Authorization")
        seen.append(auth)
        if auth != f"Bearer {DEVICE_TOKEN}":
            return httpx.Response(401, json={"detail": "unauthorized"})
        return httpx.Response(200, json=HISTORICAL_ROW)

    _serve(httpx.MockTransport(handler))
    result = _find()

    assert seen == [f"Bearer {DEVICE_TOKEN}"]
    assert result.found
    first = result.matches.first
    assert first is not None
    assert (first.x, first.y) == (111, 222)


def _assert_refusal_reads_as_no_data(
    status: int, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pin the generated-match fallback to "nothing found" so the outcome is
    # deterministic: a refusal must leave the find with no historical match.
    monkeypatch.setattr(MockFindImplementation, "_generate_mock_matches", lambda self, *a: [])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"detail": "refused"})

    _serve(httpx.MockTransport(handler))
    with caplog.at_level(logging.WARNING):
        result = _find()

    assert result is not None
    assert not result.found
    assert "no historical data: unauthorized" in caplog.text
    assert DEVICE_TOKEN not in caplog.text


def test_mock_find_reads_a_401_as_no_data(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _assert_refusal_reads_as_no_data(401, caplog, monkeypatch)


def test_mock_find_reads_a_403_as_no_data(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _assert_refusal_reads_as_no_data(403, caplog, monkeypatch)
