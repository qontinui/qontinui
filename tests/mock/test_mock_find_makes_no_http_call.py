"""Regression: mock-mode find makes no network call.

The historical-data client that fetched matches from qontinui-web's
``/api/v1/testing/historical/*`` routes was retired together with those routes
(plan 2026-11-06-verification-stores-nothing-writes-are-deleted-and-their-open-write-doors-close,
Phase 1). Mock mode now takes matches from the pattern's own history or
generates them, so a mock find must never reach the network.
"""

from __future__ import annotations

import asyncio
import importlib.util
import socket
from pathlib import Path

import httpx
import numpy as np
import pytest

from qontinui.actions.find import mock_find_implementation as mfi
from qontinui.actions.find.find_options import FindOptions
from qontinui.actions.find.find_result import FindResult
from qontinui.actions.find.mock_find_implementation import MockFindImplementation
from qontinui.model.element import Pattern


def _pattern() -> Pattern:
    return Pattern(
        id="no_http_pattern",
        name="no_http_pattern",
        pixel_data=np.zeros((8, 8, 3), dtype=np.uint8),
        mask=np.ones((8, 8), dtype=np.float32),
    )


def _find() -> FindResult:
    """One mock-mode find, entered where FindWrapper enters (``execute``)."""
    return asyncio.run(MockFindImplementation().execute([_pattern()], FindOptions()))[0]


def test_imports_this_trees_source() -> None:
    # A venv whose editable install points at another checkout would make the
    # tests below observe someone else's code; refuse that loudly.
    src = Path(__file__).resolve().parents[2] / "src"
    assert Path(mfi.__file__).resolve().is_relative_to(src)


def test_historical_data_client_module_is_gone() -> None:
    assert importlib.util.find_spec("qontinui.mock.historical_data_client") is None


def test_mock_find_sends_no_http_request_and_opens_no_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[str] = []

    def record_send(self: httpx.Client, request: httpx.Request, **_: object) -> httpx.Response:
        attempts.append(f"httpx {request.method} {request.url}")
        return httpx.Response(503, request=request)

    def record_connect(self: socket.socket, address: object) -> None:
        attempts.append(f"socket connect {address!r}")
        raise OSError("network disabled in this test")

    monkeypatch.setattr(httpx.Client, "send", record_send)
    monkeypatch.setattr(socket.socket, "connect", record_connect)

    result = _find()

    assert result is not None
    assert attempts == []
