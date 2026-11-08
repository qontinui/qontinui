"""Tests for the credential half of the mock-mode historical data client.

The client resolves a bearer per request: ``QONTINUI_WEB_TOKEN``, else the
fleet's device JWT file -- the latter only for the exact base URL
``https://api.qontinui.io``, never for loopback or any other base. A 401/403
reads as no data with one warning, and the token never reaches a log.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest

from qontinui.mock import historical_data_client as hdc
from qontinui.mock.historical_data_client import (
    FILE_TOKEN_BASE_URL,
    UNAUTHORIZED_WARNING,
    HistoricalDataClient,
)

LOOPBACK_DEFAULT = "http://localhost:8000"
FOREIGN_BASE = "https://api.example.invalid"

RESULT = {
    "id": 7,
    "pattern_id": "p",
    "pattern_name": "n",
    "action_type": "FIND",
    "active_states": ["s"],
    "success": True,
    "match_count": 1,
    "best_match_score": 0.9,
    "match_x": 1,
    "match_y": 2,
    "match_width": 3,
    "match_height": 4,
    "frame_timestamp_ms": None,
    "has_frame": False,
}


def _jwt(exp_offset: int, marker: str) -> str:
    """Build an unsigned JWT-shaped token whose exp is now + exp_offset."""

    def seg(obj: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

    payload = {"exp": int(time.time()) + exp_offset, "user_id": marker}
    return f"{seg({'alg': 'none'})}.{seg(payload)}.sig-{marker}"


ENV_TOKEN = _jwt(3600, "env-secret")
FILE_TOKEN = _jwt(3600, "file-secret")
EXPIRED_ENV_TOKEN = _jwt(-60, "expired-env-secret")
EXPIRED_FILE_TOKEN = _jwt(-60, "expired-file-secret")
ALL_TOKENS = (ENV_TOKEN, FILE_TOKEN, EXPIRED_ENV_TOKEN, EXPIRED_FILE_TOKEN)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Fresh singleton, no ambient env token, and a HOME with no device JWT."""
    for name in (
        "QONTINUI_WEB_TOKEN",
        "QONTINUI_WEB_URL",
        "QONTINUI_API_URL",
        "QONTINUI_API_ENABLED",
        "QONTINUI_API_TIMEOUT",
    ):
        monkeypatch.delenv(name, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(HistoricalDataClient, "_instance", None)
    monkeypatch.setattr(hdc, "_client", None)
    yield


def _write_device_jwt(token: str) -> None:
    path = Path.home() / ".qontinui" / "coord-device-jwt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token + "\n", encoding="utf-8")


class Recorder:
    """MockTransport handler that records requests and answers a fixed status."""

    def __init__(self, status: int = 200, body: object = RESULT) -> None:
        self.status = status
        self.body = body
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status == 200:
            return httpx.Response(200, json=self.body)
        return httpx.Response(self.status, json={"detail": "nope"})

    @property
    def auth(self) -> str | None:
        assert self.requests, "no request reached the transport"
        value: str | None = self.requests[-1].headers.get("Authorization")
        return value


def _client(
    monkeypatch: pytest.MonkeyPatch, base_url: str | None, recorder: Recorder
) -> HistoricalDataClient:
    if base_url is not None:
        monkeypatch.setenv("QONTINUI_WEB_URL", base_url)
    client = HistoricalDataClient()
    client._client = httpx.Client(base_url=client.api_url, transport=httpx.MockTransport(recorder))
    return client


# --- credential resolution -------------------------------------------------


def test_env_token_is_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QONTINUI_WEB_TOKEN", ENV_TOKEN)
    rec = Recorder()
    result = _client(monkeypatch, FILE_TOKEN_BASE_URL, rec).get_random_historical_result()
    assert result is not None and result.id == 7
    assert rec.auth == f"Bearer {ENV_TOKEN}"


def test_file_token_is_sent_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    _write_device_jwt(FILE_TOKEN)
    rec = Recorder()
    _client(monkeypatch, FILE_TOKEN_BASE_URL, rec).get_random_historical_result()
    assert rec.auth == f"Bearer {FILE_TOKEN}"


def test_env_token_wins_over_file_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QONTINUI_WEB_TOKEN", ENV_TOKEN)
    _write_device_jwt(FILE_TOKEN)
    rec = Recorder()
    _client(monkeypatch, FILE_TOKEN_BASE_URL, rec).get_random_historical_result()
    assert rec.auth == f"Bearer {ENV_TOKEN}"


def test_expired_env_token_falls_through_to_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QONTINUI_WEB_TOKEN", EXPIRED_ENV_TOKEN)
    _write_device_jwt(FILE_TOKEN)
    rec = Recorder()
    _client(monkeypatch, FILE_TOKEN_BASE_URL, rec).get_random_historical_result()
    assert rec.auth == f"Bearer {FILE_TOKEN}"


def test_expired_file_token_is_not_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    _write_device_jwt(EXPIRED_FILE_TOKEN)
    rec = Recorder()
    _client(monkeypatch, FILE_TOKEN_BASE_URL, rec).get_random_historical_result()
    assert rec.auth is None


def test_refreshed_file_token_is_picked_up_per_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_device_jwt(EXPIRED_FILE_TOKEN)
    rec = Recorder()
    client = _client(monkeypatch, FILE_TOKEN_BASE_URL, rec)
    client.get_random_historical_result()
    assert rec.auth is None
    _write_device_jwt(FILE_TOKEN)
    client.get_random_historical_result()
    assert rec.auth == f"Bearer {FILE_TOKEN}"


def test_no_header_when_neither_token_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = Recorder()
    _client(monkeypatch, FILE_TOKEN_BASE_URL, rec).get_random_historical_result()
    assert rec.auth is None


def test_foreign_base_gets_no_file_token(monkeypatch: pytest.MonkeyPatch) -> None:
    _write_device_jwt(FILE_TOKEN)
    rec = Recorder()
    _client(monkeypatch, FOREIGN_BASE, rec).get_random_historical_result()
    assert rec.auth is None


def test_foreign_base_still_gets_explicit_env_token(monkeypatch: pytest.MonkeyPatch) -> None:
    _write_device_jwt(FILE_TOKEN)
    monkeypatch.setenv("QONTINUI_WEB_TOKEN", ENV_TOKEN)
    rec = Recorder()
    _client(monkeypatch, FOREIGN_BASE, rec).get_random_historical_result()
    assert rec.auth == f"Bearer {ENV_TOKEN}"


@pytest.mark.parametrize(
    "base",
    [
        "https://api.qontinui.io.example.invalid",
        "http://api.qontinui.io",
        "https://api.qontinui.io:8443",
        "https://API.qontinui.io",
        "https://evil@api.qontinui.io",
        "https://api.qontinui.io/api",
    ],
)
def test_lookalike_base_gets_no_file_token(monkeypatch: pytest.MonkeyPatch, base: str) -> None:
    _write_device_jwt(FILE_TOKEN)
    rec = Recorder()
    client = _client(monkeypatch, base, rec)
    # httpx replaces Authorization with its own Basic header for URL userinfo
    # before the transport sees it, so assert on the resolution, not the wire.
    assert client._resolve_bearer() is None
    client.get_random_historical_result()
    assert FILE_TOKEN not in (rec.auth or "")


def test_trailing_slash_api_base_gets_file_token(monkeypatch: pytest.MonkeyPatch) -> None:
    _write_device_jwt(FILE_TOKEN)
    rec = Recorder()
    _client(monkeypatch, FILE_TOKEN_BASE_URL + "/", rec).get_random_historical_result()
    assert rec.auth == f"Bearer {FILE_TOKEN}"


@pytest.mark.parametrize("base", [LOOPBACK_DEFAULT, FOREIGN_BASE])
def test_expired_env_token_is_not_sent_off_the_api_base(
    monkeypatch: pytest.MonkeyPatch, base: str
) -> None:
    _write_device_jwt(FILE_TOKEN)
    monkeypatch.setenv("QONTINUI_WEB_TOKEN", EXPIRED_ENV_TOKEN)
    rec = Recorder()
    _client(monkeypatch, base, rec).get_random_historical_result()
    assert rec.auth is None


def _h11_like(request: httpx.Request) -> httpx.Response:
    """Refuse an illegal header value the way h11 does, echoing it in the error.

    MockTransport validates nothing, so this stands in for h11: a value holding
    a byte outside visible ASCII is refused with the value in the message, which
    the client's unchanged RequestError arms would then log.
    """
    for value in request.headers.raw:
        if any(b < 0x20 or b > 0x7E for b in value[1]):
            raise httpx.LocalProtocolError(f"Illegal header value {value[1]!r}")
    return httpx.Response(200, json=RESULT)


@pytest.mark.parametrize("bad", ["\r\nX-Evil: 1", "\t", "\u00e9"], ids=["crlf", "tab", "non-ascii"])
def test_malformed_env_token_is_not_sent_and_never_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, bad: str
) -> None:
    token = f"SECRETHEAD{bad}SECRETTAIL"
    monkeypatch.setenv("QONTINUI_WEB_TOKEN", token)
    client = HistoricalDataClient()
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("Authorization"))
        return _h11_like(request)

    client._client = httpx.Client(base_url=client.api_url, transport=httpx.MockTransport(handler))
    with caplog.at_level(logging.DEBUG):
        result = client.get_random_historical_result()
    assert result is not None
    assert seen == [None]
    assert "SECRETHEAD" not in caplog.text
    assert "SECRETTAIL" not in caplog.text


def test_malformed_file_token_is_not_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    _write_device_jwt(FILE_TOKEN + "\n" + FILE_TOKEN)
    rec = Recorder()
    _client(monkeypatch, FILE_TOKEN_BASE_URL, rec).get_random_historical_result()
    assert rec.auth is None


def test_non_utf8_device_jwt_file_reads_as_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    path = Path.home() / ".qontinui" / "coord-device-jwt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xfe\x00garbage")
    rec = Recorder()
    result = _client(monkeypatch, FILE_TOKEN_BASE_URL, rec).get_random_historical_result()
    assert result is not None
    assert rec.auth is None


def test_loopback_default_gets_no_file_token(monkeypatch: pytest.MonkeyPatch) -> None:
    _write_device_jwt(FILE_TOKEN)
    rec = Recorder()
    client = _client(monkeypatch, None, rec)
    assert client.api_url == LOOPBACK_DEFAULT
    client.get_random_historical_result()
    assert rec.auth is None


def test_loopback_default_gets_explicit_env_token(monkeypatch: pytest.MonkeyPatch) -> None:
    _write_device_jwt(FILE_TOKEN)
    monkeypatch.setenv("QONTINUI_WEB_TOKEN", ENV_TOKEN)
    rec = Recorder()
    client = _client(monkeypatch, None, rec)
    assert client.api_url == LOOPBACK_DEFAULT
    client.get_random_historical_result()
    assert rec.auth == f"Bearer {ENV_TOKEN}"


@pytest.mark.parametrize(
    ("call", "body"),
    [
        (lambda c: c.get_historical_results_for_pattern("p", "FIND"), [RESULT]),
        (lambda c: c.get_frame_for_result(7), RESULT),
        (lambda c: c.get_playback_frames([7]), [RESULT]),
    ],
    ids=["pattern", "frame", "playback"],
)
def test_every_read_sends_the_credential(
    monkeypatch: pytest.MonkeyPatch,
    call: Callable[[HistoricalDataClient], object],
    body: object,
) -> None:
    _write_device_jwt(FILE_TOKEN)
    rec = Recorder(body=body)
    call(_client(monkeypatch, FILE_TOKEN_BASE_URL, rec))
    assert rec.auth == f"Bearer {FILE_TOKEN}"


# --- refused requests read as no data --------------------------------------


@pytest.mark.parametrize("status", [401, 403])
def test_refused_request_returns_no_data_and_warns_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, status: int
) -> None:
    monkeypatch.setenv("QONTINUI_WEB_TOKEN", ENV_TOKEN)
    rec = Recorder(status=status)
    client = _client(monkeypatch, FILE_TOKEN_BASE_URL, rec)
    with caplog.at_level(logging.WARNING, logger=hdc.__name__):
        assert client.get_random_historical_result() is None
        assert client.get_historical_results_for_pattern("p", "FIND") == []
        assert client.get_frame_for_result(7) is None
        assert client.get_playback_frames([7]) == []
    warnings = [r for r in caplog.records if UNAUTHORIZED_WARNING in r.getMessage()]
    assert len(warnings) == 1
    assert warnings[0].levelno == logging.WARNING
    assert len(rec.requests) == 4


def test_other_non_200_keeps_its_existing_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    rec = Recorder(status=500)
    client = _client(monkeypatch, FILE_TOKEN_BASE_URL, rec)
    with caplog.at_level(logging.WARNING, logger=hdc.__name__):
        assert client.get_random_historical_result() is None
    assert "Historical data API returned 500" in caplog.text
    assert UNAUTHORIZED_WARNING not in caplog.text


def test_request_error_still_returns_no_data(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    monkeypatch.setenv("QONTINUI_WEB_TOKEN", ENV_TOKEN)
    client = HistoricalDataClient()
    client._client = httpx.Client(base_url=client.api_url, transport=httpx.MockTransport(boom))
    assert client.get_random_historical_result() is None
    assert client.get_playback_frames([1]) == []


def test_request_error_arm_never_logs_the_token(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    monkeypatch.setenv("QONTINUI_WEB_TOKEN", EXPIRED_ENV_TOKEN)
    _write_device_jwt(FILE_TOKEN)
    monkeypatch.setenv("QONTINUI_WEB_URL", FILE_TOKEN_BASE_URL)
    client = HistoricalDataClient()
    client._client = httpx.Client(base_url=client.api_url, transport=httpx.MockTransport(boom))
    with caplog.at_level(logging.DEBUG):
        assert client.get_random_historical_result() is None
        assert client.get_historical_results_for_pattern("p", "FIND") == []
        assert client.get_frame_for_result(7) is None
        assert client.get_playback_frames([7]) == []
    assert "connection refused" in caplog.text
    for token in ALL_TOKENS:
        assert token not in caplog.text
        assert token.split(".")[1] not in caplog.text


# --- the token never reaches a log ----------------------------------------


def test_caplog_never_holds_a_token(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    scenarios = [
        (ENV_TOKEN, FILE_TOKEN, FILE_TOKEN_BASE_URL, 200),
        (EXPIRED_ENV_TOKEN, FILE_TOKEN, FILE_TOKEN_BASE_URL, 401),
        (None, FILE_TOKEN, FILE_TOKEN_BASE_URL, 403),
        (ENV_TOKEN, EXPIRED_FILE_TOKEN, LOOPBACK_DEFAULT, 500),
        (None, FILE_TOKEN, FOREIGN_BASE, 401),
    ]
    for env_token, file_token, base, status in scenarios:
        monkeypatch.setattr(HistoricalDataClient, "_instance", None)
        if env_token is None:
            monkeypatch.delenv("QONTINUI_WEB_TOKEN", raising=False)
        else:
            monkeypatch.setenv("QONTINUI_WEB_TOKEN", env_token)
        _write_device_jwt(file_token)
        client = _client(monkeypatch, base, Recorder(status=status))
        client.get_random_historical_result()
        client.get_playback_frames([7])
    assert caplog.records, "scenarios logged nothing, so the check below is vacuous"
    for token in ALL_TOKENS:
        assert token not in caplog.text
        assert token.split(".")[1] not in caplog.text
