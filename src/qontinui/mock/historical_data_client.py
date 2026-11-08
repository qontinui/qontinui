"""Client for fetching historical data from qontinui-web.

This module provides an interface for retrieving historical automation
results from the qontinui-web backend during integration testing.

When running in mock mode with historical data enabled, the mock system
uses this client to fetch random historical results that match the
current action context, making each test run different.

Note: Historical data endpoints are served by qontinui-web (port 8000)
in the testing API namespace.
"""

import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast

import httpx

logger = logging.getLogger(__name__)

# The only base URL the fleet's device JWT file may be sent to. That file is a
# fleet credential nobody handed to this client, so a mis-set base URL -- and
# loopback, which is not an identity -- must never receive it. Any other base
# gets a bearer only from an explicit QONTINUI_WEB_TOKEN.
FILE_TOKEN_BASE_URL = "https://api.qontinui.io"

WEB_TOKEN_ENV = "QONTINUI_WEB_TOKEN"

UNAUTHORIZED_WARNING = "no historical data: unauthorized"

# The characters a bearer token may hold (JWT base64url plus RFC 6750's
# token68 set). Anything else -- an embedded CR/LF, a NUL, a non-ASCII
# character -- is treated as no token: httpx would refuse it as a header and
# the RequestError arms would log the refusal, header value included.
_TOKEN_RE = re.compile(r"[A-Za-z0-9._~+/=-]+")


def _device_jwt_path() -> Path:
    """Path of the fleet's device JWT file (``~/.qontinui/coord-device-jwt``)."""
    return Path.home() / ".qontinui" / "coord-device-jwt"


def _token_expired(token: str) -> bool:
    """Return True only when ``token`` is a JWT whose ``exp`` has passed.

    The payload is read unverified: this decides whether a token is worth
    sending, not whether it is genuine. A token with no readable ``exp`` is
    treated as live and left for the server to judge.
    """
    parts = token.split(".")
    if len(parts) != 3:
        return False
    payload = parts[1]
    try:
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except (ValueError, RecursionError):
        return False
    if not isinstance(claims, dict):
        return False
    exp = claims.get("exp")
    if isinstance(exp, bool) or not isinstance(exp, int | float):
        return False
    return exp <= time.time()


def _read_device_jwt() -> str | None:
    """Read the device JWT file, or None when it is absent or unreadable."""
    try:
        token = _device_jwt_path().read_text(encoding="utf-8").strip()
    except (OSError, RuntimeError, ValueError):
        return None
    return token or None


def _usable(token: str) -> bool:
    """True when ``token`` is well-formed and its ``exp`` has not passed."""
    if not token or _TOKEN_RE.fullmatch(token) is None:
        return False
    return not _token_expired(token)


@dataclass
class HistoricalMatchData:
    """Data from a historical match result."""

    id: int
    pattern_id: str | None
    pattern_name: str | None
    action_type: str
    active_states: list[str]
    success: bool
    match_count: int | None
    best_match_score: float | None
    match_x: int | None
    match_y: int | None
    match_width: int | None
    match_height: int | None
    frame_timestamp_ms: int | None
    has_frame: bool

    @classmethod
    def from_api_response(cls, data: dict) -> "HistoricalMatchData":
        """Create from API response dict."""
        return cls(
            id=data["id"],
            pattern_id=data.get("pattern_id"),
            pattern_name=data.get("pattern_name"),
            action_type=data["action_type"],
            active_states=data.get("active_states") or [],
            success=data["success"],
            match_count=data.get("match_count"),
            best_match_score=data.get("best_match_score"),
            match_x=data.get("match_x"),
            match_y=data.get("match_y"),
            match_width=data.get("match_width"),
            match_height=data.get("match_height"),
            frame_timestamp_ms=data.get("frame_timestamp_ms"),
            has_frame=data.get("has_frame", False),
        )


class HistoricalDataClient:
    """Client for fetching historical data from qontinui-web.

    This client is used during integration testing to retrieve
    historical automation results. It supports:

    - Random result selection for realistic test variation
    - Filtering by pattern, action type, and active states
    - Frame retrieval for visual playback

    Configuration is via environment variables:
    - QONTINUI_WEB_URL: Base URL of qontinui-web (default: http://localhost:8000)
    - QONTINUI_API_URL: Legacy alias for QONTINUI_WEB_URL
    - QONTINUI_API_ENABLED: Enable API calls (default: true in mock mode)
    - QONTINUI_WEB_TOKEN: Bearer token sent to whichever base URL is configured

    Credentials are resolved per request: QONTINUI_WEB_TOKEN, else the device
    JWT file ``~/.qontinui/coord-device-jwt`` -- the latter only when the base
    URL is exactly ``https://api.qontinui.io``. A token whose ``exp`` has passed
    is skipped. A 401 or 403 is read as no data, never as a crash.
    """

    _instance: Optional["HistoricalDataClient"] = None
    _initialized: bool = False

    def __new__(cls) -> "HistoricalDataClient":
        """Singleton pattern for client instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the client (only once for singleton)."""
        if self._initialized:
            return

        # Use QONTINUI_WEB_URL (new) or fall back to QONTINUI_API_URL (legacy)
        self.api_url = os.getenv(
            "QONTINUI_WEB_URL",
            os.getenv("QONTINUI_API_URL", "http://localhost:8000"),
        )
        self.enabled = os.getenv("QONTINUI_API_ENABLED", "true").lower() == "true"
        self.timeout = float(os.getenv("QONTINUI_API_TIMEOUT", "5.0"))
        self._client: httpx.Client | None = None
        self._unauthorized_warned = False
        self._initialized = True

        logger.info(
            f"HistoricalDataClient initialized: api_url={self.api_url}, enabled={self.enabled}"
        )

    @property
    def client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.api_url,
                timeout=self.timeout,
            )
        return self._client

    def _resolve_bearer(self) -> str | None:
        """Resolve the bearer token for one request, or None.

        Never log the returned value.
        """
        env_token = os.getenv(WEB_TOKEN_ENV, "").strip()
        if _usable(env_token):
            return env_token
        if self.api_url.rstrip("/") != FILE_TOKEN_BASE_URL:
            return None
        file_token = _read_device_jwt()
        if file_token is not None and _usable(file_token):
            return file_token
        return None

    def _auth_headers(self) -> dict[str, str]:
        """Authorization header for one request, empty when no token resolves."""
        token = self._resolve_bearer()
        if token is None:
            return {}
        return {"Authorization": f"Bearer {token}"}

    def _is_unauthorized(self, response: httpx.Response) -> bool:
        """True for a 401/403, warning once per client that no data was read."""
        if response.status_code not in (401, 403):
            return False
        if not self._unauthorized_warned:
            self._unauthorized_warned = True
            logger.warning(f"{UNAUTHORIZED_WARNING} (HTTP {response.status_code})")
        return True

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            self._client.close()
            self._client = None

    def get_random_historical_result(
        self,
        pattern_id: str | None = None,
        action_type: str | None = None,
        active_states: list[str] | None = None,
        success_only: bool = True,
        workflow_id: int | None = None,
        project_id: int | None = None,
    ) -> HistoricalMatchData | None:
        """Get a random historical result matching criteria.

        This is the main method for integration testing. It returns
        a randomly selected historical result, making each test
        run different.

        Args:
            pattern_id: Filter by pattern ID
            action_type: Filter by action type (e.g., "FIND", "CLICK")
            active_states: Filter by active states (any match)
            success_only: Only return successful results
            workflow_id: Filter by workflow ID
            project_id: Filter by project ID

        Returns:
            HistoricalMatchData or None if no matches or API disabled
        """
        if not self.enabled:
            logger.debug("Historical data API disabled")
            return None

        try:
            response = self.client.post(
                "/api/v1/testing/historical/random",
                json={
                    "pattern_id": pattern_id,
                    "action_type": action_type,
                    "active_states": active_states,
                    "success_only": success_only,
                    "workflow_id": workflow_id,
                    "project_id": str(project_id) if project_id else None,
                },
                headers=self._auth_headers(),
            )

            if response.status_code == 200:
                data = response.json()
                if data:
                    return HistoricalMatchData.from_api_response(data)
                return None

            if self._is_unauthorized(response):
                return None

            logger.warning(
                f"Historical data API returned {response.status_code}: {response.text}"
            )
            return None

        except httpx.RequestError as e:
            logger.warning(f"Failed to fetch historical data: {e}")
            return None

    def get_historical_results_for_pattern(
        self,
        pattern_id: str,
        action_type: str,
        active_states: list[str] | None = None,
        limit: int = 10,
    ) -> list[HistoricalMatchData]:
        """Get all historical results for a pattern context.

        Args:
            pattern_id: Pattern ID
            action_type: Action type
            active_states: Filter by active states
            limit: Maximum results to return

        Returns:
            List of HistoricalMatchData objects
        """
        if not self.enabled:
            return []

        try:
            params: dict[str, str | int] = {
                "action_type": action_type,
                "limit": limit,
            }
            if active_states:
                params["active_states"] = ",".join(active_states)

            response = self.client.get(
                f"/api/v1/testing/historical/pattern/{pattern_id}",
                params=params,
                headers=self._auth_headers(),
            )

            if response.status_code == 200:
                data = response.json()
                return [HistoricalMatchData.from_api_response(item) for item in data]

            if self._is_unauthorized(response):
                return []

            logger.warning(
                f"Historical data API returned {response.status_code}: {response.text}"
            )
            return []

        except httpx.RequestError as e:
            logger.warning(f"Failed to fetch historical results: {e}")
            return []

    def get_frame_for_result(
        self,
        historical_result_id: int,
        frame_type: str = "action",
    ) -> bytes | None:
        """Get the frame image for a historical result.

        Args:
            historical_result_id: Historical result ID
            frame_type: Frame type ('before', 'action', 'after', 'result')

        Returns:
            Frame image data as bytes (JPEG), or None
        """
        if not self.enabled:
            return None

        try:
            response = self.client.get(
                f"/api/v1/testing/historical/frames/{historical_result_id}",
                params={"frame_type": frame_type},
                headers=self._auth_headers(),
            )

            if response.status_code == 200:
                return cast(bytes, response.content)

            if self._is_unauthorized(response):
                return None

            logger.warning(
                f"Frame API returned {response.status_code}: {response.text}"
            )
            return None

        except httpx.RequestError as e:
            logger.warning(f"Failed to fetch frame: {e}")
            return None

    def get_playback_frames(
        self,
        historical_result_ids: list[int],
    ) -> list[dict]:
        """Get frames for integration test playback.

        Args:
            historical_result_ids: List of historical result IDs in order

        Returns:
            List of dicts with frame data and metadata
        """
        if not self.enabled:
            return []

        try:
            response = self.client.post(
                "/api/v1/testing/historical/playback",
                json={"historical_result_ids": historical_result_ids},
                headers=self._auth_headers(),
            )

            if response.status_code == 200:
                return cast(list[dict[Any, Any]], response.json())

            if self._is_unauthorized(response):
                return []

            logger.warning(
                f"Playback API returned {response.status_code}: {response.text}"
            )
            return []

        except httpx.RequestError as e:
            logger.warning(f"Failed to fetch playback frames: {e}")
            return []


# Global client instance
_client: HistoricalDataClient | None = None


def get_historical_data_client() -> HistoricalDataClient:
    """Get the global historical data client instance."""
    global _client
    if _client is None:
        _client = HistoricalDataClient()
    return _client
