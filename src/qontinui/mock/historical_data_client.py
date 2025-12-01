"""Client for fetching historical data from qontinui-api.

This module provides an interface for retrieving historical automation
results from the qontinui-api database during integration testing.

When running in mock mode with historical data enabled, the mock system
uses this client to fetch random historical results that match the
current action context, making each test run different.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional, cast

import httpx

logger = logging.getLogger(__name__)


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
    """Client for fetching historical data from qontinui-api.

    This client is used during integration testing to retrieve
    historical automation results. It supports:

    - Random result selection for realistic test variation
    - Filtering by pattern, action type, and active states
    - Frame retrieval for visual playback

    Configuration is via environment variables:
    - QONTINUI_API_URL: Base URL of qontinui-api (default: http://localhost:8001)
    - QONTINUI_API_ENABLED: Enable API calls (default: true in mock mode)
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

        self.api_url = os.getenv("QONTINUI_API_URL", "http://localhost:8001")
        self.enabled = os.getenv("QONTINUI_API_ENABLED", "true").lower() == "true"
        self.timeout = float(os.getenv("QONTINUI_API_TIMEOUT", "5.0"))
        self._client: httpx.Client | None = None
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
                "/api/capture/historical/random",
                json={
                    "pattern_id": pattern_id,
                    "action_type": action_type,
                    "active_states": active_states,
                    "success_only": success_only,
                    "workflow_id": workflow_id,
                    "project_id": project_id,
                },
            )

            if response.status_code == 200:
                data = response.json()
                if data:
                    return HistoricalMatchData.from_api_response(data)
                return None

            logger.warning(f"Historical data API returned {response.status_code}: {response.text}")
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
                f"/api/capture/historical/pattern/{pattern_id}",
                params=params,
            )

            if response.status_code == 200:
                data = response.json()
                return [HistoricalMatchData.from_api_response(item) for item in data]

            logger.warning(f"Historical data API returned {response.status_code}: {response.text}")
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
                f"/api/capture/frames/{historical_result_id}",
                params={"frame_type": frame_type},
            )

            if response.status_code == 200:
                return cast(bytes, response.content)

            logger.warning(f"Frame API returned {response.status_code}: {response.text}")
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
                "/api/capture/frames/playback",
                json={"historical_result_ids": historical_result_ids},
            )

            if response.status_code == 200:
                return cast(list[dict[Any, Any]], response.json())

            logger.warning(f"Playback API returned {response.status_code}: {response.text}")
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
