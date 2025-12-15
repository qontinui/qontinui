"""Result streaming to remote servers (e.g., qontinui-web)."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ResultStreamer:
    """Stream test results to a remote server in real-time.

    This class enables streaming test results to qontinui-web or other servers
    as tests complete, allowing for real-time monitoring of CI/CD test runs.
    """

    def __init__(self, url: str, api_token: str | None = None):
        """Initialize result streamer.

        Args:
            url: URL endpoint to stream results to
            api_token: Optional API token for authentication
        """
        self.url = url
        self.api_token = api_token
        self._validate_url()

    def _validate_url(self):
        """Validate the streaming URL.

        Raises:
            ValueError: If URL is invalid
        """
        if not self.url:
            raise ValueError("Streaming URL cannot be empty")

        if not (self.url.startswith("http://") or self.url.startswith("https://")):
            raise ValueError(f"Invalid URL scheme. Must be http:// or https://: {self.url}")

    def stream_result(self, result: dict[str, Any]) -> bool:
        """Stream a single test result to the server.

        Args:
            result: Test result dictionary

        Returns:
            True if streaming succeeded, False otherwise
        """
        try:
            # Lazy import to avoid hard dependency
            import requests

            payload = {
                "workflow_id": result["workflow_id"],
                "workflow_name": result["workflow_name"],
                "success": result["success"],
                "duration": result["duration"],
                "timestamp": result["start_time"],
                "error": result.get("error"),
            }

            headers = {"Content-Type": "application/json"}
            if self.api_token:
                headers["Authorization"] = f"Bearer {self.api_token}"

            response = requests.post(
                self.url,
                json=payload,
                timeout=10,
                headers=headers,
            )

            if response.status_code >= 400:
                logger.warning(
                    f"Failed to stream result to {self.url}: "
                    f"HTTP {response.status_code} - {response.text}"
                )
                return False

            logger.debug(f"Successfully streamed result for {result['workflow_name']}")
            return True

        except ImportError:
            logger.error("requests library not available. Install with: pip install requests")
            return False
        except Exception as e:
            logger.warning(f"Failed to stream result to {self.url}: {e}")
            return False

    def stream_summary(self, summary: dict[str, Any]) -> bool:
        """Stream test summary to the server.

        Args:
            summary: Test summary dictionary

        Returns:
            True if streaming succeeded, False otherwise
        """
        try:
            import requests

            headers = {"Content-Type": "application/json"}
            if self.api_token:
                headers["Authorization"] = f"Bearer {self.api_token}"

            response = requests.post(
                self.url,
                json=summary,
                timeout=10,
                headers=headers,
            )

            if response.status_code >= 400:
                logger.warning(
                    f"Failed to stream summary to {self.url}: "
                    f"HTTP {response.status_code} - {response.text}"
                )
                return False

            logger.debug("Successfully streamed test summary")
            return True

        except ImportError:
            logger.error("requests library not available. Install with: pip install requests")
            return False
        except Exception as e:
            logger.warning(f"Failed to stream summary to {self.url}: {e}")
            return False
