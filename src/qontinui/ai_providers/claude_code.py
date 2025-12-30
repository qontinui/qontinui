"""Claude Code AI provider implementation.

This provider invokes Claude Code via the qontinui-runner API.
It handles submitting prompts and polling for task completion.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from .base import AIProvider, AnalysisRequest, AnalysisResult

logger = logging.getLogger(__name__)

# Default runner URL
DEFAULT_RUNNER_URL = "http://localhost:9876"

# Default polling interval in seconds
DEFAULT_POLL_INTERVAL = 1.0


def _json_dumps(obj: Any) -> str:
    """Simple JSON serialization without external dependency."""
    import json

    return json.dumps(obj)


def _json_loads(data: str) -> Any:
    """Simple JSON deserialization without external dependency."""
    import json

    return json.loads(data)


class ClaudeCodeProvider(AIProvider):
    """AI provider that uses Claude Code via qontinui-runner.

    This provider submits prompts to the qontinui-runner's /prompts/run API
    and polls for completion via /task-runs/{id}.

    Configuration via environment variables:
        QONTINUI_RUNNER_URL: URL of the runner API (default: http://localhost:9876)
    """

    def __init__(self, runner_url: str | None = None) -> None:
        """Initialize the provider.

        Args:
            runner_url: URL of the qontinui-runner API. Defaults to localhost:9876.
        """
        import os

        self._runner_url = runner_url or os.environ.get("QONTINUI_RUNNER_URL", DEFAULT_RUNNER_URL)

    @property
    def name(self) -> str:
        """Get the provider name."""
        return "claude_code"

    @property
    def description(self) -> str:
        """Get provider description."""
        return f"Claude Code via qontinui-runner ({self._runner_url})"

    def is_available(self) -> bool:
        """Check if qontinui-runner is available.

        Returns:
            True if runner API is reachable
        """
        try:
            url = f"{self._runner_url}/health"
            request = Request(url, method="GET")
            request.add_header("Accept", "application/json")

            with urlopen(request, timeout=5) as response:
                return bool(response.status == 200)
        except Exception as e:
            logger.debug(f"Runner not available: {e}")
            return False

    def analyze(
        self,
        request: AnalysisRequest,
        *,
        task_name: str | None = None,
        max_sessions: int | None = 1,
        image_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
        trace_path: str | None = None,
    ) -> AnalysisResult:
        """Run analysis synchronously via the runner API.

        Args:
            request: The analysis request
            task_name: Name for the task in the runner UI
            max_sessions: Maximum Claude sessions (1 = one-shot, None = unlimited)
            image_paths: Paths to images to include in the prompt
            video_paths: Paths to videos for frame extraction
            trace_path: Path to Playwright trace file

        Returns:
            The analysis result
        """
        result = AnalysisResult(success=False, provider=self.name)

        try:
            # Submit the prompt
            task_run_id = self._submit_prompt(
                prompt=request.prompt,
                task_name=task_name or "ai-analysis",
                max_sessions=max_sessions,
                timeout_seconds=request.timeout_seconds,
                image_paths=image_paths,
                video_paths=video_paths,
                trace_path=trace_path,
            )

            if not task_run_id:
                result.error = "Failed to submit prompt to runner"
                return result

            logger.info(f"Task submitted: {task_run_id} (timeout: {request.timeout_seconds}s)")

            # Poll for completion
            output, error = self._poll_task_completion(task_run_id, request.timeout_seconds)

            if error:
                result.error = error
                result.metadata["task_run_id"] = task_run_id
                logger.error(f"Task {task_run_id} failed: {error}")
            else:
                result.success = True
                result.output = output
                result.metadata["task_run_id"] = task_run_id
                logger.info(f"Task {task_run_id} completed successfully")

        except TimeoutError:
            result.error = f"Analysis timed out after {request.timeout_seconds} seconds"
            logger.error(result.error)

        except URLError as e:
            result.error = f"Failed to connect to runner: {e}"
            logger.error(result.error)

        except Exception as e:
            result.error = f"Failed to run analysis: {e}"
            logger.error(result.error, exc_info=True)

        return result

    async def stream_analyze(
        self,
        request: AnalysisRequest,
        *,
        task_name: str | None = None,
        max_sessions: int | None = 1,
        image_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
        trace_path: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream analysis output asynchronously.

        Args:
            request: The analysis request
            task_name: Name for the task in the runner UI
            max_sessions: Maximum Claude sessions (1 = one-shot, None = unlimited)
            image_paths: Paths to images to include in the prompt
            video_paths: Paths to videos for frame extraction
            trace_path: Path to Playwright trace file

        Yields:
            Lines of output from the task
        """
        try:
            # Submit the prompt
            task_run_id = self._submit_prompt(
                prompt=request.prompt,
                task_name=task_name or "ai-analysis",
                max_sessions=max_sessions,
                timeout_seconds=request.timeout_seconds,
                image_paths=image_paths,
                video_paths=video_paths,
                trace_path=trace_path,
            )

            if not task_run_id:
                yield "[ERROR: Failed to submit prompt to runner]\n"
                return

            logger.info(f"Streaming task: {task_run_id}")

            # Poll and stream output
            async for chunk in self._poll_task_output_async(task_run_id, request.timeout_seconds):
                yield chunk

        except TimeoutError:
            yield f"\n[ERROR: Analysis timed out after {request.timeout_seconds}s]\n"
            logger.error("Streaming analysis timed out")

        except URLError as e:
            yield f"\n[ERROR: Failed to connect to runner: {e}]\n"
            logger.error(f"Runner connection error: {e}")

        except Exception as e:
            yield f"\n[ERROR: {e}]\n"
            logger.error(f"Streaming analysis failed: {e}", exc_info=True)

    def _submit_prompt(
        self,
        prompt: str,
        task_name: str,
        max_sessions: int | None,
        timeout_seconds: int,
        image_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
        trace_path: str | None = None,
    ) -> str | None:
        """Submit a prompt to the runner API.

        Args:
            prompt: The prompt content
            task_name: Name for the task
            max_sessions: Maximum sessions to spawn
            timeout_seconds: Timeout for the request
            image_paths: Optional image paths
            video_paths: Optional video paths
            trace_path: Optional trace path

        Returns:
            Task run ID or None if failed
        """
        url = f"{self._runner_url}/prompts/run"

        # Build request body
        body: dict[str, Any] = {
            "name": task_name,
            "content": prompt,
            "timeout_seconds": timeout_seconds,
        }

        if max_sessions is not None:
            body["max_sessions"] = max_sessions

        if image_paths:
            body["image_paths"] = image_paths

        if video_paths:
            body["video_paths"] = video_paths

        if trace_path:
            body["trace_path"] = trace_path

        # Make POST request
        request_data = _json_dumps(body).encode("utf-8")
        request = Request(url, data=request_data, method="POST")
        request.add_header("Content-Type", "application/json")
        request.add_header("Accept", "application/json")

        logger.debug(f"Submitting prompt to {url}")

        with urlopen(request, timeout=30) as response:
            response_data = response.read().decode("utf-8")
            result = _json_loads(response_data)

            # The API returns {"success": true, "data": {"task_run_id": "..."}}
            if result.get("success"):
                data = result.get("data", {})
                task_run_id = data.get("task_run_id")
                return str(task_run_id) if task_run_id else None
            else:
                error = result.get("error", "Unknown error")
                logger.error(f"Failed to submit prompt: {error}")
                return None

    def _poll_task_completion(
        self,
        task_run_id: str,
        timeout_seconds: int,
    ) -> tuple[str, str]:
        """Poll for task completion.

        Args:
            task_run_id: The task run ID to poll
            timeout_seconds: Maximum time to wait

        Returns:
            Tuple of (output, error)
        """
        url = f"{self._runner_url}/task-runs/{task_run_id}"
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                raise TimeoutError()

            # Fetch task status
            request = Request(url, method="GET")
            request.add_header("Accept", "application/json")

            with urlopen(request, timeout=10) as response:
                response_data = response.read().decode("utf-8")
                task = _json_loads(response_data)

            if task is None:
                return "", f"Task not found: {task_run_id}"

            status = task.get("status", "")
            output_log = task.get("output_log", "")

            if status == "complete":
                return output_log, ""
            elif status == "failed":
                error_message = task.get("error_message", "Task failed")
                return output_log, error_message
            elif status == "stopped":
                return output_log, "Task was stopped"
            elif status == "running":
                time.sleep(DEFAULT_POLL_INTERVAL)
            else:
                logger.warning(f"Unknown task status: {status}")
                time.sleep(DEFAULT_POLL_INTERVAL)

    async def _poll_task_output_async(
        self,
        task_run_id: str,
        timeout_seconds: int,
    ) -> AsyncIterator[str]:
        """Poll task and yield output incrementally.

        Args:
            task_run_id: The task run ID to poll
            timeout_seconds: Maximum time to wait

        Yields:
            New output chunks as they become available
        """
        url = f"{self._runner_url}/task-runs/{task_run_id}"
        start_time = time.time()
        last_output_len = 0

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                raise TimeoutError()

            # Fetch task status (sync call in async context)
            task = await asyncio.get_event_loop().run_in_executor(None, self._fetch_task, url)

            if task is None:
                yield f"[ERROR: Task not found: {task_run_id}]\n"
                return

            status = task.get("status", "")
            output_log = task.get("output_log", "")

            # Yield any new output
            if len(output_log) > last_output_len:
                new_output = output_log[last_output_len:]
                last_output_len = len(output_log)
                yield new_output

            if status == "complete":
                return
            elif status == "failed":
                error_message = task.get("error_message", "Task failed")
                yield f"\n[ERROR: {error_message}]\n"
                return
            elif status == "stopped":
                yield "\n[Task was stopped]\n"
                return
            elif status == "running":
                await asyncio.sleep(DEFAULT_POLL_INTERVAL)
            else:
                logger.warning(f"Unknown task status: {status}")
                await asyncio.sleep(DEFAULT_POLL_INTERVAL)

    def _fetch_task(self, url: str) -> dict[str, Any] | None:
        """Fetch task from the runner API (sync helper for async polling).

        Args:
            url: The task URL

        Returns:
            Task data or None
        """
        request = Request(url, method="GET")
        request.add_header("Accept", "application/json")

        with urlopen(request, timeout=10) as response:
            response_data = response.read().decode("utf-8")
            result = _json_loads(response_data)
            if isinstance(result, dict):
                return result
            return None
