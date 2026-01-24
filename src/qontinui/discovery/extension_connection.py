"""Extension-based connection for UI Bridge exploration via Chrome extension.

This module provides a connection handler that uses the Chrome extension
as a bridge to interact with web pages. Unlike the web connection that
requires UI Bridge SDK to be installed in the target app, the extension
connection works with any page that has the extension installed.

The communication flow:
    Python -> Runner HTTP API -> Runner WebSocket -> Extension -> Content Script -> Page

Example:
    >>> from qontinui.discovery.extension_connection import ExtensionTargetConnection
    >>> async with ExtensionTargetConnection("http://localhost:9876") as conn:
    ...     elements = await conn.find_elements()
    ...     result = await conn.execute_action("btn-submit", "click")
"""

import logging
import time
from datetime import datetime
from typing import Any

import httpx

from .target_connection import (
    ActionResult,
    BoundingBox,
    DOMSnapshot,
    Element,
    ElementRole,
    TargetConnection,
)

logger = logging.getLogger(__name__)


class ExtensionTargetConnection(TargetConnection):
    """Connection handler for web pages via Chrome extension.

    Uses the qontinui-runner's extension bridge API to communicate with
    the Chrome extension, which in turn communicates with the active browser tab.

    This provides an alternative to the WebTargetConnection when:
    - The target app doesn't have UI Bridge SDK installed
    - You want to explore any arbitrary web page
    - You need to interact with the actual browser environment

    API endpoints used:
    - GET /extension/status - Check if extension is connected
    - POST /extension/command - Send command and wait for response

    Example:
        >>> async with ExtensionTargetConnection("http://localhost:9876") as conn:
        ...     # Verify extension is connected
        ...     if await conn.connect():
        ...         elements = await conn.find_elements()
        ...         for elem in elements:
        ...             print(f"{elem.id}: {elem.text_content}")
    """

    def __init__(
        self,
        runner_url: str = "http://localhost:9876",
        timeout_seconds: float = 30.0,
    ):
        """Initialize extension connection.

        Args:
            runner_url: URL of the qontinui-runner (default: http://localhost:9876)
            timeout_seconds: Request timeout in seconds
        """
        super().__init__(runner_url, timeout_seconds)
        self._active_tab_id: int | None = None
        self._active_url: str | None = None
        self._active_title: str | None = None

    async def connect(self) -> bool:
        """Connect to the browser via Chrome extension.

        Verifies that:
        1. The runner is running
        2. The Chrome extension is connected to the runner

        Returns:
            True if connection successful

        Raises:
            ConnectionError: With descriptive message if connection fails
        """
        if self._connected:
            return True

        self._client = httpx.AsyncClient(timeout=self._timeout)

        try:
            # Check extension status via runner API
            response = await self._client.get(f"{self._connection_url}/extension/status")

            if not response.is_success:
                error_msg = (
                    f"Could not get extension status from runner at {self._connection_url}. "
                    f"Make sure the runner is running."
                )
                logger.error(error_msg)
                raise ConnectionError(error_msg)

            data = response.json()
            result_data = data.get("data", data)

            if not result_data.get("connected", False):
                error_msg = (
                    "Chrome extension not connected to runner. "
                    "Make sure the Qontinui DevTools extension is installed and enabled in Chrome. "
                    "The extension should automatically connect when Chrome is open."
                )
                logger.warning(error_msg)
                raise ConnectionError(error_msg)

            # Try to connect to the active tab
            connect_result = await self._send_command("connect", {})
            self._active_tab_id = connect_result.get("tabId")
            self._active_url = connect_result.get("url")
            self._active_title = connect_result.get("title")

            self._connected = True
            logger.info(
                f"Connected to browser via extension: {self._active_url} (tab {self._active_tab_id})"
            )
            return True

        except httpx.ConnectError as e:
            error_msg = (
                f"Could not connect to qontinui-runner at {self._connection_url}. "
                f"Make sure the runner application is running. "
                f"You can start it with: cd qontinui-runner && npm run tauri dev"
            )
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e

        except httpx.TimeoutException as e:
            error_msg = (
                f"Connection to runner at {self._connection_url} timed out after {self._timeout}s. "
                f"The runner may be unresponsive or starting up."
            )
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e

        except httpx.RequestError as e:
            error_msg = f"Failed to connect to runner at {self._connection_url}: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e

    async def disconnect(self) -> None:
        """Disconnect from the browser."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._connected = False
        self._active_tab_id = None
        self._active_url = None
        self._active_title = None
        logger.info("Disconnected from browser extension")

    async def _send_command(
        self,
        action: str,
        params: dict[str, Any],
        timeout_secs: int = 30,
    ) -> dict[str, Any]:
        """Send a command to the extension via the runner.

        Args:
            action: The command action (e.g., "getElements", "executeAction")
            params: Command parameters
            timeout_secs: Timeout in seconds

        Returns:
            The command result data

        Raises:
            RuntimeError: If command fails
        """
        if not self._client:
            raise RuntimeError("Not connected to runner")

        try:
            response = await self._client.post(
                f"{self._connection_url}/extension/command",
                json={
                    "action": action,
                    "params": params,
                    "timeout_secs": timeout_secs,
                },
            )

            if not response.is_success:
                error_text = response.text
                raise RuntimeError(f"Extension command failed: {response.status_code} {error_text}")

            data = response.json()

            if not data.get("success", True):
                error = data.get("error", "Unknown error")
                raise RuntimeError(f"Extension command error: {error}")

            result: dict[str, Any] = data.get("data", {})
            return result

        except httpx.TimeoutException as e:
            raise RuntimeError(f"Extension command timed out: {action}") from e

        except httpx.RequestError as e:
            self._connected = False
            raise RuntimeError(f"Extension command failed: {e}") from e

    async def find_elements(self, selector: str | None = None) -> list[Element]:
        """Find interactive elements in the active browser tab.

        Uses the extension's content script to find elements with data-ui-id
        attributes on the current page.

        Args:
            selector: Optional CSS selector to filter elements

        Returns:
            List of Element objects
        """
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to browser")

        try:
            params: dict[str, Any] = {}
            if selector:
                params["selector"] = selector

            result = await self._send_command("getElements", params)

            elements = []
            elem_list = result.get("elements", result) if isinstance(result, dict) else result

            if isinstance(elem_list, list):
                for elem_data in elem_list:
                    elements.append(self._parse_element(elem_data))

            logger.debug(f"Found {len(elements)} elements via extension")
            return elements

        except RuntimeError as e:
            if "not connected" in str(e).lower() or "disconnected" in str(e).lower():
                self._connected = False
            logger.error(f"find_elements failed: {e}")
            return []

    async def execute_action(
        self,
        element_id: str,
        action: str,
        value: str | None = None,
    ) -> ActionResult:
        """Execute an action on an element in the browser.

        Args:
            element_id: Element's data-ui-id value
            action: Action to perform (click, type, clear, focus, blur, etc.)
            value: Optional value for the action (e.g., text to type)

        Returns:
            ActionResult with execution details
        """
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to browser")

        start_time = time.time()

        try:
            params: dict[str, Any] = {
                "elementId": element_id,
                "action": action,
            }
            if value is not None:
                params["params"] = {"value": value}

            result = await self._send_command("executeAction", params)

            response_time_ms = int((time.time() - start_time) * 1000)

            return ActionResult(
                success=result.get("success", True),
                element_id=element_id,
                action=action,
                error=result.get("error"),
                response_time_ms=response_time_ms,
                state_changed=result.get("stateChanged", False),
                new_elements=result.get("newElements", []),
                removed_elements=result.get("removedElements", []),
            )

        except RuntimeError as e:
            response_time_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e)

            if "not connected" in error_msg.lower() or "disconnected" in error_msg.lower():
                self._connected = False

            return ActionResult(
                success=False,
                element_id=element_id,
                action=action,
                error=error_msg,
                response_time_ms=response_time_ms,
            )

    async def capture_snapshot(self, include_screenshot: bool = False) -> DOMSnapshot:
        """Capture current state snapshot from the browser.

        Args:
            include_screenshot: Whether to include base64 screenshot (not yet supported)

        Returns:
            DOMSnapshot of current state
        """
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to browser")

        try:
            params: dict[str, Any] = {}
            if include_screenshot:
                params["includeScreenshot"] = True

            result = await self._send_command("captureSnapshot", params)

            # Parse elements from snapshot
            elements = []
            elem_list = result.get("elements", [])
            for elem_data in elem_list:
                elements.append(self._parse_element(elem_data))

            # Parse timestamp
            timestamp_str = result.get("timestamp")
            try:
                timestamp = (
                    datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now()
                )
            except (TypeError, ValueError):
                timestamp = datetime.now()

            return DOMSnapshot(
                id=result.get("id", f"snapshot-{int(time.time() * 1000)}"),
                timestamp=timestamp,
                url=result.get("url", self._active_url or self._connection_url),
                title=result.get("title", self._active_title or ""),
                root=result.get("root", {}),
                elements=elements,
                screenshot_base64=result.get("screenshotBase64"),
            )

        except RuntimeError as e:
            if "not connected" in str(e).lower() or "disconnected" in str(e).lower():
                self._connected = False
            raise RuntimeError(f"Failed to capture snapshot: {e}") from e

    def _parse_element(self, elem_data: dict[str, Any]) -> Element:
        """Parse element data from extension response.

        Args:
            elem_data: Raw element data from extension

        Returns:
            Element object
        """
        bbox = None
        bbox_data = elem_data.get("bbox") or elem_data.get("bounds")
        if bbox_data:
            bbox = BoundingBox(
                x=bbox_data.get("x", 0),
                y=bbox_data.get("y", 0),
                width=bbox_data.get("width", 0),
                height=bbox_data.get("height", 0),
            )

        role = None
        role_str = elem_data.get("role")
        if role_str:
            try:
                role = ElementRole(role_str)
            except ValueError:
                role = ElementRole.GENERIC

        return Element(
            id=elem_data.get("id", ""),
            tag_name=elem_data.get("tagName", elem_data.get("tag_name", "unknown")),
            text_content=elem_data.get(
                "textContent", elem_data.get("text_content", elem_data.get("text"))
            ),
            role=role,
            bbox=bbox,
            attributes=elem_data.get("attributes", {}),
            is_visible=elem_data.get(
                "isVisible", elem_data.get("is_visible", elem_data.get("visible", True))
            ),
            is_enabled=elem_data.get(
                "isEnabled", elem_data.get("is_enabled", elem_data.get("enabled", True))
            ),
            component_name=elem_data.get("componentName", elem_data.get("component_name")),
        )

    @property
    def active_url(self) -> str | None:
        """Get the URL of the active browser tab."""
        return self._active_url

    @property
    def active_title(self) -> str | None:
        """Get the title of the active browser tab."""
        return self._active_title

    @property
    def active_tab_id(self) -> int | None:
        """Get the ID of the active browser tab."""
        return self._active_tab_id
