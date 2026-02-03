"""Extension-based connection for UI Bridge exploration via Chrome extension.

This module provides a connection handler that uses the Chrome extension
as a bridge to interact with web pages. Unlike the web connection that
requires UI Bridge SDK to be installed in the target app, the extension
connection works with any page that has the extension installed.

The communication flow:
    Python -> Runner HTTP API -> Runner WebSocket -> Extension -> Content Script -> Page

Features:
- Element discovery with fingerprints for cross-page matching
- Action execution (click, type, etc.)
- Capture session management for state discovery
- Tab selection for targeting specific browser tabs

Example:
    >>> from qontinui.discovery.extension_connection import ExtensionTargetConnection
    >>> async with ExtensionTargetConnection("http://localhost:9876") as conn:
    ...     # List and select a tab
    ...     tabs = await conn.list_tabs()
    ...     await conn.select_tab(tabs[0].id)
    ...
    ...     # Start capture session for state discovery
    ...     await conn.start_capture_session()
    ...
    ...     # Get elements with fingerprints
    ...     elements = await conn.find_elements()
    ...     for elem in elements:
    ...         print(f"{elem.id}: {elem.fingerprint.hash if elem.fingerprint else 'no fp'}")
    ...
    ...     # Execute action and capture state change
    ...     before = await conn.create_capture()
    ...     result = await conn.execute_action("btn-submit", "click")
    ...     after = await conn.create_capture()
    ...
    ...     # Export session for state discovery
    ...     export = await conn.export_capture_session()
"""

import logging
import time
from dataclasses import dataclass
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


# =============================================================================
# Data Types for Fingerprint-Enhanced Discovery
# =============================================================================


@dataclass
class ElementFingerprint:
    """Fingerprint data for an element from the browser extension.

    Fingerprints enable cross-page element matching by providing a stable
    identity based on structural and semantic properties rather than just IDs.

    Attributes:
        hash: Unique hash identifying this element across pages
        structural_path: Tag-only path like "nav > ul > li > a"
        position_zone: Position zone (header, footer, main, modal, etc.)
        landmark_context: Nearest ARIA landmark
        role: ARIA role
        tag_name: HTML tag name
        accessible_name: Computed accessible name
        size_category: Size classification (icon, button, small, medium, large, etc.)
        is_repeating: Whether this is a repeating element (list item, etc.)
        repeat_pattern: Pattern info if repeating
    """

    hash: str
    structural_path: str = ""
    position_zone: str = "main"
    landmark_context: str = ""
    role: str = ""
    tag_name: str = ""
    accessible_name: str | None = None
    size_category: str = "medium"
    is_repeating: bool = False
    repeat_pattern: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ElementFingerprint":
        """Create from dictionary."""
        return cls(
            hash=data.get("hash", ""),
            structural_path=data.get("structuralPath", ""),
            position_zone=data.get("positionZone", "main"),
            landmark_context=data.get("landmarkContext", ""),
            role=data.get("role", ""),
            tag_name=data.get("tagName", ""),
            accessible_name=data.get("accessibleName"),
            size_category=data.get("sizeCategory", "medium"),
            is_repeating=data.get("isRepeating", False),
            repeat_pattern=data.get("repeatPattern"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hash": self.hash,
            "structuralPath": self.structural_path,
            "positionZone": self.position_zone,
            "landmarkContext": self.landmark_context,
            "role": self.role,
            "tagName": self.tag_name,
            "accessibleName": self.accessible_name,
            "sizeCategory": self.size_category,
            "isRepeating": self.is_repeating,
            "repeatPattern": self.repeat_pattern,
        }


@dataclass
class ExtensionElement(Element):
    """Element with fingerprint data from browser extension.

    Extends the base Element with fingerprint information for
    cross-page element matching and state discovery.
    """

    fingerprint: ElementFingerprint | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        if self.fingerprint:
            data["fingerprint"] = self.fingerprint.to_dict()
        return data


@dataclass
class BrowserTab:
    """Information about a browser tab."""

    id: int
    url: str
    title: str
    active: bool
    window_id: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BrowserTab":
        """Create from dictionary."""
        return cls(
            id=data.get("id", 0),
            url=data.get("url", ""),
            title=data.get("title", ""),
            active=data.get("active", False),
            window_id=data.get("windowId", 0),
        )


@dataclass
class CaptureRecord:
    """A capture record from a capture session.

    Represents a snapshot of the page state with fingerprint data.
    """

    capture_id: str
    timestamp: int
    url: str
    title: str
    fingerprint_hashes: list[str]
    element_count: int
    triggered_by: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CaptureRecord":
        """Create from dictionary."""
        return cls(
            capture_id=data.get("captureId", ""),
            timestamp=data.get("timestamp", 0),
            url=data.get("url", ""),
            title=data.get("title", ""),
            fingerprint_hashes=data.get("elementFingerprints", []),
            element_count=data.get("elementCount", 0),
            triggered_by=data.get("triggeredBy"),
        )


@dataclass
class ActionRecord:
    """Record of an action taken during a capture session."""

    action_id: str
    timestamp: int
    action_type: str
    target_fingerprint: str
    before_capture_id: str
    after_capture_id: str
    added_fingerprints: list[str]
    removed_fingerprints: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionRecord":
        """Create from dictionary."""
        return cls(
            action_id=data.get("actionId", ""),
            timestamp=data.get("timestamp", 0),
            action_type=data.get("actionType", ""),
            target_fingerprint=data.get("targetFingerprint", ""),
            before_capture_id=data.get("beforeCaptureId", ""),
            after_capture_id=data.get("afterCaptureId", ""),
            added_fingerprints=data.get("addedFingerprints", []),
            removed_fingerprints=data.get("removedFingerprints", []),
        )


@dataclass
class CaptureSession:
    """A complete capture session with all data for state discovery."""

    session_id: str
    started_at: int
    ended_at: int | None
    captures: list[CaptureRecord]
    actions: list[ActionRecord]
    fingerprint_catalog: dict[str, ElementFingerprint]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CaptureSession":
        """Create from dictionary."""
        catalog = {}
        raw_catalog = data.get("fingerprintCatalog", {})
        for hash_key, fp_data in raw_catalog.items():
            catalog[hash_key] = ElementFingerprint.from_dict(fp_data)

        return cls(
            session_id=data.get("sessionId", ""),
            started_at=data.get("startedAt", 0),
            ended_at=data.get("endedAt"),
            captures=[CaptureRecord.from_dict(c) for c in data.get("captures", [])],
            actions=[ActionRecord.from_dict(a) for a in data.get("actions", [])],
            fingerprint_catalog=catalog,
        )


# =============================================================================
# Extension Target Connection
# =============================================================================


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
        self._capture_session_active: bool = False
        self._selected_tab_id: int | None = None

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

            elements: list[Element] = []
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
            elements: list[Element] = []
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

    def _parse_element(self, elem_data: dict[str, Any]) -> ExtensionElement:
        """Parse element data from extension response.

        Args:
            elem_data: Raw element data from extension

        Returns:
            ExtensionElement object with fingerprint if available
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

        # Parse fingerprint if present
        fingerprint = None
        fp_data = elem_data.get("fingerprint")
        if fp_data and isinstance(fp_data, dict):
            fingerprint = ElementFingerprint.from_dict(fp_data)

        return ExtensionElement(
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
            fingerprint=fingerprint,
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

    @property
    def selected_tab_id(self) -> int | None:
        """Get the ID of the selected tab for exploration."""
        return self._selected_tab_id

    @property
    def capture_session_active(self) -> bool:
        """Check if a capture session is currently active."""
        return self._capture_session_active

    # =========================================================================
    # Tab Management
    # =========================================================================

    async def list_tabs(self) -> list[BrowserTab]:
        """List available browser tabs.

        Returns:
            List of browser tabs
        """
        data = await self._send_command("listTabs", {})
        tabs = data.get("tabs", [])
        return [BrowserTab.from_dict(t) for t in tabs]

    async def select_tab(self, tab_id: int) -> bool:
        """Select a tab for exploration.

        Args:
            tab_id: Browser tab ID to select

        Returns:
            True if tab was selected successfully
        """
        data = await self._send_command("selectTab", {"tabId": tab_id})
        if data.get("success"):
            self._selected_tab_id = tab_id
            # Update active tab info
            tab_info = data.get("tab", {})
            self._active_tab_id = tab_id
            self._active_url = tab_info.get("url", self._active_url)
            self._active_title = tab_info.get("title", self._active_title)
            logger.info(f"Selected browser tab: {tab_id} - {self._active_url}")
            return True
        return False

    async def get_selected_tab(self) -> BrowserTab | None:
        """Get the currently selected tab.

        Returns:
            Selected tab info or None
        """
        data = await self._send_command("getSelectedTab", {})
        if data.get("tab"):
            return BrowserTab.from_dict(data["tab"])
        return None

    async def clear_selected_tab(self) -> None:
        """Clear the tab selection (use active tab)."""
        await self._send_command("clearSelectedTab", {})
        self._selected_tab_id = None

    # =========================================================================
    # Capture Session Management (for State Discovery)
    # =========================================================================

    async def start_capture_session(self) -> dict[str, Any]:
        """Start a new capture session for state discovery.

        A capture session records element fingerprints and actions across
        multiple page states, enabling co-occurrence analysis.

        Returns:
            Session info with sessionId and startedAt
        """
        data = await self._send_command("startCaptureSession", {})
        self._capture_session_active = True
        logger.info(f"Started capture session: {data.get('sessionId')}")
        return data

    async def end_capture_session(self) -> CaptureSession | None:
        """End the current capture session.

        Returns:
            Complete session data or None if no session was active
        """
        data = await self._send_command("endCaptureSession", {})
        self._capture_session_active = False

        if data and data.get("sessionId"):
            logger.info(
                f"Ended capture session: {data.get('sessionId')} "
                f"with {len(data.get('captures', []))} captures"
            )
            return CaptureSession.from_dict(data)
        return None

    async def get_capture_session_status(self) -> dict[str, Any]:
        """Get status of the current capture session.

        Returns:
            Session status including active state and counts
        """
        return await self._send_command("getCaptureSessionStatus", {})

    async def create_capture(
        self,
        triggered_by: dict[str, Any] | None = None,
    ) -> CaptureRecord:
        """Create a capture record in the current session.

        This captures the current page state with all element fingerprints.

        Args:
            triggered_by: Optional info about what triggered this capture
                (e.g., action type, target fingerprint, previous capture ID)

        Returns:
            The created capture record
        """
        params: dict[str, Any] = {}
        if triggered_by:
            params["triggeredBy"] = triggered_by

        data = await self._send_command("createCapture", params, timeout_secs=60)
        return CaptureRecord.from_dict(data)

    async def record_action(
        self,
        action_type: str,
        target_fingerprint: str,
        before_capture_id: str,
        after_capture_id: str,
    ) -> ActionRecord:
        """Record an action in the capture session.

        Args:
            action_type: Type of action (click, type, etc.)
            target_fingerprint: Fingerprint hash of target element
            before_capture_id: Capture ID before the action
            after_capture_id: Capture ID after the action

        Returns:
            The created action record
        """
        data = await self._send_command(
            "recordAction",
            {
                "actionType": action_type,
                "targetFingerprint": target_fingerprint,
                "beforeCaptureId": before_capture_id,
                "afterCaptureId": after_capture_id,
            },
        )
        return ActionRecord.from_dict(data)

    async def export_capture_session(self) -> dict[str, Any]:
        """Export the current capture session in CooccurrenceExport format.

        The exported data can be used with FingerprintStateDiscovery to
        discover application states.

        Returns:
            Export data suitable for FingerprintStateDiscovery.load_cooccurrence_export()
        """
        return await self._send_command("exportCaptureSession", {}, timeout_secs=60)

    # =========================================================================
    # Screenshot
    # =========================================================================

    async def capture_screenshot(self) -> str | None:
        """Capture a screenshot of the current page.

        Returns:
            Base64-encoded screenshot or None on failure
        """
        try:
            data = await self._send_command("capturePageScreenshot", {}, timeout_secs=30)
            return data.get("screenshot")
        except Exception as e:
            logger.warning(f"Screenshot capture failed: {e}")
            return None
