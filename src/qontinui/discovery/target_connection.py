"""Connection handlers for different target types in UI Bridge exploration.

This module provides connection abstractions for connecting to applications
that have UI Bridge integrated. It supports:
- Web applications via HTTP URL
- Desktop (Tauri) applications via local port
- Mobile (React Native) applications via device IP

Example:
    >>> from qontinui.discovery.target_connection import WebTargetConnection
    >>> async with WebTargetConnection("http://localhost:3000") as conn:
    ...     elements = await conn.find_elements()
    ...     result = await conn.execute_action("btn-submit", "click")
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Self

import httpx

logger = logging.getLogger(__name__)


class ActionType(str, Enum):
    """Types of actions that can be executed on elements."""

    CLICK = "click"
    TYPE = "type"
    CLEAR = "clear"
    FOCUS = "focus"
    BLUR = "blur"
    SELECT = "select"
    SCROLL = "scroll"
    HOVER = "hover"


class ElementRole(str, Enum):
    """ARIA roles for UI elements."""

    BUTTON = "button"
    LINK = "link"
    TEXTBOX = "textbox"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    COMBOBOX = "combobox"
    LISTBOX = "listbox"
    MENU = "menu"
    MENUITEM = "menuitem"
    TAB = "tab"
    TABPANEL = "tabpanel"
    DIALOG = "dialog"
    ALERT = "alert"
    NAVIGATION = "navigation"
    MAIN = "main"
    REGION = "region"
    GENERIC = "generic"


@dataclass
class BoundingBox:
    """Bounding box coordinates for an element."""

    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> tuple[float, float]:
        """Get the center point of the bounding box."""
        return (self.x + self.width / 2, self.y + self.height / 2)

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class Element:
    """Represents an interactive element discovered via UI Bridge.

    Attributes:
        id: Unique element identifier (data-ui-id value)
        tag_name: HTML tag name (e.g., 'button', 'input')
        text_content: Visible text content of the element
        role: ARIA role if available
        bbox: Bounding box coordinates
        attributes: All HTML attributes
        is_visible: Whether element is currently visible
        is_enabled: Whether element is enabled for interaction
        component_name: React component name if available
    """

    id: str
    tag_name: str
    text_content: str | None = None
    role: ElementRole | None = None
    bbox: BoundingBox | None = None
    attributes: dict[str, str] = field(default_factory=dict)
    is_visible: bool = True
    is_enabled: bool = True
    component_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "tagName": self.tag_name,
            "textContent": self.text_content,
            "role": self.role.value if self.role else None,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "attributes": self.attributes,
            "isVisible": self.is_visible,
            "isEnabled": self.is_enabled,
            "componentName": self.component_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Element":
        """Create Element from dictionary."""
        bbox = None
        if data.get("bbox"):
            bbox = BoundingBox(**data["bbox"])

        role = None
        if data.get("role"):
            try:
                role = ElementRole(data["role"])
            except ValueError:
                role = ElementRole.GENERIC

        return cls(
            id=data["id"],
            tag_name=data.get("tagName", data.get("tag_name", "unknown")),
            text_content=data.get("textContent", data.get("text_content")),
            role=role,
            bbox=bbox,
            attributes=data.get("attributes", {}),
            is_visible=data.get("isVisible", data.get("is_visible", True)),
            is_enabled=data.get("isEnabled", data.get("is_enabled", True)),
            component_name=data.get("componentName", data.get("component_name")),
        )


@dataclass
class ActionResult:
    """Result of executing an action on an element.

    Attributes:
        success: Whether the action completed successfully
        element_id: ID of the element the action was performed on
        action: Type of action that was executed
        error: Error message if action failed
        response_time_ms: Time taken to execute the action
        state_changed: Whether the action caused a state change
        new_elements: New elements that appeared after the action
        removed_elements: Elements that disappeared after the action
    """

    success: bool
    element_id: str
    action: str
    error: str | None = None
    response_time_ms: int = 0
    state_changed: bool = False
    new_elements: list[str] = field(default_factory=list)
    removed_elements: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "elementId": self.element_id,
            "action": self.action,
            "error": self.error,
            "responseTimeMs": self.response_time_ms,
            "stateChanged": self.state_changed,
            "newElements": self.new_elements,
            "removedElements": self.removed_elements,
        }


@dataclass
class DOMSnapshot:
    """A snapshot of the DOM state captured via UI Bridge.

    Attributes:
        id: Unique snapshot identifier
        timestamp: When the snapshot was captured
        url: Current page URL
        title: Page title
        root: Root DOM element tree
        elements: Flat list of interactive elements
        screenshot_base64: Optional screenshot as base64
    """

    id: str
    timestamp: datetime
    url: str
    title: str
    root: dict[str, Any]
    elements: list[Element]
    screenshot_base64: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "url": self.url,
            "title": self.title,
            "root": self.root,
            "elements": [e.to_dict() for e in self.elements],
            "screenshotBase64": self.screenshot_base64,
        }


@dataclass
class ExplorationConfig:
    """Configuration for UI Bridge exploration.

    Attributes:
        target_type: Type of target application ("web", "desktop", "mobile", "extension")
        connection_url: URL, local port, or device IP to connect to
        max_depth: Maximum navigation depth from starting page
        max_elements_per_page: Maximum elements to interact with per page
        max_total_elements: Maximum total elements to explore
        action_delay_ms: Delay between actions in milliseconds
        blocked_keywords: Keywords in element text/id to skip (e.g., "delete", "logout")
        safe_keywords: Keywords that are always safe to interact with
        blocked_selectors: CSS selectors to never interact with
        timeout_seconds: HTTP request timeout
        capture_screenshots: Whether to capture screenshots with snapshots
        record_render_logs: Whether to record render logs for analysis
    """

    target_type: Literal["web", "desktop", "mobile", "extension"]
    connection_url: str
    max_depth: int = 2
    max_elements_per_page: int = 20
    max_total_elements: int = 100
    action_delay_ms: int = 500
    blocked_keywords: list[str] = field(default_factory=list)
    safe_keywords: list[str] = field(default_factory=list)
    blocked_selectors: list[str] = field(default_factory=list)
    timeout_seconds: float = 30.0
    capture_screenshots: bool = False
    record_render_logs: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "targetType": self.target_type,
            "connectionUrl": self.connection_url,
            "maxDepth": self.max_depth,
            "maxElementsPerPage": self.max_elements_per_page,
            "maxTotalElements": self.max_total_elements,
            "actionDelayMs": self.action_delay_ms,
            "blockedKeywords": self.blocked_keywords,
            "safeKeywords": self.safe_keywords,
            "blockedSelectors": self.blocked_selectors,
            "timeoutSeconds": self.timeout_seconds,
            "captureScreenshots": self.capture_screenshots,
            "recordRenderLogs": self.record_render_logs,
        }


class TargetConnection(ABC):
    """Abstract base class for target application connections.

    Provides a common interface for connecting to different types of
    applications (web, desktop, mobile) that have UI Bridge integrated.

    Subclasses must implement:
    - connect(): Establish connection to the target
    - disconnect(): Close the connection
    - find_elements(): Discover interactive elements
    - execute_action(): Perform an action on an element
    - capture_snapshot(): Capture current DOM state
    """

    def __init__(self, connection_url: str, timeout_seconds: float = 30.0):
        """Initialize the connection.

        Args:
            connection_url: URL or address to connect to
            timeout_seconds: Request timeout in seconds
        """
        self._connection_url = connection_url
        self._timeout = timeout_seconds
        self._connected = False
        self._client: httpx.AsyncClient | None = None

    @property
    def is_connected(self) -> bool:
        """Check if currently connected."""
        return self._connected

    @property
    def connection_url(self) -> str:
        """Get the connection URL."""
        return self._connection_url

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the target application.

        Returns:
            True if connection successful, False otherwise
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the connection to the target application."""
        ...

    @abstractmethod
    async def find_elements(self, selector: str | None = None) -> list[Element]:
        """Find interactive elements in the target application.

        Args:
            selector: Optional CSS selector to filter elements

        Returns:
            List of discovered Element objects
        """
        ...

    @abstractmethod
    async def execute_action(
        self,
        element_id: str,
        action: str,
        value: str | None = None,
    ) -> ActionResult:
        """Execute an action on an element.

        Args:
            element_id: ID of the element (data-ui-id value)
            action: Action to perform (click, type, etc.)
            value: Optional value for the action (e.g., text to type)

        Returns:
            ActionResult with success status and any changes
        """
        ...

    @abstractmethod
    async def capture_snapshot(self, include_screenshot: bool = False) -> DOMSnapshot:
        """Capture the current DOM state.

        Args:
            include_screenshot: Whether to include a screenshot

        Returns:
            DOMSnapshot of current state
        """
        ...


class WebTargetConnection(TargetConnection):
    """Connection handler for web applications via HTTP URL.

    Connects to web applications that have UI Bridge installed and
    expose the control API endpoint.

    Example:
        >>> async with WebTargetConnection("http://localhost:3000") as conn:
        ...     elements = await conn.find_elements()
        ...     for elem in elements:
        ...         print(f"{elem.id}: {elem.text_content}")
    """

    UI_BRIDGE_API_PATH = "/__ui-bridge__"

    def __init__(
        self,
        connection_url: str,
        timeout_seconds: float = 30.0,
        api_path: str | None = None,
    ):
        """Initialize web connection.

        Args:
            connection_url: Base URL of the web application
            timeout_seconds: Request timeout
            api_path: Custom UI Bridge API path (default: /__ui-bridge__)
        """
        super().__init__(connection_url, timeout_seconds)
        self._api_path = api_path or self.UI_BRIDGE_API_PATH

    def _get_api_url(self, endpoint: str) -> str:
        """Build full API URL for an endpoint."""
        base = self._connection_url.rstrip("/")
        path = self._api_path.rstrip("/")
        return f"{base}{path}/{endpoint.lstrip('/')}"

    async def connect(self) -> bool:
        """Connect to the web application.

        Verifies UI Bridge is available by calling the health endpoint.

        Returns:
            True if connection successful

        Raises:
            ConnectionError: With descriptive message if connection fails
        """
        if self._connected:
            return True

        self._client = httpx.AsyncClient(timeout=self._timeout)

        try:
            # Check if UI Bridge is available
            response = await self._client.get(self._get_api_url("health"))

            if response.is_success:
                self._connected = True
                logger.info(f"Connected to web target at {self._connection_url}")
                return True

            if response.status_code == 404:
                error_msg = (
                    f"UI Bridge SDK not found at {self._connection_url}. "
                    f"The target web application must have the UI Bridge SDK installed. "
                    f"Install with: npm install @qontinui/ui-bridge "
                    f"and add the UIBridgeProvider to your app."
                )
                logger.error(error_msg)
                raise ConnectionError(error_msg)

            logger.warning(
                f"UI Bridge not available at {self._connection_url}: {response.status_code}"
            )
            return False

        except httpx.ConnectError as e:
            error_msg = (
                f"Could not connect to {self._connection_url}. "
                f"Make sure the web application is running and accessible. "
                f"Error: {e}"
            )
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e

        except httpx.TimeoutException as e:
            error_msg = (
                f"Connection to {self._connection_url} timed out after {self._timeout}s. "
                f"The web application may be unresponsive or the network connection is slow."
            )
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e

        except httpx.RequestError as e:
            error_msg = f"Failed to connect to {self._connection_url}: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e

    async def disconnect(self) -> None:
        """Disconnect from the web application."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._connected = False
        logger.info(f"Disconnected from web target at {self._connection_url}")

    async def find_elements(self, selector: str | None = None) -> list[Element]:
        """Find interactive elements using UI Bridge's find() method.

        Args:
            selector: Optional CSS selector to filter elements

        Returns:
            List of Element objects
        """
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to target")

        try:
            payload: dict[str, Any] = {}
            if selector:
                payload["selector"] = selector

            # Use POST to /control/discover as per UI Bridge API
            response = await self._client.post(
                self._get_api_url("control/discover"),
                json=payload,
            )

            if not response.is_success:
                logger.warning(f"find_elements failed: {response.status_code}")
                return []

            data = response.json()
            elements = []

            # Handle API response format: {success: bool, data: {...}}
            result_data = data.get("data", data)
            for elem_data in result_data.get("elements", []):
                elements.append(Element.from_dict(elem_data))

            logger.debug(f"Found {len(elements)} elements")
            return elements

        except httpx.ConnectError as e:
            logger.error(
                f"Lost connection to web app at {self._connection_url}. "
                f"The application may have stopped or restarted. Error: {e}"
            )
            self._connected = False
            return []

        except httpx.TimeoutException as e:
            logger.error(
                f"Request to {self._connection_url} timed out. "
                f"The application may be unresponsive. Error: {e}"
            )
            return []

        except httpx.RequestError as e:
            logger.error(f"find_elements request failed: {e}")
            return []

    async def execute_action(
        self,
        element_id: str,
        action: str,
        value: str | None = None,
    ) -> ActionResult:
        """Execute an action on an element via UI Bridge control API.

        Args:
            element_id: Element's data-ui-id value
            action: Action to perform
            value: Optional value for the action

        Returns:
            ActionResult with execution details
        """
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to target")

        import time

        start_time = time.time()

        try:
            payload: dict[str, Any] = {
                "action": action,
            }
            if value is not None:
                payload["params"] = {"value": value}

            # Use POST to /control/element/:id/action as per UI Bridge API
            response = await self._client.post(
                self._get_api_url(f"control/element/{element_id}/action"),
                json=payload,
            )

            response_time_ms = int((time.time() - start_time) * 1000)

            if not response.is_success:
                error_detail = response.text
                if response.status_code == 404:
                    error_detail = (
                        f"Element '{element_id}' not found. "
                        f"The element may have been removed from the DOM."
                    )
                return ActionResult(
                    success=False,
                    element_id=element_id,
                    action=action,
                    error=f"HTTP {response.status_code}: {error_detail}",
                    response_time_ms=response_time_ms,
                )

            data = response.json()

            return ActionResult(
                success=data.get("success", True),
                element_id=element_id,
                action=action,
                error=data.get("error"),
                response_time_ms=response_time_ms,
                state_changed=data.get("stateChanged", False),
                new_elements=data.get("newElements", []),
                removed_elements=data.get("removedElements", []),
            )

        except httpx.ConnectError as e:
            response_time_ms = int((time.time() - start_time) * 1000)
            error_msg = (
                f"Lost connection to web app at {self._connection_url}. "
                f"The application may have stopped or restarted."
            )
            logger.error(f"{error_msg} Error: {e}")
            self._connected = False
            return ActionResult(
                success=False,
                element_id=element_id,
                action=action,
                error=error_msg,
                response_time_ms=response_time_ms,
            )

        except httpx.TimeoutException as e:
            response_time_ms = int((time.time() - start_time) * 1000)
            error_msg = (
                f"Action '{action}' on element '{element_id}' timed out. "
                f"The application may be unresponsive."
            )
            logger.error(f"{error_msg} Error: {e}")
            return ActionResult(
                success=False,
                element_id=element_id,
                action=action,
                error=error_msg,
                response_time_ms=response_time_ms,
            )

        except httpx.RequestError as e:
            response_time_ms = int((time.time() - start_time) * 1000)
            logger.error(f"execute_action request failed: {e}")
            return ActionResult(
                success=False,
                element_id=element_id,
                action=action,
                error=str(e),
                response_time_ms=response_time_ms,
            )

    async def capture_snapshot(self, include_screenshot: bool = False) -> DOMSnapshot:
        """Capture current DOM snapshot via UI Bridge.

        Args:
            include_screenshot: Whether to include base64 screenshot

        Returns:
            DOMSnapshot of current state
        """
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to target")

        try:
            params = {"includeScreenshot": str(include_screenshot).lower()}

            # Use GET to /control/snapshot as per UI Bridge API
            response = await self._client.get(
                self._get_api_url("control/snapshot"),
                params=params,
            )

            if not response.is_success:
                raise RuntimeError(f"Failed to capture snapshot: {response.status_code}")

            data = response.json()

            # Check for API-level error (HTTP 200 but success: false)
            if data.get("success") is False:
                error_msg = data.get("error", "Unknown error from UI Bridge")
                raise RuntimeError(f"Snapshot failed: {error_msg}")

            # Handle API response format: {success: bool, data: {...}}
            result_data = data.get("data", data)

            # Parse elements from snapshot
            elements = []
            for elem_data in result_data.get("elements", []):
                elements.append(Element.from_dict(elem_data))

            # Parse timestamp: SDK returns epoch ms (number), but may also be ISO string
            ts_raw = data.get("timestamp")
            if isinstance(ts_raw, int | float):
                timestamp = datetime.fromtimestamp(ts_raw / 1000)
            elif isinstance(ts_raw, str):
                try:
                    timestamp = datetime.fromisoformat(ts_raw)
                except ValueError:
                    timestamp = datetime.now()
            else:
                timestamp = datetime.now()

            return DOMSnapshot(
                id=data.get("id", ""),
                timestamp=timestamp,
                url=data.get("url", self._connection_url),
                title=data.get("title", ""),
                root=data.get("root", {}),
                elements=elements,
                screenshot_base64=data.get("screenshotBase64"),
            )

        except httpx.RequestError as e:
            logger.error(f"capture_snapshot request failed: {e}")
            raise RuntimeError(f"Failed to capture snapshot: {e}") from e


class DesktopTargetConnection(TargetConnection):
    """Connection handler for Tauri desktop applications via qontinui-runner.

    Connects to Tauri applications via the qontinui-runner's HTTP API.
    The runner exposes UI Bridge control endpoints at /ui-bridge/control/*.

    The runner forwards requests to the connected Tauri/React app via WebSocket.

    API endpoints used:
    - GET /health - Check runner availability
    - GET /ui-bridge/control/elements - Get all registered elements
    - GET /ui-bridge/control/element/:id - Get specific element by ID
    - POST /ui-bridge/control/element/:id/action - Execute action on element
    - POST /ui-bridge/control/discover - Discover elements with filtering options
    - GET /ui-bridge/control/snapshot - Capture full UI state snapshot

    Example:
        >>> async with DesktopTargetConnection("http://localhost:9876") as conn:
        ...     elements = await conn.find_elements()
        ...     await conn.execute_action("btn-save", "click")
    """

    # Runner's UI Bridge control API prefix
    UI_BRIDGE_API_PREFIX = "/ui-bridge/control"

    def __init__(
        self,
        connection_url: str,
        timeout_seconds: float = 30.0,
    ):
        """Initialize desktop connection.

        Args:
            connection_url: Runner URL with port (e.g., http://localhost:9876)
            timeout_seconds: Request timeout
        """
        super().__init__(connection_url, timeout_seconds)

    def _get_api_url(self, endpoint: str) -> str:
        """Build full API URL for an endpoint."""
        base = self._connection_url.rstrip("/")
        return f"{base}{self.UI_BRIDGE_API_PREFIX}/{endpoint.lstrip('/')}"

    async def connect(self) -> bool:
        """Connect to the desktop application via qontinui-runner.

        Verifies the runner is running and UI Bridge control API is available.

        Returns:
            True if connection successful

        Raises:
            ConnectionError: With descriptive message if connection fails
        """
        if self._connected:
            return True

        self._client = httpx.AsyncClient(timeout=self._timeout)

        try:
            # Check runner health endpoint
            response = await self._client.get(f"{self._connection_url.rstrip('/')}/health")

            if response.is_success:
                # Check if the response indicates a connected desktop app
                try:
                    health_data = response.json()
                    # If the runner reports no connected app, provide helpful error
                    if health_data.get("desktop_app_connected") is False:
                        error_msg = (
                            f"No desktop application connected to the runner at {self._connection_url}. "
                            f"Start your Tauri app and ensure the UI Bridge SDK connects to the runner. "
                            f"The app should connect via WebSocket when it starts."
                        )
                        logger.warning(error_msg)
                        # Still mark as connected to runner, but warn about missing app
                except (ValueError, KeyError):
                    pass  # Health endpoint may not return JSON or this field

                self._connected = True
                logger.info(f"Connected to runner at {self._connection_url}")
                return True

            logger.warning(
                f"Runner not available at {self._connection_url}: {response.status_code}"
            )
            return False

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
        """Disconnect from the desktop application."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._connected = False
        logger.info(f"Disconnected from runner at {self._connection_url}")

    async def find_elements(self, selector: str | None = None) -> list[Element]:
        """Find interactive elements via runner's UI Bridge control API.

        Uses POST /ui-bridge/control/discover for discovery with filtering options,
        or falls back to GET /ui-bridge/control/elements for basic listing.

        Args:
            selector: Optional CSS selector to filter elements

        Returns:
            List of Element objects
        """
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to runner")

        try:
            # Use discover endpoint for more control (POST with camelCase body)
            payload: dict[str, Any] = {
                "interactiveOnly": True,
                "includeHidden": False,
            }
            if selector:
                payload["selector"] = selector

            response = await self._client.post(
                self._get_api_url("discover"),
                json=payload,
            )

            if not response.is_success:
                # Fall back to elements endpoint (GET, no body needed)
                response = await self._client.get(self._get_api_url("elements"))
                if not response.is_success:
                    logger.warning(f"find_elements failed: {response.status_code}")
                    return []

            data = response.json()
            elements = []

            # Handle runner's API response format: {success: bool, data: ...}
            result_data = data.get("data", data)
            elem_list = (
                result_data.get("elements", result_data)
                if isinstance(result_data, dict)
                else result_data
            )

            if isinstance(elem_list, list):
                for elem_data in elem_list:
                    elements.append(Element.from_dict(elem_data))

            logger.debug(f"Found {len(elements)} elements via runner")
            return elements

        except httpx.ConnectError as e:
            error_msg = (
                f"Lost connection to runner at {self._connection_url}. "
                f"The runner may have stopped or restarted."
            )
            logger.error(f"{error_msg} Error: {e}")
            self._connected = False
            return []

        except httpx.TimeoutException as e:
            logger.error(
                f"Request to runner at {self._connection_url} timed out. "
                f"The runner or desktop app may be unresponsive. Error: {e}"
            )
            return []

        except httpx.RequestError as e:
            logger.error(f"find_elements request failed: {e}")
            return []

    async def execute_action(
        self,
        element_id: str,
        action: str,
        value: str | None = None,
    ) -> ActionResult:
        """Execute an action on an element via runner's UI Bridge control API.

        Uses POST /ui-bridge/control/element/:id/action endpoint.
        The request body uses camelCase: {action, params, waitOptions}

        Args:
            element_id: Element's data-ui-id value
            action: Action to perform (click, type, clear, focus, blur, etc.)
            value: Optional value for the action (e.g., text to type)

        Returns:
            ActionResult with execution details
        """
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to runner")

        import time

        start_time = time.time()

        try:
            # Runner's action endpoint expects camelCase body
            payload: dict[str, Any] = {
                "action": action,
            }
            if value is not None:
                payload["params"] = {"value": value}

            response = await self._client.post(
                self._get_api_url(f"element/{element_id}/action"),
                json=payload,
            )

            response_time_ms = int((time.time() - start_time) * 1000)

            if not response.is_success:
                return ActionResult(
                    success=False,
                    element_id=element_id,
                    action=action,
                    error=f"HTTP {response.status_code}: {response.text}",
                    response_time_ms=response_time_ms,
                )

            data = response.json()
            result_data = data.get("data", data)

            return ActionResult(
                success=data.get("success", True),
                element_id=element_id,
                action=action,
                error=data.get("error"),
                response_time_ms=response_time_ms,
                state_changed=(
                    result_data.get("stateChanged", False)
                    if isinstance(result_data, dict)
                    else False
                ),
                new_elements=(
                    result_data.get("newElements", []) if isinstance(result_data, dict) else []
                ),
                removed_elements=(
                    result_data.get("removedElements", []) if isinstance(result_data, dict) else []
                ),
            )

        except httpx.RequestError as e:
            response_time_ms = int((time.time() - start_time) * 1000)
            logger.error(f"execute_action request failed: {e}")
            return ActionResult(
                success=False,
                element_id=element_id,
                action=action,
                error=str(e),
                response_time_ms=response_time_ms,
            )

    async def capture_snapshot(self, include_screenshot: bool = False) -> DOMSnapshot:
        """Capture current DOM snapshot via runner's UI Bridge control API.

        Uses GET /ui-bridge/control/snapshot endpoint.

        Args:
            include_screenshot: Whether to include base64 screenshot

        Returns:
            DOMSnapshot of current state
        """
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to runner")

        try:
            params = {"includeScreenshot": str(include_screenshot).lower()}

            response = await self._client.get(
                self._get_api_url("snapshot"),
                params=params,
            )

            if not response.is_success:
                error_detail = f"HTTP {response.status_code}"
                if response.status_code == 503:
                    error_detail = (
                        "No desktop application connected to the runner. "
                        "Make sure your Tauri app is running and connected."
                    )
                raise RuntimeError(f"Failed to capture snapshot: {error_detail}")

            data = response.json()

            # Check for API-level error (HTTP 200 but success: false)
            if data.get("success") is False:
                error_msg = data.get("error", "Unknown error from UI Bridge")
                raise RuntimeError(f"Snapshot failed: {error_msg}")

            result_data = data.get("data", data)

            elements = []
            elem_list = result_data.get("elements", []) if isinstance(result_data, dict) else []
            for elem_data in elem_list:
                elements.append(Element.from_dict(elem_data))

            ts_raw = result_data.get("timestamp") if isinstance(result_data, dict) else None
            if isinstance(ts_raw, int | float):
                timestamp = datetime.fromtimestamp(ts_raw / 1000)
            elif isinstance(ts_raw, str):
                try:
                    timestamp = datetime.fromisoformat(ts_raw)
                except ValueError:
                    timestamp = datetime.now()
            else:
                timestamp = datetime.now()

            return DOMSnapshot(
                id=result_data.get("id", "") if isinstance(result_data, dict) else "",
                timestamp=timestamp,
                url=(
                    result_data.get("url", self._connection_url)
                    if isinstance(result_data, dict)
                    else self._connection_url
                ),
                title=result_data.get("title", "") if isinstance(result_data, dict) else "",
                root=result_data.get("root", {}) if isinstance(result_data, dict) else {},
                elements=elements,
                screenshot_base64=(
                    result_data.get("screenshotBase64") if isinstance(result_data, dict) else None
                ),
            )

        except httpx.ConnectError as e:
            error_msg = (
                f"Lost connection to runner at {self._connection_url}. "
                f"The runner may have stopped or restarted."
            )
            logger.error(f"{error_msg} Error: {e}")
            self._connected = False
            raise RuntimeError(error_msg) from e

        except httpx.TimeoutException as e:
            error_msg = (
                f"Request to runner at {self._connection_url} timed out. "
                f"The runner or desktop app may be unresponsive."
            )
            logger.error(f"{error_msg} Error: {e}")
            raise RuntimeError(error_msg) from e

        except httpx.RequestError as e:
            logger.error(f"capture_snapshot request failed: {e}")
            raise RuntimeError(f"Failed to capture snapshot: {e}") from e


class MobileTargetConnection(TargetConnection):
    """Connection handler for React Native mobile applications via qontinui-runner.

    Connects to React Native applications via the qontinui-runner's HTTP API.
    The runner forwards requests to the connected mobile app via WebSocket.

    This uses the same API as DesktopTargetConnection since the runner handles
    both desktop (Tauri) and mobile (React Native) connections through the
    same UI Bridge control endpoints.

    API endpoints used:
    - GET /health - Check runner availability
    - GET /ui-bridge/control/elements - Get all registered elements
    - GET /ui-bridge/control/element/:id - Get specific element by ID
    - POST /ui-bridge/control/element/:id/action - Execute action on element
    - POST /ui-bridge/control/discover - Discover elements with filtering options
    - GET /ui-bridge/control/snapshot - Capture full UI state snapshot

    Example:
        >>> async with MobileTargetConnection("http://localhost:9876") as conn:
        ...     elements = await conn.find_elements()
        ...     await conn.execute_action("btn-login", "click")
    """

    # Runner's UI Bridge control API prefix (same as desktop)
    UI_BRIDGE_API_PREFIX = "/ui-bridge/control"

    def __init__(
        self,
        connection_url: str,
        timeout_seconds: float = 30.0,
    ):
        """Initialize mobile connection.

        Args:
            connection_url: Runner URL with port (e.g., http://localhost:9876)
            timeout_seconds: Request timeout
        """
        super().__init__(connection_url, timeout_seconds)

    def _get_api_url(self, endpoint: str) -> str:
        """Build full API URL for an endpoint."""
        base = self._connection_url.rstrip("/")
        return f"{base}{self.UI_BRIDGE_API_PREFIX}/{endpoint.lstrip('/')}"

    async def connect(self) -> bool:
        """Connect to the mobile application via qontinui-runner.

        Verifies the runner is running and UI Bridge control API is available.

        Returns:
            True if connection successful

        Raises:
            ConnectionError: With descriptive message if connection fails
        """
        if self._connected:
            return True

        self._client = httpx.AsyncClient(timeout=self._timeout)

        try:
            # Check runner health endpoint
            response = await self._client.get(f"{self._connection_url.rstrip('/')}/health")

            if response.is_success:
                # Check if the response indicates a connected mobile app
                try:
                    health_data = response.json()
                    # If the runner reports no connected app, provide helpful error
                    if health_data.get("mobile_app_connected") is False:
                        error_msg = (
                            f"No mobile application connected to the runner at {self._connection_url}. "
                            f"Start your React Native app and ensure the UI Bridge SDK connects to the runner. "
                            f"The app should connect via WebSocket when it starts."
                        )
                        logger.warning(error_msg)
                        # Still mark as connected to runner, but warn about missing app
                except (ValueError, KeyError):
                    pass  # Health endpoint may not return JSON or this field

                self._connected = True
                logger.info(f"Connected to runner for mobile target at {self._connection_url}")
                return True

            logger.warning(
                f"Runner not available at {self._connection_url}: {response.status_code}"
            )
            return False

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
        """Disconnect from the mobile application."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._connected = False
        logger.info(f"Disconnected from runner for mobile target at {self._connection_url}")

    async def find_elements(self, selector: str | None = None) -> list[Element]:
        """Find interactive elements via runner's UI Bridge control API.

        Uses POST /ui-bridge/control/discover for discovery with filtering options,
        or falls back to GET /ui-bridge/control/elements for basic listing.

        Args:
            selector: Optional selector to filter elements

        Returns:
            List of Element objects
        """
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to runner")

        try:
            # Use discover endpoint for more control (POST with camelCase body)
            payload: dict[str, Any] = {
                "interactiveOnly": True,
                "includeHidden": False,
            }
            if selector:
                payload["selector"] = selector

            response = await self._client.post(
                self._get_api_url("discover"),
                json=payload,
            )

            if not response.is_success:
                # Fall back to elements endpoint (GET, no body needed)
                response = await self._client.get(self._get_api_url("elements"))
                if not response.is_success:
                    logger.warning(f"find_elements failed: {response.status_code}")
                    return []

            data = response.json()
            elements = []

            # Handle runner's API response format: {success: bool, data: ...}
            result_data = data.get("data", data)
            elem_list = (
                result_data.get("elements", result_data)
                if isinstance(result_data, dict)
                else result_data
            )

            if isinstance(elem_list, list):
                for elem_data in elem_list:
                    elements.append(Element.from_dict(elem_data))

            logger.debug(f"Found {len(elements)} elements in mobile app via runner")
            return elements

        except httpx.ConnectError as e:
            error_msg = (
                f"Lost connection to runner at {self._connection_url}. "
                f"The runner may have stopped or restarted."
            )
            logger.error(f"{error_msg} Error: {e}")
            self._connected = False
            return []

        except httpx.TimeoutException as e:
            logger.error(
                f"Request to runner at {self._connection_url} timed out. "
                f"The runner or mobile app may be unresponsive. Error: {e}"
            )
            return []

        except httpx.RequestError as e:
            logger.error(f"find_elements request failed: {e}")
            return []

    async def execute_action(
        self,
        element_id: str,
        action: str,
        value: str | None = None,
    ) -> ActionResult:
        """Execute an action on an element via runner's UI Bridge control API.

        Uses POST /ui-bridge/control/element/:id/action endpoint.
        The request body uses camelCase: {action, params, waitOptions}

        Args:
            element_id: Element's data-ui-id value
            action: Action to perform (click, type, clear, focus, blur, etc.)
            value: Optional value for the action (e.g., text to type)

        Returns:
            ActionResult with execution details
        """
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to runner")

        import time

        start_time = time.time()

        try:
            # Runner's action endpoint expects camelCase body for mobile
            payload: dict[str, Any] = {
                "action": action,
            }
            if value is not None:
                payload["params"] = {"value": value}

            response = await self._client.post(
                self._get_api_url(f"element/{element_id}/action"),
                json=payload,
            )

            response_time_ms = int((time.time() - start_time) * 1000)

            if not response.is_success:
                error_detail = response.text
                if response.status_code == 404:
                    error_detail = (
                        f"Element '{element_id}' not found in the mobile app. "
                        f"The element may have been removed or the app may have navigated away."
                    )
                elif response.status_code == 503:
                    error_detail = (
                        "No mobile application connected to the runner. "
                        "Make sure your React Native app is running and connected."
                    )
                return ActionResult(
                    success=False,
                    element_id=element_id,
                    action=action,
                    error=f"HTTP {response.status_code}: {error_detail}",
                    response_time_ms=response_time_ms,
                )

            data = response.json()
            result_data = data.get("data", data)

            return ActionResult(
                success=data.get("success", True),
                element_id=element_id,
                action=action,
                error=data.get("error"),
                response_time_ms=response_time_ms,
                state_changed=(
                    result_data.get("stateChanged", False)
                    if isinstance(result_data, dict)
                    else False
                ),
                new_elements=(
                    result_data.get("newElements", []) if isinstance(result_data, dict) else []
                ),
                removed_elements=(
                    result_data.get("removedElements", []) if isinstance(result_data, dict) else []
                ),
            )

        except httpx.ConnectError as e:
            response_time_ms = int((time.time() - start_time) * 1000)
            error_msg = (
                f"Lost connection to runner at {self._connection_url}. "
                f"The runner may have stopped or restarted."
            )
            logger.error(f"{error_msg} Error: {e}")
            self._connected = False
            return ActionResult(
                success=False,
                element_id=element_id,
                action=action,
                error=error_msg,
                response_time_ms=response_time_ms,
            )

        except httpx.TimeoutException as e:
            response_time_ms = int((time.time() - start_time) * 1000)
            error_msg = (
                f"Action '{action}' on element '{element_id}' timed out. "
                f"The runner or mobile app may be unresponsive."
            )
            logger.error(f"{error_msg} Error: {e}")
            return ActionResult(
                success=False,
                element_id=element_id,
                action=action,
                error=error_msg,
                response_time_ms=response_time_ms,
            )

        except httpx.RequestError as e:
            response_time_ms = int((time.time() - start_time) * 1000)
            logger.error(f"execute_action request failed: {e}")
            return ActionResult(
                success=False,
                element_id=element_id,
                action=action,
                error=str(e),
                response_time_ms=response_time_ms,
            )

    async def capture_snapshot(self, include_screenshot: bool = False) -> DOMSnapshot:
        """Capture current DOM snapshot via runner's UI Bridge control API.

        Uses GET /ui-bridge/control/snapshot endpoint.

        Args:
            include_screenshot: Whether to include base64 screenshot

        Returns:
            DOMSnapshot of current state
        """
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to runner")

        try:
            params = {"includeScreenshot": str(include_screenshot).lower()}

            response = await self._client.get(
                self._get_api_url("snapshot"),
                params=params,
            )

            if not response.is_success:
                error_detail = f"HTTP {response.status_code}"
                if response.status_code == 503:
                    error_detail = (
                        "No mobile application connected to the runner. "
                        "Make sure your React Native app is running and connected."
                    )
                raise RuntimeError(f"Failed to capture snapshot: {error_detail}")

            data = response.json()

            # Check for API-level error (HTTP 200 but success: false)
            if data.get("success") is False:
                error_msg = data.get("error", "Unknown error from UI Bridge")
                raise RuntimeError(f"Snapshot failed: {error_msg}")

            result_data = data.get("data", data)

            elements = []
            elem_list = result_data.get("elements", []) if isinstance(result_data, dict) else []
            for elem_data in elem_list:
                elements.append(Element.from_dict(elem_data))

            ts_raw = result_data.get("timestamp") if isinstance(result_data, dict) else None
            if isinstance(ts_raw, int | float):
                timestamp = datetime.fromtimestamp(ts_raw / 1000)
            elif isinstance(ts_raw, str):
                try:
                    timestamp = datetime.fromisoformat(ts_raw)
                except ValueError:
                    timestamp = datetime.now()
            else:
                timestamp = datetime.now()

            return DOMSnapshot(
                id=result_data.get("id", "") if isinstance(result_data, dict) else "",
                timestamp=timestamp,
                url=(
                    result_data.get("url", self._connection_url)
                    if isinstance(result_data, dict)
                    else self._connection_url
                ),
                title=result_data.get("title", "") if isinstance(result_data, dict) else "",
                root=result_data.get("root", {}) if isinstance(result_data, dict) else {},
                elements=elements,
                screenshot_base64=(
                    result_data.get("screenshotBase64") if isinstance(result_data, dict) else None
                ),
            )

        except httpx.ConnectError as e:
            error_msg = (
                f"Lost connection to runner at {self._connection_url}. "
                f"The runner may have stopped or restarted."
            )
            logger.error(f"{error_msg} Error: {e}")
            self._connected = False
            raise RuntimeError(error_msg) from e

        except httpx.TimeoutException as e:
            error_msg = (
                f"Request to runner at {self._connection_url} timed out. "
                f"The runner or mobile app may be unresponsive."
            )
            logger.error(f"{error_msg} Error: {e}")
            raise RuntimeError(error_msg) from e

        except httpx.RequestError as e:
            logger.error(f"capture_snapshot request failed: {e}")
            raise RuntimeError(f"Failed to capture snapshot: {e}") from e


def create_connection(config: ExplorationConfig) -> TargetConnection:
    """Factory function to create the appropriate connection type.

    Args:
        config: Exploration configuration with target type and URL

    Returns:
        Appropriate TargetConnection subclass instance

    Raises:
        ValueError: If target_type is not recognized
    """
    if config.target_type == "web":
        return WebTargetConnection(
            connection_url=config.connection_url,
            timeout_seconds=config.timeout_seconds,
        )
    elif config.target_type == "desktop":
        return DesktopTargetConnection(
            connection_url=config.connection_url,
            timeout_seconds=config.timeout_seconds,
        )
    elif config.target_type == "mobile":
        return MobileTargetConnection(
            connection_url=config.connection_url,
            timeout_seconds=config.timeout_seconds,
        )
    elif config.target_type == "extension":
        from .extension_connection import ExtensionTargetConnection

        return ExtensionTargetConnection(
            runner_url=config.connection_url,
            timeout_seconds=config.timeout_seconds,
        )
    else:
        raise ValueError(f"Unknown target type: {config.target_type}")
