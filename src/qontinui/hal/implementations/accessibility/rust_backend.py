"""Rust backend accessibility capture.

Delegates all accessibility operations to the Rust runner's native
accessibility layer via HTTP API. The runner exposes accessibility
endpoints on its MCP API server (default port 9876).

This backend provides the fastest and most capable accessibility
capture by leveraging Rust-native platform adapters (UIA on Windows,
AT-SPI on Linux, AX on macOS).

Example:
    >>> capture = RustBackendCapture()
    >>> await capture.connect(target="desktop")
    >>> snapshot = await capture.capture_tree()
    >>> await capture.click_by_ref("@e3")
"""

from __future__ import annotations

import logging
import time
from typing import Any

import aiohttp
from qontinui_schemas.accessibility import (
    AccessibilityCaptureOptions,
    AccessibilityNode,
    AccessibilityRole,
    AccessibilitySelector,
    AccessibilitySnapshot,
    AccessibilityState,
)
from qontinui_schemas.accessibility.enums import AccessibilityBackend

from qontinui.hal.interfaces.accessibility_capture import IAccessibilityCapture

logger = logging.getLogger(__name__)


class RustBackendCapture(IAccessibilityCapture):
    """Delegates accessibility to the Rust runner's native accessibility layer.

    All operations are forwarded as HTTP requests to the runner's MCP API
    server. The runner performs the actual platform-specific accessibility
    operations using its Rust-native adapters.

    Args:
        runner_url: Base URL of the runner's HTTP API (default: http://localhost:9876).
        timeout: HTTP request timeout in seconds (default: 30).
    """

    def __init__(
        self,
        runner_url: str = "http://localhost:9876",
        timeout: float = 30.0,
    ) -> None:
        self._runner_url = runner_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._connected = False
        self._last_snapshot: AccessibilitySnapshot | None = None
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def _post(self, path: str, json: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make a POST request to the runner API.

        Args:
            path: API path (e.g., "/a11y/connect")
            json: Optional JSON body

        Returns:
            Response JSON dict

        Raises:
            RuntimeError: If the request fails
        """
        session = await self._get_session()
        url = f"{self._runner_url}{path}"
        try:
            async with session.post(url, json=json or {}) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    raise RuntimeError(f"Runner API error {resp.status} on POST {path}: {body}")
                return await resp.json()
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Failed to reach runner at {url}: {e}") from e

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make a GET request to the runner API.

        Args:
            path: API path
            params: Optional query parameters

        Returns:
            Response JSON dict

        Raises:
            RuntimeError: If the request fails
        """
        session = await self._get_session()
        url = f"{self._runner_url}{path}"
        try:
            async with session.get(url, params=params) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    raise RuntimeError(f"Runner API error {resp.status} on GET {path}: {body}")
                return await resp.json()
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Failed to reach runner at {url}: {e}") from e

    # ========================================================================
    # IAccessibilityCapture implementation
    # ========================================================================

    async def connect(
        self,
        target: str | int | None = None,
        *,
        host: str = "localhost",
        port: int = 9222,
        timeout: float = 30.0,
    ) -> bool:
        """Connect to an accessibility source via the Rust runner.

        Args:
            target: Target identifier. Accepts:
                - None or "desktop": entire desktop
                - str: window title or "pid:1234"
                - int: process ID
            host: Unused, kept for interface compatibility
            port: Unused, kept for interface compatibility
            timeout: Connection timeout in seconds

        Returns:
            True if connection succeeded
        """
        # Build target string for the runner
        if target is None:
            target_str = "desktop"
        elif isinstance(target, int):
            target_str = f"pid:{target}"
        else:
            target_str = str(target)

        try:
            result = await self._post(
                "/a11y/connect",
                json={
                    "target": target_str,
                    "backend": "auto",
                },
            )
            self._connected = result.get("connected", False)
            logger.info(
                "Rust backend connected: backend=%s",
                result.get("backend", "unknown"),
            )
            return self._connected
        except RuntimeError:
            logger.exception("Failed to connect via Rust backend")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from the current accessibility source."""
        try:
            await self._post("/a11y/disconnect")
        except RuntimeError:
            logger.debug("Disconnect request failed (runner may be down)", exc_info=True)
        finally:
            self._connected = False
            self._last_snapshot = None
            if self._session and not self._session.closed:
                await self._session.close()
                self._session = None

    def is_connected(self) -> bool:
        """Check if connected to an accessibility source."""
        return self._connected

    async def capture_tree(
        self,
        options: AccessibilityCaptureOptions | None = None,
    ) -> AccessibilitySnapshot:
        """Capture the accessibility tree via the Rust runner.

        Args:
            options: Capture options (max_depth, include_hidden, etc.)

        Returns:
            AccessibilitySnapshot with the captured tree

        Raises:
            RuntimeError: If not connected or capture fails
        """
        if not self._connected:
            raise RuntimeError("Not connected to an accessibility source")

        include_hidden = False
        max_depth = None
        if options is not None:
            include_hidden = options.config.include_hidden
            max_depth = options.config.max_depth

        result = await self._post(
            "/a11y/capture",
            json={
                "include_hidden": include_hidden,
                "max_depth": max_depth,
            },
        )

        snapshot = _parse_snapshot(result)
        self._last_snapshot = snapshot
        return snapshot

    async def get_node_by_ref(self, ref: str) -> AccessibilityNode | None:
        """Get a node by its ref ID from the last snapshot.

        Args:
            ref: Reference ID (e.g., "@e1")

        Returns:
            The node if found, None otherwise
        """
        if self._last_snapshot is None:
            return None
        return self._last_snapshot.get_node_by_ref(ref)

    async def find_nodes(
        self,
        selector: AccessibilitySelector,
    ) -> list[AccessibilityNode]:
        """Find nodes matching a selector via the Rust runner's query engine.

        Args:
            selector: Selection criteria

        Returns:
            List of matching nodes
        """
        # Use the runner's query endpoint for server-side filtering
        params: dict[str, Any] = {
            "interactive_only": selector.is_interactive or False,
        }

        if selector.role is not None:
            if isinstance(selector.role, list):
                # Query endpoint supports a single role; use first
                params["role"] = selector.role[0].value if selector.role else None
            else:
                params["role"] = selector.role.value

        if selector.name is not None:
            params["label"] = selector.name

        if selector.name_contains is not None:
            params["label_contains"] = selector.name_contains

        result = await self._post("/a11y/query", json=params)

        nodes_data = result.get("nodes", [])
        return [_parse_query_node(n) for n in nodes_data]

    async def click_by_ref(self, ref: str) -> bool:
        """Click an element by ref via the Rust runner.

        Args:
            ref: Reference ID of element to click

        Returns:
            True if click succeeded
        """
        result = await self._post("/a11y/click", json={"ref_id": ref})
        return result.get("success", False)

    async def type_by_ref(
        self,
        ref: str,
        text: str,
        *,
        clear_first: bool = False,
    ) -> bool:
        """Type text into an element by ref via the Rust runner.

        Args:
            ref: Reference ID of element to type into
            text: Text to type
            clear_first: If True, clear existing content before typing

        Returns:
            True if typing succeeded
        """
        result = await self._post(
            "/a11y/type_text",
            json={
                "ref_id": ref,
                "text": text,
                "clear_first": clear_first,
            },
        )
        return result.get("success", False)

    async def focus_by_ref(self, ref: str) -> bool:
        """Focus an element by ref via the Rust runner.

        Args:
            ref: Reference ID of element to focus

        Returns:
            True if focus succeeded
        """
        result = await self._post("/a11y/focus", json={"ref_id": ref})
        return result.get("success", False)

    def get_backend_name(self) -> str:
        """Get the name of this backend implementation."""
        return "rust"

    # ========================================================================
    # Health check
    # ========================================================================

    @classmethod
    async def is_runner_available(
        cls,
        runner_url: str = "http://localhost:9876",
        timeout: float = 2.0,
    ) -> bool:
        """Check if the Rust runner is reachable and has Rust accessibility enabled.

        Performs a health check and then queries the runner's accessibility
        settings to respect the ``use_rust_accessibility`` flag. If the
        runner is reachable but ``use_rust_accessibility`` is ``false``,
        this returns ``False`` so the Python HAL falls back to native
        Python backends.

        Args:
            runner_url: Base URL of the runner
            timeout: Timeout for the health check request

        Returns:
            True if the runner is reachable, healthy, **and** has
            ``use_rust_accessibility`` enabled.
        """
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as session:
                # 1. Health check — runner must be up
                async with session.get(f"{runner_url}/health") as resp:
                    if resp.status != 200:
                        return False

                # 2. Check use_rust_accessibility setting
                try:
                    async with session.get(f"{runner_url}/settings/accessibility") as settings_resp:
                        if settings_resp.status == 200:
                            data = await settings_resp.json()
                            # The runner wraps responses in ApiResponse with a
                            # "data" field; fall back to top-level keys for
                            # forward compatibility.
                            settings_data = data.get("data", data)
                            use_rust = settings_data.get("use_rust_accessibility", True)
                            if not use_rust:
                                logger.info(
                                    "Runner is available but use_rust_accessibility"
                                    " is disabled — skipping Rust backend"
                                )
                                return False
                except (aiohttp.ClientError, KeyError, ValueError):
                    # Settings endpoint unavailable (older runner version) —
                    # fall back to assuming Rust accessibility is enabled.
                    logger.debug(
                        "Could not query accessibility settings; "
                        "assuming use_rust_accessibility=true"
                    )

                return True
        except (aiohttp.ClientError, OSError):
            return False


# ============================================================================
# Parsing helpers
# ============================================================================


def _parse_snapshot(data: dict[str, Any]) -> AccessibilitySnapshot:
    """Parse a snapshot response from the Rust runner into a Pydantic model.

    The Rust runner serializes UnifiedSnapshot with snake_case fields that
    map closely to the Python AccessibilitySnapshot model, but require
    some field name translation.

    Args:
        data: Raw JSON dict from the runner

    Returns:
        AccessibilitySnapshot model
    """
    root_data = data.get("root", {})
    root_node = _parse_node(root_data)

    backend_str = data.get("source", data.get("backend", "unknown"))
    try:
        backend = AccessibilityBackend(backend_str)
    except ValueError:
        backend = AccessibilityBackend.AUTO

    return AccessibilitySnapshot(
        root=root_node,
        timestamp=data.get("timestamp", time.time()),
        backend=backend,
        url=data.get("url"),
        title=data.get("title"),
        total_nodes=data.get("total_nodes", 0),
        interactive_nodes=data.get("interactive_nodes", 0),
    )


def _parse_node(data: dict[str, Any]) -> AccessibilityNode:
    """Recursively parse a node from the Rust runner's JSON format.

    Args:
        data: Raw JSON dict for a single node

    Returns:
        AccessibilityNode model
    """
    # Parse role
    role_str = data.get("role", "unknown")
    try:
        role = AccessibilityRole(role_str)
    except ValueError:
        role = AccessibilityRole.UNKNOWN

    # Parse state
    state_data = data.get("state", {})
    state = AccessibilityState(
        is_focused=state_data.get("is_focused", False),
        is_disabled=state_data.get("is_disabled", False),
        is_hidden=state_data.get("is_hidden", False),
        is_expanded=state_data.get("is_expanded"),
        is_selected=state_data.get("is_selected"),
        is_checked=state_data.get("is_checked"),
        is_pressed=state_data.get("is_pressed"),
        is_readonly=state_data.get("is_readonly", False),
        is_required=state_data.get("is_required", False),
        is_editable=state_data.get("is_editable", False),
        is_focusable=state_data.get("is_focusable", False),
        is_modal=state_data.get("is_modal", False),
    )

    # Parse children recursively
    children = [_parse_node(c) for c in data.get("children", [])]

    return AccessibilityNode(
        ref=data.get("ref_id", data.get("ref", "")),
        role=role,
        name=data.get("name"),
        value=data.get("value"),
        description=data.get("description"),
        bounds=data.get("bounds"),
        state=state,
        is_interactive=data.get("is_interactive", False),
        level=data.get("level"),
        automation_id=data.get("automation_id"),
        class_name=data.get("class_name"),
        html_tag=data.get("html_tag"),
        url=data.get("url"),
        children=children,
    )


def _parse_query_node(data: dict[str, Any]) -> AccessibilityNode:
    """Parse a query result node (flat, no children).

    Args:
        data: Node dict from the query endpoint

    Returns:
        AccessibilityNode model (leaf node, no children)
    """
    role_str = data.get("role", "unknown")
    try:
        role = AccessibilityRole(role_str)
    except ValueError:
        role = AccessibilityRole.UNKNOWN

    return AccessibilityNode(
        ref=data.get("ref_id", data.get("ref", "")),
        role=role,
        name=data.get("name"),
        value=data.get("value"),
        is_interactive=data.get("is_interactive", False),
        bounds=data.get("bounds"),
    )
