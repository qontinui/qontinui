"""Chrome DevTools Protocol (CDP) accessibility capture implementation.

This module provides accessibility tree capture via CDP, supporting:
- Chrome/Chromium browsers
- Edge browser
- Electron applications
- Tauri applications (Windows only, via WebView2)

CDP provides access to the browser's accessibility tree, which is a semantic
representation of the DOM optimized for assistive technologies. This tree
is much smaller than the DOM and ideal for AI-driven automation.
"""

import asyncio
import json
import logging
import re
import time
from typing import Any

from qontinui_schemas.accessibility import (
    AccessibilityBackend,
    AccessibilityBounds,
    AccessibilityCaptureOptions,
    AccessibilityConfig,
    AccessibilityNode,
    AccessibilityRole,
    AccessibilitySelector,
    AccessibilitySnapshot,
    AccessibilityState,
)

from qontinui.hal.implementations.accessibility.ref_manager import RefManager
from qontinui.hal.interfaces.accessibility_capture import IAccessibilityCapture

logger = logging.getLogger(__name__)


class CDPAccessibilityCapture(IAccessibilityCapture):
    """CDP-based accessibility capture implementation.

    Uses Chrome DevTools Protocol to capture accessibility trees from
    web browsers and Chromium-based applications.

    Example:
        >>> capture = CDPAccessibilityCapture()
        >>> await capture.connect(port=9222)
        >>> snapshot = await capture.capture_tree()
        >>> print(capture.to_ai_context(snapshot))
        >>> await capture.click_by_ref("@e3")
        >>> await capture.disconnect()
    """

    def __init__(self, config: AccessibilityConfig | None = None) -> None:
        """Initialize CDP capture.

        Args:
            config: Accessibility configuration. If None, uses defaults.
        """
        self._config = config or AccessibilityConfig()
        self._ws: Any = None  # aiohttp WebSocket connection
        self._session: Any = None  # aiohttp ClientSession
        self._ws_url: str | None = None
        self._message_id: int = 0
        self._pending_responses: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._ref_manager = RefManager()
        self._current_snapshot: AccessibilitySnapshot | None = None
        self._receive_task: asyncio.Task[None] | None = None
        # Map ref -> backendDOMNodeId for bounds retrieval and interaction
        self._ref_to_backend_node_id: dict[str, int] = {}

    async def connect(
        self,
        target: str | int | None = None,
        *,
        host: str = "localhost",
        port: int = 9222,
        timeout: float = 30.0,
    ) -> bool:
        """Connect to a browser via CDP.

        Args:
            target: Target page URL, index, or None for first available
            host: CDP host
            port: CDP port
            timeout: Connection timeout

        Returns:
            True if connected successfully
        """
        import aiohttp

        # Close existing connection
        if self.is_connected():
            await self.disconnect()

        try:
            self._session = aiohttp.ClientSession()

            # Get list of available targets
            targets_url = f"http://{host}:{port}/json"
            async with asyncio.timeout(timeout):
                async with self._session.get(targets_url) as resp:
                    if resp.status != 200:
                        logger.error(f"Failed to get targets: {resp.status}")
                        return False
                    targets = await resp.json()

            # Filter to page targets
            page_targets = [t for t in targets if t.get("type") == "page"]
            if not page_targets:
                logger.error("No page targets available")
                return False

            # Select target
            selected_target = None
            if target is None:
                selected_target = page_targets[0]
            elif isinstance(target, int):
                if 0 <= target < len(page_targets):
                    selected_target = page_targets[target]
            elif isinstance(target, str):
                # Match by URL
                for t in page_targets:
                    if target in t.get("url", ""):
                        selected_target = t
                        break

            if not selected_target:
                logger.error(f"Target not found: {target}")
                return False

            # Connect to WebSocket
            self._ws_url = selected_target.get("webSocketDebuggerUrl")
            if not self._ws_url:
                logger.error("No WebSocket URL for target")
                return False

            async with asyncio.timeout(timeout):
                self._ws = await self._session.ws_connect(self._ws_url)

            # Start message receive task
            self._receive_task = asyncio.create_task(self._receive_messages())

            # Enable required CDP domains
            await self._send_command("Accessibility.enable")
            await self._send_command("DOM.enable")
            await self._send_command("Page.enable")

            logger.info(f"Connected to CDP: {selected_target.get('url', 'unknown')}")
            return True

        except TimeoutError:
            logger.error(f"Connection timeout after {timeout}s")
            await self.disconnect()
            return False
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            await self.disconnect()
            return False

    async def disconnect(self) -> None:
        """Disconnect from CDP."""
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        if self._ws:
            await self._ws.close()
            self._ws = None

        if self._session:
            await self._session.close()
            self._session = None

        self._ws_url = None
        self._pending_responses.clear()
        self._ref_to_backend_node_id.clear()

    def is_connected(self) -> bool:
        """Check if connected to CDP.

        Returns:
            True if connected
        """
        return self._ws is not None and not self._ws.closed

    async def _send_command(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Send CDP command and wait for response.

        Args:
            method: CDP method name
            params: Method parameters
            timeout: Response timeout

        Returns:
            Response result

        Raises:
            RuntimeError: If not connected or command fails
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to CDP")

        self._message_id += 1
        msg_id = self._message_id

        message = {"id": msg_id, "method": method}
        if params:
            message["params"] = params

        # Create future for response
        future: asyncio.Future[dict[str, Any]] = asyncio.get_event_loop().create_future()
        self._pending_responses[msg_id] = future

        try:
            await self._ws.send_json(message)
            async with asyncio.timeout(timeout):
                result = await future
                return result
        except TimeoutError as e:
            raise RuntimeError(f"CDP command timeout: {method}") from e
        finally:
            self._pending_responses.pop(msg_id, None)

    async def _receive_messages(self) -> None:
        """Background task to receive CDP messages."""
        import aiohttp

        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)

                    # Handle response
                    if "id" in data:
                        msg_id = data["id"]
                        future = self._pending_responses.get(msg_id)
                        if future and not future.done():
                            if "error" in data:
                                future.set_exception(
                                    RuntimeError(data["error"].get("message", "CDP error"))
                                )
                            else:
                                future.set_result(data.get("result", {}))

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {self._ws.exception()}")
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Receive error: {e}")

    async def capture_tree(
        self,
        options: AccessibilityCaptureOptions | None = None,
    ) -> AccessibilitySnapshot:
        """Capture the accessibility tree via CDP.

        Args:
            options: Capture options

        Returns:
            AccessibilitySnapshot with the captured tree
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to CDP")

        opts = options or AccessibilityCaptureOptions()
        config = opts.config

        # Clear ref to backend node ID mapping for new capture
        self._ref_to_backend_node_id.clear()

        # Get the full accessibility tree
        result = await self._send_command(
            "Accessibility.getFullAXTree",
            {"depth": config.max_depth} if config.max_depth else None,
        )

        ax_nodes = result.get("nodes", [])
        if not ax_nodes:
            raise RuntimeError("No accessibility nodes returned")

        # Build node index
        node_index: dict[str, dict[str, Any]] = {}
        for ax_node in ax_nodes:
            node_id = ax_node.get("nodeId")
            if node_id:
                node_index[node_id] = ax_node

        # Find root node
        root_ax = ax_nodes[0] if ax_nodes else None
        if not root_ax:
            raise RuntimeError("No root accessibility node")

        # Convert to our model
        root_node = self._convert_ax_node(root_ax, node_index, config)

        # Assign refs
        total_refs = self._ref_manager.assign_refs(
            root_node,
            interactive_only=config.interactive_only,
        )

        # Build ref to backend DOM node ID mapping
        self._build_ref_to_backend_node_mapping(root_node)

        # Get page info
        page_info = await self._send_command("Page.getNavigationHistory")
        current_entry = None
        entries = page_info.get("entries", [])
        current_index = page_info.get("currentIndex", 0)
        if 0 <= current_index < len(entries):
            current_entry = entries[current_index]

        # Count interactive nodes
        interactive_count = len(self._ref_manager.get_interactive_nodes())

        # Create snapshot
        snapshot = AccessibilitySnapshot(
            root=root_node,
            timestamp=time.time(),
            backend=AccessibilityBackend.CDP,
            url=current_entry.get("url") if current_entry else None,
            title=current_entry.get("title") if current_entry else None,
            total_nodes=total_refs,
            interactive_nodes=interactive_count,
        )

        self._current_snapshot = snapshot
        return snapshot

    def _convert_ax_node(
        self,
        ax_node: dict[str, Any],
        node_index: dict[str, dict[str, Any]],
        config: AccessibilityConfig,
    ) -> AccessibilityNode:
        """Convert CDP AX node to our model.

        Args:
            ax_node: CDP accessibility node
            node_index: Index of all nodes by ID
            config: Capture configuration

        Returns:
            AccessibilityNode
        """
        # Extract role
        role_info = ax_node.get("role", {})
        role_value = role_info.get("value", "generic")
        try:
            role = AccessibilityRole(role_value)
        except ValueError:
            role = AccessibilityRole.GENERIC

        # Extract name
        name_info = ax_node.get("name", {})
        name = name_info.get("value") if name_info.get("value") else None

        # Extract value
        value_info = ax_node.get("value", {})
        value = value_info.get("value") if value_info.get("value") else None

        # Extract description
        desc_info = ax_node.get("description", {})
        description = desc_info.get("value") if desc_info.get("value") else None

        # Extract state properties
        properties = ax_node.get("properties", [])
        state = self._extract_state(properties)

        # Determine if interactive
        is_interactive = self._is_interactive(role, state, properties)

        # Store backendDOMNodeId for later interaction (bounds, click, focus, etc.)
        # We store it in automation_id field temporarily; will be moved to internal mapping
        backend_dom_node_id = ax_node.get("backendDOMNodeId")
        automation_id_str = str(backend_dom_node_id) if backend_dom_node_id else None

        # Process children
        children: list[AccessibilityNode] = []
        child_ids = ax_node.get("childIds", [])
        for child_id in child_ids:
            child_ax = node_index.get(child_id)
            if child_ax:
                # Skip hidden nodes if not configured to include them
                if not config.include_hidden:
                    child_props = child_ax.get("properties", [])
                    if self._is_hidden(child_props):
                        continue
                children.append(self._convert_ax_node(child_ax, node_index, config))

        return AccessibilityNode(
            ref="",  # Will be assigned by RefManager
            role=role,
            name=name,
            value=value if config.include_value else None,
            description=description,
            bounds=None,  # Bounds will be fetched on-demand via _get_bounds_for_node
            state=state,
            is_interactive=is_interactive,
            automation_id=automation_id_str,  # Store backendDOMNodeId here temporarily
            children=children,
        )

    def _extract_state(self, properties: list[dict[str, Any]]) -> AccessibilityState:
        """Extract state flags from CDP properties.

        Args:
            properties: CDP property list

        Returns:
            AccessibilityState
        """
        state = AccessibilityState()

        for prop in properties:
            name = prop.get("name")
            value = prop.get("value", {}).get("value")

            if name == "focused":
                state.is_focused = bool(value)
            elif name == "disabled":
                state.is_disabled = bool(value)
            elif name == "hidden":
                state.is_hidden = bool(value)
            elif name == "expanded":
                state.is_expanded = bool(value)
            elif name == "selected":
                state.is_selected = bool(value)
            elif name == "checked":
                if value == "true":
                    state.is_checked = True
                elif value == "false":
                    state.is_checked = False
                elif value == "mixed":
                    state.is_checked = None  # Indeterminate
            elif name == "pressed":
                state.is_pressed = bool(value)
            elif name == "readonly":
                state.is_readonly = bool(value)
            elif name == "required":
                state.is_required = bool(value)
            elif name == "multiselectable":
                state.is_multiselectable = bool(value)
            elif name == "editable":
                state.is_editable = bool(value)
            elif name == "focusable":
                state.is_focusable = bool(value)
            elif name == "modal":
                state.is_modal = bool(value)

        return state

    def _is_interactive(
        self,
        role: AccessibilityRole,
        state: AccessibilityState,
        properties: list[dict[str, Any]],
    ) -> bool:
        """Determine if a node is interactive.

        Args:
            role: Node role
            state: Node state
            properties: CDP properties

        Returns:
            True if interactive
        """
        # Disabled elements are not interactive
        if state.is_disabled:
            return False

        # Interactive roles
        interactive_roles = {
            AccessibilityRole.BUTTON,
            AccessibilityRole.CHECKBOX,
            AccessibilityRole.COMBOBOX,
            AccessibilityRole.LINK,
            AccessibilityRole.LISTBOX,
            AccessibilityRole.MENU,
            AccessibilityRole.MENUITEM,
            AccessibilityRole.MENUITEMCHECKBOX,
            AccessibilityRole.MENUITEMRADIO,
            AccessibilityRole.OPTION,
            AccessibilityRole.RADIO,
            AccessibilityRole.SEARCHBOX,
            AccessibilityRole.SLIDER,
            AccessibilityRole.SPINBUTTON,
            AccessibilityRole.SWITCH,
            AccessibilityRole.TAB,
            AccessibilityRole.TEXTBOX,
            AccessibilityRole.TREEITEM,
        }

        if role in interactive_roles:
            return True

        # Check for focusable property
        if state.is_focusable:
            return True

        return False

    def _is_hidden(self, properties: list[dict[str, Any]]) -> bool:
        """Check if node is hidden.

        Args:
            properties: CDP properties

        Returns:
            True if hidden
        """
        for prop in properties:
            if prop.get("name") == "hidden":
                return bool(prop.get("value", {}).get("value"))
        return False

    def _build_ref_to_backend_node_mapping(self, node: AccessibilityNode) -> None:
        """Build mapping from ref to backendDOMNodeId by walking the tree.

        Args:
            node: Current node to process
        """
        # If node has a ref and automation_id (which stores backendDOMNodeId)
        if node.ref and node.automation_id:
            try:
                backend_node_id = int(node.automation_id)
                self._ref_to_backend_node_id[node.ref] = backend_node_id
            except ValueError:
                pass  # Not a valid backend node ID

        # Process children
        for child in node.children:
            self._build_ref_to_backend_node_mapping(child)

    async def _get_bounds_for_backend_node(
        self, backend_node_id: int
    ) -> AccessibilityBounds | None:
        """Get bounds for a node using CDP DOM.getBoxModel.

        Args:
            backend_node_id: The backend DOM node ID

        Returns:
            AccessibilityBounds if available, None otherwise
        """
        if not self.is_connected():
            return None

        try:
            # Use DOM.getBoxModel to get the element's bounding box
            result = await self._send_command(
                "DOM.getBoxModel",
                {"backendNodeId": backend_node_id},
            )

            model = result.get("model")
            if not model:
                return None

            # The content quad gives the innermost bounds
            # Format: [x1, y1, x2, y2, x3, y3, x4, y4] for the 4 corners
            content = model.get("content")
            if not content or len(content) < 8:
                # Fall back to border quad
                content = model.get("border")
                if not content or len(content) < 8:
                    return None

            # Extract bounding rectangle from quad (4 corners)
            # Quad format: [x1, y1, x2, y2, x3, y3, x4, y4]
            # Corners: top-left, top-right, bottom-right, bottom-left
            x_coords = [content[0], content[2], content[4], content[6]]
            y_coords = [content[1], content[3], content[5], content[7]]

            x = int(min(x_coords))
            y = int(min(y_coords))
            width = int(max(x_coords) - x)
            height = int(max(y_coords) - y)

            if width > 0 and height > 0:
                return AccessibilityBounds(x=x, y=y, width=width, height=height)

            return None

        except Exception as e:
            logger.debug(f"Failed to get bounds for backend node {backend_node_id}: {e}")
            return None

    async def _get_bounds_for_ref(self, ref: str) -> AccessibilityBounds | None:
        """Get bounds for a node by its ref.

        Args:
            ref: Reference ID

        Returns:
            AccessibilityBounds if available, None otherwise
        """
        backend_node_id = self._ref_to_backend_node_id.get(ref)
        if not backend_node_id:
            return None

        return await self._get_bounds_for_backend_node(backend_node_id)

    async def get_node_by_ref(self, ref: str) -> AccessibilityNode | None:
        """Get a node by its ref ID.

        Args:
            ref: Reference ID

        Returns:
            Node if found
        """
        return self._ref_manager.get_node_by_ref(ref)

    async def find_nodes(
        self,
        selector: AccessibilitySelector,
    ) -> list[AccessibilityNode]:
        """Find nodes matching a selector.

        Args:
            selector: Selection criteria

        Returns:
            List of matching nodes
        """
        if not self._current_snapshot:
            return []

        matches: list[AccessibilityNode] = []
        self._find_nodes_recursive(self._current_snapshot.root, selector, matches)
        return matches

    def _find_nodes_recursive(
        self,
        node: AccessibilityNode,
        selector: AccessibilitySelector,
        matches: list[AccessibilityNode],
    ) -> None:
        """Recursively find matching nodes.

        Args:
            node: Current node
            selector: Selection criteria
            matches: List to append matches to
        """
        if self._matches_selector(node, selector):
            matches.append(node)

        for child in node.children:
            self._find_nodes_recursive(child, selector, matches)

    def _matches_selector(
        self,
        node: AccessibilityNode,
        selector: AccessibilitySelector,
    ) -> bool:
        """Check if node matches selector.

        Args:
            node: Node to check
            selector: Selection criteria

        Returns:
            True if matches
        """
        # Role match
        if selector.role is not None:
            if isinstance(selector.role, list):
                if node.role not in selector.role:
                    return False
            elif node.role != selector.role:
                return False

        # Name matches
        if selector.name is not None:
            if node.name is None:
                return False
            if selector.case_sensitive:
                if node.name != selector.name:
                    return False
            else:
                if node.name.lower() != selector.name.lower():
                    return False

        if selector.name_contains is not None:
            if node.name is None:
                return False
            name = node.name if selector.case_sensitive else node.name.lower()
            contains = (
                selector.name_contains
                if selector.case_sensitive
                else selector.name_contains.lower()
            )
            if contains not in name:
                return False

        if selector.name_pattern is not None:
            if node.name is None:
                return False
            flags = 0 if selector.case_sensitive else re.IGNORECASE
            if not re.search(selector.name_pattern, node.name, flags):
                return False

        # Value matches
        if selector.value is not None:
            if node.value != selector.value:
                return False

        if selector.value_contains is not None:
            if node.value is None:
                return False
            if selector.value_contains not in node.value:
                return False

        # ID matches
        if selector.automation_id is not None:
            if node.automation_id != selector.automation_id:
                return False

        if selector.class_name is not None:
            if node.class_name != selector.class_name:
                return False

        if selector.html_tag is not None:
            if node.html_tag != selector.html_tag:
                return False

        # Interactive filter
        if selector.is_interactive is not None:
            if node.is_interactive != selector.is_interactive:
                return False

        return True

    async def click_by_ref(self, ref: str) -> bool:
        """Click an element by ref using bounds-based coordinates.

        Uses CDP Input.dispatchMouseEvent to perform a real mouse click
        at the center of the element's bounding box.

        Args:
            ref: Reference ID

        Returns:
            True if clicked successfully
        """
        node = await self.get_node_by_ref(ref)
        if not node:
            logger.warning(f"Node not found for ref: {ref}")
            return False

        # Get bounds for the element
        bounds = await self._get_bounds_for_ref(ref)
        if not bounds:
            logger.warning(f"Could not get bounds for ref: {ref}")
            # Fall back to DOM.focus + click via JavaScript
            return await self._click_by_ref_fallback(ref)

        # Calculate center point
        center_x = bounds.center_x
        center_y = bounds.center_y

        try:
            # Perform mouse click sequence: move -> mousePressed -> mouseReleased
            # First move to the location
            await self._send_command(
                "Input.dispatchMouseEvent",
                {
                    "type": "mouseMoved",
                    "x": center_x,
                    "y": center_y,
                },
            )

            # Mouse down
            await self._send_command(
                "Input.dispatchMouseEvent",
                {
                    "type": "mousePressed",
                    "x": center_x,
                    "y": center_y,
                    "button": "left",
                    "clickCount": 1,
                },
            )

            # Mouse up
            await self._send_command(
                "Input.dispatchMouseEvent",
                {
                    "type": "mouseReleased",
                    "x": center_x,
                    "y": center_y,
                    "button": "left",
                    "clickCount": 1,
                },
            )

            logger.info(f"Clicked {ref} at ({center_x}, {center_y})")
            return True

        except Exception as e:
            logger.error(f"Click failed for {ref}: {e}")
            return False

    async def _click_by_ref_fallback(self, ref: str) -> bool:
        """Fallback click using DOM.focus and JavaScript click.

        Used when bounds are not available.

        Args:
            ref: Reference ID

        Returns:
            True if clicked successfully
        """
        backend_node_id = self._ref_to_backend_node_id.get(ref)
        if not backend_node_id:
            logger.warning(f"No backend node ID for ref: {ref}")
            return False

        try:
            # Resolve the backend node ID to a regular node ID
            result = await self._send_command(
                "DOM.describeNode",
                {"backendNodeId": backend_node_id},
            )

            node_info = result.get("node", {})
            node_id = node_info.get("nodeId")

            if not node_id:
                # Try to push node to get a node ID
                result = await self._send_command(
                    "DOM.pushNodesByBackendIdsToFrontend",
                    {"backendNodeIds": [backend_node_id]},
                )
                node_ids = result.get("nodeIds", [])
                if node_ids:
                    node_id = node_ids[0]

            if not node_id:
                logger.warning(f"Could not resolve node ID for ref: {ref}")
                return False

            # Focus the element
            await self._send_command("DOM.focus", {"nodeId": node_id})

            # Get the remote object for the node
            result = await self._send_command(
                "DOM.resolveNode",
                {"nodeId": node_id},
            )

            object_info = result.get("object", {})
            object_id = object_info.get("objectId")

            if object_id:
                # Call click() on the element
                await self._send_command(
                    "Runtime.callFunctionOn",
                    {
                        "objectId": object_id,
                        "functionDeclaration": "function() { this.click(); }",
                        "returnByValue": True,
                    },
                )
                logger.info(f"Clicked {ref} via JavaScript fallback")
                return True

            return False

        except Exception as e:
            logger.error(f"Fallback click failed for {ref}: {e}")
            return False

    async def type_by_ref(
        self,
        ref: str,
        text: str,
        *,
        clear_first: bool = False,
    ) -> bool:
        """Type text into an element by ref.

        Focuses the element first, optionally clears existing content,
        then types text character by character using proper key events.

        Args:
            ref: Reference ID
            text: Text to type
            clear_first: Clear existing content first

        Returns:
            True if typed successfully
        """
        node = await self.get_node_by_ref(ref)
        if not node:
            logger.warning(f"Node not found for ref: {ref}")
            return False

        # Focus the element first
        focused = await self.focus_by_ref(ref)
        if not focused:
            logger.warning(f"Could not focus element: {ref}")
            return False

        # Small delay after focus
        await asyncio.sleep(0.05)

        try:
            # Clear existing content if requested
            if clear_first:
                # Select all (Ctrl+A)
                await self._send_command(
                    "Input.dispatchKeyEvent",
                    {
                        "type": "keyDown",
                        "key": "a",
                        "code": "KeyA",
                        "modifiers": 2,  # Ctrl modifier
                    },
                )
                await self._send_command(
                    "Input.dispatchKeyEvent",
                    {
                        "type": "keyUp",
                        "key": "a",
                        "code": "KeyA",
                        "modifiers": 2,
                    },
                )

                # Delete
                await self._send_command(
                    "Input.dispatchKeyEvent",
                    {
                        "type": "keyDown",
                        "key": "Backspace",
                        "code": "Backspace",
                    },
                )
                await self._send_command(
                    "Input.dispatchKeyEvent",
                    {
                        "type": "keyUp",
                        "key": "Backspace",
                        "code": "Backspace",
                    },
                )

                await asyncio.sleep(0.02)

            # Type each character using Input.insertText for reliability
            # This handles Unicode and special characters better than key events
            await self._send_command(
                "Input.insertText",
                {"text": text},
            )

            logger.info(f"Typed into {ref}: {text[:20]}{'...' if len(text) > 20 else ''}")
            return True

        except Exception as e:
            logger.error(f"Type failed for {ref}: {e}")
            return False

    async def focus_by_ref(self, ref: str) -> bool:
        """Focus an element by ref.

        Uses CDP DOM.focus to properly focus the element. If that fails,
        falls back to clicking the element to focus it.

        Args:
            ref: Reference ID

        Returns:
            True if focused successfully
        """
        node = await self.get_node_by_ref(ref)
        if not node:
            logger.warning(f"Node not found for ref: {ref}")
            return False

        backend_node_id = self._ref_to_backend_node_id.get(ref)
        if not backend_node_id:
            logger.warning(f"No backend node ID for ref: {ref}")
            return False

        try:
            # Try DOM.focus with backend node ID directly
            await self._send_command(
                "DOM.focus",
                {"backendNodeId": backend_node_id},
            )
            logger.debug(f"Focused {ref} via DOM.focus")
            return True

        except Exception as e:
            logger.debug(f"DOM.focus failed for {ref}: {e}, trying fallback")

            # Fallback: resolve to object and call focus()
            try:
                # Push the node to frontend to get a node ID
                result = await self._send_command(
                    "DOM.pushNodesByBackendIdsToFrontend",
                    {"backendNodeIds": [backend_node_id]},
                )
                node_ids = result.get("nodeIds", [])

                if not node_ids:
                    logger.warning(f"Could not resolve node for ref: {ref}")
                    return False

                node_id = node_ids[0]

                # Try DOM.focus with the resolved node ID
                await self._send_command(
                    "DOM.focus",
                    {"nodeId": node_id},
                )
                logger.debug(f"Focused {ref} via resolved node ID")
                return True

            except Exception as e2:
                logger.debug(f"Fallback DOM.focus failed: {e2}, trying click focus")

                # Last resort: click to focus
                bounds = await self._get_bounds_for_ref(ref)
                if bounds:
                    await self._send_command(
                        "Input.dispatchMouseEvent",
                        {
                            "type": "mousePressed",
                            "x": bounds.center_x,
                            "y": bounds.center_y,
                            "button": "left",
                            "clickCount": 1,
                        },
                    )
                    await self._send_command(
                        "Input.dispatchMouseEvent",
                        {
                            "type": "mouseReleased",
                            "x": bounds.center_x,
                            "y": bounds.center_y,
                            "button": "left",
                            "clickCount": 1,
                        },
                    )
                    logger.debug(f"Focused {ref} via click")
                    return True

                logger.error(f"All focus methods failed for {ref}")
                return False

    def get_backend_name(self) -> str:
        """Get backend name.

        Returns:
            "cdp"
        """
        return "cdp"
