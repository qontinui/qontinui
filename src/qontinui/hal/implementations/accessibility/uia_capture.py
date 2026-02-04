"""Windows UI Automation (UIA) accessibility capture implementation.

This module provides accessibility tree capture via Windows UI Automation,
supporting native Windows applications:
- Win32 applications
- WPF applications
- UWP applications
- Electron applications (native controls)
- Most desktop software on Windows

Windows UI Automation is the platform accessibility API that provides
access to UI element hierarchies, properties, and patterns for automation.
"""

import asyncio
import logging
import platform
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

# UIA control type to AccessibilityRole mapping
UIA_ROLE_MAP: dict[int, AccessibilityRole] = {
    50000: AccessibilityRole.BUTTON,  # Button
    50001: AccessibilityRole.CALENDAR,  # Calendar
    50002: AccessibilityRole.CHECKBOX,  # CheckBox
    50003: AccessibilityRole.COMBOBOX,  # ComboBox
    50004: AccessibilityRole.EDIT,  # Edit
    50005: AccessibilityRole.HYPERLINK,  # Hyperlink
    50006: AccessibilityRole.IMG,  # Image
    50007: AccessibilityRole.LISTITEM,  # ListItem
    50008: AccessibilityRole.LIST,  # List
    50009: AccessibilityRole.MENU,  # Menu
    50010: AccessibilityRole.MENUBAR,  # MenuBar
    50011: AccessibilityRole.MENUITEM,  # MenuItem
    50012: AccessibilityRole.PROGRESSBAR,  # ProgressBar
    50013: AccessibilityRole.RADIO,  # RadioButton
    50014: AccessibilityRole.SCROLLBAR,  # ScrollBar
    50015: AccessibilityRole.SLIDER,  # Slider
    50016: AccessibilityRole.SPINBUTTON,  # Spinner
    50017: AccessibilityRole.STATUS,  # StatusBar
    50018: AccessibilityRole.TAB,  # Tab
    50019: AccessibilityRole.TABPANEL,  # TabItem
    50020: AccessibilityRole.STATIC_TEXT,  # Text
    50021: AccessibilityRole.TOOLBAR,  # ToolBar
    50022: AccessibilityRole.TOOLTIP,  # ToolTip
    50023: AccessibilityRole.TREE,  # Tree
    50024: AccessibilityRole.TREEITEM,  # TreeItem
    50025: AccessibilityRole.CUSTOM,  # Custom
    50026: AccessibilityRole.GROUP,  # Group
    50027: AccessibilityRole.SCROLLBAR,  # Thumb
    50028: AccessibilityRole.DATAITEM,  # DataItem
    50029: AccessibilityRole.DOCUMENT,  # Document
    50030: AccessibilityRole.SPLITBUTTON,  # SplitButton
    50031: AccessibilityRole.WINDOW,  # Window
    50032: AccessibilityRole.PANE,  # Pane
    50033: AccessibilityRole.HEADING,  # Header
    50034: AccessibilityRole.HEADING,  # HeaderItem
    50035: AccessibilityRole.TABLE,  # Table
    50036: AccessibilityRole.TITLEBAR,  # TitleBar
    50037: AccessibilityRole.SEPARATOR,  # Separator
}


def _is_windows() -> bool:
    """Check if running on Windows."""
    return platform.system() == "Windows"


class UIAAccessibilityCapture(IAccessibilityCapture):
    """Windows UI Automation accessibility capture implementation.

    Uses Windows UI Automation API to capture accessibility trees from
    native Windows applications.

    Note: This implementation only works on Windows. On other platforms,
    connect() will always return False.

    Example:
        >>> capture = UIAAccessibilityCapture()
        >>> await capture.connect(target="Notepad")  # Match by window title
        >>> snapshot = await capture.capture_tree()
        >>> print(capture.to_ai_context(snapshot))
        >>> await capture.click_by_ref("@e3")
        >>> await capture.disconnect()
    """

    def __init__(self, config: AccessibilityConfig | None = None) -> None:
        """Initialize UIA capture.

        Args:
            config: Accessibility configuration. If None, uses defaults.
        """
        self._config = config or AccessibilityConfig()
        self._ref_manager = RefManager()
        self._current_snapshot: AccessibilitySnapshot | None = None
        self._target_element: Any = None  # uiautomation.Control
        self._uia: Any = None  # uiautomation module
        self._is_connected = False

    def _ensure_uia_available(self) -> bool:
        """Ensure the uiautomation module is available.

        Returns:
            True if uiautomation is available
        """
        if self._uia is not None:
            return True

        if not _is_windows():
            logger.warning("UIA accessibility capture is only available on Windows")
            return False

        try:
            import uiautomation as auto

            self._uia = auto
            return True
        except ImportError:
            logger.error(
                "uiautomation module not available. Install with: pip install uiautomation"
            )
            return False

    async def connect(
        self,
        target: str | int | None = None,
        *,
        host: str = "localhost",
        port: int = 9222,
        timeout: float = 30.0,
    ) -> bool:
        """Connect to a Windows application via UIA.

        Args:
            target: Window title (partial match), process ID, or None for desktop
            host: Ignored for UIA (local only)
            port: Ignored for UIA
            timeout: Search timeout for finding the window

        Returns:
            True if connected successfully
        """
        if not self._ensure_uia_available():
            return False

        # Disconnect from any existing target
        if self._is_connected:
            await self.disconnect()

        try:
            auto = self._uia

            if target is None:
                # Use the desktop root
                self._target_element = auto.GetRootControl()
                self._is_connected = True
                logger.info("Connected to UIA: Desktop root")
                return True

            elif isinstance(target, int):
                # Find by process ID
                def find_by_pid() -> Any:
                    root = auto.GetRootControl()
                    for window in root.GetChildren():
                        if hasattr(window, "ProcessId") and window.ProcessId == target:
                            return window
                    return None

                # Run synchronously in executor to avoid blocking
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, find_by_pid),
                    timeout=timeout,
                )

                if result:
                    self._target_element = result
                    self._is_connected = True
                    logger.info(f"Connected to UIA: Process {target}")
                    return True
                else:
                    logger.error(f"Window not found for process ID: {target}")
                    return False

            elif isinstance(target, str):
                # Find by window title (partial match)
                def find_by_title() -> Any:
                    root = auto.GetRootControl()
                    for window in root.GetChildren():
                        if hasattr(window, "Name") and target.lower() in window.Name.lower():
                            return window
                    return None

                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, find_by_title),
                    timeout=timeout,
                )

                if result:
                    self._target_element = result
                    self._is_connected = True
                    logger.info(f"Connected to UIA: {result.Name}")
                    return True
                else:
                    logger.error(f"Window not found matching: {target}")
                    return False

            return False

        except TimeoutError:
            logger.error(f"Timeout finding window: {target}")
            return False
        except Exception as e:
            logger.exception(f"Failed to connect via UIA: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from the current UIA target."""
        self._target_element = None
        self._is_connected = False
        self._current_snapshot = None
        self._ref_manager.clear()
        logger.info("Disconnected from UIA")

    def is_connected(self) -> bool:
        """Check if connected to a UIA target."""
        return self._is_connected and self._target_element is not None

    def _element_to_node(
        self,
        element: Any,
        config: AccessibilityConfig,
        depth: int = 0,
    ) -> AccessibilityNode | None:
        """Convert a UIA element to an AccessibilityNode.

        Args:
            element: UIA Control object
            config: Capture configuration
            depth: Current tree depth

        Returns:
            AccessibilityNode or None if element should be skipped
        """
        if config.max_depth is not None and depth > config.max_depth:
            return None

        try:
            # Get control type and map to role
            control_type = getattr(element, "ControlType", 50025)  # Default to Custom
            role = UIA_ROLE_MAP.get(control_type, AccessibilityRole.GENERIC)

            # Get element properties
            name = getattr(element, "Name", "") or ""
            value = getattr(element, "Value", None)
            automation_id = getattr(element, "AutomationId", None) or None
            class_name = getattr(element, "ClassName", None) or None

            # Get bounding rectangle
            bounds = None
            rect = getattr(element, "BoundingRectangle", None)
            if rect and hasattr(rect, "left"):
                width = rect.right - rect.left
                height = rect.bottom - rect.top
                if width > 0 and height > 0:
                    bounds = AccessibilityBounds(
                        x=int(rect.left),
                        y=int(rect.top),
                        width=int(width),
                        height=int(height),
                    )

            # Skip hidden elements if not configured to include
            if not config.include_hidden:
                if bounds is None or (bounds.width == 0 and bounds.height == 0):
                    # Still process children - element might be a container
                    pass

            # Get state flags
            is_enabled = getattr(element, "IsEnabled", True)
            is_offscreen = getattr(element, "IsOffscreen", False)
            has_keyboard_focus = getattr(element, "HasKeyboardFocus", False)

            # Check for toggle/check state
            is_checked = None
            toggle_state = getattr(element, "ToggleState", None)
            if toggle_state is not None:
                is_checked = toggle_state == 1  # 1 = On

            # Check for expand/collapse state
            is_expanded = None
            expand_state = getattr(element, "ExpandCollapseState", None)
            if expand_state is not None:
                is_expanded = expand_state == 1  # 1 = Expanded

            state = AccessibilityState(
                is_disabled=not is_enabled,
                is_hidden=is_offscreen,
                is_focused=has_keyboard_focus,
                is_checked=is_checked,
                is_expanded=is_expanded,
            )

            # Determine if interactive
            is_interactive = role in {
                AccessibilityRole.BUTTON,
                AccessibilityRole.LINK,
                AccessibilityRole.HYPERLINK,
                AccessibilityRole.CHECKBOX,
                AccessibilityRole.RADIO,
                AccessibilityRole.COMBOBOX,
                AccessibilityRole.EDIT,
                AccessibilityRole.TEXTBOX,
                AccessibilityRole.SLIDER,
                AccessibilityRole.SPINBUTTON,
                AccessibilityRole.TAB,
                AccessibilityRole.MENUITEM,
                AccessibilityRole.TREEITEM,
                AccessibilityRole.LISTITEM,
            }

            # Skip non-interactive if configured
            if config.interactive_only and not is_interactive:
                # Still process children
                children: list[AccessibilityNode] = []
                try:
                    for child in element.GetChildren():
                        child_node = self._element_to_node(child, config, depth + 1)
                        if child_node:
                            children.append(child_node)
                except Exception:
                    pass

                # If we have interactive children, create a container node
                if children:
                    return AccessibilityNode(
                        ref=self._ref_manager.assign_ref(is_interactive=False),
                        role=role,
                        name=name or None,
                        value=value,
                        bounds=bounds,
                        state=state,
                        is_interactive=False,
                        automation_id=automation_id,
                        class_name=class_name,
                        children=children,
                    )
                return None

            # Assign ref
            ref = self._ref_manager.assign_ref(is_interactive=is_interactive)

            # Process children
            children = []
            try:
                for child in element.GetChildren():
                    child_node = self._element_to_node(child, config, depth + 1)
                    if child_node:
                        children.append(child_node)
            except Exception as e:
                logger.debug(f"Error getting children: {e}")

            return AccessibilityNode(
                ref=ref,
                role=role,
                name=name or None,
                value=value,
                bounds=bounds,
                state=state,
                is_interactive=is_interactive,
                automation_id=automation_id,
                class_name=class_name,
                children=children,
            )

        except Exception as e:
            logger.debug(f"Error converting element: {e}")
            return None

    async def capture_tree(
        self,
        options: AccessibilityCaptureOptions | None = None,
    ) -> AccessibilitySnapshot:
        """Capture the accessibility tree from the connected UIA target.

        Args:
            options: Capture options

        Returns:
            AccessibilitySnapshot with the captured tree

        Raises:
            RuntimeError: If not connected
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to a UIA target")

        opts = options or AccessibilityCaptureOptions()
        config = opts.config or self._config

        # Reset ref manager for new capture
        self._ref_manager.clear()

        # Capture in executor to avoid blocking
        loop = asyncio.get_event_loop()

        def do_capture() -> AccessibilityNode | None:
            return self._element_to_node(self._target_element, config, depth=0)

        root = await loop.run_in_executor(None, do_capture)

        if root is None:
            # Create empty root if capture failed
            root = AccessibilityNode(
                ref="@e0",
                role=AccessibilityRole.WINDOW,
                name=getattr(self._target_element, "Name", "Unknown"),
                is_interactive=False,
                children=[],
            )

        # Count nodes
        def count_nodes(node: AccessibilityNode) -> tuple[int, int]:
            total = 1
            interactive = 1 if node.is_interactive else 0
            for child in node.children:
                t, i = count_nodes(child)
                total += t
                interactive += i
            return total, interactive

        total_nodes, interactive_nodes = count_nodes(root)

        # Get window title
        title = getattr(self._target_element, "Name", None)

        snapshot = AccessibilitySnapshot(
            root=root,
            timestamp=time.time(),
            backend=AccessibilityBackend.UIA,
            title=title,
            total_nodes=total_nodes,
            interactive_nodes=interactive_nodes,
        )

        self._current_snapshot = snapshot
        logger.info(f"Captured UIA tree: {total_nodes} nodes, {interactive_nodes} interactive")

        return snapshot

    async def get_node_by_ref(self, ref: str) -> AccessibilityNode | None:
        """Get a node by its ref ID from the current snapshot."""
        if not self._current_snapshot:
            return None

        def find_node(node: AccessibilityNode) -> AccessibilityNode | None:
            if node.ref == ref:
                return node
            for child in node.children:
                found = find_node(child)
                if found:
                    return found
            return None

        return find_node(self._current_snapshot.root)

    async def find_nodes(
        self,
        selector: AccessibilitySelector,
    ) -> list[AccessibilityNode]:
        """Find nodes matching a selector in the current snapshot."""
        if not self._current_snapshot:
            return []

        matches: list[AccessibilityNode] = []

        def check_match(node: AccessibilityNode) -> bool:
            # Check role
            if selector.role is not None:
                if isinstance(selector.role, list):
                    if node.role not in selector.role:
                        return False
                elif node.role != selector.role:
                    return False

            # Check name
            if selector.name is not None:
                case_sensitive = (
                    selector.case_sensitive if selector.case_sensitive is not None else True
                )
                node_name = node.name or ""
                sel_name = selector.name
                if not case_sensitive:
                    node_name = node_name.lower()
                    sel_name = sel_name.lower()
                if node_name != sel_name:
                    return False

            # Check name_contains
            if selector.name_contains is not None:
                case_sensitive = (
                    selector.case_sensitive if selector.case_sensitive is not None else True
                )
                node_name = node.name or ""
                sel_contains = selector.name_contains
                if not case_sensitive:
                    node_name = node_name.lower()
                    sel_contains = sel_contains.lower()
                if sel_contains not in node_name:
                    return False

            # Check is_interactive
            if selector.is_interactive is not None:
                if node.is_interactive != selector.is_interactive:
                    return False

            # Check automation_id
            if selector.automation_id is not None:
                if node.automation_id != selector.automation_id:
                    return False

            # Check class_name
            if selector.class_name is not None:
                if node.class_name != selector.class_name:
                    return False

            return True

        def search(node: AccessibilityNode, depth: int = 0) -> None:
            if selector.max_depth is not None and depth > selector.max_depth:
                return

            if check_match(node):
                matches.append(node)

            for child in node.children:
                search(child, depth + 1)

        search(self._current_snapshot.root)
        return matches

    async def click_by_ref(self, ref: str) -> bool:
        """Click an element by ref using its center coordinates."""
        node = await self.get_node_by_ref(ref)
        if not node:
            logger.warning(f"Element not found: {ref}")
            return False

        if not node.bounds:
            logger.warning(f"Element has no bounds: {ref}")
            return False

        # Calculate center
        center_x = node.bounds.x + node.bounds.width // 2
        center_y = node.bounds.y + node.bounds.height // 2

        # Click using platform API
        try:
            if self._uia:
                self._uia.Click(center_x, center_y)
                logger.info(f"Clicked {ref} at ({center_x}, {center_y})")
                return True
        except Exception as e:
            logger.error(f"Click failed: {e}")

        return False

    async def type_by_ref(
        self,
        ref: str,
        text: str,
        *,
        clear_first: bool = False,
    ) -> bool:
        """Type text into an element by ref."""
        node = await self.get_node_by_ref(ref)
        if not node:
            logger.warning(f"Element not found: {ref}")
            return False

        # Focus the element first
        await self.focus_by_ref(ref)
        await asyncio.sleep(0.1)  # Small delay for focus

        try:
            if self._uia:
                # Clear if requested
                if clear_first:
                    self._uia.SendKeys("{Ctrl}a", waitTime=0)
                    await asyncio.sleep(0.05)

                # Type the text
                # Use SendKeys for special characters, otherwise direct text
                self._uia.SendKeys(text, interval=0.02, waitTime=0)
                logger.info(f"Typed into {ref}: {text[:20]}...")
                return True
        except Exception as e:
            logger.error(f"Type failed: {e}")

        return False

    async def focus_by_ref(self, ref: str) -> bool:
        """Focus an element by ref."""
        node = await self.get_node_by_ref(ref)
        if not node:
            logger.warning(f"Element not found: {ref}")
            return False

        if not node.bounds:
            logger.warning(f"Element has no bounds for focus: {ref}")
            return False

        # Click to focus
        center_x = node.bounds.x + node.bounds.width // 2
        center_y = node.bounds.y + node.bounds.height // 2

        try:
            if self._uia:
                self._uia.Click(center_x, center_y)
                await asyncio.sleep(0.1)
                logger.info(f"Focused {ref}")
                return True
        except Exception as e:
            logger.error(f"Focus failed: {e}")

        return False

    def get_backend_name(self) -> str:
        """Get the backend name."""
        return "uia"
