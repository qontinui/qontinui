"""Linux AT-SPI2 accessibility capture implementation.

This module provides accessibility tree capture via AT-SPI2 (Assistive
Technology Service Provider Interface), supporting native Linux applications:
- GTK applications
- Qt applications
- LibreOffice
- Firefox (native a11y)
- Most desktop software on Linux

AT-SPI2 is the platform accessibility API on Linux that provides access to
UI element hierarchies, properties, and actions for automation.
"""

import asyncio
import logging
import platform
import time
from pathlib import Path
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

# AT-SPI role name to AccessibilityRole mapping.
# AT-SPI roles are exposed as role name strings via pyatspi2 or as
# Atspi.Role enum members via GObject introspection.  We normalise
# them to lower-case strings for matching.
ATSPI_ROLE_MAP: dict[str, AccessibilityRole] = {
    "push button": AccessibilityRole.BUTTON,
    "text": AccessibilityRole.TEXTBOX,
    "combo box": AccessibilityRole.COMBOBOX,
    "check box": AccessibilityRole.CHECKBOX,
    "radio button": AccessibilityRole.RADIO,
    "menu": AccessibilityRole.MENU,
    "menu bar": AccessibilityRole.MENUBAR,
    "menu item": AccessibilityRole.MENUITEM,
    "link": AccessibilityRole.LINK,
    "list": AccessibilityRole.LIST,
    "list item": AccessibilityRole.LISTITEM,
    "tree": AccessibilityRole.TREE,
    "tree item": AccessibilityRole.TREEITEM,
    "tab": AccessibilityRole.TAB,
    "slider": AccessibilityRole.SLIDER,
    "scroll bar": AccessibilityRole.SCROLLBAR,
    "dialog": AccessibilityRole.DIALOG,
    "toolbar": AccessibilityRole.TOOLBAR,
    "table": AccessibilityRole.TABLE,
    "heading": AccessibilityRole.HEADING,
    "image": AccessibilityRole.IMG,
    "progress bar": AccessibilityRole.PROGRESSBAR,
    "spin button": AccessibilityRole.SPINBUTTON,
    "status bar": AccessibilityRole.STATUS,
    "alert": AccessibilityRole.ALERT,
    "frame": AccessibilityRole.WINDOW,
    "panel": AccessibilityRole.GROUP,
    "label": AccessibilityRole.STATIC_TEXT,
    "entry": AccessibilityRole.EDIT,
    "password text": AccessibilityRole.TEXTBOX,
    "toggle button": AccessibilityRole.SWITCH,
    "separator": AccessibilityRole.SEPARATOR,
    "filler": AccessibilityRole.GENERIC,
    "page tab": AccessibilityRole.TAB,
    "page tab list": AccessibilityRole.TABLIST,
    "check menu item": AccessibilityRole.MENUITEMCHECKBOX,
    "radio menu item": AccessibilityRole.MENUITEMRADIO,
    "split pane": AccessibilityRole.PANE,
    "window": AccessibilityRole.WINDOW,
    "application": AccessibilityRole.APPLICATION,
    "document frame": AccessibilityRole.DOCUMENT,
    "scroll pane": AccessibilityRole.PANE,
    "option pane": AccessibilityRole.PANE,
    "glass pane": AccessibilityRole.PANE,
    "layered pane": AccessibilityRole.PANE,
    "tree table": AccessibilityRole.TREEGRID,
    "table cell": AccessibilityRole.CELL,
    "table column header": AccessibilityRole.COLUMNHEADER,
    "table row header": AccessibilityRole.ROWHEADER,
    "paragraph": AccessibilityRole.PARAGRAPH,
    "form": AccessibilityRole.FORM,
    "section": AccessibilityRole.REGION,
    "redundant object": AccessibilityRole.GENERIC,
    "unknown": AccessibilityRole.UNKNOWN,
    "icon": AccessibilityRole.IMG,
    "embedded": AccessibilityRole.GENERIC,
    "desktop icon": AccessibilityRole.IMG,
    "desktop frame": AccessibilityRole.WINDOW,
    "internal frame": AccessibilityRole.WINDOW,
    "file chooser": AccessibilityRole.DIALOG,
    "notification": AccessibilityRole.ALERT,
    "date editor": AccessibilityRole.DATEPICKER,
    "animation": AccessibilityRole.IMG,
    "canvas": AccessibilityRole.IMG,
    "color chooser": AccessibilityRole.DIALOG,
    "tooltip": AccessibilityRole.TOOLTIP,
}


def _is_linux() -> bool:
    """Check if running on Linux."""
    return platform.system() == "Linux"


class ATSPIAccessibilityCapture(IAccessibilityCapture):
    """Linux AT-SPI2 accessibility capture implementation.

    Uses AT-SPI2 (via pyatspi2 or gi.repository.Atspi) to capture
    accessibility trees from native Linux applications.

    Note: This implementation only works on Linux. On other platforms,
    connect() will always return False.

    Example:
        >>> capture = ATSPIAccessibilityCapture()
        >>> await capture.connect(target="gedit")  # Match by window title
        >>> snapshot = await capture.capture_tree()
        >>> print(capture.to_ai_context(snapshot))
        >>> await capture.click_by_ref("@e3")
        >>> await capture.disconnect()
    """

    def __init__(self, config: AccessibilityConfig | None = None) -> None:
        """Initialize AT-SPI capture.

        Args:
            config: Accessibility configuration. If None, uses defaults.
        """
        self._config = config or AccessibilityConfig()
        self._ref_manager = RefManager()
        self._current_snapshot: AccessibilitySnapshot | None = None
        self._target_element: Any = None  # AT-SPI Accessible object
        self._atspi: Any = None  # pyatspi2 or gi.repository.Atspi module
        self._is_connected = False
        self._persistence_dir = Path.home() / ".qontinui" / "atspi_refs"
        self._use_gi = False  # True if using gi.repository.Atspi
        # Map ref → live AT-SPI element for click/type/focus operations
        self._atspi_elements_by_ref: dict[str, Any] = {}

    def _ensure_atspi_available(self) -> bool:
        """Ensure the AT-SPI module is available.

        Tries pyatspi2 first, then falls back to gi.repository.Atspi.

        Returns:
            True if an AT-SPI binding is available
        """
        if self._atspi is not None:
            return True

        if not _is_linux():
            logger.warning("AT-SPI accessibility capture is only available on Linux")
            return False

        # Try pyatspi2 first (higher-level, more Pythonic API)
        try:
            import pyatspi  # type: ignore[import-untyped]

            self._atspi = pyatspi
            self._use_gi = False
            return True
        except ImportError:
            pass

        # Fall back to gi.repository.Atspi (GObject introspection bindings)
        try:
            import gi  # type: ignore[import-untyped]

            gi.require_version("Atspi", "2.0")
            from gi.repository import Atspi  # type: ignore[import-untyped]

            self._atspi = Atspi
            self._use_gi = True
            return True
        except (ImportError, ValueError):
            pass

        logger.error(
            "AT-SPI module not available. Install pyatspi2 or python3-gi with "
            "gir1.2-atspi-2.0: pip install pyatspi or apt install python3-gi gir1.2-atspi-2.0"
        )
        return False

    def _get_desktop(self) -> Any:
        """Get the AT-SPI desktop object.

        Returns:
            The desktop accessible object
        """
        if self._use_gi:
            return self._atspi.get_desktop(0)
        else:
            return self._atspi.Registry.getDesktop(0)

    def _get_children(self, element: Any) -> list[Any]:
        """Get children of an AT-SPI accessible element.

        Args:
            element: AT-SPI Accessible object

        Returns:
            List of child Accessible objects
        """
        children = []
        try:
            if self._use_gi:
                count = element.get_child_count()
                for i in range(count):
                    child = element.get_child_at_index(i)
                    if child is not None:
                        children.append(child)
            else:
                count = element.childCount
                for i in range(count):
                    child = element.getChildAtIndex(i)
                    if child is not None:
                        children.append(child)
        except Exception as e:
            logger.debug(f"Error getting children: {e}")
        return children

    def _get_role_name(self, element: Any) -> str:
        """Get the role name of an AT-SPI element as a lower-case string.

        Args:
            element: AT-SPI Accessible object

        Returns:
            Role name string (lower-case)
        """
        try:
            if self._use_gi:
                return str(element.get_role_name().lower())
            else:
                return str(element.getRoleName().lower())
        except Exception:
            return "unknown"

    def _get_name(self, element: Any) -> str:
        """Get the name of an AT-SPI element.

        Args:
            element: AT-SPI Accessible object

        Returns:
            Element name or empty string
        """
        try:
            if self._use_gi:
                return element.get_name() or ""
            else:
                return element.name or ""
        except Exception:
            return ""

    def _get_description(self, element: Any) -> str | None:
        """Get the description of an AT-SPI element.

        Args:
            element: AT-SPI Accessible object

        Returns:
            Description string or None
        """
        try:
            if self._use_gi:
                desc = element.get_description()
            else:
                desc = element.description
            return desc if desc else None
        except Exception:
            return None

    def _get_pid(self, element: Any) -> int:
        """Get the process ID of an AT-SPI element.

        Args:
            element: AT-SPI Accessible object

        Returns:
            Process ID or -1 on error
        """
        try:
            if self._use_gi:
                return int(element.get_process_id())
            else:
                pid = getattr(element, "pid", None)
                if pid is not None:
                    return int(pid)
                app = element.getApplication()
                if app is not None:
                    return int(getattr(app, "pid", -1))
                return -1
        except Exception:
            return -1

    def _get_extents(self, element: Any) -> tuple[int, int, int, int] | None:
        """Get the bounding rectangle of an AT-SPI element.

        Args:
            element: AT-SPI Accessible object

        Returns:
            Tuple (x, y, width, height) or None
        """
        try:
            if self._use_gi:
                component = element.get_component_iface()
                if component is None:
                    return None
                rect = component.get_extents(self._atspi.CoordType.SCREEN)
                return (rect.x, rect.y, rect.width, rect.height)
            else:
                try:
                    component = element.queryComponent()
                except Exception:
                    return None
                rect = component.getExtents(self._atspi.DESKTOP_COORDS)
                return (rect.x, rect.y, rect.width, rect.height)
        except Exception:
            return None

    def _get_states(self, element: Any) -> set[str]:
        """Get the state names of an AT-SPI element.

        Args:
            element: AT-SPI Accessible object

        Returns:
            Set of state name strings (lower-case)
        """
        states: set[str] = set()
        try:
            if self._use_gi:
                state_set = element.get_state_set()
                # GI Atspi exposes states as Atspi.StateType enum values
                for st in dir(self._atspi.StateType):
                    if st.startswith("_"):
                        continue
                    try:
                        val = getattr(self._atspi.StateType, st)
                        if state_set.contains(val):
                            states.add(st.lower())
                    except Exception:
                        continue
            else:
                state_set = element.getState()
                # pyatspi2 exposes getStates() returning a list
                try:
                    raw_states = state_set.getStates()
                    for st in raw_states:
                        name = self._atspi.stateToString(st)
                        if name:
                            states.add(name.lower())
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Error getting states: {e}")
        return states

    def _get_value(self, element: Any) -> str | None:
        """Get the value of an AT-SPI element (for text/value interfaces).

        Args:
            element: AT-SPI Accessible object

        Returns:
            Value string or None
        """
        # Try text interface first
        try:
            if self._use_gi:
                text_iface = element.get_text_iface()
                if text_iface is not None:
                    count = text_iface.get_character_count()
                    if count > 0:
                        result: str | None = text_iface.get_text(0, count)
                        return result
            else:
                try:
                    text_iface = element.queryText()
                    count = text_iface.characterCount
                    if count > 0:
                        result2: str | None = text_iface.getText(0, count)
                        return result2
                except Exception:
                    pass
        except Exception:
            pass

        # Try value interface
        try:
            if self._use_gi:
                value_iface = element.get_value_iface()
                if value_iface is not None:
                    return str(value_iface.get_current_value())
            else:
                try:
                    value_iface = element.queryValue()
                    return str(value_iface.currentValue)
                except Exception:
                    pass
        except Exception:
            pass

        return None

    async def connect(
        self,
        target: str | int | None = None,
        *,
        host: str = "localhost",
        port: int = 9222,
        timeout: float = 30.0,
    ) -> bool:
        """Connect to a Linux application via AT-SPI.

        Args:
            target: Window title (partial match), process ID, or None for desktop
            host: Ignored for AT-SPI (local only)
            port: Ignored for AT-SPI
            timeout: Search timeout for finding the window

        Returns:
            True if connected successfully
        """
        if not self._ensure_atspi_available():
            return False

        # Disconnect from any existing target
        if self._is_connected:
            await self.disconnect()

        try:
            desktop = self._get_desktop()

            if target is None:
                # Use the desktop root
                self._target_element = desktop
                self._is_connected = True
                logger.info("Connected to AT-SPI: Desktop root")
                return True

            elif isinstance(target, int):
                # Find by process ID
                def find_by_pid() -> Any:
                    for i in range(self._get_children(desktop).__len__()):
                        app = self._get_children(desktop)[i]
                        if self._get_pid(app) == target:
                            # Return the first window of the application
                            app_children = self._get_children(app)
                            if app_children:
                                return app_children[0]
                            return app
                    return None

                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, find_by_pid),
                    timeout=timeout,
                )

                if result:
                    self._target_element = result
                    self._is_connected = True
                    logger.info(f"Connected to AT-SPI: Process {target}")
                    return True
                else:
                    logger.error(f"Application not found for process ID: {target}")
                    return False

            elif isinstance(target, str):
                # Find by window title (partial match)
                def find_by_title() -> Any:
                    for app in self._get_children(desktop):
                        for window in self._get_children(app):
                            window_name = self._get_name(window)
                            if target.lower() in window_name.lower():
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
                    window_name = self._get_name(result)
                    logger.info(f"Connected to AT-SPI: {window_name}")
                    return True
                else:
                    logger.error(f"Window not found matching: {target}")
                    return False

            return False

        except TimeoutError:
            logger.error(f"Timeout finding window: {target}")
            return False
        except Exception as e:
            logger.exception(f"Failed to connect via AT-SPI: {e}")
            return False

    def _ref_persistence_path(self) -> Path | None:
        """Get the persistence file path for the current target's refs."""
        if self._target_element is None:
            return None
        title = self._get_name(self._target_element)
        if not title:
            return None
        # Sanitise title into a safe filename
        safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in title).strip()[:80]
        return self._persistence_dir / f"{safe}.json"

    async def disconnect(self) -> None:
        """Disconnect from the current AT-SPI target.

        Saves ref fingerprints to disk before clearing state so they can
        be re-resolved on the next connect.
        """
        # Persist refs before clearing
        if self._ref_manager.count > 0:
            path = self._ref_persistence_path()
            if path is not None:
                try:
                    self._ref_manager.save(path)
                except Exception as e:
                    logger.debug("Failed to persist AT-SPI refs: %s", e)

        self._target_element = None
        self._is_connected = False
        self._current_snapshot = None
        self._ref_manager.clear()
        logger.info("Disconnected from AT-SPI")

    def is_connected(self) -> bool:
        """Check if connected to an AT-SPI target."""
        return self._is_connected and self._target_element is not None

    def _element_to_node(
        self,
        element: Any,
        config: AccessibilityConfig,
        depth: int = 0,
    ) -> AccessibilityNode | None:
        """Convert an AT-SPI element to an AccessibilityNode.

        Args:
            element: AT-SPI Accessible object
            config: Capture configuration
            depth: Current tree depth

        Returns:
            AccessibilityNode or None if element should be skipped
        """
        if config.max_depth is not None and depth > config.max_depth:
            return None

        try:
            # Get role and map to AccessibilityRole
            role_name = self._get_role_name(element)
            role = ATSPI_ROLE_MAP.get(role_name, AccessibilityRole.GENERIC)

            # Get element properties
            name = self._get_name(element) or ""
            value = self._get_value(element)
            description = self._get_description(element)

            # AT-SPI does not have an automation_id concept; use the
            # accessible's unique path/id when available.
            automation_id: str | None = None
            try:
                if self._use_gi:
                    # GI Atspi exposes get_id() on some builds
                    aid = getattr(element, "get_id", None)
                    if callable(aid):
                        automation_id = str(aid()) or None
                else:
                    aid = getattr(element, "id", None)
                    if aid is not None:
                        automation_id = str(aid) or None
            except Exception:
                pass

            # Get bounding rectangle
            bounds = None
            extents = self._get_extents(element)
            if extents is not None:
                x, y, width, height = extents
                if width > 0 and height > 0:
                    bounds = AccessibilityBounds(
                        x=int(x),
                        y=int(y),
                        width=int(width),
                        height=int(height),
                    )

            # Skip hidden elements if not configured to include
            if not config.include_hidden:
                if bounds is None or (bounds.width == 0 and bounds.height == 0):
                    # Still process children - element might be a container
                    pass

            # Get state flags
            states = self._get_states(element)

            is_enabled = "enabled" in states or "sensitive" in states
            is_showing = "showing" in states
            is_visible = "visible" in states
            has_focus = "focused" in states
            is_checked_state = "checked" in states
            is_expanded_state = "expanded" in states
            is_selected = "selected" in states

            # Determine hidden: not showing or not visible
            is_hidden = not is_showing or not is_visible

            # Determine checked state
            is_checked: bool | None = None
            if "checked" in states or role in {
                AccessibilityRole.CHECKBOX,
                AccessibilityRole.RADIO,
                AccessibilityRole.SWITCH,
            }:
                is_checked = is_checked_state

            # Determine expanded state
            is_expanded: bool | None = None
            if "expandable" in states or "expanded" in states:
                is_expanded = is_expanded_state

            state = AccessibilityState(
                is_disabled=not is_enabled,
                is_hidden=is_hidden,
                is_focused=has_focus,
                is_checked=is_checked,
                is_expanded=is_expanded,
                is_selected=is_selected,
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
                AccessibilityRole.MENUITEMCHECKBOX,
                AccessibilityRole.MENUITEMRADIO,
                AccessibilityRole.TREEITEM,
                AccessibilityRole.LISTITEM,
                AccessibilityRole.SWITCH,
            }

            # Skip non-interactive if configured
            if config.interactive_only and not is_interactive:
                # Still process children
                children: list[AccessibilityNode] = []
                for child in self._get_children(element):
                    child_node = self._element_to_node(child, config, depth + 1)
                    if child_node:
                        children.append(child_node)

                # If we have interactive children, create a container node
                if children:
                    container_ref = self._ref_manager.assign_ref(is_interactive=False)
                    self._atspi_elements_by_ref[container_ref] = element
                    container_node = AccessibilityNode(
                        ref=container_ref,
                        role=role,
                        name=name or None,
                        value=value,
                        description=description,
                        bounds=bounds,
                        state=state,
                        is_interactive=False,
                        automation_id=automation_id,
                        children=children,
                    )
                    self._ref_manager.register_node(container_ref, container_node)
                    return container_node
                return None

            # Assign ref and store AT-SPI element mapping
            ref = self._ref_manager.assign_ref(is_interactive=is_interactive)
            self._atspi_elements_by_ref[ref] = element

            # Process children
            children = []
            for child in self._get_children(element):
                child_node = self._element_to_node(child, config, depth + 1)
                if child_node:
                    children.append(child_node)

            node = AccessibilityNode(
                ref=ref,
                role=role,
                name=name or None,
                value=value,
                description=description,
                bounds=bounds,
                state=state,
                is_interactive=is_interactive,
                automation_id=automation_id,
                children=children,
            )
            # Register in ref manager so save() can persist fingerprints
            self._ref_manager.register_node(ref, node)
            return node

        except Exception as e:
            logger.debug(f"Error converting element: {e}")
            return None

    async def capture_tree(
        self,
        options: AccessibilityCaptureOptions | None = None,
    ) -> AccessibilitySnapshot:
        """Capture the accessibility tree from the connected AT-SPI target.

        Args:
            options: Capture options

        Returns:
            AccessibilitySnapshot with the captured tree

        Raises:
            RuntimeError: If not connected
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to an AT-SPI target")

        opts = options or AccessibilityCaptureOptions()
        config = opts.config or self._config

        # Reset ref manager and element map for new capture
        self._ref_manager.clear()
        self._atspi_elements_by_ref.clear()

        # Capture in executor to avoid blocking
        loop = asyncio.get_event_loop()

        def do_capture() -> AccessibilityNode | None:
            return self._element_to_node(self._target_element, config, depth=0)

        root = await loop.run_in_executor(None, do_capture)

        if root is None:
            # Create empty root if capture failed
            target_name = (
                self._get_name(self._target_element) if self._target_element else "Unknown"
            )
            root = AccessibilityNode(
                ref="@e0",
                role=AccessibilityRole.WINDOW,
                name=target_name,
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
        title = self._get_name(self._target_element) if self._target_element else None

        snapshot = AccessibilitySnapshot(
            root=root,
            timestamp=time.time(),
            backend=AccessibilityBackend.ATSPI,
            title=title,
            total_nodes=total_nodes,
            interactive_nodes=interactive_nodes,
        )

        self._current_snapshot = snapshot

        # Attempt to re-resolve persisted ref fingerprints against the
        # new tree so that refs from previous sessions stay valid.
        self._try_restore_persisted_refs(snapshot)

        logger.info(f"Captured AT-SPI tree: {total_nodes} nodes, {interactive_nodes} interactive")

        return snapshot

    def _try_restore_persisted_refs(self, snapshot: AccessibilitySnapshot) -> None:
        """Try to re-resolve persisted ref fingerprints against a new tree.

        Loads fingerprints saved during a prior ``disconnect()`` and matches
        them against the current tree by automation_id, then role+name.
        Successfully re-resolved refs are registered in the ref manager so
        callers can continue using the same ref IDs across sessions.
        """
        path = self._ref_persistence_path()
        if path is None:
            return

        fingerprints = self._ref_manager.load(path)
        if not fingerprints:
            return

        # Flatten interactive nodes for matching
        def flatten(node: AccessibilityNode) -> list[AccessibilityNode]:
            result: list[AccessibilityNode] = []
            if node.is_interactive:
                result.append(node)
            for child in node.children:
                result.extend(flatten(child))
            return result

        nodes = flatten(snapshot.root)
        restored = 0

        for ref, fp in fingerprints.items():
            auto_id = fp.get("automation_id")
            role = fp.get("role")
            name = fp.get("name")

            # Strategy 1: automation_id (most stable)
            matched = False
            if auto_id:
                for node in nodes:
                    if node.automation_id == auto_id:
                        self._ref_manager.register_node(ref, node)
                        restored += 1
                        matched = True
                        break

            # Strategy 2: role + name (fallback)
            if not matched and role and name:
                for node in nodes:
                    node_role = getattr(node.role, "value", str(node.role))
                    if node_role == role and node.name == name:
                        self._ref_manager.register_node(ref, node)
                        restored += 1
                        break

        if restored:
            logger.info("Restored %d/%d persisted refs", restored, len(fingerprints))

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

    def _find_atspi_element_by_ref(self, ref: str) -> Any | None:
        """Find the original AT-SPI element corresponding to a node ref.

        Uses the ref→element mapping built during ``_element_to_node``
        to return the live AT-SPI Accessible object for click/type/focus.

        Args:
            ref: Reference ID (e.g., "@e3")

        Returns:
            The AT-SPI Accessible object or None
        """
        return self._atspi_elements_by_ref.get(ref)

    async def click_by_ref(self, ref: str) -> bool:
        """Click an element by ref using AT-SPI action interface.

        Falls back to coordinate-based click via xdotool if the action
        interface is not available.

        Args:
            ref: Reference ID of element to click

        Returns:
            True if click succeeded
        """
        node = await self.get_node_by_ref(ref)
        if not node:
            logger.warning(f"Element not found: {ref}")
            return False

        # Try AT-SPI action interface first
        loop = asyncio.get_event_loop()

        def do_click() -> bool:
            element = self._find_atspi_element_by_ref(ref)
            if element is None:
                return False

            # Try the Action interface
            try:
                if self._use_gi:
                    action = element.get_action_iface()
                    if action is not None:
                        n_actions = action.get_n_actions()
                        for i in range(n_actions):
                            action_name = action.get_action_name(i)
                            if action_name and action_name.lower() in (
                                "click",
                                "activate",
                                "press",
                            ):
                                action.do_action(i)
                                return True
                        # If no matching action name, try the first action
                        if n_actions > 0:
                            action.do_action(0)
                            return True
                else:
                    try:
                        action = element.queryAction()
                        n_actions = action.nActions
                        for i in range(n_actions):
                            action_name = action.getName(i)
                            if action_name and action_name.lower() in (
                                "click",
                                "activate",
                                "press",
                            ):
                                action.doAction(i)
                                return True
                        if n_actions > 0:
                            action.doAction(0)
                            return True
                    except Exception:
                        pass
            except Exception:
                pass

            return False

        result = await loop.run_in_executor(None, do_click)
        if result:
            logger.info(f"Clicked {ref} via AT-SPI action")
            return True

        # Fall back to coordinate-based click
        if not node.bounds:
            logger.warning(f"Element has no bounds: {ref}")
            return False

        center_x = node.bounds.x + node.bounds.width // 2
        center_y = node.bounds.y + node.bounds.height // 2

        try:
            import subprocess

            await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ["xdotool", "mousemove", str(center_x), str(center_y), "click", "1"],
                    check=True,
                    capture_output=True,
                ),
            )
            logger.info(f"Clicked {ref} at ({center_x}, {center_y}) via xdotool")
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
        """Type text into an element by ref.

        Uses AT-SPI's editable text interface, falling back to xdotool.

        Args:
            ref: Reference ID of element to type into
            text: Text to type
            clear_first: If True, clear existing content before typing

        Returns:
            True if typing succeeded
        """
        node = await self.get_node_by_ref(ref)
        if not node:
            logger.warning(f"Element not found: {ref}")
            return False

        # Focus the element first
        await self.focus_by_ref(ref)
        await asyncio.sleep(0.1)  # Small delay for focus

        loop = asyncio.get_event_loop()

        def do_type() -> bool:
            element = self._find_atspi_element_by_ref(ref)
            if element is None:
                return False

            # Try AT-SPI editable text interface
            try:
                if self._use_gi:
                    editable = element.get_editable_text_iface()
                    if editable is not None:
                        if clear_first:
                            text_iface = element.get_text_iface()
                            if text_iface is not None:
                                count = text_iface.get_character_count()
                                if count > 0:
                                    editable.delete_text(0, count)
                        editable.insert_text(0 if clear_first else -1, text, len(text))
                        return True
                else:
                    try:
                        editable = element.queryEditableText()
                        if clear_first:
                            try:
                                text_iface = element.queryText()
                                count = text_iface.characterCount
                                if count > 0:
                                    editable.deleteText(0, count)
                            except Exception:
                                pass
                        # Insert at cursor position (-1) or at start (0)
                        editable.insertText(0 if clear_first else -1, text, len(text))
                        return True
                    except Exception:
                        pass
            except Exception:
                pass

            return False

        result = await loop.run_in_executor(None, do_type)
        if result:
            logger.info(f"Typed into {ref}: {text[:20]}...")
            return True

        # Fall back to xdotool
        try:
            import subprocess

            if clear_first:
                await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ["xdotool", "key", "ctrl+a"],
                        check=True,
                        capture_output=True,
                    ),
                )
                await asyncio.sleep(0.05)

            await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ["xdotool", "type", "--clearmodifiers", text],
                    check=True,
                    capture_output=True,
                ),
            )
            logger.info(f"Typed into {ref} via xdotool: {text[:20]}...")
            return True
        except Exception as e:
            logger.error(f"Type failed: {e}")

        return False

    async def focus_by_ref(self, ref: str) -> bool:
        """Focus an element by ref.

        Uses AT-SPI's component interface grabFocus, falling back to
        coordinate-based click.

        Args:
            ref: Reference ID of element to focus

        Returns:
            True if focus succeeded
        """
        node = await self.get_node_by_ref(ref)
        if not node:
            logger.warning(f"Element not found: {ref}")
            return False

        loop = asyncio.get_event_loop()

        def do_focus() -> bool:
            element = self._find_atspi_element_by_ref(ref)
            if element is None:
                return False

            try:
                if self._use_gi:
                    component = element.get_component_iface()
                    if component is not None:
                        component.grab_focus()
                        return True
                else:
                    try:
                        component = element.queryComponent()
                        component.grabFocus()
                        return True
                    except Exception:
                        pass
            except Exception:
                pass

            return False

        result = await loop.run_in_executor(None, do_focus)
        if result:
            logger.info(f"Focused {ref}")
            return True

        # Fall back to coordinate-based click
        if not node.bounds:
            logger.warning(f"Element has no bounds for focus: {ref}")
            return False

        center_x = node.bounds.x + node.bounds.width // 2
        center_y = node.bounds.y + node.bounds.height // 2

        try:
            import subprocess

            await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ["xdotool", "mousemove", str(center_x), str(center_y), "click", "1"],
                    check=True,
                    capture_output=True,
                ),
            )
            await asyncio.sleep(0.1)
            logger.info(f"Focused {ref} via click")
            return True
        except Exception as e:
            logger.error(f"Focus failed: {e}")

        return False

    def get_backend_name(self) -> str:
        """Get the backend name."""
        return "atspi"
