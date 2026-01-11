"""
Accessibility-based extraction using OS Accessibility APIs.

Extracts UI elements and states from native applications using:
- Windows: UI Automation (UIA)
- macOS: Accessibility API (AX)
- Linux: AT-SPI
"""

import logging
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..abstract_extractor import (
    AbstractExtractor,
    ExtractedElement,
    ExtractedState,
    ExtractionContext,
    ExtractionResult,
)

if TYPE_CHECKING:
    from ..extractor_config import AccessibilityConfig, ExtractorConfig

logger = logging.getLogger(__name__)


class AccessibilityExtractor(AbstractExtractor):
    """
    Accessibility-based extraction using OS Accessibility APIs.

    Uses platform-specific accessibility APIs to extract UI structure:
    - Windows: UI Automation (UIA) via comtypes/pywinauto
    - macOS: Accessibility API (AX) via pyobjc
    - Linux: AT-SPI via pyatspi2

    Advantages over other extraction methods:
    - Rich semantic information (roles, labels, states)
    - Reliable element hierarchy
    - Works with native applications
    - Provides accessible names and descriptions

    Example:
        >>> extractor = AccessibilityExtractor()
        >>> context = ExtractionContext(
        ...     app_name="Notepad",
        ...     platform="win32"
        ... )
        >>> config = ExtractorConfig()
        >>> result = await extractor.extract(context, config)
    """

    def __init__(self) -> None:
        """Initialize the accessibility extractor."""
        self._platform = sys.platform
        self._api: PlatformAccessibilityAPI | None = None  # Lazy loaded platform API
        self._element_counter = 0

    async def extract(
        self,
        context: ExtractionContext,
        config: "ExtractorConfig",
    ) -> ExtractionResult:
        """
        Perform accessibility-based extraction.

        Args:
            context: Extraction context with app information
            config: Extractor configuration

        Returns:
            ExtractionResult with extracted elements and states
        """
        await self.validate_context(context)

        extraction_id = str(uuid.uuid4())
        result = ExtractionResult(
            extraction_id=extraction_id,
            extraction_method="accessibility",
            context=context,
        )

        try:
            # Initialize platform API
            self._init_platform_api(config.accessibility)

            # Find target window/application
            root_element = await self._find_target(context, config.accessibility)
            if root_element is None:
                result.add_error(
                    f"Could not find target application: "
                    f"{context.app_name or context.window_title}"
                )
                result.complete()
                return result

            # Extract elements from accessibility tree
            elements = await self.extract_elements(context, config)
            result.elements = self.filter_elements(elements, config)

            # Detect states from hierarchy
            if config.detect_states:
                states = await self.extract_states(context, config, result.elements)
                result.states = states

            # Capture screenshot if requested
            if config.capture_screenshots:
                try:
                    screenshot_path = await self.capture_screenshot(context)
                    result.screenshots = [str(screenshot_path)]
                    result.screenshots_dir = screenshot_path.parent
                except Exception as e:
                    result.add_warning(f"Screenshot capture failed: {e}")

            result.complete()
            logger.info(
                f"Accessibility extraction complete: {len(result.elements)} elements, "
                f"{len(result.states)} states"
            )

        except Exception as e:
            logger.error(f"Accessibility extraction failed: {e}", exc_info=True)
            result.add_error(str(e))
            result.complete()

        return result

    async def extract_elements(
        self,
        context: ExtractionContext,
        config: "ExtractorConfig",
    ) -> list[ExtractedElement]:
        """
        Extract elements from accessibility tree.

        Traverses the accessibility tree and converts nodes to
        ExtractedElement format.
        """
        # Initialize API if needed
        self._init_platform_api(config.accessibility)

        # Find target
        root = await self._find_target(context, config.accessibility)
        if root is None:
            return []

        # Traverse tree
        elements = await self._traverse_tree(root, config.accessibility, depth=0)

        return elements

    async def extract_states(
        self,
        context: ExtractionContext,
        config: "ExtractorConfig",
        elements: list[ExtractedElement] | None = None,
    ) -> list[ExtractedState]:
        """
        Detect UI states from accessibility hierarchy.

        Uses accessibility roles and hierarchy to identify logical
        UI groupings (windows, dialogs, panels, etc.).
        """
        if elements is None:
            elements = await self.extract_elements(context, config)

        states: list[ExtractedState] = []

        # Find top-level containers
        container_roles = {
            "window",
            "dialog",
            "pane",
            "menu",
            "toolbar",
            "tab_panel",
            "group",
        }

        for element in elements:
            if element.aria_role in container_roles:
                # Create state from container
                contained_ids = self._find_contained_elements(element, elements)

                state = ExtractedState(
                    id=f"a11y_state_{element.id}",
                    name=element.name or element.aria_label or element.aria_role or "Unknown",
                    state_type=self._role_to_state_type(element.aria_role),
                    bbox=element.bbox,
                    element_ids=contained_ids,
                    confidence=1.0,
                    extraction_method="accessibility",
                    detection_method="hierarchy",
                )
                states.append(state)

        # If no container states found, create a single root state
        if not states and elements:
            state = ExtractedState(
                id="a11y_state_root",
                name=context.app_name or "Application",
                state_type="window",
                element_ids=[e.id for e in elements],
                confidence=1.0,
                extraction_method="accessibility",
                detection_method="root",
            )
            states.append(state)

        return states

    async def capture_screenshot(
        self,
        context: ExtractionContext,
        region: tuple[int, int, int, int] | None = None,
    ) -> Path:
        """
        Capture screenshot of the target application.

        Uses platform-specific methods to capture the window.
        """
        # TODO: Implement platform-specific screenshot capture
        # For now, use HAL screen capture
        raise NotImplementedError(
            "Screenshot capture not yet implemented for AccessibilityExtractor"
        )

    @classmethod
    def supports_target(cls, context: ExtractionContext) -> bool:
        """Check if accessibility extraction can handle this target."""
        # Must have app identification and be on supported platform
        has_app_id = (
            context.app_name is not None
            or context.window_title is not None
            or context.process_id is not None
        )
        supported_platform = context.platform in ("win32", "darwin", "linux")

        return has_app_id and supported_platform

    @classmethod
    def get_name(cls) -> str:
        """Return extractor name."""
        return "accessibility"

    @classmethod
    def get_priority(cls) -> int:
        """Return priority (higher than vision, lower than DOM for web)."""
        return 50

    def _init_platform_api(self, config: "AccessibilityConfig") -> None:
        """Initialize platform-specific accessibility API."""
        if self._api is not None:
            return

        if self._platform == "win32" and config.use_uia:
            self._api = WindowsUIAutomationAPI()
        elif self._platform == "darwin" and config.use_ax:
            self._api = MacOSAccessibilityAPI()
        elif self._platform.startswith("linux") and config.use_atspi:
            self._api = LinuxATSPIAPI()
        else:
            raise RuntimeError(f"No accessibility API available for platform: {self._platform}")

    async def _find_target(
        self,
        context: ExtractionContext,
        config: "AccessibilityConfig",
    ) -> Any:
        """Find the target application/window."""
        if self._api is None:
            return None

        # Try different identification methods
        if context.process_id:
            return await self._api.find_by_pid(context.process_id)
        elif context.window_title:
            return await self._api.find_by_title(context.window_title)
        elif context.app_name:
            return await self._api.find_by_name(context.app_name)

        return None

    async def _traverse_tree(
        self,
        node: Any,
        config: "AccessibilityConfig",
        depth: int = 0,
    ) -> list[ExtractedElement]:
        """Recursively traverse accessibility tree."""
        if depth > config.max_depth:
            return []

        if self._api is None:
            return []

        elements: list[ExtractedElement] = []

        # Get element properties from node
        props = await self._api.get_properties(node)
        if props is None:
            return elements

        # Apply filters
        if not config.include_invisible and not props.get("is_visible", True):
            return elements
        if not config.include_disabled and not props.get("is_enabled", True):
            return elements
        if not config.include_offscreen and props.get("is_offscreen", False):
            return elements

        # Role filtering
        role = props.get("role", "unknown")
        if config.include_roles and role not in config.include_roles:
            return elements
        if config.exclude_roles and role in config.exclude_roles:
            return elements

        # Create element
        self._element_counter += 1
        element = ExtractedElement(
            id=f"a11y_element_{self._element_counter:04d}",
            element_type=self._role_to_element_type(role),
            bbox=props.get("bbox", (0, 0, 0, 0)),
            confidence=1.0,  # Accessibility info is authoritative
            text=props.get("value") or props.get("name"),
            name=props.get("name"),
            aria_role=role,
            aria_label=props.get("description"),
            is_interactive=props.get("is_interactive", False),
            is_visible=props.get("is_visible", True),
            is_enabled=props.get("is_enabled", True),
            is_focused=props.get("is_focused", False),
            extraction_method="accessibility",
            source_backend=self._api.get_name(),
            attributes=props.get("attributes", {}),
        )
        elements.append(element)

        # Traverse children
        if config.include_children:
            children = await self._api.get_children(node)
            for child in children:
                child_elements = await self._traverse_tree(child, config, depth + 1)
                elements.extend(child_elements)

        return elements

    def _role_to_element_type(self, role: str | None) -> str:
        """Convert accessibility role to element type."""
        role_mapping = {
            "button": "button",
            "push_button": "button",
            "toggle_button": "button",
            "check_box": "checkbox",
            "radio_button": "radio",
            "text": "text",
            "text_field": "input",
            "edit": "input",
            "combo_box": "select",
            "list_box": "select",
            "link": "link",
            "hyperlink": "link",
            "image": "image",
            "label": "label",
            "static_text": "label",
            "menu": "menu",
            "menu_item": "menuitem",
            "menu_bar": "menubar",
            "tab": "tab",
            "tab_list": "tablist",
            "tree": "tree",
            "tree_item": "treeitem",
            "table": "table",
            "cell": "cell",
            "row": "row",
            "column": "column",
            "scroll_bar": "scrollbar",
            "slider": "slider",
            "progress_bar": "progressbar",
            "tooltip": "tooltip",
            "window": "window",
            "dialog": "dialog",
            "pane": "pane",
            "panel": "panel",
            "group": "group",
            "toolbar": "toolbar",
        }
        return role_mapping.get(role or "", "unknown")

    def _role_to_state_type(self, role: str | None) -> str:
        """Convert accessibility role to state type."""
        state_mapping = {
            "window": "window",
            "dialog": "modal",
            "alert": "modal",
            "pane": "panel",
            "panel": "panel",
            "menu": "menu",
            "menu_bar": "menubar",
            "toolbar": "toolbar",
            "tab_panel": "tab",
            "group": "group",
        }
        return state_mapping.get(role or "", "container")

    def _find_contained_elements(
        self,
        container: ExtractedElement,
        all_elements: list[ExtractedElement],
    ) -> list[str]:
        """Find element IDs contained within a container's bbox."""
        if container.bbox is None:
            return []

        cx, cy, cw, ch = container.bbox
        contained = []

        for element in all_elements:
            if element.id == container.id:
                continue
            if element.bbox is None:
                continue

            ex, ey, ew, eh = element.bbox

            # Check if element is fully contained
            if ex >= cx and ey >= cy and ex + ew <= cx + cw and ey + eh <= cy + ch:
                contained.append(element.id)

        return contained


# Platform-specific API stubs
# These will be implemented in separate files under windows/, macos/, linux/


class PlatformAccessibilityAPI:
    """Base class for platform-specific accessibility APIs."""

    async def find_by_pid(self, pid: int) -> Any:
        """Find application by process ID."""
        raise NotImplementedError

    async def find_by_title(self, title: str) -> Any:
        """Find window by title."""
        raise NotImplementedError

    async def find_by_name(self, name: str) -> Any:
        """Find application by name."""
        raise NotImplementedError

    async def get_properties(self, node: Any) -> dict[str, Any] | None:
        """Get properties of an accessibility node."""
        raise NotImplementedError

    async def get_children(self, node: Any) -> list[Any]:
        """Get child nodes."""
        raise NotImplementedError

    def get_name(self) -> str:
        """Get API name."""
        raise NotImplementedError


class WindowsUIAutomationAPI(PlatformAccessibilityAPI):
    """
    Windows UI Automation API.

    TODO: Implement using comtypes or pywinauto.
    """

    def __init__(self) -> None:
        """Initialize Windows UI Automation."""
        self._uia = None
        try:
            # Try to import comtypes for UIA
            import comtypes.client

            self._uia = comtypes.client.CreateObject(
                "{ff48dba4-60ef-4201-aa87-54103eef594e}",  # CUIAutomation
                interface_name="IUIAutomation",
            )
        except ImportError:
            logger.warning("comtypes not installed, UIA not available")
        except Exception as e:
            logger.warning(f"Failed to initialize UIA: {e}")

    async def find_by_pid(self, pid: int) -> Any:
        """Find by process ID."""
        # TODO: Implement
        return None

    async def find_by_title(self, title: str) -> Any:
        """Find by window title."""
        # TODO: Implement
        return None

    async def find_by_name(self, name: str) -> Any:
        """Find by app name."""
        # TODO: Implement
        return None

    async def get_properties(self, node: Any) -> dict[str, Any] | None:
        """Get node properties."""
        # TODO: Implement
        return None

    async def get_children(self, node: Any) -> list[Any]:
        """Get children."""
        # TODO: Implement
        return []

    def get_name(self) -> str:
        """Get API name."""
        return "windows_uia"


class MacOSAccessibilityAPI(PlatformAccessibilityAPI):
    """
    macOS Accessibility API.

    TODO: Implement using pyobjc.
    """

    def __init__(self) -> None:
        """Initialize macOS Accessibility."""
        self._ax = None
        try:
            import ApplicationServices

            self._ax = ApplicationServices
        except ImportError:
            logger.warning("pyobjc not installed, AX not available")

    async def find_by_pid(self, pid: int) -> Any:
        """Find by process ID."""
        # TODO: Implement
        return None

    async def find_by_title(self, title: str) -> Any:
        """Find by window title."""
        # TODO: Implement
        return None

    async def find_by_name(self, name: str) -> Any:
        """Find by app name."""
        # TODO: Implement
        return None

    async def get_properties(self, node: Any) -> dict[str, Any] | None:
        """Get node properties."""
        # TODO: Implement
        return None

    async def get_children(self, node: Any) -> list[Any]:
        """Get children."""
        # TODO: Implement
        return []

    def get_name(self) -> str:
        """Get API name."""
        return "macos_ax"


class LinuxATSPIAPI(PlatformAccessibilityAPI):
    """
    Linux AT-SPI API.

    TODO: Implement using pyatspi2.
    """

    def __init__(self) -> None:
        """Initialize Linux AT-SPI."""
        self._atspi = None
        try:
            import pyatspi

            self._atspi = pyatspi
        except ImportError:
            logger.warning("pyatspi not installed, AT-SPI not available")

    async def find_by_pid(self, pid: int) -> Any:
        """Find by process ID."""
        # TODO: Implement
        return None

    async def find_by_title(self, title: str) -> Any:
        """Find by window title."""
        # TODO: Implement
        return None

    async def find_by_name(self, name: str) -> Any:
        """Find by app name."""
        # TODO: Implement
        return None

    async def get_properties(self, node: Any) -> dict[str, Any] | None:
        """Get node properties."""
        # TODO: Implement
        return None

    async def get_children(self, node: Any) -> list[Any]:
        """Get children."""
        # TODO: Implement
        return []

    def get_name(self) -> str:
        """Get API name."""
        return "linux_atspi"
