"""
Abstract base class for runtime UI extraction.

This module defines the interface for runtime extractors that connect to
running applications and extract UI state through observation and interaction.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..models.base import BoundingBox, Screenshot
from ..web.models import ExtractedElement, ExtractedTransition
from .types import ExtractionTarget, RuntimeStateCapture


@dataclass
class DetectedRegion:
    """A detected UI region from runtime analysis."""

    id: str
    name: str
    bbox: BoundingBox
    region_type: str
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InteractionAction:
    """An interaction to perform on the UI."""

    action_type: str  # click, type, hover, scroll, etc.
    target_selector: str | None = None
    target_element_id: str | None = None
    action_value: str | None = None  # Text to type, scroll distance, etc.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StateChange:
    """Changes observed after an interaction."""

    appeared_elements: list[str] = field(default_factory=list)
    disappeared_elements: list[str] = field(default_factory=list)
    modified_elements: list[str] = field(default_factory=list)
    appeared_states: list[str] = field(default_factory=list)
    disappeared_states: list[str] = field(default_factory=list)
    url_changed: bool = False
    new_url: str | None = None
    screenshot_before: str | None = None
    screenshot_after: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeExtractionResult:
    """Complete results from runtime extraction."""

    session_id: str
    target: ExtractionTarget
    captures: list[RuntimeStateCapture] = field(default_factory=list)
    transitions: list[ExtractedTransition] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class RuntimeExtractor(ABC):
    """
    Abstract base class for runtime UI extraction.

    Runtime extractors connect to running applications and extract UI state
    through:
    - DOM/UI tree inspection
    - Visual analysis (screenshots, bounding boxes)
    - Interaction simulation (clicks, typing, etc.)
    - State observation (before/after comparisons)

    This complements static analysis by providing the actual runtime behavior
    and discovering dynamic UI elements that may not be evident from source code.
    """

    @abstractmethod
    async def connect(self, target: ExtractionTarget) -> None:
        """
        Connect to the target application.

        Establishes a connection to the running application for extraction.
        The connection method varies by platform:
        - Web: Browser automation via Playwright/Selenium
        - Desktop: Accessibility APIs, UI automation frameworks
        - Mobile: Appium, device bridges

        Args:
            target: Configuration specifying how to connect to the application
                   (URL for web, process ID for desktop, device ID for mobile, etc.)

        Raises:
            ConnectionError: If unable to connect to the target application.
            TimeoutError: If connection takes too long.
        """
        pass

    @abstractmethod
    async def extract_current_state(self) -> RuntimeStateCapture:
        """
        Extract the current visible UI state.

        Captures a snapshot of the current UI state including:
        - All visible elements and their properties
        - Current route/URL
        - Application state (loading, error, success, etc.)
        - Active regions (open dialogs, expanded menus, etc.)
        - Screenshot for visual reference

        This provides a point-in-time view of the application's UI.

        Returns:
            RuntimeStateCapture containing all information about the current
            visible state.

        Raises:
            ExtractionError: If unable to extract the current state.
        """
        pass

    @abstractmethod
    async def extract_elements(self) -> list[ExtractedElement]:
        """
        Extract all interactive elements from the current UI.

        Identifies and extracts properties of UI elements:
        - Buttons, links, inputs, dropdowns, etc.
        - Element locations (bounding boxes)
        - Element properties (text, labels, values, state)
        - Element identifiers (IDs, selectors, accessibility info)
        - Interactive capabilities (clickable, editable, etc.)

        Returns:
            List of ExtractedElement objects representing all interactive
            elements in the current UI.

        Raises:
            ExtractionError: If unable to extract elements.
        """
        pass

    @abstractmethod
    async def detect_regions(self) -> list[DetectedRegion]:
        """
        Detect UI regions (navigation, modals, sidebars, etc.).

        Identifies logical groupings of UI elements that represent states:
        - Navigation bars and menus
        - Modal dialogs and overlays
        - Sidebars and panels
        - Forms and input groups
        - Content regions
        - Toolbars and action bars

        Uses heuristics like:
        - Semantic HTML/accessibility roles
        - Visual layout analysis
        - Common UI patterns
        - Element grouping and containment

        Returns:
            List of DetectedRegion objects representing identified UI regions.

        Raises:
            ExtractionError: If unable to detect regions.
        """
        pass

    @abstractmethod
    async def capture_screenshot(self, region: BoundingBox | None = None) -> Screenshot:
        """
        Capture a screenshot of the current state.

        Takes a visual snapshot of the application for:
        - Documentation and visualization
        - Visual comparison between states
        - Computer vision analysis
        - Test verification

        Args:
            region: Optional bounding box to capture only a specific region.
                   If None, captures the entire visible viewport.

        Returns:
            Screenshot object containing the image data and metadata.

        Raises:
            ScreenshotError: If unable to capture the screenshot.
        """
        pass

    @abstractmethod
    async def navigate_to_route(self, route: str) -> None:
        """
        Navigate to a specific route/URL.

        Triggers navigation within the application:
        - For web: Navigate to URL
        - For desktop: Open window/view
        - For mobile: Navigate to screen

        This is used to explore different parts of the application during
        state discovery.

        Args:
            route: The route/path to navigate to (URL for web, view ID for
                  desktop/mobile).

        Raises:
            NavigationError: If unable to navigate to the specified route.
            TimeoutError: If navigation takes too long.
        """
        pass

    @abstractmethod
    async def simulate_interaction(self, action: InteractionAction) -> StateChange:
        """
        Simulate a user interaction and observe the resulting state changes.

        Performs an action (click, type, hover, etc.) and captures:
        - Elements that appeared/disappeared
        - Elements that changed properties
        - Route/URL changes
        - New UI regions that became visible
        - Screenshot before and after

        This is the core of transition discovery - by simulating interactions
        and observing changes, we can map out the state machine.

        Args:
            action: The interaction to simulate (click on element, type text,
                   hover over element, etc.)

        Returns:
            StateChange object describing what changed as a result of the
            interaction.

        Raises:
            InteractionError: If unable to perform the interaction.
            TimeoutError: If waiting for state change takes too long.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Disconnect from the target application.

        Cleanly closes the connection and releases resources:
        - Close browser sessions
        - Release UI automation handles
        - Clean up temporary files
        - Save any pending data

        Should be called when extraction is complete or if an error occurs.

        Raises:
            DisconnectionError: If unable to cleanly disconnect.
        """
        pass

    @classmethod
    @abstractmethod
    def supports_target(cls, target: ExtractionTarget) -> bool:
        """
        Check if this extractor supports the given target type.

        Different extractors are designed for different platforms:
        - Web extractors for browser-based applications
        - Desktop extractors for native applications
        - Mobile extractors for iOS/Android apps

        Args:
            target: The extraction target to check support for.

        Returns:
            True if this extractor can extract from the given target type,
            False otherwise.
        """
        pass
