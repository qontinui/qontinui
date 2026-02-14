"""UI Bridge Explorer for automatic application exploration.

This module provides systematic exploration of applications that have UI Bridge
integrated. It discovers interactive elements, performs actions, and captures
DOM snapshots for co-occurrence analysis.

The explorer:
- Connects to target apps via UI Bridge control API (HTTP)
- Discovers interactive elements using UI Bridge's find() method
- Performs systematic exploration with safety controls
- Captures DOM snapshots after each action
- Returns render logs for co-occurrence analysis

When using the "extension" target type, the explorer uses a capture session
to build fingerprint-based CooccurrenceExport data during exploration:
- Element fingerprints enable cross-page element matching
- Capture sessions track state transitions via actions
- Fingerprint-based state discovery produces higher-quality states

Example:
    >>> from qontinui.discovery.ui_bridge_explorer import UIBridgeExplorer
    >>> from qontinui.discovery.target_connection import ExplorationConfig
    >>>
    >>> config = ExplorationConfig(
    ...     target_type="web",
    ...     connection_url="http://localhost:3000",
    ...     blocked_keywords=["delete", "logout", "remove"],
    ... )
    >>>
    >>> async with UIBridgeExplorer(config) as explorer:
    ...     result = await explorer.explore()
    ...     print(f"Explored {result.elements_explored} elements")
    ...     print(f"Captured {len(result.render_logs)} snapshots")
    >>>
    >>> # For fingerprint-based exploration via Chrome extension:
    >>> config = ExplorationConfig(
    ...     target_type="extension",
    ...     connection_url="http://localhost:9876",  # Runner URL
    ... )
    >>> async with UIBridgeExplorer(config) as explorer:
    ...     result = await explorer.explore()
    ...     # result.cooccurrence_export contains fingerprint data
    ...     print(f"States: {len(result.state_discovery_result.states)}")
"""

import asyncio
import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Self

from .target_connection import (
    ActionResult,
    ActionType,
    DOMSnapshot,
    Element,
    ExplorationConfig,
    TargetConnection,
    create_connection,
)
from .ui_bridge_adapter import UIBridgeStateDiscoveryResult, discover_states_from_renders

# Type alias for progress callback
# Returns False to stop exploration, True to continue
ProgressCallback = Callable[[str, int, int, str | None], bool]

logger = logging.getLogger(__name__)


@dataclass
class ExplorationStep:
    """A single step in the exploration process.

    Attributes:
        step_id: Unique identifier for this step
        timestamp: When the step was executed
        element_id: ID of the element interacted with (if any)
        action: Action performed
        action_result: Result of the action
        snapshot_before: DOM state before the action
        snapshot_after: DOM state after the action
        depth: Navigation depth from starting page
        parent_step_id: ID of the parent step (if navigated from another step)
    """

    step_id: str
    timestamp: datetime
    element_id: str | None
    action: str
    action_result: ActionResult | None
    snapshot_before: DOMSnapshot | None
    snapshot_after: DOMSnapshot
    depth: int = 0
    parent_step_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "stepId": self.step_id,
            "timestamp": self.timestamp.isoformat(),
            "elementId": self.element_id,
            "action": self.action,
            "actionResult": self.action_result.to_dict() if self.action_result else None,
            "snapshotBefore": self.snapshot_before.to_dict() if self.snapshot_before else None,
            "snapshotAfter": self.snapshot_after.to_dict(),
            "depth": self.depth,
            "parentStepId": self.parent_step_id,
        }


@dataclass
class ExplorationResult:
    """Result of a complete exploration session.

    Attributes:
        exploration_id: Unique identifier for this exploration
        config: Configuration used for exploration
        steps: All exploration steps performed
        render_logs: DOM snapshots in format suitable for co-occurrence analysis
        elements_discovered: Total unique elements discovered
        elements_explored: Total elements interacted with
        errors: Errors encountered during exploration
        start_time: When exploration started
        end_time: When exploration completed
        state_discovery_result: Result of co-occurrence state discovery
        cooccurrence_export: Raw cooccurrence export data (when using extension target)
    """

    exploration_id: str
    config: ExplorationConfig
    steps: list[ExplorationStep] = field(default_factory=list)
    render_logs: list[dict[str, Any]] = field(default_factory=list)
    elements_discovered: int = 0
    elements_explored: int = 0
    errors: list[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    state_discovery_result: UIBridgeStateDiscoveryResult | None = None
    cooccurrence_export: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "explorationId": self.exploration_id,
            "config": self.config.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
            "renderLogs": self.render_logs,
            "elementsDiscovered": self.elements_discovered,
            "elementsExplored": self.elements_explored,
            "errors": self.errors,
            "startTime": self.start_time.isoformat(),
            "endTime": self.end_time.isoformat() if self.end_time else None,
            "stateDiscoveryResult": (
                self.state_discovery_result.to_dict() if self.state_discovery_result else None
            ),
            "cooccurrenceExport": self.cooccurrence_export,
        }


class SafetyFilter:
    """Filters elements based on safety rules.

    Ensures that dangerous elements (delete buttons, logout links, etc.)
    are not interacted with during exploration.
    """

    # Default blocked keywords (dangerous operations)
    DEFAULT_BLOCKED_KEYWORDS = [
        "delete",
        "remove",
        "logout",
        "signout",
        "sign-out",
        "sign out",
        "cancel subscription",
        "deactivate",
        "close account",
        "terminate",
        "destroy",
        "clear all",
        "reset",
        "unsubscribe",
        "revoke",
    ]

    # Default safe keywords (always safe to interact with)
    DEFAULT_SAFE_KEYWORDS = [
        "view",
        "show",
        "details",
        "expand",
        "collapse",
        "toggle",
        "open",
        "select",
        "filter",
        "sort",
        "search",
        "help",
        "info",
        "about",
    ]

    def __init__(
        self,
        blocked_keywords: list[str] | None = None,
        safe_keywords: list[str] | None = None,
        blocked_selectors: list[str] | None = None,
    ):
        """Initialize the safety filter.

        Args:
            blocked_keywords: Additional keywords to block
            safe_keywords: Additional keywords to always allow
            blocked_selectors: CSS selectors to never interact with
        """
        self._blocked_keywords = set(self.DEFAULT_BLOCKED_KEYWORDS)
        if blocked_keywords:
            self._blocked_keywords.update(kw.lower() for kw in blocked_keywords)

        self._safe_keywords = set(self.DEFAULT_SAFE_KEYWORDS)
        if safe_keywords:
            self._safe_keywords.update(kw.lower() for kw in safe_keywords)

        self._blocked_selectors = set(blocked_selectors or [])

    def is_safe(self, element: Element) -> bool:
        """Check if an element is safe to interact with.

        Args:
            element: Element to check

        Returns:
            True if safe to interact, False otherwise
        """
        # Check if element matches blocked selector
        if self._matches_blocked_selector(element):
            logger.debug(f"Element {element.id} blocked by selector")
            return False

        # Get text to analyze (id, text content, aria-label)
        text_to_check = self._get_element_text(element).lower()

        # Check for explicitly safe keywords first
        for safe_kw in self._safe_keywords:
            if safe_kw in text_to_check:
                return True

        # Check for blocked keywords
        for blocked_kw in self._blocked_keywords:
            if blocked_kw in text_to_check:
                logger.debug(f"Element {element.id} blocked by keyword: {blocked_kw}")
                return False

        # Check for dangerous patterns
        if self._has_dangerous_pattern(text_to_check):
            logger.debug(f"Element {element.id} blocked by dangerous pattern")
            return False

        return True

    def _get_element_text(self, element: Element) -> str:
        """Extract all text associated with an element."""
        texts = [element.id]

        if element.text_content:
            texts.append(element.text_content)

        aria_label = element.attributes.get("aria-label")
        if aria_label:
            texts.append(aria_label)

        title = element.attributes.get("title")
        if title:
            texts.append(title)

        return " ".join(texts)

    def _matches_blocked_selector(self, element: Element) -> bool:
        """Check if element matches any blocked selector."""
        for selector in self._blocked_selectors:
            # Simple selector matching (could be extended)
            if selector.startswith("#") and element.id == selector[1:]:
                return True
            if selector.startswith("."):
                class_name = selector[1:]
                element_classes = element.attributes.get("class", "").split()
                if class_name in element_classes:
                    return True
            if selector.startswith("[data-ui-id="):
                # Extract value from [data-ui-id="value"]
                match = re.match(r'\[data-ui-id="([^"]+)"\]', selector)
                if match and element.id == match.group(1):
                    return True

        return False

    def _has_dangerous_pattern(self, text: str) -> bool:
        """Check for dangerous action patterns in text."""
        dangerous_patterns = [
            r"confirm.*delete",
            r"permanently.*remove",
            r"cannot.*undo",
            r"irreversible",
            r"final.*action",
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False


class ElementPrioritizer:
    """Prioritizes elements for exploration order.

    Elements are prioritized based on:
    - Interactive potential (buttons > links > inputs)
    - Visibility and prominence
    - Unexplored status
    """

    # Priority weights by tag/role
    PRIORITY_WEIGHTS = {
        "button": 100,
        "a": 90,
        "input": 80,
        "select": 75,
        "textarea": 70,
        "checkbox": 65,
        "radio": 60,
        "tab": 55,
        "menuitem": 50,
        "link": 45,
    }

    def prioritize(
        self,
        elements: list[Element],
        explored_ids: set[str],
    ) -> list[Element]:
        """Sort elements by exploration priority.

        Args:
            elements: Elements to prioritize
            explored_ids: Set of already explored element IDs

        Returns:
            Elements sorted by priority (highest first)
        """

        def get_priority(elem: Element) -> int:
            # Base priority from tag/role
            priority = self.PRIORITY_WEIGHTS.get(elem.tag_name.lower(), 30)

            # Boost for role
            if elem.role:
                role_boost = self.PRIORITY_WEIGHTS.get(elem.role.value.lower(), 0)
                priority = max(priority, role_boost)

            # Penalize already explored
            if elem.id in explored_ids:
                priority -= 1000

            # Penalize invisible/disabled
            if not elem.is_visible:
                priority -= 500
            if not elem.is_enabled:
                priority -= 500

            # Boost for prominent position (top of page)
            if elem.bbox and elem.bbox.y < 200:
                priority += 10

            return priority

        return sorted(elements, key=get_priority, reverse=True)


class UIBridgeExplorer:
    """Main explorer class for UI Bridge applications.

    Systematically explores applications by:
    1. Connecting to the target via UI Bridge
    2. Discovering interactive elements
    3. Executing actions with safety controls
    4. Capturing DOM snapshots for analysis
    5. Running co-occurrence analysis on results

    When using target_type="extension", the explorer:
    - Uses capture sessions to track element fingerprints
    - Records actions with before/after captures
    - Exports CooccurrenceExport data for fingerprint-based state discovery
    - Produces higher-quality states through cross-page element matching

    Example:
        >>> config = ExplorationConfig(
        ...     target_type="web",
        ...     connection_url="http://localhost:3000",
        ... )
        >>> async with UIBridgeExplorer(config) as explorer:
        ...     result = await explorer.explore()
    """

    # Keywords that indicate a close button or modal dismiss action
    CLOSE_BUTTON_KEYWORDS = [
        "close",
        "dismiss",
        "cancel",
        "x",
        "\u00d7",  # multiplication sign (x)
        "\u2715",  # multiplication x
        "\u2716",  # heavy multiplication x
        "\u2717",  # ballot x
        "\u2718",  # heavy ballot x
    ]

    # Keywords that indicate navigation elements to go back/home
    NAVIGATION_KEYWORDS = [
        "home",
        "back",
        "return",
        "go back",
        "dashboard",
        "main",
        "menu",
    ]

    def __init__(
        self,
        config: ExplorationConfig,
        on_progress: ProgressCallback | None = None,
    ):
        """Initialize the explorer.

        Args:
            config: Exploration configuration
            on_progress: Optional callback for progress updates.
                Signature: (message, elements_discovered, elements_explored, current_element) -> bool
                Return False to stop exploration, True to continue.
        """
        self._config = config
        self._connection: TargetConnection | None = None
        self._safety_filter = SafetyFilter(
            blocked_keywords=config.blocked_keywords,
            safe_keywords=config.safe_keywords,
            blocked_selectors=config.blocked_selectors,
        )
        self._prioritizer = ElementPrioritizer()
        self._explored_elements: set[str] = set()
        self._visited_states: set[str] = set()
        self._on_progress = on_progress
        self._stop_requested = False
        self._elements_discovered = 0

        # Capture session tracking (for extension target type)
        self._uses_capture_session = config.target_type == "extension"
        self._capture_session_active = False
        self._last_capture_id: str | None = None
        self._element_fingerprints: dict[str, str] = {}  # element_id -> fingerprint_hash

        # Navigation history tracking for smarter back navigation
        self._url_history: list[str] = []  # URLs visited during exploration
        self._state_history: list[str] = []  # State fingerprint hashes for visited states

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()

    async def connect(self) -> bool:
        """Connect to the target application.

        Returns:
            True if connection successful
        """
        self._connection = create_connection(self._config)
        return await self._connection.connect()

    async def disconnect(self) -> None:
        """Disconnect from the target application."""
        if self._connection:
            await self._connection.disconnect()
            self._connection = None

    def _report_progress(
        self,
        message: str,
        current_element: str | None = None,
    ) -> bool:
        """Report progress and check if exploration should continue.

        Args:
            message: Progress message
            current_element: Currently processing element ID

        Returns:
            True to continue, False to stop
        """
        if self._stop_requested:
            return False

        if self._on_progress:
            should_continue = self._on_progress(
                message,
                self._elements_discovered,
                len(self._explored_elements),
                current_element,
            )
            if not should_continue:
                self._stop_requested = True
                return False

        return True

    def stop(self) -> None:
        """Request exploration to stop gracefully."""
        self._stop_requested = True

    def _get_extension_connection(self) -> Any | None:
        """Get the connection as ExtensionTargetConnection if applicable.

        Returns:
            ExtensionTargetConnection if using extension target type, None otherwise.
        """
        if not self._uses_capture_session or not self._connection:
            return None

        # Import here to avoid circular dependency
        from .extension_connection import ExtensionTargetConnection

        if isinstance(self._connection, ExtensionTargetConnection):
            return self._connection
        return None

    async def _start_capture_session(self) -> bool:
        """Start a capture session for fingerprint-based exploration.

        Returns:
            True if session started successfully, False otherwise.
        """
        ext_conn = self._get_extension_connection()
        if not ext_conn:
            return False

        try:
            await ext_conn.start_capture_session()
            self._capture_session_active = True
            logger.info("Started capture session for fingerprint-based exploration")
            return True
        except Exception as e:
            logger.warning(f"Failed to start capture session: {e}")
            return False

    async def _create_capture(self, triggered_by: dict[str, Any] | None = None) -> str | None:
        """Create a capture in the active session.

        Args:
            triggered_by: Optional info about what triggered this capture

        Returns:
            Capture ID if successful, None otherwise.
        """
        ext_conn = self._get_extension_connection()
        if not ext_conn or not self._capture_session_active:
            return None

        try:
            capture = await ext_conn.create_capture(triggered_by=triggered_by)
            capture_id: str = capture.capture_id
            self._last_capture_id = capture_id

            # Store element fingerprints from the capture for later lookup
            # Note: The extension sends fingerprint hashes with the capture
            return capture_id
        except Exception as e:
            logger.warning(f"Failed to create capture: {e}")
            return None

    async def _record_action(
        self,
        action_type: str,
        target_element_id: str,
        before_capture_id: str,
        after_capture_id: str,
    ) -> None:
        """Record an action in the capture session.

        Args:
            action_type: Type of action performed
            target_element_id: Element ID that was acted upon
            before_capture_id: Capture ID before the action
            after_capture_id: Capture ID after the action
        """
        ext_conn = self._get_extension_connection()
        if not ext_conn or not self._capture_session_active:
            return

        # Get fingerprint hash for the target element
        target_fingerprint = self._element_fingerprints.get(target_element_id, target_element_id)

        try:
            await ext_conn.record_action(
                action_type=action_type,
                target_fingerprint=target_fingerprint,
                before_capture_id=before_capture_id,
                after_capture_id=after_capture_id,
            )
        except Exception as e:
            logger.warning(f"Failed to record action: {e}")

    async def _end_capture_session(self) -> dict[str, Any] | None:
        """End the capture session and export data.

        Returns:
            CooccurrenceExport data if successful, None otherwise.
        """
        ext_conn = self._get_extension_connection()
        if not ext_conn or not self._capture_session_active:
            return None

        try:
            # Export the session data in CooccurrenceExport format
            export_data: dict[str, Any] = await ext_conn.export_capture_session()
            await ext_conn.end_capture_session()
            self._capture_session_active = False
            logger.info("Ended capture session and exported cooccurrence data")
            return export_data
        except Exception as e:
            logger.warning(f"Failed to end capture session: {e}")
            self._capture_session_active = False
            return None

    async def _store_element_fingerprints(self, elements: list[Element]) -> None:
        """Store fingerprint hashes for elements.

        Args:
            elements: List of elements (may include ExtensionElement with fingerprints)
        """
        # Import here to avoid circular dependency
        from .extension_connection import ExtensionElement

        for elem in elements:
            if isinstance(elem, ExtensionElement) and elem.fingerprint:
                self._element_fingerprints[elem.id] = elem.fingerprint.hash

    async def explore(self) -> ExplorationResult:
        """Perform systematic exploration of the application.

        When using the extension target type, this method:
        1. Starts a capture session to track fingerprints
        2. Creates captures before/after each action
        3. Records actions with fingerprint data
        4. Exports CooccurrenceExport for fingerprint-based state discovery

        Returns:
            ExplorationResult with all captured data
        """
        if not self._connection or not self._connection.is_connected:
            raise RuntimeError("Not connected to target")

        result = ExplorationResult(
            exploration_id=str(uuid.uuid4()),
            config=self._config,
            start_time=datetime.now(),
        )

        try:
            # Start capture session if using extension target
            if self._uses_capture_session:
                if not self._report_progress("Starting capture session..."):
                    result.end_time = datetime.now()
                    return result
                await self._start_capture_session()

            # Report initial progress
            if not self._report_progress("Capturing initial state..."):
                result.end_time = datetime.now()
                return result

            # Capture initial state
            initial_snapshot = await self._connection.capture_snapshot(
                include_screenshot=self._config.capture_screenshots
            )

            # Update discovered elements count
            self._elements_discovered = len(initial_snapshot.elements)

            # Store fingerprints from initial elements
            await self._store_element_fingerprints(initial_snapshot.elements)

            # Create initial capture in session
            if self._capture_session_active:
                await self._create_capture(triggered_by={"type": "initial"})

            # Check if we found any elements - if not, provide helpful guidance
            if self._elements_discovered == 0:
                target_type = self._config.target_type
                if target_type == "web":
                    error_msg = (
                        "No elements found. For web apps, the UI Bridge SDK server "
                        "endpoints cannot access browser-side elements. "
                        "Try using target type 'extension' instead, which uses the Chrome "
                        "extension to access elements directly in the browser."
                    )
                else:
                    error_msg = (
                        "No elements found. Make sure the target application has "
                        "UI Bridge elements registered (elements with data-ui-id attributes)."
                    )
                logger.warning(error_msg)
                result.errors.append(error_msg)

            # Record initial state as first step
            initial_step = ExplorationStep(
                step_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                element_id=None,
                action="initial",
                action_result=None,
                snapshot_before=None,
                snapshot_after=initial_snapshot,
                depth=0,
            )
            result.steps.append(initial_step)

            if self._config.record_render_logs:
                result.render_logs.append(self._snapshot_to_render_log(initial_snapshot))

            # Report progress after initial capture
            if not self._report_progress(
                f"Found {self._elements_discovered} elements, starting exploration..."
            ):
                result.end_time = datetime.now()
                result.elements_discovered = self._elements_discovered
                result.elements_explored = len(self._explored_elements)
                return result

            # Start exploration from initial state
            await self._explore_state(
                snapshot=initial_snapshot,
                depth=0,
                parent_step_id=initial_step.step_id,
                result=result,
            )

        except Exception as e:
            logger.error(f"Exploration error: {e}")
            result.errors.append(str(e))

        result.end_time = datetime.now()
        result.elements_discovered = len(self._get_all_discovered_elements(result))
        result.elements_explored = len(self._explored_elements)

        # End capture session and export cooccurrence data
        cooccurrence_export = None
        if self._capture_session_active:
            if not self._report_progress("Exporting capture session..."):
                pass  # Continue anyway to get partial results
            cooccurrence_export = await self._end_capture_session()
            result.cooccurrence_export = cooccurrence_export

        # Run state discovery - prefer fingerprint-based if we have export data
        if cooccurrence_export:
            logger.info("Running fingerprint-based state discovery from capture session")
            result.state_discovery_result = discover_states_from_renders(
                renders=result.render_logs,
                cooccurrence_export=cooccurrence_export,
            )
        elif result.render_logs:
            logger.info("Running render-log-based state discovery")
            result.state_discovery_result = discover_states_from_renders(result.render_logs)

        return result

    async def _explore_state(
        self,
        snapshot: DOMSnapshot,
        depth: int,
        parent_step_id: str,
        result: ExplorationResult,
    ) -> None:
        """Recursively explore a state by interacting with elements.

        Args:
            snapshot: Current DOM snapshot
            depth: Current navigation depth
            parent_step_id: ID of the parent step
            result: Result object to accumulate data
        """
        # Check for stop request
        if self._stop_requested:
            return

        # Check depth limit
        if depth >= self._config.max_depth:
            logger.debug(f"Max depth {self._config.max_depth} reached")
            return

        # Check total elements limit
        if len(self._explored_elements) >= self._config.max_total_elements:
            logger.info(f"Max total elements {self._config.max_total_elements} reached")
            return

        # Get state hash to avoid re-exploring same state
        state_hash = self._compute_state_hash(snapshot)
        if state_hash in self._visited_states:
            logger.debug("State already visited, skipping")
            return
        self._visited_states.add(state_hash)

        # Get and prioritize elements
        elements = snapshot.elements
        safe_elements = [e for e in elements if self._safety_filter.is_safe(e)]
        prioritized = self._prioritizer.prioritize(safe_elements, self._explored_elements)

        # Limit elements per page
        elements_to_explore = prioritized[: self._config.max_elements_per_page]

        logger.info(
            f"Exploring {len(elements_to_explore)} of {len(safe_elements)} safe elements "
            f"at depth {depth}"
        )

        # Report progress
        if not self._report_progress(
            f"Exploring depth {depth}: {len(elements_to_explore)} elements to check"
        ):
            return

        for element in elements_to_explore:
            # Check for stop request
            if self._stop_requested:
                break

            # Check limits again in loop
            if len(self._explored_elements) >= self._config.max_total_elements:
                break

            if element.id in self._explored_elements:
                continue

            await self._explore_element(
                element=element,
                current_snapshot=snapshot,
                depth=depth,
                parent_step_id=parent_step_id,
                result=result,
            )

            # Delay between actions
            if self._config.action_delay_ms > 0:
                await asyncio.sleep(self._config.action_delay_ms / 1000.0)

    async def _explore_element(
        self,
        element: Element,
        current_snapshot: DOMSnapshot,
        depth: int,
        parent_step_id: str,
        result: ExplorationResult,
    ) -> None:
        """Explore a single element by executing an action.

        When using extension target with capture session:
        1. Creates a capture before the action
        2. Executes the action
        3. Creates a capture after the action
        4. Records the action with before/after capture IDs

        Args:
            element: Element to explore
            current_snapshot: Snapshot before action
            depth: Current depth
            parent_step_id: Parent step ID
            result: Result object
        """
        if not self._connection:
            return

        # Check for stop request
        if self._stop_requested:
            return

        self._explored_elements.add(element.id)

        # Determine action based on element type
        action = self._determine_action(element)

        logger.debug(f"Exploring element {element.id} with action {action}")

        # Report progress with current element
        element_name = element.text_content or element.id
        if not self._report_progress(
            f"Exploring: {element_name[:50]}...",
            current_element=element.id,
        ):
            return

        try:
            # Create capture before action (for fingerprint tracking)
            before_capture_id = None
            if self._capture_session_active:
                before_capture_id = await self._create_capture(
                    triggered_by={
                        "type": "before_action",
                        "action": action,
                        "elementId": element.id,
                    }
                )

            # Execute the action
            action_result = await self._connection.execute_action(
                element_id=element.id,
                action=action,
            )

            # Capture state after action
            after_snapshot = await self._connection.capture_snapshot(
                include_screenshot=self._config.capture_screenshots
            )

            # Store fingerprints from new elements
            await self._store_element_fingerprints(after_snapshot.elements)

            # Create capture after action (for fingerprint tracking)
            after_capture_id = None
            if self._capture_session_active:
                after_capture_id = await self._create_capture(
                    triggered_by={
                        "type": "after_action",
                        "action": action,
                        "elementId": element.id,
                        "success": action_result.success,
                        "stateChanged": action_result.state_changed,
                    }
                )

                # Record the action with before/after captures
                if before_capture_id and after_capture_id:
                    await self._record_action(
                        action_type=action,
                        target_element_id=element.id,
                        before_capture_id=before_capture_id,
                        after_capture_id=after_capture_id,
                    )

            # Update discovered elements count
            new_elements = len(after_snapshot.elements)
            if new_elements > self._elements_discovered:
                self._elements_discovered = new_elements

            # Record the step
            step = ExplorationStep(
                step_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                element_id=element.id,
                action=action,
                action_result=action_result,
                snapshot_before=current_snapshot,
                snapshot_after=after_snapshot,
                depth=depth,
                parent_step_id=parent_step_id,
            )
            result.steps.append(step)

            if self._config.record_render_logs:
                result.render_logs.append(self._snapshot_to_render_log(after_snapshot))

            # If action caused state change, explore the new state
            if action_result.success and action_result.state_changed:
                logger.debug(f"State changed after action on {element.id}")
                await self._explore_state(
                    snapshot=after_snapshot,
                    depth=depth + 1,
                    parent_step_id=step.step_id,
                    result=result,
                )

                # Navigate back if possible
                await self._navigate_back(result)

        except Exception as e:
            logger.warning(f"Error exploring element {element.id}: {e}")
            result.errors.append(f"Element {element.id}: {e}")

    def _determine_action(self, element: Element) -> str:
        """Determine the appropriate action for an element.

        Args:
            element: Element to analyze

        Returns:
            Action type string
        """
        tag = element.tag_name.lower()
        role = element.role.value.lower() if element.role else None
        elem_type = element.attributes.get("type", "").lower()

        # Input elements need special handling
        if tag == "input":
            if elem_type in ("checkbox", "radio"):
                return ActionType.CLICK.value
            if elem_type in ("text", "email", "password", "search", "tel", "url"):
                return ActionType.FOCUS.value
            return ActionType.CLICK.value

        # Select elements
        if tag == "select":
            return ActionType.FOCUS.value

        # Textarea
        if tag == "textarea":
            return ActionType.FOCUS.value

        # Buttons and links
        if tag in ("button", "a") or role in ("button", "link"):
            return ActionType.CLICK.value

        # Default to click
        return ActionType.CLICK.value

    async def _navigate_back(self, result: ExplorationResult) -> bool:
        """Navigate back to the previous state with multiple fallback strategies.

        Tries multiple strategies in order:
        1. Browser back button
        2. Close modal (Escape key or close button)
        3. Navigate to previous URL if tracked
        4. Click known navigation element (home, back button in UI)

        Args:
            result: Current exploration result

        Returns:
            True if navigation succeeded and we're now in a known state,
            False if all strategies failed.
        """
        if not self._connection:
            return False

        # Try each strategy with retries
        for _attempt in range(3):
            if await self._try_navigate_back_once():
                return True
            # Brief delay between retry attempts
            await asyncio.sleep(0.2)

        # All strategies failed
        logger.warning("All navigate back strategies failed, continuing from current state")
        return False

    async def _try_navigate_back_once(self) -> bool:
        """Try all navigation strategies once.

        Returns:
            True if any strategy succeeded and we're in a known state.
        """
        if not self._connection:
            return False

        # Strategy 1: Try browser back
        try:
            logger.debug("Navigate back: trying browser back")
            await self._connection.execute_action(
                element_id="__browser__",
                action="back",
            )
            await asyncio.sleep(self._config.action_delay_ms / 1000.0)
            new_snapshot = await self._connection.capture_snapshot()
            if self._is_known_state(new_snapshot):
                logger.debug("Navigate back: browser back succeeded")
                return True
        except Exception as e:
            logger.debug(f"Browser back failed: {e}")

        # Strategy 2: Try closing modal (Escape key or close button)
        try:
            logger.debug("Navigate back: trying to close modal")
            if await self._try_close_modal():
                return True
        except Exception as e:
            logger.debug(f"Close modal failed: {e}")

        # Strategy 3: Navigate to previous URL if tracked
        if self._url_history and len(self._url_history) > 1:
            try:
                prev_url = self._url_history[-2]
                logger.debug(f"Navigate back: trying to navigate to previous URL: {prev_url}")
                await self._connection.execute_action(
                    element_id="__browser__",
                    action="navigate",
                    value=prev_url,
                )
                await asyncio.sleep(self._config.action_delay_ms / 1000.0)
                new_snapshot = await self._connection.capture_snapshot()
                if self._is_known_state(new_snapshot):
                    logger.debug("Navigate back: URL navigation succeeded")
                    # Remove the current URL from history since we navigated back
                    self._url_history.pop()
                    return True
            except Exception as e:
                logger.debug(f"URL navigation failed: {e}")

        # Strategy 4: Click known navigation element (home, back button in UI)
        try:
            logger.debug("Navigate back: trying to find navigation element")
            if await self._try_click_navigation_element():
                return True
        except Exception as e:
            logger.debug(f"Click navigation element failed: {e}")

        return False

    def _is_known_state(self, snapshot: DOMSnapshot) -> bool:
        """Check if snapshot matches a previously visited state.

        Args:
            snapshot: The snapshot to check.

        Returns:
            True if this state has been visited before.
        """
        state_hash = self._compute_state_hash(snapshot)
        return state_hash in self._visited_states or state_hash in self._state_history

    async def _try_close_modal(self) -> bool:
        """Try to close a modal dialog if one is detected.

        Strategies:
        1. Press Escape key
        2. Find and click close button by aria-label or text content

        Returns:
            True if modal was closed and we're now in a known state.
        """
        if not self._connection:
            return False

        # Strategy 1: Try Escape key to close modal
        try:
            logger.debug("Close modal: trying Escape key")
            await self._connection.execute_action(
                element_id="__browser__",
                action="keypress",
                value="Escape",
            )
            await asyncio.sleep(self._config.action_delay_ms / 1000.0)
            new_snapshot = await self._connection.capture_snapshot()
            if self._is_known_state(new_snapshot):
                logger.debug("Close modal: Escape key succeeded")
                return True
        except Exception as e:
            logger.debug(f"Escape key failed: {e}")

        # Strategy 2: Find and click close button
        try:
            snapshot = await self._connection.capture_snapshot()
            close_button = self._find_close_button(snapshot)
            if close_button:
                logger.debug(f"Close modal: found close button {close_button.id}")
                await self._connection.execute_action(
                    element_id=close_button.id,
                    action="click",
                )
                await asyncio.sleep(self._config.action_delay_ms / 1000.0)
                new_snapshot = await self._connection.capture_snapshot()
                if self._is_known_state(new_snapshot):
                    logger.debug("Close modal: close button click succeeded")
                    return True
        except Exception as e:
            logger.debug(f"Close button click failed: {e}")

        return False

    def _find_close_button(self, snapshot: DOMSnapshot) -> Element | None:
        """Find a close button in the current snapshot.

        Looks for elements that appear to be modal close buttons:
        - Elements with role="dialog" close buttons
        - Elements with aria-label containing "close"
        - Buttons with "x" or close-related text

        Args:
            snapshot: Current DOM snapshot.

        Returns:
            Close button element if found, None otherwise.
        """
        # First, check if we're likely in a modal
        modal_detected = self._detect_modal(snapshot)
        if not modal_detected:
            return None

        for element in snapshot.elements:
            # Check aria-label
            aria_label = element.attributes.get("aria-label", "").lower()
            if any(kw in aria_label for kw in self.CLOSE_BUTTON_KEYWORDS):
                return element

            # Check text content
            text = (element.text_content or "").strip().lower()
            if text in self.CLOSE_BUTTON_KEYWORDS:
                return element

            # Check for close button by common class names
            class_attr = element.attributes.get("class", "").lower()
            if any(cls in class_attr for cls in ["close", "modal-close", "dialog-close"]):
                return element

            # Check title attribute
            title = element.attributes.get("title", "").lower()
            if any(kw in title for kw in self.CLOSE_BUTTON_KEYWORDS):
                return element

        return None

    def _detect_modal(self, snapshot: DOMSnapshot) -> bool:
        """Detect if a modal/dialog is currently displayed.

        Looks for:
        - Elements with role="dialog" or aria-modal="true"
        - Elements with common modal class names
        - Elements positioned as overlays (position:fixed with high z-index)

        Args:
            snapshot: Current DOM snapshot.

        Returns:
            True if a modal appears to be displayed.
        """
        for element in snapshot.elements:
            # Check ARIA attributes
            role = element.attributes.get("role", "").lower()
            if role in ("dialog", "alertdialog"):
                return True

            aria_modal = element.attributes.get("aria-modal", "").lower()
            if aria_modal == "true":
                return True

            # Check for common modal class names
            class_attr = element.attributes.get("class", "").lower()
            modal_classes = ["modal", "dialog", "popup", "overlay", "lightbox"]
            if any(cls in class_attr for cls in modal_classes):
                # Check that it's likely visible (not just a hidden modal container)
                if element.is_visible:
                    return True

        return False

    async def _try_click_navigation_element(self) -> bool:
        """Try to find and click a navigation element to go back/home.

        Looks for:
        - Home links
        - Back buttons in the UI (not browser back)
        - Navigation breadcrumbs
        - Menu items leading to main/dashboard

        Returns:
            True if navigation succeeded and we're in a known state.
        """
        if not self._connection:
            return False

        try:
            snapshot = await self._connection.capture_snapshot()
            nav_element = self._find_navigation_element(snapshot)
            if nav_element:
                logger.debug(f"Navigation: found element {nav_element.id}")
                await self._connection.execute_action(
                    element_id=nav_element.id,
                    action="click",
                )
                await asyncio.sleep(self._config.action_delay_ms / 1000.0)
                new_snapshot = await self._connection.capture_snapshot()
                if self._is_known_state(new_snapshot):
                    logger.debug("Navigation: click succeeded")
                    return True
        except Exception as e:
            logger.debug(f"Navigation element click failed: {e}")

        return False

    def _find_navigation_element(self, snapshot: DOMSnapshot) -> Element | None:
        """Find a navigation element that leads back/home.

        Args:
            snapshot: Current DOM snapshot.

        Returns:
            Navigation element if found, None otherwise.
        """
        candidates: list[tuple[int, Element]] = []

        for element in snapshot.elements:
            if not element.is_visible or not element.is_enabled:
                continue

            # Calculate a score based on navigation relevance
            score = 0
            text_lower = (element.text_content or "").lower()
            aria_label = element.attributes.get("aria-label", "").lower()
            id_lower = element.id.lower()
            href = element.attributes.get("href", "").lower()

            # Check for navigation keywords
            for kw in self.NAVIGATION_KEYWORDS:
                if kw in text_lower:
                    score += 10
                if kw in aria_label:
                    score += 8
                if kw in id_lower:
                    score += 5

            # Bonus for links to root/home
            if href in ("/", "/home", "/dashboard", "#"):
                score += 15

            # Bonus for nav elements
            role = element.attributes.get("role", "").lower()
            if role in ("navigation", "link"):
                score += 3

            # Bonus if it's in a breadcrumb
            class_attr = element.attributes.get("class", "").lower()
            if "breadcrumb" in class_attr:
                score += 5

            # Prefer buttons and links
            if element.tag_name.lower() in ("a", "button"):
                score += 2

            if score > 0:
                candidates.append((score, element))

        if not candidates:
            return None

        # Return the highest scoring candidate
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def _track_url(self, url: str) -> None:
        """Track a visited URL for history-based navigation.

        Args:
            url: The URL to track.
        """
        if not self._url_history or self._url_history[-1] != url:
            self._url_history.append(url)
            # Keep history bounded to avoid memory issues
            if len(self._url_history) > 100:
                self._url_history = self._url_history[-50:]

    def _track_state(self, state_hash: str) -> None:
        """Track a visited state for known-state checking.

        Args:
            state_hash: The state fingerprint hash to track.
        """
        if state_hash not in self._state_history:
            self._state_history.append(state_hash)
            # Keep history bounded
            if len(self._state_history) > 200:
                self._state_history = self._state_history[-100:]

    def _snapshot_to_render_log(self, snapshot: DOMSnapshot) -> dict[str, Any]:
        """Convert a DOM snapshot to render log format for co-occurrence analysis.

        Args:
            snapshot: DOM snapshot to convert

        Returns:
            Render log entry compatible with discover_states_from_renders()
        """
        # The UI Bridge snapshot API returns a flat element list, not a DOM tree.
        # If root is empty, synthesize a tree from elements so the legacy
        # co-occurrence strategy can extract element IDs (data-ui-id).
        root = snapshot.root
        if not root and snapshot.elements:
            root = {
                "tagName": "body",
                "children": [
                    {
                        "tagName": "div",
                        "attributes": {"data-ui-id": elem.id},
                        "children": [],
                    }
                    for elem in snapshot.elements
                ],
            }

        return {
            "id": snapshot.id,
            "type": "dom_snapshot",
            "timestamp": snapshot.timestamp.isoformat(),
            "snapshot": {
                "root": root,
                "url": snapshot.url,
                "title": snapshot.title,
            },
        }

    def _compute_state_hash(self, snapshot: DOMSnapshot) -> str:
        """Compute a hash of the DOM state for deduplication.

        Args:
            snapshot: Snapshot to hash

        Returns:
            Hash string representing the state
        """
        # Use sorted element IDs as state signature
        element_ids = sorted(e.id for e in snapshot.elements)
        return "|".join(element_ids)

    def _get_all_discovered_elements(self, result: ExplorationResult) -> set[str]:
        """Get all unique element IDs discovered during exploration.

        Args:
            result: Exploration result

        Returns:
            Set of all discovered element IDs
        """
        all_ids: set[str] = set()
        for step in result.steps:
            for elem in step.snapshot_after.elements:
                all_ids.add(elem.id)
        return all_ids


async def explore_application(
    connection_url: str,
    target_type: str = "web",
    max_depth: int = 2,
    max_elements: int = 100,
    blocked_keywords: list[str] | None = None,
    on_progress: ProgressCallback | None = None,
) -> ExplorationResult:
    """Convenience function to explore an application.

    Args:
        connection_url: URL to connect to
        target_type: Type of target ("web", "desktop", "mobile")
        max_depth: Maximum navigation depth
        max_elements: Maximum elements to explore
        blocked_keywords: Additional keywords to block
        on_progress: Optional callback for progress updates.
            Signature: (message, elements_discovered, elements_explored, current_element) -> bool
            Return False to stop exploration, True to continue.

    Returns:
        ExplorationResult with all captured data

    Example:
        >>> result = await explore_application(
        ...     "http://localhost:3000",
        ...     blocked_keywords=["delete", "logout"],
        ... )
        >>> print(f"Discovered {len(result.state_discovery_result.states)} states")
    """
    config = ExplorationConfig(
        target_type=target_type,  # type: ignore[arg-type]
        connection_url=connection_url,
        max_depth=max_depth,
        max_total_elements=max_elements,
        blocked_keywords=blocked_keywords or [],
    )

    async with UIBridgeExplorer(config, on_progress=on_progress) as explorer:
        return await explorer.explore()
