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

    Example:
        >>> config = ExplorationConfig(
        ...     target_type="web",
        ...     connection_url="http://localhost:3000",
        ... )
        >>> async with UIBridgeExplorer(config) as explorer:
        ...     result = await explorer.explore()
    """

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

    async def explore(self) -> ExplorationResult:
        """Perform systematic exploration of the application.

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

        # Run state discovery on render logs
        if result.render_logs:
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
            # Execute the action
            action_result = await self._connection.execute_action(
                element_id=element.id,
                action=action,
            )

            # Capture state after action
            after_snapshot = await self._connection.capture_snapshot(
                include_screenshot=self._config.capture_screenshots
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

    async def _navigate_back(self, result: ExplorationResult) -> None:
        """Navigate back to the previous state if possible.

        Args:
            result: Current exploration result
        """
        if not self._connection:
            return

        try:
            # Try browser back if available
            await self._connection.execute_action(
                element_id="__browser__",
                action="back",
            )

            # Wait for navigation
            await asyncio.sleep(self._config.action_delay_ms / 1000.0)

        except Exception as e:
            logger.debug(f"Could not navigate back: {e}")

    def _snapshot_to_render_log(self, snapshot: DOMSnapshot) -> dict[str, Any]:
        """Convert a DOM snapshot to render log format for co-occurrence analysis.

        Args:
            snapshot: DOM snapshot to convert

        Returns:
            Render log entry compatible with discover_states_from_renders()
        """
        return {
            "id": snapshot.id,
            "type": "dom_snapshot",
            "timestamp": snapshot.timestamp.isoformat(),
            "snapshot": {
                "root": snapshot.root,
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
