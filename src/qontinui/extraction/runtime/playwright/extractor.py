"""
PlaywrightExtractor for runtime web extraction.

Implements RuntimeExtractor using Playwright for web targets,
reusing the existing web extraction components.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from ...models.base import BoundingBox as BaseBoundingBox
from ...web.element_classifier import ElementClassifier
from ...web.models import BoundingBox as WebBoundingBox
from ...web.models import ExtractedElement
from ...web.region_detector import RegionDetector
from ..base import DetectedRegion, InteractionAction, RuntimeExtractor, StateChange

if TYPE_CHECKING:
    from ...models.base import Screenshot
    from ...web.models import ExtractedState
    from ..types import ExtractionTarget, RuntimeExtractionSession, RuntimeStateCapture

logger = logging.getLogger(__name__)


class PlaywrightExtractor(RuntimeExtractor):
    """
    Runtime extractor using Playwright for web targets.

    This extractor connects to web applications via Playwright and extracts
    their GUI state, reusing the existing element classifier and region detector
    components from the web extraction module.
    """

    def __init__(self):
        """Initialize the Playwright extractor."""
        super().__init__()
        self.playwright = None
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None

        # Reuse existing web extraction components
        self.element_classifier = ElementClassifier()
        self.region_detector = RegionDetector()

        # State
        self._screenshot_counter = 0
        self._capture_counter = 0
        self.is_connected = False
        self.session: RuntimeExtractionSession | None = None

    async def connect(self, target: ExtractionTarget) -> None:
        """
        Connect to web target via Playwright.

        Args:
            target: ExtractionTarget with URL and configuration.

        Raises:
            ConnectionError: If unable to connect to the target.
        """
        if not target.url:
            raise ValueError("ExtractionTarget must have a URL for PlaywrightExtractor")

        try:
            # Launch Playwright
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=target.headless,
            )

            # Create context with viewport
            self.context = await self.browser.new_context(
                viewport={"width": target.viewport[0], "height": target.viewport[1]},
            )

            # Set cookies if provided
            if target.auth_cookies:
                from typing import cast

                from playwright._impl._api_structures import SetCookieParam

                cookies = [
                    {"name": name, "value": value, "url": target.url}
                    for name, value in target.auth_cookies.items()
                ]
                await self.context.add_cookies(cast(list[SetCookieParam], cookies))

            # Create page
            self.page = await self.context.new_page()

            # Navigate to URL
            await self.page.goto(target.url, wait_until="networkidle")
            await self.wait_for_stability()

            self.is_connected = True
            logger.info(f"Connected to {target.url}")

        except Exception as e:
            logger.error(f"Failed to connect to {target.url}: {e}")
            await self._cleanup()
            raise ConnectionError(f"Failed to connect to {target.url}") from e

    async def disconnect(self) -> None:
        """Disconnect from web target and cleanup resources."""
        await self._cleanup()
        self.is_connected = False
        logger.info("Disconnected from web target")

    async def _cleanup(self) -> None:
        """Clean up Playwright resources."""
        if self.page:
            await self.page.close()
            self.page = None

        if self.context:
            await self.context.close()
            self.context = None

        if self.browser:
            await self.browser.close()
            self.browser = None

        if self.playwright:
            await self.playwright.stop()
            self.playwright = None

    async def extract_current_state(self) -> RuntimeStateCapture:
        """
        Extract current page state.

        Returns:
            RuntimeStateCapture with all extracted elements and states.

        Raises:
            RuntimeError: If not connected or extraction fails.
        """
        if not self.is_connected or not self.page:
            raise RuntimeError("Not connected to target")

        try:
            self._capture_counter += 1
            capture_id = f"capture_{self._capture_counter:04d}"

            # Extract elements
            elements = await self.extract_elements()

            # Detect regions
            regions = await self.detect_regions()

            # Convert regions to states
            states = await self._regions_to_states(regions, elements)

            # Capture screenshot
            screenshot_path = None
            if self.session and self.session.storage_dir:
                screenshot = await self.capture_screenshot()
                screenshot_path = screenshot.path

            # Get current URL and title
            url = self.page.url
            title = await self.page.title()

            # Get scroll position
            scroll_position = await self.page.evaluate(
                "() => ({ x: window.scrollX, y: window.scrollY })"
            )

            # Create RuntimeStateCapture
            from ..types import RuntimeStateCapture

            capture = RuntimeStateCapture(
                capture_id=capture_id,
                timestamp=datetime.now(),
                elements=elements,
                states=states,
                screenshot_path=screenshot_path,
                url=url,
                title=title,
                viewport=(
                    (
                        self.page.viewport_size["width"],
                        self.page.viewport_size["height"],
                    )
                    if self.page.viewport_size
                    else (1920, 1080)
                ),
                scroll_position=(
                    int(scroll_position["x"]),
                    int(scroll_position["y"]),
                ),
            )

            # Add to session
            self.add_capture(capture)

            logger.info(f"Extracted state: {len(elements)} elements, {len(states)} states")
            return capture

        except Exception as e:
            logger.error(f"Failed to extract current state: {e}")
            raise RuntimeError("Failed to extract current state") from e

    async def extract_elements(self) -> list[ExtractedElement]:
        """
        Extract all interactive elements from the current page.

        Returns:
            List of ExtractedElement objects.

        Raises:
            ExtractionError: If unable to extract elements.
        """
        if not self.page:
            raise RuntimeError("Not connected to target")

        try:
            elements = await self.element_classifier.extract_all_elements(
                self.page,
                min_size=(10, 10),
                max_size=(2000, 2000),
                include_hidden=False,
            )
            return list(elements)

        except Exception as e:
            logger.error(f"Failed to extract elements: {e}")
            raise RuntimeError("Failed to extract elements") from e

    async def detect_regions(self) -> list[DetectedRegion]:
        """
        Detect UI regions (navigation, modals, sidebars, etc.).

        Returns:
            List of DetectedRegion objects.

        Raises:
            ExtractionError: If unable to detect regions.
        """
        if not self.page:
            raise RuntimeError("Not connected to target")

        try:
            web_regions = await self.region_detector.detect_regions(
                self.page,
                min_size=(50, 50),
            )
            # Convert web.region_detector.DetectedRegion to runtime.base.DetectedRegion
            regions = [
                DetectedRegion(
                    id=r.id,
                    name=r.aria_label or r.region_type.value,
                    bbox=BaseBoundingBox(
                        x=r.bbox.x, y=r.bbox.y, width=r.bbox.width, height=r.bbox.height
                    ),
                    region_type=r.region_type.value,
                    confidence=r.confidence,
                    metadata=r.metadata,
                )
                for r in web_regions
            ]
            return regions

        except Exception as e:
            logger.error(f"Failed to detect regions: {e}")
            raise RuntimeError("Failed to detect regions") from e

    async def _regions_to_states(
        self,
        regions: list[DetectedRegion],
        elements: list[ExtractedElement],
    ) -> list[ExtractedState]:
        """Convert detected regions to extracted states."""
        from ...web.models import ExtractedState, StateType

        states: list[ExtractedState] = []

        for region in regions:
            # Find elements contained in this region
            contained_elements = [
                e.id
                for e in elements
                if region.bbox.contains(
                    BaseBoundingBox(
                        x=e.bbox.x, y=e.bbox.y, width=e.bbox.width, height=e.bbox.height
                    )
                )
            ]

            # Generate name from region
            name = region.name or f"{region.region_type.replace('_', ' ').title()}"

            # Generate screenshot ID (placeholder for now)
            screenshot_id = f"screenshot_{self._screenshot_counter:04d}"

            # Convert region_type string back to StateType enum
            try:
                state_type = StateType(region.region_type)
            except ValueError:
                state_type = StateType.UNKNOWN

            state = ExtractedState(
                id=region.id.replace("region_", "state_"),
                name=name,
                bbox=WebBoundingBox(
                    x=region.bbox.x,
                    y=region.bbox.y,
                    width=region.bbox.width,
                    height=region.bbox.height,
                ),
                state_type=state_type,
                element_ids=contained_elements,
                screenshot_id=screenshot_id,
                detection_method="semantic",
                confidence=region.confidence,
                semantic_role=None,
                aria_label=None,
                source_url=self.page.url if self.page else "",
                metadata=region.metadata,
            )
            states.append(state)

        return states

    async def capture_screenshot(self, region: BaseBoundingBox | None = None) -> Screenshot:
        """
        Capture a screenshot of the current state.

        Args:
            region: Optional bounding box to capture only a specific region.
                   If None, captures the entire visible viewport.

        Returns:
            Screenshot object containing the image data and metadata.

        Raises:
            ScreenshotError: If unable to capture the screenshot.
        """
        from ...models.base import Screenshot, Viewport

        if not self.page:
            raise RuntimeError("Not connected to target")

        try:
            # Generate path
            if self.session and self.session.storage_dir:
                screenshots_dir = self.session.storage_dir / "screenshots"
                screenshots_dir.mkdir(exist_ok=True, parents=True)
                self._screenshot_counter += 1
                screenshot_id = f"{self._screenshot_counter:04d}"
                path = screenshots_dir / f"{screenshot_id}.png"
            else:
                # Fallback to temp directory
                import tempfile

                temp_dir = Path(tempfile.gettempdir()) / "qontinui_screenshots"
                temp_dir.mkdir(exist_ok=True, parents=True)
                self._screenshot_counter += 1
                screenshot_id = f"{self._screenshot_counter:04d}"
                path = temp_dir / f"{screenshot_id}.png"

            # Capture screenshot (optionally clipped to region)
            if region:
                await self.page.screenshot(
                    path=str(path),
                    clip={
                        "x": region.x,
                        "y": region.y,
                        "width": region.width,
                        "height": region.height,
                    },
                )
                viewport = Viewport(width=region.width, height=region.height)
            else:
                await self.page.screenshot(path=str(path), full_page=False)
                viewport_size = self.page.viewport_size
                if viewport_size:
                    viewport = Viewport(
                        width=viewport_size["width"], height=viewport_size["height"]
                    )
                else:
                    viewport = Viewport(width=1920, height=1080)

            # Create Screenshot object
            screenshot = Screenshot(
                id=screenshot_id,
                path=path,
                viewport=viewport,
                timestamp=datetime.now().isoformat(),
            )

            logger.debug(f"Screenshot saved to {path}")
            return screenshot

        except Exception as e:
            logger.error(f"Failed to capture screenshot: {e}")
            raise RuntimeError("Failed to capture screenshot") from e

    async def navigate_to_route(self, route: str) -> None:
        """
        Navigate to a specific URL.

        Args:
            route: The URL to navigate to.

        Raises:
            NavigationError: If unable to navigate to the specified route.
        """
        if not self.page:
            raise RuntimeError("Not connected to target")

        try:
            await self.page.goto(route, wait_until="networkidle")
            await self.wait_for_stability()
            logger.info(f"Navigated to {route}")

        except Exception as e:
            logger.error(f"Failed to navigate to {route}: {e}")
            raise RuntimeError(f"Failed to navigate to {route}") from e

    async def simulate_interaction(self, action: InteractionAction) -> StateChange:
        """
        Simulate a user interaction and observe the resulting state changes.

        Args:
            action: The interaction to simulate.

        Returns:
            StateChange object describing what changed.

        Raises:
            InteractionError: If unable to perform the interaction.
        """
        if not self.page:
            raise RuntimeError("Not connected to target")

        # Capture state before action
        before_capture = await self.extract_current_state()
        before_url = self.page.url

        try:
            # Perform action based on type
            action_type = action.action_type.lower()

            if action_type == "click":
                if action.target_selector:
                    element = await self.page.query_selector(action.target_selector)
                    if element:
                        await element.click()
                    else:
                        raise ValueError(f"Element not found: {action.target_selector}")
                else:
                    raise ValueError("target_selector is required for click action")

            elif action_type in ("type", "fill"):
                if action.target_selector:
                    element = await self.page.query_selector(action.target_selector)
                    if element:
                        await element.fill(action.action_value or "")
                    else:
                        raise ValueError(f"Element not found: {action.target_selector}")
                else:
                    raise ValueError("target_selector is required for type/fill action")

            elif action_type == "hover":
                if action.target_selector:
                    element = await self.page.query_selector(action.target_selector)
                    if element:
                        await element.hover()
                    else:
                        raise ValueError(f"Element not found: {action.target_selector}")
                else:
                    raise ValueError("target_selector is required for hover action")

            elif action_type == "scroll":
                scroll_amount = int(action.action_value) if action.action_value else 100
                await self.page.evaluate(f"window.scrollBy(0, {scroll_amount})")

            else:
                raise ValueError(f"Unsupported action type: {action_type}")

            # Wait for stability
            await self.wait_for_stability()

            # Capture state after action
            after_capture = await self.extract_current_state()
            after_url = self.page.url

            # Find elements that appeared/disappeared
            before_element_ids = {e.id for e in before_capture.elements}
            after_element_ids = {e.id for e in after_capture.elements}

            appeared_elements = [
                e.id for e in after_capture.elements if e.id not in before_element_ids
            ]
            disappeared_elements = [
                e.id for e in before_capture.elements if e.id not in after_element_ids
            ]

            # Find states that appeared/disappeared
            before_state_ids = {s.id for s in before_capture.states}
            after_state_ids = {s.id for s in after_capture.states}

            appeared_states = [s.id for s in after_capture.states if s.id not in before_state_ids]
            disappeared_states = [
                s.id for s in before_capture.states if s.id not in after_state_ids
            ]

            # Create StateChange object
            state_change = StateChange(
                appeared_elements=appeared_elements,
                disappeared_elements=disappeared_elements,
                modified_elements=[],
                appeared_states=appeared_states,
                disappeared_states=disappeared_states,
                url_changed=(before_url != after_url),
                new_url=after_url if before_url != after_url else None,
                screenshot_before=(
                    str(before_capture.screenshot_path) if before_capture.screenshot_path else None
                ),
                screenshot_after=(
                    str(after_capture.screenshot_path) if after_capture.screenshot_path else None
                ),
                metadata={
                    "action": {
                        "action_type": action.action_type,
                        "target_selector": action.target_selector,
                        "target_element_id": action.target_element_id,
                        "action_value": action.action_value,
                    }
                },
            )

            return state_change

        except Exception as e:
            logger.error(f"Failed to simulate interaction: {e}")
            raise RuntimeError("Failed to simulate interaction") from e

    async def perform_action(
        self,
        action_type: str,
        target_selector: str,
        value: str | None = None,
    ) -> RuntimeStateCapture:
        """
        Perform an action on the application and capture the resulting state.

        Args:
            action_type: Type of action (click, type, hover, etc.).
            target_selector: Selector for the target element.
            value: Optional value for the action (e.g., text to type).

        Returns:
            RuntimeStateCapture after the action completes.

        Raises:
            RuntimeError: If action fails.
        """
        if not self.page:
            raise RuntimeError("Not connected to target")

        try:
            # Perform action based on type
            action_type_lower = action_type.lower()

            if action_type_lower == "click":
                element = await self.page.query_selector(target_selector)
                if element:
                    await element.click()
                else:
                    raise ValueError(f"Element not found: {target_selector}")

            elif action_type_lower in ("type", "fill"):
                element = await self.page.query_selector(target_selector)
                if element:
                    await element.fill(value or "")
                else:
                    raise ValueError(f"Element not found: {target_selector}")

            elif action_type_lower == "hover":
                element = await self.page.query_selector(target_selector)
                if element:
                    await element.hover()
                else:
                    raise ValueError(f"Element not found: {target_selector}")

            elif action_type_lower == "scroll":
                scroll_amount = int(value) if value else 100
                await self.page.evaluate(f"window.scrollBy(0, {scroll_amount})")

            else:
                raise ValueError(f"Unsupported action type: {action_type}")

            # Wait for stability
            await self.wait_for_stability()

            # Capture and return new state
            return await self.extract_current_state()

        except Exception as e:
            logger.error(f"Failed to perform action {action_type}: {e}")
            raise RuntimeError(f"Failed to perform action {action_type}") from e

    async def wait_for_stability(self, timeout_ms: int = 500) -> None:
        """
        Wait for the UI to stabilize after an action.

        Args:
            timeout_ms: Maximum time to wait in milliseconds.
        """
        if self.page:
            await self.page.wait_for_timeout(timeout_ms)

    def add_capture(self, capture: RuntimeStateCapture) -> None:
        """
        Add a capture to the current session.

        Args:
            capture: The RuntimeStateCapture to add.
        """
        if self.session:
            self.session.captures.append(capture)

    async def extract(self, target: ExtractionTarget, config: Any = None) -> Any:
        """
        Extract state from the target application (orchestrator API).

        This is the main entry point called by the orchestrator. It wraps
        the lower-level extraction methods.

        Args:
            target: ExtractionTarget specifying what to extract from.
            config: Optional ExtractionConfig with extraction settings.

        Returns:
            RuntimeExtractionResult containing all extracted states, elements, etc.

        Raises:
            RuntimeError: If extraction fails.
        """
        import uuid
        from pathlib import Path

        from ...models.base import RuntimeExtractionResult
        from ..types import RuntimeExtractionSession

        try:
            # Create session
            session_id = str(uuid.uuid4())
            storage_dir = Path.cwd() / ".qontinui" / "extraction" / session_id
            storage_dir.mkdir(parents=True, exist_ok=True)

            self.session = RuntimeExtractionSession(
                session_id=session_id,
                target=target,
                storage_dir=storage_dir,
                captures=[],
            )

            # Connect to target
            await self.connect(target)

            # Extract current state
            capture = await self.extract_current_state()

            # Disconnect
            await self.disconnect()

            # Convert to RuntimeExtractionResult
            result = RuntimeExtractionResult(
                elements=capture.elements,
                states=capture.states,
                transitions=[],
                screenshots=[str(capture.screenshot_path)] if capture.screenshot_path else [],
                pages_visited=1,
                extraction_duration_ms=0.0,
                errors=[],
            )

            logger.info(
                f"Extraction complete: {len(result.elements)} elements, "
                f"{len(result.states)} states"
            )

            return result

        except Exception as e:
            logger.error(f"Extraction failed: {e}", exc_info=True)
            # Make sure we disconnect even on error
            if self.is_connected:
                await self.disconnect()
            raise RuntimeError(f"Extraction failed: {e}") from e

    @classmethod
    def supports_target(cls, target: ExtractionTarget) -> bool:
        """
        Check if this extractor can handle the given target.

        Args:
            target: Target to check.

        Returns:
            True if this extractor can handle the target.
        """
        from ..types import RuntimeType

        # Check if target has runtime_type (from runtime/types.py ExtractionTarget)
        if hasattr(target, "runtime_type"):
            return (
                target.runtime_type == RuntimeType.WEB
                and target.url is not None
                and target.url.startswith(("http://", "https://"))
            )

        # Also support ExtractionTarget from models/base.py which has url but no runtime_type
        # If url is set and looks like a web URL, we can handle it
        if hasattr(target, "url") and target.url is not None:
            return target.url.startswith(("http://", "https://"))

        return False
