"""
Web extractor orchestrator.

Coordinates element classification, region detection, and visibility
tracking to extract GUI elements and states from web applications.
"""

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from .config import ExtractionConfig
from .element_classifier import ElementClassifier
from .models import (
    BoundingBox,
    ExtractedElement,
    ExtractedState,
    ExtractedTransition,
    ExtractionResult,
    PageExtraction,
    StateType,
    TransitionType,
)
from .region_detector import DetectedRegion, RegionDetector
from .visibility_tracker import VisibilityTracker

logger = logging.getLogger(__name__)


# Type for progress callback
ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]] | Callable[[dict[str, Any]], None]


class WebExtractor:
    """
    Extracts GUI elements and states from web applications.

    Uses Playwright to browse web pages and extract:
    - Interactive elements (buttons, inputs, links, etc.)
    - UI regions/states (menus, dialogs, navigation, etc.)
    - Transitions between states (click, hover, etc.)
    """

    def __init__(
        self,
        config: ExtractionConfig,
        storage_dir: Path | None = None,
    ):
        """
        Initialize web extractor.

        Args:
            config: Extraction configuration.
            storage_dir: Directory for storing screenshots. Defaults to
                        ~/.qontinui/extraction/{extraction_id}/
        """
        self.config = config
        self.extraction_id = str(uuid.uuid4())

        # Set up storage directory
        if storage_dir:
            self.storage_dir = storage_dir
        elif config.output_dir:
            self.storage_dir = Path(config.output_dir)
        else:
            self.storage_dir = Path.home() / ".qontinui" / "extraction" / self.extraction_id
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir = self.storage_dir / "screenshots"
        self.screenshots_dir.mkdir(exist_ok=True)

        # Components
        self.element_classifier = ElementClassifier()
        self.region_detector = RegionDetector()
        self.visibility_tracker = VisibilityTracker(
            correlation_threshold=config.visibility_threshold
        )

        # State
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None
        self._is_running = False
        self._screenshot_counter = 0

        # Results
        self.result = ExtractionResult(
            extraction_id=self.extraction_id,
            source_urls=config.urls,
            viewports=config.viewports,
            config=config.to_dict(),
        )

        # Progress tracking
        self._progress_callback: ProgressCallback | None = None
        self._visited_urls: set[str] = set()
        self._pages_extracted = 0

    async def start(
        self,
        progress_callback: ProgressCallback | None = None,
    ) -> ExtractionResult:
        """
        Start the extraction process.

        Args:
            progress_callback: Optional callback for progress updates.
                             Called with dict containing status, stats, etc.

        Returns:
            ExtractionResult with all extracted data.
        """
        self._progress_callback = progress_callback
        self._is_running = True

        try:
            async with async_playwright() as playwright:
                # Launch browser
                self.browser = await playwright.chromium.launch(
                    headless=True,
                )

                # Process each URL
                for url in self.config.urls:
                    if not self._is_running:
                        break
                    await self._extract_url(url)

                # Complete
                self.result.completed_at = datetime.now()
                await self._emit_progress(
                    {
                        "status": "complete",
                        "extraction_id": self.extraction_id,
                        "summary": {
                            "total_pages": self._pages_extracted,
                            "total_states": len(self.result.states),
                            "total_elements": len(self.result.elements),
                            "total_transitions": len(self.result.transitions),
                        },
                    }
                )

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            await self._emit_progress(
                {
                    "status": "error",
                    "error": str(e),
                }
            )
            raise

        finally:
            self._is_running = False
            if self.browser:
                await self.browser.close()
                self.browser = None

        return self.result

    async def stop(self) -> None:
        """Stop the extraction process."""
        self._is_running = False

    async def _extract_url(self, url: str) -> None:
        """Extract from a single URL and its linked pages."""
        if not self.browser:
            return

        # Process each viewport size
        for viewport in self.config.viewports:
            if not self._is_running:
                break

            # Create context with viewport
            self.context = await self.browser.new_context(
                viewport={"width": viewport[0], "height": viewport[1]},
            )

            # Set cookies if configured
            if self.config.auth_cookies:
                cookies = [
                    {"name": name, "value": value, "url": url}
                    for name, value in self.config.auth_cookies.items()
                ]
                await self.context.add_cookies(cookies)

            try:
                self.page = await self.context.new_page()

                # Navigate to URL
                await self.page.goto(url, wait_until="networkidle")
                await self._wait_for_stability()

                # Extract current page (visible viewport)
                await self._extract_current_page(viewport)

                # Extract scroll-based states for long pages
                if self.config.capture_scroll_states:
                    await self._extract_scroll_states(viewport)

                # Discover transitions if enabled
                if self.config.capture_hover_states or self.config.capture_focus_states:
                    await self._discover_transitions(viewport)

            finally:
                await self.context.close()
                self.context = None
                self.page = None

    async def _extract_current_page(self, viewport: tuple[int, int]) -> PageExtraction:
        """Extract elements and regions from the current page."""
        if not self.page:
            raise RuntimeError("No page available")

        url = self.page.url
        title = await self.page.title()

        await self._emit_progress(
            {
                "status": "running",
                "current_url": url,
                "viewport": viewport,
            }
        )

        # Capture screenshot
        screenshot_id = await self._capture_screenshot()

        # Extract elements
        elements = await self.element_classifier.extract_all_elements(
            self.page,
            min_size=self.config.min_element_size,
            max_size=self.config.max_element_size,
            include_hidden=self.config.include_hidden_elements,
        )

        # Detect regions
        regions = await self.region_detector.detect_regions(
            self.page,
            min_size=self.config.min_state_size,
        )

        # Convert regions to states
        states = await self._regions_to_states(regions, elements, screenshot_id, url)

        # Record visibility snapshot
        element_selectors = {e.id: e.selector for e in elements}
        await self.visibility_tracker.record_snapshot(
            self.page,
            element_selectors,
            trigger_action=f"navigate:{url}",
        )

        # Create page extraction
        page_extraction = PageExtraction(
            url=url,
            title=title,
            viewport=viewport,
            elements=elements,
            states=states,
            screenshot_ids=[screenshot_id],
        )

        # Add to results
        self.result.elements.extend(elements)
        self.result.states.extend(states)
        self.result.page_extractions.append(page_extraction)
        self.result.screenshot_ids.append(screenshot_id)

        self._pages_extracted += 1
        self._visited_urls.add(url)

        await self._emit_progress(
            {
                "status": "running",
                "pages_visited": self._pages_extracted,
                "states_found": len(self.result.states),
                "elements_found": len(self.result.elements),
            }
        )

        # Emit detected states and elements
        for state in states:
            thumbnail = await self._get_thumbnail(screenshot_id)
            await self._emit_progress(
                {
                    "type": "state_detected",
                    "state": state.to_dict(),
                    "thumbnail": thumbnail,
                }
            )

        return page_extraction

    async def _regions_to_states(
        self,
        regions: list[DetectedRegion],
        elements: list[ExtractedElement],
        screenshot_id: str,
        source_url: str,
    ) -> list[ExtractedState]:
        """Convert detected regions to extracted states."""
        states: list[ExtractedState] = []

        for region in regions:
            # Find elements contained in this region
            contained_elements = [e.id for e in elements if region.bbox.contains(e.bbox)]

            # Generate name from region type and aria label
            name = region.aria_label or f"{region.region_type.value.replace('_', ' ').title()}"

            state = ExtractedState(
                id=region.id.replace("region_", "state_"),
                name=name,
                bbox=region.bbox,
                state_type=region.region_type,
                element_ids=contained_elements,
                screenshot_id=screenshot_id,
                detection_method="semantic",
                confidence=region.confidence,
                semantic_role=region.semantic_role,
                aria_label=region.aria_label,
                source_url=source_url,
                metadata=region.metadata,
            )
            states.append(state)

        return states

    async def _extract_scroll_states(self, viewport: tuple[int, int]) -> None:
        """
        Extract scroll-based states for pages longer than the viewport.

        Creates separate states for each scroll position, with transitions
        representing the scroll action between them.
        """
        if not self.page:
            return

        url = self.page.url

        # Get page dimensions
        page_height = await self.page.evaluate("document.documentElement.scrollHeight")
        viewport_height = viewport[1]

        # Check if page is scrollable
        if page_height <= viewport_height:
            logger.debug(f"Page {url} fits in viewport, no scroll states needed")
            return

        logger.info(f"Extracting scroll states for {url} (height: {page_height}px)")

        # Calculate scroll step (viewport height minus overlap)
        overlap_pixels = int(viewport_height * self.config.scroll_overlap_percent)
        scroll_step = viewport_height - overlap_pixels

        # Track scroll positions and their states
        scroll_positions: list[dict] = []
        scroll_state_counter = 0

        # Ensure we start at the top
        await self.page.evaluate("window.scrollTo(0, 0)")
        await self._wait_for_stability()

        # Create state for initial viewport (scroll position 0)
        initial_screenshot_id = await self._capture_screenshot()
        initial_elements = await self.element_classifier.extract_all_elements(
            self.page,
            min_size=self.config.min_element_size,
            max_size=self.config.max_element_size,
            include_hidden=self.config.include_hidden_elements,
        )

        initial_state = ExtractedState(
            id="scroll_state_000",
            name="Scroll Position 0 (Top)",
            bbox=BoundingBox(
                x=0,
                y=0,
                width=viewport[0],
                height=viewport_height,
            ),
            state_type=StateType.CONTENT,
            element_ids=[e.id for e in initial_elements],
            screenshot_id=initial_screenshot_id,
            detection_method="scroll",
            confidence=1.0,
            source_url=url,
            metadata={
                "scroll_y": 0,
                "scroll_index": 0,
                "page_height": page_height,
                "viewport_height": viewport_height,
            },
        )

        self.result.states.append(initial_state)
        self.result.elements.extend(initial_elements)
        self.result.screenshot_ids.append(initial_screenshot_id)

        scroll_positions.append(
            {
                "scroll_y": 0,
                "state_id": initial_state.id,
                "screenshot_id": initial_screenshot_id,
            }
        )

        await self._emit_progress(
            {
                "type": "state_detected",
                "state": initial_state.to_dict(),
                "thumbnail": await self._get_thumbnail(initial_screenshot_id),
            }
        )

        # Get current scroll position
        current_scroll = 0

        # Iterate through remaining scroll positions
        while current_scroll < page_height - viewport_height:
            scroll_state_counter += 1

            # Scroll to next position
            next_scroll = min(current_scroll + scroll_step, page_height - viewport_height)
            await self.page.evaluate(f"window.scrollTo(0, {next_scroll})")
            await self._wait_for_stability()

            # Verify scroll happened
            actual_scroll = await self.page.evaluate("window.scrollY")

            # Capture screenshot at this scroll position
            screenshot_id = await self._capture_screenshot()

            # Extract elements visible at this scroll position
            elements = await self.element_classifier.extract_all_elements(
                self.page,
                min_size=self.config.min_element_size,
                max_size=self.config.max_element_size,
                include_hidden=self.config.include_hidden_elements,
            )

            # Detect regions at this scroll position
            await self.region_detector.detect_regions(
                self.page,
                min_size=self.config.min_state_size,
            )

            # Create a scroll-based state representing this viewport position
            scroll_state = ExtractedState(
                id=f"scroll_state_{scroll_state_counter:03d}",
                name=f"Scroll Position {scroll_state_counter} ({int(actual_scroll)}px)",
                bbox=BoundingBox(
                    x=0,
                    y=int(actual_scroll),
                    width=viewport[0],
                    height=viewport_height,
                ),
                state_type=StateType.CONTENT,
                element_ids=[e.id for e in elements],
                screenshot_id=screenshot_id,
                detection_method="scroll",
                confidence=1.0,
                source_url=url,
                metadata={
                    "scroll_y": int(actual_scroll),
                    "scroll_index": scroll_state_counter,
                    "page_height": page_height,
                    "viewport_height": viewport_height,
                },
            )

            self.result.states.append(scroll_state)
            self.result.elements.extend(elements)
            self.result.screenshot_ids.append(screenshot_id)

            # Create scroll transition from previous state
            previous_pos = scroll_positions[-1]
            scroll_distance = int(actual_scroll - previous_pos["scroll_y"])
            transition = ExtractedTransition(
                id=f"scroll_trans_{len(self.result.transitions):04d}",
                action_type=TransitionType.SCROLL,
                target_element_id="",  # No specific element for scroll
                target_selector="window",
                causes_appear=[scroll_state.id],
                causes_disappear=[previous_pos["state_id"]],
                action_value=str(scroll_distance),
                metadata={
                    "scroll_from": previous_pos["scroll_y"],
                    "scroll_to": int(actual_scroll),
                    "scroll_distance": scroll_distance,
                    "direction": "down",
                },
            )
            self.result.transitions.append(transition)

            # Track this scroll position
            scroll_positions.append(
                {
                    "scroll_y": int(actual_scroll),
                    "state_id": scroll_state.id,
                    "screenshot_id": screenshot_id,
                }
            )

            # Emit progress
            await self._emit_progress(
                {
                    "type": "state_detected",
                    "state": scroll_state.to_dict(),
                    "thumbnail": await self._get_thumbnail(screenshot_id),
                }
            )

            current_scroll = actual_scroll

            # Safety check to prevent infinite loops
            if scroll_state_counter > 100:
                logger.warning(f"Reached max scroll states (100) for {url}")
                break

        # Also create reverse scroll transitions (scroll up)
        for i in range(len(scroll_positions) - 1, 0, -1):
            current_pos = scroll_positions[i]
            prev_pos = scroll_positions[i - 1]
            scroll_distance = prev_pos["scroll_y"] - current_pos["scroll_y"]

            transition = ExtractedTransition(
                id=f"scroll_trans_{len(self.result.transitions):04d}",
                action_type=TransitionType.SCROLL,
                target_element_id="",
                target_selector="window",
                causes_appear=[prev_pos["state_id"]],
                causes_disappear=[current_pos["state_id"]],
                action_value=str(scroll_distance),
                metadata={
                    "scroll_from": current_pos["scroll_y"],
                    "scroll_to": prev_pos["scroll_y"],
                    "scroll_distance": scroll_distance,
                    "direction": "up",
                },
            )
            self.result.transitions.append(transition)

        # Scroll back to top
        await self.page.evaluate("window.scrollTo(0, 0)")
        await self._wait_for_stability()

        total_scroll_states = len(scroll_positions)  # Includes initial state
        total_scroll_transitions = (total_scroll_states - 1) * 2  # Up and down for each pair
        logger.info(
            f"Extracted {total_scroll_states} scroll states and "
            f"{total_scroll_transitions} scroll transitions for {url}"
        )

    async def _discover_transitions(self, viewport: tuple[int, int]) -> None:
        """Discover transitions by interacting with elements."""
        if not self.page:
            return

        interactive_elements = [
            e for e in self.result.elements if e.is_interactive and e.is_enabled
        ]

        for element in interactive_elements[:50]:  # Limit to prevent infinite loops
            if not self._is_running:
                break

            try:
                # Get element handle
                el = await self.page.query_selector(element.selector)
                if not el:
                    continue

                # Record state before action
                before_selectors = {e.id: e.selector for e in self.result.elements}
                await self.visibility_tracker.record_snapshot(
                    self.page, before_selectors, trigger_action="before"
                )

                # Try hover if enabled
                if self.config.capture_hover_states:
                    await self._try_hover_transition(element, el, viewport)

                # Try focus if enabled
                if self.config.capture_focus_states:
                    await self._try_focus_transition(element, el, viewport)

            except Exception as e:
                logger.debug(f"Error discovering transitions for {element.id}: {e}")

    async def _try_hover_transition(
        self,
        element: ExtractedElement,
        el,
        viewport: tuple[int, int],
    ) -> None:
        """Try hover action and detect any state changes."""
        if not self.page:
            return

        try:
            # Hover over element
            await el.hover()
            await self._wait_for_stability()

            # Check for new regions
            new_regions = await self.region_detector.detect_regions(self.page)

            # Find regions that weren't there before
            existing_state_selectors = {
                s.metadata.get("matched_selector") for s in self.result.states
            }
            new_region_ids = []

            for region in new_regions:
                if region.metadata.get("matched_selector") not in existing_state_selectors:
                    # This is a new region - likely a dropdown/tooltip
                    screenshot_id = await self._capture_screenshot()
                    states = await self._regions_to_states(
                        [region], [], screenshot_id, self.page.url
                    )
                    for state in states:
                        self.result.states.append(state)
                        new_region_ids.append(state.id)

            # Record transition if new regions appeared
            if new_region_ids:
                transition = ExtractedTransition(
                    id=f"trans_{len(self.result.transitions):04d}",
                    action_type=TransitionType.HOVER,
                    target_element_id=element.id,
                    target_selector=element.selector,
                    causes_appear=new_region_ids,
                    causes_disappear=[],
                )
                self.result.transitions.append(transition)

            # Move mouse away to reset state
            await self.page.mouse.move(0, 0)
            await self._wait_for_stability()

        except Exception as e:
            logger.debug(f"Hover transition error: {e}")

    async def _try_focus_transition(
        self,
        element: ExtractedElement,
        el,
        viewport: tuple[int, int],
    ) -> None:
        """Try focus action and detect any state changes."""
        if not self.page:
            return

        try:
            # Focus element
            await el.focus()
            await self._wait_for_stability()

            # Record visibility snapshot
            element_selectors = {e.id: e.selector for e in self.result.elements}
            await self.visibility_tracker.record_snapshot(
                self.page,
                element_selectors,
                trigger_action=f"focus:{element.id}",
            )

            # Blur to reset
            await self.page.evaluate("document.activeElement?.blur()")

        except Exception as e:
            logger.debug(f"Focus transition error: {e}")

    async def _capture_screenshot(self, region: BoundingBox | None = None) -> str:
        """Capture a screenshot and save it locally."""
        if not self.page:
            raise RuntimeError("No page available")

        self._screenshot_counter += 1
        screenshot_id = f"{self._screenshot_counter:04d}"
        filename = f"{screenshot_id}.png"
        filepath = self.screenshots_dir / filename

        if region:
            await self.page.screenshot(
                path=str(filepath),
                clip={
                    "x": region.x,
                    "y": region.y,
                    "width": region.width,
                    "height": region.height,
                },
            )
        else:
            await self.page.screenshot(path=str(filepath), full_page=False)

        return screenshot_id

    async def _get_thumbnail(self, screenshot_id: str) -> str:
        """Get base64 thumbnail of a screenshot."""
        import base64
        import io

        from PIL import Image

        filepath = self.screenshots_dir / f"{screenshot_id}.png"
        if not filepath.exists():
            return ""

        try:
            with Image.open(filepath) as img:
                # Resize to thumbnail
                img.thumbnail(self.config.thumbnail_size)

                # Convert to base64
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                return base64.b64encode(buffer.getvalue()).decode()
        except Exception as e:
            logger.warning(f"Error creating thumbnail: {e}")
            return ""

    async def _wait_for_stability(self) -> None:
        """Wait for page to stabilize after an action."""
        if not self.page:
            return

        if self.config.wait_for_animations:
            await self.page.wait_for_timeout(self.config.animation_timeout_ms)

    async def _emit_progress(self, data: dict[str, Any]) -> None:
        """Emit progress update to callback."""
        if self._progress_callback:
            try:
                result = self._progress_callback(data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    def get_results(self) -> ExtractionResult:
        """Get current extraction results."""
        return self.result

    def get_screenshot_path(self, screenshot_id: str) -> Path | None:
        """Get the path to a screenshot file."""
        path = self.screenshots_dir / f"{screenshot_id}.png"
        return path if path.exists() else None

    @property
    def is_running(self) -> bool:
        """Check if extraction is currently running."""
        return self._is_running
