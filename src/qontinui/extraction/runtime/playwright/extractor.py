"""
PlaywrightExtractor for runtime web extraction.

Implements RuntimeExtractor using Playwright for web targets.
Extracts interactive elements (buttons, links, inputs) from web pages.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)
from qontinui_schemas.common import utc_now

from ...models.base import BoundingBox as BaseBoundingBox
from ...web.interactive_element_extractor import InteractiveElementExtractor
from ...web.models import BoundingBox as WebBoundingBox
from ...web.models import (
    ExtractedElement,
    ExtractedState,
    InteractiveElement,
    StateType,
)
from ..base import DetectedRegion, InteractionAction, RuntimeExtractor, StateChange

if TYPE_CHECKING:
    from ...models.base import Screenshot
    from ..types import ExtractionTarget, RuntimeExtractionSession, RuntimeStateCapture

logger = logging.getLogger(__name__)


class PlaywrightExtractor(RuntimeExtractor):
    """
    Runtime extractor using Playwright for web targets.

    Extracts interactive elements (buttons, links, form inputs) from web pages
    using DOM-based analysis.
    """

    def __init__(self) -> None:
        """Initialize the Playwright extractor."""
        super().__init__()
        self.playwright: Playwright | None = None
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None

        # Interactive element extractor (DOM-based)
        self.element_extractor = InteractiveElementExtractor()

        # State
        self._screenshot_counter = 0
        self._capture_counter = 0
        self.is_connected = False
        self.session: RuntimeExtractionSession | None = None

        # Configuration
        self._configured_viewport: tuple[int, int] | None = None

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
            headless = getattr(target, "headless", True)

            if hasattr(self, "_configured_viewport") and self._configured_viewport:
                viewport = self._configured_viewport
            else:
                viewport = getattr(target, "viewport", (1920, 1080))

            logger.info(f"Connecting with headless={headless}, viewport={viewport}")

            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(headless=headless)

            self.context = await self.browser.new_context(
                viewport={"width": viewport[0], "height": viewport[1]},
            )

            if target.auth_cookies:
                from typing import cast

                from playwright._impl._api_structures import SetCookieParam

                cookies = [
                    {"name": name, "value": value, "url": target.url}
                    for name, value in target.auth_cookies.items()
                ]
                await self.context.add_cookies(cast(list[SetCookieParam], cookies))

            self.page = await self.context.new_page()
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
        """
        if not self.is_connected or not self.page:
            raise RuntimeError("Not connected to target")

        try:
            self._capture_counter += 1
            capture_id = f"capture_{self._capture_counter:04d}"
            logger.info(f"Extracting state: {capture_id}")

            # Capture screenshot first
            screenshot_path = None
            screenshot_id = None
            if self.session and self.session.storage_dir:
                screenshot = await self.capture_screenshot()
                screenshot_path = screenshot.path
                screenshot_id = screenshot.id

            # Extract interactive elements
            self.element_extractor.reset_counter()
            interactive_elements = await self.element_extractor.extract_interactive_elements(
                self.page,
                screenshot_id or f"page_{self._capture_counter:04d}",
            )

            # Convert InteractiveElement to ExtractedElement for compatibility
            elements = self._convert_to_extracted_elements(interactive_elements)

            # Get viewport
            viewport_size = self.page.viewport_size
            width = viewport_size["width"] if viewport_size else 1920
            height = viewport_size["height"] if viewport_size else 1080

            # Create a single page-level state containing all elements
            page_state = ExtractedState(
                id=f"state_page_{self._capture_counter:04d}",
                name=await self.page.title() or f"Page {self._capture_counter}",
                bbox=WebBoundingBox(x=0, y=0, width=width, height=height),
                state_type=StateType.PAGE,
                element_ids=[e.id for e in elements],
                screenshot_id=screenshot_id,
                detection_method="interactive_extraction",
                confidence=1.0,
                source_url=self.page.url,
            )
            states = [page_state]

            # Get scroll position
            scroll_position = await self.page.evaluate(
                "() => ({ x: window.scrollX, y: window.scrollY })"
            )

            # Create RuntimeStateCapture
            from ..types import RuntimeStateCapture

            capture = RuntimeStateCapture(
                capture_id=capture_id,
                timestamp=utc_now(),
                elements=elements,
                states=states,
                screenshot_path=screenshot_path,
                url=self.page.url,
                title=await self.page.title(),
                viewport=(width, height),
                scroll_position=(int(scroll_position["x"]), int(scroll_position["y"])),
            )

            self.add_capture(capture)

            logger.info(f"Extracted {len(elements)} interactive elements")
            return capture

        except Exception as e:
            logger.error(f"Failed to extract current state: {e}")
            raise RuntimeError("Failed to extract current state") from e

    def _convert_to_extracted_elements(
        self,
        interactive_elements: list[InteractiveElement],
    ) -> list[ExtractedElement]:
        """Convert InteractiveElement list to ExtractedElement list for compatibility."""

        elements = []
        for ie in interactive_elements:
            # Map element_type string to ElementType enum
            element_type = self._map_element_type(ie.element_type)

            elem = ExtractedElement(
                id=ie.id,
                bbox=ie.bbox,
                element_type=element_type,
                selector=ie.selector,
                text_content=ie.text,
                aria_label=ie.aria_label,
                semantic_role=ie.aria_role,
                is_interactive=True,
                is_visible=True,
                tag_name=ie.tag_name,
                extraction_category=ie.element_type,  # Use element_type as category
            )
            elements.append(elem)

        return elements

    def _map_element_type(self, type_str: str) -> Any:
        """Map element type string to ElementType enum."""
        from ...web.models import ElementType

        type_mapping = {
            "button": ElementType.BUTTON,
            "a": ElementType.LINK,
            "input": ElementType.TEXT_INPUT,
            "select": ElementType.DROPDOWN,
            "textarea": ElementType.TEXTAREA,
            "checkbox": ElementType.CHECKBOX,
            "radio": ElementType.RADIO,
            "label": ElementType.LABEL,
        }

        # Handle prefixed types like "aria_button", "tabindex_div"
        if type_str.startswith("aria_"):
            base = type_str.replace("aria_", "")
            return type_mapping.get(base, ElementType.UNKNOWN)

        if type_str.startswith("tabindex_") or type_str.startswith("onclick_"):
            return ElementType.BUTTON  # Clickable element

        return type_mapping.get(type_str, ElementType.UNKNOWN)

    async def extract_elements(self) -> list[ExtractedElement]:
        """Extract all interactive elements from the current page."""
        if not self.page:
            raise RuntimeError("Not connected to target")

        self.element_extractor.reset_counter()
        interactive_elements = await self.element_extractor.extract_interactive_elements(
            self.page,
            f"page_{self._capture_counter:04d}",
        )
        return self._convert_to_extracted_elements(interactive_elements)

    async def detect_regions(self) -> list[DetectedRegion]:
        """
        Detect UI regions. Returns empty list as we no longer use region detection.
        """
        return []

    async def capture_screenshot(self, region: BaseBoundingBox | None = None) -> Screenshot:
        """Capture a screenshot of the current state."""
        from ...models.base import Screenshot, Viewport

        if not self.page:
            raise RuntimeError("Not connected to target")

        try:
            if self.session and self.session.storage_dir:
                screenshots_dir = self.session.storage_dir / "screenshots"
                screenshots_dir.mkdir(exist_ok=True, parents=True)
                self._screenshot_counter += 1
                screenshot_id = f"{self._screenshot_counter:04d}"
                path = screenshots_dir / f"{screenshot_id}.png"
            else:
                import tempfile

                temp_dir = Path(tempfile.gettempdir()) / "qontinui_screenshots"
                temp_dir.mkdir(exist_ok=True, parents=True)
                self._screenshot_counter += 1
                screenshot_id = f"{self._screenshot_counter:04d}"
                path = temp_dir / f"{screenshot_id}.png"

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
                await self.page.screenshot(path=str(path), full_page=True)
                page_width = await self.page.evaluate("document.body.scrollWidth")
                page_height = await self.page.evaluate("document.body.scrollHeight")
                if page_width and page_height:
                    viewport = Viewport(width=page_width, height=page_height)
                else:
                    viewport_size = self.page.viewport_size
                    if viewport_size:
                        viewport = Viewport(
                            width=viewport_size["width"], height=viewport_size["height"]
                        )
                    else:
                        viewport = Viewport(width=1920, height=1080)

            screenshot = Screenshot(
                id=screenshot_id,
                path=path,
                viewport=viewport,
                timestamp=utc_now().isoformat(),
            )

            logger.debug(f"Screenshot saved to {path}")
            return screenshot

        except Exception as e:
            logger.error(f"Failed to capture screenshot: {e}")
            raise RuntimeError("Failed to capture screenshot") from e

    async def navigate_to_route(self, route: str) -> None:
        """Navigate to a specific URL."""
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
        """Simulate a user interaction and observe the resulting state changes."""
        if not self.page:
            raise RuntimeError("Not connected to target")

        before_capture = await self.extract_current_state()
        before_url = self.page.url

        try:
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

            await self.wait_for_stability()

            after_capture = await self.extract_current_state()
            after_url = self.page.url

            before_element_ids = {e.id for e in before_capture.elements}
            after_element_ids = {e.id for e in after_capture.elements}

            appeared_elements = [
                e.id for e in after_capture.elements if e.id not in before_element_ids
            ]
            disappeared_elements = [
                e.id for e in before_capture.elements if e.id not in after_element_ids
            ]

            before_state_ids = {s.id for s in before_capture.states}
            after_state_ids = {s.id for s in after_capture.states}

            appeared_states = [s.id for s in after_capture.states if s.id not in before_state_ids]
            disappeared_states = [
                s.id for s in before_capture.states if s.id not in after_state_ids
            ]

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
                metadata={"action": action.__dict__},
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
        """Perform an action on the application and capture the resulting state."""
        if not self.page:
            raise RuntimeError("Not connected to target")

        try:
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

            await self.wait_for_stability()
            return await self.extract_current_state()

        except Exception as e:
            logger.error(f"Failed to perform action {action_type}: {e}")
            raise RuntimeError(f"Failed to perform action {action_type}") from e

    async def wait_for_stability(self, timeout_ms: int = 500) -> None:
        """Wait for the UI to stabilize after an action."""
        if self.page:
            await self.page.wait_for_timeout(timeout_ms)

    def add_capture(self, capture: RuntimeStateCapture) -> None:
        """Add a capture to the current session."""
        if self.session:
            self.session.captures.append(capture)

    async def _discover_links(self) -> list[dict[str, Any]]:
        """Discover all internal links on the current page."""
        if not self.page:
            return []

        try:
            current_url = self.page.url
            from urllib.parse import urlparse

            parsed = urlparse(current_url)
            origin = f"{parsed.scheme}://{parsed.netloc}"

            links = await self.page.evaluate(
                """() => {
                const anchors = document.querySelectorAll('a[href]');
                const results = [];
                for (const a of anchors) {
                    const href = a.href;
                    if (href && !href.startsWith('javascript:') && !href.startsWith('#') &&
                        !href.startsWith('mailto:') && !href.startsWith('tel:')) {
                        const rect = a.getBoundingClientRect();
                        let selector = '';
                        if (a.id) {
                            selector = '#' + a.id;
                        } else if (a.className) {
                            selector = 'a.' + a.className.split(' ').join('.');
                        } else {
                            selector = `a[href="${a.getAttribute('href')}"]`;
                        }
                        results.push({
                            url: href,
                            selector: selector,
                            text: a.innerText.trim().substring(0, 100),
                            bbox: {
                                x: rect.x,
                                y: rect.y,
                                width: rect.width,
                                height: rect.height
                            }
                        });
                    }
                }
                return results;
            }"""
            )

            same_origin_links = []
            seen_urls = set()
            for link in links:
                url = link["url"]
                if url.startswith(origin):
                    normalized = url.split("#")[0].rstrip("/")
                    if normalized and normalized not in seen_urls:
                        seen_urls.add(normalized)
                        link["url"] = normalized
                        same_origin_links.append(link)

            logger.info(f"Discovered {len(same_origin_links)} same-origin links")
            return same_origin_links

        except Exception as e:
            logger.warning(f"Failed to discover links: {e}")
            return []

    async def extract(self, target: ExtractionTarget, config: Any = None) -> Any:
        """
        Extract state from the target application.

        This is the main entry point called by the orchestrator.
        """
        from pathlib import Path
        from urllib.parse import urlparse

        from ...models.base import InferredTransition, RuntimeExtractionResult
        from ..types import RuntimeExtractionSession

        logger.info("=" * 60)
        logger.info("PLAYWRIGHT EXTRACTOR: Starting extraction")
        logger.info("=" * 60)
        logger.info(f"  Target URL: {target.url}")

        max_pages = 100
        max_depth = 5
        if config is not None:
            max_pages = getattr(config, "max_pages", None) or 100
            max_depth = getattr(config, "max_interaction_depth", None) or 5
        logger.info(f"  Max pages: {max_pages}, Max depth: {max_depth}")

        viewport = (1920, 1080)
        if config is not None:
            viewports = getattr(config, "viewports", None)
            if viewports and len(viewports) > 0:
                viewport = viewports[0]
        self._configured_viewport = viewport

        try:
            session_id = str(uuid.uuid4())
            storage_dir = Path.home() / ".qontinui" / "extraction" / session_id
            storage_dir.mkdir(parents=True, exist_ok=True)

            self.session = RuntimeExtractionSession(
                session_id=session_id,
                target=target,
                storage_dir=storage_dir,
                captures=[],
            )

            await self.connect(target)

            all_elements: list[Any] = []
            all_states: list[Any] = []
            all_screenshots: list[str] = []
            all_transitions: list[Any] = []
            visited_urls: set[str] = set()
            urls_to_visit: list[tuple[str, int, str | None, dict | None]] = []

            url_to_state_id: dict[str, str] = {}

            start_url = target.url
            if start_url:
                parsed = urlparse(start_url)
                normalized_start = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
                urls_to_visit.append((normalized_start, 0, None, None))

            pages_visited = 0
            errors: list[str] = []
            transition_counter = 0

            while urls_to_visit and pages_visited < max_pages:
                current_url, current_depth, source_url, link_info = urls_to_visit.pop(0)

                normalized_current = current_url.split("#")[0].rstrip("/")
                if normalized_current in visited_urls:
                    continue

                visited_urls.add(normalized_current)

                logger.info(f"Crawling page {pages_visited + 1}/{max_pages}: {current_url}")

                try:
                    if pages_visited > 0:
                        if self.page is None:
                            raise RuntimeError("Page not available")
                        await self.page.goto(current_url, wait_until="networkidle")
                        await self.wait_for_stability()

                    capture = await self.extract_current_state()

                    all_elements.extend(capture.elements)
                    all_states.extend(capture.states)
                    if capture.screenshot_path:
                        all_screenshots.append(str(capture.screenshot_path))

                    if capture.states:
                        primary_state = capture.states[0]
                        primary_state_id = getattr(primary_state, "id", f"state_{pages_visited}")
                        url_to_state_id[normalized_current] = primary_state_id

                        if source_url and source_url in url_to_state_id:
                            source_state_id = url_to_state_id[source_url]
                            transition_counter += 1

                            trigger_selector = link_info.get("selector", "") if link_info else ""
                            trigger_text = link_info.get("text", "") if link_info else ""

                            transition = InferredTransition(
                                id=f"trans_{transition_counter:04d}",
                                from_state_id=source_state_id,
                                to_state_id=primary_state_id,
                                trigger_type="click",
                                target_element=trigger_selector,
                                confidence=0.9,
                                metadata={
                                    "source_url": source_url,
                                    "target_url": normalized_current,
                                    "trigger_text": trigger_text,
                                    "link_info": link_info,
                                },
                            )
                            all_transitions.append(transition)

                    pages_visited += 1
                    logger.info(f"  Extracted {len(capture.elements)} elements")

                    if current_depth < max_depth:
                        discovered_links = await self._discover_links()
                        for link in discovered_links:
                            link_url = link["url"]
                            normalized_link = link_url.split("#")[0].rstrip("/")
                            if normalized_link not in visited_urls:
                                if not any(
                                    url == normalized_link for url, _, _, _ in urls_to_visit
                                ):
                                    urls_to_visit.append(
                                        (
                                            normalized_link,
                                            current_depth + 1,
                                            normalized_current,
                                            link,
                                        )
                                    )

                except Exception as e:
                    error_msg = f"Failed to extract {current_url}: {e}"
                    logger.warning(error_msg)
                    errors.append(error_msg)
                    continue

            await self.disconnect()

            result = RuntimeExtractionResult(
                elements=all_elements,
                states=all_states,
                transitions=all_transitions,
                screenshots=all_screenshots,
                pages_visited=pages_visited,
                extraction_duration_ms=0.0,
                errors=errors,
                extraction_id=session_id,
            )

            logger.info("=" * 60)
            logger.info("EXTRACTION COMPLETE")
            logger.info(f"  Pages: {result.pages_visited}")
            logger.info(f"  Elements: {len(result.elements)}")
            logger.info(f"  States: {len(result.states)}")
            logger.info(f"  Transitions: {len(result.transitions)}")
            logger.info("=" * 60)

            return result

        except Exception as e:
            logger.error(f"Extraction failed: {e}", exc_info=True)
            if self.is_connected:
                await self.disconnect()
            raise RuntimeError(f"Extraction failed: {e}") from e

    @classmethod
    def supports_target(cls, target: ExtractionTarget) -> bool:
        """Check if this extractor can handle the given target."""
        from ..types import RuntimeType

        if hasattr(target, "runtime_type"):
            return (
                target.runtime_type == RuntimeType.WEB
                and target.url is not None
                and target.url.startswith(("http://", "https://"))
            )

        if hasattr(target, "url") and target.url is not None:
            return target.url.startswith(("http://", "https://"))

        return False
