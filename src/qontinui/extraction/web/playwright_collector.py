"""
Safe Playwright-based state collector for building State Machines.

This module provides automated web crawling with safety mechanisms to prevent
clicking dangerous buttons (delete, purchase, etc.) while collecting
state machine data from web applications.
"""

import io
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image as PILImage
from playwright.async_api import Browser, BrowserContext, ElementHandle, Page, async_playwright

from .safety import (
    ActionRisk,
    ConfirmationDialogHandler,
    ElementSafetyAnalyzer,
    SafetyConfig,
)
from .verification import (
    BatchVerifier,
    ClickableVerifier,
    ExtractedClickable,
)

logger = logging.getLogger(__name__)


# CSS selectors for clickable elements
CLICKABLE_SELECTORS = [
    "button",
    "a[href]",
    '[role="button"]',
    '[role="link"]',
    '[role="menuitem"]',
    '[role="tab"]',
    'input[type="submit"]',
    'input[type="button"]',
    "[onclick]",
    '[tabindex]:not([tabindex="-1"])',
]


@dataclass
class CollectionResult:
    """Result of a state collection run."""

    clickables: list[ExtractedClickable]
    skipped_dangerous: list[dict[str, Any]]
    metrics: dict[str, Any]
    pages_visited: list[str]
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "clickables": [c.to_dict() for c in self.clickables],
            "skipped_dangerous": self.skipped_dangerous,
            "metrics": self.metrics,
            "pages_visited": self.pages_visited,
            "errors": self.errors,
        }


@dataclass
class TransitionData:
    """Data captured for a state transition."""

    element_id: str
    selector: str
    before_screenshot: np.ndarray
    after_screenshot: np.ndarray
    before_url: str
    after_url: str
    click_point: tuple[int, int]
    url_changed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "selector": self.selector,
            "before_url": self.before_url,
            "after_url": self.after_url,
            "click_point": self.click_point,
            "url_changed": self.url_changed,
            "has_before_screenshot": self.before_screenshot is not None,
            "has_after_screenshot": self.after_screenshot is not None,
        }


class SafePlaywrightStateCollector:
    """Safely collect state machine data from web applications using Playwright.

    This class provides:
    - DOM-based extraction of clickable elements
    - Safety analysis to prevent dangerous clicks
    - Screenshot capture for visual verification
    - Transition recording (before/after states)
    - Pattern matching verification for extracted elements
    """

    def __init__(
        self,
        safety_config: SafetyConfig | None = None,
        verify_extractions: bool = True,
        verification_threshold: float = 0.85,
    ):
        """Initialize the collector.

        Args:
            safety_config: Safety configuration for risk analysis
            verify_extractions: Whether to verify extracted elements with pattern matching
            verification_threshold: Minimum similarity for verification
        """
        self.safety = safety_config or SafetyConfig()
        self.analyzer = ElementSafetyAnalyzer(self.safety)
        self.dialog_handler = ConfirmationDialogHandler()
        self.verify_extractions = verify_extractions
        self.verification_threshold = verification_threshold

        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._element_counter = 0

    def _generate_id(self) -> str:
        """Generate a unique element ID."""
        self._element_counter += 1
        return f"elem_{self._element_counter:06d}"

    async def _take_screenshot(self, page: Page) -> np.ndarray:
        """Take a full page screenshot as numpy array."""
        screenshot_bytes = await page.screenshot(full_page=True)
        pil_image = PILImage.open(io.BytesIO(screenshot_bytes))
        return np.array(pil_image)

    async def _is_truly_clickable(self, element: ElementHandle) -> bool:
        """Verify element is actually interactive."""
        try:
            return await element.evaluate(
                """(el) => {
                const style = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();

                // Must be visible
                if (style.display === 'none') return false;
                if (style.visibility === 'hidden') return false;
                if (parseFloat(style.opacity) === 0) return false;
                if (rect.width === 0 || rect.height === 0) return false;

                // Must not be disabled
                if (el.disabled) return false;
                if (el.getAttribute('aria-disabled') === 'true') return false;

                // Must have pointer events
                if (style.pointerEvents === 'none') return false;

                // Must be in viewport or scrollable
                if (rect.bottom < 0) return false;

                return true;
            }"""
            )
        except Exception:
            return False

    async def _get_unique_selector(self, element: ElementHandle) -> str:
        """Generate a unique CSS selector for an element."""
        try:
            return await element.evaluate(
                """(el) => {
                // Try ID first
                if (el.id) {
                    return '#' + CSS.escape(el.id);
                }

                // Build path from root
                const path = [];
                let current = el;

                while (current && current !== document.body) {
                    let selector = current.tagName.toLowerCase();

                    if (current.id) {
                        selector = '#' + CSS.escape(current.id);
                        path.unshift(selector);
                        break;
                    }

                    // Add classes
                    if (current.className && typeof current.className === 'string') {
                        const classes = current.className.split(' ')
                            .filter(c => c && !c.includes(':') && !c.includes('['))
                            .slice(0, 2);
                        if (classes.length > 0) {
                            selector += '.' + classes.map(c => CSS.escape(c)).join('.');
                        }
                    }

                    // Add nth-child if needed
                    const siblings = Array.from(current.parentElement?.children || []);
                    const sameTagSiblings = siblings.filter(s => s.tagName === current.tagName);
                    if (sameTagSiblings.length > 1) {
                        const index = sameTagSiblings.indexOf(current) + 1;
                        selector += ':nth-of-type(' + index + ')';
                    }

                    path.unshift(selector);
                    current = current.parentElement;
                }

                return path.join(' > ');
            }"""
            )
        except Exception:
            return "unknown"

    async def _extract_element_data(
        self,
        element: ElementHandle,
        page: Page,
        page_screenshot: np.ndarray,
    ) -> ExtractedClickable | None:
        """Extract all data from an element."""
        try:
            bbox = await element.bounding_box()
            if not bbox or bbox["width"] < 10 or bbox["height"] < 10:
                return None

            # Get element screenshot
            element_screenshot_bytes = await element.screenshot()
            element_pil = PILImage.open(io.BytesIO(element_screenshot_bytes))
            element_image = np.array(element_pil)

            # Get metadata
            tag_name = await element.evaluate("el => el.tagName.toLowerCase()")
            text_content = await element.text_content()
            aria_label = await element.get_attribute("aria-label")
            selector = await self._get_unique_selector(element)

            return ExtractedClickable(
                element_id=self._generate_id(),
                selector=selector,
                tag_name=tag_name,
                text=text_content.strip()[:200] if text_content else None,
                aria_label=aria_label,
                bounding_box={
                    "x": int(bbox["x"]),
                    "y": int(bbox["y"]),
                    "width": int(bbox["width"]),
                    "height": int(bbox["height"]),
                },
                screenshot=element_image,
                page_screenshot_before=page_screenshot,
                page_screenshot_after=None,
            )
        except Exception as e:
            logger.debug(f"Failed to extract element: {e}")
            return None

    async def _find_clickable_elements(self, page: Page) -> list[ElementHandle]:
        """Find all clickable elements on the page."""
        elements: list[ElementHandle] = []

        for selector in CLICKABLE_SELECTORS:
            try:
                found = await page.query_selector_all(selector)
                for element in found:
                    if await self._is_truly_clickable(element):
                        elements.append(element)
            except Exception:
                continue

        return elements

    async def collect(
        self,
        url: str,
        max_depth: int = 2,
        max_elements_per_page: int = 50,
        on_progress: Callable[[str, int], None] | None = None,
    ) -> CollectionResult:
        """Safely collect clickable elements from a website.

        Args:
            url: Starting URL to crawl
            max_depth: Maximum depth of clicks to explore
            max_elements_per_page: Maximum elements to extract per page
            on_progress: Optional callback for progress updates

        Returns:
            CollectionResult with extracted clickables and metrics
        """
        all_clickables: list[ExtractedClickable] = []
        skipped: list[dict[str, Any]] = []
        visited_urls: set[str] = {url}
        errors: list[str] = []

        try:
            async with async_playwright() as p:
                self._browser = await p.chromium.launch(headless=True)
                self._context = await self._browser.new_context(
                    viewport={"width": 1920, "height": 1080}
                )
                self._page = await self._context.new_page()

                # Set up safety handlers
                await self.dialog_handler.setup_dialog_handler(self._page)

                # Navigate to starting URL
                if on_progress:
                    on_progress("Loading initial page", 0)

                await self._page.goto(url)
                await self._page.wait_for_load_state("networkidle")

                # BFS exploration
                urls_to_visit: list[tuple[str, int]] = [(url, 0)]

                while urls_to_visit:
                    current_url, depth = urls_to_visit.pop(0)

                    if depth > max_depth:
                        continue

                    if on_progress:
                        percent = min(
                            int(
                                (len(visited_urls) / (len(visited_urls) + len(urls_to_visit) + 1))
                                * 80
                            ),
                            80,
                        )
                        on_progress(f"Processing {current_url[:50]}...", percent)

                    try:
                        # Navigate if needed
                        if self._page.url != current_url:
                            await self._page.goto(current_url)
                            await self._page.wait_for_load_state("networkidle")

                        # Take page screenshot
                        page_screenshot = await self._take_screenshot(self._page)

                        # Find clickable elements
                        elements = await self._find_clickable_elements(self._page)

                        for element in elements[:max_elements_per_page]:
                            # Analyze risk
                            risk, reason = await self.analyzer.analyze_risk(element, self._page)

                            # Extract element data
                            extracted = await self._extract_element_data(
                                element, self._page, page_screenshot
                            )

                            if not extracted:
                                continue

                            extracted.risk_level = risk.value
                            extracted.risk_reason = reason

                            # Check if we should click
                            if not self.analyzer.should_click(risk):
                                skipped.append(
                                    {
                                        "selector": extracted.selector,
                                        "text": extracted.text,
                                        "risk": risk.value,
                                        "reason": reason,
                                        "url": current_url,
                                    }
                                )
                                extracted.was_clicked = False
                                all_clickables.append(extracted)
                                continue

                            # Safe to click - capture transition
                            if not self.safety.dry_run:
                                try:
                                    # Click the element
                                    await element.click()
                                    await self._page.wait_for_timeout(500)

                                    # Check for unexpected dialog
                                    dismissed = await self.dialog_handler.dismiss_dialog(self._page)
                                    if dismissed:
                                        skipped.append(
                                            {
                                                "selector": extracted.selector,
                                                "text": extracted.text,
                                                "risk": "dialog_triggered",
                                                "reason": "Triggered confirmation dialog",
                                                "url": current_url,
                                            }
                                        )
                                        await self._page.goto(current_url)
                                        await self._page.wait_for_load_state("networkidle")
                                        continue

                                    # Capture after state
                                    extracted.page_screenshot_after = await self._take_screenshot(
                                        self._page
                                    )
                                    extracted.was_clicked = True

                                    # Check if we navigated to a new page
                                    new_url = self._page.url
                                    if new_url != current_url and new_url not in visited_urls:
                                        visited_urls.add(new_url)
                                        if depth + 1 <= max_depth:
                                            urls_to_visit.append((new_url, depth + 1))

                                    # Go back for next element
                                    await self._page.goto(current_url)
                                    await self._page.wait_for_load_state("networkidle")

                                except Exception as e:
                                    extracted.error = str(e)
                                    logger.debug(f"Error clicking element: {e}")

                            all_clickables.append(extracted)

                    except Exception as e:
                        error_msg = f"Error processing {current_url}: {e}"
                        errors.append(error_msg)
                        logger.warning(error_msg)

                await self._browser.close()

        except Exception as e:
            error_msg = f"Collection failed: {e}"
            errors.append(error_msg)
            logger.error(error_msg)

        # Verify extractions if enabled
        verify_metrics: dict[str, Any] = {}
        if self.verify_extractions and all_clickables:
            if on_progress:
                on_progress("Verifying extractions", 85)

            verifier = ClickableVerifier(
                similarity_threshold=self.verification_threshold,
            )
            batch_verifier = BatchVerifier(
                verifier=verifier,
                on_progress=on_progress,
            )
            all_clickables, metrics = batch_verifier.verify_batch(
                all_clickables,
                batch_size=10,
            )
            verify_metrics = metrics.to_dict()

        if on_progress:
            on_progress("Complete", 100)

        return CollectionResult(
            clickables=all_clickables,
            skipped_dangerous=skipped,
            metrics={
                "total_found": len(all_clickables),
                "clicked": len([c for c in all_clickables if c.was_clicked]),
                "skipped_dangerous": len(skipped),
                "pages_visited": len(visited_urls),
                "errors": len(errors),
                **verify_metrics,
            },
            pages_visited=list(visited_urls),
            errors=errors,
        )

    async def collect_single_page(
        self,
        url: str,
        max_elements: int = 100,
    ) -> CollectionResult:
        """Collect elements from a single page without navigation.

        This is a faster, simpler collection that only extracts elements
        from one page without clicking or following links.

        Args:
            url: URL to collect from
            max_elements: Maximum elements to extract

        Returns:
            CollectionResult with extracted clickables
        """
        all_clickables: list[ExtractedClickable] = []
        skipped: list[dict[str, Any]] = []
        errors: list[str] = []

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(viewport={"width": 1920, "height": 1080})
                page = await context.new_page()

                await page.goto(url)
                await page.wait_for_load_state("networkidle")

                # Take page screenshot
                page_screenshot = await self._take_screenshot(page)

                # Find clickable elements
                elements = await self._find_clickable_elements(page)

                for element in elements[:max_elements]:
                    # Analyze risk
                    risk, reason = await self.analyzer.analyze_risk(element, page)

                    # Extract element data
                    extracted = await self._extract_element_data(element, page, page_screenshot)

                    if not extracted:
                        continue

                    extracted.risk_level = risk.value
                    extracted.risk_reason = reason
                    extracted.was_clicked = False

                    if risk in (ActionRisk.DANGEROUS, ActionRisk.BLOCKED):
                        skipped.append(
                            {
                                "selector": extracted.selector,
                                "text": extracted.text,
                                "risk": risk.value,
                                "reason": reason,
                                "url": url,
                            }
                        )

                    all_clickables.append(extracted)

                await browser.close()

        except Exception as e:
            error_msg = f"Collection failed: {e}"
            errors.append(error_msg)
            logger.error(error_msg)

        # Verify extractions if enabled
        verify_metrics: dict[str, Any] = {}
        if self.verify_extractions and all_clickables:
            verifier = ClickableVerifier(
                similarity_threshold=self.verification_threshold,
            )
            all_clickables, metrics = verifier.verify_all(all_clickables)
            verify_metrics = metrics.to_dict()

        return CollectionResult(
            clickables=all_clickables,
            skipped_dangerous=skipped,
            metrics={
                "total_found": len(all_clickables),
                "clicked": 0,
                "skipped_dangerous": len(skipped),
                "pages_visited": 1,
                "errors": len(errors),
                **verify_metrics,
            },
            pages_visited=[url],
            errors=errors,
        )


async def collect_web_states(
    url: str,
    max_depth: int = 2,
    max_elements_per_page: int = 50,
    safety_config: SafetyConfig | None = None,
    verify: bool = True,
    on_progress: Callable[[str, int], None] | None = None,
) -> CollectionResult:
    """Convenience function to collect states from a website.

    Args:
        url: Starting URL
        max_depth: Maximum click depth
        max_elements_per_page: Maximum elements per page
        safety_config: Safety configuration
        verify: Whether to verify extractions
        on_progress: Progress callback

    Returns:
        CollectionResult with extracted data
    """
    collector = SafePlaywrightStateCollector(
        safety_config=safety_config,
        verify_extractions=verify,
    )
    return await collector.collect(
        url=url,
        max_depth=max_depth,
        max_elements_per_page=max_elements_per_page,
        on_progress=on_progress,
    )
