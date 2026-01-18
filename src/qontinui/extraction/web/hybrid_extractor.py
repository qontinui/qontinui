"""
Hybrid DOM + Visual extractor for LLM consumption.

This module provides multimodal context for LLMs by combining DOM extraction
with screenshot capture. Similar to OpenManus's approach, it gives LLMs both
visual and semantic understanding of web pages.

LLM Context Includes
--------------------
- **Screenshot**: Base64-encoded viewport image
- **Interactive Elements**: Indexed list with bounding boxes
- **Accessibility Tree**: Optional semantic structure
- **Page Metadata**: URL, title, scroll position, viewport

Performance Optimizations
-------------------------
- Accessibility extraction OFF by default (100-200ms savings)
- DOM extraction and screenshot capture run in parallel
- Configurable screenshot quality (JPEG default)
- Max elements limit to control context size

Classes
-------
HybridExtractor
    Main extractor for multimodal context.
HybridContext
    Container for all extracted context.
StateTracker
    Track page state changes over time.

Functions
---------
extract_hybrid_context
    Convenience function for extraction.
build_llm_prompt
    Build complete LLM prompt from context.

Usage Examples
--------------
Basic extraction::

    from qontinui.extraction.web import extract_hybrid_context

    context = await extract_hybrid_context(page)
    print(f"Page: {context.title}")
    print(f"Elements: {len(context.elements)}")

    # Get LLM message format
    message = context.to_llm_message(include_screenshot=True)
    # message = {"text": "...", "image": {"type": "jpeg", "data": "..."}}

With HybridExtractor::

    from qontinui.extraction.web import HybridExtractor

    extractor = HybridExtractor(
        include_accessibility=True,  # Include a11y tree
        include_iframes=True,        # Extract from iframes
        screenshot_quality=90,       # JPEG quality
        max_elements=200,            # Limit elements
    )
    context = await extractor.extract(page)

Build LLM prompt::

    from qontinui.extraction.web import build_llm_prompt

    prompt = build_llm_prompt(
        context,
        instruction="Click the submit button",
        include_screenshot=True,
    )
    # prompt = {"system": "...", "user": {"text": "...", "image": {...}}}

Track state changes::

    from qontinui.extraction.web import StateTracker

    tracker = StateTracker()
    before = await tracker.capture_state(page)
    await page.click("#button")
    after = await tracker.capture_state(page)

    diff = tracker.get_state_diff(before, after)
    print(f"URL changed: {diff['url_changed']}")

See Also
--------
- interactive_element_extractor: DOM extraction
- accessibility_extractor: A11y tree extraction
- llm_formatter: Element formatting
- llm_clients: LLM client adapters
"""

import asyncio
import base64
import logging
from dataclasses import dataclass, field
from typing import Any

from playwright.async_api import Page

from .accessibility_extractor import AccessibilityExtractor, EnrichedElement
from .frame_manager import FrameAwareElement, extract_across_frames
from .llm_formatter import LLMFormatter
from .models import InteractiveElement

logger = logging.getLogger(__name__)


@dataclass
class HybridContext:
    """
    Combined DOM + visual context for LLM consumption.

    Contains all information needed for an LLM to understand and
    interact with a web page:
    - Screenshot (base64 encoded)
    - Interactive elements with indices
    - Page metadata (URL, title, scroll position)
    - Accessibility data (optional)
    """

    # Page metadata
    url: str
    title: str
    viewport: tuple[int, int]

    # Visual context
    screenshot_base64: str
    screenshot_format: str = "jpeg"

    # DOM context
    elements: list[InteractiveElement] | list[FrameAwareElement] = field(default_factory=list)
    elements_formatted: str = ""  # LLM-friendly format

    # Accessibility context (optional)
    accessibility_tree_text: str = ""

    # Scroll info
    scroll_x: int = 0
    scroll_y: int = 0
    scroll_height: int = 0
    viewport_height: int = 0

    # Frame info (for multi-frame pages)
    frame_count: int = 1
    has_iframes: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "viewport": list(self.viewport),
            "screenshot_base64": self.screenshot_base64[:100] + "...",  # Truncate for display
            "screenshot_format": self.screenshot_format,
            "element_count": len(self.elements),
            "elements_formatted": self.elements_formatted,
            "accessibility_tree_text": (
                self.accessibility_tree_text[:500] + "..." if self.accessibility_tree_text else ""
            ),
            "scroll_x": self.scroll_x,
            "scroll_y": self.scroll_y,
            "scroll_height": self.scroll_height,
            "viewport_height": self.viewport_height,
            "frame_count": self.frame_count,
            "has_iframes": self.has_iframes,
        }

    def to_llm_message(self, include_screenshot: bool = True) -> dict[str, Any]:
        """
        Format as a message suitable for LLM APIs.

        Returns a dict with 'text' and optionally 'image' fields,
        compatible with most LLM API formats.
        """
        text_parts = [
            f"Page: {self.title}",
            f"URL: {self.url}",
            f"Viewport: {self.viewport[0]}x{self.viewport[1]}",
            f"Scroll Position: ({self.scroll_x}, {self.scroll_y})",
            "",
            f"Interactive Elements ({len(self.elements)}):",
            self.elements_formatted,
        ]

        if self.accessibility_tree_text:
            text_parts.extend(
                [
                    "",
                    "Accessibility Tree:",
                    self.accessibility_tree_text[:2000],  # Truncate for token limits
                ]
            )

        message: dict[str, Any] = {"text": "\n".join(text_parts)}

        if include_screenshot and self.screenshot_base64:
            message["image"] = {
                "type": self.screenshot_format,
                "data": self.screenshot_base64,
            }

        return message


@dataclass
class PageState:
    """
    Complete state of a web page for automation.

    Used to track state changes and transitions.
    """

    context: HybridContext
    timestamp: float
    state_hash: str = ""  # Hash for comparing states

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": self.context.to_dict(),
            "timestamp": self.timestamp,
            "state_hash": self.state_hash,
        }


class HybridExtractor:
    """
    Extract combined DOM + visual context from web pages.

    This extractor provides the multimodal context that LLMs need
    for effective web automation:

    1. Screenshot - visual understanding of the page
    2. DOM elements - interactive elements with indices
    3. Accessibility tree - semantic structure (optional, OFF by default)
    4. Page metadata - URL, title, scroll position

    Similar to OpenManus's approach of sending both screenshot
    and JSON DOM state to the LLM for decision making.

    Performance optimizations:
    - Accessibility extraction is OFF by default (saves 100-200ms)
    - DOM extraction and screenshot capture run in parallel
    - Screenshot quality reduced to 70 for faster encoding
    """

    def __init__(
        self,
        include_accessibility: bool = False,
        include_shadow_dom: bool = True,
        include_iframes: bool = True,
        screenshot_quality: int = 70,
        screenshot_format: str = "jpeg",
        max_elements: int = 200,
        parallel_extraction: bool = True,
    ):
        """
        Initialize the extractor.

        Args:
            include_accessibility: Whether to include accessibility tree (default: False for performance)
            include_shadow_dom: Whether to extract from shadow DOMs (default: True)
            include_iframes: Whether to extract from iframes (default: True)
            screenshot_quality: JPEG quality 1-100 (default: 70 for performance)
            screenshot_format: Image format jpeg or png (default: jpeg)
            max_elements: Maximum elements to extract (default: 200)
            parallel_extraction: Whether to run extractions in parallel (default: True)
        """
        self.include_accessibility = include_accessibility
        self.include_shadow_dom = include_shadow_dom
        self.include_iframes = include_iframes
        self.screenshot_quality = screenshot_quality
        self.screenshot_format = screenshot_format
        self.max_elements = max_elements
        self.parallel_extraction = parallel_extraction

        self.formatter = LLMFormatter()
        self.a11y_extractor = AccessibilityExtractor() if include_accessibility else None

    async def extract(self, page: Page) -> HybridContext:
        """
        Extract hybrid context from a page.

        Performance optimization: Runs DOM extraction, screenshot capture,
        and scroll info retrieval in parallel when parallel_extraction is True.

        Args:
            page: Playwright Page to extract from

        Returns:
            HybridContext with all extracted data
        """
        # Get page metadata (fast, synchronous-like)
        url = page.url
        title = await page.title()
        viewport = page.viewport_size or {"width": 1920, "height": 1080}

        # Define extraction tasks
        async def extract_elements() -> (
            tuple[list[InteractiveElement] | list[FrameAwareElement], int, bool]
        ):
            """Extract interactive elements."""
            if self.include_iframes:
                frame_result = await extract_across_frames(
                    page,
                    screenshot_id="hybrid_extract",
                    include_shadow_dom=self.include_shadow_dom,
                )
                return (
                    frame_result.elements[: self.max_elements],
                    len(frame_result.frames),
                    len(frame_result.frames) > 1,
                )
            else:
                from .interactive_element_extractor import InteractiveElementExtractor

                extractor = InteractiveElementExtractor()
                elements = await extractor.extract_interactive_elements(page, "hybrid_extract")
                return elements[: self.max_elements], 1, False

        async def extract_a11y() -> str:
            """Extract accessibility tree."""
            if self.include_accessibility and self.a11y_extractor:
                try:
                    tree = await self.a11y_extractor.extract_tree(page)
                    return tree.to_text()
                except Exception as e:
                    logger.warning(f"Failed to extract accessibility tree: {e}")
            return ""

        # Run extractions in parallel or sequentially
        if self.parallel_extraction:
            # Run all extractions in parallel
            (elements, frame_count, has_iframes), screenshot_b64, scroll_info, a11y_text = (
                await asyncio.gather(
                    extract_elements(),
                    self._take_screenshot(page),
                    self._get_scroll_info(page),
                    extract_a11y(),
                )
            )
        else:
            # Sequential extraction (for debugging or specific use cases)
            scroll_info = await self._get_scroll_info(page)
            screenshot_b64 = await self._take_screenshot(page)
            elements, frame_count, has_iframes = await extract_elements()
            a11y_text = await extract_a11y()

        # Format elements for LLM (CPU-bound, fast)
        formatted = self.formatter.format_elements(elements)

        return HybridContext(
            url=url,
            title=title,
            viewport=(viewport["width"], viewport["height"]),
            screenshot_base64=screenshot_b64,
            screenshot_format=self.screenshot_format,
            elements=elements,
            elements_formatted=formatted.text,
            accessibility_tree_text=a11y_text,
            scroll_x=scroll_info["scroll_x"],
            scroll_y=scroll_info["scroll_y"],
            scroll_height=scroll_info["scroll_height"],
            viewport_height=scroll_info["viewport_height"],
            frame_count=frame_count,
            has_iframes=has_iframes,
        )

    async def extract_with_enrichment(
        self, page: Page
    ) -> tuple[HybridContext, list[EnrichedElement]]:
        """
        Extract hybrid context with enriched element data.

        Returns both the context and elements enriched with
        accessibility data for more detailed analysis.

        Note: This method requires accessibility extraction. If the extractor
        was initialized with include_accessibility=False, a temporary
        AccessibilityExtractor will be created.

        Args:
            page: Playwright Page

        Returns:
            Tuple of (HybridContext, list of EnrichedElement)
        """
        context = await self.extract(page)

        # Enrich elements with accessibility data
        enriched: list[EnrichedElement] = []
        if context.elements:
            # Convert to InteractiveElement list if frame-aware
            plain_elements = [
                e.element if isinstance(e, FrameAwareElement) else e for e in context.elements
            ]
            # Use existing extractor or create temporary one
            a11y = self.a11y_extractor or AccessibilityExtractor()
            enriched = await a11y.enrich_elements(plain_elements, page)

        return context, enriched

    async def _take_screenshot(self, page: Page) -> str:
        """Take a screenshot and encode as base64."""
        try:
            if self.screenshot_format == "jpeg":
                screenshot_bytes = await page.screenshot(
                    type="jpeg",
                    quality=self.screenshot_quality,
                    full_page=False,  # Viewport only for speed
                )
            else:
                screenshot_bytes = await page.screenshot(
                    type="png",
                    full_page=False,
                )

            return base64.b64encode(screenshot_bytes).decode("utf-8")

        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            return ""

    async def _get_scroll_info(self, page: Page) -> dict[str, int]:
        """Get scroll position information."""
        try:
            info = await page.evaluate(
                """() => ({
                scroll_x: window.scrollX || window.pageXOffset || 0,
                scroll_y: window.scrollY || window.pageYOffset || 0,
                scroll_height: document.documentElement.scrollHeight || document.body.scrollHeight || 0,
                viewport_height: window.innerHeight || 0,
            })"""
            )
            return info
        except Exception:
            return {
                "scroll_x": 0,
                "scroll_y": 0,
                "scroll_height": 0,
                "viewport_height": 0,
            }


class StateTracker:
    """
    Track page state changes over time.

    Useful for detecting state transitions during automation
    and understanding what changed after an action.
    """

    def __init__(self, extractor: HybridExtractor | None = None):
        """
        Initialize the tracker.

        Args:
            extractor: HybridExtractor to use (creates default if None)
        """
        self.extractor = extractor or HybridExtractor()
        self.states: list[PageState] = []

    async def capture_state(self, page: Page) -> PageState:
        """
        Capture current page state.

        Args:
            page: Playwright Page

        Returns:
            PageState with current context
        """
        import hashlib
        import time

        context = await self.extractor.extract(page)

        # Generate state hash from element fingerprints
        element_signatures = [
            f"{e.element.selector if isinstance(e, FrameAwareElement) else e.selector}"
            for e in context.elements[:50]  # Use first 50 elements
        ]
        state_hash = hashlib.md5("|".join(element_signatures).encode()).hexdigest()[:16]

        state = PageState(
            context=context,
            timestamp=time.time(),
            state_hash=state_hash,
        )

        self.states.append(state)
        return state

    def get_state_diff(
        self,
        before: PageState,
        after: PageState,
    ) -> dict[str, Any]:
        """
        Compare two states and return differences.

        Args:
            before: State before action
            after: State after action

        Returns:
            Dict describing the differences
        """
        diff: dict[str, Any] = {
            "url_changed": before.context.url != after.context.url,
            "title_changed": before.context.title != after.context.title,
            "element_count_changed": len(before.context.elements) != len(after.context.elements),
            "state_hash_changed": before.state_hash != after.state_hash,
        }

        # URL change details
        if diff["url_changed"]:
            diff["url_before"] = before.context.url
            diff["url_after"] = after.context.url

        # Element count details
        diff["elements_before"] = len(before.context.elements)
        diff["elements_after"] = len(after.context.elements)

        return diff

    def is_same_state(self, state1: PageState, state2: PageState) -> bool:
        """Check if two states represent the same page state."""
        return state1.state_hash == state2.state_hash

    def clear_history(self) -> None:
        """Clear captured state history."""
        self.states = []


def build_llm_prompt(
    context: HybridContext,
    instruction: str,
    include_screenshot: bool = True,
) -> dict[str, Any]:
    """
    Build a complete LLM prompt from hybrid context.

    Creates a prompt suitable for multimodal LLMs like GPT-4V or Claude.

    Args:
        context: HybridContext from extraction
        instruction: User's instruction or goal
        include_screenshot: Whether to include screenshot

    Returns:
        Dict with 'system', 'user' message content
    """
    system_prompt = """You are a web automation assistant. You analyze web pages and select elements to interact with.

Given:
- A screenshot of the current page
- A list of interactive elements with indices [0], [1], etc.
- The user's instruction

Your task:
1. Analyze the page to understand its structure
2. Find the element(s) relevant to the instruction
3. Return the element index and action to perform

Response format:
ELEMENT: <index number>
ACTION: <click|type|hover|scroll>
VALUE: <text to type, if action is type>
REASONING: <brief explanation>"""

    user_message = context.to_llm_message(include_screenshot=include_screenshot)
    user_message["text"] = f"{user_message['text']}\n\nInstruction: {instruction}"

    return {
        "system": system_prompt,
        "user": user_message,
    }


async def extract_hybrid_context(page: Page) -> HybridContext:
    """
    Convenience function to extract hybrid context.

    Args:
        page: Playwright Page

    Returns:
        HybridContext with all extracted data
    """
    extractor = HybridExtractor()
    return await extractor.extract(page)
