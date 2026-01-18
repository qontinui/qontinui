"""
Frame manager for multi-frame DOM extraction.

This module handles extraction across frame boundaries (main document + iframes),
providing frame-aware element IDs and deep locator support for cross-frame
element selection.

Key Features
------------
- **Iframe Traversal**: Extract from main document and all nested iframes
- **Frame-Aware IDs**: Unique element IDs that encode frame context
- **Deep Locators**: Stagehand-style selectors for cross-frame access
- **Graceful Degradation**: Continue extraction if individual frames fail
- **Retry Logic**: Configurable retries for transient failures
- **Configurable Timeouts**: Per-frame timeout settings

Classes
-------
FrameInfo
    Metadata about a frame (ID, URL, selector, parent).
FrameAwareElement
    Element with frame context and deep locator.
FrameExtractionResult
    Container for extracted elements and frame info with fast lookups.
FrameManager
    Manager for frame enumeration and ID encoding.

Functions
---------
extract_across_frames
    Main entry point for multi-frame extraction.
extract_from_frame
    Extract elements from a single frame.

Usage Examples
--------------
Basic multi-frame extraction::

    from qontinui.extraction.web import extract_across_frames

    result = await extract_across_frames(page, "screenshot_001")
    print(f"Found {len(result.elements)} elements across {len(result.frames)} frames")

    # Access elements by frame
    for frame in result.frames:
        frame_elements = result.get_frame_elements(frame.frame_id)
        print(f"Frame {frame.name}: {len(frame_elements)} elements")

Using deep locators::

    # Deep locators enable cross-frame selection
    for elem in result.elements:
        print(f"Deep selector: {elem.deep_selector}")
        # e.g., "iframe#sidebar >> button.submit"

With FrameManager::

    from qontinui.extraction.web import FrameManager

    manager = FrameManager()
    frames = await manager.enumerate_frames(page)

    # Encode/decode frame-aware IDs
    encoded = manager.encode_element_id(2, "elem_000001")  # "2-elem_000001"
    frame_id, elem_id = manager.decode_element_id(encoded)

    # Generate deep locators
    deep = manager.generate_deep_locator("iframe#sidebar", "button.submit")
    # Result: "iframe#sidebar >> button.submit"

See Also
--------
- interactive_element_extractor: Base element extraction
- deep_locator: Cross-frame selector resolution
- exceptions: FrameExtractionError and retry utilities
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from playwright.async_api import Frame, Page

from .exceptions import with_timeout
from .models import BoundingBox, InteractiveElement

logger = logging.getLogger(__name__)

# Default settings
DEFAULT_FRAME_TIMEOUT = 30.0
DEFAULT_FRAME_RETRIES = 2
DEFAULT_RETRY_DELAY = 0.3


@dataclass
class FrameInfo:
    """Information about a frame in the page."""

    frame_id: int  # Ordinal index (0 = main frame)
    name: str  # Frame name attribute
    url: str  # Frame URL (may be about:blank for same-origin)
    selector: str  # CSS selector to reach this frame
    parent_frame_id: int | None  # Parent frame ordinal (None for main)
    is_main: bool  # Whether this is the main frame

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "name": self.name,
            "url": self.url,
            "selector": self.selector,
            "parent_frame_id": self.parent_frame_id,
            "is_main": self.is_main,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrameInfo":
        return cls(
            frame_id=data["frame_id"],
            name=data["name"],
            url=data["url"],
            selector=data["selector"],
            parent_frame_id=data.get("parent_frame_id"),
            is_main=data["is_main"],
        )


@dataclass
class FrameAwareElement:
    """
    An interactive element with frame context.

    Extends InteractiveElement with frame information for multi-frame pages.
    The encoded_id uniquely identifies an element across all frames.
    """

    frame_id: int  # Frame ordinal (0 = main)
    frame_selector: str  # Deep locator path to frame (e.g., "iframe#sidebar")
    encoded_id: str  # Unique ID: "{frame_id}-{element_id}"
    element: InteractiveElement  # The actual element data

    # Deep locator for cross-frame selection
    deep_selector: str = ""  # e.g., "iframe#sidebar >> button.submit"

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "frame_selector": self.frame_selector,
            "encoded_id": self.encoded_id,
            "element": self.element.to_dict(),
            "deep_selector": self.deep_selector,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrameAwareElement":
        return cls(
            frame_id=data["frame_id"],
            frame_selector=data["frame_selector"],
            encoded_id=data["encoded_id"],
            element=InteractiveElement.from_dict(data["element"]),
            deep_selector=data.get("deep_selector", ""),
        )


@dataclass
class FrameExtractionResult:
    """Result of extracting elements from all frames."""

    elements: list[FrameAwareElement] = field(default_factory=list)
    frames: list[FrameInfo] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # Lookup maps for fast access
    _element_by_encoded_id: dict[str, FrameAwareElement] = field(
        default_factory=dict, repr=False
    )
    _elements_by_frame: dict[int, list[FrameAwareElement]] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        self._build_indexes()

    def _build_indexes(self) -> None:
        """Build lookup indexes for fast access."""
        self._element_by_encoded_id = {e.encoded_id: e for e in self.elements}
        self._elements_by_frame = {}
        for elem in self.elements:
            if elem.frame_id not in self._elements_by_frame:
                self._elements_by_frame[elem.frame_id] = []
            self._elements_by_frame[elem.frame_id].append(elem)

    def get_by_encoded_id(self, encoded_id: str) -> FrameAwareElement | None:
        """Get element by its frame-aware encoded ID."""
        return self._element_by_encoded_id.get(encoded_id)

    def get_frame_elements(self, frame_id: int) -> list[FrameAwareElement]:
        """Get all elements from a specific frame."""
        return self._elements_by_frame.get(frame_id, [])

    def to_dict(self) -> dict[str, Any]:
        return {
            "elements": [e.to_dict() for e in self.elements],
            "frames": [f.to_dict() for f in self.frames],
            "errors": self.errors,
        }


class FrameManager:
    """
    Manages extraction across multiple frames (main document + iframes).

    Features:
    - Enumerate all frames in a page
    - Generate frame-aware element IDs
    - Build deep locator selectors for cross-frame access
    - Handle cross-origin iframe restrictions gracefully
    """

    def __init__(self) -> None:
        self._frame_counter = 0
        self._frame_selectors: dict[str, str] = {}  # frame_name -> selector

    def reset(self) -> None:
        """Reset state between pages."""
        self._frame_counter = 0
        self._frame_selectors = {}

    async def enumerate_frames(self, page: Page) -> list[FrameInfo]:
        """
        Enumerate all frames in a page.

        Returns frame info for main frame and all iframes, including nested.
        """
        frames: list[FrameInfo] = []
        self._frame_counter = 0

        for frame in page.frames:
            frame_info = await self._get_frame_info(frame, page)
            if frame_info:
                frames.append(frame_info)
                self._frame_counter += 1

        logger.info(f"Enumerated {len(frames)} frames")
        return frames

    async def _get_frame_info(self, frame: Frame, page: Page) -> FrameInfo | None:
        """Get information about a single frame."""
        try:
            is_main = frame == page.main_frame
            parent_frame_id = None

            if not is_main and frame.parent_frame:
                # Find parent frame's ordinal
                for i, f in enumerate(page.frames):
                    if f == frame.parent_frame:
                        parent_frame_id = i
                        break

            # Get frame selector
            selector = await self._get_frame_selector(frame, page)

            return FrameInfo(
                frame_id=self._frame_counter,
                name=frame.name or f"frame_{self._frame_counter}",
                url=frame.url,
                selector=selector,
                parent_frame_id=parent_frame_id,
                is_main=is_main,
            )
        except Exception as e:
            logger.warning(f"Failed to get frame info: {e}")
            return None

    async def _get_frame_selector(self, frame: Frame, page: Page) -> str:
        """Generate a CSS selector to reach this frame."""
        if frame == page.main_frame:
            return "main"

        try:
            # Try to find the iframe element that hosts this frame
            frame_element = await frame.frame_element()
            if frame_element:
                selector = await page.evaluate(
                    """(el) => {
                    if (el.id) return 'iframe#' + CSS.escape(el.id);
                    if (el.name) return 'iframe[name="' + el.name + '"]';
                    if (el.src) return 'iframe[src="' + el.src.substring(0, 50) + '"]';
                    return 'iframe';
                }""",
                    frame_element,
                )
                return str(selector)
        except Exception:
            pass

        return f"iframe[name='{frame.name}']" if frame.name else "iframe"

    def encode_element_id(self, frame_id: int, element_id: str) -> str:
        """
        Create a frame-aware encoded ID.

        Format: "{frame_id}-{element_id}"
        Example: "0-elem_000001" (main frame), "2-elem_000015" (iframe #2)
        """
        return f"{frame_id}-{element_id}"

    def decode_element_id(self, encoded_id: str) -> tuple[int, str]:
        """
        Decode a frame-aware encoded ID.

        Returns (frame_id, element_id) tuple.
        """
        parts = encoded_id.split("-", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid encoded ID format: {encoded_id}")
        return int(parts[0]), parts[1]

    def generate_deep_locator(self, frame_selector: str, element_selector: str) -> str:
        """
        Generate a Stagehand-style deep locator.

        Format: "frame_selector >> element_selector"
        Example: "iframe#sidebar >> button.submit"

        For main frame, just returns the element selector.
        """
        if frame_selector == "main" or not frame_selector:
            return element_selector
        return f"{frame_selector} >> {element_selector}"

    def parse_deep_locator(self, deep_locator: str) -> tuple[str, str]:
        """
        Parse a deep locator into frame and element parts.

        Returns (frame_selector, element_selector) tuple.
        """
        if " >> " not in deep_locator:
            return "main", deep_locator

        parts = deep_locator.split(" >> ", 1)
        return parts[0], parts[1]


async def extract_from_frame(
    frame: Frame,
    frame_id: int,
    frame_selector: str,
    screenshot_id: str,
    min_size: tuple[int, int] = (10, 10),
    include_shadow_dom: bool = True,
    max_retries: int = DEFAULT_FRAME_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    timeout_seconds: float = DEFAULT_FRAME_TIMEOUT,
) -> list[FrameAwareElement]:
    """
    Extract interactive elements from a single frame.

    Args:
        frame: Playwright Frame to extract from.
        frame_id: Ordinal index of this frame.
        frame_selector: CSS selector to reach this frame.
        screenshot_id: ID of the page screenshot.
        min_size: Minimum (width, height) for elements.
        include_shadow_dom: Whether to extract from shadow DOMs.
        max_retries: Maximum number of retry attempts.
        retry_delay: Delay between retries in seconds.
        timeout_seconds: Timeout for extraction.

    Returns:
        List of FrameAwareElement objects.
    """
    try:
        return await with_timeout(
            _extract_from_frame_with_retry(
                frame=frame,
                frame_id=frame_id,
                frame_selector=frame_selector,
                screenshot_id=screenshot_id,
                min_size=min_size,
                include_shadow_dom=include_shadow_dom,
                max_retries=max_retries,
                retry_delay=retry_delay,
            ),
            timeout_seconds=timeout_seconds,
            operation_name=f"extract_from_frame_{frame_id}",
        )
    except Exception as e:
        logger.warning(f"Frame {frame_id} extraction failed: {e}")
        return []


async def _extract_from_frame_with_retry(
    frame: Frame,
    frame_id: int,
    frame_selector: str,
    screenshot_id: str,
    min_size: tuple[int, int],
    include_shadow_dom: bool,
    max_retries: int,
    retry_delay: float,
) -> list[FrameAwareElement]:
    """
    Internal implementation with retry logic.
    """
    from .interactive_element_extractor import InteractiveElementExtractor

    extractor = InteractiveElementExtractor()
    extractor.reset_counter()

    frame_manager = FrameManager()

    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            # Extract elements from this frame
            # Note: We pass the frame as if it were a page for the evaluate call
            elements_data = await frame.evaluate(
                _get_extraction_script(include_shadow_dom),
                {
                    "minWidth": min_size[0],
                    "minHeight": min_size[1],
                    "skipTags": [
                        "script",
                        "style",
                        "noscript",
                        "template",
                        "slot",
                        "head",
                        "meta",
                        "link",
                        "title",
                        "base",
                        "br",
                        "wbr",
                        "hr",
                    ],
                },
            )

            frame_elements: list[FrameAwareElement] = []

            for data in elements_data:
                try:
                    bbox = BoundingBox(
                        x=data["bbox"]["x"],
                        y=data["bbox"]["y"],
                        width=data["bbox"]["width"],
                        height=data["bbox"]["height"],
                    )

                    element_id = f"elem_{len(frame_elements) + 1:06d}"

                    element = InteractiveElement(
                        id=element_id,
                        bbox=bbox,
                        tag_name=data["tagName"],
                        element_type=data["elementType"],
                        screenshot_id=screenshot_id,
                        selector=data["selector"],
                        text=data.get("text"),
                        href=data.get("href"),
                        aria_label=data.get("ariaLabel"),
                        aria_role=data.get("ariaRole"),
                    )

                    encoded_id = frame_manager.encode_element_id(frame_id, element_id)
                    deep_selector = frame_manager.generate_deep_locator(
                        frame_selector, element.selector
                    )

                    frame_element = FrameAwareElement(
                        frame_id=frame_id,
                        frame_selector=frame_selector,
                        encoded_id=encoded_id,
                        element=element,
                        deep_selector=deep_selector,
                    )
                    frame_elements.append(frame_element)

                except Exception as e:
                    logger.debug(f"Failed to process element: {e}")
                    continue

            return frame_elements

        except Exception as e:
            last_error = e
            if attempt == max_retries:
                logger.warning(
                    f"Failed to extract from frame {frame_id} after "
                    f"{max_retries + 1} attempts: {e}"
                )
                return []

            logger.debug(
                f"Frame {frame_id} extraction attempt {attempt + 1} failed: {e}. "
                f"Retrying in {retry_delay}s..."
            )
            await asyncio.sleep(retry_delay)

    # Should never reach here
    return []


async def extract_across_frames(
    page: Page,
    screenshot_id: str,
    min_size: tuple[int, int] = (10, 10),
    include_shadow_dom: bool = True,
) -> FrameExtractionResult:
    """
    Extract interactive elements from all frames in a page.

    This is the main entry point for multi-frame extraction.

    Args:
        page: Playwright Page to extract from
        screenshot_id: ID of the page screenshot
        min_size: Minimum (width, height) for elements
        include_shadow_dom: Whether to extract from shadow DOMs

    Returns:
        FrameExtractionResult with all elements and frame info
    """
    frame_manager = FrameManager()
    frame_manager.reset()

    all_elements: list[FrameAwareElement] = []
    all_frames: list[FrameInfo] = []
    errors: list[str] = []

    # Enumerate all frames
    frames = await frame_manager.enumerate_frames(page)
    all_frames = frames

    # Extract from each frame
    for frame_info in frames:
        try:
            # Get the actual Frame object
            if frame_info.is_main:
                frame = page.main_frame
            else:
                frame = page.frames[frame_info.frame_id]

            # Extract elements from this frame
            frame_elements = await extract_from_frame(
                frame=frame,
                frame_id=frame_info.frame_id,
                frame_selector=frame_info.selector,
                screenshot_id=screenshot_id,
                min_size=min_size,
                include_shadow_dom=include_shadow_dom,
            )

            all_elements.extend(frame_elements)
            logger.info(
                f"Extracted {len(frame_elements)} elements from frame {frame_info.frame_id}"
            )

        except Exception as e:
            error_msg = f"Error extracting from frame {frame_info.frame_id}: {e}"
            errors.append(error_msg)
            logger.warning(error_msg)

    result = FrameExtractionResult(
        elements=all_elements,
        frames=all_frames,
        errors=errors,
    )

    logger.info(
        f"Multi-frame extraction complete: {len(all_elements)} elements "
        f"from {len(all_frames)} frames"
    )

    return result


def _get_extraction_script(include_shadow_dom: bool) -> str:
    """
    Get the JavaScript extraction script.

    This is a modified version of the script from InteractiveElementExtractor
    that supports shadow DOM extraction.
    """
    shadow_dom_code = """
        // Shadow DOM extraction helper
        function extractFromShadowRoot(shadowRoot, results, scrollX, scrollY, shadowPath) {
            for (const el of shadowRoot.querySelectorAll('*')) {
                processElement(el, results, scrollX, scrollY, shadowPath);

                // Recursively extract from nested shadow roots
                if (el.shadowRoot) {
                    extractFromShadowRoot(
                        el.shadowRoot,
                        results,
                        scrollX,
                        scrollY,
                        shadowPath + '/shadow'
                    );
                }
            }
        }
    """ if include_shadow_dom else ""

    shadow_dom_call = """
                // Extract from shadow DOM if present
                if (el.shadowRoot) {
                    extractFromShadowRoot(el.shadowRoot, results, scrollX, scrollY, '/shadow');
                }
    """ if include_shadow_dom else ""

    return f"""
        (config) => {{
            const minWidth = config.minWidth;
            const minHeight = config.minHeight;
            const skipTags = new Set(config.skipTags);

            // Interactive HTML tags
            const interactiveTags = new Set([
                'button', 'a', 'input', 'select', 'textarea', 'label',
                'details', 'summary', 'option', 'optgroup'
            ]);

            // Interactive ARIA roles
            const interactiveRoles = new Set([
                'button', 'link', 'menuitem', 'tab', 'checkbox',
                'radio', 'switch', 'slider', 'textbox', 'combobox'
            ]);

            const results = [];
            const scrollX = window.scrollX || window.pageXOffset;
            const scrollY = window.scrollY || window.pageYOffset;

            {shadow_dom_code}

            function processElement(el, results, scrollX, scrollY, shadowPath = '') {{
                const tagName = el.tagName.toLowerCase();

                // Skip non-visible tags
                if (skipTags.has(tagName)) return;

                // Get bounding box
                const rect = el.getBoundingClientRect();
                if (rect.width < minWidth || rect.height < minHeight) return;

                // Check visibility
                const style = window.getComputedStyle(el);
                if (style.display === 'none') return;
                if (style.visibility === 'hidden') return;
                if (style.opacity === '0') return;

                // Get ARIA role
                const ariaRole = el.getAttribute('role');
                const hasCursorPointer = style.cursor === 'pointer';

                // Determine if this is an interactive element
                let elementType = null;

                if (interactiveTags.has(tagName)) {{
                    elementType = tagName;
                }}
                else if (ariaRole && interactiveRoles.has(ariaRole)) {{
                    elementType = 'aria_' + ariaRole;
                }}
                else if (el.getAttribute('tabindex') !== null) {{
                    elementType = 'tabindex_' + tagName;
                }}
                else if (el.onclick !== null || el.getAttribute('onclick') !== null) {{
                    elementType = 'onclick_' + tagName;
                }}
                else if (hasCursorPointer) {{
                    const textTags = new Set(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'ul', 'ol', 'span', 'strong', 'em', 'b', 'i', 'blockquote', 'pre', 'code']);
                    if (textTags.has(tagName)) return;

                    const text = el.innerText?.trim() || '';
                    if (text.length === 0 || text.length > 30) return;
                    if (text.includes('\\n')) return;
                    if (rect.width > 300 || rect.height > 60) return;

                    const hasInteractiveChildren = el.querySelector('button, a, input, select, textarea, [role="button"], [role="link"]');
                    if (hasInteractiveChildren) return;

                    elementType = 'clickable_' + tagName;
                }}

                // Skip non-interactive elements
                if (!elementType) return;

                // Get text content
                let directText = '';
                for (const node of el.childNodes) {{
                    if (node.nodeType === Node.TEXT_NODE) {{
                        directText += node.textContent.trim();
                    }}
                }}
                const fullText = el.innerText?.trim() || '';

                // Generate a CSS selector
                let selector = tagName;
                if (el.id) {{
                    selector = '#' + CSS.escape(el.id);
                }} else if (el.className && typeof el.className === 'string') {{
                    const classes = el.className.split(' ')
                        .filter(c => c && !c.includes(':'))
                        .slice(0, 2);
                    if (classes.length > 0) {{
                        selector = tagName + '.' + classes.map(c => CSS.escape(c)).join('.');
                    }}
                }}

                // Add shadow path suffix for shadow DOM elements
                if (shadowPath) {{
                    selector = selector + '[shadow-path="' + shadowPath + '"]';
                }}

                results.push({{
                    tagName: tagName,
                    elementType: elementType + (shadowPath ? '_shadow' : ''),
                    bbox: {{
                        x: Math.round(rect.x + scrollX),
                        y: Math.round(rect.y + scrollY),
                        width: Math.round(rect.width),
                        height: Math.round(rect.height),
                    }},
                    selector: selector,
                    text: directText.substring(0, 200) || fullText.substring(0, 200) || null,
                    href: el.href || el.getAttribute('href') || null,
                    ariaLabel: el.getAttribute('aria-label') || null,
                    ariaRole: ariaRole || null,
                    shadowPath: shadowPath || null,
                }});
            }}

            // Process main document
            for (const el of document.querySelectorAll('*')) {{
                processElement(el, results, scrollX, scrollY);
                {shadow_dom_call}
            }}

            return results;
        }}
    """
