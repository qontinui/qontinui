"""
Interactive Element Extractor for web extraction.

This module provides DOM-based extraction of interactive UI elements using Playwright.
It identifies elements that users can interact with, enabling automated testing
and AI-driven web automation.

Key Features
------------
- **Element Detection**: Buttons, links, inputs, ARIA roles, clickable elements
- **Shadow DOM Support**: Pierces open/closed shadow roots up to configurable depth
- **Frame Integration**: Multi-frame extraction via `extract_with_frames()`
- **Stability Waiting**: DOM stability detection via `extract_with_stability()`
- **Full Extraction**: Combined stability + frames via `extract_full()`
- **Retry Logic**: Configurable retries for transient failures
- **Timeouts**: Configurable timeout settings

Classes
-------
ExtractionOptions
    Configuration for element extraction including min size, shadow DOM depth,
    cursor:pointer detection, retry settings, and timeouts.

InteractiveElementExtractor
    Main extractor class with multiple extraction methods.

Usage Examples
--------------
Basic extraction::

    from qontinui.extraction.web import InteractiveElementExtractor

    extractor = InteractiveElementExtractor()
    elements = await extractor.extract_interactive_elements(page, "screenshot_001")
    for elem in elements:
        print(f"{elem.element_type}: {elem.text} at {elem.bbox}")

With custom options::

    options = ExtractionOptions(
        include_cursor_pointer=False,
        min_width=20,
        min_height=20,
        max_retries=5,
        timeout_seconds=60.0,
    )
    extractor = InteractiveElementExtractor(options)
    elements = await extractor.extract_interactive_elements(page, "screenshot_001")

Multi-frame extraction (main document + iframes)::

    result = await extractor.extract_with_frames(page, "screenshot_001")
    print(f"Found {len(result.elements)} elements across {len(result.frames)} frames")
    for elem in result.elements:
        print(f"  Frame {elem.frame_id}: {elem.element.text}")

With DOM stability waiting (for dynamic pages)::

    elements = await extractor.extract_with_stability(
        page,
        screenshot_id="screenshot_001",
        stability_ms=500,   # Wait for 500ms without mutations
        max_wait_ms=5000,   # Max 5 seconds total
    )

Full extraction (stability + frames)::

    result = await extractor.extract_full(
        page,
        screenshot_id="screenshot_001",
        include_iframes=True,
        wait_for_stability=True,
    )

See Also
--------
- frame_manager: Multi-frame extraction with deep locators
- dom_stability: DOM stability detection and lazy loading
- accessibility_extractor: Accessibility tree enrichment
- exceptions: Custom exceptions and retry/timeout decorators
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .frame_manager import FrameExtractionResult

from playwright.async_api import Page

from .exceptions import (
    ElementExtractionError,
    ShadowDOMError,
    ValidationError,
    with_timeout,
)
from .models import BoundingBox, InteractiveElement

logger = logging.getLogger(__name__)

# Default timeout for extraction operations (seconds)
DEFAULT_EXTRACTION_TIMEOUT = 30.0


@dataclass
class ExtractionOptions:
    """
    Configuration options for interactive element extraction.

    Controls which elements are extracted and how extraction behaves.

    Attributes
    ----------
    min_width : int
        Minimum element width in pixels (default: 10).
    min_height : int
        Minimum element height in pixels (default: 10).
    include_shadow_dom : bool
        Whether to pierce shadow DOM boundaries (default: True).
    max_shadow_depth : int
        Maximum nesting depth for shadow DOM traversal (default: 5).
    include_cursor_pointer : bool
        Detect clickable elements via CSS cursor:pointer (default: True).
    max_cursor_pointer_text_length : int
        Max text length for cursor:pointer elements (default: 30).
    max_cursor_pointer_width : int
        Max width for cursor:pointer elements (default: 300).
    max_cursor_pointer_height : int
        Max height for cursor:pointer elements (default: 60).
    max_retries : int
        Number of retry attempts for transient failures (default: 3).
    retry_delay : float
        Delay between retries in seconds (default: 0.5).
    timeout_seconds : float
        Timeout for extraction operations (default: 30.0).

    Example
    -------
    ::

        # Strict extraction: larger elements, no cursor detection
        options = ExtractionOptions(
            min_width=50,
            min_height=30,
            include_cursor_pointer=False,
        )

        # Deep shadow DOM traversal with longer timeout
        options = ExtractionOptions(
            max_shadow_depth=10,
            timeout_seconds=60.0,
        )
    """

    min_width: int = 10
    min_height: int = 10
    include_shadow_dom: bool = True
    max_shadow_depth: int = 5
    include_cursor_pointer: bool = True
    max_cursor_pointer_text_length: int = 30
    max_cursor_pointer_width: int = 300
    max_cursor_pointer_height: int = 60
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 0.5
    # Timeout settings
    timeout_seconds: float = DEFAULT_EXTRACTION_TIMEOUT

    def __post_init__(self) -> None:
        """Validate options after initialization."""
        self.validate()

    def validate(self) -> None:
        """
        Validate extraction options.

        Raises:
            ValidationError: If any option is invalid.
        """
        if self.min_width < 1:
            raise ValidationError(f"min_width must be >= 1, got {self.min_width}")
        if self.min_height < 1:
            raise ValidationError(f"min_height must be >= 1, got {self.min_height}")
        if self.max_shadow_depth < 0:
            raise ValidationError(
                f"max_shadow_depth must be >= 0, got {self.max_shadow_depth}"
            )
        if self.max_cursor_pointer_text_length < 1:
            raise ValidationError(
                f"max_cursor_pointer_text_length must be >= 1, got {self.max_cursor_pointer_text_length}"
            )
        if self.max_cursor_pointer_width < 1:
            raise ValidationError(
                f"max_cursor_pointer_width must be >= 1, got {self.max_cursor_pointer_width}"
            )
        if self.max_cursor_pointer_height < 1:
            raise ValidationError(
                f"max_cursor_pointer_height must be >= 1, got {self.max_cursor_pointer_height}"
            )
        if self.max_retries < 0:
            raise ValidationError(f"max_retries must be >= 0, got {self.max_retries}")
        if self.retry_delay < 0:
            raise ValidationError(f"retry_delay must be >= 0, got {self.retry_delay}")
        if self.timeout_seconds <= 0:
            raise ValidationError(
                f"timeout_seconds must be > 0, got {self.timeout_seconds}"
            )

# Tags to skip (not visible/not useful)
SKIP_TAGS = frozenset(
    {
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
    }
)


class InteractiveElementExtractor:
    """
    Extracts interactive elements from a web page using DOM analysis.

    Only captures:
    - Buttons (<button>, role="button")
    - Links (<a>, role="link")
    - Form inputs (<input>, <select>, <textarea>)
    - Other interactive ARIA roles (tab, checkbox, radio, etc.)
    - Elements with onclick or tabindex

    Enhanced features:
    - Shadow DOM piercing (extracts from shadow roots)
    - Configurable extraction behavior
    """

    def __init__(self, options: ExtractionOptions | None = None) -> None:
        """
        Initialize the extractor.

        Args:
            options: Extraction options. Uses defaults if not provided.
        """
        self._element_counter = 0
        self.options = options or ExtractionOptions()

    def _generate_id(self) -> str:
        """Generate a unique element ID."""
        self._element_counter += 1
        return f"elem_{self._element_counter:06d}"

    def reset_counter(self) -> None:
        """Reset the element counter (call between pages)."""
        self._element_counter = 0

    async def extract_interactive_elements(
        self,
        page: Page,
        screenshot_id: str,
        min_size: tuple[int, int] | None = None,
        timeout: float | None = None,
    ) -> list[InteractiveElement]:
        """
        Extract interactive elements from a page.

        Args:
            page: Playwright page to extract from.
            screenshot_id: ID of the page screenshot (for reference).
            min_size: Minimum (width, height) for elements. Uses options if not provided.
            timeout: Timeout in seconds. Uses options if not provided.

        Returns:
            List of InteractiveElement objects.

        Raises:
            ElementExtractionError: If extraction fails after retries.
            ExtractionTimeoutError: If extraction times out.
        """
        actual_timeout = timeout if timeout is not None else self.options.timeout_seconds
        return await with_timeout(
            self._extract_interactive_elements_impl(page, screenshot_id, min_size),
            timeout_seconds=actual_timeout,
            operation_name="extract_interactive_elements",
        )

    async def _extract_interactive_elements_impl(
        self,
        page: Page,
        screenshot_id: str,
        min_size: tuple[int, int] | None = None,
    ) -> list[InteractiveElement]:
        """
        Internal implementation of element extraction with retry logic.
        """
        # Use provided min_size or fall back to options
        actual_min_size = min_size or (self.options.min_width, self.options.min_height)

        if self.options.include_shadow_dom:
            logger.info("Extracting interactive elements from DOM (with shadow DOM)...")
        else:
            logger.info("Extracting interactive elements from DOM...")

        # Retry logic for transient failures
        last_error: Exception | None = None
        for attempt in range(self.options.max_retries + 1):
            try:
                return await self._do_extraction(page, screenshot_id, actual_min_size)
            except Exception as e:
                last_error = e
                if attempt == self.options.max_retries:
                    logger.error(
                        f"Element extraction failed after {self.options.max_retries + 1} attempts: {e}"
                    )
                    raise ElementExtractionError(
                        f"Failed to extract elements after {self.options.max_retries + 1} attempts: {e}"
                    ) from e

                import asyncio

                logger.warning(
                    f"Element extraction attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {self.options.retry_delay}s..."
                )
                await asyncio.sleep(self.options.retry_delay)

        # Should never reach here
        if last_error:
            raise ElementExtractionError(f"Extraction failed: {last_error}") from last_error
        return []

    async def _do_extraction(
        self,
        page: Page,
        screenshot_id: str,
        actual_min_size: tuple[int, int],
    ) -> list[InteractiveElement]:
        """
        Perform the actual element extraction.
        """
        try:
            # Get all interactive element data in a single evaluate call
            # Enhanced with shadow DOM traversal
            # Performance optimized:
            # - Use targeted selector for interactive elements first
            # - Batch getBoundingClientRect calls
            # - Cache scroll position
            # - Minimize getComputedStyle calls
            elements_data = await page.evaluate(
                """
                (config) => {
                    const minWidth = config.minWidth;
                    const minHeight = config.minHeight;
                    const skipTags = new Set(config.skipTags);
                    const includeShadowDOM = config.includeShadowDOM;
                    const maxShadowDepth = config.maxShadowDepth;
                    const includeCursorPointer = config.includeCursorPointer;
                    const maxCursorPointerTextLength = config.maxCursorPointerTextLength;
                    const maxCursorPointerWidth = config.maxCursorPointerWidth;
                    const maxCursorPointerHeight = config.maxCursorPointerHeight;

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

                    // Text tags to skip for cursor:pointer detection
                    const textTags = new Set([
                        'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'ul', 'ol',
                        'span', 'strong', 'em', 'b', 'i', 'blockquote', 'pre', 'code'
                    ]);

                    const results = [];

                    // Cache scroll position once (avoid reflow)
                    const scrollX = window.scrollX || window.pageXOffset || 0;
                    const scrollY = window.scrollY || window.pageYOffset || 0;

                    // Pre-compute targeted selector for likely interactive elements
                    // This reduces the number of elements we need to process
                    const interactiveSelector = includeCursorPointer
                        ? 'button, a, input, select, textarea, label, details, summary, [role], [tabindex], [onclick]'
                        : 'button, a, input, select, textarea, label, details, summary, [role], [tabindex], [onclick]';

                    // Process a single element - optimized version
                    function processElement(el, shadowPath, forceCheckCursor) {
                        const tagName = el.tagName.toLowerCase();

                        // Skip non-visible tags (fast check, no style computation)
                        if (skipTags.has(tagName)) return;

                        // Get bounding box early (single reflow batch)
                        const rect = el.getBoundingClientRect();
                        if (rect.width < minWidth || rect.height < minHeight) return;

                        // Get ARIA role (fast attribute access)
                        const ariaRole = el.getAttribute('role');

                        // Determine element type BEFORE checking visibility
                        // This avoids expensive getComputedStyle for non-interactive elements
                        let elementType = null;
                        let needsStyleCheck = false;

                        if (interactiveTags.has(tagName)) {
                            elementType = tagName;
                            needsStyleCheck = true;
                        }
                        else if (ariaRole && interactiveRoles.has(ariaRole)) {
                            elementType = 'aria_' + ariaRole;
                            needsStyleCheck = true;
                        }
                        else if (el.getAttribute('tabindex') !== null) {
                            elementType = 'tabindex_' + tagName;
                            needsStyleCheck = true;
                        }
                        else if (el.onclick !== null || el.getAttribute('onclick') !== null) {
                            elementType = 'onclick_' + tagName;
                            needsStyleCheck = true;
                        }
                        else if (forceCheckCursor && includeCursorPointer && !textTags.has(tagName)) {
                            // Only check cursor:pointer for elements that passed initial filter
                            needsStyleCheck = true;
                        }

                        // Skip if not interactive and not checking cursor
                        if (!elementType && !needsStyleCheck) return;

                        // Now check visibility (expensive getComputedStyle)
                        const style = window.getComputedStyle(el);
                        if (style.display === 'none') return;
                        if (style.visibility === 'hidden') return;
                        if (style.opacity === '0') return;

                        // Handle cursor:pointer detection
                        if (!elementType && includeCursorPointer && style.cursor === 'pointer') {
                            if (textTags.has(tagName)) return;

                            const text = el.innerText?.trim() || '';
                            if (text.length === 0 || text.length > maxCursorPointerTextLength) return;
                            if (text.includes('\\n')) return;
                            if (rect.width > maxCursorPointerWidth || rect.height > maxCursorPointerHeight) return;

                            const hasInteractiveChildren = el.querySelector(
                                'button, a, input, select, textarea, [role="button"], [role="link"]'
                            );
                            if (hasInteractiveChildren) return;

                            elementType = 'clickable_' + tagName;
                        }

                        // Skip if still not interactive
                        if (!elementType) return;

                        // Get text content (lazy evaluation)
                        let directText = '';
                        for (const node of el.childNodes) {
                            if (node.nodeType === Node.TEXT_NODE) {
                                directText += node.textContent.trim();
                            }
                        }
                        const fullText = el.innerText?.trim() || '';

                        // Generate a CSS selector (optimized)
                        let selector = tagName;
                        if (el.id) {
                            selector = '#' + CSS.escape(el.id);
                        } else if (el.className && typeof el.className === 'string') {
                            const classes = el.className.split(' ')
                                .filter(c => c && !c.includes(':'))
                                .slice(0, 2);
                            if (classes.length > 0) {
                                selector = tagName + '.' + classes.map(c => CSS.escape(c)).join('.');
                            }
                        }

                        results.push({
                            tagName: tagName,
                            elementType: elementType,
                            bbox: {
                                x: Math.round(rect.x + scrollX),
                                y: Math.round(rect.y + scrollY),
                                width: Math.round(rect.width),
                                height: Math.round(rect.height),
                            },
                            selector: selector,
                            text: directText.substring(0, 200) || fullText.substring(0, 200) || null,
                            href: el.href || el.getAttribute('href') || null,
                            ariaLabel: el.getAttribute('aria-label') || null,
                            ariaRole: ariaRole || null,
                            shadowPath: shadowPath || null,
                        });
                    }

                    // Recursively extract from a root (document or shadow root)
                    // Optimized: use targeted selector first, then check remaining for cursor:pointer
                    function extractFromRoot(root, shadowPath, depth) {
                        if (depth > maxShadowDepth) return;

                        // First pass: targeted selector for known interactive elements
                        const interactiveElements = root.querySelectorAll(interactiveSelector);
                        const processedElements = new Set();

                        for (const el of interactiveElements) {
                            processElement(el, shadowPath, false);
                            processedElements.add(el);

                            // Check for shadow root (if shadow DOM enabled)
                            if (includeShadowDOM && el.shadowRoot) {
                                const newPath = shadowPath
                                    ? shadowPath + ' >> shadow'
                                    : 'shadow';
                                extractFromRoot(el.shadowRoot, newPath, depth + 1);
                            }
                        }

                        // Second pass: check remaining elements for cursor:pointer (if enabled)
                        // This is the expensive path, only used when needed
                        if (includeCursorPointer) {
                            const allElements = root.querySelectorAll('*');
                            for (const el of allElements) {
                                if (!processedElements.has(el)) {
                                    processElement(el, shadowPath, true);

                                    // Check for shadow root
                                    if (includeShadowDOM && el.shadowRoot && !processedElements.has(el)) {
                                        const newPath = shadowPath
                                            ? shadowPath + ' >> shadow'
                                            : 'shadow';
                                        extractFromRoot(el.shadowRoot, newPath, depth + 1);
                                    }
                                }
                            }
                        }
                    }

                    // Start extraction from document
                    extractFromRoot(document, '', 0);

                    return results;
                }
                """,
                {
                    "minWidth": actual_min_size[0],
                    "minHeight": actual_min_size[1],
                    "skipTags": list(SKIP_TAGS),
                    "includeShadowDOM": self.options.include_shadow_dom,
                    "maxShadowDepth": self.options.max_shadow_depth,
                    "includeCursorPointer": self.options.include_cursor_pointer,
                    "maxCursorPointerTextLength": self.options.max_cursor_pointer_text_length,
                    "maxCursorPointerWidth": self.options.max_cursor_pointer_width,
                    "maxCursorPointerHeight": self.options.max_cursor_pointer_height,
                },
            )

            # Convert to InteractiveElement objects
            elements: list[InteractiveElement] = []
            for data in elements_data:
                element = self._data_to_element(data, screenshot_id)
                if element:
                    elements.append(element)

            # Log summary
            type_counts: dict[str, int] = {}
            for elem in elements:
                type_counts[elem.element_type] = type_counts.get(elem.element_type, 0) + 1
            logger.info(f"Extracted {len(elements)} interactive elements: {type_counts}")

            return elements

        except Exception as e:
            # Check if this is a shadow DOM specific error and gracefully degrade
            error_str = str(e).lower()
            if self.options.include_shadow_dom and (
                "shadow" in error_str or "shadowroot" in error_str
            ):
                logger.warning(
                    f"Shadow DOM extraction failed: {e}. "
                    "Falling back to regular DOM extraction."
                )
                # Retry without shadow DOM
                try:
                    elements_data = await page.evaluate(
                        """
                        (config) => {
                            const minWidth = config.minWidth;
                            const minHeight = config.minHeight;
                            const skipTags = new Set(config.skipTags);

                            const interactiveTags = new Set([
                                'button', 'a', 'input', 'select', 'textarea', 'label',
                                'details', 'summary', 'option', 'optgroup'
                            ]);

                            const interactiveRoles = new Set([
                                'button', 'link', 'menuitem', 'tab', 'checkbox',
                                'radio', 'switch', 'slider', 'textbox', 'combobox'
                            ]);

                            const results = [];
                            const scrollX = window.scrollX || window.pageXOffset;
                            const scrollY = window.scrollY || window.pageYOffset;

                            for (const el of document.querySelectorAll('*')) {
                                const tagName = el.tagName.toLowerCase();
                                if (skipTags.has(tagName)) continue;

                                const rect = el.getBoundingClientRect();
                                if (rect.width < minWidth || rect.height < minHeight) continue;

                                const style = window.getComputedStyle(el);
                                if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') continue;

                                const ariaRole = el.getAttribute('role');
                                let elementType = null;

                                if (interactiveTags.has(tagName)) {
                                    elementType = tagName;
                                } else if (ariaRole && interactiveRoles.has(ariaRole)) {
                                    elementType = 'aria_' + ariaRole;
                                } else if (el.getAttribute('tabindex') !== null) {
                                    elementType = 'tabindex_' + tagName;
                                } else if (el.onclick !== null || el.getAttribute('onclick') !== null) {
                                    elementType = 'onclick_' + tagName;
                                }

                                if (!elementType) continue;

                                let directText = '';
                                for (const node of el.childNodes) {
                                    if (node.nodeType === Node.TEXT_NODE) {
                                        directText += node.textContent.trim();
                                    }
                                }

                                let selector = tagName;
                                if (el.id) {
                                    selector = '#' + CSS.escape(el.id);
                                } else if (el.className && typeof el.className === 'string') {
                                    const classes = el.className.split(' ').filter(c => c && !c.includes(':')).slice(0, 2);
                                    if (classes.length > 0) {
                                        selector = tagName + '.' + classes.map(c => CSS.escape(c)).join('.');
                                    }
                                }

                                results.push({
                                    tagName: tagName,
                                    elementType: elementType,
                                    bbox: {
                                        x: Math.round(rect.x + scrollX),
                                        y: Math.round(rect.y + scrollY),
                                        width: Math.round(rect.width),
                                        height: Math.round(rect.height),
                                    },
                                    selector: selector,
                                    text: directText.substring(0, 200) || el.innerText?.trim().substring(0, 200) || null,
                                    href: el.href || el.getAttribute('href') || null,
                                    ariaLabel: el.getAttribute('aria-label') || null,
                                    ariaRole: ariaRole || null,
                                    shadowPath: null,
                                });
                            }
                            return results;
                        }
                        """,
                        {
                            "minWidth": actual_min_size[0],
                            "minHeight": actual_min_size[1],
                            "skipTags": list(SKIP_TAGS),
                        },
                    )

                    elements: list[InteractiveElement] = []
                    for data in elements_data:
                        element = self._data_to_element(data, screenshot_id)
                        if element:
                            elements.append(element)

                    logger.info(
                        f"Fallback extraction found {len(elements)} elements (without shadow DOM)"
                    )
                    return elements

                except Exception as fallback_error:
                    logger.error(f"Fallback extraction also failed: {fallback_error}")
                    raise ShadowDOMError(
                        f"Both shadow DOM and fallback extraction failed: {e}"
                    ) from fallback_error

            # Re-raise for retry logic to handle
            logger.error(f"Failed to extract elements: {e}")
            raise

    def _data_to_element(
        self,
        data: dict[str, Any],
        screenshot_id: str,
    ) -> InteractiveElement | None:
        """Convert JavaScript data to InteractiveElement."""
        try:
            bbox = BoundingBox(
                x=data["bbox"]["x"],
                y=data["bbox"]["y"],
                width=data["bbox"]["width"],
                height=data["bbox"]["height"],
            )

            return InteractiveElement(
                id=self._generate_id(),
                bbox=bbox,
                tag_name=data["tagName"],
                element_type=data["elementType"],
                screenshot_id=screenshot_id,
                selector=data["selector"],
                text=data.get("text"),
                href=data.get("href"),
                aria_label=data.get("ariaLabel"),
                aria_role=data.get("ariaRole"),
                shadow_path=data.get("shadowPath"),
            )

        except Exception as e:
            logger.warning(f"Failed to convert element data: {e}")
            return None

    async def extract_with_frames(
        self,
        page: Page,
        screenshot_id: str,
    ) -> FrameExtractionResult:
        """
        Extract interactive elements from all frames (main + iframes).

        Uses the frame_manager module for multi-frame extraction with
        frame-aware element IDs.

        Args:
            page: Playwright page to extract from.
            screenshot_id: ID of the page screenshot (for reference).

        Returns:
            FrameExtractionResult with elements from all frames.
        """
        from .frame_manager import extract_across_frames

        logger.info("Extracting interactive elements from all frames...")

        return await extract_across_frames(
            page,
            screenshot_id=screenshot_id,
            include_shadow_dom=self.options.include_shadow_dom,
            min_size=(self.options.min_width, self.options.min_height),
        )

    async def extract_with_stability(
        self,
        page: Page,
        screenshot_id: str,
        stability_ms: int = 100,
        max_wait_ms: int = 3000,
    ) -> list[InteractiveElement]:
        """
        Wait for DOM stability, then extract interactive elements.

        Uses the dom_stability module to wait for the page to settle
        before extraction, which helps with dynamic content.

        Performance optimized defaults:
        - stability_ms: 100ms (reduced from 500ms)
        - max_wait_ms: 3000ms (reduced from 5000ms)

        Args:
            page: Playwright page to extract from.
            screenshot_id: ID of the page screenshot.
            stability_ms: Time without mutations for stability (default: 100ms).
            max_wait_ms: Maximum wait time (default: 3000ms).

        Returns:
            List of InteractiveElement objects.
        """
        from .dom_stability import wait_for_stable_extraction

        logger.info("Waiting for DOM stability before extraction...")

        result = await wait_for_stable_extraction(
            page,
            stability_ms=stability_ms,
            max_wait_ms=max_wait_ms,
        )

        if result.stable:
            logger.info(
                f"DOM stable after {result.wait_time_ms:.0f}ms "
                f"({result.mutation_count} mutations)"
            )
        else:
            logger.warning(
                f"DOM did not stabilize within {max_wait_ms}ms "
                f"({result.mutation_count} mutations detected)"
            )

        return await self.extract_interactive_elements(page, screenshot_id)

    async def extract_full(
        self,
        page: Page,
        screenshot_id: str,
        include_iframes: bool = True,
        wait_for_stability: bool = True,
        stability_ms: int = 100,
        max_wait_ms: int = 3000,
    ) -> list[InteractiveElement] | FrameExtractionResult:
        """
        Full extraction with all enhanced capabilities.

        Combines DOM stability waiting, shadow DOM extraction, and
        iframe traversal for comprehensive element extraction.

        Performance optimized defaults:
        - stability_ms: 100ms (reduced from 500ms)
        - max_wait_ms: 3000ms (reduced from 5000ms)

        Args:
            page: Playwright page to extract from.
            screenshot_id: ID of the page screenshot.
            include_iframes: Whether to extract from iframes (default: True).
            wait_for_stability: Whether to wait for DOM stability (default: True).
            stability_ms: Time without mutations for stability (default: 100ms).
            max_wait_ms: Maximum wait time for stability (default: 3000ms).

        Returns:
            FrameExtractionResult if include_iframes, else list of InteractiveElement.
        """
        # Wait for stability if requested
        if wait_for_stability:
            from .dom_stability import wait_for_stable_extraction

            result = await wait_for_stable_extraction(
                page,
                stability_ms=stability_ms,
                max_wait_ms=max_wait_ms,
            )
            logger.debug(
                f"Stability result: stable={result.stable}, "
                f"wait={result.wait_time_ms:.0f}ms, mutations={result.mutation_count}"
            )

        # Extract with or without iframes
        if include_iframes:
            return await self.extract_with_frames(page, screenshot_id)
        else:
            return await self.extract_interactive_elements(page, screenshot_id)
