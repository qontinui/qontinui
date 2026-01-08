"""
Interactive Element Extractor for web extraction.

Extracts interactive UI elements from the DOM using Playwright.
Only captures elements users can interact with: buttons, links, inputs.
"""

import logging
from typing import Any

from playwright.async_api import Page

from .models import BoundingBox, InteractiveElement

logger = logging.getLogger(__name__)

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
    """

    def __init__(self) -> None:
        """Initialize the extractor."""
        self._element_counter = 0

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
        min_size: tuple[int, int] = (10, 10),
    ) -> list[InteractiveElement]:
        """
        Extract interactive elements from a page.

        Args:
            page: Playwright page to extract from.
            screenshot_id: ID of the page screenshot (for reference).
            min_size: Minimum (width, height) for elements.

        Returns:
            List of InteractiveElement objects.
        """
        logger.info("Extracting interactive elements from DOM...")

        try:
            # Get all interactive element data in a single evaluate call
            elements_data = await page.evaluate(
                """
                (config) => {
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
                    const allElements = document.querySelectorAll('*');

                    // Get scroll position for converting viewport coords to page coords
                    const scrollX = window.scrollX || window.pageXOffset;
                    const scrollY = window.scrollY || window.pageYOffset;

                    for (const el of allElements) {
                        const tagName = el.tagName.toLowerCase();

                        // Skip non-visible tags
                        if (skipTags.has(tagName)) continue;

                        // Get bounding box
                        const rect = el.getBoundingClientRect();
                        if (rect.width < minWidth || rect.height < minHeight) continue;

                        // Check visibility
                        const style = window.getComputedStyle(el);
                        if (style.display === 'none') continue;
                        if (style.visibility === 'hidden') continue;
                        if (style.opacity === '0') continue;

                        // Get ARIA role
                        const ariaRole = el.getAttribute('role');

                        // Check for cursor pointer (indicates clickable in modern frameworks)
                        const hasCursorPointer = style.cursor === 'pointer';

                        // Determine if this is an interactive element
                        let elementType = null;

                        if (interactiveTags.has(tagName)) {
                            elementType = tagName;
                        }
                        else if (ariaRole && interactiveRoles.has(ariaRole)) {
                            elementType = 'aria_' + ariaRole;
                        }
                        else if (el.getAttribute('tabindex') !== null) {
                            elementType = 'tabindex_' + tagName;
                        }
                        else if (el.onclick !== null || el.getAttribute('onclick') !== null) {
                            elementType = 'onclick_' + tagName;
                        }
                        // Detect clickable elements via cursor:pointer (React, Vue, etc.)
                        // Be VERY restrictive to avoid capturing text content
                        else if (hasCursorPointer) {
                            // Skip text-content tags that often inherit cursor:pointer
                            const textTags = new Set(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'ul', 'ol', 'span', 'strong', 'em', 'b', 'i', 'blockquote', 'pre', 'code']);
                            if (textTags.has(tagName)) continue;

                            // Only capture div/section elements that look like buttons
                            const text = el.innerText?.trim() || '';

                            // Must have short text (like "Sign Up", not a paragraph)
                            if (text.length === 0 || text.length > 30) continue;

                            // Must not span multiple lines (button text is usually single line)
                            if (text.includes('\\n')) continue;

                            // Must be reasonably button-sized (not full-width containers)
                            if (rect.width > 300 || rect.height > 60) continue;

                            // Must not contain interactive children (avoid capturing parent containers)
                            const hasInteractiveChildren = el.querySelector('button, a, input, select, textarea, [role="button"], [role="link"]');
                            if (hasInteractiveChildren) continue;

                            elementType = 'clickable_' + tagName;
                        }

                        // Skip non-interactive elements
                        if (!elementType) continue;

                        // Get text content
                        let directText = '';
                        for (const node of el.childNodes) {
                            if (node.nodeType === Node.TEXT_NODE) {
                                directText += node.textContent.trim();
                            }
                        }
                        const fullText = el.innerText?.trim() || '';

                        // Generate a CSS selector
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
                                // Convert viewport coords to page coords for full-page screenshots
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
                        });
                    }

                    return results;
                }
                """,
                {
                    "minWidth": min_size[0],
                    "minHeight": min_size[1],
                    "skipTags": list(SKIP_TAGS),
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
            logger.error(f"Failed to extract elements: {e}")
            return []

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
            )

        except Exception as e:
            logger.warning(f"Failed to convert element data: {e}")
            return None
