"""
Element classifier for web extraction.

Classifies DOM elements by analyzing their tag name, ARIA role,
attributes, and visual characteristics.
"""

import logging
from typing import Any, cast

from playwright.async_api import ElementHandle, Page

from .models import (
    BoundingBox,
    ElementType,
    ExtractedElement,
)

logger = logging.getLogger(__name__)


class ElementClassifier:
    """Classifies DOM elements by type."""

    # Selectors for each element type
    ELEMENT_SELECTORS: dict[ElementType, list[str]] = {
        ElementType.BUTTON: [
            "button",
            '[role="button"]',
            'input[type="button"]',
            'input[type="submit"]',
            'input[type="reset"]',
            '[class*="btn"]',
            '[class*="button"]',
        ],
        ElementType.TEXT_INPUT: [
            'input[type="text"]',
            'input[type="email"]',
            'input[type="tel"]',
            'input[type="url"]',
            'input[type="search"]',
            'input[type="number"]',
            "input:not([type])",  # Default input type is text
            '[role="textbox"]:not(textarea)',
            '[contenteditable="true"]',
        ],
        ElementType.PASSWORD_INPUT: [
            'input[type="password"]',
        ],
        ElementType.TEXTAREA: [
            "textarea",
            '[role="textbox"][aria-multiline="true"]',
        ],
        ElementType.LINK: [
            "a[href]",
            '[role="link"]',
        ],
        ElementType.DROPDOWN: [
            "select",
            '[role="combobox"]',
            '[role="listbox"]',
            '[class*="select"]',
            '[class*="dropdown"]',
        ],
        ElementType.CHECKBOX: [
            'input[type="checkbox"]',
            '[role="checkbox"]',
        ],
        ElementType.RADIO: [
            'input[type="radio"]',
            '[role="radio"]',
        ],
        ElementType.SLIDER: [
            'input[type="range"]',
            '[role="slider"]',
        ],
        ElementType.TOGGLE: [
            '[role="switch"]',
            '[class*="toggle"]',
            '[class*="switch"]',
        ],
        ElementType.TAB: [
            '[role="tab"]',
        ],
        ElementType.MENU_ITEM: [
            '[role="menuitem"]',
            '[role="menuitemcheckbox"]',
            '[role="menuitemradio"]',
        ],
        ElementType.ICON_BUTTON: [
            "button:has(svg)",
            "button:has(i)",
            '[role="button"]:has(svg)',
            '[role="button"]:has(i)',
        ],
        ElementType.IMAGE: [
            "img",
            '[role="img"]',
            "svg",
            "picture",
        ],
        ElementType.LABEL: [
            "label",
        ],
        ElementType.HEADING: [
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            '[role="heading"]',
        ],
        ElementType.PARAGRAPH: [
            "p",
        ],
        ElementType.LIST_ITEM: [
            "li",
            '[role="listitem"]',
        ],
        ElementType.TABLE_CELL: [
            "td",
            "th",
            '[role="cell"]',
            '[role="gridcell"]',
            '[role="columnheader"]',
            '[role="rowheader"]',
        ],
    }

    # Interactive element types
    INTERACTIVE_TYPES = {
        ElementType.BUTTON,
        ElementType.TEXT_INPUT,
        ElementType.PASSWORD_INPUT,
        ElementType.TEXTAREA,
        ElementType.LINK,
        ElementType.DROPDOWN,
        ElementType.CHECKBOX,
        ElementType.RADIO,
        ElementType.SLIDER,
        ElementType.TOGGLE,
        ElementType.TAB,
        ElementType.MENU_ITEM,
        ElementType.ICON_BUTTON,
    }

    def __init__(self):
        self._element_counter = 0

    def _generate_id(self) -> str:
        """Generate a unique element ID."""
        self._element_counter += 1
        return f"elem_{self._element_counter:04d}"

    async def get_all_interactive_selectors(self) -> str:
        """Get a combined selector for all interactive elements."""
        selectors = []
        for element_type in self.INTERACTIVE_TYPES:
            selectors.extend(self.ELEMENT_SELECTORS.get(element_type, []))
        return ", ".join(selectors)

    async def classify_element(self, element: ElementHandle) -> ElementType:
        """Determine element type from tag, role, and attributes."""
        try:
            # Get element properties
            tag_name = await element.evaluate("el => el.tagName.toLowerCase()")
            role = await element.get_attribute("role")
            input_type = await element.get_attribute("type")
            class_name = await element.get_attribute("class") or ""

            # Check role-based classification first (most specific)
            if role:
                role_lower = role.lower()
                role_mapping = {
                    "button": ElementType.BUTTON,
                    "link": ElementType.LINK,
                    "textbox": ElementType.TEXT_INPUT,
                    "checkbox": ElementType.CHECKBOX,
                    "radio": ElementType.RADIO,
                    "combobox": ElementType.DROPDOWN,
                    "listbox": ElementType.DROPDOWN,
                    "slider": ElementType.SLIDER,
                    "switch": ElementType.TOGGLE,
                    "tab": ElementType.TAB,
                    "menuitem": ElementType.MENU_ITEM,
                    "menuitemcheckbox": ElementType.MENU_ITEM,
                    "menuitemradio": ElementType.MENU_ITEM,
                    "img": ElementType.IMAGE,
                    "heading": ElementType.HEADING,
                    "listitem": ElementType.LIST_ITEM,
                    "cell": ElementType.TABLE_CELL,
                    "gridcell": ElementType.TABLE_CELL,
                }
                if role_lower in role_mapping:
                    return role_mapping[role_lower]

            # Check tag-based classification
            if tag_name == "button":
                # Check if it's an icon button
                has_icon = await element.evaluate(
                    "el => el.querySelector('svg, i, [class*=\"icon\"]') !== null"
                )
                if has_icon:
                    text = await element.inner_text()
                    if not text.strip():
                        return ElementType.ICON_BUTTON
                return ElementType.BUTTON

            if tag_name == "input":
                type_mapping = {
                    "button": ElementType.BUTTON,
                    "submit": ElementType.BUTTON,
                    "reset": ElementType.BUTTON,
                    "text": ElementType.TEXT_INPUT,
                    "email": ElementType.TEXT_INPUT,
                    "tel": ElementType.TEXT_INPUT,
                    "url": ElementType.TEXT_INPUT,
                    "search": ElementType.TEXT_INPUT,
                    "number": ElementType.TEXT_INPUT,
                    "password": ElementType.PASSWORD_INPUT,
                    "checkbox": ElementType.CHECKBOX,
                    "radio": ElementType.RADIO,
                    "range": ElementType.SLIDER,
                }
                return type_mapping.get(input_type or "text", ElementType.TEXT_INPUT)

            if tag_name == "a":
                return ElementType.LINK

            if tag_name == "select":
                return ElementType.DROPDOWN

            if tag_name == "textarea":
                return ElementType.TEXTAREA

            if tag_name in ("img", "svg", "picture"):
                return ElementType.IMAGE

            if tag_name == "label":
                return ElementType.LABEL

            if tag_name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                return ElementType.HEADING

            if tag_name == "p":
                return ElementType.PARAGRAPH

            if tag_name == "li":
                return ElementType.LIST_ITEM

            if tag_name in ("td", "th"):
                return ElementType.TABLE_CELL

            # Check class-based hints
            class_lower = class_name.lower()
            if "btn" in class_lower or "button" in class_lower:
                return ElementType.BUTTON
            if "dropdown" in class_lower or "select" in class_lower:
                return ElementType.DROPDOWN
            if "toggle" in class_lower or "switch" in class_lower:
                return ElementType.TOGGLE

            return ElementType.UNKNOWN

        except Exception as e:
            logger.warning(f"Error classifying element: {e}")
            return ElementType.UNKNOWN

    async def extract_element_info(
        self,
        element: ElementHandle,
        element_type: ElementType | None = None,
    ) -> ExtractedElement | None:
        """Extract full element information."""
        try:
            # Get bounding box
            bbox_dict = await element.bounding_box()
            if not bbox_dict:
                return None

            bbox = BoundingBox(
                x=int(bbox_dict["x"]),
                y=int(bbox_dict["y"]),
                width=int(bbox_dict["width"]),
                height=int(bbox_dict["height"]),
            )

            # Skip zero-size elements
            if bbox.width <= 0 or bbox.height <= 0:
                return None

            # Classify element if not provided
            if element_type is None:
                element_type = await self.classify_element(element)

            # Get element properties
            properties = await element.evaluate(
                """el => {
                const computedStyle = window.getComputedStyle(el);
                return {
                    tagName: el.tagName.toLowerCase(),
                    id: el.id || null,
                    className: el.className || '',
                    textContent: el.textContent?.trim().substring(0, 500) || null,
                    innerText: el.innerText?.trim().substring(0, 500) || null,
                    value: el.value || null,
                    placeholder: el.placeholder || null,
                    alt: el.alt || null,
                    href: el.href || null,
                    src: el.src || null,
                    type: el.type || null,
                    name: el.name || null,
                    disabled: el.disabled || false,
                    checked: el.checked || false,
                    selected: el.selected || false,
                    required: el.required || false,
                    readOnly: el.readOnly || false,
                    ariaLabel: el.getAttribute('aria-label'),
                    ariaExpanded: el.getAttribute('aria-expanded'),
                    ariaChecked: el.getAttribute('aria-checked'),
                    ariaSelected: el.getAttribute('aria-selected'),
                    ariaDisabled: el.getAttribute('aria-disabled'),
                    ariaHidden: el.getAttribute('aria-hidden'),
                    role: el.getAttribute('role'),
                    tabIndex: el.tabIndex,
                    isVisible: computedStyle.display !== 'none' &&
                               computedStyle.visibility !== 'hidden' &&
                               computedStyle.opacity !== '0',
                };
            }"""
            )

            # Generate selector
            selector = await self._generate_selector(element, properties)

            # Determine element state
            element_state = None
            if properties.get("checked"):
                element_state = "checked"
            elif properties.get("ariaExpanded") == "true":
                element_state = "expanded"
            elif properties.get("ariaSelected") == "true":
                element_state = "selected"

            # Get class names as list
            class_names = []
            if properties.get("className"):
                class_names = properties["className"].split()

            # Determine if interactive
            is_interactive = element_type in self.INTERACTIVE_TYPES

            # Determine if enabled
            is_enabled = not properties.get("disabled") and properties.get("ariaDisabled") != "true"

            # Determine if visible
            is_visible = (
                properties.get("isVisible", True) and properties.get("ariaHidden") != "true"
            )

            # Compute accessible name
            name = (
                properties.get("ariaLabel")
                or properties.get("alt")
                or properties.get("placeholder")
                or properties.get("innerText", "")[:100]
            )

            return ExtractedElement(
                id=self._generate_id(),
                bbox=bbox,
                element_type=element_type,
                selector=selector,
                text_content=properties.get("innerText"),
                placeholder=properties.get("placeholder"),
                value=properties.get("value"),
                alt_text=properties.get("alt"),
                semantic_role=properties.get("role"),
                aria_label=properties.get("ariaLabel"),
                name=name,
                is_interactive=is_interactive,
                is_enabled=is_enabled,
                is_visible=is_visible,
                is_focused=False,  # Would need to check document.activeElement
                element_state=element_state,
                attributes={
                    "href": properties.get("href"),
                    "src": properties.get("src"),
                    "type": properties.get("type"),
                    "required": properties.get("required"),
                    "readOnly": properties.get("readOnly"),
                    "tabIndex": properties.get("tabIndex"),
                },
                tag_name=properties.get("tagName", ""),
                class_names=class_names,
            )

        except Exception as e:
            logger.warning(f"Error extracting element info: {e}")
            return None

    async def _generate_selector(self, element: ElementHandle, properties: dict[str, Any]) -> str:
        """Generate a CSS selector for the element."""
        try:
            # Try to generate a unique selector
            selector = await element.evaluate(
                """el => {
                // Try ID first
                if (el.id) {
                    return '#' + CSS.escape(el.id);
                }

                // Try data-testid or similar attributes
                const testAttrs = ['data-testid', 'data-test-id', 'data-cy', 'data-test'];
                for (const attr of testAttrs) {
                    const value = el.getAttribute(attr);
                    if (value) {
                        return `[${attr}="${CSS.escape(value)}"]`;
                    }
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

                    // Add class if unique among siblings
                    if (current.className) {
                        const classes = current.className.split(' ')
                            .filter(c => c && !c.includes(':'))
                            .slice(0, 2);
                        if (classes.length) {
                            selector += '.' + classes.map(c => CSS.escape(c)).join('.');
                        }
                    }

                    // Add nth-child if needed
                    const parent = current.parentElement;
                    if (parent) {
                        const siblings = Array.from(parent.children)
                            .filter(s => s.tagName === current.tagName);
                        if (siblings.length > 1) {
                            const index = siblings.indexOf(current) + 1;
                            selector += `:nth-child(${index})`;
                        }
                    }

                    path.unshift(selector);
                    current = current.parentElement;
                }

                return path.join(' > ');
            }"""
            )
            return cast(str, selector)
        except Exception:
            # Fallback to basic selector
            tag = properties.get("tagName", "div")
            if properties.get("id"):
                return cast(str, f"#{properties['id']}")
            return cast(str, tag)

    async def extract_all_elements(
        self,
        page: Page,
        min_size: tuple[int, int] = (10, 10),
        max_size: tuple[int, int] = (2000, 2000),
        include_hidden: bool = False,
    ) -> list[ExtractedElement]:
        """Extract all elements from a page."""
        elements = []
        seen_selectors = set()

        # Get all interactive elements
        selector = await self.get_all_interactive_selectors()

        try:
            handles = await page.query_selector_all(selector)

            for handle in handles:
                try:
                    element = await self.extract_element_info(handle)
                    if element is None:
                        continue

                    # Skip if too small or too large
                    if (
                        element.bbox.width < min_size[0]
                        or element.bbox.height < min_size[1]
                        or element.bbox.width > max_size[0]
                        or element.bbox.height > max_size[1]
                    ):
                        continue

                    # Skip hidden elements unless requested
                    if not include_hidden and not element.is_visible:
                        continue

                    # Skip duplicates
                    if element.selector in seen_selectors:
                        continue
                    seen_selectors.add(element.selector)

                    elements.append(element)

                except Exception as e:
                    logger.debug(f"Error processing element: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error extracting elements: {e}")

        logger.info(f"Extracted {len(elements)} elements from page")
        return elements
