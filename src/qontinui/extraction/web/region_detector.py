"""
Region detector for web extraction.

Detects UI regions (menus, dialogs, navigation bars, etc.) that
represent potential states in the GUI.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from playwright.async_api import ElementHandle, Page

from .models import BoundingBox, StateType

logger = logging.getLogger(__name__)


@dataclass
class DetectedRegion:
    """A detected UI region."""

    id: str
    bbox: BoundingBox
    region_type: StateType
    selector: str
    semantic_role: str | None = None
    aria_label: str | None = None
    element_count: int = 0
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class RegionDetector:
    """Detects UI regions that represent potential states."""

    # Selectors that typically define state boundaries
    STATE_SELECTORS: dict[StateType, list[str]] = {
        StateType.DIALOG: [
            "dialog",
            '[role="dialog"]',
            '[role="alertdialog"]',
            ".modal",
            ".dialog",
            '[class*="modal"]',
            '[class*="dialog"]',
        ],
        StateType.MENU: [
            '[role="menu"]',
            '[role="menubar"]',
            ".menu",
            '[class*="menu"]:not([class*="menuitem"])',
        ],
        StateType.DROPDOWN_MENU: [
            '[role="listbox"]',
            ".dropdown-menu",
            ".dropdown-content",
            '[class*="dropdown"]:not(button):not(select)',
            '[class*="popover"]',
        ],
        StateType.NAVIGATION: [
            "nav",
            '[role="navigation"]',
            ".nav",
            ".navbar",
            '[class*="navigation"]',
        ],
        StateType.SIDEBAR: [
            "aside",
            '[role="complementary"]',
            ".sidebar",
            '[class*="sidebar"]',
            '[class*="side-panel"]',
        ],
        StateType.TOOLBAR: [
            '[role="toolbar"]',
            ".toolbar",
            '[class*="toolbar"]',
        ],
        StateType.HEADER: [
            "header",
            '[role="banner"]',
            ".header",
            '[class*="header"]',
        ],
        StateType.FOOTER: [
            "footer",
            '[role="contentinfo"]',
            ".footer",
            '[class*="footer"]',
        ],
        StateType.FORM: [
            "form",
            '[role="form"]',
            ".form",
        ],
        StateType.TOAST: [
            '[role="alert"]',
            '[role="status"]',
            ".toast",
            ".notification",
            '[class*="toast"]',
            '[class*="snackbar"]',
        ],
        StateType.TOOLTIP: [
            '[role="tooltip"]',
            ".tooltip",
            '[class*="tooltip"]',
        ],
        StateType.POPOVER: [
            ".popover",
            '[class*="popover"]',
        ],
        StateType.CARD: [
            '[class*="card"]',
        ],
        StateType.PANEL: [
            '[class*="panel"]',
            '[class*="pane"]',
        ],
    }

    def __init__(self):
        self._region_counter = 0

    def _generate_id(self) -> str:
        """Generate a unique region ID."""
        self._region_counter += 1
        return f"region_{self._region_counter:04d}"

    async def detect_regions(
        self,
        page: Page,
        min_size: tuple[int, int] = (50, 50),
    ) -> list[DetectedRegion]:
        """
        Detect all UI regions on the current page.

        Args:
            page: Playwright page to analyze.
            min_size: Minimum (width, height) for detected regions.

        Returns:
            List of detected regions.
        """
        regions: list[DetectedRegion] = []
        seen_selectors: set[str] = set()

        for region_type, selectors in self.STATE_SELECTORS.items():
            for selector in selectors:
                try:
                    elements = await page.query_selector_all(selector)

                    for element in elements:
                        try:
                            region = await self._extract_region(
                                element, region_type, selector, min_size
                            )
                            if region and region.selector not in seen_selectors:
                                regions.append(region)
                                seen_selectors.add(region.selector)
                        except Exception as e:
                            logger.debug(f"Error processing region element: {e}")

                except Exception as e:
                    logger.debug(f"Error querying selector {selector}: {e}")

        # Remove overlapping regions (keep larger ones)
        regions = self._filter_overlapping(regions)

        logger.info(f"Detected {len(regions)} UI regions")
        return regions

    async def _extract_region(
        self,
        element: ElementHandle,
        region_type: StateType,
        matched_selector: str,
        min_size: tuple[int, int],
    ) -> DetectedRegion | None:
        """Extract region information from an element."""
        # Check visibility
        is_visible = await element.is_visible()
        if not is_visible:
            return None

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

        # Check minimum size
        if bbox.width < min_size[0] or bbox.height < min_size[1]:
            return None

        # Get semantic info
        properties = await element.evaluate(
            """el => ({
            role: el.getAttribute('role'),
            ariaLabel: el.getAttribute('aria-label'),
            tagName: el.tagName.toLowerCase(),
            id: el.id || null,
            className: el.className || '',
            childCount: el.children.length,
        })"""
        )

        # Generate unique selector
        selector = await self._generate_selector(element, properties)

        # Count interactive elements inside
        element_count = await element.evaluate(
            """el => {
            const interactives = el.querySelectorAll(
                'button, a, input, select, textarea, [role="button"], [role="link"]'
            );
            return interactives.length;
        }"""
        )

        return DetectedRegion(
            id=self._generate_id(),
            bbox=bbox,
            region_type=region_type,
            selector=selector,
            semantic_role=properties.get("role"),
            aria_label=properties.get("ariaLabel"),
            element_count=element_count,
            confidence=1.0,
            metadata={
                "matched_selector": matched_selector,
                "tag_name": properties.get("tagName"),
                "child_count": properties.get("childCount"),
            },
        )

    async def _generate_selector(self, element: ElementHandle, properties: dict[str, Any]) -> str:
        """Generate a CSS selector for the region."""
        try:
            selector = await element.evaluate(
                """el => {
                if (el.id) {
                    return '#' + CSS.escape(el.id);
                }

                const testAttrs = ['data-testid', 'data-test-id', 'data-cy'];
                for (const attr of testAttrs) {
                    const value = el.getAttribute(attr);
                    if (value) {
                        return `[${attr}="${CSS.escape(value)}"]`;
                    }
                }

                // Build path
                const path = [];
                let current = el;
                while (current && current !== document.body) {
                    let selector = current.tagName.toLowerCase();

                    if (current.id) {
                        path.unshift('#' + CSS.escape(current.id));
                        break;
                    }

                    const role = current.getAttribute('role');
                    if (role) {
                        selector = `[role="${role}"]`;
                    } else if (current.className) {
                        const mainClass = current.className.split(' ')
                            .find(c => c && !c.includes(':'));
                        if (mainClass) {
                            selector += '.' + CSS.escape(mainClass);
                        }
                    }

                    path.unshift(selector);
                    current = current.parentElement;
                }

                return path.slice(-3).join(' > ');
            }"""
            )
            return selector
        except Exception:
            return properties.get("tagName", "div")

    def _filter_overlapping(self, regions: list[DetectedRegion]) -> list[DetectedRegion]:
        """Remove smaller regions that are fully contained in larger ones."""
        if len(regions) <= 1:
            return regions

        # Sort by area (largest first)
        sorted_regions = sorted(regions, key=lambda r: r.bbox.area, reverse=True)

        filtered: list[DetectedRegion] = []
        for region in sorted_regions:
            is_contained = False
            for existing in filtered:
                if existing.bbox.contains(region.bbox):
                    # Only filter if same type or existing is more important
                    if region.region_type == existing.region_type or self._is_more_important(
                        existing.region_type, region.region_type
                    ):
                        is_contained = True
                        break

            if not is_contained:
                filtered.append(region)

        return filtered

    def _is_more_important(self, type_a: StateType, type_b: StateType) -> bool:
        """Check if type_a is more important than type_b."""
        # Dialogs and modals are always important
        important_types = {StateType.DIALOG, StateType.MODAL}
        if type_a in important_types:
            return True
        return False

    async def detect_dynamic_regions(
        self,
        page: Page,
        before_action: dict[str, Any],
        after_action: dict[str, Any],
    ) -> list[DetectedRegion]:
        """
        Detect regions that appeared after an action.

        This is useful for finding dropdown menus, tooltips, etc.
        that only appear after user interaction.

        Args:
            page: Playwright page.
            before_action: Snapshot before action.
            after_action: Snapshot after action.

        Returns:
            List of newly appeared regions.
        """
        # Get current regions
        current_regions = await self.detect_regions(page)

        # Filter to only new regions
        before_selectors = set(before_action.get("region_selectors", []))
        new_regions = [r for r in current_regions if r.selector not in before_selectors]

        return new_regions

    async def get_region_hierarchy(
        self, page: Page, regions: list[DetectedRegion]
    ) -> dict[str, list[str]]:
        """
        Determine parent-child relationships between regions.

        Returns:
            Dict mapping region_id to list of child region_ids.
        """
        hierarchy: dict[str, list[str]] = {r.id: [] for r in regions}

        for region in regions:
            for other in regions:
                if region.id != other.id and region.bbox.contains(other.bbox):
                    hierarchy[region.id].append(other.id)

        return hierarchy

    async def classify_region_by_content(self, element: ElementHandle) -> StateType:
        """
        Classify a region based on its content analysis.

        This is a fallback when semantic hints aren't available.
        """
        analysis = await element.evaluate(
            """el => {
            const hasForm = el.querySelector('form, input, select, textarea') !== null;
            const hasNav = el.querySelector('a, nav') !== null;
            const hasButtons = el.querySelectorAll('button').length;
            const hasLinks = el.querySelectorAll('a').length;
            const hasInputs = el.querySelectorAll('input, select, textarea').length;
            const isAtTop = el.getBoundingClientRect().top < 100;
            const isAtBottom = el.getBoundingClientRect().bottom > window.innerHeight - 100;
            const isAtSide = el.getBoundingClientRect().left < 100 ||
                            el.getBoundingClientRect().right > window.innerWidth - 100;

            return {
                hasForm, hasNav, hasButtons, hasLinks, hasInputs,
                isAtTop, isAtBottom, isAtSide
            };
        }"""
        )

        if analysis["hasForm"] and analysis["hasInputs"] > 2:
            return StateType.FORM
        if analysis["isAtTop"] and analysis["hasNav"]:
            return StateType.HEADER
        if analysis["isAtBottom"]:
            return StateType.FOOTER
        if analysis["isAtSide"] and analysis["hasLinks"] > 3:
            return StateType.SIDEBAR
        if analysis["hasButtons"] > 3:
            return StateType.TOOLBAR

        return StateType.PANEL
