"""
Deep locator resolution for cross-frame element access.

Provides utilities for resolving deep locator strings (e.g., "iframe#sidebar >> button.submit")
to actual Playwright locators, handling frame traversal transparently.

Inspired by Stagehand's deep locator syntax.
"""

import logging
from dataclasses import dataclass

from playwright.async_api import ElementHandle, Frame, Locator, Page

logger = logging.getLogger(__name__)


@dataclass
class DeepLocatorParts:
    """Parsed components of a deep locator string."""

    frame_selectors: list[str]  # Chain of frame selectors
    element_selector: str  # Final element selector
    original: str  # Original deep locator string

    @property
    def has_frames(self) -> bool:
        """Whether this locator crosses frame boundaries."""
        return len(self.frame_selectors) > 0

    @property
    def frame_depth(self) -> int:
        """Number of frame hops required."""
        return len(self.frame_selectors)


class DeepLocatorResolver:
    """
    Resolves deep locator strings to Playwright locators.

    Deep locators use the ">>" separator to indicate frame boundaries:
    - "button.submit" - element in main frame
    - "iframe#sidebar >> button.submit" - element in sidebar iframe
    - "iframe#outer >> iframe#inner >> button" - nested iframes

    This enables clean, human-readable selectors that work across frames.
    """

    FRAME_SEPARATOR = " >> "

    def __init__(self, page: Page):
        """
        Initialize the resolver.

        Args:
            page: Playwright Page to resolve locators against
        """
        self.page = page

    def parse(self, deep_locator: str) -> DeepLocatorParts:
        """
        Parse a deep locator string into its components.

        Args:
            deep_locator: The deep locator string

        Returns:
            DeepLocatorParts with frame selectors and element selector
        """
        parts = deep_locator.split(self.FRAME_SEPARATOR)

        if len(parts) == 1:
            # No frame separator - element in main frame
            return DeepLocatorParts(
                frame_selectors=[],
                element_selector=parts[0],
                original=deep_locator,
            )

        # Last part is the element selector, rest are frame selectors
        return DeepLocatorParts(
            frame_selectors=parts[:-1],
            element_selector=parts[-1],
            original=deep_locator,
        )

    async def resolve_frame(self, frame_selector: str) -> Frame | None:
        """
        Resolve a frame selector to a Frame object.

        Args:
            frame_selector: CSS selector for the iframe

        Returns:
            Frame object or None if not found
        """
        try:
            # Handle "main" special case
            if frame_selector == "main":
                return self.page.main_frame

            # Find the iframe element
            iframe_element = await self.page.query_selector(frame_selector)
            if not iframe_element:
                logger.warning(f"Frame not found: {frame_selector}")
                return None

            # Get the content frame
            frame = await iframe_element.content_frame()
            if not frame:
                logger.warning(f"No content frame for: {frame_selector}")
                return None

            return frame

        except Exception as e:
            logger.error(f"Error resolving frame {frame_selector}: {e}")
            return None

    async def resolve_frame_chain(self, frame_selectors: list[str]) -> Frame | None:
        """
        Resolve a chain of frame selectors to the deepest frame.

        Args:
            frame_selectors: List of frame selectors to traverse

        Returns:
            The deepest Frame, or None if any frame not found
        """
        if not frame_selectors:
            return self.page.main_frame

        current_frame = self.page.main_frame

        for frame_selector in frame_selectors:
            # Find iframe in current frame
            try:
                iframe_element = await current_frame.query_selector(frame_selector)
                if not iframe_element:
                    logger.warning(f"Frame not found in chain: {frame_selector}")
                    return None

                next_frame = await iframe_element.content_frame()
                if not next_frame:
                    logger.warning(f"No content frame for: {frame_selector}")
                    return None

                current_frame = next_frame

            except Exception as e:
                logger.error(f"Error traversing frame chain at {frame_selector}: {e}")
                return None

        return current_frame

    async def resolve(self, deep_locator: str) -> Locator | None:
        """
        Resolve a deep locator string to a Playwright Locator.

        Args:
            deep_locator: The deep locator string

        Returns:
            Playwright Locator or None if resolution failed
        """
        parts = self.parse(deep_locator)

        try:
            if not parts.has_frames:
                # Simple case - no frame traversal needed
                return self.page.locator(parts.element_selector)

            # Resolve frame chain
            target_frame = await self.resolve_frame_chain(parts.frame_selectors)
            if not target_frame:
                return None

            # Return locator in the target frame
            return target_frame.locator(parts.element_selector)

        except Exception as e:
            logger.error(f"Error resolving deep locator {deep_locator}: {e}")
            return None

    async def resolve_to_element(self, deep_locator: str) -> ElementHandle | None:
        """
        Resolve a deep locator to an ElementHandle.

        Args:
            deep_locator: The deep locator string

        Returns:
            ElementHandle or None if not found
        """
        locator = await self.resolve(deep_locator)
        if not locator:
            return None

        try:
            return await locator.element_handle()
        except Exception as e:
            logger.error(f"Error getting element handle for {deep_locator}: {e}")
            return None

    async def click(self, deep_locator: str) -> bool:
        """
        Click an element using a deep locator.

        Args:
            deep_locator: The deep locator string

        Returns:
            True if click succeeded, False otherwise
        """
        locator = await self.resolve(deep_locator)
        if not locator:
            return False

        try:
            await locator.click()
            return True
        except Exception as e:
            logger.error(f"Error clicking {deep_locator}: {e}")
            return False

    async def fill(self, deep_locator: str, value: str) -> bool:
        """
        Fill an input using a deep locator.

        Args:
            deep_locator: The deep locator string
            value: The value to fill

        Returns:
            True if fill succeeded, False otherwise
        """
        locator = await self.resolve(deep_locator)
        if not locator:
            return False

        try:
            await locator.fill(value)
            return True
        except Exception as e:
            logger.error(f"Error filling {deep_locator}: {e}")
            return False

    async def get_text(self, deep_locator: str) -> str | None:
        """
        Get text content using a deep locator.

        Args:
            deep_locator: The deep locator string

        Returns:
            Text content or None if not found
        """
        locator = await self.resolve(deep_locator)
        if not locator:
            return None

        try:
            return await locator.text_content()
        except Exception as e:
            logger.error(f"Error getting text from {deep_locator}: {e}")
            return None

    async def is_visible(self, deep_locator: str) -> bool:
        """
        Check if element is visible using a deep locator.

        Args:
            deep_locator: The deep locator string

        Returns:
            True if visible, False otherwise
        """
        locator = await self.resolve(deep_locator)
        if not locator:
            return False

        try:
            return await locator.is_visible()
        except Exception:
            return False


def build_deep_locator(frame_path: list[str], element_selector: str) -> str:
    """
    Build a deep locator string from frame path and element selector.

    Args:
        frame_path: List of frame selectors (empty for main frame)
        element_selector: The element's CSS selector

    Returns:
        Deep locator string
    """
    if not frame_path:
        return element_selector
    return " >> ".join(frame_path + [element_selector])


def is_deep_locator(selector: str) -> bool:
    """
    Check if a selector string is a deep locator.

    Args:
        selector: The selector string to check

    Returns:
        True if it contains frame separators
    """
    return " >> " in selector


async def resolve_deep_locator(page: Page, deep_locator: str) -> Locator | None:
    """
    Convenience function to resolve a deep locator.

    Args:
        page: Playwright Page
        deep_locator: The deep locator string

    Returns:
        Playwright Locator or None
    """
    resolver = DeepLocatorResolver(page)
    return await resolver.resolve(deep_locator)


async def click_deep(page: Page, deep_locator: str) -> bool:
    """
    Convenience function to click using a deep locator.

    Args:
        page: Playwright Page
        deep_locator: The deep locator string

    Returns:
        True if click succeeded
    """
    resolver = DeepLocatorResolver(page)
    return await resolver.click(deep_locator)


async def fill_deep(page: Page, deep_locator: str, value: str) -> bool:
    """
    Convenience function to fill using a deep locator.

    Args:
        page: Playwright Page
        deep_locator: The deep locator string
        value: The value to fill

    Returns:
        True if fill succeeded
    """
    resolver = DeepLocatorResolver(page)
    return await resolver.fill(deep_locator, value)
