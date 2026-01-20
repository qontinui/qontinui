#!/usr/bin/env python
"""
Form filling example using qontinui's web extraction capabilities.

Demonstrates using NaturalLanguageSelector to find form fields and
fill them programmatically.

Usage:
    poetry run python scripts/examples/form_filling_example.py
    poetry run python scripts/examples/form_filling_example.py --url https://example.com/form
    poetry run python scripts/examples/form_filling_example.py --no-headless

Features demonstrated:
- Natural language element selection
- Finding form fields by description
- Using MockLLMClient for demo (no API key needed)
- Action selection (click, type, hover)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from playwright.async_api import Page, async_playwright

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from qontinui.extraction.web import (
    FallbackSelector,
    InteractiveElement,
    InteractiveElementExtractor,
    NaturalLanguageSelector,
    format_for_llm,
)
from qontinui.extraction.web.llm_clients import MockLLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class FormFillingDemo:
    """
    Demonstrates form filling using natural language element selection.

    Uses MockLLMClient so no API key is required for the demo.
    """

    def __init__(self, headless: bool = True):
        self.headless = headless
        self.extractor = InteractiveElementExtractor()

        # Create mock responses for common form queries
        # In production, use AnthropicClient or OpenAIClient instead
        mock_responses = {
            "username": """INDEX: 0
CONFIDENCE: 0.95
REASONING: Found username input field
ALTERNATIVES: none""",
            "email": """INDEX: 1
CONFIDENCE: 0.95
REASONING: Found email input field
ALTERNATIVES: none""",
            "password": """INDEX: 2
CONFIDENCE: 0.95
REASONING: Found password input field
ALTERNATIVES: none""",
            "search": """INDEX: 0
CONFIDENCE: 0.90
REASONING: Found search input
ALTERNATIVES: none""",
            "submit": """INDEX: 3
CONFIDENCE: 0.95
REASONING: Found submit button
ALTERNATIVES: none""",
            "login": """INDEX: 3
CONFIDENCE: 0.90
REASONING: Found login/submit button
ALTERNATIVES: none""",
            "sign": """INDEX: 4
CONFIDENCE: 0.85
REASONING: Found sign-in related element
ALTERNATIVES: none""",
        }

        self.llm_client = MockLLMClient(responses=mock_responses)
        self.selector = NaturalLanguageSelector(self.llm_client)
        self.fallback = FallbackSelector()

    async def extract_elements(self, page: Page) -> list[InteractiveElement]:
        """Extract interactive elements from the page."""
        elements = await self.extractor.extract_interactive_elements(page, "form_demo")
        logger.info(f"Extracted {len(elements)} interactive elements")
        return elements

    async def find_element(
        self,
        description: str,
        elements: list[InteractiveElement],
        use_llm: bool = True,
    ) -> InteractiveElement | None:
        """
        Find an element using natural language description.

        Args:
            description: Natural language description of the element
            elements: List of extracted elements
            use_llm: Whether to use LLM-based selection (or fallback to text matching)

        Returns:
            The matching element, or None if not found
        """
        if use_llm:
            result = await self.selector.find_element(description, elements)
        else:
            result = self.fallback.find_by_text(description, elements)

        if result.found:
            logger.info(f"Found element for '{description}':")
            logger.info(f"  Index: {result.index}")
            logger.info(f"  Confidence: {result.confidence:.2f}")
            logger.info(f"  Reasoning: {result.reasoning}")
            return result.element
        else:
            logger.warning(f"No element found for '{description}'")
            logger.warning(f"  Reason: {result.reasoning}")
            return None

    async def fill_form_field(
        self,
        page: Page,
        element: InteractiveElement,
        value: str,
    ) -> bool:
        """
        Fill a form field with a value.

        Args:
            page: Playwright page
            element: The element to fill
            value: The value to enter

        Returns:
            True if successful, False otherwise
        """
        try:
            # Try to find the element using its selector
            locator = page.locator(element.selector)

            # Check if element exists
            count = await locator.count()
            if count == 0:
                # Fallback: try by position (click then type)
                logger.info(f"  Using position-based click at {element.bbox.center}")
                await page.mouse.click(element.bbox.center[0], element.bbox.center[1])
                await page.keyboard.type(value)
                return True

            # Clear and fill
            await locator.first.fill(value)
            logger.info(f"  Filled with: {value}")
            return True

        except Exception as e:
            logger.error(f"  Failed to fill: {e}")
            return False

    async def click_element(
        self,
        page: Page,
        element: InteractiveElement,
    ) -> bool:
        """
        Click an element.

        Args:
            page: Playwright page
            element: The element to click

        Returns:
            True if successful, False otherwise
        """
        try:
            locator = page.locator(element.selector)
            count = await locator.count()

            if count == 0:
                # Fallback: click by position
                await page.mouse.click(element.bbox.center[0], element.bbox.center[1])
            else:
                await locator.first.click()

            logger.info("  Clicked element")
            return True

        except Exception as e:
            logger.error(f"  Failed to click: {e}")
            return False


async def demo_github_search(demo: FormFillingDemo, url: str):
    """
    Demo: Search on GitHub using natural language selection.

    This demonstrates finding a search input and performing a search.
    """
    logger.info("=" * 60)
    logger.info("DEMO: GitHub Search")
    logger.info("=" * 60)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=demo.headless)
        page = await browser.new_page()

        logger.info(f"Navigating to: {url}")
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(2000)
        except Exception as e:
            logger.error(f"Failed to load page: {e}")
            await browser.close()
            return

        # Extract elements
        elements = await demo.extract_elements(page)

        # Show what we extracted
        formatted = format_for_llm(elements)
        logger.info("\nExtracted elements (first 10):")
        for line in formatted.split("\n")[:10]:
            logger.info(f"  {line}")

        # Find the search input using natural language
        logger.info("\n--- Finding search input ---")
        search_input = await demo.find_element("the search input field", elements)

        if search_input:
            logger.info(f"  Tag: <{search_input.tag_name}>")
            logger.info(f"  Selector: {search_input.selector}")
            logger.info(f"  Aria-label: {search_input.aria_label}")

            # In a real scenario, we would fill the search field
            # await demo.fill_form_field(page, search_input, "qontinui automation")
            logger.info("  (Skipping actual fill to preserve demo state)")

        # Find buttons using natural language
        logger.info("\n--- Finding buttons ---")
        submit_button = await demo.find_element("a submit or search button", elements)

        if submit_button:
            logger.info(f"  Tag: <{submit_button.tag_name}>")
            logger.info(f"  Text: {submit_button.text}")
            logger.info(f"  Selector: {submit_button.selector}")

        await browser.close()


async def demo_action_selection(demo: FormFillingDemo, url: str):
    """
    Demo: Using action selection to determine what to do.

    The select_action method returns both the element AND the action to perform.
    """
    logger.info("=" * 60)
    logger.info("DEMO: Action Selection")
    logger.info("=" * 60)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=demo.headless)
        page = await browser.new_page()

        logger.info(f"Navigating to: {url}")
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(2000)
        except Exception as e:
            logger.error(f"Failed to load page: {e}")
            await browser.close()
            return

        elements = await demo.extract_elements(page)

        # Test various natural language instructions
        instructions = [
            "click the sign in button",
            "type in the search box",
            "hover over the navigation menu",
            "focus on the first input field",
        ]

        logger.info("\n--- Action Selection Results ---")
        for instruction in instructions:
            logger.info(f'\nInstruction: "{instruction}"')

            result, action = await demo.selector.select_action(instruction, elements)

            if result.found:
                logger.info(f"  Element: [{result.index}] <{result.element.tag_name}>")
                logger.info(f"  Text: {result.element.text or '(no text)'}")
                logger.info(f"  Action: {action}")
                logger.info(f"  Confidence: {result.confidence:.2f}")
            else:
                logger.info("  Could not determine action")
                logger.info(f"  Reason: {result.reasoning}")

        await browser.close()


async def demo_fallback_selection(demo: FormFillingDemo, url: str):
    """
    Demo: Using fallback selector (no LLM required).

    The fallback selector uses simple text matching and role-based queries.
    """
    logger.info("=" * 60)
    logger.info("DEMO: Fallback Selection (No LLM)")
    logger.info("=" * 60)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=demo.headless)
        page = await browser.new_page()

        logger.info(f"Navigating to: {url}")
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(2000)
        except Exception as e:
            logger.error(f"Failed to load page: {e}")
            await browser.close()
            return

        elements = await demo.extract_elements(page)

        # Text-based selection (works without LLM)
        logger.info("\n--- Text-based Selection ---")
        test_texts = ["Sign in", "Search", "About", "Help", "Contact"]

        for text in test_texts:
            result = demo.fallback.find_by_text(text, elements)
            if result.found:
                logger.info(
                    f"  '{text}' -> [{result.index}] <{result.element.tag_name}> (confidence: {result.confidence:.2f})"
                )
            else:
                logger.info(f"  '{text}' -> No match")

        # Role-based selection
        logger.info("\n--- Role-based Selection ---")
        for role in ["button", "link", "textbox"]:
            results = demo.fallback.find_by_role(role, elements)
            logger.info(f"  Role '{role}': Found {len(results)} elements")

            # Show top 3
            for r in results[:3]:
                text = r.element.text[:30] if r.element.text else "(no text)"
                logger.info(f"    [{r.index}] {text}")

        await browser.close()


async def demo_find_multiple(demo: FormFillingDemo, url: str):
    """
    Demo: Finding multiple matching elements.

    Useful when you want to find all buttons, all links, etc.
    """
    logger.info("=" * 60)
    logger.info("DEMO: Find Multiple Elements")
    logger.info("=" * 60)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=demo.headless)
        page = await browser.new_page()

        logger.info(f"Navigating to: {url}")
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(2000)
        except Exception as e:
            logger.error(f"Failed to load page: {e}")
            await browser.close()
            return

        elements = await demo.extract_elements(page)

        # Find multiple elements matching a description
        queries = [
            "navigation links",
            "form input fields",
            "call-to-action buttons",
        ]

        logger.info("\n--- Finding Multiple Elements ---")
        for query in queries:
            logger.info(f'\nQuery: "{query}"')

            results = await demo.selector.find_multiple(query, elements, max_results=5)

            if results:
                for result in results:
                    text = result.element.text[:30] if result.element.text else "(no text)"
                    logger.info(f"  [{result.index}] <{result.element.tag_name}> {text}")
                    logger.info(f"       Confidence: {result.confidence:.2f}")
            else:
                logger.info("  No matches found")

        await browser.close()


async def main():
    parser = argparse.ArgumentParser(
        description="Form filling example using qontinui NaturalLanguageSelector"
    )
    parser.add_argument(
        "--url",
        default="https://github.com",
        help="URL to test (default: https://github.com)",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Run browser in visible mode",
    )
    parser.add_argument(
        "--demo",
        choices=["search", "actions", "fallback", "multiple", "all"],
        default="all",
        help="Which demo to run (default: all)",
    )
    args = parser.parse_args()

    headless = not args.no_headless
    demo = FormFillingDemo(headless=headless)

    logger.info("Form Filling Example")
    logger.info("Using MockLLMClient (no API key required)")
    logger.info(f"URL: {args.url}")
    logger.info(f"Headless: {headless}")
    logger.info("")

    if args.demo in ("search", "all"):
        await demo_github_search(demo, args.url)

    if args.demo in ("actions", "all"):
        await demo_action_selection(demo, args.url)

    if args.demo in ("fallback", "all"):
        await demo_fallback_selection(demo, args.url)

    if args.demo in ("multiple", "all"):
        await demo_find_multiple(demo, args.url)

    logger.info("\n" + "=" * 60)
    logger.info("Demo complete!")
    logger.info("=" * 60)
    logger.info("\nTip: Use a real LLM provider for better results:")
    logger.info("  from qontinui.extraction.web.llm_clients import AnthropicClient")
    logger.info("  client = AnthropicClient()  # Uses ANTHROPIC_API_KEY env var")
    logger.info("  selector = NaturalLanguageSelector(client)")


if __name__ == "__main__":
    asyncio.run(main())
