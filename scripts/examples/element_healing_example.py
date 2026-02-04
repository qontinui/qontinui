#!/usr/bin/env python
"""
Element healing example using qontinui's selector healer.

Demonstrates how the SelectorHealer can automatically repair broken
CSS selectors when the DOM changes.

Usage:
    poetry run python scripts/examples/element_healing_example.py
    poetry run python scripts/examples/element_healing_example.py --no-headless

Features demonstrated:
- Selector healing strategies (variations, text match, aria match)
- Healing history (learning from past repairs)
- Strategy statistics
- Position-based healing
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import tempfile
from pathlib import Path

from playwright.async_api import Page, async_playwright

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from qontinui.extraction.web import (
    BoundingBox,
    HealingHistory,
    HealingResult,
    InteractiveElement,
    SelectorHealer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_mock_element(
    selector: str,
    tag_name: str = "button",
    text: str | None = None,
    aria_label: str | None = None,
    bbox: tuple[int, int, int, int] = (100, 100, 100, 50),
) -> InteractiveElement:
    """Create a mock InteractiveElement for testing."""
    return InteractiveElement(
        id="mock_elem_001",
        bbox=BoundingBox(x=bbox[0], y=bbox[1], width=bbox[2], height=bbox[3]),
        tag_name=tag_name,
        element_type=tag_name,
        screenshot_id="mock_screenshot",
        selector=selector,
        text=text,
        aria_label=aria_label,
    )


async def demo_selector_variations(page: Page):
    """
    Demo: Selector variations strategy.

    Shows how the healer tries variations of a broken selector.
    """
    logger.info("=" * 60)
    logger.info("DEMO: Selector Variations Strategy")
    logger.info("=" * 60)

    healer = SelectorHealer()

    # Simulate a broken selector with nth-child
    broken_selector = "div.container > ul.nav-list > li:nth-child(3) > a.nav-link"
    element = create_mock_element(
        selector=broken_selector,
        tag_name="a",
        text="Navigation Link",
    )

    logger.info(f"Original selector: {broken_selector}")
    logger.info("Simulating selector breakage (nth-child changed)...")

    # The healer generates variations internally
    # Let's show what variations it would try
    variations = healer._generate_selector_variations(broken_selector)

    logger.info("\nGenerated variations:")
    for i, var in enumerate(variations):
        logger.info(f"  {i + 1}. {var}")

    # Now try healing on an actual page
    logger.info("\n--- Attempting to heal on actual page ---")

    result = await healer.heal_selector(broken_selector, element, page)

    logger.info("\nHealing result:")
    logger.info(f"  Success: {result.success}")
    logger.info(f"  Strategy used: {result.strategy_used}")
    logger.info(f"  Confidence: {result.confidence:.2f}")
    if result.healed_selector:
        logger.info(f"  Healed selector: {result.healed_selector}")

    logger.info(f"\nAttempts made: {len(result.attempts)}")
    for attempt in result.attempts[:5]:  # Show first 5 attempts
        status = "SUCCESS" if attempt.success else "failed"
        logger.info(f"  - [{attempt.strategy}] {attempt.selector_tried[:50]}... ({status})")


async def demo_text_and_aria_healing(page: Page):
    """
    Demo: Text and aria-label based healing.

    Shows how the healer can find elements by their text content
    or aria-label when the selector breaks.
    """
    logger.info("=" * 60)
    logger.info("DEMO: Text and Aria-Label Healing")
    logger.info("=" * 60)

    healer = SelectorHealer()

    # Element with text content
    text_element = create_mock_element(
        selector="#old-button-id",  # This ID might have changed
        tag_name="button",
        text="Sign In",
    )

    logger.info(f"Element with text: '{text_element.text}'")
    logger.info(f"Broken selector: {text_element.selector}")

    result = await healer.heal_selector(
        text_element.selector,
        text_element,
        page,
    )

    logger.info("\nHealing result (text-based):")
    logger.info(f"  Success: {result.success}")
    logger.info(f"  Strategy: {result.strategy_used}")
    if result.healed_selector:
        logger.info(f"  Healed selector: {result.healed_selector}")

    # Element with aria-label
    aria_element = create_mock_element(
        selector="#old-search-button",
        tag_name="button",
        aria_label="Search",
    )

    logger.info(f"\nElement with aria-label: '{aria_element.aria_label}'")
    logger.info(f"Broken selector: {aria_element.selector}")

    result = await healer.heal_selector(
        aria_element.selector,
        aria_element,
        page,
    )

    logger.info("\nHealing result (aria-based):")
    logger.info(f"  Success: {result.success}")
    logger.info(f"  Strategy: {result.strategy_used}")
    if result.healed_selector:
        logger.info(f"  Healed selector: {result.healed_selector}")


async def demo_healing_history():
    """
    Demo: Healing history and learning.

    Shows how the healer learns from past repairs and uses that
    knowledge to fix similar selectors faster.
    """
    logger.info("=" * 60)
    logger.info("DEMO: Healing History and Learning")
    logger.info("=" * 60)

    # Create a temporary file for history storage
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        history_path = Path(f.name)

    logger.info(f"History file: {history_path}")

    # Create history instance
    history = HealingHistory(storage_path=history_path)

    # Simulate some past healing records
    logger.info("\n--- Adding simulated healing records ---")

    # Record 1: Navigation link selector change
    mock_result_1 = HealingResult(
        success=True,
        original_selector="nav.main-nav > a.link:nth-child(1)",
        healed_selector="nav.main-nav > a.link:first-child",
        element=None,
        confidence=0.9,
        strategy_used="selector_variation",
    )
    mock_element_1 = create_mock_element(
        selector="nav.main-nav > a.link:nth-child(1)",
        tag_name="a",
        text="Home",
    )
    history.add_record(mock_result_1, "https://example.com", mock_element_1)
    logger.info("  Added: nav link nth-child -> first-child")

    # Record 2: Button ID change
    mock_result_2 = HealingResult(
        success=True,
        original_selector="#submit-btn-v1",
        healed_selector="button[type='submit']",
        element=None,
        confidence=0.85,
        strategy_used="text_match",
    )
    mock_element_2 = create_mock_element(
        selector="#submit-btn-v1",
        tag_name="button",
        text="Submit",
    )
    history.add_record(mock_result_2, "https://example.com", mock_element_2)
    logger.info("  Added: submit button ID -> type selector")

    # Record 3: Same selector fixed multiple times (increases priority)
    for _ in range(3):
        mock_result_3 = HealingResult(
            success=True,
            original_selector=".menu-item.active",
            healed_selector=".menu-item[aria-current='page']",
            element=None,
            confidence=0.95,
            strategy_used="aria_match",
        )
        mock_element_3 = create_mock_element(
            selector=".menu-item.active",
            tag_name="div",
            aria_label="Current Page",
        )
        history.add_record(mock_result_3, "https://example.com", mock_element_3)
    logger.info("  Added: menu item active class (3x - high priority)")

    # Show history contents
    logger.info(f"\nHistory now contains {len(history.records)} records")

    # Look up past healings
    logger.info("\n--- Looking up past healings ---")

    # Query 1: Exact match
    records = history.lookup(".menu-item.active")
    logger.info(f"Lookup '.menu-item.active': {len(records)} matches")
    if records:
        logger.info(f"  Best match: {records[0].healed_selector}")
        logger.info(f"  Success count: {records[0].success_count}")

    # Query 2: Pattern match
    records = history.lookup("nav.main-nav > a.link:nth-child(5)")
    logger.info(f"Lookup 'nav.main-nav > a.link:nth-child(5)': {len(records)} matches")
    if records:
        logger.info(f"  Pattern match: {records[0].healed_selector}")

    # Show strategy statistics
    logger.info("\n--- Strategy Statistics ---")
    stats = history.get_strategy_stats()
    for strategy, data in stats.items():
        logger.info(f"  {strategy}:")
        logger.info(f"    Total uses: {data['total_uses']}")
        logger.info(f"    Success count: {data['success_count']}")
        logger.info(f"    Avg confidence: {data['avg_confidence']:.2f}")

    # Clean up
    history_path.unlink(missing_ok=True)


async def demo_healing_with_history(page: Page):
    """
    Demo: Using healing history for faster repairs.

    Shows how history-based lookup is tried first before
    other strategies.
    """
    logger.info("=" * 60)
    logger.info("DEMO: Healing with History")
    logger.info("=" * 60)

    # Create healer with in-memory history
    history = HealingHistory()
    healer = SelectorHealer(history=history)

    # First healing attempt (no history)
    element1 = create_mock_element(
        selector="a.nav-link",
        tag_name="a",
        text="About",
    )

    logger.info("--- First healing (no history) ---")
    result1 = await healer.heal_selector("a.nav-link.old-class", element1, page)

    logger.info(f"Result: success={result1.success}, strategy={result1.strategy_used}")
    if result1.healed_selector:
        logger.info(f"Healed to: {result1.healed_selector}")

    # Record is automatically added to history if successful
    logger.info(f"\nHistory now has {len(history.records)} records")

    # Second healing attempt (should use history first)
    element2 = create_mock_element(
        selector="a.nav-link.old-class",  # Same broken selector
        tag_name="a",
        text="Contact",
    )

    logger.info("\n--- Second healing (with history) ---")
    result2 = await healer.heal_selector("a.nav-link.old-class", element2, page)

    logger.info(f"Result: success={result2.success}, strategy={result2.strategy_used}")

    # Check if history lookup was tried first
    strategies_tried = [a.strategy for a in result2.attempts]
    logger.info(f"Strategies tried (in order): {strategies_tried[:5]}")

    if "history_lookup" in strategies_tried:
        history_idx = strategies_tried.index("history_lookup")
        logger.info(f"History lookup was attempted at position {history_idx + 1}")


async def demo_all_strategies(page: Page):
    """
    Demo: Overview of all healing strategies.

    Provides a comprehensive look at what the healer can do.
    """
    logger.info("=" * 60)
    logger.info("DEMO: All Healing Strategies")
    logger.info("=" * 60)

    strategies = [
        {
            "name": "History Lookup",
            "description": "Uses past successful healings to fix similar selectors",
            "priority": 1,
            "confidence": "0.95 (very high)",
        },
        {
            "name": "Selector Variation",
            "description": "Tries variations: remove nth-child, change combinators, etc.",
            "priority": 2,
            "confidence": "0.90 (high)",
        },
        {
            "name": "Text Match",
            "description": "Finds element by its visible text content",
            "priority": 3,
            "confidence": "0.85 (high)",
        },
        {
            "name": "Aria Match",
            "description": "Finds element by its aria-label attribute",
            "priority": 4,
            "confidence": "0.85 (high)",
        },
        {
            "name": "Position Match",
            "description": "Finds element near the original position (within tolerance)",
            "priority": 5,
            "confidence": "0.70 (medium)",
        },
        {
            "name": "LLM Recovery",
            "description": "Uses AI to find similar element by description",
            "priority": 6,
            "confidence": "0.75 (medium)",
        },
    ]

    logger.info("\nHealing strategies (in order of priority):\n")

    for strategy in strategies:
        logger.info(f"{strategy['priority']}. {strategy['name']}")
        logger.info(f"   Description: {strategy['description']}")
        logger.info(f"   Confidence: {strategy['confidence']}")
        logger.info("")

    logger.info("The healer tries strategies in order until one succeeds.")
    logger.info("Higher confidence means more reliable match.")


async def main():
    parser = argparse.ArgumentParser(
        description="Element healing example using qontinui SelectorHealer"
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
        choices=["variations", "text", "history", "with-history", "overview", "all"],
        default="all",
        help="Which demo to run (default: all)",
    )
    args = parser.parse_args()

    headless = not args.no_headless

    logger.info("Element Healing Example")
    logger.info(f"URL: {args.url}")
    logger.info(f"Headless: {headless}")
    logger.info("")

    # Run demos that don't need a browser first
    if args.demo in ("history", "all"):
        await demo_healing_history()

    if args.demo in ("overview", "all"):
        await demo_all_strategies(None)

    # Run demos that need a browser
    if args.demo in ("variations", "text", "with-history", "all"):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=headless)
            page = await browser.new_page()

            logger.info(f"Navigating to: {args.url}")
            try:
                await page.goto(args.url, wait_until="domcontentloaded", timeout=30000)
                await page.wait_for_timeout(2000)
            except Exception as e:
                logger.error(f"Failed to load page: {e}")
                await browser.close()
                return

            if args.demo in ("variations", "all"):
                await demo_selector_variations(page)

            if args.demo in ("text", "all"):
                await demo_text_and_aria_healing(page)

            if args.demo in ("with-history", "all"):
                await demo_healing_with_history(page)

            await browser.close()

    logger.info("\n" + "=" * 60)
    logger.info("Demo complete!")
    logger.info("=" * 60)
    logger.info("\nKey takeaways:")
    logger.info("1. Healing history learns from past repairs")
    logger.info("2. Multiple strategies are tried in order")
    logger.info("3. Text and aria-label provide reliable fallbacks")
    logger.info("4. History-based healing is fastest and most reliable")


if __name__ == "__main__":
    asyncio.run(main())
