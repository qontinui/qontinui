#!/usr/bin/env python
"""
Demonstration of LLM-powered natural language element selection.

This script shows how to use the NaturalLanguageSelector with various
LLM providers to find elements using natural language descriptions.

Usage:
    # With mock LLM (no API key needed)
    poetry run python scripts/demo_llm_selector.py --mock

    # With Anthropic Claude
    poetry run python scripts/demo_llm_selector.py --provider anthropic

    # With OpenAI GPT
    poetry run python scripts/demo_llm_selector.py --provider openai

    # Custom URL
    poetry run python scripts/demo_llm_selector.py --url https://example.com --mock
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from playwright.async_api import async_playwright

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qontinui.extraction.web import (
    FallbackSelector,
    InteractiveElementExtractor,
    NaturalLanguageSelector,
    format_for_llm,
)
from qontinui.extraction.web.llm_clients import (
    MockLLMClient,
    create_llm_client,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Test queries to demonstrate natural language selection
DEMO_QUERIES = [
    "the search input field",
    "a button to submit",
    "the main navigation menu",
    "link to sign in or log in",
    "the first link on the page",
]


async def demo_with_mock(url: str) -> None:
    """Demonstrate using the mock LLM client."""
    logger.info("=" * 60)
    logger.info("Demo: Mock LLM Client (no API required)")
    logger.info("=" * 60)

    # Create mock responses for common queries
    mock_responses = {
        "search": """INDEX: 0
CONFIDENCE: 0.95
REASONING: Found an input element that appears to be for search
ALTERNATIVES: none""",
        "submit": """INDEX: 1
CONFIDENCE: 0.90
REASONING: Found a button element
ALTERNATIVES: 2, 3""",
        "navigation": """INDEX: 5
CONFIDENCE: 0.85
REASONING: Found navigation-related element
ALTERNATIVES: 6, 7""",
        "sign in": """INDEX: 10
CONFIDENCE: 0.92
REASONING: Found sign in link
ALTERNATIVES: none""",
        "log in": """INDEX: 10
CONFIDENCE: 0.92
REASONING: Found login link
ALTERNATIVES: none""",
        "first link": """INDEX: 0
CONFIDENCE: 0.99
REASONING: Selected the first link element
ALTERNATIVES: 1, 2""",
    }

    mock_client = MockLLMClient(responses=mock_responses)
    await run_demo(url, mock_client, "Mock")


async def demo_with_provider(url: str, provider: str) -> None:
    """Demonstrate using a real LLM provider."""
    logger.info("=" * 60)
    logger.info(f"Demo: {provider.title()} LLM Client")
    logger.info("=" * 60)

    try:
        client = create_llm_client(provider)
        await run_demo(url, client, provider.title())
    except ValueError as e:
        logger.error(f"Failed to create client: {e}")
        logger.info("Tip: Set the appropriate API key environment variable")
        if provider == "anthropic":
            logger.info("  export ANTHROPIC_API_KEY=sk-ant-...")
        elif provider == "openai":
            logger.info("  export OPENAI_API_KEY=sk-...")


async def demo_fallback(url: str) -> None:
    """Demonstrate the fallback selector (no LLM required)."""
    logger.info("=" * 60)
    logger.info("Demo: Fallback Selector (text matching, no LLM)")
    logger.info("=" * 60)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        logger.info(f"Loading: {url}")
        await page.goto(url, wait_until="domcontentloaded")
        await page.wait_for_timeout(2000)

        # Extract elements
        extractor = InteractiveElementExtractor()
        elements = await extractor.extract_interactive_elements(page, "demo")

        logger.info(f"Found {len(elements)} interactive elements")

        # Show element list
        formatted = format_for_llm(elements)
        logger.info("\nExtracted elements (first 10):")
        for line in formatted.split("\n")[:10]:
            logger.info(f"  {line}")

        # Use fallback selector
        fallback = FallbackSelector()

        # Test text-based selection
        test_texts = ["More information", "Example", "Click", "Submit", "Search"]

        logger.info("\n--- Text-based Selection ---")
        for text in test_texts:
            result = fallback.find_by_text(text, elements)
            if result.found:
                logger.info(
                    f"  '{text}' -> Index {result.index} (confidence: {result.confidence:.2f})"
                )
            else:
                logger.info(f"  '{text}' -> No match")

        # Test role-based selection
        logger.info("\n--- Role-based Selection ---")
        for role in ["button", "link", "input"]:
            results = fallback.find_by_role(role, elements)
            logger.info(f"  Role '{role}': Found {len(results)} elements")
            for r in results[:3]:
                inner = r.element.element if hasattr(r.element, "element") else r.element
                text = inner.text[:30] if inner.text else "(no text)"
                logger.info(f"    [{r.index}] {text}")

        await browser.close()


async def run_demo(url: str, llm_client, provider_name: str) -> None:
    """Run the demo with the given LLM client."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        logger.info(f"Loading: {url}")
        await page.goto(url, wait_until="domcontentloaded")
        await page.wait_for_timeout(2000)

        # Extract elements
        extractor = InteractiveElementExtractor()
        elements = await extractor.extract_interactive_elements(page, "demo")

        logger.info(f"Found {len(elements)} interactive elements")

        # Show element list
        formatted = format_for_llm(elements)
        logger.info("\nExtracted elements (first 15):")
        for line in formatted.split("\n")[:15]:
            logger.info(f"  {line}")

        # Create selector
        selector = NaturalLanguageSelector(llm_client)

        # Run demo queries
        logger.info(f"\n--- Natural Language Selection ({provider_name}) ---")

        for query in DEMO_QUERIES:
            logger.info(f'\nQuery: "{query}"')

            result = await selector.find_element(query, elements)

            if result.found:
                inner = (
                    result.element.element if hasattr(result.element, "element") else result.element
                )
                text = inner.text[:40] if inner.text else "(no text)"
                logger.info(f"  Found: [{result.index}] <{inner.tag_name}> {text}")
                logger.info(f"  Confidence: {result.confidence:.2f}")
                logger.info(f"  Reasoning: {result.reasoning[:60]}...")
                if result.alternatives:
                    logger.info(f"  Alternatives: {result.alternatives}")
            else:
                logger.info(f"  Not found: {result.reasoning}")

        # Demo action selection
        logger.info("\n--- Action Selection Demo ---")
        instructions = [
            "click the first link",
            "type in the search box",
            "hover over the menu",
        ]

        for instruction in instructions:
            logger.info(f'\nInstruction: "{instruction}"')
            result, action = await selector.select_action(instruction, elements)

            if result.found:
                inner = (
                    result.element.element if hasattr(result.element, "element") else result.element
                )
                logger.info(f"  Element: [{result.index}] <{inner.tag_name}>")
                logger.info(f"  Action: {action}")
                logger.info(f"  Confidence: {result.confidence:.2f}")
            else:
                logger.info(f"  Could not determine action: {result.reasoning}")

        await browser.close()


async def main():
    parser = argparse.ArgumentParser(description="Demo LLM-powered element selection")
    parser.add_argument(
        "--url",
        default="https://github.com",
        help="URL to test (default: https://github.com)",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "litellm"],
        help="LLM provider to use",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLM client (no API key needed)",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Demo fallback selector (text matching)",
    )
    args = parser.parse_args()

    if args.fallback:
        await demo_fallback(args.url)
    elif args.mock:
        await demo_with_mock(args.url)
    elif args.provider:
        await demo_with_provider(args.url, args.provider)
    else:
        # Default: run mock demo
        logger.info("No provider specified. Running mock demo.")
        logger.info("Use --provider anthropic/openai or --fallback for other modes.\n")
        await demo_with_mock(args.url)


if __name__ == "__main__":
    asyncio.run(main())
