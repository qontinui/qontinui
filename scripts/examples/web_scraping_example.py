#!/usr/bin/env python
"""
Web scraping example using qontinui's web extraction capabilities.

Demonstrates extracting interactive elements (links, buttons) from a webpage
and formatting the output as JSON.

Usage:
    poetry run python scripts/examples/web_scraping_example.py
    poetry run python scripts/examples/web_scraping_example.py --url https://example.com
    poetry run python scripts/examples/web_scraping_example.py --no-headless

Features demonstrated:
- Basic interactive element extraction (buttons, links, inputs)
- LLM-friendly element formatting
- JSON output formatting
- Shadow DOM extraction
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from playwright.async_api import async_playwright

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from qontinui.extraction.web import (
    ExtractionOptions,
    InteractiveElement,
    InteractiveElementExtractor,
    format_for_llm,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def filter_by_type(
    elements: list[InteractiveElement],
    element_types: set[str],
) -> list[InteractiveElement]:
    """Filter elements by their type."""
    return [e for e in elements if e.element_type in element_types]


def format_links_as_json(elements: list[InteractiveElement]) -> list[dict]:
    """Format link elements as a list of dictionaries."""
    links = []
    for elem in elements:
        if elem.element_type == "a" or "link" in elem.element_type:
            links.append(
                {
                    "text": elem.text or elem.aria_label or "(no text)",
                    "href": elem.href or "",
                    "selector": elem.selector,
                    "position": {
                        "x": elem.bbox.x,
                        "y": elem.bbox.y,
                        "width": elem.bbox.width,
                        "height": elem.bbox.height,
                    },
                }
            )
    return links


def format_buttons_as_json(elements: list[InteractiveElement]) -> list[dict]:
    """Format button elements as a list of dictionaries."""
    buttons = []
    for elem in elements:
        if elem.element_type in ("button", "aria_button", "input"):
            # Check for submit/button input types
            if elem.element_type == "input":
                # Only include button-like inputs
                if not any(x in elem.selector.lower() for x in ["submit", "button"]):
                    continue

            buttons.append(
                {
                    "label": elem.text or elem.aria_label or "(no label)",
                    "type": elem.element_type,
                    "selector": elem.selector,
                    "aria_role": elem.aria_role,
                    "position": {
                        "x": elem.bbox.x,
                        "y": elem.bbox.y,
                    },
                }
            )
    return buttons


async def basic_extraction(url: str, headless: bool = True) -> dict:
    """
    Perform basic element extraction from a webpage.

    This is the simplest form of extraction - just getting all
    interactive elements from the page.
    """
    logger.info("=" * 60)
    logger.info("BASIC EXTRACTION")
    logger.info("=" * 60)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page()

        logger.info(f"Navigating to: {url}")
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(2000)  # Wait for dynamic content
        except Exception as e:
            logger.error(f"Failed to load page: {e}")
            await browser.close()
            return {"error": str(e)}

        # Create extractor with default options
        extractor = InteractiveElementExtractor()

        # Extract all interactive elements
        elements = await extractor.extract_interactive_elements(page, "basic_demo")

        logger.info(f"Found {len(elements)} interactive elements")

        # Categorize elements
        element_counts = {}
        for elem in elements:
            elem_type = elem.element_type
            element_counts[elem_type] = element_counts.get(elem_type, 0) + 1

        logger.info(f"Element types: {json.dumps(element_counts, indent=2)}")

        # Extract links
        links = format_links_as_json(elements)
        logger.info(f"Found {len(links)} links")

        # Extract buttons
        buttons = format_buttons_as_json(elements)
        logger.info(f"Found {len(buttons)} buttons")

        await browser.close()

        return {
            "url": url,
            "total_elements": len(elements),
            "element_counts": element_counts,
            "links": links,
            "buttons": buttons,
        }


async def enhanced_extraction(url: str, headless: bool = True) -> dict:
    """
    Perform enhanced extraction with custom options.

    This demonstrates:
    - Custom extraction options
    - Shadow DOM extraction
    - Cursor pointer detection (for framework-based UIs)
    """
    logger.info("=" * 60)
    logger.info("ENHANCED EXTRACTION")
    logger.info("=" * 60)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page()

        logger.info(f"Navigating to: {url}")
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(2000)
        except Exception as e:
            logger.error(f"Failed to load page: {e}")
            await browser.close()
            return {"error": str(e)}

        # Create extractor with custom options
        options = ExtractionOptions(
            min_width=20,  # Smaller minimum size
            min_height=20,
            include_shadow_dom=True,  # Extract from shadow DOM
            max_shadow_depth=5,  # Traverse up to 5 levels deep
            include_cursor_pointer=True,  # Include clickable elements (React, Vue, etc.)
            max_cursor_pointer_text_length=50,  # Longer text for clickable
        )

        extractor = InteractiveElementExtractor(options=options)

        # Extract elements
        elements = await extractor.extract_interactive_elements(page, "enhanced_demo")

        logger.info(f"Found {len(elements)} interactive elements (enhanced)")

        # Check for shadow DOM elements
        shadow_elements = [e for e in elements if e.shadow_path]
        if shadow_elements:
            logger.info(f"Found {len(shadow_elements)} elements in shadow DOM")
            for elem in shadow_elements[:5]:  # Show first 5
                logger.info(f"  Shadow path: {elem.shadow_path}")
                logger.info(f"    Text: {elem.text or '(no text)'}")

        # Get LLM-friendly format
        formatted = format_for_llm(elements)
        llm_lines = formatted.split("\n")

        logger.info("\nLLM-friendly format (first 15 elements):")
        for line in llm_lines[:15]:
            logger.info(f"  {line}")

        await browser.close()

        return {
            "url": url,
            "total_elements": len(elements),
            "shadow_dom_elements": len(shadow_elements),
            "llm_format_sample": "\n".join(llm_lines[:20]),
            "elements": [e.to_dict() for e in elements[:20]],  # First 20 as JSON
        }


async def extract_forms(url: str, headless: bool = True) -> dict:
    """
    Extract form-related elements from a webpage.

    Useful for understanding form structure before automation.
    """
    logger.info("=" * 60)
    logger.info("FORM EXTRACTION")
    logger.info("=" * 60)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page()

        logger.info(f"Navigating to: {url}")
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(2000)
        except Exception as e:
            logger.error(f"Failed to load page: {e}")
            await browser.close()
            return {"error": str(e)}

        extractor = InteractiveElementExtractor()
        elements = await extractor.extract_interactive_elements(page, "form_demo")

        # Filter for form-related elements
        form_types = {"input", "select", "textarea", "button", "label"}
        form_elements = [
            e for e in elements if e.element_type in form_types or e.tag_name in form_types
        ]

        logger.info(f"Found {len(form_elements)} form-related elements")

        # Organize by type
        inputs = []
        buttons = []
        selects = []

        for elem in form_elements:
            if elem.tag_name == "input":
                inputs.append(
                    {
                        "type": elem.element_type,
                        "selector": elem.selector,
                        "aria_label": elem.aria_label,
                        "placeholder": elem.text,
                    }
                )
            elif elem.tag_name == "button" or elem.element_type == "button":
                buttons.append(
                    {
                        "text": elem.text or elem.aria_label,
                        "selector": elem.selector,
                    }
                )
            elif elem.tag_name == "select":
                selects.append(
                    {
                        "aria_label": elem.aria_label,
                        "selector": elem.selector,
                    }
                )

        logger.info(f"  Inputs: {len(inputs)}")
        logger.info(f"  Buttons: {len(buttons)}")
        logger.info(f"  Select dropdowns: {len(selects)}")

        await browser.close()

        return {
            "url": url,
            "total_form_elements": len(form_elements),
            "inputs": inputs,
            "buttons": buttons,
            "selects": selects,
        }


async def main():
    parser = argparse.ArgumentParser(description="Web scraping example using qontinui extraction")
    parser.add_argument(
        "--url",
        default="https://github.com",
        help="URL to extract from (default: https://github.com)",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Run browser in visible mode",
    )
    parser.add_argument(
        "--output",
        help="Output file for JSON results",
    )
    parser.add_argument(
        "--mode",
        choices=["basic", "enhanced", "forms", "all"],
        default="all",
        help="Extraction mode (default: all)",
    )
    args = parser.parse_args()

    headless = not args.no_headless
    results = {}

    if args.mode in ("basic", "all"):
        results["basic"] = await basic_extraction(args.url, headless)

    if args.mode in ("enhanced", "all"):
        results["enhanced"] = await enhanced_extraction(args.url, headless)

    if args.mode in ("forms", "all"):
        results["forms"] = await extract_forms(args.url, headless)

    # Output results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to: {output_path}")
    else:
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)

        if "basic" in results and "error" not in results["basic"]:
            logger.info(f"Basic extraction: {results['basic']['total_elements']} elements")
            logger.info(f"  - Links: {len(results['basic']['links'])}")
            logger.info(f"  - Buttons: {len(results['basic']['buttons'])}")

        if "enhanced" in results and "error" not in results["enhanced"]:
            logger.info(f"Enhanced extraction: {results['enhanced']['total_elements']} elements")
            logger.info(f"  - Shadow DOM: {results['enhanced']['shadow_dom_elements']}")

        if "forms" in results and "error" not in results["forms"]:
            logger.info(f"Form extraction: {results['forms']['total_form_elements']} elements")
            logger.info(f"  - Inputs: {len(results['forms']['inputs'])}")

    return results


if __name__ == "__main__":
    asyncio.run(main())
