#!/usr/bin/env python
"""
Real-world validation script for enhanced DOM extraction.

Tests shadow DOM, iframe traversal, and full extraction pipeline
against actual websites.

Usage:
    poetry run python scripts/validate_dom_extraction.py
    poetry run python scripts/validate_dom_extraction.py --headless
    poetry run python scripts/validate_dom_extraction.py --url https://example.com
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from playwright.async_api import async_playwright

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qontinui.extraction.web import (
    ExtractionOptions,
    InteractiveElementExtractor,
    format_for_llm,
)
from qontinui.extraction.web.accessibility_extractor import (
    AccessibilityExtractor,
    extract_accessibility_tree,
)
from qontinui.extraction.web.frame_manager import FrameManager, extract_across_frames
from qontinui.extraction.web.hybrid_extractor import HybridExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation test."""

    test_name: str
    url: str
    success: bool
    elements_found: int
    shadow_elements: int
    iframe_elements: int
    details: dict
    error: str | None = None


# Test URLs with known shadow DOM / iframe usage
TEST_URLS = {
    "github_explore": {
        "url": "https://github.com/explore",
        "description": "GitHub Explore - uses shadow DOM for some components",
        "expect_shadow": False,  # GitHub doesn't use much shadow DOM anymore
        "expect_iframes": False,
    },
    "youtube_embed": {
        "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "description": "YouTube - complex page with potential shadow DOM",
        "expect_shadow": True,
        "expect_iframes": True,
    },
    "mdn_webcomponents": {
        "url": "https://developer.mozilla.org/en-US/docs/Web/API/Web_components",
        "description": "MDN Web Components docs - may have examples",
        "expect_shadow": False,
        "expect_iframes": False,
    },
    "codepen_embed": {
        "url": "https://codepen.io/pen/",
        "description": "CodePen - uses iframes for code preview",
        "expect_shadow": False,
        "expect_iframes": True,
    },
}


async def test_basic_extraction(page, url: str) -> ValidationResult:
    """Test basic interactive element extraction."""
    test_name = "basic_extraction"
    logger.info(f"Testing basic extraction on {url}")

    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(2000)  # Wait for JS to settle

        extractor = InteractiveElementExtractor()
        elements = await extractor.extract_interactive_elements(page, "test_screenshot")

        # Categorize elements
        element_types = {}
        for el in elements:
            element_types[el.element_type] = element_types.get(el.element_type, 0) + 1

        return ValidationResult(
            test_name=test_name,
            url=url,
            success=len(elements) > 0,
            elements_found=len(elements),
            shadow_elements=0,
            iframe_elements=0,
            details={
                "element_types": element_types,
                "sample_elements": [
                    {"id": el.id, "type": el.element_type, "text": el.text[:50] if el.text else None}
                    for el in elements[:10]
                ],
            },
        )

    except Exception as e:
        logger.error(f"Basic extraction failed: {e}")
        return ValidationResult(
            test_name=test_name,
            url=url,
            success=False,
            elements_found=0,
            shadow_elements=0,
            iframe_elements=0,
            details={},
            error=str(e),
        )


async def test_shadow_dom_extraction(page, url: str) -> ValidationResult:
    """Test shadow DOM extraction."""
    test_name = "shadow_dom_extraction"
    logger.info(f"Testing shadow DOM extraction on {url}")

    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(2000)

        # Extract with shadow DOM enabled
        options = ExtractionOptions(
            include_shadow_dom=True,
            max_shadow_depth=5,
        )
        extractor = InteractiveElementExtractor(options)
        elements = await extractor.extract_interactive_elements(page, "test_screenshot")

        # Count elements from shadow DOM
        shadow_elements = [el for el in elements if el.shadow_path]

        # Also check for shadow roots on the page
        shadow_root_count = await page.evaluate("""
            () => {
                let count = 0;
                const checkShadow = (root) => {
                    for (const el of root.querySelectorAll('*')) {
                        if (el.shadowRoot) {
                            count++;
                            checkShadow(el.shadowRoot);
                        }
                    }
                };
                checkShadow(document);
                return count;
            }
        """)

        return ValidationResult(
            test_name=test_name,
            url=url,
            success=True,
            elements_found=len(elements),
            shadow_elements=len(shadow_elements),
            iframe_elements=0,
            details={
                "shadow_roots_on_page": shadow_root_count,
                "elements_from_shadow": len(shadow_elements),
                "shadow_element_samples": [
                    {"id": el.id, "shadow_path": el.shadow_path, "text": el.text[:30] if el.text else None}
                    for el in shadow_elements[:5]
                ],
            },
        )

    except Exception as e:
        logger.error(f"Shadow DOM extraction failed: {e}")
        return ValidationResult(
            test_name=test_name,
            url=url,
            success=False,
            elements_found=0,
            shadow_elements=0,
            iframe_elements=0,
            details={},
            error=str(e),
        )


async def test_iframe_extraction(page, url: str) -> ValidationResult:
    """Test iframe traversal extraction."""
    test_name = "iframe_extraction"
    logger.info(f"Testing iframe extraction on {url}")

    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(3000)  # More time for iframes to load

        # Get frame count
        frames = page.frames
        frame_count = len(frames)

        # Extract across all frames
        result = await extract_across_frames(
            page,
            screenshot_id="test_screenshot",
            include_shadow_dom=True,
        )

        # Count elements per frame
        elements_per_frame = {}
        for el in result.elements:
            frame_id = el.frame_id
            elements_per_frame[frame_id] = elements_per_frame.get(frame_id, 0) + 1

        iframe_elements = sum(
            count for frame_id, count in elements_per_frame.items() if frame_id > 0
        )

        return ValidationResult(
            test_name=test_name,
            url=url,
            success=True,
            elements_found=len(result.elements),
            shadow_elements=0,
            iframe_elements=iframe_elements,
            details={
                "total_frames": frame_count,
                "frames_with_elements": len(result.frames),
                "elements_per_frame": elements_per_frame,
                "frame_info": [
                    {"id": f.frame_id, "url": f.url[:50] if f.url else "about:blank", "name": f.name}
                    for f in result.frames[:5]
                ],
            },
        )

    except Exception as e:
        logger.error(f"Iframe extraction failed: {e}")
        return ValidationResult(
            test_name=test_name,
            url=url,
            success=False,
            elements_found=0,
            shadow_elements=0,
            iframe_elements=0,
            details={},
            error=str(e),
        )


async def test_accessibility_extraction(page, url: str) -> ValidationResult:
    """Test accessibility tree extraction."""
    test_name = "accessibility_extraction"
    logger.info(f"Testing accessibility extraction on {url}")

    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(2000)

        # Extract accessibility tree
        a11y_tree = await extract_accessibility_tree(page)

        # Get stats
        node_count = 0
        roles = {}

        def count_nodes(node):
            nonlocal node_count
            node_count += 1
            if node.role:
                roles[node.role] = roles.get(node.role, 0) + 1
            for child in node.children:
                count_nodes(child)

        if a11y_tree.root:
            count_nodes(a11y_tree.root)

        return ValidationResult(
            test_name=test_name,
            url=url,
            success=node_count > 0,
            elements_found=node_count,
            shadow_elements=0,
            iframe_elements=0,
            details={
                "total_nodes": node_count,
                "roles": dict(sorted(roles.items(), key=lambda x: -x[1])[:10]),
                "tree_preview": a11y_tree.to_text()[:500] if a11y_tree.root else "Empty tree",
            },
        )

    except Exception as e:
        logger.error(f"Accessibility extraction failed: {e}")
        return ValidationResult(
            test_name=test_name,
            url=url,
            success=False,
            elements_found=0,
            shadow_elements=0,
            iframe_elements=0,
            details={},
            error=str(e),
        )


async def test_llm_formatting(page, url: str) -> ValidationResult:
    """Test LLM-friendly element formatting."""
    test_name = "llm_formatting"
    logger.info(f"Testing LLM formatting on {url}")

    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(2000)

        extractor = InteractiveElementExtractor()
        elements = await extractor.extract_interactive_elements(page, "test_screenshot")

        # Format for LLM
        formatted = format_for_llm(elements)
        lines = formatted.split("\n")

        return ValidationResult(
            test_name=test_name,
            url=url,
            success=len(lines) > 0,
            elements_found=len(elements),
            shadow_elements=0,
            iframe_elements=0,
            details={
                "formatted_count": len(lines),
                "sample_output": "\n".join(lines[:10]),
                "total_chars": len(formatted),
            },
        )

    except Exception as e:
        logger.error(f"LLM formatting failed: {e}")
        return ValidationResult(
            test_name=test_name,
            url=url,
            success=False,
            elements_found=0,
            shadow_elements=0,
            iframe_elements=0,
            details={},
            error=str(e),
        )


async def test_hybrid_extraction(page, url: str) -> ValidationResult:
    """Test hybrid DOM + screenshot extraction."""
    test_name = "hybrid_extraction"
    logger.info(f"Testing hybrid extraction on {url}")

    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(2000)

        extractor = HybridExtractor()
        context = await extractor.extract(page)

        element_count = len(context.elements) if context.elements else 0

        return ValidationResult(
            test_name=test_name,
            url=url,
            success=element_count > 0,
            elements_found=element_count,
            shadow_elements=0,
            iframe_elements=0,
            details={
                "page_title": context.title,
                "page_url": context.url,
                "viewport": context.viewport,
                "scroll_position": (context.scroll_x, context.scroll_y),
                "has_screenshot": context.screenshot_base64 is not None,
                "element_count": element_count,
            },
        )

    except Exception as e:
        logger.error(f"Hybrid extraction failed: {e}")
        return ValidationResult(
            test_name=test_name,
            url=url,
            success=False,
            elements_found=0,
            shadow_elements=0,
            iframe_elements=0,
            details={},
            error=str(e),
        )


async def test_full_extraction(page, url: str) -> ValidationResult:
    """Test full extraction with all capabilities."""
    test_name = "full_extraction"
    logger.info(f"Testing full extraction on {url}")

    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(2000)

        options = ExtractionOptions(
            include_shadow_dom=True,
            max_shadow_depth=5,
            include_cursor_pointer=True,
        )
        extractor = InteractiveElementExtractor(options)

        # Use full extraction with stability waiting
        result = await extractor.extract_full(
            page,
            screenshot_id="test_screenshot",
            include_iframes=True,
            wait_for_stability=True,
            stability_ms=500,
            max_wait_ms=3000,
        )

        # Result could be FrameExtractionResult or list
        if hasattr(result, "elements"):
            elements = result.elements
            frame_count = len(result.frames)
        else:
            elements = result
            frame_count = 1

        shadow_elements = [
            el for el in elements
            if hasattr(el, "shadow_path") and el.shadow_path
        ]

        return ValidationResult(
            test_name=test_name,
            url=url,
            success=len(elements) > 0,
            elements_found=len(elements),
            shadow_elements=len(shadow_elements),
            iframe_elements=0,  # Would need frame_id check
            details={
                "total_elements": len(elements),
                "from_shadow_dom": len(shadow_elements),
                "frame_count": frame_count,
                "stability_waited": True,
            },
        )

    except Exception as e:
        logger.error(f"Full extraction failed: {e}")
        return ValidationResult(
            test_name=test_name,
            url=url,
            success=False,
            elements_found=0,
            shadow_elements=0,
            iframe_elements=0,
            details={},
            error=str(e),
        )


async def run_validation(urls: list[str], headless: bool = True) -> list[ValidationResult]:
    """Run all validation tests on given URLs."""
    all_results = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        )
        page = await context.new_page()

        for url in urls:
            logger.info(f"\n{'='*60}")
            logger.info(f"Validating: {url}")
            logger.info(f"{'='*60}")

            # Run each test
            tests = [
                test_basic_extraction,
                test_shadow_dom_extraction,
                test_iframe_extraction,
                test_accessibility_extraction,
                test_llm_formatting,
                test_hybrid_extraction,
                test_full_extraction,
            ]

            for test_fn in tests:
                result = await test_fn(page, url)
                all_results.append(result)

                status = "✓" if result.success else "✗"
                logger.info(
                    f"  {status} {result.test_name}: "
                    f"{result.elements_found} elements"
                    + (f" (shadow: {result.shadow_elements})" if result.shadow_elements else "")
                    + (f" (iframes: {result.iframe_elements})" if result.iframe_elements else "")
                    + (f" ERROR: {result.error}" if result.error else "")
                )

        await browser.close()

    return all_results


def print_summary(results: list[ValidationResult]) -> None:
    """Print validation summary."""
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    # Group by URL
    by_url = {}
    for r in results:
        if r.url not in by_url:
            by_url[r.url] = []
        by_url[r.url].append(r)

    for url, url_results in by_url.items():
        print(f"\n{url}")
        print("-" * 40)

        passed = sum(1 for r in url_results if r.success)
        failed = sum(1 for r in url_results if not r.success)

        for r in url_results:
            status = "PASS" if r.success else "FAIL"
            print(f"  [{status}] {r.test_name}: {r.elements_found} elements")
            if r.error:
                print(f"         Error: {r.error[:50]}...")

        print(f"  Summary: {passed}/{len(url_results)} tests passed")

    # Overall stats
    total_passed = sum(1 for r in results if r.success)
    total_failed = sum(1 for r in results if not r.success)
    print(f"\n{'='*60}")
    print(f"OVERALL: {total_passed}/{len(results)} tests passed")
    print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="Validate DOM extraction capabilities")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--url", type=str, help="Test specific URL instead of defaults")
    parser.add_argument("--all", action="store_true", help="Test all predefined URLs")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    # Determine which URLs to test
    if args.url:
        urls = [args.url]
    elif args.all:
        urls = [info["url"] for info in TEST_URLS.values()]
    else:
        # Default: test a simple reliable page
        urls = ["https://example.com", "https://github.com"]

    # Run validation
    results = await run_validation(urls, headless=args.headless)

    # Print summary
    print_summary(results)

    # Save to file if requested
    if args.output:
        output_data = [
            {
                "test_name": r.test_name,
                "url": r.url,
                "success": r.success,
                "elements_found": r.elements_found,
                "shadow_elements": r.shadow_elements,
                "iframe_elements": r.iframe_elements,
                "details": r.details,
                "error": r.error,
            }
            for r in results
        ]
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    # Return exit code based on results
    failed = sum(1 for r in results if not r.success)
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
