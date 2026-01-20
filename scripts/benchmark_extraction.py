#!/usr/bin/env python
"""
Performance benchmarking for DOM extraction capabilities.

Measures extraction times with and without enhanced features:
- Shadow DOM extraction
- Iframe traversal
- DOM stability waiting
- Accessibility tree extraction
- LLM formatting

Usage:
    poetry run python scripts/benchmark_extraction.py
    poetry run python scripts/benchmark_extraction.py --iterations 5
    poetry run python scripts/benchmark_extraction.py --url https://example.com
    poetry run python scripts/benchmark_extraction.py --all --output results.json
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import statistics
import sys
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from playwright.async_api import Page, async_playwright

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qontinui.extraction.web import (
    ExtractionOptions,
    InteractiveElementExtractor,
    format_for_llm,
)
from qontinui.extraction.web.accessibility_extractor import extract_accessibility_tree
from qontinui.extraction.web.frame_manager import extract_across_frames
from qontinui.extraction.web.hybrid_extractor import HybridExtractor

logging.basicConfig(
    level=logging.WARNING,  # Reduce noise during benchmarks
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""

    name: str
    times_ms: list[float] = field(default_factory=list)
    elements_found: int = 0
    extra_data: dict[str, Any] = field(default_factory=dict)

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.times_ms) if self.times_ms else 0

    @property
    def median_ms(self) -> float:
        return statistics.median(self.times_ms) if self.times_ms else 0

    @property
    def stdev_ms(self) -> float:
        return statistics.stdev(self.times_ms) if len(self.times_ms) > 1 else 0

    @property
    def min_ms(self) -> float:
        return min(self.times_ms) if self.times_ms else 0

    @property
    def max_ms(self) -> float:
        return max(self.times_ms) if self.times_ms else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "iterations": len(self.times_ms),
            "mean_ms": round(self.mean_ms, 2),
            "median_ms": round(self.median_ms, 2),
            "stdev_ms": round(self.stdev_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "elements_found": self.elements_found,
            "extra_data": self.extra_data,
        }


@dataclass
class PageBenchmarkResults:
    """Results for all benchmarks on a single page."""

    url: str
    benchmarks: list[BenchmarkResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "benchmarks": [b.to_dict() for b in self.benchmarks],
        }


async def time_async(
    func: Callable[[], Coroutine[Any, Any, Any]],
    iterations: int = 3,
) -> tuple[list[float], Any]:
    """Time an async function over multiple iterations."""
    times: list[float] = []
    result = None

    for _i in range(iterations):
        gc.collect()  # Clean up before each iteration
        start = time.perf_counter()
        result = await func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return times, result


async def benchmark_basic_extraction(page: Page, iterations: int) -> BenchmarkResult:
    """Benchmark basic extraction without enhanced features."""
    options = ExtractionOptions(
        include_shadow_dom=False,
        include_cursor_pointer=False,
    )
    extractor = InteractiveElementExtractor(options)

    async def run():
        extractor.reset_counter()
        return await extractor.extract_interactive_elements(page, "benchmark")

    times, elements = await time_async(run, iterations)

    return BenchmarkResult(
        name="basic_extraction",
        times_ms=times,
        elements_found=len(elements) if elements else 0,
    )


async def benchmark_shadow_dom_extraction(page: Page, iterations: int) -> BenchmarkResult:
    """Benchmark extraction with shadow DOM enabled."""
    options = ExtractionOptions(
        include_shadow_dom=True,
        max_shadow_depth=5,
        include_cursor_pointer=False,
    )
    extractor = InteractiveElementExtractor(options)

    async def run():
        extractor.reset_counter()
        return await extractor.extract_interactive_elements(page, "benchmark")

    times, elements = await time_async(run, iterations)

    # Count shadow elements
    shadow_count = sum(1 for e in (elements or []) if e.shadow_path)

    return BenchmarkResult(
        name="shadow_dom_extraction",
        times_ms=times,
        elements_found=len(elements) if elements else 0,
        extra_data={"shadow_elements": shadow_count},
    )


async def benchmark_cursor_pointer_extraction(page: Page, iterations: int) -> BenchmarkResult:
    """Benchmark extraction with cursor:pointer detection."""
    options = ExtractionOptions(
        include_shadow_dom=False,
        include_cursor_pointer=True,
    )
    extractor = InteractiveElementExtractor(options)

    async def run():
        extractor.reset_counter()
        return await extractor.extract_interactive_elements(page, "benchmark")

    times, elements = await time_async(run, iterations)

    # Count clickable elements found via cursor:pointer
    clickable_count = sum(1 for e in (elements or []) if "clickable_" in e.element_type)

    return BenchmarkResult(
        name="cursor_pointer_extraction",
        times_ms=times,
        elements_found=len(elements) if elements else 0,
        extra_data={"clickable_elements": clickable_count},
    )


async def benchmark_iframe_extraction(page: Page, iterations: int) -> BenchmarkResult:
    """Benchmark multi-frame extraction."""

    async def run():
        return await extract_across_frames(
            page,
            screenshot_id="benchmark",
            include_shadow_dom=False,
        )

    times, result = await time_async(run, iterations)

    frame_count = len(result.frames) if result else 0
    element_count = len(result.elements) if result else 0

    return BenchmarkResult(
        name="iframe_extraction",
        times_ms=times,
        elements_found=element_count,
        extra_data={"frame_count": frame_count},
    )


async def benchmark_accessibility_extraction(page: Page, iterations: int) -> BenchmarkResult:
    """Benchmark accessibility tree extraction."""

    async def run():
        return await extract_accessibility_tree(page)

    times, tree = await time_async(run, iterations)

    node_count = tree.node_count if tree else 0

    return BenchmarkResult(
        name="accessibility_extraction",
        times_ms=times,
        elements_found=node_count,
        extra_data={"a11y_nodes": node_count},
    )


async def benchmark_llm_formatting(page: Page, iterations: int) -> BenchmarkResult:
    """Benchmark LLM formatting (extraction + formatting)."""
    options = ExtractionOptions(include_shadow_dom=False)
    extractor = InteractiveElementExtractor(options)

    # First extract elements
    extractor.reset_counter()
    elements = await extractor.extract_interactive_elements(page, "benchmark")

    # Benchmark just the formatting
    def run_sync():
        return format_for_llm(elements)

    times: list[float] = []
    result = None

    for _ in range(iterations):
        gc.collect()
        start = time.perf_counter()
        result = run_sync()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    char_count = len(result) if result else 0

    return BenchmarkResult(
        name="llm_formatting",
        times_ms=times,
        elements_found=len(elements) if elements else 0,
        extra_data={"formatted_chars": char_count},
    )


async def benchmark_hybrid_extraction(page: Page, iterations: int) -> BenchmarkResult:
    """Benchmark hybrid extraction (DOM + screenshot + a11y)."""
    extractor = HybridExtractor()

    async def run():
        return await extractor.extract(page)

    times, context = await time_async(run, iterations)

    element_count = len(context.elements) if context and context.elements else 0
    has_screenshot = context.screenshot_base64 is not None if context else False

    return BenchmarkResult(
        name="hybrid_extraction",
        times_ms=times,
        elements_found=element_count,
        extra_data={"has_screenshot": has_screenshot},
    )


async def benchmark_full_extraction(page: Page, iterations: int) -> BenchmarkResult:
    """Benchmark full extraction with all features enabled (optimized defaults)."""
    options = ExtractionOptions(
        include_shadow_dom=True,
        max_shadow_depth=5,
        include_cursor_pointer=True,
    )
    extractor = InteractiveElementExtractor(options)

    async def run():
        extractor.reset_counter()
        # Use optimized defaults: 100ms stability, 3000ms max wait
        return await extractor.extract_full(
            page,
            screenshot_id="benchmark",
            include_iframes=True,
            wait_for_stability=True,
            stability_ms=100,
            max_wait_ms=3000,
        )

    times, result = await time_async(run, iterations)

    # Result could be FrameExtractionResult or list
    if hasattr(result, "elements"):
        element_count = len(result.elements)
        frame_count = len(result.frames)
    else:
        element_count = len(result) if result else 0
        frame_count = 1

    return BenchmarkResult(
        name="full_extraction",
        times_ms=times,
        elements_found=element_count,
        extra_data={"frame_count": frame_count},
    )


async def benchmark_stability_waiting(page: Page, iterations: int) -> BenchmarkResult:
    """Benchmark extraction with stability waiting (optimized defaults)."""
    options = ExtractionOptions(include_shadow_dom=False)
    extractor = InteractiveElementExtractor(options)

    async def run():
        extractor.reset_counter()
        # Use optimized defaults: 100ms stability, 3000ms max wait
        return await extractor.extract_with_stability(
            page,
            screenshot_id="benchmark",
            stability_ms=100,
            max_wait_ms=3000,
        )

    times, elements = await time_async(run, iterations)

    return BenchmarkResult(
        name="stability_waiting",
        times_ms=times,
        elements_found=len(elements) if elements else 0,
    )


# All benchmark functions
BENCHMARKS = [
    ("basic_extraction", benchmark_basic_extraction),
    ("shadow_dom_extraction", benchmark_shadow_dom_extraction),
    ("cursor_pointer_extraction", benchmark_cursor_pointer_extraction),
    ("iframe_extraction", benchmark_iframe_extraction),
    ("accessibility_extraction", benchmark_accessibility_extraction),
    ("llm_formatting", benchmark_llm_formatting),
    ("stability_waiting", benchmark_stability_waiting),
    ("hybrid_extraction", benchmark_hybrid_extraction),
    ("full_extraction", benchmark_full_extraction),
]

# Test URLs with different complexity levels
TEST_URLS = {
    "simple": {
        "url": "https://example.com",
        "description": "Simple static page",
    },
    "medium": {
        "url": "https://github.com",
        "description": "Medium complexity (GitHub home)",
    },
    "complex": {
        "url": "https://www.youtube.com",
        "description": "Complex dynamic page (YouTube)",
    },
    "documentation": {
        "url": "https://developer.mozilla.org/en-US/docs/Web/API/Web_components",
        "description": "Large documentation page (MDN)",
    },
}


async def run_benchmarks_on_page(
    page: Page,
    url: str,
    iterations: int = 3,
    benchmarks: list[str] | None = None,
) -> PageBenchmarkResults:
    """Run all benchmarks on a single page."""
    results = PageBenchmarkResults(url=url)

    # Filter benchmarks if specified
    bench_to_run = BENCHMARKS
    if benchmarks:
        bench_to_run = [(n, f) for n, f in BENCHMARKS if n in benchmarks]

    for name, bench_func in bench_to_run:
        try:
            result = await bench_func(page, iterations)
            results.benchmarks.append(result)
        except Exception as e:
            logger.error(f"Benchmark {name} failed: {e}")
            results.benchmarks.append(
                BenchmarkResult(
                    name=name,
                    extra_data={"error": str(e)},
                )
            )

    return results


def print_results(results: PageBenchmarkResults) -> None:
    """Print benchmark results in a formatted table."""
    print(f"\n{'='*80}")
    print(f"Benchmark Results: {results.url}")
    print(f"{'='*80}")

    # Header
    print(f"{'Benchmark':<30} {'Mean':>10} {'Median':>10} {'StdDev':>10} {'Elements':>10}")
    print(f"{'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    # Results
    for b in results.benchmarks:
        print(
            f"{b.name:<30} "
            f"{b.mean_ms:>9.1f}ms "
            f"{b.median_ms:>9.1f}ms "
            f"{b.stdev_ms:>9.1f}ms "
            f"{b.elements_found:>10}"
        )

        # Print extra data if present
        if b.extra_data:
            for key, value in b.extra_data.items():
                if key != "error":
                    print(f"  └─ {key}: {value}")

    print()


def print_comparison(all_results: list[PageBenchmarkResults]) -> None:
    """Print comparison across all pages."""
    print(f"\n{'='*80}")
    print("Performance Comparison Across Pages")
    print(f"{'='*80}")

    # Get all benchmark names
    bench_names = [b.name for b in all_results[0].benchmarks]

    for bench_name in bench_names:
        print(f"\n{bench_name}:")
        print(f"  {'Page':<50} {'Mean':>12} {'Elements':>10}")
        print(f"  {'-'*50} {'-'*12} {'-'*10}")

        for page_result in all_results:
            bench = next((b for b in page_result.benchmarks if b.name == bench_name), None)
            if bench:
                # Truncate URL for display
                url_display = (
                    page_result.url[:47] + "..." if len(page_result.url) > 50 else page_result.url
                )
                print(
                    f"  {url_display:<50} "
                    f"{bench.mean_ms:>10.1f}ms "
                    f"{bench.elements_found:>10}"
                )


def calculate_overhead(all_results: list[PageBenchmarkResults]) -> None:
    """Calculate and print overhead of enhanced features vs basic extraction."""
    print(f"\n{'='*80}")
    print("Feature Overhead Analysis (vs basic extraction)")
    print(f"{'='*80}")

    for page_result in all_results:
        basic = next((b for b in page_result.benchmarks if b.name == "basic_extraction"), None)
        if not basic or basic.mean_ms == 0:
            continue

        print(f"\n{page_result.url}")
        print(f"  Basic extraction: {basic.mean_ms:.1f}ms ({basic.elements_found} elements)")

        for bench in page_result.benchmarks:
            if bench.name == "basic_extraction" or bench.mean_ms == 0:
                continue

            overhead_ms = bench.mean_ms - basic.mean_ms
            overhead_pct = (overhead_ms / basic.mean_ms) * 100 if basic.mean_ms > 0 else 0

            sign = "+" if overhead_ms >= 0 else ""
            print(
                f"  {bench.name:<28} {bench.mean_ms:>8.1f}ms "
                f"({sign}{overhead_ms:.1f}ms, {sign}{overhead_pct:.0f}%)"
            )


async def main():
    parser = argparse.ArgumentParser(description="Benchmark DOM extraction performance")
    parser.add_argument(
        "--url",
        help="Single URL to benchmark",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Benchmark all predefined test URLs",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations per benchmark (default: 3)",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        help="Specific benchmarks to run (space-separated)",
    )
    parser.add_argument(
        "--output",
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run a warmup iteration before benchmarking",
    )
    args = parser.parse_args()

    # Determine URLs to test
    if args.url:
        urls = [args.url]
    elif args.all:
        urls = [info["url"] for info in TEST_URLS.values()]
    else:
        # Default: test simple and medium
        urls = [TEST_URLS["simple"]["url"], TEST_URLS["medium"]["url"]]

    all_results: list[PageBenchmarkResults] = []

    print(f"Running benchmarks with {args.iterations} iterations per test")
    print(f"Testing {len(urls)} URL(s)")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 720},
        )
        page = await context.new_page()

        for url in urls:
            print(f"\nLoading: {url}")
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                await page.wait_for_timeout(2000)  # Let JS settle

                # Warmup run if requested
                if args.warmup:
                    print("  Running warmup...")
                    await run_benchmarks_on_page(page, url, iterations=1)

                # Actual benchmark
                print("  Running benchmarks...")
                results = await run_benchmarks_on_page(page, url, args.iterations, args.benchmarks)
                all_results.append(results)

                # Print results for this page
                print_results(results)

            except Exception as e:
                print(f"  ERROR: {e}")
                all_results.append(
                    PageBenchmarkResults(
                        url=url,
                        benchmarks=[BenchmarkResult(name="error", extra_data={"error": str(e)})],
                    )
                )

        await browser.close()

    # Print comparison if multiple pages
    if len(all_results) > 1:
        print_comparison(all_results)
        calculate_overhead(all_results)

    # Save results if requested
    if args.output:
        output_data = {
            "iterations": args.iterations,
            "results": [r.to_dict() for r in all_results],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
