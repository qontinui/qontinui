#!/usr/bin/env python
"""
Comprehensive test script for NaturalLanguageSelector with real LLM providers.

Tests the natural language element selection capability against real websites
using actual LLM providers (Anthropic Claude, OpenAI GPT).

Usage:
    # With Anthropic (if ANTHROPIC_API_KEY is set)
    ANTHROPIC_API_KEY=xxx poetry run python scripts/test_real_llm_selector.py

    # With OpenAI (if OPENAI_API_KEY is set)
    OPENAI_API_KEY=xxx poetry run python scripts/test_real_llm_selector.py

    # With specific provider
    poetry run python scripts/test_real_llm_selector.py --provider anthropic
    poetry run python scripts/test_real_llm_selector.py --provider openai

    # Test specific URL
    poetry run python scripts/test_real_llm_selector.py --url https://github.com

    # Verbose output
    poetry run python scripts/test_real_llm_selector.py --verbose

    # Generate JSON report
    poetry run python scripts/test_real_llm_selector.py --output results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from playwright.async_api import async_playwright

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qontinui.extraction.web import (
    InteractiveElementExtractor,
    NaturalLanguageSelector,
    format_for_llm,
)
from qontinui.extraction.web.llm_clients import (
    AnthropicClient,
    BaseLLMClient,
    MockLLMClient,
    OpenAIClient,
    create_llm_client,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Test Configurations
# =============================================================================

@dataclass
class TestQuery:
    """A test query with expected behavior."""
    description: str
    category: str  # button, link, input, action, ambiguous
    expected_tags: list[str] = field(default_factory=list)  # Expected tag names
    expected_keywords: list[str] = field(default_factory=list)  # Text/aria keywords
    min_confidence: float = 0.5
    notes: str = ""


# Test websites with known structure
TEST_SITES = {
    "github": {
        "url": "https://github.com",
        "name": "GitHub",
        "queries": [
            TestQuery(
                description="the search input field",
                category="input",
                expected_tags=["input", "button"],
                expected_keywords=["search", "type", "find"],
                notes="GitHub has both search button and input",
            ),
            TestQuery(
                description="sign in button or link",
                category="button",
                expected_tags=["a", "button"],
                expected_keywords=["sign in", "signin", "login"],
            ),
            TestQuery(
                description="sign up button",
                category="button",
                expected_tags=["a", "button"],
                expected_keywords=["sign up", "signup", "register"],
            ),
            TestQuery(
                description="the main navigation menu",
                category="link",
                expected_tags=["nav", "ul", "a"],
                expected_keywords=["product", "solutions", "open source", "pricing"],
            ),
            TestQuery(
                description="explore repositories link",
                category="link",
                expected_tags=["a"],
                expected_keywords=["explore", "repositories"],
            ),
        ],
    },
    "example": {
        "url": "https://example.com",
        "name": "Example.com",
        "queries": [
            TestQuery(
                description="the More information link",
                category="link",
                expected_tags=["a"],
                expected_keywords=["more information"],
            ),
            TestQuery(
                description="any clickable link on the page",
                category="link",
                expected_tags=["a"],
                min_confidence=0.7,
            ),
        ],
    },
    "duckduckgo": {
        "url": "https://duckduckgo.com",
        "name": "DuckDuckGo",
        "queries": [
            TestQuery(
                description="the search input box",
                category="input",
                expected_tags=["input", "textarea"],
                expected_keywords=["search", "query"],
            ),
            TestQuery(
                description="the search submit button",
                category="button",
                expected_tags=["button", "input"],
                expected_keywords=["search", "submit"],
            ),
        ],
    },
    "wikipedia": {
        "url": "https://www.wikipedia.org",
        "name": "Wikipedia",
        "queries": [
            TestQuery(
                description="the search input field",
                category="input",
                expected_tags=["input"],
                expected_keywords=["search"],
            ),
            TestQuery(
                description="English language link",
                category="link",
                expected_tags=["a"],
                expected_keywords=["english"],
            ),
            TestQuery(
                description="any language selection link",
                category="ambiguous",
                expected_tags=["a"],
                min_confidence=0.5,
            ),
        ],
    },
}

# Action selection test queries
ACTION_QUERIES = [
    {
        "instruction": "click the sign in button",
        "expected_action": "click",
        "category": "click",
    },
    {
        "instruction": "type in the search field",
        "expected_action": "type",
        "category": "type",
    },
    {
        "instruction": "hover over the menu",
        "expected_action": "hover",
        "category": "hover",
    },
    {
        "instruction": "focus on the email input",
        "expected_action": "focus",
        "category": "focus",
    },
    {
        "instruction": "select an option from the dropdown",
        "expected_action": "select",
        "category": "select",
    },
]


# =============================================================================
# Test Result Tracking
# =============================================================================

@dataclass
class QueryResult:
    """Result of a single query test."""
    query: str
    category: str
    success: bool
    found_element: bool
    element_tag: str | None
    element_text: str | None
    element_index: int | None
    confidence: float
    reasoning: str
    response_time_ms: float
    error: str | None = None
    meets_expectations: bool = False


@dataclass
class ActionResult:
    """Result of an action selection test."""
    instruction: str
    expected_action: str
    actual_action: str
    action_correct: bool
    element_found: bool
    confidence: float
    response_time_ms: float
    error: str | None = None


@dataclass
class SiteTestResult:
    """Results for testing a single site."""
    url: str
    site_name: str
    element_count: int
    query_results: list[QueryResult] = field(default_factory=list)
    action_results: list[ActionResult] = field(default_factory=list)
    extraction_time_ms: float = 0.0
    total_time_ms: float = 0.0


@dataclass
class TestSummary:
    """Overall test summary."""
    provider: str
    model: str
    total_queries: int
    successful_queries: int
    failed_queries: int

    # By category
    category_results: dict[str, dict[str, int]] = field(default_factory=dict)

    # Timing
    avg_response_time_ms: float = 0.0
    min_response_time_ms: float = 0.0
    max_response_time_ms: float = 0.0

    # Confidence distribution
    confidence_buckets: dict[str, int] = field(default_factory=dict)
    avg_confidence: float = 0.0

    # Site results
    site_results: list[SiteTestResult] = field(default_factory=list)

    # Timestamp
    timestamp: str = ""


# =============================================================================
# Test Runner
# =============================================================================

class LLMSelectorTester:
    """Tests NaturalLanguageSelector with real LLM providers."""

    def __init__(
        self,
        client: BaseLLMClient,
        provider_name: str,
        model_name: str,
        verbose: bool = False,
    ):
        self.client = client
        self.provider_name = provider_name
        self.model_name = model_name
        self.verbose = verbose
        self.selector = NaturalLanguageSelector(client, confidence_threshold=0.3)

    async def test_site(self, site_key: str, site_config: dict[str, Any]) -> SiteTestResult:
        """Test all queries for a single site."""
        url = site_config["url"]
        site_name = site_config["name"]
        queries = site_config.get("queries", [])

        logger.info(f"\n{'='*60}")
        logger.info(f"Testing site: {site_name} ({url})")
        logger.info(f"{'='*60}")

        result = SiteTestResult(
            url=url,
            site_name=site_name,
            element_count=0,
        )

        start_time = time.time()

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            try:
                logger.info(f"Loading: {url}")
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                await page.wait_for_timeout(2000)  # Wait for JS

                # Extract elements
                extraction_start = time.time()
                extractor = InteractiveElementExtractor()
                elements = await extractor.extract_interactive_elements(page, "test")
                result.extraction_time_ms = (time.time() - extraction_start) * 1000
                result.element_count = len(elements)

                logger.info(f"Extracted {len(elements)} interactive elements in {result.extraction_time_ms:.0f}ms")

                if self.verbose:
                    formatted = format_for_llm(elements)
                    logger.info(f"\nExtracted elements (first 20):")
                    for line in formatted.split("\n")[:20]:
                        logger.info(f"  {line}")

                # Test each query
                for query in queries:
                    query_result = await self._test_query(query, elements)
                    result.query_results.append(query_result)

                # Test action selection on this site
                for action_query in ACTION_QUERIES[:3]:  # Test first 3 action queries
                    action_result = await self._test_action(
                        action_query["instruction"],
                        action_query["expected_action"],
                        elements,
                    )
                    result.action_results.append(action_result)

            except Exception as e:
                logger.error(f"Error testing {site_name}: {e}")
                result.query_results.append(QueryResult(
                    query="<site load>",
                    category="error",
                    success=False,
                    found_element=False,
                    element_tag=None,
                    element_text=None,
                    element_index=None,
                    confidence=0.0,
                    reasoning="",
                    response_time_ms=0.0,
                    error=str(e),
                ))

            finally:
                await browser.close()

        result.total_time_ms = (time.time() - start_time) * 1000
        return result

    async def _test_query(
        self,
        query: TestQuery,
        elements: list,
    ) -> QueryResult:
        """Test a single query."""
        logger.info(f"\nQuery: \"{query.description}\" (category: {query.category})")

        start_time = time.time()

        try:
            result = await self.selector.find_element(query.description, elements)
            response_time_ms = (time.time() - start_time) * 1000

            # Analyze result
            found = result.found
            element_tag = None
            element_text = None
            meets_expectations = False

            if found:
                inner = result.element.element if hasattr(result.element, 'element') else result.element
                element_tag = inner.tag_name.lower()
                element_text = (inner.text or inner.aria_label or "")[:50]

                # Check if meets expectations
                tag_ok = not query.expected_tags or element_tag in [t.lower() for t in query.expected_tags]
                text_ok = not query.expected_keywords or any(
                    kw.lower() in (inner.text or "").lower() or
                    kw.lower() in (inner.aria_label or "").lower()
                    for kw in query.expected_keywords
                )
                conf_ok = result.confidence >= query.min_confidence
                meets_expectations = tag_ok and text_ok and conf_ok

            success = found and result.confidence >= query.min_confidence

            # Log result
            if found:
                status = "PASS" if meets_expectations else "PARTIAL"
                logger.info(f"  [{status}] Found: [{result.index}] <{element_tag}> \"{element_text}\"")
                logger.info(f"  Confidence: {result.confidence:.2f} (min: {query.min_confidence})")
                logger.info(f"  Reasoning: {result.reasoning[:80]}...")
                logger.info(f"  Response time: {response_time_ms:.0f}ms")

                if not meets_expectations:
                    if query.expected_tags and element_tag not in [t.lower() for t in query.expected_tags]:
                        logger.warning(f"    Expected tags: {query.expected_tags}, got: {element_tag}")
            else:
                logger.info(f"  [FAIL] Not found: {result.reasoning}")
                logger.info(f"  Response time: {response_time_ms:.0f}ms")

            return QueryResult(
                query=query.description,
                category=query.category,
                success=success,
                found_element=found,
                element_tag=element_tag,
                element_text=element_text,
                element_index=result.index,
                confidence=result.confidence,
                reasoning=result.reasoning,
                response_time_ms=response_time_ms,
                meets_expectations=meets_expectations,
            )

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            logger.error(f"  [ERROR] {e}")
            return QueryResult(
                query=query.description,
                category=query.category,
                success=False,
                found_element=False,
                element_tag=None,
                element_text=None,
                element_index=None,
                confidence=0.0,
                reasoning="",
                response_time_ms=response_time_ms,
                error=str(e),
            )

    async def _test_action(
        self,
        instruction: str,
        expected_action: str,
        elements: list,
    ) -> ActionResult:
        """Test action selection."""
        logger.info(f"\nAction: \"{instruction}\" (expected: {expected_action})")

        start_time = time.time()

        try:
            result, action = await self.selector.select_action(instruction, elements)
            response_time_ms = (time.time() - start_time) * 1000

            action_correct = action.lower() == expected_action.lower()

            if result.found:
                inner = result.element.element if hasattr(result.element, 'element') else result.element
                status = "PASS" if action_correct else "WRONG_ACTION"
                logger.info(f"  [{status}] Element: [{result.index}] <{inner.tag_name}>")
                logger.info(f"  Action: {action} (expected: {expected_action})")
                logger.info(f"  Confidence: {result.confidence:.2f}")
                logger.info(f"  Response time: {response_time_ms:.0f}ms")
            else:
                logger.info(f"  [FAIL] No element found: {result.reasoning}")

            return ActionResult(
                instruction=instruction,
                expected_action=expected_action,
                actual_action=action,
                action_correct=action_correct,
                element_found=result.found,
                confidence=result.confidence,
                response_time_ms=response_time_ms,
            )

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            logger.error(f"  [ERROR] {e}")
            return ActionResult(
                instruction=instruction,
                expected_action=expected_action,
                actual_action="error",
                action_correct=False,
                element_found=False,
                confidence=0.0,
                response_time_ms=response_time_ms,
                error=str(e),
            )

    def generate_summary(self, site_results: list[SiteTestResult]) -> TestSummary:
        """Generate a summary of all test results."""
        all_queries: list[QueryResult] = []
        all_actions: list[ActionResult] = []

        for site in site_results:
            all_queries.extend(site.query_results)
            all_actions.extend(site.action_results)

        # Filter out error queries
        valid_queries = [q for q in all_queries if q.error is None]

        # Calculate category results
        category_results: dict[str, dict[str, int]] = {}
        for q in valid_queries:
            if q.category not in category_results:
                category_results[q.category] = {"total": 0, "success": 0, "meets_expectations": 0}
            category_results[q.category]["total"] += 1
            if q.success:
                category_results[q.category]["success"] += 1
            if q.meets_expectations:
                category_results[q.category]["meets_expectations"] += 1

        # Calculate timing stats
        response_times = [q.response_time_ms for q in valid_queries if q.response_time_ms > 0]
        avg_time = statistics.mean(response_times) if response_times else 0
        min_time = min(response_times) if response_times else 0
        max_time = max(response_times) if response_times else 0

        # Calculate confidence distribution
        confidence_buckets = {
            "0.0-0.3": 0,
            "0.3-0.5": 0,
            "0.5-0.7": 0,
            "0.7-0.9": 0,
            "0.9-1.0": 0,
        }
        confidences = []
        for q in valid_queries:
            if q.found_element:
                confidences.append(q.confidence)
                if q.confidence < 0.3:
                    confidence_buckets["0.0-0.3"] += 1
                elif q.confidence < 0.5:
                    confidence_buckets["0.3-0.5"] += 1
                elif q.confidence < 0.7:
                    confidence_buckets["0.5-0.7"] += 1
                elif q.confidence < 0.9:
                    confidence_buckets["0.7-0.9"] += 1
                else:
                    confidence_buckets["0.9-1.0"] += 1

        avg_confidence = statistics.mean(confidences) if confidences else 0

        return TestSummary(
            provider=self.provider_name,
            model=self.model_name,
            total_queries=len(valid_queries),
            successful_queries=sum(1 for q in valid_queries if q.success),
            failed_queries=sum(1 for q in valid_queries if not q.success),
            category_results=category_results,
            avg_response_time_ms=avg_time,
            min_response_time_ms=min_time,
            max_response_time_ms=max_time,
            confidence_buckets=confidence_buckets,
            avg_confidence=avg_confidence,
            site_results=site_results,
            timestamp=datetime.now().isoformat(),
        )


# =============================================================================
# Report Generation
# =============================================================================

def print_summary(summary: TestSummary) -> None:
    """Print a formatted summary to console."""
    print("\n" + "=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nProvider: {summary.provider}")
    print(f"Model: {summary.model}")
    print(f"Timestamp: {summary.timestamp}")

    print(f"\n--- Overall Results ---")
    success_rate = (summary.successful_queries / summary.total_queries * 100) if summary.total_queries > 0 else 0
    print(f"Total Queries: {summary.total_queries}")
    print(f"Successful: {summary.successful_queries} ({success_rate:.1f}%)")
    print(f"Failed: {summary.failed_queries}")

    print(f"\n--- Results by Category ---")
    for category, results in summary.category_results.items():
        cat_success = (results["success"] / results["total"] * 100) if results["total"] > 0 else 0
        cat_meets = (results["meets_expectations"] / results["total"] * 100) if results["total"] > 0 else 0
        print(f"  {category}: {results['success']}/{results['total']} success ({cat_success:.0f}%), "
              f"{results['meets_expectations']} meet expectations ({cat_meets:.0f}%)")

    print(f"\n--- Response Times ---")
    print(f"Average: {summary.avg_response_time_ms:.0f}ms")
    print(f"Min: {summary.min_response_time_ms:.0f}ms")
    print(f"Max: {summary.max_response_time_ms:.0f}ms")

    print(f"\n--- Confidence Distribution ---")
    print(f"Average confidence: {summary.avg_confidence:.2f}")
    for bucket, count in summary.confidence_buckets.items():
        print(f"  {bucket}: {count}")

    print(f"\n--- Per-Site Results ---")
    for site in summary.site_results:
        site_queries = [q for q in site.query_results if q.error is None]
        site_success = sum(1 for q in site_queries if q.success)
        site_rate = (site_success / len(site_queries) * 100) if site_queries else 0
        print(f"  {site.site_name}: {site_success}/{len(site_queries)} ({site_rate:.0f}%) "
              f"- {site.element_count} elements")

    print("\n" + "=" * 70)


def save_report(summary: TestSummary, output_path: str) -> None:
    """Save the test report as JSON."""
    report = {
        "provider": summary.provider,
        "model": summary.model,
        "timestamp": summary.timestamp,
        "total_queries": summary.total_queries,
        "successful_queries": summary.successful_queries,
        "failed_queries": summary.failed_queries,
        "success_rate": (summary.successful_queries / summary.total_queries) if summary.total_queries > 0 else 0,
        "category_results": summary.category_results,
        "timing": {
            "avg_ms": summary.avg_response_time_ms,
            "min_ms": summary.min_response_time_ms,
            "max_ms": summary.max_response_time_ms,
        },
        "confidence": {
            "average": summary.avg_confidence,
            "distribution": summary.confidence_buckets,
        },
        "site_results": [
            {
                "url": site.url,
                "name": site.site_name,
                "element_count": site.element_count,
                "extraction_time_ms": site.extraction_time_ms,
                "queries": [
                    {
                        "query": q.query,
                        "category": q.category,
                        "success": q.success,
                        "found": q.found_element,
                        "element_tag": q.element_tag,
                        "element_text": q.element_text,
                        "confidence": q.confidence,
                        "response_time_ms": q.response_time_ms,
                        "meets_expectations": q.meets_expectations,
                        "error": q.error,
                    }
                    for q in site.query_results
                ],
                "actions": [
                    {
                        "instruction": a.instruction,
                        "expected_action": a.expected_action,
                        "actual_action": a.actual_action,
                        "action_correct": a.action_correct,
                        "element_found": a.element_found,
                        "confidence": a.confidence,
                        "response_time_ms": a.response_time_ms,
                        "error": a.error,
                    }
                    for a in site.action_results
                ],
            }
            for site in summary.site_results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Report saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def get_available_providers() -> list[tuple[str, str]]:
    """Check which LLM providers are available."""
    available = []

    if os.getenv("ANTHROPIC_API_KEY"):
        available.append(("anthropic", "claude-3-5-sonnet-20241022"))

    if os.getenv("OPENAI_API_KEY"):
        available.append(("openai", "gpt-4o"))

    return available


async def main():
    parser = argparse.ArgumentParser(
        description="Test NaturalLanguageSelector with real LLM providers"
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "mock"],
        help="LLM provider to use (defaults to first available)",
    )
    parser.add_argument(
        "--model",
        help="Model to use (uses provider default if not specified)",
    )
    parser.add_argument(
        "--url",
        help="Test only this URL instead of all test sites",
    )
    parser.add_argument(
        "--site",
        choices=list(TEST_SITES.keys()),
        help="Test only this site",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output including element lists",
    )
    parser.add_argument(
        "--output", "-o",
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    # Determine provider
    available = get_available_providers()

    if args.provider:
        provider = args.provider
        model = args.model
    elif available:
        provider, model = available[0]
        if args.model:
            model = args.model
        logger.info(f"Using detected provider: {provider} ({model})")
    else:
        logger.warning("No API keys found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY")
        logger.info("Available options:")
        logger.info("  export ANTHROPIC_API_KEY=sk-ant-...")
        logger.info("  export OPENAI_API_KEY=sk-...")
        logger.info("\nFalling back to mock LLM client for demonstration...")
        provider = "mock"
        model = "mock"

    # Create client
    try:
        if provider == "mock":
            # Create a smarter mock client that always returns the first element
            # This is just for testing the infrastructure; real LLM tests need API keys
            class SmartMockClient(BaseLLMClient):
                """Mock client that returns first element and parses action verbs."""

                def _default_config(self):
                    from qontinui.extraction.web.llm_clients import LLMConfig
                    return LLMConfig(model="mock")

                async def complete(self, prompt: str) -> str:
                    # Determine action from prompt - check instruction line first
                    prompt_lower = prompt.lower()
                    action = "click"
                    # Look for action keywords (order matters - more specific first)
                    if "type in" in prompt_lower or "type into" in prompt_lower:
                        action = "type"
                    elif "hover over" in prompt_lower or "hover on" in prompt_lower:
                        action = "hover"
                    elif "focus on" in prompt_lower:
                        action = "focus"
                    elif "select" in prompt_lower and "option" in prompt_lower:
                        action = "select"

                    if "ACTION:" in prompt:
                        # Action selection prompt
                        return f"""INDEX: 0
ACTION: {action}
CONFIDENCE: 0.90
REASONING: Mock selection of first element for action '{action}'"""
                    else:
                        # Element selection prompt
                        return """INDEX: 0
CONFIDENCE: 0.90
REASONING: Mock selection of first matching element
ALTERNATIVES: none"""

            client = SmartMockClient()
        else:
            client = create_llm_client(provider, model=model)
    except ValueError as e:
        logger.error(f"Failed to create LLM client: {e}")
        return

    # Create tester
    tester = LLMSelectorTester(
        client=client,
        provider_name=provider,
        model_name=model or "default",
        verbose=args.verbose,
    )

    # Determine which sites to test
    if args.url:
        # Custom URL
        sites_to_test = {
            "custom": {
                "url": args.url,
                "name": "Custom URL",
                "queries": [
                    TestQuery(description="the main search input", category="input"),
                    TestQuery(description="any button", category="button"),
                    TestQuery(description="the first link", category="link"),
                    TestQuery(description="a navigation menu item", category="link"),
                ],
            }
        }
    elif args.site:
        # Specific site
        sites_to_test = {args.site: TEST_SITES[args.site]}
    else:
        # All test sites
        sites_to_test = TEST_SITES

    # Run tests
    logger.info(f"\nStarting LLM Selector Tests")
    logger.info(f"Provider: {provider}")
    logger.info(f"Model: {model or 'default'}")
    logger.info(f"Sites to test: {len(sites_to_test)}")

    site_results = []
    for site_key, site_config in sites_to_test.items():
        result = await tester.test_site(site_key, site_config)
        site_results.append(result)

    # Generate summary
    summary = tester.generate_summary(site_results)

    # Print summary
    print_summary(summary)

    # Save report if requested
    if args.output:
        save_report(summary, args.output)


if __name__ == "__main__":
    asyncio.run(main())
