#!/usr/bin/env python
"""
Accessibility audit example using qontinui's accessibility extractor.

Demonstrates extracting the accessibility tree from a webpage and
analyzing it for common accessibility issues.

Usage:
    poetry run python scripts/examples/accessibility_audit_example.py
    poetry run python scripts/examples/accessibility_audit_example.py --url https://example.com
    poetry run python scripts/examples/accessibility_audit_example.py --no-headless

Features demonstrated:
- Accessibility tree extraction
- Analyzing missing labels and roles
- Interactive element accessibility check
- Generating an accessibility report
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

from playwright.async_api import Page, async_playwright

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from qontinui.extraction.web import (
    A11yNode,
    A11yTree,
    AccessibilityExtractor,
    InteractiveElementExtractor,
    enrich_with_accessibility,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class AccessibilityIssue:
    """Represents an accessibility issue found during audit."""

    severity: str  # "error", "warning", "info"
    category: str  # "missing-label", "missing-role", etc.
    message: str
    element_info: dict = field(default_factory=dict)


@dataclass
class AccessibilityReport:
    """Complete accessibility audit report."""

    url: str
    total_elements: int
    total_a11y_nodes: int
    issues: list[AccessibilityIssue] = field(default_factory=list)
    statistics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "total_elements": self.total_elements,
            "total_a11y_nodes": self.total_a11y_nodes,
            "issue_count": len(self.issues),
            "issues_by_severity": {
                "error": len([i for i in self.issues if i.severity == "error"]),
                "warning": len([i for i in self.issues if i.severity == "warning"]),
                "info": len([i for i in self.issues if i.severity == "info"]),
            },
            "issues": [
                {
                    "severity": i.severity,
                    "category": i.category,
                    "message": i.message,
                    "element": i.element_info,
                }
                for i in self.issues
            ],
            "statistics": self.statistics,
        }


class AccessibilityAuditor:
    """
    Performs accessibility audits on web pages.

    Uses qontinui's AccessibilityExtractor to get the a11y tree,
    then analyzes it for common issues.
    """

    def __init__(self):
        self.extractor = InteractiveElementExtractor()
        self.a11y_extractor = AccessibilityExtractor()

    async def audit_page(self, page: Page) -> AccessibilityReport:
        """
        Perform a comprehensive accessibility audit.

        Args:
            page: Playwright page to audit

        Returns:
            AccessibilityReport with findings
        """
        url = page.url
        issues: list[AccessibilityIssue] = []

        # Extract accessibility tree
        logger.info("Extracting accessibility tree...")
        a11y_tree = await self.a11y_extractor.extract_tree(page)

        # Extract interactive elements
        logger.info("Extracting interactive elements...")
        elements = await self.extractor.extract_interactive_elements(page, "audit")

        logger.info(f"Found {len(elements)} interactive elements")
        logger.info(f"Accessibility tree has {a11y_tree.node_count} nodes")

        # Enrich elements with accessibility data
        logger.info("Matching elements to accessibility tree...")
        enriched = await enrich_with_accessibility(elements, page)

        # Analyze for issues
        logger.info("Analyzing for accessibility issues...")

        # Check 1: Interactive elements without accessible names
        issues.extend(self._check_missing_names(enriched))

        # Check 2: Missing roles on interactive elements
        issues.extend(self._check_missing_roles(enriched))

        # Check 3: Images without alt text (from a11y tree)
        issues.extend(self._check_missing_alt_text(a11y_tree))

        # Check 4: Form inputs without labels
        issues.extend(self._check_form_labels(enriched))

        # Check 5: Low confidence matches (might indicate a11y issues)
        issues.extend(self._check_low_confidence_matches(enriched))

        # Collect statistics
        statistics = self._collect_statistics(enriched, a11y_tree)

        return AccessibilityReport(
            url=url,
            total_elements=len(elements),
            total_a11y_nodes=a11y_tree.node_count,
            issues=issues,
            statistics=statistics,
        )

    def _check_missing_names(self, enriched) -> list[AccessibilityIssue]:
        """Check for interactive elements without accessible names."""
        issues = []

        for item in enriched:
            elem = item.element
            # Interactive elements should have accessible names
            if elem.element_type in ("button", "a", "aria_button", "aria_link"):
                has_name = bool(elem.text or elem.aria_label or item.a11y_name)
                if not has_name:
                    issues.append(
                        AccessibilityIssue(
                            severity="error",
                            category="missing-name",
                            message=f"Interactive element <{elem.tag_name}> has no accessible name",
                            element_info={
                                "tag": elem.tag_name,
                                "type": elem.element_type,
                                "selector": elem.selector,
                            },
                        )
                    )

        return issues

    def _check_missing_roles(self, enriched) -> list[AccessibilityIssue]:
        """Check for interactive elements without proper ARIA roles."""
        issues = []

        for item in enriched:
            elem = item.element

            # Check clickable divs/spans (might need button role)
            if elem.element_type.startswith("clickable_"):
                if not elem.aria_role:
                    issues.append(
                        AccessibilityIssue(
                            severity="warning",
                            category="missing-role",
                            message=f"Clickable <{elem.tag_name}> should have role='button'",
                            element_info={
                                "tag": elem.tag_name,
                                "text": elem.text[:30] if elem.text else None,
                                "selector": elem.selector,
                            },
                        )
                    )

            # Check tabindex elements
            if elem.element_type.startswith("tabindex_"):
                if not elem.aria_role:
                    issues.append(
                        AccessibilityIssue(
                            severity="info",
                            category="missing-role",
                            message="Element with tabindex should have an ARIA role",
                            element_info={
                                "tag": elem.tag_name,
                                "selector": elem.selector,
                            },
                        )
                    )

        return issues

    def _check_missing_alt_text(self, tree: A11yTree) -> list[AccessibilityIssue]:
        """Check for images without alt text in the a11y tree."""
        issues = []

        images = tree.find_by_role("img") + tree.find_by_role("image")

        for img in images:
            if not img.name and not img.description:
                issues.append(
                    AccessibilityIssue(
                        severity="error",
                        category="missing-alt",
                        message="Image element has no alt text",
                        element_info={
                            "role": img.role,
                            "a11y_name": img.name or "(none)",
                        },
                    )
                )

        return issues

    def _check_form_labels(self, enriched) -> list[AccessibilityIssue]:
        """Check for form inputs without labels."""
        issues = []

        for item in enriched:
            elem = item.element

            # Check input elements
            if elem.tag_name == "input":
                has_label = bool(elem.aria_label or item.a11y_name or item.a11y_description)
                if not has_label:
                    issues.append(
                        AccessibilityIssue(
                            severity="error",
                            category="missing-label",
                            message="Input element has no label",
                            element_info={
                                "tag": elem.tag_name,
                                "type": elem.element_type,
                                "selector": elem.selector,
                            },
                        )
                    )

        return issues

    def _check_low_confidence_matches(self, enriched) -> list[AccessibilityIssue]:
        """Flag elements with low a11y match confidence."""
        issues = []

        for item in enriched:
            if 0 < item.match_confidence < 0.5:
                issues.append(
                    AccessibilityIssue(
                        severity="info",
                        category="low-confidence",
                        message=f"Element has low accessibility match confidence ({item.match_confidence:.2f})",
                        element_info={
                            "tag": item.element.tag_name,
                            "text": item.element.text[:30] if item.element.text else None,
                        },
                    )
                )

        return issues

    def _collect_statistics(self, enriched, tree: A11yTree) -> dict:
        """Collect statistics about the page's accessibility."""
        # Count elements by type
        element_types = {}
        for item in enriched:
            t = item.element.element_type
            element_types[t] = element_types.get(t, 0) + 1

        # Count a11y roles
        a11y_roles = {}
        if tree.root:
            self._count_roles(tree.root, a11y_roles)

        # Match statistics
        matched = sum(1 for e in enriched if e.match_confidence > 0)
        high_confidence = sum(1 for e in enriched if e.match_confidence >= 0.8)

        return {
            "element_types": element_types,
            "a11y_roles": a11y_roles,
            "total_enriched": len(enriched),
            "matched_to_a11y": matched,
            "high_confidence_matches": high_confidence,
            "match_rate": matched / len(enriched) if enriched else 0,
        }

    def _count_roles(self, node: A11yNode, counts: dict):
        """Recursively count a11y roles."""
        if node.role:
            counts[node.role] = counts.get(node.role, 0) + 1
        for child in node.children:
            self._count_roles(child, counts)


async def demo_tree_extraction(page: Page):
    """
    Demo: Extract and display the accessibility tree.
    """
    logger.info("=" * 60)
    logger.info("DEMO: Accessibility Tree Extraction")
    logger.info("=" * 60)

    extractor = AccessibilityExtractor()
    tree = await extractor.extract_tree(page)

    logger.info(f"\nAccessibility tree has {tree.node_count} nodes")

    # Show tree structure (abbreviated)
    logger.info("\nTree structure (first 30 lines):")
    tree_text = tree.to_text()
    for line in tree_text.split("\n")[:30]:
        logger.info(f"  {line}")

    # Find specific roles
    logger.info("\n--- Elements by role ---")
    roles_to_find = ["button", "link", "textbox", "heading", "navigation"]

    for role in roles_to_find:
        nodes = tree.find_by_role(role)
        logger.info(f"  {role}: {len(nodes)} found")

        # Show first 3
        for node in nodes[:3]:
            name = node.name[:40] if node.name else "(no name)"
            logger.info(f"    - {name}")


async def demo_enrichment(page: Page):
    """
    Demo: Enrich DOM elements with accessibility data.
    """
    logger.info("=" * 60)
    logger.info("DEMO: Element Enrichment")
    logger.info("=" * 60)

    elem_extractor = InteractiveElementExtractor()
    elements = await elem_extractor.extract_interactive_elements(page, "enrich")

    logger.info(f"Extracted {len(elements)} interactive elements")

    # Enrich with accessibility
    enriched = await enrich_with_accessibility(elements, page)

    logger.info(f"Enriched {len(enriched)} elements with accessibility data")

    # Show some examples
    logger.info("\n--- Enriched element examples ---")
    for item in enriched[:10]:
        elem = item.element
        logger.info(f"\n<{elem.tag_name}> {elem.text[:30] if elem.text else '(no text)'}")
        logger.info(f"  DOM selector: {elem.selector}")
        logger.info(f"  A11y role: {item.a11y_role or '(none)'}")
        logger.info(f"  A11y name: {item.a11y_name or '(none)'}")
        logger.info(f"  Match confidence: {item.match_confidence:.2f}")
        if item.a11y_disabled is not None:
            logger.info(f"  Disabled: {item.a11y_disabled}")


async def demo_full_audit(page: Page, output_path: Path | None = None):
    """
    Demo: Full accessibility audit with report.
    """
    logger.info("=" * 60)
    logger.info("DEMO: Full Accessibility Audit")
    logger.info("=" * 60)

    auditor = AccessibilityAuditor()
    report = await auditor.audit_page(page)

    # Print summary
    logger.info("\n--- Audit Summary ---")
    logger.info(f"URL: {report.url}")
    logger.info(f"Interactive elements: {report.total_elements}")
    logger.info(f"Accessibility nodes: {report.total_a11y_nodes}")
    logger.info(f"Issues found: {len(report.issues)}")

    # Issues by severity
    errors = [i for i in report.issues if i.severity == "error"]
    warnings = [i for i in report.issues if i.severity == "warning"]
    info = [i for i in report.issues if i.severity == "info"]

    logger.info(f"  - Errors: {len(errors)}")
    logger.info(f"  - Warnings: {len(warnings)}")
    logger.info(f"  - Info: {len(info)}")

    # Show errors
    if errors:
        logger.info("\n--- Errors ---")
        for issue in errors[:10]:
            logger.info(f"  [{issue.category}] {issue.message}")

    # Show warnings
    if warnings:
        logger.info("\n--- Warnings ---")
        for issue in warnings[:10]:
            logger.info(f"  [{issue.category}] {issue.message}")

    # Statistics
    logger.info("\n--- Statistics ---")
    stats = report.statistics
    logger.info(f"  A11y match rate: {stats.get('match_rate', 0) * 100:.1f}%")
    logger.info(f"  High confidence matches: {stats.get('high_confidence_matches', 0)}")

    # Element types
    if stats.get("element_types"):
        logger.info(f"  Element types: {json.dumps(stats['element_types'], indent=4)}")

    # Save report if output path provided
    if output_path:
        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"\nReport saved to: {output_path}")

    return report


async def main():
    parser = argparse.ArgumentParser(description="Accessibility audit example using qontinui")
    parser.add_argument(
        "--url",
        default="https://github.com",
        help="URL to audit (default: https://github.com)",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Run browser in visible mode",
    )
    parser.add_argument(
        "--output",
        help="Output file for JSON report",
    )
    parser.add_argument(
        "--demo",
        choices=["tree", "enrich", "audit", "all"],
        default="all",
        help="Which demo to run (default: all)",
    )
    args = parser.parse_args()

    headless = not args.no_headless

    logger.info("Accessibility Audit Example")
    logger.info(f"URL: {args.url}")
    logger.info(f"Headless: {headless}")
    logger.info("")

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

        if args.demo in ("tree", "all"):
            await demo_tree_extraction(page)

        if args.demo in ("enrich", "all"):
            await demo_enrichment(page)

        if args.demo in ("audit", "all"):
            output_path = Path(args.output) if args.output else None
            await demo_full_audit(page, output_path)

        await browser.close()

    logger.info("\n" + "=" * 60)
    logger.info("Demo complete!")
    logger.info("=" * 60)
    logger.info("\nKey accessibility concepts:")
    logger.info("1. Every interactive element needs an accessible name")
    logger.info("2. Custom clickable elements need proper ARIA roles")
    logger.info("3. Form inputs need labels (explicit or via aria-label)")
    logger.info("4. Images need alt text for screen readers")


if __name__ == "__main__":
    asyncio.run(main())
