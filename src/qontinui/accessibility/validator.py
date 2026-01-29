"""
Accessibility Validator

Validates UI elements against WCAG guidelines.
"""

from __future__ import annotations

import time
from typing import Any

from qontinui.accessibility.rules import AccessibilityRule, get_all_rules
from qontinui.accessibility.types import (
    AccessibilityIssue,
    AccessibilityReport,
    AccessibilitySeverity,
    WCAGLevel,
)


class AccessibilityValidator:
    """Validates UI elements against WCAG guidelines.

    This validator checks elements for accessibility issues based on
    WCAG 2.1 success criteria. It supports different conformance levels
    (A, AA, AAA) and provides detailed reports with suggested fixes.

    Example:
        ```python
        validator = AccessibilityValidator(level=WCAGLevel.AA)

        # Validate a single element
        issues = validator.validate_element(element_data)

        # Validate multiple elements and get a report
        report = validator.validate_elements(elements, url="https://example.com")
        print(f"Found {report.failed_count} issues")
        ```
    """

    def __init__(
        self,
        level: WCAGLevel = WCAGLevel.AA,
        custom_rules: list[AccessibilityRule] | None = None,
    ) -> None:
        """Initialize the validator.

        Args:
            level: The WCAG conformance level to validate against.
                   Defaults to AA, which is the most common requirement.
            custom_rules: Additional custom rules to include in validation.
        """
        self.level = level
        self.rules = get_all_rules(level)
        if custom_rules:
            self.rules.extend(custom_rules)

    def validate_element(self, element: dict[str, Any]) -> list[AccessibilityIssue]:
        """Validate a single element against accessibility rules.

        Args:
            element: Element data including type, role, accessibility info, etc.
                     Expected to have an 'accessibility' key with ElementAccessibility data.

        Returns:
            List of accessibility issues found (empty if element passes all checks).
        """
        issues: list[AccessibilityIssue] = []

        for rule in self.rules:
            if rule.applies_to(element):
                rule_issues = rule.check(element)
                issues.extend(rule_issues)

        return issues

    def validate_elements(
        self,
        elements: list[dict[str, Any]],
        url: str = "",
    ) -> AccessibilityReport:
        """Validate multiple elements and generate a report.

        Args:
            elements: List of element data dictionaries.
            url: The URL of the page being validated (for reporting).

        Returns:
            An AccessibilityReport with all issues and summary statistics.
        """
        start_time = time.time()
        all_issues: list[AccessibilityIssue] = []
        passed_count = 0
        failed_count = 0

        for element in elements:
            element_issues = self.validate_element(element)
            if element_issues:
                all_issues.extend(element_issues)
                failed_count += 1
            else:
                passed_count += 1

        duration_ms = (time.time() - start_time) * 1000

        # Determine WCAG compliance
        level_a_issues = [issue for issue in all_issues if issue.level == WCAGLevel.A]
        level_aa_issues = [
            issue for issue in all_issues if issue.level in (WCAGLevel.A, WCAGLevel.AA)
        ]

        # Consider critical/serious issues as blocking for compliance
        blocking_a = any(
            issue.severity in (AccessibilitySeverity.CRITICAL, AccessibilitySeverity.SERIOUS)
            for issue in level_a_issues
        )
        blocking_aa = any(
            issue.severity in (AccessibilitySeverity.CRITICAL, AccessibilitySeverity.SERIOUS)
            for issue in level_aa_issues
        )

        meets_wcag_a = not blocking_a
        meets_wcag_aa = not blocking_aa

        # Generate summary
        summary = self._generate_summary(all_issues, meets_wcag_a, meets_wcag_aa)

        return AccessibilityReport(
            timestamp=int(time.time() * 1000),
            url=url,
            elements_scanned=len(elements),
            issues=all_issues,
            passed_count=passed_count,
            failed_count=failed_count,
            meets_wcag_a=meets_wcag_a,
            meets_wcag_aa=meets_wcag_aa,
            summary=summary,
            duration_ms=duration_ms,
        )

    def _generate_summary(
        self,
        issues: list[AccessibilityIssue],
        meets_wcag_a: bool,
        meets_wcag_aa: bool,
    ) -> str:
        """Generate a human-readable summary of the validation results.

        Args:
            issues: All issues found.
            meets_wcag_a: Whether the page meets WCAG A.
            meets_wcag_aa: Whether the page meets WCAG AA.

        Returns:
            Summary string.
        """
        if not issues:
            return "No accessibility issues found. Page meets WCAG 2.1 AA standards."

        # Count by severity
        critical = sum(1 for i in issues if i.severity == AccessibilitySeverity.CRITICAL)
        serious = sum(1 for i in issues if i.severity == AccessibilitySeverity.SERIOUS)
        moderate = sum(1 for i in issues if i.severity == AccessibilitySeverity.MODERATE)
        minor = sum(1 for i in issues if i.severity == AccessibilitySeverity.MINOR)

        parts = []
        if critical:
            parts.append(f"{critical} critical")
        if serious:
            parts.append(f"{serious} serious")
        if moderate:
            parts.append(f"{moderate} moderate")
        if minor:
            parts.append(f"{minor} minor")

        issues_summary = ", ".join(parts)
        total = len(issues)

        compliance = []
        if meets_wcag_a:
            compliance.append("WCAG 2.1 A")
        if meets_wcag_aa:
            compliance.append("WCAG 2.1 AA")

        if compliance:
            compliance_str = f"Meets {' and '.join(compliance)}."
        else:
            compliance_str = "Does not meet WCAG 2.1 A compliance."

        return f"Found {total} accessibility issues ({issues_summary}). {compliance_str}"

    def add_rule(self, rule: AccessibilityRule) -> None:
        """Add a custom rule to the validator.

        Args:
            rule: The rule to add.
        """
        self.rules.append(rule)

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by its ID.

        Args:
            rule_id: The ID of the rule to remove.

        Returns:
            True if a rule was removed, False if not found.
        """
        original_count = len(self.rules)
        self.rules = [r for r in self.rules if r.rule_id != rule_id]
        return len(self.rules) < original_count

    def get_rule_ids(self) -> list[str]:
        """Get the IDs of all active rules.

        Returns:
            List of rule IDs.
        """
        return [rule.rule_id for rule in self.rules]
