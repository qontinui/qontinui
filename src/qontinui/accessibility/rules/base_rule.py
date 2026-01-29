"""
Base Accessibility Rule

Abstract base class for WCAG accessibility rules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from qontinui.accessibility.types import (
    AccessibilityIssue,
    AccessibilitySeverity,
    WCAGLevel,
)

if TYPE_CHECKING:
    pass


class AccessibilityRule(ABC):
    """Abstract base class for accessibility validation rules.

    Each rule checks for a specific WCAG success criterion.
    """

    @property
    @abstractmethod
    def rule_id(self) -> str:
        """Unique identifier for this rule."""
        ...

    @property
    @abstractmethod
    def wcag_criterion(self) -> str:
        """The WCAG success criterion this rule checks (e.g., '4.1.2')."""
        ...

    @property
    @abstractmethod
    def level(self) -> WCAGLevel:
        """The WCAG conformance level for this rule."""
        ...

    @property
    @abstractmethod
    def severity(self) -> AccessibilitySeverity:
        """Default severity when this rule fails."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this rule checks."""
        ...

    @abstractmethod
    def applies_to(self, element: dict[str, Any]) -> bool:
        """Check if this rule should be applied to the given element.

        Args:
            element: Element data including type, role, accessibility info, etc.

        Returns:
            True if this rule should check this element.
        """
        ...

    @abstractmethod
    def check(self, element: dict[str, Any]) -> list[AccessibilityIssue]:
        """Check the element for accessibility issues.

        Args:
            element: Element data including type, role, accessibility info, etc.

        Returns:
            List of issues found (empty if element passes).
        """
        ...

    def _create_issue(
        self,
        element: dict[str, Any],
        message: str,
        suggestion: str,
        severity: AccessibilitySeverity | None = None,
    ) -> AccessibilityIssue:
        """Helper to create an accessibility issue.

        Args:
            element: The element with the issue.
            message: Description of the issue.
            suggestion: How to fix the issue.
            severity: Override default severity (optional).

        Returns:
            AccessibilityIssue instance.
        """
        import uuid

        element_id = element.get("id", "unknown")
        element_selector = element.get("_selector", None)

        return AccessibilityIssue(
            id=str(uuid.uuid4()),
            wcag_criterion=self.wcag_criterion,
            severity=severity or self.severity,
            level=self.level,
            message=message,
            element_id=element_id,
            element_selector=element_selector,
            suggestion=suggestion,
            rule_id=self.rule_id,
        )

    def get_accessibility(self, element: dict[str, Any]) -> dict[str, Any] | None:
        """Get the accessibility data from an element.

        Args:
            element: Element data.

        Returns:
            Accessibility data dictionary or None.
        """
        return element.get("accessibility")

    def get_role(self, element: dict[str, Any]) -> str | None:
        """Get the computed role of an element.

        Args:
            element: Element data.

        Returns:
            The element's role or None.
        """
        accessibility = self.get_accessibility(element)
        if accessibility:
            return accessibility.get("role")
        return None

    def get_accessible_name(self, element: dict[str, Any]) -> str | None:
        """Get the accessible name of an element.

        Args:
            element: Element data.

        Returns:
            The accessible name or None.
        """
        accessibility = self.get_accessibility(element)
        if accessibility:
            return accessibility.get("accessibleName")
        return None
