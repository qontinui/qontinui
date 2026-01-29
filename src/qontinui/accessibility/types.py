"""
Accessibility Type Definitions

Pydantic models for accessibility validation and WCAG compliance.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class WCAGLevel(str, Enum):
    """WCAG conformance level."""

    A = "A"
    AA = "AA"
    AAA = "AAA"


class AccessibilitySeverity(str, Enum):
    """Severity of accessibility issues."""

    CRITICAL = "critical"
    SERIOUS = "serious"
    MODERATE = "moderate"
    MINOR = "minor"


# Type alias for ARIA checked state
AriaCheckedState = bool | Literal["mixed"]


class ElementAccessibility(BaseModel):
    """Accessibility information for a UI element.

    Captures ARIA attributes and accessibility-relevant properties
    following the WAI-ARIA specification.
    """

    role: str = Field(description="The element's computed role (explicit or implicit)")
    accessible_name: str | None = Field(
        None,
        alias="accessibleName",
        description="Computed accessible name following ARIA name computation",
    )
    accessible_description: str | None = Field(
        None,
        alias="accessibleDescription",
        description="Computed accessible description",
    )
    aria_label: str | None = Field(
        None, alias="ariaLabel", description="Value of aria-label attribute"
    )
    aria_labelled_by: str | None = Field(
        None, alias="ariaLabelledBy", description="Value of aria-labelledby attribute"
    )
    aria_described_by: str | None = Field(
        None, alias="ariaDescribedBy", description="Value of aria-describedby attribute"
    )
    aria_expanded: bool | None = Field(
        None,
        alias="ariaExpanded",
        description="Whether element is expanded (for expandable elements)",
    )
    aria_selected: bool | None = Field(
        None,
        alias="ariaSelected",
        description="Whether element is selected (for selectable elements)",
    )
    aria_checked: AriaCheckedState | None = Field(
        None,
        alias="ariaChecked",
        description="Checked state (for checkboxes, can be true/false/'mixed')",
    )
    aria_hidden: bool | None = Field(
        None,
        alias="ariaHidden",
        description="Whether element is hidden from accessibility tree",
    )
    aria_disabled: bool | None = Field(
        None,
        alias="ariaDisabled",
        description="Whether element is disabled via aria-disabled",
    )
    aria_required: bool | None = Field(
        None,
        alias="ariaRequired",
        description="Whether element is required (for form inputs)",
    )
    aria_live: Literal["off", "polite", "assertive"] | None = Field(
        None, alias="ariaLive", description="Current aria-live value for live regions"
    )
    tab_index: int = Field(alias="tabIndex", description="Tab index value")
    is_in_tab_order: bool = Field(
        alias="isInTabOrder",
        description="Whether element is in the tab order (tabindex >= 0 or naturally focusable)",
    )
    is_keyboard_accessible: bool = Field(
        alias="isKeyboardAccessible",
        description="Whether element can receive keyboard focus",
    )
    implicit_role: str | None = Field(
        None,
        alias="implicitRole",
        description="The implicit role based on element type (before explicit role override)",
    )
    has_explicit_role: bool = Field(
        alias="hasExplicitRole",
        description="Whether element has an explicit role attribute",
    )

    model_config = {"populate_by_name": True}


class AccessibilityIssue(BaseModel):
    """An accessibility issue found during validation."""

    id: str = Field(description="Unique identifier for this issue instance")
    wcag_criterion: str = Field(
        alias="wcagCriterion",
        description="The WCAG success criterion this issue relates to (e.g., '4.1.2')",
    )
    severity: AccessibilitySeverity = Field(description="How severe this issue is")
    level: WCAGLevel = Field(description="WCAG conformance level this criterion belongs to")
    message: str = Field(description="Human-readable description of the issue")
    element_id: str = Field(alias="elementId", description="ID of the element with the issue")
    element_selector: str | None = Field(
        None, alias="elementSelector", description="Selector to find the element"
    )
    suggestion: str = Field(description="Suggested fix for the issue")
    rule_id: str = Field(alias="ruleId", description="The rule ID that detected this issue")

    model_config = {"populate_by_name": True}


class AccessibilityReport(BaseModel):
    """Accessibility validation report."""

    timestamp: int = Field(description="When the validation was performed (Unix timestamp)")
    url: str = Field(description="URL of the page that was validated")
    elements_scanned: int = Field(
        alias="elementsScanned", description="Number of elements that were scanned"
    )
    issues: list[AccessibilityIssue] = Field(
        default_factory=list, description="All issues found during validation"
    )
    passed_count: int = Field(alias="passedCount", description="Number of checks that passed")
    failed_count: int = Field(alias="failedCount", description="Number of checks that failed")
    meets_wcag_a: bool = Field(
        alias="meetsWCAG_A", description="Whether the page meets WCAG 2.1 Level A"
    )
    meets_wcag_aa: bool = Field(
        alias="meetsWCAG_AA", description="Whether the page meets WCAG 2.1 Level AA"
    )
    summary: str = Field(description="Human-readable summary of the validation")
    duration_ms: float = Field(
        alias="durationMs", description="Duration of the validation in milliseconds"
    )

    model_config = {"populate_by_name": True}

    def get_issues_by_severity(self, severity: AccessibilitySeverity) -> list[AccessibilityIssue]:
        """Get issues filtered by severity."""
        return [issue for issue in self.issues if issue.severity == severity]

    def get_issues_by_level(self, level: WCAGLevel) -> list[AccessibilityIssue]:
        """Get issues filtered by WCAG level."""
        return [issue for issue in self.issues if issue.level == level]

    def get_critical_issues(self) -> list[AccessibilityIssue]:
        """Get only critical issues."""
        return self.get_issues_by_severity(AccessibilitySeverity.CRITICAL)

    def has_blocking_issues(self) -> bool:
        """Check if there are any critical or serious issues."""
        return any(
            issue.severity in (AccessibilitySeverity.CRITICAL, AccessibilitySeverity.SERIOUS)
            for issue in self.issues
        )
