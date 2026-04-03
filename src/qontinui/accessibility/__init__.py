"""
Accessibility Module

Provides WCAG validation and accessibility-aware automation support.
"""

from qontinui.accessibility.types import (
    AccessibilityIssue,
    AccessibilityReport,
    AccessibilitySeverity,
    AriaCheckedState,
    ElementAccessibility,
    WCAGLevel,
)
from qontinui.accessibility.validator import AccessibilityValidator

__all__ = [
    "AccessibilityValidator",
    "AccessibilityIssue",
    "AccessibilityReport",
    "AccessibilitySeverity",
    "AriaCheckedState",
    "ElementAccessibility",
    "WCAGLevel",
]
