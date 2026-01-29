"""
Accessibility Rules Module

Provides WCAG validation rules for accessibility checking.
"""

from qontinui.accessibility.rules.base_rule import AccessibilityRule
from qontinui.accessibility.rules.robust import (
    ButtonNameRule,
    FormInputLabelRule,
    ImageAltRule,
    KeyboardAccessRule,
    LinkNameRule,
)
from qontinui.accessibility.types import WCAGLevel


def get_all_rules(level: WCAGLevel = WCAGLevel.AA) -> list[AccessibilityRule]:
    """Get all accessibility rules up to and including the specified level.

    Args:
        level: The maximum WCAG level to include rules for.
               WCAGLevel.A includes only Level A rules.
               WCAGLevel.AA includes Level A and AA rules.
               WCAGLevel.AAA includes all rules.

    Returns:
        List of accessibility rules.
    """
    all_rules: list[AccessibilityRule] = [
        # Level A rules (WCAG 2.1)
        ButtonNameRule(),
        LinkNameRule(),
        FormInputLabelRule(),
        ImageAltRule(),
        KeyboardAccessRule(),
    ]

    # Filter based on level
    level_order = {"A": 1, "AA": 2, "AAA": 3}
    max_level = level_order.get(level.value, 2)

    return [rule for rule in all_rules if level_order.get(rule.level.value, 1) <= max_level]


__all__ = [
    "AccessibilityRule",
    "ButtonNameRule",
    "FormInputLabelRule",
    "ImageAltRule",
    "KeyboardAccessRule",
    "LinkNameRule",
    "get_all_rules",
]
