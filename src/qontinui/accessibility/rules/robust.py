"""
WCAG Robust Rules (Principle 4)

These rules ensure content is robust enough to be interpreted reliably
by a wide variety of user agents, including assistive technologies.

Also includes related rules from other WCAG principles that are commonly checked.
"""

from __future__ import annotations

from typing import Any

from qontinui.accessibility.rules.base_rule import AccessibilityRule
from qontinui.accessibility.types import (
    AccessibilityIssue,
    AccessibilitySeverity,
    WCAGLevel,
)


class ButtonNameRule(AccessibilityRule):
    """WCAG 4.1.2: Buttons must have an accessible name.

    Buttons must have discernible text that describes their purpose.
    This can come from:
    - Button text content
    - aria-label
    - aria-labelledby
    - title (as fallback)
    """

    @property
    def rule_id(self) -> str:
        return "button-name"

    @property
    def wcag_criterion(self) -> str:
        return "4.1.2"

    @property
    def level(self) -> WCAGLevel:
        return WCAGLevel.A

    @property
    def severity(self) -> AccessibilitySeverity:
        return AccessibilitySeverity.CRITICAL

    @property
    def description(self) -> str:
        return "Buttons must have an accessible name"

    def applies_to(self, element: dict[str, Any]) -> bool:
        """Check if element is a button."""
        role = self.get_role(element)
        element_type = element.get("type")
        tag_name = element.get("tagName", "").lower()

        return (
            role == "button"
            or element_type == "button"
            or tag_name == "button"
            or (tag_name == "input" and element.get("inputType") in ["button", "submit", "reset"])
        )

    def check(self, element: dict[str, Any]) -> list[AccessibilityIssue]:
        """Check if button has an accessible name."""
        accessible_name = self.get_accessible_name(element)

        if not accessible_name or not accessible_name.strip():
            return [
                self._create_issue(
                    element,
                    message="Button does not have an accessible name",
                    suggestion=(
                        "Add text content to the button, or use aria-label or "
                        "aria-labelledby to provide an accessible name"
                    ),
                )
            ]

        return []


class LinkNameRule(AccessibilityRule):
    """WCAG 4.1.2 / 2.4.4: Links must have an accessible name.

    Links must have discernible text that describes their destination.
    """

    @property
    def rule_id(self) -> str:
        return "link-name"

    @property
    def wcag_criterion(self) -> str:
        return "2.4.4"

    @property
    def level(self) -> WCAGLevel:
        return WCAGLevel.A

    @property
    def severity(self) -> AccessibilitySeverity:
        return AccessibilitySeverity.SERIOUS

    @property
    def description(self) -> str:
        return "Links must have an accessible name"

    def applies_to(self, element: dict[str, Any]) -> bool:
        """Check if element is a link."""
        role = self.get_role(element)
        element_type = element.get("type")
        tag_name = element.get("tagName", "").lower()

        return role == "link" or element_type == "link" or tag_name == "a"

    def check(self, element: dict[str, Any]) -> list[AccessibilityIssue]:
        """Check if link has an accessible name."""
        accessible_name = self.get_accessible_name(element)

        if not accessible_name or not accessible_name.strip():
            return [
                self._create_issue(
                    element,
                    message="Link does not have an accessible name",
                    suggestion=(
                        "Add text content to the link, or use aria-label or "
                        "aria-labelledby to provide an accessible name. "
                        "Avoid using generic text like 'click here' or 'read more'."
                    ),
                )
            ]

        # Check for generic link text
        generic_texts = ["click here", "here", "read more", "more", "link", "learn more"]
        if accessible_name.strip().lower() in generic_texts:
            return [
                self._create_issue(
                    element,
                    message=f"Link has generic text '{accessible_name}' that does not describe its destination",
                    suggestion=(
                        "Use descriptive link text that indicates where the link goes. "
                        "For example, instead of 'Read more', use 'Read more about accessibility'."
                    ),
                    severity=AccessibilitySeverity.MODERATE,
                )
            ]

        return []


class FormInputLabelRule(AccessibilityRule):
    """WCAG 1.3.1 / 4.1.2: Form inputs must have labels.

    All form inputs must have an associated label that describes their purpose.
    """

    @property
    def rule_id(self) -> str:
        return "form-input-label"

    @property
    def wcag_criterion(self) -> str:
        return "1.3.1"

    @property
    def level(self) -> WCAGLevel:
        return WCAGLevel.A

    @property
    def severity(self) -> AccessibilitySeverity:
        return AccessibilitySeverity.CRITICAL

    @property
    def description(self) -> str:
        return "Form inputs must have labels"

    def applies_to(self, element: dict[str, Any]) -> bool:
        """Check if element is a form input."""
        role = self.get_role(element)
        element_type = element.get("type")
        tag_name = element.get("tagName", "").lower()

        form_roles = ["textbox", "searchbox", "spinbutton", "combobox", "listbox"]
        form_types = ["input", "textarea", "select"]

        return role in form_roles or element_type in form_types or tag_name in form_types

    def check(self, element: dict[str, Any]) -> list[AccessibilityIssue]:
        """Check if form input has a label."""
        accessible_name = self.get_accessible_name(element)

        if not accessible_name or not accessible_name.strip():
            tag_name = element.get("tagName", "input").lower()
            return [
                self._create_issue(
                    element,
                    message=f"{tag_name.capitalize()} does not have an accessible label",
                    suggestion=(
                        f"Add a <label> element associated with this {tag_name} using the 'for' attribute, "
                        "or use aria-label or aria-labelledby to provide an accessible name"
                    ),
                )
            ]

        return []


class ImageAltRule(AccessibilityRule):
    """WCAG 1.1.1: Images must have alternative text.

    Images that convey information must have alternative text.
    Decorative images should have empty alt attribute or role="presentation".
    """

    @property
    def rule_id(self) -> str:
        return "image-alt"

    @property
    def wcag_criterion(self) -> str:
        return "1.1.1"

    @property
    def level(self) -> WCAGLevel:
        return WCAGLevel.A

    @property
    def severity(self) -> AccessibilitySeverity:
        return AccessibilitySeverity.CRITICAL

    @property
    def description(self) -> str:
        return "Images must have alternative text"

    def applies_to(self, element: dict[str, Any]) -> bool:
        """Check if element is an image."""
        role = self.get_role(element)
        tag_name = element.get("tagName", "").lower()

        return role == "img" or tag_name == "img"

    def check(self, element: dict[str, Any]) -> list[AccessibilityIssue]:
        """Check if image has alternative text."""
        role = self.get_role(element)
        accessibility = self.get_accessibility(element)

        # Decorative images with role="presentation" or role="none" are OK
        if role in ["presentation", "none"]:
            return []

        accessible_name = self.get_accessible_name(element)

        # Check for aria-hidden="true" (decorative)
        if accessibility and accessibility.get("ariaHidden"):
            return []

        if accessible_name is None:
            # No alt attribute at all
            return [
                self._create_issue(
                    element,
                    message="Image does not have an alt attribute",
                    suggestion=(
                        "Add an alt attribute describing the image content, "
                        "or use alt='' for decorative images, "
                        "or use role='presentation' for decorative images"
                    ),
                )
            ]

        # Empty alt is OK for decorative images
        if accessible_name == "":
            return []

        return []


class KeyboardAccessRule(AccessibilityRule):
    """WCAG 2.1.1: Interactive elements must be keyboard accessible.

    All functionality must be operable through a keyboard interface.
    """

    @property
    def rule_id(self) -> str:
        return "keyboard-access"

    @property
    def wcag_criterion(self) -> str:
        return "2.1.1"

    @property
    def level(self) -> WCAGLevel:
        return WCAGLevel.A

    @property
    def severity(self) -> AccessibilitySeverity:
        return AccessibilitySeverity.CRITICAL

    @property
    def description(self) -> str:
        return "Interactive elements must be keyboard accessible"

    def applies_to(self, element: dict[str, Any]) -> bool:
        """Check if element should be keyboard accessible."""
        # Interactive roles that should be keyboard accessible
        interactive_roles = [
            "button",
            "link",
            "checkbox",
            "radio",
            "textbox",
            "searchbox",
            "spinbutton",
            "combobox",
            "listbox",
            "menuitem",
            "menuitemcheckbox",
            "menuitemradio",
            "option",
            "tab",
            "switch",
            "slider",
        ]

        role = self.get_role(element)
        element_type = element.get("type")

        # Check role
        if role in interactive_roles:
            return True

        # Check element type
        interactive_types = ["button", "input", "link", "checkbox", "radio", "select", "tab"]
        if element_type in interactive_types:
            return True

        # Check for click handlers or explicit tabindex
        if element.get("hasClickHandler"):
            return True

        return False

    def check(self, element: dict[str, Any]) -> list[AccessibilityIssue]:
        """Check if element is keyboard accessible."""
        accessibility = self.get_accessibility(element)

        if not accessibility:
            # No accessibility data - can't verify
            return []

        # Check if element is hidden from accessibility tree
        if accessibility.get("ariaHidden"):
            return []  # Hidden elements don't need keyboard access

        # Check if element is disabled
        if accessibility.get("ariaDisabled") or not element.get("enabled", True):
            return []  # Disabled elements don't need keyboard access

        # Check keyboard accessibility
        is_keyboard_accessible = accessibility.get("isKeyboardAccessible", False)
        is_in_tab_order = accessibility.get("isInTabOrder", False)

        if not is_keyboard_accessible and not is_in_tab_order:
            role = self.get_role(element) or element.get("type", "element")
            return [
                self._create_issue(
                    element,
                    message=f"Interactive {role} is not keyboard accessible",
                    suggestion=(
                        f"Add tabindex='0' to make this {role} focusable, "
                        "and ensure it responds to keyboard events (Enter/Space for activation)"
                    ),
                )
            ]

        return []
