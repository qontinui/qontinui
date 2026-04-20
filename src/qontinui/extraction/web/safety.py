"""
Safety configuration and element risk analysis for web extraction.

This module provides mechanisms to prevent automated Playwright crawlers
from clicking dangerous buttons like "Delete Account", "Purchase", etc.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from playwright.async_api import ElementHandle, Page


class ActionRisk(Enum):
    """Risk levels for element interactions."""

    SAFE = "safe"  # Navigation, view, open
    CAUTION = "caution"  # Edit, change, update
    DANGEROUS = "dangerous"  # Delete, remove, cancel subscription
    BLOCKED = "blocked"  # Never click automatically


@dataclass
class SafetyConfig:
    """Configuration for safe element interaction during web extraction."""

    # Keywords that indicate dangerous actions (case-insensitive)
    dangerous_keywords: list[str] = field(
        default_factory=lambda: [
            # Deletion
            "delete",
            "remove",
            "erase",
            "destroy",
            "purge",
            # Account actions
            "deactivate",
            "close account",
            "cancel subscription",
            "unsubscribe",
            "terminate",
            "revoke",
            # Irreversible
            "permanent",
            "cannot be undone",
            "irreversible",
            # Financial
            "purchase",
            "buy now",
            "pay",
            "checkout",
            "place order",
            "confirm payment",
            "submit order",
            "add to cart",
            # Auth
            "logout",
            "log out",
            "sign out",
            "disconnect",
            # Data
            "clear all",
            "reset",
            "factory reset",
            "wipe",
            # Destructive actions
            "confirm delete",
            "yes, delete",
            "i understand",
        ]
    )

    # Keywords that indicate safe navigation
    safe_keywords: list[str] = field(
        default_factory=lambda: [
            "view",
            "see",
            "show",
            "open",
            "expand",
            "collapse",
            "details",
            "more",
            "less",
            "next",
            "previous",
            "back",
            "menu",
            "nav",
            "tab",
            "home",
            "about",
            "help",
            "info",
            "search",
            "filter",
            "sort",
            "list",
            "grid",
            "settings",
            "preferences",
            "profile",
            "dashboard",
            "overview",
        ]
    )

    # CSS selectors to always block
    blocked_selectors: list[str] = field(
        default_factory=lambda: [
            '[data-testid*="delete"]',
            '[data-testid*="remove"]',
            '[data-testid*="destroy"]',
            '[class*="danger"]',
            '[class*="destructive"]',
            'button[type="submit"][form*="delete"]',
            ".btn-danger",
            ".destructive-action",
            '[aria-label*="delete"]',
            '[aria-label*="remove"]',
            '[title*="delete"]',
            '[title*="remove"]',
        ]
    )

    # Selectors that are always safe
    safe_selectors: list[str] = field(
        default_factory=lambda: [
            "nav a",
            '[role="navigation"] a',
            ".nav-link",
            ".menu-item",
            '[role="tab"]',
            ".breadcrumb a",
            '[role="menuitem"]',
            ".sidebar a",
            ".header a",
            ".footer a",
        ]
    )

    # ARIA roles that indicate dangerous actions
    dangerous_roles: list[str] = field(
        default_factory=lambda: [
            "alertdialog",  # Usually confirmation dialogs
        ]
    )

    # Maximum risk level to auto-click
    max_auto_click_risk: ActionRisk = ActionRisk.SAFE

    # Require user confirmation for caution-level actions
    prompt_for_caution: bool = True

    # Dry run mode - identify but don't click
    dry_run: bool = False

    # Maximum elements to extract per page
    max_elements_per_page: int = 100

    # Maximum depth for crawling (clicks deep)
    max_depth: int = 3

    # Minimum element size to consider
    min_element_size: tuple[int, int] = (10, 10)

    def add_dangerous_keyword(self, keyword: str) -> None:
        """Add a custom dangerous keyword."""
        if keyword.lower() not in [k.lower() for k in self.dangerous_keywords]:
            self.dangerous_keywords.append(keyword.lower())

    def add_safe_keyword(self, keyword: str) -> None:
        """Add a custom safe keyword."""
        if keyword.lower() not in [k.lower() for k in self.safe_keywords]:
            self.safe_keywords.append(keyword.lower())

    def add_blocked_selector(self, selector: str) -> None:
        """Add a custom blocked selector."""
        if selector not in self.blocked_selectors:
            self.blocked_selectors.append(selector)

    def add_safe_selector(self, selector: str) -> None:
        """Add a custom safe selector."""
        if selector not in self.safe_selectors:
            self.safe_selectors.append(selector)


class ElementSafetyAnalyzer:
    """Analyze elements for safety before clicking."""

    def __init__(self, config: SafetyConfig | None = None):
        self.config = config or SafetyConfig()

    async def analyze_risk(self, element: ElementHandle, page: Page) -> tuple[ActionRisk, str]:
        """
        Analyze an element and return its risk level with reason.

        Args:
            element: Playwright ElementHandle to analyze
            page: Playwright Page containing the element

        Returns:
            Tuple of (risk_level, reason_string)
        """
        # Get element properties via JavaScript evaluation
        props = await element.evaluate(
            """(el) => ({
            tagName: el.tagName.toLowerCase(),
            text: el.textContent?.trim() || '',
            ariaLabel: el.getAttribute('aria-label') || '',
            title: el.getAttribute('title') || '',
            className: el.className || '',
            id: el.id || '',
            type: el.type || '',
            formAction: el.formAction || '',
            href: el.href || '',
            dataTestId: el.getAttribute('data-testid') || '',
            role: el.getAttribute('role') || '',
            disabled: el.disabled || false,
            ariaDisabled: el.getAttribute('aria-disabled') === 'true',
        })"""
        )

        # Combine all text for analysis
        all_text = " ".join(
            [
                props["text"],
                props["ariaLabel"],
                props["title"],
                props["dataTestId"],
            ]
        ).lower()

        # Check blocked selectors first (highest priority)
        for selector in self.config.blocked_selectors:
            try:
                matches = await element.evaluate(f'(el) => el.matches("{selector}")', selector)
                if matches:
                    return ActionRisk.BLOCKED, f"Matches blocked selector: {selector}"
            except Exception:
                pass

        # Check for dangerous keywords
        for keyword in self.config.dangerous_keywords:
            if keyword.lower() in all_text:
                return ActionRisk.DANGEROUS, f"Contains dangerous keyword: {keyword}"

        # Check class names for danger indicators
        danger_classes = ["danger", "destructive", "delete", "remove", "warning", "error"]
        for cls in danger_classes:
            if cls in props["className"].lower():
                return ActionRisk.DANGEROUS, f"Has danger class: {cls}"

        # Check for dangerous ARIA roles
        if props["role"] in self.config.dangerous_roles:
            return ActionRisk.DANGEROUS, f"Has dangerous role: {props['role']}"

        # Check safe selectors
        for selector in self.config.safe_selectors:
            try:
                matches = await element.evaluate(f'(el) => el.matches("{selector}")', selector)
                if matches:
                    return ActionRisk.SAFE, f"Matches safe selector: {selector}"
            except Exception:
                pass

        # Check for safe keywords
        for keyword in self.config.safe_keywords:
            if keyword.lower() in all_text:
                return ActionRisk.SAFE, f"Contains safe keyword: {keyword}"

        # Check for form submissions (caution)
        if props["tagName"] == "button" and props["type"] == "submit":
            return ActionRisk.CAUTION, "Form submit button"

        # Check for external links (caution - might navigate away)
        if props["href"] and not props["href"].startswith("#"):
            if "javascript:" not in props["href"]:
                # Internal links are safer
                if props["href"].startswith("/") or props["href"].startswith("./"):
                    return ActionRisk.SAFE, "Internal navigation link"
                return ActionRisk.CAUTION, "External navigation link"

        # Navigation elements are generally safe
        if props["tagName"] == "a" and props["href"]:
            return ActionRisk.SAFE, "Standard link element"

        # Default to caution for unknown elements
        return ActionRisk.CAUTION, "Unknown action type"

    def should_click(self, risk: ActionRisk) -> bool:
        """Determine if element should be auto-clicked based on config."""
        if self.config.dry_run:
            return False

        risk_order = [
            ActionRisk.SAFE,
            ActionRisk.CAUTION,
            ActionRisk.DANGEROUS,
            ActionRisk.BLOCKED,
        ]
        max_idx = risk_order.index(self.config.max_auto_click_risk)
        current_idx = risk_order.index(risk)

        return current_idx <= max_idx


class ConfirmationDialogHandler:
    """Detect and handle confirmation dialogs safely."""

    CANCEL_SELECTORS = [
        'button:has-text("Cancel")',
        'button:has-text("No")',
        'button:has-text("Go Back")',
        'button:has-text("Never mind")',
        'button:has-text("Close")',
        "[data-testid='cancel']",
        ".cancel-btn",
        ".btn-cancel",
        '[aria-label="Close"]',
        '[aria-label="Cancel"]',
        # Avoid confirm/danger buttons
        '[role="dialog"] button:not([class*="confirm"]):not([class*="danger"]):not([class*="primary"])',
    ]

    def __init__(self) -> None:
        self._dialog_detected = False

    async def setup_dialog_handler(self, page: Page) -> None:
        """Set up automatic dialog dismissal."""
        # Handle native browser dialogs (alert, confirm, prompt)
        page.on("dialog", lambda dialog: dialog.dismiss())

        # Set up mutation observer for modal dialogs
        await page.evaluate(
            """() => {
            window.__qontinui_dialog_detected = false;

            const observer = new MutationObserver((mutations) => {
                for (const mutation of mutations) {
                    for (const node of mutation.addedNodes) {
                        if (node.nodeType === 1) {
                            const role = node.getAttribute?.('role');
                            if (role === 'dialog' || role === 'alertdialog') {
                                window.__qontinui_dialog_detected = true;
                            }
                            // Also check for common modal classes
                            const classList = node.classList;
                            if (classList && (
                                classList.contains('modal') ||
                                classList.contains('dialog') ||
                                classList.contains('overlay')
                            )) {
                                window.__qontinui_dialog_detected = true;
                            }
                        }
                    }
                }
            });

            observer.observe(document.body, {
                childList: true,
                subtree: true
            });
        }"""
        )

    async def check_for_dialog(self, page: Page) -> bool:
        """Check if a dialog has been detected."""
        try:
            result = await page.evaluate("() => window.__qontinui_dialog_detected || false")
            return bool(result)
        except Exception:
            return False

    async def dismiss_dialog(self, page: Page) -> bool:
        """
        Try to dismiss any detected dialog by clicking cancel.

        Returns:
            True if a dialog was dismissed, False otherwise
        """
        dialog_detected = await self.check_for_dialog(page)

        if not dialog_detected:
            return False

        # Try to find and click cancel button
        for selector in self.CANCEL_SELECTORS:
            try:
                cancel_btn = await page.query_selector(selector)
                if cancel_btn and await cancel_btn.is_visible():
                    await cancel_btn.click()
                    await page.evaluate("() => window.__qontinui_dialog_detected = false")
                    return True
            except Exception:
                continue

        # Fallback: press Escape
        try:
            await page.keyboard.press("Escape")
            await page.evaluate("() => window.__qontinui_dialog_detected = false")
            return True
        except Exception:
            pass

        return False

    async def reset_dialog_state(self, page: Page) -> None:
        """Reset the dialog detection state."""
        try:
            await page.evaluate("() => window.__qontinui_dialog_detected = false")
        except Exception:
            pass


@dataclass
class ElementRiskAssessment:
    """Result of risk assessment for an element."""

    element_id: str
    selector: str
    risk_level: ActionRisk
    risk_reason: str
    text_content: str | None = None
    aria_label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "selector": self.selector,
            "risk_level": self.risk_level.value,
            "risk_reason": self.risk_reason,
            "text_content": self.text_content,
            "aria_label": self.aria_label,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ElementRiskAssessment":
        return cls(
            element_id=data["element_id"],
            selector=data["selector"],
            risk_level=ActionRisk(data["risk_level"]),
            risk_reason=data["risk_reason"],
            text_content=data.get("text_content"),
            aria_label=data.get("aria_label"),
        )
