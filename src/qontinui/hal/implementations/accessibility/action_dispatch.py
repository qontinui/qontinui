"""Control-type-aware action dispatch registry for UIA/accessibility elements.

This module implements Phase 3 of PLAN-10: UIA Cascade Hardening.

Each UIA control type (button, combobox, treeitem, etc.) requires a different
interaction strategy.  Rather than using a single generic click/type path for
everything, this registry maps ``AccessibilityRole`` → ``ActionStrategy`` and
lets each strategy try the appropriate UIA pattern first (invoke, toggle,
expand_collapse, …), falling back to generic click/type when the pattern is
unavailable.

Usage::

    registry = ActionDispatchRegistry()           # reads QONTINUI_ACTION_SPEED_PROFILE
    result   = await registry.dispatch(node, "click", capture)
    if result.success:
        await asyncio.sleep(result.wait_after_s)
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from qontinui_schemas.accessibility import AccessibilityNode, AccessibilityRole

if TYPE_CHECKING:
    from qontinui.hal.interfaces.accessibility_capture import IAccessibilityCapture

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Post-action stabilization timings (seconds)
# ---------------------------------------------------------------------------

_POST_ACTION_WAIT_S: dict[AccessibilityRole, float] = {
    AccessibilityRole.BUTTON: 0.09,
    AccessibilityRole.MENUITEM: 0.1,
    AccessibilityRole.TREEITEM: 0.1,
    AccessibilityRole.TAB: 0.05,
    AccessibilityRole.COMBOBOX: 0.15,
    AccessibilityRole.CHECKBOX: 0.05,
    AccessibilityRole.SLIDER: 0.1,
    AccessibilityRole.TEXTBOX: 0.03,
}
_DEFAULT_WAIT_S: float = 0.05


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ActionResult:
    """Outcome of an action dispatch.

    Attributes:
        success:          Whether the action was performed without error.
        action_performed: Human-readable name of the action that ran
                          (e.g. "invoke_pattern", "click_center").
        wait_after_s:     Recommended pause after this action (seconds).
                          The registry multiplies this by the speed profile
                          multiplier before returning it to callers.
        message:          Optional diagnostic message (error text, notes).
    """

    success: bool
    action_performed: str
    wait_after_s: float = 0.0
    message: str = ""


# ---------------------------------------------------------------------------
# Abstract base strategy
# ---------------------------------------------------------------------------


class ActionStrategy(ABC):
    """Abstract base for per-role action strategies.

    Subclasses implement the pattern-based primary path and the generic
    fallback.  The registry calls :meth:`execute` and then scales
    ``wait_after_s`` by the speed-profile multiplier.
    """

    @abstractmethod
    async def execute(
        self,
        node: AccessibilityNode,
        action: str,
        capture: IAccessibilityCapture,
        **kwargs: object,
    ) -> ActionResult:
        """Execute the action on *node* via *capture*.

        Args:
            node:    The accessibility node to act on.
            action:  Logical action name ("click", "type", "toggle", …).
            capture: Live capture instance whose ``perform_action`` /
                     ``click_by_ref`` / ``type_by_ref`` are available.
            **kwargs: Action-specific keyword arguments
                      (e.g. ``text`` for type, ``value`` for sliders).

        Returns:
            :class:`ActionResult` describing what happened.
        """

    @abstractmethod
    def supported_actions(self) -> list[str]:
        """Return the list of logical action names this strategy handles."""


# ---------------------------------------------------------------------------
# Helper: generic click-center fallback
# ---------------------------------------------------------------------------


async def _click_center(
    node: AccessibilityNode,
    capture: IAccessibilityCapture,
) -> ActionResult:
    """Perform a generic click via :meth:`IAccessibilityCapture.click_by_ref`."""
    try:
        ok = await capture.click_by_ref(node.ref)
        return ActionResult(
            success=ok,
            action_performed="click_center",
            wait_after_s=_POST_ACTION_WAIT_S.get(node.role, _DEFAULT_WAIT_S),
            message="" if ok else f"click_by_ref returned False for {node.ref}",
        )
    except Exception as exc:
        logger.warning("click_center failed for %s: %s", node.ref, exc)
        return ActionResult(
            success=False,
            action_performed="click_center",
            wait_after_s=_DEFAULT_WAIT_S,
            message=str(exc),
        )


async def _perform_pattern_action(
    node: AccessibilityNode,
    pattern_action: str,
    capture: IAccessibilityCapture,
) -> ActionResult:
    """Try a UIA pattern action via *capture*.

    ``IAccessibilityCapture`` may not expose ``perform_action`` on all
    backends.  This helper calls it if available; otherwise it returns a
    failure result so the caller can try the fallback.
    """
    perform = getattr(capture, "perform_action", None)
    if perform is None:
        return ActionResult(
            success=False,
            action_performed=pattern_action,
            wait_after_s=0.0,
            message="capture does not support perform_action",
        )
    try:
        ok = await perform(node.ref, pattern_action)
        return ActionResult(
            success=bool(ok),
            action_performed=pattern_action,
            wait_after_s=_POST_ACTION_WAIT_S.get(node.role, _DEFAULT_WAIT_S),
            message="" if ok else f"{pattern_action} returned False for {node.ref}",
        )
    except Exception as exc:
        logger.debug("Pattern action %r failed for %s: %s", pattern_action, node.ref, exc)
        return ActionResult(
            success=False,
            action_performed=pattern_action,
            wait_after_s=0.0,
            message=str(exc),
        )


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------


class ButtonStrategy(ActionStrategy):
    """Strategy for BUTTON control type.

    Primary:  UIA Invoke pattern (``invoke``).
    Fallback: Click at element center.
    """

    def supported_actions(self) -> list[str]:
        return ["click", "invoke", "press"]

    async def execute(
        self,
        node: AccessibilityNode,
        action: str,
        capture: IAccessibilityCapture,
        **kwargs: object,
    ) -> ActionResult:
        # Try Invoke pattern first — the preferred UIA approach for buttons.
        result = await _perform_pattern_action(node, "invoke", capture)
        if result.success:
            return result

        # Fallback: generic click at center coordinates.
        logger.debug("ButtonStrategy: invoke failed, falling back to click_center (%s)", node.ref)
        return await _click_center(node, capture)


class ComboBoxStrategy(ActionStrategy):
    """Strategy for COMBOBOX control type.

    Primary:  UIA ExpandCollapse pattern (``expand_collapse``) to open the
              list, then optionally select an item by value.
    Fallback: Click to open → send arrow keys to navigate.
    """

    def supported_actions(self) -> list[str]:
        return ["click", "expand", "collapse", "select", "open"]

    async def execute(
        self,
        node: AccessibilityNode,
        action: str,
        capture: IAccessibilityCapture,
        **kwargs: object,
    ) -> ActionResult:
        # Try ExpandCollapse pattern first.
        result = await _perform_pattern_action(node, "expand_collapse", capture)
        if result.success:
            # If a target item value was supplied, try to select it.
            item_value = kwargs.get("value") or kwargs.get("item")
            if item_value:
                sel_result = await _perform_pattern_action(
                    node, f"select_item:{item_value}", capture
                )
                if sel_result.success:
                    return ActionResult(
                        success=True,
                        action_performed="expand_collapse+select_item",
                        wait_after_s=_POST_ACTION_WAIT_S.get(node.role, _DEFAULT_WAIT_S),
                    )
            return result

        # Fallback 1: click to open.
        logger.debug(
            "ComboBoxStrategy: expand_collapse failed, falling back to click (%s)", node.ref
        )
        click_result = await _click_center(node, capture)
        if not click_result.success:
            return click_result

        # Fallback 2: arrow-key navigation when an item value is requested.
        item_value = kwargs.get("value") or kwargs.get("item")
        if item_value:
            send_keys = getattr(capture, "send_keys", None)
            if send_keys is not None:
                try:
                    await send_keys("{Down}")
                except Exception as exc:
                    logger.debug("ComboBoxStrategy: send_keys Down failed: %s", exc)

        return ActionResult(
            success=click_result.success,
            action_performed="click_open",
            wait_after_s=_POST_ACTION_WAIT_S.get(node.role, _DEFAULT_WAIT_S),
            message=click_result.message,
        )


class TreeItemStrategy(ActionStrategy):
    """Strategy for TREEITEM control type.

    Primary:  UIA SelectionItem pattern (``select``) then scroll into view.
    Fallback: Click.
    """

    def supported_actions(self) -> list[str]:
        return ["click", "select", "expand", "collapse"]

    async def execute(
        self,
        node: AccessibilityNode,
        action: str,
        capture: IAccessibilityCapture,
        **kwargs: object,
    ) -> ActionResult:
        # Try SelectionItem Select pattern.
        result = await _perform_pattern_action(node, "select", capture)
        if result.success:
            # Also attempt to scroll the item into view.
            await _perform_pattern_action(node, "ensure_visible", capture)
            return result

        # Fallback: generic click.
        logger.debug("TreeItemStrategy: select failed, falling back to click_center (%s)", node.ref)
        return await _click_center(node, capture)


class MenuItemStrategy(ActionStrategy):
    """Strategy for MENUITEM control type.

    Primary:  UIA Invoke pattern (``invoke``).
    Fallback: Click.
    """

    def supported_actions(self) -> list[str]:
        return ["click", "invoke", "select"]

    async def execute(
        self,
        node: AccessibilityNode,
        action: str,
        capture: IAccessibilityCapture,
        **kwargs: object,
    ) -> ActionResult:
        result = await _perform_pattern_action(node, "invoke", capture)
        if result.success:
            return result

        logger.debug("MenuItemStrategy: invoke failed, falling back to click_center (%s)", node.ref)
        return await _click_center(node, capture)


class CheckboxStrategy(ActionStrategy):
    """Strategy for CHECKBOX control type.

    Primary:  UIA Toggle pattern (``toggle``).
    Fallback: Click.
    """

    def supported_actions(self) -> list[str]:
        return ["click", "toggle", "check", "uncheck"]

    async def execute(
        self,
        node: AccessibilityNode,
        action: str,
        capture: IAccessibilityCapture,
        **kwargs: object,
    ) -> ActionResult:
        result = await _perform_pattern_action(node, "toggle", capture)
        if result.success:
            return result

        logger.debug("CheckboxStrategy: toggle failed, falling back to click_center (%s)", node.ref)
        return await _click_center(node, capture)


class SliderStrategy(ActionStrategy):
    """Strategy for SLIDER control type.

    Primary:  UIA RangeValue pattern (``set_range_value:<value>``).
    Fallback: Click at the proportional position along the slider track.
    """

    def supported_actions(self) -> list[str]:
        return ["click", "set", "set_value", "drag"]

    async def execute(
        self,
        node: AccessibilityNode,
        action: str,
        capture: IAccessibilityCapture,
        **kwargs: object,
    ) -> ActionResult:
        target_value = kwargs.get("value")
        if target_value is not None:
            result = await _perform_pattern_action(node, f"set_range_value:{target_value}", capture)
            if result.success:
                return result

        # Fallback: click at the center (or proportional position if bounds available).
        logger.debug("SliderStrategy: set_range_value failed, falling back to click (%s)", node.ref)
        return await _click_center(node, capture)


class TextBoxStrategy(ActionStrategy):
    """Strategy for TEXTBOX / EDIT control type.

    Primary:  UIA Value pattern (``set_value:<text>``).
    Fallback: Click to focus then type via :meth:`IAccessibilityCapture.type_by_ref`.
    """

    def supported_actions(self) -> list[str]:
        return ["click", "type", "set", "clear", "focus"]

    async def execute(
        self,
        node: AccessibilityNode,
        action: str,
        capture: IAccessibilityCapture,
        **kwargs: object,
    ) -> ActionResult:
        text = kwargs.get("text", "")
        clear_first = bool(kwargs.get("clear_first", False))

        if text:
            # Try Value.SetValue pattern first.
            result = await _perform_pattern_action(node, f"set_value:{text}", capture)
            if result.success:
                return result

        # Fallback: click to focus then type via the capture interface.
        logger.debug("TextBoxStrategy: set_value failed, falling back to click+type (%s)", node.ref)
        click_result = await _click_center(node, capture)
        if not click_result.success:
            return click_result

        if text:
            try:
                ok = await capture.type_by_ref(node.ref, str(text), clear_first=clear_first)
                return ActionResult(
                    success=ok,
                    action_performed="click_then_type",
                    wait_after_s=_POST_ACTION_WAIT_S.get(node.role, _DEFAULT_WAIT_S),
                    message="" if ok else f"type_by_ref returned False for {node.ref}",
                )
            except Exception as exc:
                logger.warning("TextBoxStrategy type_by_ref failed for %s: %s", node.ref, exc)
                return ActionResult(
                    success=False,
                    action_performed="click_then_type",
                    wait_after_s=_DEFAULT_WAIT_S,
                    message=str(exc),
                )

        # Action was just a focus/click with no text.
        return ActionResult(
            success=click_result.success,
            action_performed="click_focus",
            wait_after_s=_POST_ACTION_WAIT_S.get(node.role, _DEFAULT_WAIT_S),
            message=click_result.message,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ActionDispatchRegistry:
    """Central registry that maps :class:`AccessibilityRole` → :class:`ActionStrategy`.

    The registry selects the best strategy for a node's role, delegates
    execution, and then scales the resulting ``wait_after_s`` by the
    speed-profile multiplier.

    Speed profile is read from the ``QONTINUI_ACTION_SPEED_PROFILE``
    environment variable (``"fast"`` / ``"default"`` / ``"slow"``).
    It may also be overridden at construction time via *speed_profile*.

    Example::

        registry = ActionDispatchRegistry()          # uses env var
        result   = await registry.dispatch(node, "click", capture)
    """

    _SPEED_MULTIPLIERS: dict[str, float] = {
        "fast": 0.5,
        "default": 1.0,
        "slow": 3.0,
    }

    def __init__(self, speed_profile: str | None = None) -> None:
        """Initialise the registry.

        Args:
            speed_profile: One of ``"fast"``, ``"default"``, ``"slow"``.
                           If *None*, the value is read from the
                           ``QONTINUI_ACTION_SPEED_PROFILE`` environment
                           variable, defaulting to ``"default"``.
        """
        resolved = speed_profile or os.environ.get("QONTINUI_ACTION_SPEED_PROFILE", "default")
        self._speed_multiplier: float = self._SPEED_MULTIPLIERS.get(resolved, 1.0)
        self._strategies: dict[AccessibilityRole, ActionStrategy] = {}
        self._register_defaults()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def _register_defaults(self) -> None:
        """Register the built-in per-role strategies."""
        self.register(AccessibilityRole.BUTTON, ButtonStrategy())
        self.register(AccessibilityRole.COMBOBOX, ComboBoxStrategy())
        self.register(AccessibilityRole.TREEITEM, TreeItemStrategy())
        self.register(AccessibilityRole.MENUITEM, MenuItemStrategy())
        self.register(AccessibilityRole.MENUITEMCHECKBOX, MenuItemStrategy())
        self.register(AccessibilityRole.MENUITEMRADIO, MenuItemStrategy())
        self.register(AccessibilityRole.CHECKBOX, CheckboxStrategy())
        self.register(AccessibilityRole.SLIDER, SliderStrategy())
        self.register(AccessibilityRole.TEXTBOX, TextBoxStrategy())
        self.register(AccessibilityRole.EDIT, TextBoxStrategy())
        self.register(AccessibilityRole.SEARCHBOX, TextBoxStrategy())

    def register(self, role: AccessibilityRole, strategy: ActionStrategy) -> None:
        """Register (or replace) a strategy for *role*.

        Args:
            role:     The accessibility role to handle.
            strategy: The strategy instance to use.
        """
        self._strategies[role] = strategy

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_strategy(self, role: AccessibilityRole) -> ActionStrategy | None:
        """Return the registered strategy for *role*, or *None*.

        Args:
            role: Accessibility role to look up.

        Returns:
            The registered :class:`ActionStrategy`, or ``None`` if no
            strategy is registered for this role.
        """
        return self._strategies.get(role)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def dispatch(
        self,
        node: AccessibilityNode,
        action: str,
        capture: IAccessibilityCapture,
        **kwargs: object,
    ) -> ActionResult:
        """Dispatch *action* on *node* using the best available strategy.

        If no strategy is registered for ``node.role``, falls back to a
        generic click (for "click"-like actions) or a no-op failure result.

        The ``wait_after_s`` field of the returned result is already
        scaled by the speed-profile multiplier.

        Args:
            node:    Target accessibility node.
            action:  Logical action name ("click", "type", "toggle", …).
            capture: Live capture backend for performing actions.
            **kwargs: Additional action parameters forwarded to the strategy
                      (e.g. ``text="hello"`` for TextBoxStrategy).

        Returns:
            :class:`ActionResult` with ``wait_after_s`` already adjusted
            for the configured speed profile.
        """
        strategy = self.get_strategy(node.role)

        if strategy is not None:
            result = await strategy.execute(node, action, capture, **kwargs)
        else:
            result = await self._default_action(node, action, capture, **kwargs)

        # Apply speed-profile scaling to the stabilisation wait.
        result.wait_after_s *= self._speed_multiplier
        return result

    async def _default_action(
        self,
        node: AccessibilityNode,
        action: str,
        capture: IAccessibilityCapture,
        **kwargs: object,
    ) -> ActionResult:
        """Generic fallback when no strategy is registered for a role.

        For "type" actions attempts ``type_by_ref``; for everything else
        falls back to ``click_by_ref``.
        """
        if action == "type":
            text = str(kwargs.get("text", ""))
            clear_first = bool(kwargs.get("clear_first", False))
            try:
                ok = await capture.type_by_ref(node.ref, text, clear_first=clear_first)
                return ActionResult(
                    success=ok,
                    action_performed="default_type",
                    wait_after_s=_POST_ACTION_WAIT_S.get(node.role, _DEFAULT_WAIT_S),
                    message="" if ok else f"type_by_ref returned False for {node.ref}",
                )
            except Exception as exc:
                return ActionResult(
                    success=False,
                    action_performed="default_type",
                    wait_after_s=_DEFAULT_WAIT_S,
                    message=str(exc),
                )

        return await _click_center(node, capture)


__all__ = [
    "ActionResult",
    "ActionStrategy",
    "ActionDispatchRegistry",
    "ButtonStrategy",
    "ComboBoxStrategy",
    "TreeItemStrategy",
    "MenuItemStrategy",
    "CheckboxStrategy",
    "SliderStrategy",
    "TextBoxStrategy",
    "_POST_ACTION_WAIT_S",
    "_DEFAULT_WAIT_S",
]
