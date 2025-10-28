"""Fluent API for action chaining following Brobot principles."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

from .pure import ActionResult, PureActions


@dataclass
class ActionChain:
    """Represents a chain of actions to be executed."""

    actions: list[tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]] = field(
        default_factory=list
    )
    results: list[ActionResult] = field(default_factory=list)
    stop_on_failure: bool = True

    def add(self, action: Callable[..., Any], *args, **kwargs) -> "ActionChain":
        """Add an action to the chain.

        Args:
            action: Action method to execute
            *args: Positional arguments for the action
            **kwargs: Keyword arguments for the action

        Returns:
            Self for chaining
        """
        self.actions.append((action, args, kwargs))
        return self

    def execute(self) -> list[ActionResult]:
        """Execute all actions in the chain.

        Returns:
            List of ActionResult objects
        """

        self.results = []

        for action, args, kwargs in self.actions:
            result = action(*args, **kwargs)
            self.results.append(result)

            if not result.success and self.stop_on_failure:
                break

        return cast(list[ActionResult], self.results)

    def clear(self) -> "ActionChain":
        """Clear the action chain.

        Returns:
            Self for chaining
        """
        self.actions = []
        self.results = []
        return self

    @property
    def success(self) -> bool:
        """Check if all actions succeeded.

        Returns:
            True if all actions succeeded
        """

        return cast(bool, all(r.success for r in self.results))

    @property
    def last_result(self) -> ActionResult | None:
        """Get the last action result.

        Returns:
            Last ActionResult or None
        """
        return self.results[-1] if self.results else None


class FluentActions:
    """Fluent API for chaining pure actions.

    Following Brobot principles:
    - Composite actions are built from pure actions
    - Fluent interface for readability
    - Clear separation between atomic and composite actions
    """

    def __init__(self) -> None:
        """Initialize fluent actions."""
        self.pure = PureActions()
        self.chain = ActionChain()
        self._current_position: tuple[int, int] | None = None

    # Fluent API Methods

    def at(self, x: int, y: int) -> "FluentActions":
        """Set position for next action.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Self for chaining
        """
        self._current_position = (x, y)
        return self

    def move_to(self, x: int, y: int, duration: float = 0.0) -> "FluentActions":
        """Add mouse move to chain.

        Args:
            x: Target X coordinate
            y: Target Y coordinate
            duration: Duration of movement

        Returns:
            Self for chaining
        """
        self.chain.add(self.pure.mouse_move, x, y, duration)
        self._current_position = (x, y)
        return self

    def click(self, button: str = "left") -> "FluentActions":
        """Add click at current position to chain.

        Args:
            button: Mouse button to click

        Returns:
            Self for chaining
        """
        if self._current_position:
            x, y = self._current_position
            self.chain.add(self.pure.mouse_click, x, y, button)
        return self

    def double_click(self, button: str = "left") -> "FluentActions":
        """Add double click to chain.

        Args:
            button: Mouse button to click

        Returns:
            Self for chaining
        """
        if self._current_position:
            x, y = self._current_position
            self.chain.add(self.pure.mouse_click, x, y, button)
            self.chain.add(self.pure.wait, 0.05)  # Small delay between clicks
            self.chain.add(self.pure.mouse_click, x, y, button)
        return self

    def right_click(self) -> "FluentActions":
        """Add right click to chain.

        Returns:
            Self for chaining
        """
        return self.click("right")

    def mouse_down(self, button: str = "left") -> "FluentActions":
        """Add mouse down to chain.

        Args:
            button: Mouse button

        Returns:
            Self for chaining
        """
        if self._current_position:
            x, y = self._current_position
            self.chain.add(self.pure.mouse_down, x, y, button)
        else:
            self.chain.add(self.pure.mouse_down, None, None, button)
        return self

    def mouse_up(self, button: str = "left") -> "FluentActions":
        """Add mouse up to chain.

        Args:
            button: Mouse button

        Returns:
            Self for chaining
        """
        if self._current_position:
            x, y = self._current_position
            self.chain.add(self.pure.mouse_up, x, y, button)
        else:
            self.chain.add(self.pure.mouse_up, None, None, button)
        return self

    def drag_to(
        self, x: int, y: int, duration: float = 1.0, button: str = "left"
    ) -> "FluentActions":
        """Add drag operation to chain (mouseDown + move + mouseUp).

        This is a composite action built from pure actions.

        Args:
            x: Target X coordinate
            y: Target Y coordinate
            duration: Duration of drag
            button: Mouse button

        Returns:
            Self for chaining
        """
        # Composite action: mouseDown -> move -> mouseUp
        self.mouse_down(button)
        self.move_to(x, y, duration)
        self.mouse_up(button)
        return self

    def scroll(self, clicks: int) -> "FluentActions":
        """Add scroll to chain.

        Args:
            clicks: Number of scroll clicks

        Returns:
            Self for chaining
        """
        if self._current_position:
            x, y = self._current_position
            self.chain.add(self.pure.mouse_scroll, clicks, x, y)
        else:
            self.chain.add(self.pure.mouse_scroll, clicks)
        return self

    def type_text(self, text: str, interval: float = 0.0) -> "FluentActions":
        """Add text typing to chain.

        Args:
            text: Text to type
            interval: Delay between characters

        Returns:
            Self for chaining
        """
        for char in text:
            self.chain.add(self.pure.type_character, char)
            if interval > 0:
                self.chain.add(self.pure.wait, interval)
        return self

    def key(self, key: str) -> "FluentActions":
        """Add key press to chain.

        Args:
            key: Key to press

        Returns:
            Self for chaining
        """
        self.chain.add(self.pure.key_press, key)
        return self

    def hotkey(self, *keys: str) -> "FluentActions":
        """Add hotkey combination to chain.

        Args:
            *keys: Keys to press together

        Returns:
            Self for chaining
        """
        # Press all keys down
        for key in keys:
            self.chain.add(self.pure.key_down, key)

        # Release all keys in reverse order
        for key in reversed(keys):
            self.chain.add(self.pure.key_up, key)

        return self

    def wait(self, seconds: float) -> "FluentActions":
        """Add wait to chain.

        Args:
            seconds: Seconds to wait

        Returns:
            Self for chaining
        """
        self.chain.add(self.pure.wait, seconds)
        return self

    def pause(self, milliseconds: int) -> "FluentActions":
        """Add pause to chain.

        Args:
            milliseconds: Milliseconds to pause

        Returns:
            Self for chaining
        """
        self.chain.add(self.pure.pause, milliseconds)
        return self

    def screenshot(self, region: tuple[int, int, int, int] | None = None) -> "FluentActions":
        """Add screenshot to chain.

        Args:
            region: Optional region to capture

        Returns:
            Self for chaining
        """
        self.chain.add(self.pure.capture_screen, region)
        return self

    def execute(self) -> list[ActionResult]:
        """Execute the action chain.

        Returns:
            List of ActionResult objects
        """

        return cast(list[ActionResult], self.chain.execute())

    def clear(self) -> "FluentActions":
        """Clear the action chain.

        Returns:
            Self for chaining
        """
        self.chain.clear()
        self._current_position = None
        return self

    @property
    def success(self) -> bool:
        """Check if all actions succeeded.

        Returns:
            True if all actions succeeded
        """

        return cast(bool, self.chain.success)

    @property
    def results(self) -> list[ActionResult]:
        """Get all action results.

        Returns:
            List of ActionResult objects
        """
        return cast(list[ActionResult], self.chain.results)


# Example usage following Brobot patterns:
#
# actions = FluentActions()
#
# # Simple click
# actions.at(100, 200).click().execute()
#
# # Drag operation (composite)
# actions.at(100, 200).drag_to(300, 400).execute()
#
# # Complex chain
# results = (actions
#     .at(100, 200)
#     .click()
#     .wait(0.5)
#     .type_text("Hello")
#     .key("enter")
#     .execute())
#
# if actions.success:
#     print("All actions succeeded")
# else:
#     print("Some actions failed")
