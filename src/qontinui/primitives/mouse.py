"""Mouse primitives - ported from Qontinui framework.

Each primitive mouse action does exactly one thing and extends Action
to get lifecycle management capabilities.
"""

from ..actions import (
    Action,
    ActionConfig,
    ActionResult,
    ClickOptions,
    DragOptions,
    MoveOptions,
    ScrollOptions,
)
from ..actions.pure import PureActions


class MouseMove(Action):
    """Primitive mouse move action.

    Port of MouseMove from Qontinui framework primitive.
    Moves the mouse to a specific position.
    """

    def __init__(self, config: MoveOptions | None = None) -> None:
        """Initialize with optional MoveOptions.

        Args:
            config: MoveOptions instance or None for defaults
        """
        super().__init__()
        self._config = config or MoveOptions()
        self._pure = PureActions()

    def execute_at(self, x: int, y: int, duration: float | None = None) -> ActionResult:
        """Execute mouse move to specific coordinates.

        Args:
            x: Target X coordinate
            y: Target Y coordinate
            duration: Optional duration override

        Returns:
            ActionResult
        """
        move_duration = (
            duration if duration is not None else getattr(self._config, "_move_duration", 0.5)
        )

        return self.execute(lambda: self._pure.mouse_move(x, y, move_duration), target=(x, y))


class MouseClick(Action):
    """Primitive mouse click action.

    Port of MouseClick from Qontinui framework primitive.
    Performs a single click at a position.
    """

    def __init__(self, config: ClickOptions | None = None) -> None:
        """Initialize with optional ClickOptions.

        Args:
            config: ClickOptions instance or None for defaults
        """
        super().__init__()
        if config is None:
            from ..actions.basic.click import ClickOptionsBuilder

            config = ClickOptionsBuilder().build()
        self._config = config
        self._pure = PureActions()

    def execute_at(self, x: int, y: int, button: str = "left") -> ActionResult:
        """Execute click at specific coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
            button: Mouse button ('left', 'right', 'middle')

        Returns:
            ActionResult
        """
        # Handle multiple clicks if configured
        click_count = getattr(self._config, "_click_count", 1)
        pause_between = getattr(self._config, "_pause_between_clicks", 0.05)

        def click_action():
            results = []
            for i in range(click_count):
                if i > 0:
                    self._pure.wait(pause_between)
                result = self._pure.mouse_click(x, y, button)
                results.append(result)
                if not result.success:
                    return result
            return result  # Return the last successful result from PureActions

        return self.execute(click_action, target=(x, y))


class MouseDown(Action):
    """Primitive mouse down action.

    Port of MouseDown from Qontinui framework primitive.
    Presses and holds a mouse button.
    """

    def __init__(self, config: ActionConfig | None = None) -> None:
        """Initialize with optional ActionConfig.

        Args:
            config: ActionConfig instance or None for defaults
        """
        super().__init__()
        self._config = config or ActionConfig()
        self._pure = PureActions()

    def execute_at(
        self, x: int | None = None, y: int | None = None, button: str = "left"
    ) -> ActionResult:
        """Execute mouse down at optional coordinates.

        Args:
            x: Optional X coordinate
            y: Optional Y coordinate
            button: Mouse button

        Returns:
            ActionResult
        """
        return self.execute(
            lambda: self._pure.mouse_down(x, y, button), target=(x, y) if x and y else None
        )


class MouseUp(Action):
    """Primitive mouse up action.

    Port of MouseUp from Qontinui framework primitive.
    Releases a mouse button.
    """

    def __init__(self, config: ActionConfig | None = None) -> None:
        """Initialize with optional ActionConfig.

        Args:
            config: ActionConfig instance or None for defaults
        """
        super().__init__()
        self._config = config or ActionConfig()
        self._pure = PureActions()

    def execute_at(
        self, x: int | None = None, y: int | None = None, button: str = "left"
    ) -> ActionResult:
        """Execute mouse up at optional coordinates.

        Args:
            x: Optional X coordinate
            y: Optional Y coordinate
            button: Mouse button

        Returns:
            ActionResult
        """
        return self.execute(
            lambda: self._pure.mouse_up(x, y, button), target=(x, y) if x and y else None
        )


class MouseDrag(Action):
    """Primitive mouse drag action.

    Port of MouseDrag from Qontinui framework primitive.
    Drags from one position to another.
    """

    def __init__(self, config: DragOptions | None = None) -> None:
        """Initialize with optional DragOptions.

        Args:
            config: DragOptions instance or None for defaults
        """
        super().__init__()
        if config is None:
            from ..actions.composite.drag import DragOptionsBuilder

            config = DragOptionsBuilder().build()
        self._config = config
        self._pure = PureActions()

    def execute_from_to(self, start_x: int, start_y: int, end_x: int, end_y: int) -> ActionResult:
        """Execute drag from start to end position.

        Args:
            start_x: Start X coordinate
            start_y: Start Y coordinate
            end_x: End X coordinate
            end_y: End Y coordinate

        Returns:
            ActionResult
        """
        drag_duration = getattr(self._config, "_drag_duration", 1.0)
        button = getattr(self._config, "_button", "left")
        pause_before = getattr(self._config, "_pause_before_drag", 0.1)
        pause_after = getattr(self._config, "_pause_after_drag", 0.1)

        def drag_action():
            # Move to start position
            move_result = self._pure.mouse_move(start_x, start_y, 0.2)
            if not move_result.success:
                return move_result

            # Pause before drag
            if pause_before > 0:
                self._pure.wait(pause_before)

            # Mouse down
            down_result = self._pure.mouse_down(button=button)
            if not down_result.success:
                return down_result

            # Drag to end position
            drag_result = self._pure.mouse_move(end_x, end_y, drag_duration)
            if not drag_result.success:
                self._pure.mouse_up(button=button)  # Ensure mouse is released
                return drag_result

            # Pause after drag (before release)
            if pause_after > 0:
                self._pure.wait(pause_after)

            # Mouse up
            up_result = self._pure.mouse_up(button=button)
            return up_result  # Return the PureActions ActionResult directly

        return self.execute(drag_action, target=((start_x, start_y), (end_x, end_y)))


class MouseWheel(Action):
    """Primitive mouse wheel/scroll action.

    Port of MouseWheel from Qontinui framework primitive.
    Scrolls the mouse wheel.
    """

    def __init__(self, config: ScrollOptions | None = None) -> None:
        """Initialize with optional ScrollOptions.

        Args:
            config: ScrollOptions instance or None for defaults
        """
        super().__init__()
        if config is None:
            from ..actions.basic.mouse.scroll_options import ScrollOptionsBuilder

            config = ScrollOptionsBuilder().build()
        self._config = config
        self._pure = PureActions()

    def execute_scroll(
        self, clicks: int | None = None, x: int | None = None, y: int | None = None
    ) -> ActionResult:
        """Execute scroll action.

        Args:
            clicks: Number of scroll clicks (None uses config)
            x: Optional X coordinate
            y: Optional Y coordinate

        Returns:
            ActionResult
        """
        scroll_amount = clicks if clicks is not None else getattr(self._config, "_scroll_amount", 3)
        direction = getattr(self._config, "_scroll_direction", "down")

        # Convert direction to scroll clicks
        if direction in ["down", "right"]:
            scroll_amount = -abs(scroll_amount)
        else:
            scroll_amount = abs(scroll_amount)

        return self.execute(
            lambda: self._pure.mouse_scroll(scroll_amount, x, y), target=(x, y, scroll_amount)
        )
