"""Type text action - ported from Qontinui framework.

Types text to the window in focus.
"""

from typing import Optional

from ....model.state.state_string import StateString
from ...action_interface import ActionInterface
from ...action_result import ActionResult
from ...action_type import ActionType
from ...object_collection import ObjectCollection
from .type_options import TypeOptions

# Default type delay
DEFAULT_TYPE_DELAY = 0.02


class TypeText(ActionInterface):
    """Types text to the window in focus in the Qontinui model-based GUI automation framework.

    Port of TypeText from Qontinui framework class.

    TypeText is a fundamental action in the Action Model (α) that enables keyboard input
    simulation. It bridges the gap between high-level text input requirements and low-level
    keyboard event generation, providing a reliable way to enter text into GUI applications
    regardless of their underlying technology.

    Key features:
    - Focus-based Input: Types to whatever window or field currently has focus,
      following standard GUI interaction patterns
    - Configurable Timing: Supports custom delays between keystrokes to accommodate
      different application response times
    - Batch Processing: Can type multiple strings in sequence with configurable
      pauses between them
    - State-aware: Works with StateString objects that maintain context about
      their owning states

    Common use cases:
    - Filling form fields during automated testing
    - Entering search queries or commands
    - Providing credentials during login sequences
    - Interacting with text-based interfaces or terminals

    The type delay mechanism is particularly important for handling applications that
    process keystrokes asynchronously or have input validation that runs between keystrokes.
    By adjusting the type delay through ActionOptions, automation scripts can adapt to
    different application behaviors without modifying the core logic.

    This action exemplifies the framework's approach to abstracting platform-specific
    details while providing fine-grained control when needed. The underlying implementation
    delegates to platform-specific wrappers while maintaining a consistent interface.
    """

    def __init__(
        self,
        text_typer: Optional["TextTyper"] = None,
        time: Optional["TimeProvider"] = None,
    ) -> None:
        """Initialize TypeText action.

        Args:
            text_typer: Component that performs the actual typing
            time: Time provider for delays
        """
        self.text_typer = text_typer
        self.time = time
        self._saved_type_delay = DEFAULT_TYPE_DELAY

    def get_action_type(self) -> ActionType:
        """Return the action type.

        Returns:
            ActionType.TYPE
        """
        return ActionType.TYPE

    async def perform(self, matches: ActionResult, *object_collections: ObjectCollection) -> None:
        """Type text strings from the object collections.

        Args:
            matches: The ActionResult containing configuration
            object_collections: Collections containing StateString objects to type

        Raises:
            ValueError: If matches does not contain TypeOptions configuration
        """
        # Get the configuration - expecting TypeOptions
        action_config = matches.action_config
        if not isinstance(action_config, TypeOptions):
            raise ValueError("TypeText requires TypeOptions configuration")

        type_options = action_config

        # Save and set the type delay
        self._saved_type_delay = DEFAULT_TYPE_DELAY
        current_type_delay = type_options.get_type_delay()

        if object_collections and len(object_collections) > 0:
            strings = object_collections[0].state_strings
            for i, state_string in enumerate(strings):
                if self.text_typer:
                    self.text_typer.type(state_string, type_options)
                else:
                    # Placeholder implementation
                    print(f"Typing: {state_string.string} with delay {current_type_delay}")

                # Pause between typing different strings (except after the last one)
                if i < len(strings) - 1:
                    if self.time:
                        self.time.wait(type_options.get_pause_after_end())

        # Note: In Python we don't need to restore Settings.TypeDelay as we don't have global settings


class TextTyper:
    """Placeholder for TextTyper class.

    Performs the actual text typing.
    """

    def type(self, state_string: StateString, type_options: TypeOptions) -> None:
        """Type a state string.

        Args:
            state_string: The string to type
            type_options: Configuration for typing
        """
        text = state_string.string
        modifiers = type_options.get_modifiers()
        delay = type_options.get_type_delay()

        print(f"Typing '{text}' with modifiers '{modifiers}' and delay {delay}s")


class TimeProvider:
    """Placeholder for TimeProvider class.

    Provides time and delay functionality.
    """

    def wait(self, seconds: float) -> None:
        """Wait for specified duration.

        Args:
            seconds: Duration to wait
        """
        import time

        time.sleep(seconds)
