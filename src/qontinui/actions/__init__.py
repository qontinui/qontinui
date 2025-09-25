"""Actions package - ported from Qontinui framework.

Core action system including ObjectCollection.
"""

from .action import Action
from .action_config import ActionConfig, ActionConfigBuilder, Illustrate, LoggingOptions
from .action_interface import ActionInterface
from .action_options import FindOptions, KeyModifier
from .action_result import ActionResult
from .action_type import ActionType

# Import unified Actions class
from .actions import Actions
from .basic.click import ClickOptions, ClickOptionsBuilder

# Import basic action options
from .basic.find import (
    DoOnEach,
    FindStrategy,
    PatternFindOptions,
    PatternFindOptionsBuilder,
    Strategy,
)
from .basic.mouse import (
    Direction,
    MouseButton,
    MouseMoveOptions,
    MousePressOptions,
    MousePressOptionsBuilder,
    ScrollOptions,
    ScrollOptionsBuilder,
)
from .basic.type import TypeOptions, TypeOptionsBuilder
from .basic.type.key_down_options import KeyDownOptions
from .basic.type.key_up_options import KeyUpOptions

# Import wait options
from .basic.wait.wait import WaitOptions

# Import composite action options
from .composite import DragOptions, DragOptionsBuilder
from .composite.chains.action_chain import ActionChain
from .fluent import FluentActions
from .keys import Key, KeyCombo, KeyCombos
from .object_collection import ObjectCollection, ObjectCollectionBuilder

# Import pure and fluent actions
from .pure import PureActions
from .repetition_options import RepetitionOptions, RepetitionOptionsBuilder
from .verification_options import Event, VerificationOptions, VerificationOptionsBuilder

# Create aliases
MoveOptions = MouseMoveOptions

__all__ = [
    # Core action classes
    "ObjectCollection",
    "ObjectCollectionBuilder",
    "ActionResult",
    "ActionType",
    "ActionConfig",
    "ActionConfigBuilder",
    "Illustrate",
    "LoggingOptions",
    "ActionInterface",
    "Action",
    "VerificationOptions",
    "VerificationOptionsBuilder",
    "Event",
    "RepetitionOptions",
    "RepetitionOptionsBuilder",
    # Find options
    "FindOptions",
    "PatternFindOptions",
    "PatternFindOptionsBuilder",
    "FindStrategy",
    "Strategy",
    "DoOnEach",
    # Click options
    "ClickOptions",
    "ClickOptionsBuilder",
    # Type options
    "TypeOptions",
    "TypeOptionsBuilder",
    "KeyDownOptions",
    "KeyUpOptions",
    "KeyModifier",
    "Key",
    "KeyCombo",
    "KeyCombos",
    # Mouse options
    "MousePressOptions",
    "MousePressOptionsBuilder",
    "MouseButton",
    "MouseMoveOptions",
    "ScrollOptions",
    "ScrollOptionsBuilder",
    "Direction",
    # Composite options
    "DragOptions",
    "DragOptionsBuilder",
    "ActionChain",
    # Pure and fluent actions
    "PureActions",
    "FluentActions",
    "Actions",
    # Wait options
    "WaitOptions",
    # Aliases for convenience
    "MoveOptions",  # Alias for MouseMoveOptions
]
