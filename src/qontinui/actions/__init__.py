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
    MouseMoveOptionsBuilder,
    MousePressOptions,
    MousePressOptionsBuilder,
    ScrollOptions,
    ScrollOptionsBuilder,
)
from .basic.type import TypeOptions, TypeOptionsBuilder
from .basic.type.key_down_options import KeyDownOptions, KeyDownOptionsBuilder
from .basic.type.key_up_options import KeyUpOptions, KeyUpOptionsBuilder

# Import wait options
from .basic.wait.wait import WaitOptions

# Import composite action options
from .composite import DragOptions, DragOptionsBuilder
from .composite.chains.action_chain import ActionChain

# Import control flow
from .control_flow import BreakLoop, ContinueLoop, ControlFlowExecutor
from .fluent import FluentActions
from .keys import Key, KeyCombo, KeyCombos
from .object_collection import ObjectCollection, ObjectCollectionBuilder

# Import pure and fluent actions
from .pure import PureActions
from .repetition_options import RepetitionOptions, RepetitionOptionsBuilder
from .result_builder import ActionResultBuilder
from .result_extractors import ResultExtractor
from .result_mergers import ResultMerger
from .verification_options import Event, VerificationOptions, VerificationOptionsBuilder

__all__ = [
    # Core action classes
    "ObjectCollection",
    "ObjectCollectionBuilder",
    "ActionResult",
    "ActionResultBuilder",
    "ResultExtractor",
    "ResultMerger",
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
    "KeyDownOptionsBuilder",
    "KeyUpOptions",
    "KeyUpOptionsBuilder",
    "KeyModifier",
    "Key",
    "KeyCombo",
    "KeyCombos",
    # Mouse options
    "MousePressOptions",
    "MousePressOptionsBuilder",
    "MouseButton",
    "MouseMoveOptions",
    "MouseMoveOptionsBuilder",
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
    # Control flow
    "ControlFlowExecutor",
    "BreakLoop",
    "ContinueLoop",
]
