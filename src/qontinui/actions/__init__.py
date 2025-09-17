"""Actions package - ported from Qontinui framework.

Core action system including ObjectCollection.
"""

from .object_collection import ObjectCollection, ObjectCollectionBuilder
from .action_result import ActionResult
from .action_type import ActionType
from .action_config import ActionConfig, ActionConfigBuilder, Illustrate, LoggingOptions
from .action_interface import ActionInterface
from .action import Action
from .action_options import FindOptions, KeyModifier
from .keys import Key, KeyCombo, KeyCombos
from .verification_options import VerificationOptions, VerificationOptionsBuilder, Event
from .repetition_options import RepetitionOptions, RepetitionOptionsBuilder

# Import basic action options
from .basic.find import (
    PatternFindOptions, PatternFindOptionsBuilder,
    FindStrategy, Strategy, DoOnEach
)
from .basic.click import ClickOptions, ClickOptionsBuilder
from .basic.type import TypeOptions, TypeOptionsBuilder
from .basic.type.key_down_options import KeyDownOptions
from .basic.type.key_up_options import KeyUpOptions
from .basic.mouse import (
    MousePressOptions, MousePressOptionsBuilder, MouseButton,
    MouseMoveOptions, ScrollOptions, ScrollOptionsBuilder, Direction
)

# Import composite action options
from .composite import DragOptions, DragOptionsBuilder
from .composite.chains.action_chain import ActionChain

# Import pure and fluent actions
from .pure import PureActions
from .fluent import FluentActions

# Import unified Actions class
from .actions import Actions

# Import wait options
from .basic.wait.wait import WaitOptions

# Create aliases
MoveOptions = MouseMoveOptions

__all__ = [
    # Core action classes
    'ObjectCollection',
    'ObjectCollectionBuilder',
    'ActionResult',
    'ActionType',
    'ActionConfig',
    'ActionConfigBuilder',
    'Illustrate',
    'LoggingOptions',
    'ActionInterface',
    'Action',
    'VerificationOptions',
    'VerificationOptionsBuilder',
    'Event',
    'RepetitionOptions',
    'RepetitionOptionsBuilder',
    
    # Find options
    'FindOptions',
    'PatternFindOptions',
    'PatternFindOptionsBuilder',
    'FindStrategy',
    'Strategy',
    'DoOnEach',
    
    # Click options
    'ClickOptions',
    'ClickOptionsBuilder',
    
    # Type options
    'TypeOptions',
    'TypeOptionsBuilder',
    'KeyDownOptions',
    'KeyUpOptions',
    'KeyModifier',
    'Key',
    'KeyCombo',
    'KeyCombos',
    
    # Mouse options
    'MousePressOptions',
    'MousePressOptionsBuilder',
    'MouseButton',
    'MouseMoveOptions',
    'ScrollOptions',
    'ScrollOptionsBuilder',
    'Direction',
    
    # Composite options
    'DragOptions',
    'DragOptionsBuilder',
    'ActionChain',
    
    # Pure and fluent actions
    'PureActions',
    'FluentActions',
    'Actions',
    
    # Wait options
    'WaitOptions',
    
    # Aliases for convenience
    'MoveOptions',  # Alias for MouseMoveOptions
]