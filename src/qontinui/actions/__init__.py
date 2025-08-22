"""Actions package - ported from Qontinui framework.

Core action system including ObjectCollection.
"""

from .object_collection import ObjectCollection, ObjectCollectionBuilder
from .action_result import ActionResult
from .action_type import ActionType
from .action_config import ActionConfig, ActionConfigBuilder, Illustrate, LoggingOptions
from .action_interface import ActionInterface
from .action import Action
from .verification_options import VerificationOptions, VerificationOptionsBuilder, Event
from .repetition_options import RepetitionOptions, RepetitionOptionsBuilder

# Import basic action options
from .basic.find import (
    PatternFindOptions, PatternFindOptionsBuilder,
    FindStrategy, Strategy, DoOnEach
)
from .basic.click import ClickOptions, ClickOptionsBuilder
from .basic.type import TypeOptions, TypeOptionsBuilder
from .basic.mouse import (
    MousePressOptions, MousePressOptionsBuilder, MouseButton,
    MouseMoveOptions, ScrollOptions, ScrollOptionsBuilder, Direction
)

# Import composite action options
from .composite import DragOptions, DragOptionsBuilder

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
]