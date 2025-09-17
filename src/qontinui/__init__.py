"""Qontinui Core: Model-based GUI automation with AI-enhanced perception.

Following Brobot principles with modern Python implementation.
"""

# CRITICAL: Initialize DPI awareness BEFORE any other imports
# This ensures physical resolution capture on Windows
from .startup import PhysicalResolutionInitializer
PhysicalResolutionInitializer.force_initialization()

__version__ = "0.1.0"

# State Management
from .state_management import QontinuiStateManager
from .state_management.models import Element, Transition, StateGraph
from .state_management.traversal import StateTraversal
from .model.state import State

# Perception Pipeline
# TODO: Enable after installing AI dependencies (faiss, torch, transformers)
# from .perception import (
#     ScreenSegmenter,
#     ObjectVectorizer,
#     ElementMatcher,
# )

# Model Elements (Brobot-style)
from .model.element import (
    Location,
    Region,
    Image,
    Pattern,
    RGB,
    HSV,
    Position,
    Anchor,
)
from .model.search_regions import SearchRegions

# Find System (Brobot-style)
from .find import (
    Find,
    FindImage,
    Match,
    Matches,
    FindResults,
)

# Actions (Brobot-style)
from .actions import (
    PureActions,
    FluentActions,
    ActionResult,
    ActionChain,
    Action,
    ActionConfig,
    ActionInterface,
    ClickOptions,
    DragOptions,
    MoveOptions,
    TypeOptions,
    ScrollOptions,
    WaitOptions,
    FindOptions,
)

# Primitives (Brobot-style)
from .primitives import (
    MouseMove,
    MouseClick,
    MouseDrag,
    MouseWheel,
    MouseDown,
    MouseUp,
    KeyPress,
    KeyDown,
    KeyUp,
    TypeText,
)

# State System (Brobot-style)
from .model.state import (
    State as BrobotState,
    StateEnum,
    StateImage,
    StateObject,
    StateLocation,
    StateRegion,
    StateString,
    StateMemory,
    StateService,
    Path,
    PathFinder,
)

# Transition System (Brobot-style)
from .model.transition import (
    StateTransition,
    StateTransitions,
    TransitionType,
)

# Match System (Brobot-style)
from .model.match import Match as MatchObject

# Migration Tools
# TODO: Enable after installing AI dependencies
# from .migrations import BrobotConverter

# DSL Parser
from .dsl import QontinuiDSLParser

__all__ = [
    # Original State Management (from state_management)
    "QontinuiStateManager",
    "State",
    "Element",
    "Transition",
    "StateGraph",
    "StateTraversal",
    
    # Perception
    # TODO: Enable after installing AI dependencies
    # "ScreenSegmenter",
    # "ObjectVectorizer",
    # "ElementMatcher",
    
    # Model Elements (Brobot-style)
    "Location",
    "Region",
    "SearchRegion",
    "Image",
    "Pattern",
    "RGB",
    "HSV",
    "Position",
    "Anchor",
    
    # Find System (Brobot-style)
    "Find",
    "FindImage",
    "Match",
    "Matches",
    "FindResults",
    
    # Actions (Brobot-style)
    "PureActions",
    "FluentActions",
    "ActionResult",
    "ActionChain",
    "Action",
    "ActionConfig",
    "ActionInterface",
    "ClickOptions",
    "DragOptions",
    "MoveOptions",
    "TypeOptions",
    "ScrollOptions",
    "WaitOptions",
    "FindOptions",
    
    # Primitives (Brobot-style)
    "MouseMove",
    "MouseClick",
    "MouseDrag",
    "MouseWheel",
    "MouseDown",
    "MouseUp",
    "KeyPress",
    "KeyDown",
    "KeyUp",
    "TypeText",
    
    # State System (Brobot-style)
    "BrobotState",
    "StateEnum",
    "StateImage",
    "StateObject",
    "StateImageObject",
    "StateLocation",
    "StateRegion",
    "StateString",
    "StateMemory",
    "StateService",
    "Path",
    "PathFinder",
    
    # Transition System (Brobot-style)
    "StateTransition",
    "StateTransitions",
    "TransitionType",
    
    # Match System (Brobot-style)
    "MatchObject",
    
    # Tools
    # "BrobotConverter",  # TODO: Enable after installing AI dependencies
    "QontinuiDSLParser",
]