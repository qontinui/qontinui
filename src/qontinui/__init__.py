"""Qontinui Core: Model-based GUI automation with AI-enhanced perception.

Following Brobot principles with modern Python implementation.
"""

# CRITICAL: Initialize DPI awareness BEFORE any other imports
# This ensures physical resolution capture on Windows
# Actions (Brobot-style)
from .actions import (
    Action,
    ActionChain,
    ActionConfig,
    ActionInterface,
    ActionResult,
    ClickOptions,
    DragOptions,
    FindOptions,
    FluentActions,
    MouseMoveOptions,
    PureActions,
    ScrollOptions,
    TypeOptions,
    WaitOptions,
)

# Migration Tools
# NOTE: BrobotConverter requires AI dependencies (faiss-cpu, torch, transformers)
# Uncomment when these packages are installed in your environment
# from .migrations import BrobotConverter
# DSL Parser
from .dsl import QontinuiDSLParser

# Navigation API
from . import navigation_api, registry

# Find System (Brobot-style)
from .find import Find, FindImage, FindResults, Match, Matches

# Perception Pipeline
# NOTE: Perception modules require AI dependencies (faiss-cpu, torch, transformers)
# Uncomment when these packages are installed in your environment
# from .perception import (
#     ScreenSegmenter,
#     ObjectVectorizer,
#     ElementMatcher,
# )
# Model Elements (Brobot-style)
from .model.element import HSV, RGB, Anchor, Image, Location, Pattern, Position, Region

# Match System (Brobot-style)
from .model.match import Match as MatchObject
from .model.search_regions import SearchRegions

# State System (Brobot-style)
from .model.state import (
    Path,
    PathFinder,
    State,
    StateEnum,
    StateImage,
    StateLocation,
    StateMemory,
    StateObject,
    StateRegion,
    StateService,
    StateString,
)
from .model.state import State as BrobotState

# Transition System (Brobot-style)
from .model.transition import StateTransition, StateTransitions, TransitionType

# Primitives (Brobot-style)
from .primitives import (
    KeyDown,
    KeyPress,
    KeyUp,
    MouseClick,
    MouseDown,
    MouseDrag,
    MouseMove,
    MouseUp,
    MouseWheel,
    TypeText,
)
from .startup import PhysicalResolutionInitializer  # noqa: F401 - documented for user reference
from .state_management import QontinuiStateManager
from .state_management.models import Element, StateGraph, Transition
from .state_management.traversal import StateTraversal

# Initialize DPI awareness after all imports
# NOTE: Commented out automatic initialization to prevent blocking in headless environments
# Users can manually call PhysicalResolutionInitializer.force_initialization() if needed
# PhysicalResolutionInitializer.force_initialization()

__version__ = "0.1.0"

__all__ = [
    # Navigation API
    "navigation_api",
    "registry",
    # Original State Management (from state_management)
    "QontinuiStateManager",
    "State",
    "Element",
    "Transition",
    "StateGraph",
    "StateTraversal",
    # Perception
    # NOTE: Perception exports require AI dependencies (faiss-cpu, torch, transformers)
    # "ScreenSegmenter",
    # "ObjectVectorizer",
    # "ElementMatcher",
    # Model Elements (Brobot-style)
    "Location",
    "Region",
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
    "MouseMoveOptions",
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
    # "StateImageObject",  # REMOVED: Not imported - may be deprecated
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
    "SearchRegions",
    # Tools
    # "BrobotConverter",  # NOTE: Requires AI dependencies (faiss-cpu, torch, transformers)
    "QontinuiDSLParser",
]
