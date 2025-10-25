"""Action type enumeration - ported from Qontinui framework.

Enumerates all available action types.
"""

from enum import Enum, auto


class ActionType(Enum):
    """Enumerates all available action types in the Qontinui framework.

    Port of ActionType from Qontinui framework enum.

    Actions are categorized into:
    - Basic Actions: Simple, atomic operations like finding patterns or clicking
    - Composite Actions: Complex operations that combine multiple basic actions
    """

    # Basic Actions
    FIND = auto()
    """Finds patterns, text, or regions on the screen"""

    CLICK = auto()
    """Performs mouse click on found elements (left/right/middle/double)"""

    MIDDLE_CLICK = auto()
    """Performs middle mouse click on found elements"""

    DEFINE = auto()
    """Defines regions based on found elements or criteria"""

    TYPE = auto()
    """Types text or key combinations"""

    MOVE = auto()
    """Moves the mouse cursor"""

    HOVER = auto()
    """Hovers mouse over an element (same as MOVE)"""

    VANISH = auto()
    """Waits for elements to disappear"""

    WAIT_VANISH = auto()
    """Waits for elements to disappear (alias for VANISH)"""

    HIGHLIGHT = auto()
    """Highlights elements on screen"""

    SCROLL_MOUSE_WHEEL = auto()
    """Scrolls using the mouse wheel"""

    SCROLL_UP = auto()
    """Scrolls up using the mouse wheel"""

    SCROLL_DOWN = auto()
    """Scrolls down using the mouse wheel"""

    MOUSE_DOWN = auto()
    """Presses and holds mouse button"""

    MOUSE_UP = auto()
    """Releases mouse button"""

    KEY_DOWN = auto()
    """Presses and holds keyboard key"""

    KEY_UP = auto()
    """Releases keyboard key"""

    CLASSIFY = auto()
    """Classifies images using machine learning"""

    # Composite Actions
    CLICK_UNTIL = auto()
    """Repeatedly clicks until a condition is met"""

    DRAG = auto()
    """Drags from one location to another"""

    RUN_PROCESS = auto()
    """Executes a named process (sequence of actions) with optional repetition"""
