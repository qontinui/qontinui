"""Specific Options classes - ported from Qontinui framework.

Each action type has its own Options class that inherits directly from ActionConfig.
These control the specific behavior and timing of each action type.
Note: ActionOptions was deprecated and removed from Brobot.
"""

from enum import Enum

from .action_config import ActionConfig


class ClickType(Enum):
    """Types of mouse clicks."""

    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"
    DOUBLE = "double"
    TRIPLE = "triple"


class ScrollDirection(Enum):
    """Scroll directions."""

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


class KeyModifier(Enum):
    """Keyboard modifiers."""

    CTRL = "ctrl"
    ALT = "alt"
    SHIFT = "shift"
    META = "meta"  # Windows/Command key


class ClickOptions(ActionConfig):
    """Options for click actions.

    Port of ClickOptions from Qontinui framework.
    Inherits directly from ActionConfig.
    """

    def __init__(self) -> None:
        """Initialize with click-specific defaults."""
        super().__init__()
        self._action_name = "click"
        self._click_type: ClickType = ClickType.LEFT
        self._click_count: int = 1
        self._pause_between_clicks: float = 0.05
        self._hold_duration: float = 0.0  # For click-and-hold

    def click_type(self, click_type: ClickType) -> "ClickOptions":
        """Set click type (fluent)."""
        self._click_type = click_type
        return self

    def click_count(self, count: int) -> "ClickOptions":
        """Set number of clicks (fluent)."""
        self._click_count = count
        return self

    def double_click(self) -> "ClickOptions":
        """Configure for double click (fluent)."""
        self._click_type = ClickType.DOUBLE
        self._click_count = 2
        return self

    def right_click(self) -> "ClickOptions":
        """Configure for right click (fluent)."""
        self._click_type = ClickType.RIGHT
        return self

    def pause_between_clicks(self, seconds: float) -> "ClickOptions":
        """Set pause between multiple clicks (fluent)."""
        self._pause_between_clicks = seconds
        return self

    def hold_duration(self, seconds: float) -> "ClickOptions":
        """Set hold duration for click-and-hold (fluent)."""
        self._hold_duration = seconds
        return self


class DragOptions(ActionConfig):
    """Options for drag actions.

    Port of DragOptions from Qontinui framework.
    Inherits directly from ActionConfig.
    """

    def __init__(self) -> None:
        """Initialize with drag-specific defaults."""
        super().__init__()
        self._action_name = "drag"
        self._drag_duration: float = 1.0
        self._button: str = "left"
        self._pause_before_drag: float = 0.1
        self._pause_after_drag: float = 0.1
        self._smooth_movement: bool = True

    def drag_duration(self, seconds: float) -> "DragOptions":
        """Set drag duration (fluent)."""
        self._drag_duration = seconds
        return self

    def button(self, button: str) -> "DragOptions":
        """Set mouse button for drag (fluent)."""
        self._button = button
        return self

    def pause_before_drag(self, seconds: float) -> "DragOptions":
        """Set pause before starting drag (fluent)."""
        self._pause_before_drag = seconds
        return self

    def pause_after_drag(self, seconds: float) -> "DragOptions":
        """Set pause after completing drag (fluent)."""
        self._pause_after_drag = seconds
        return self

    def smooth_movement(self, smooth: bool = True) -> "DragOptions":
        """Enable/disable smooth movement (fluent)."""
        self._smooth_movement = smooth
        return self


class TypeOptions(ActionConfig):
    """Options for typing/text input actions.

    Port of TypeOptions from Qontinui framework.
    Inherits directly from ActionConfig.
    """

    def __init__(self) -> None:
        """Initialize with type-specific defaults."""
        super().__init__()
        self._action_name = "type"
        self._typing_delay: float = 0.05  # Delay between keystrokes
        self._clear_before: bool = False  # Clear field before typing
        self._press_enter: bool = False  # Press enter after typing
        self._modifiers: list[KeyModifier] = []

    def typing_delay(self, seconds: float) -> "TypeOptions":
        """Set delay between keystrokes (fluent)."""
        self._typing_delay = seconds
        return self

    def clear_before(self, clear: bool = True) -> "TypeOptions":
        """Clear field before typing (fluent)."""
        self._clear_before = clear
        return self

    def press_enter(self, press: bool = True) -> "TypeOptions":
        """Press enter after typing (fluent)."""
        self._press_enter = press
        return self

    def with_modifiers(self, *modifiers: KeyModifier) -> "TypeOptions":
        """Add keyboard modifiers (fluent)."""
        self._modifiers = list(modifiers)
        return self


class KeyDownOptions(ActionConfig):
    """Options for key down actions.

    Port of KeyDownOptions from Qontinui framework.
    Inherits directly from ActionConfig.
    """

    def __init__(self) -> None:
        """Initialize with key down defaults."""
        super().__init__()
        self._action_name = "key_down"
        self._hold_duration: float = 0.0  # 0 = hold until key_up
        self._modifiers: list[KeyModifier] = []

    def hold_duration(self, seconds: float) -> "KeyDownOptions":
        """Set how long to hold key (fluent)."""
        self._hold_duration = seconds
        return self

    def with_modifiers(self, *modifiers: KeyModifier) -> "KeyDownOptions":
        """Add keyboard modifiers (fluent)."""
        self._modifiers = list(modifiers)
        return self


class KeyUpOptions(ActionConfig):
    """Options for key up actions.

    Port of KeyUpOptions from Qontinui framework.
    Inherits directly from ActionConfig.
    """

    def __init__(self) -> None:
        """Initialize with key up defaults."""
        super().__init__()
        self._action_name = "key_up"
        self._release_modifiers: bool = True

    def release_modifiers(self, release: bool = True) -> "KeyUpOptions":
        """Release modifiers with key (fluent)."""
        self._release_modifiers = release
        return self


class ScrollOptions(ActionConfig):
    """Options for scroll actions.

    Port of ScrollOptions from Qontinui framework.
    Inherits directly from ActionConfig.
    """

    def __init__(self) -> None:
        """Initialize with scroll defaults."""
        super().__init__()
        self._action_name = "scroll"
        self._scroll_direction: ScrollDirection = ScrollDirection.DOWN
        self._scroll_amount: int = 3  # Number of scroll units
        self._smooth_scroll: bool = True
        self._scroll_duration: float = 0.5

    def scroll_direction(self, direction: ScrollDirection) -> "ScrollOptions":
        """Set scroll direction (fluent)."""
        self._scroll_direction = direction
        return self

    def scroll_amount(self, amount: int) -> "ScrollOptions":
        """Set scroll amount (fluent)."""
        self._scroll_amount = amount
        return self

    def smooth_scroll(self, smooth: bool = True) -> "ScrollOptions":
        """Enable/disable smooth scrolling (fluent)."""
        self._smooth_scroll = smooth
        return self

    def scroll_duration(self, seconds: float) -> "ScrollOptions":
        """Set scroll animation duration (fluent)."""
        self._scroll_duration = seconds
        return self


class WaitOptions(ActionConfig):
    """Options for wait actions.

    Port of WaitOptions from Qontinui framework.
    Inherits directly from ActionConfig.
    """

    def __init__(self) -> None:
        """Initialize with wait defaults."""
        super().__init__()
        self._action_name = "wait"
        self._wait_for: str = "time"  # time, image, state, condition
        self._condition_check_interval: float = 0.5
        self._log_progress: bool = False

    def wait_for(self, wait_type: str) -> "WaitOptions":
        """Set what to wait for (fluent)."""
        self._wait_for = wait_type
        return self

    def check_interval(self, seconds: float) -> "WaitOptions":
        """Set condition check interval (fluent)."""
        self._condition_check_interval = seconds
        return self

    def log_progress(self, log: bool = True) -> "WaitOptions":
        """Enable/disable progress logging (fluent)."""
        self._log_progress = log
        return self


class FindOptions(ActionConfig):
    """Options for find/search actions.

    Port of FindOptions from Qontinui framework.
    Inherits directly from ActionConfig.
    """

    def __init__(self) -> None:
        """Initialize with find defaults."""
        super().__init__()
        self._action_name = "find"
        self._find_all: bool = False  # Find all matches vs first
        self._max_matches: int = 100
        self._sort_by: str = "similarity"  # similarity, position, size
        self._cache_result: bool = False
        self._use_cache: bool = True
        self._min_similarity: float = 0.7  # Default similarity threshold

    def find_all(self, find_all: bool = True) -> "FindOptions":
        """Find all matches vs first (fluent)."""
        self._find_all = find_all
        return self

    def max_matches(self, max_matches: int) -> "FindOptions":
        """Set maximum number of matches (fluent)."""
        self._max_matches = max_matches
        return self

    def sort_by(self, sort_criteria: str) -> "FindOptions":
        """Set sort criteria for matches (fluent)."""
        self._sort_by = sort_criteria
        return self

    def cache_result(self, cache: bool = True) -> "FindOptions":
        """Enable/disable result caching (fluent)."""
        self._cache_result = cache
        return self

    def use_cache(self, use: bool = True) -> "FindOptions":
        """Enable/disable cache usage (fluent)."""
        self._use_cache = use
        return self

    def min_similarity(self, similarity: float) -> "FindOptions":
        """Set minimum similarity threshold (fluent)."""
        self._min_similarity = similarity
        return self

    def search_region(self, x: int, y: int, width: int, height: int) -> "FindOptions":
        """Set search region (fluent).

        Args:
            x: X coordinate
            y: Y coordinate
            width: Width
            height: Height

        Returns:
            Self for chaining
        """
        # Store search region parameters if needed
        return self

    def timeout(self, seconds: float) -> "FindOptions":
        """Set timeout (fluent).

        Args:
            seconds: Timeout in seconds

        Returns:
            Self for chaining
        """
        # Store timeout if needed
        return self


class GetTextOptions(ActionConfig):
    """Options for text extraction/OCR actions.

    Port of GetTextOptions from Qontinui framework.
    Inherits directly from ActionConfig.
    """

    def __init__(self) -> None:
        """Initialize with text extraction defaults."""
        super().__init__()
        self._action_name = "get_text"
        self._language: str = "eng"  # OCR language
        self._preprocess: bool = True  # Preprocess image for OCR
        self._whitelist: str | None = None  # Character whitelist
        self._blacklist: str | None = None  # Character blacklist
        self._min_confidence: float = 0.6

    def language(self, lang: str) -> "GetTextOptions":
        """Set OCR language (fluent)."""
        self._language = lang
        return self

    def preprocess(self, preprocess: bool = True) -> "GetTextOptions":
        """Enable/disable image preprocessing (fluent)."""
        self._preprocess = preprocess
        return self

    def whitelist(self, chars: str) -> "GetTextOptions":
        """Set character whitelist (fluent)."""
        self._whitelist = chars
        return self

    def blacklist(self, chars: str) -> "GetTextOptions":
        """Set character blacklist (fluent)."""
        self._blacklist = chars
        return self

    def min_confidence(self, confidence: float) -> "GetTextOptions":
        """Set minimum OCR confidence (fluent)."""
        self._min_confidence = confidence
        return self
