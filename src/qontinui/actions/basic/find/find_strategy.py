"""Find strategy enum - ported from Qontinui framework.

Defines the various strategies available for find operations.
"""

from enum import Enum, auto


class FindStrategy(Enum):
    """Defines the various strategies available for find operations in Qontinui.

    Port of FindStrategy from Qontinui framework enum.

    This enum consolidates all find strategies that were previously scattered across
    different options classes. It provides a unified type system for find operations
    while maintaining backward compatibility with the original ActionOptions.Find enum.

    Each strategy represents a different approach to finding elements on the screen,
    from basic pattern matching to advanced motion detection and color analysis.
    """

    # Pattern-based strategies (used by PatternFindOptions)
    FIRST = auto()
    """Returns the first match found. Stops searching once any Pattern
    finds a match, making it efficient for existence checks."""

    EACH = auto()
    """Returns one match per Image object. The DoOnEach option in PatternFindOptions
    determines whether to return the first or best match per Image."""

    ALL = auto()
    """Returns all matches for all Patterns across all Images. Useful for
    counting or processing multiple instances of an element."""

    BEST = auto()
    """Performs an ALL search then returns only the match with the
    highest similarity score."""

    # Special strategies
    UNIVERSAL = auto()
    """Used for mocking. Initializing an Image with a UNIVERSAL Find allows it
    to be accessed by any find operation type."""

    CUSTOM = auto()
    """User-defined find strategy. Must be registered with FindStrategyRegistry
    before use."""

    # Color-based strategies (used by ColorFindOptions)
    COLOR = auto()
    """Finds regions based on color analysis using k-means clustering,
    mean color statistics, or classification."""

    # Histogram-based strategies (used by HistogramFindOptions)
    HISTOGRAM = auto()
    """Matches regions based on histogram similarity from the input images."""

    # Motion-based strategies (used by MotionFindOptions)
    MOTION = auto()
    """Finds the locations of a moving object across consecutive screens."""

    REGIONS_OF_MOTION = auto()
    """Finds all dynamic pixel regions from a series of screens."""

    FIXED_PIXELS = auto()
    """Returns a mask of all pixels that remain unchanged and a corresponding
    Match list from the contours."""

    DYNAMIC_PIXELS = auto()
    """Returns a mask of all pixels that have changed and a corresponding
    Match list from the contours."""

    # Text-based strategies
    ALL_WORDS = auto()
    """Finds all words and their regions. Each word is returned as a separate
    Match object. For finding all text in a specific region as one Match,
    use a normal Find operation."""

    # Image comparison strategies
    SIMILAR_IMAGES = auto()
    """Finds images in the second ObjectCollection that are above a similarity
    threshold to images in the first ObjectCollection."""

    # State analysis strategies
    STATES = auto()
    """Analyzes ObjectCollections containing screen images and screenshots to
    produce states with StateImage objects. Returns Match objects holding
    the state owner's name and Pattern."""
