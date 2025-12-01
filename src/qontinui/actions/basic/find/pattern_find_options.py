"""Pattern find options - ported from Qontinui framework.

Configuration for pattern-matching Find actions.
"""

from enum import Enum, auto

from .base_find_options import BaseFindOptions, BaseFindOptionsBuilder
from .find_strategy import FindStrategy
from .match_fusion_options import FusionMethod, MatchFusionOptions


class Strategy(Enum):
    """The pattern matching strategy."""

    FIRST = auto()
    """Returns the first match found. Stops searching once any Pattern
    finds a match, making it efficient for existence checks."""

    ALL = auto()
    """Returns all matches for all Patterns across all Images. Useful for
    counting or processing multiple instances of an element."""

    EACH = auto()
    """Returns one match per Image object. The DoOnEach option
    determines whether to return the first or best match per Image."""

    BEST = auto()
    """Performs an ALL search then returns only the match with the
    highest similarity score."""


class DoOnEach(Enum):
    """Controls match selection strategy when using Strategy.EACH."""

    FIRST = auto()
    """Returns the first match found for each Image (fastest)."""

    BEST = auto()
    """Returns the match with the highest similarity for each Image."""


class PatternFindOptions(BaseFindOptions):
    """Configuration for all standard pattern-matching Find actions.

    Port of PatternFindOptions from Qontinui framework.

    This class encapsulates parameters specific to finding objects via image or text
    pattern matching. It extends BaseFindOptions to inherit common find
    functionality while adding pattern-specific settings.

    It is an immutable object and must be constructed using its inner Builder.

    By providing a specialized configuration class, the Qontinui API ensures that only
    relevant options are available for pattern matching, enhancing type safety and ease of use.
    """

    def __init__(self, builder: "PatternFindOptionsBuilder") -> None:
        """Initialize PatternFindOptions from builder.

        Args:
            builder: The builder instance containing configuration values
        """
        super().__init__(builder)
        self.strategy = builder.strategy
        self.do_on_each = builder.do_on_each
        self.match_fusion_options = builder.match_fusion_options

    def get_find_strategy(self) -> FindStrategy:
        """Get the find strategy for this pattern find operation.

        Returns:
            The corresponding FindStrategy enum value
        """
        if self.strategy == Strategy.FIRST:
            return FindStrategy.FIRST
        elif self.strategy == Strategy.ALL:
            return FindStrategy.ALL
        elif self.strategy == Strategy.EACH:
            return FindStrategy.EACH
        elif self.strategy == Strategy.BEST:
            return FindStrategy.BEST
        else:
            return FindStrategy.FIRST

    def get_strategy(self) -> Strategy:
        """Get the pattern matching strategy."""
        return self.strategy

    def get_do_on_each(self) -> DoOnEach:
        """Get the DoOnEach strategy."""
        return self.do_on_each

    def get_match_fusion_options(self) -> MatchFusionOptions:
        """Get match fusion options."""
        return self.match_fusion_options

    @staticmethod
    def for_quick_search() -> "PatternFindOptions":
        """Create a configuration optimized for quick pattern matching.

        This factory method provides a preset configuration for scenarios where
        speed is more important than precision. It uses:
        - FIRST strategy (stops after finding one match)
        - Lower similarity threshold (0.7)
        - Disabled image capture for performance

        Returns:
            A PatternFindOptions configured for quick searches
        """
        builder: PatternFindOptionsBuilder = PatternFindOptionsBuilder()
        builder.set_strategy(Strategy.FIRST)
        builder.set_similarity(0.7)
        builder.set_capture_image(False)
        builder.set_max_matches_to_act_on(1)
        return builder.build()

    @staticmethod
    def for_precise_search() -> "PatternFindOptions":
        """Create a configuration optimized for precise pattern matching.

        This factory method provides a preset configuration for scenarios where
        accuracy is more important than speed. It uses:
        - BEST strategy (finds all matches and returns highest scoring)
        - High similarity threshold (0.9)
        - Enabled image capture for debugging
        - Conservative match fusion settings

        Returns:
            A PatternFindOptions configured for precise searches
        """
        fusion_builder = MatchFusionOptions.builder()
        fusion_builder.set_fusion_method(FusionMethod.ABSOLUTE)
        fusion_builder.set_max_fusion_distance_x(10)
        fusion_builder.set_max_fusion_distance_y(10)
        fusion_options: MatchFusionOptions = fusion_builder.build()

        builder: PatternFindOptionsBuilder = PatternFindOptionsBuilder()
        builder.set_strategy(Strategy.BEST)
        builder.set_similarity(0.9)
        builder.set_capture_image(True)
        builder.set_match_fusion(fusion_options)
        return builder.build()

    @staticmethod
    def for_all_matches() -> "PatternFindOptions":
        """Create a configuration for finding all occurrences of a pattern.

        This factory method provides a preset configuration for scenarios where
        you need to find multiple instances of an element. It uses:
        - ALL strategy (finds all matches)
        - Balanced similarity threshold (0.8)
        - Match fusion to combine adjacent matches
        - No limit on match count

        Returns:
            A PatternFindOptions configured for finding all matches
        """
        fusion_builder = MatchFusionOptions.builder()
        fusion_builder.set_fusion_method(FusionMethod.ABSOLUTE)
        fusion_builder.set_max_fusion_distance_x(20)
        fusion_builder.set_max_fusion_distance_y(20)
        fusion_options: MatchFusionOptions = fusion_builder.build()

        builder: PatternFindOptionsBuilder = PatternFindOptionsBuilder()
        builder.set_strategy(Strategy.ALL)
        builder.set_similarity(0.8)
        builder.set_capture_image(False)
        builder.set_max_matches_to_act_on(-1)
        builder.set_match_fusion(fusion_options)
        return builder.build()


class PatternFindOptionsBuilder(BaseFindOptionsBuilder["PatternFindOptionsBuilder"]):
    """Builder for constructing PatternFindOptions with a fluent API.

    Port of PatternFindOptions from Qontinui framework.Builder.
    """

    strategy: Strategy
    do_on_each: DoOnEach
    match_fusion_options: MatchFusionOptions

    def __init__(self, original: PatternFindOptions | None = None) -> None:
        """Initialize builder.

        Args:
            original: Optional PatternFindOptions instance to copy values from
        """
        super().__init__(original)

        if original:
            self.strategy = original.strategy
            self.do_on_each = original.do_on_each
            self.match_fusion_options = original.match_fusion_options.to_builder().build()
        else:
            self.strategy = Strategy.FIRST
            self.do_on_each = DoOnEach.FIRST
            self.match_fusion_options = MatchFusionOptions.builder().build()

    def set_strategy(self, strategy: Strategy) -> "PatternFindOptionsBuilder":
        """Set the pattern matching strategy.

        Args:
            strategy: The strategy to use (e.g., FIRST, ALL)

        Returns:
            This builder instance for chaining
        """
        self.strategy = strategy
        return self

    def set_do_on_each(self, do_on_each: DoOnEach) -> "PatternFindOptionsBuilder":
        """Set the strategy for selecting one match per image when using Find.EACH.

        Args:
            do_on_each: The strategy to use (e.g., FIRST, BEST)

        Returns:
            This builder instance for chaining
        """
        self.do_on_each = do_on_each
        return self

    def set_match_fusion(
        self, match_fusion_options: MatchFusionOptions
    ) -> "PatternFindOptionsBuilder":
        """Set the match fusion options for combining adjacent matches.

        Args:
            match_fusion_options: The match fusion options

        Returns:
            This builder instance for chaining
        """
        self.match_fusion_options = match_fusion_options
        return self

    def build(self) -> PatternFindOptions:
        """Build the immutable PatternFindOptions object.

        Returns:
            A new instance of PatternFindOptions
        """
        return PatternFindOptions(self)

    def _self(self) -> "PatternFindOptionsBuilder":
        """Return self for fluent interface.

        Returns:
            This builder instance
        """
        return self
