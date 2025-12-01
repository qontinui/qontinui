"""Base find options - ported from Qontinui framework.

Base configuration for all Find actions.
"""

from abc import ABC, abstractmethod
from typing import Any, TypeVar

from ....model.search_regions import SearchRegions
from ...action_config import ActionConfig, ActionConfigBuilder
from .find_strategy import FindStrategy
from .match_adjustment_options import MatchAdjustmentOptions

# Default similarity from Sikuli
DEFAULT_MIN_SIMILARITY = 0.7

# Type variable for builder pattern
TBuilder = TypeVar("TBuilder", bound="BaseFindOptionsBuilder[Any]")


class BaseFindOptions(ActionConfig, ABC):
    """Base configuration for all Find actions in the Qontinui framework.

    Port of BaseFindOptions from Qontinui framework.

    This abstract class encapsulates common parameters shared by all find operations,
    regardless of whether they use pattern matching, color analysis, or other techniques.
    It extends ActionConfig to inherit general action configuration while adding
    find-specific settings.

    Specialized find configurations (e.g., PatternFindOptions, ColorFindOptions)
    should extend this class to add their specific parameters while inheriting the common
    find functionality.

    This design promotes code reuse and ensures consistency across different find
    implementations while maintaining type safety and API clarity.
    """

    def __init__(self, builder: "BaseFindOptionsBuilder[Any]") -> None:
        """Initialize BaseFindOptions from builder.

        Args:
            builder: The builder instance containing configuration values
        """
        super().__init__(builder)
        self.similarity: float = builder.similarity
        self.search_regions: SearchRegions = builder.search_regions
        self.capture_image: bool = builder.capture_image
        self.use_defined_region: bool = builder.use_defined_region
        self.max_matches_to_act_on: int = builder.max_matches_to_act_on
        self.match_adjustment_options: MatchAdjustmentOptions = (
            builder.match_adjustment_options
        )
        self.search_duration: float = builder.search_duration

    @abstractmethod
    def get_find_strategy(self) -> FindStrategy:
        """Get the find strategy for this options instance.

        Subclasses should override this method to return their specific strategy.
        For example, PatternFindOptions would map its Strategy enum to FindStrategy,
        while ColorFindOptions would return FindStrategy.COLOR.

        Returns:
            The find strategy to use for this find operation
        """
        pass

    def get_similarity(self) -> float:
        """Get the minimum similarity threshold."""
        return self.similarity

    def get_search_regions(self) -> SearchRegions:
        """Get the search regions."""
        return self.search_regions

    def get_capture_image(self) -> bool:
        """Check if image capture is enabled."""
        return self.capture_image

    def get_use_defined_region(self) -> bool:
        """Check if using defined regions."""
        return self.use_defined_region

    def get_max_matches_to_act_on(self) -> int:
        """Get maximum matches to act on."""
        return self.max_matches_to_act_on

    def get_match_adjustment_options(self) -> MatchAdjustmentOptions:
        """Get match adjustment options."""
        return self.match_adjustment_options

    def get_search_duration(self) -> float:
        """Get search duration in seconds."""
        return self.search_duration


class BaseFindOptionsBuilder[TBuilder: "BaseFindOptionsBuilder[Any]"](
    ActionConfigBuilder
):
    """Abstract generic builder for constructing BaseFindOptions and its subclasses.

    Port of BaseFindOptions from Qontinui framework.Builder.

    This pattern allows for fluent, inheritable builder methods.
    """

    def __init__(self, original: BaseFindOptions | None = None) -> None:
        """Initialize builder.

        Args:
            original: Optional BaseFindOptions instance to copy values from
        """
        super().__init__(original)

        if original:
            self.similarity = original.similarity
            self.search_regions = SearchRegions(original.search_regions)
            self.capture_image = original.capture_image
            self.use_defined_region = original.use_defined_region
            self.max_matches_to_act_on = original.max_matches_to_act_on
            self.match_adjustment_options = (
                original.match_adjustment_options.to_builder().build()
            )
            self.search_duration = original.search_duration
        else:
            self.similarity = DEFAULT_MIN_SIMILARITY
            self.search_regions = SearchRegions()
            self.capture_image = True
            self.use_defined_region = False
            self.max_matches_to_act_on = -1
            self.match_adjustment_options = MatchAdjustmentOptions.builder().build()
            self.search_duration = 3.0  # Default 3 seconds, same as SikuliX default

    def _self(self) -> TBuilder:
        """Return self cast to the concrete builder type.

        This enables proper type inference for builder chaining.

        Returns:
            Self cast to TBuilder type
        """
        return self  # type: ignore[return-value]

    def set_similarity(self, similarity: float) -> TBuilder:
        """Set the minimum similarity score (0.0 to 1.0) for a match to be considered valid.

        This threshold determines how closely a found element must match the search pattern.
        Lower values allow for more variation but may produce false positives.

        Args:
            similarity: The minimum similarity threshold

        Returns:
            This builder instance for chaining
        """
        self.similarity = similarity
        return self._self()

    def set_search_regions(self, search_regions: SearchRegions) -> TBuilder:
        """Set the regions of the screen to search within.

        By default, the entire screen is searched. This can be restricted to improve
        performance and accuracy by limiting the search area.

        Args:
            search_regions: The regions to search within

        Returns:
            This builder instance for chaining
        """
        self.search_regions = search_regions
        return self._self()

    def set_capture_image(self, capture_image: bool) -> TBuilder:
        """Set whether to capture an image of the match for logging and debugging.

        Captured images can be useful for troubleshooting but may impact performance.

        Args:
            capture_image: True to capture match images, False otherwise

        Returns:
            This builder instance for chaining
        """
        self.capture_image = capture_image
        return self._self()

    def set_use_defined_region(self, use_defined_region: bool) -> TBuilder:
        """Set whether to use defined regions instead of searching.

        If true, bypasses image search and creates Match objects directly from
        pre-defined regions in the StateImage objects. This is useful when the
        location of elements is known in advance.

        Args:
            use_defined_region: True to use defined regions instead of searching

        Returns:
            This builder instance for chaining
        """
        self.use_defined_region = use_defined_region
        return self._self()

    def set_max_matches_to_act_on(self, max_matches_to_act_on: int) -> TBuilder:
        """Limit the number of matches to act on.

        Limits the number of matches to act on when using strategies that find
        multiple matches. A value <= 0 means no limit.

        Args:
            max_matches_to_act_on: The maximum number of matches to process

        Returns:
            This builder instance for chaining
        """
        self.max_matches_to_act_on = max_matches_to_act_on
        return self._self()

    def set_match_adjustment(
        self, match_adjustment_options: MatchAdjustmentOptions
    ) -> TBuilder:
        """Set the match adjustment options for post-processing found matches.

        This allows for resizing match regions or targeting specific points within matches.

        Args:
            match_adjustment_options: The match adjustment options

        Returns:
            This builder instance for chaining
        """
        self.match_adjustment_options = match_adjustment_options
        return self._self()

    def set_search_duration(self, seconds: float) -> TBuilder:
        """Set the search duration (in seconds) for finding a match.

        The search will continue until a match is found or this duration is reached.
        This replaces the deprecated ActionOptions.maxWait parameter.

        Args:
            seconds: The maximum duration to search for a match (default: 3.0 seconds)

        Returns:
            This builder instance for chaining
        """
        self.search_duration = seconds
        return self._self()
