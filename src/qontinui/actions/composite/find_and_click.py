"""FindAndClick composite action - ported from Qontinui framework.

Combines Find and Click operations in a single action.
"""

import logging
from dataclasses import dataclass

from ...actions.find import FindAction
from ...actions.find import FindOptions as NewFindOptions
from ..action_config import ActionConfig, ActionConfigBuilder
from ..action_interface import ActionInterface
from ..action_result import ActionResult
from ..action_type import ActionType
from ..basic.click.click import Click
from ..basic.click.click_options import ClickOptions, ClickOptionsBuilder
from ..basic.find.options.pattern_find_options import PatternFindOptions
from ..object_collection import ObjectCollection

logger = logging.getLogger(__name__)


class FindAndClickOptions(ActionConfig):
    """Configuration for FindAndClick composite action.

    Port of FindAndClick from Qontinui framework class.

    Combines Find and Click configurations for convenience.
    This class is immutable and must be constructed using FindAndClickOptionsBuilder.

    Example usage:
        options = FindAndClickOptionsBuilder()
            .set_find_options(PatternFindOptions())
            .set_similarity(0.8)
            .set_number_of_clicks(2)
            .build()
    """

    def __init__(self, builder: "FindAndClickOptionsBuilder") -> None:
        """Initialize FindAndClickOptions from builder.

        Args:
            builder: The builder instance containing configuration values
        """
        super().__init__(builder)
        self._find_options: PatternFindOptions = builder.find_options
        self._click_options: ClickOptions = builder.click_options

    def get_find_options(self) -> PatternFindOptions:
        """Get find options.

        Returns:
            PatternFindOptions instance
        """
        return self._find_options

    def get_click_options(self) -> ClickOptions:
        """Get click options.

        Returns:
            ClickOptions instance
        """
        return self._click_options


class FindAndClickOptionsBuilder(ActionConfigBuilder):
    """Builder for constructing FindAndClickOptions with a fluent API.

    Port of FindAndClickOptions from Qontinui framework.Builder.
    """

    def __init__(self, original: FindAndClickOptions | None = None) -> None:
        """Initialize builder.

        Args:
            original: Optional FindAndClickOptions instance to copy values from
        """
        super().__init__(original)

        if original:
            self.find_options = original._find_options
            self.click_options = original._click_options
        else:
            self.find_options = PatternFindOptions()
            self.click_options = ClickOptionsBuilder().build()

    def set_find_options(self, options: PatternFindOptions) -> "FindAndClickOptionsBuilder":
        """Set find options.

        Args:
            options: Find configuration

        Returns:
            This builder instance for chaining
        """
        self.find_options = options
        return self

    def set_click_options(self, options: ClickOptions) -> "FindAndClickOptionsBuilder":
        """Set click options.

        Args:
            options: Click configuration

        Returns:
            This builder instance for chaining
        """
        self.click_options = options
        return self

    def set_similarity(self, similarity: float) -> "FindAndClickOptionsBuilder":
        """Set similarity threshold for find operation.

        Convenience method that creates a new find options with updated similarity.

        Args:
            similarity: Minimum similarity score (0.0 to 1.0)

        Returns:
            This builder instance for chaining
        """
        # Create new PatternFindOptions with updated similarity instead of mutating
        self.find_options.similarity = similarity
        return self

    def set_number_of_clicks(self, count: int) -> "FindAndClickOptionsBuilder":
        """Set number of clicks.

        Convenience method that creates a new click options with updated click count.

        Args:
            count: Number of clicks

        Returns:
            This builder instance for chaining
        """
        # Create new ClickOptions with updated click count instead of mutating
        self.click_options = (
            ClickOptionsBuilder(self.click_options).set_number_of_clicks(count).build()
        )
        return self

    def build(self) -> FindAndClickOptions:
        """Build the immutable FindAndClickOptions object.

        Returns:
            A new instance of FindAndClickOptions
        """
        return FindAndClickOptions(self)

    def _self(self) -> "FindAndClickOptionsBuilder":
        """Return self for fluent interface.

        Returns:
            This builder instance
        """
        return self


@dataclass
class FindAndClick(ActionInterface):
    """Composite action that finds an element then clicks it.

    Port of FindAndClick from Qontinui framework functionality.

    Provides convenient way to configure the common pattern of
    finding an element and clicking on it.
    """

    find_action: FindAction
    click: Click

    def get_action_type(self) -> ActionType:
        """Get action type.

        Returns:
            FIND_AND_CLICK action type
        """
        return ActionType.FIND_AND_CLICK  # type: ignore[no-any-return, attr-defined]

    def perform(self, matches: ActionResult, *object_collections: ObjectCollection) -> None:
        """Find element and click on it.

        Executes:
        1. Find operation to locate target
        2. Click operation on found element

        Args:
            matches: ActionResult with configuration
            object_collections: Objects to find and click
        """
        # Get configuration
        if isinstance(matches.action_config, FindAndClickOptions):
            options = matches.action_config
            find_options_config = options.get_find_options()
            click_options = options.get_click_options()
        else:
            # Use defaults if not FindAndClickOptions
            find_options_config = PatternFindOptions()
            click_options = ClickOptionsBuilder().build()

        # Step 1: Find the target using FindAction
        found_matches = []
        for obj_coll in object_collections:
            for state_image in obj_coll.state_images:
                pattern = state_image.get_pattern()
                if pattern:
                    find_opts = NewFindOptions(
                        similarity=find_options_config.similarity,
                        find_all=True,
                    )
                    result = self.find_action.find(pattern, find_opts)
                    if result.found:
                        for match in result.matches:
                            found_matches.append(match)
                            matches.add_match(match)  # type: ignore[attr-defined]

        if not found_matches:
            object.__setattr__(matches, "success", False)
            logger.debug("FindAndClick: No matches found")
            return

        # Step 2: Click on found element(s)
        click_result = ActionResult(action_config=click_options)  # type: ignore[call-arg]
        # Pass the matches as an ObjectCollection for clicking
        click_collection = ObjectCollection()
        for match in found_matches:
            click_collection.add_match(match)  # type: ignore[attr-defined]

        self.click.perform(click_result, click_collection)

        # Update success status
        object.__setattr__(matches, "success", click_result.success)

        logger.debug(
            f"FindAndClick: Found {len(found_matches)} matches, "
            f"clicked {'successfully' if matches.success else 'unsuccessfully'}"
        )
