"""FindAndClick composite action - ported from Qontinui framework.

Combines Find and Click operations in a single action.
"""

import logging
from dataclasses import dataclass, field

from ...action_interface import ActionInterface
from ...action_result import ActionResult
from ...action_type import ActionType
from ...object_collection import ObjectCollection
from ..basic.action_config import ActionConfig
from ..basic.click.click import Click
from ..basic.click.click_options import ClickOptions, ClickOptionsBuilder
from ..basic.find.find import Find
from ..basic.find.options.pattern_find_options import PatternFindOptions

logger = logging.getLogger(__name__)


@dataclass
class FindAndClickOptions(ActionConfig):
    """Configuration for FindAndClick composite action.

    Port of FindAndClick from Qontinui framework class.

    Combines Find and Click configurations for convenience.
    """

    find_options: PatternFindOptions = field(default_factory=lambda: PatternFindOptions())
    click_options: ClickOptions = field(default_factory=lambda: ClickOptionsBuilder().build())

    def with_find_options(self, options: PatternFindOptions) -> "FindAndClickOptions":
        """Set find options.

        Args:
            options: Find configuration

        Returns:
            Self for fluent interface
        """
        self.find_options = options
        return self

    def with_click_options(self, options: ClickOptions) -> "FindAndClickOptions":
        """Set click options.

        Args:
            options: Click configuration

        Returns:
            Self for fluent interface
        """
        self.click_options = options
        return self

    def with_similarity(self, similarity: float) -> "FindAndClickOptions":
        """Set similarity threshold.

        Convenience method that modifies find options.

        Args:
            similarity: Minimum similarity score (0.0 to 1.0)

        Returns:
            Self for fluent interface
        """
        self.find_options.similarity = similarity
        return self

    def with_number_of_clicks(self, count: int) -> "FindAndClickOptions":
        """Set number of clicks.

        Convenience method that modifies click options.

        Args:
            count: Number of clicks

        Returns:
            Self for fluent interface
        """
        self.click_options.number_of_clicks = count
        return self


@dataclass
class FindAndClick(ActionInterface):
    """Composite action that finds an element then clicks it.

    Port of FindAndClick from Qontinui framework functionality.

    Provides convenient way to configure the common pattern of
    finding an element and clicking on it.
    """

    find: Find
    click: Click

    def get_action_type(self) -> ActionType:
        """Get action type.

        Returns:
            FIND_AND_CLICK action type
        """
        return ActionType.FIND_AND_CLICK

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
            find_options = options.find_options
            click_options = options.click_options
        else:
            # Use defaults if not FindAndClickOptions
            find_options = PatternFindOptions()
            click_options = ClickOptionsBuilder().build()

        # Step 1: Find the target
        find_result = ActionResult(action_config=find_options)
        self.find.perform(find_result, *object_collections)

        # Copy find results to main matches
        matches.match_list = find_result.match_list
        matches.match_locations = find_result.match_locations

        if not find_result.match_list:
            matches.success = False
            logger.debug("FindAndClick: No matches found")
            return

        # Step 2: Click on found element(s)
        click_result = ActionResult(action_config=click_options)
        # Pass the matches as an ObjectCollection for clicking
        click_collection = ObjectCollection()
        for match in find_result.match_list:
            click_collection.add_match(match)

        self.click.perform(click_result, click_collection)

        # Update success status
        matches.success = click_result.success

        logger.debug(
            f"FindAndClick: Found {len(find_result.match_list)} matches, "
            f"clicked {'successfully' if matches.success else 'unsuccessfully'}"
        )
