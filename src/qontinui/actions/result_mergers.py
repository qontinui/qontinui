"""Merging utilities for combining ActionResult instances.

Provides operations for merging matches and data from multiple ActionResult
instances into a single result.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .action_result import ActionResult


class ResultMerger:
    """Utility class for merging ActionResult instances.

    Provides methods for combining matches and non-match data from
    multiple results. Since ActionResult is immutable, these methods
    return new ActionResult instances.
    """

    @staticmethod
    def add_match_objects(
        target: "ActionResult", source: "ActionResult"
    ) -> "ActionResult":
        """Merge match objects from source into target.

        Args:
            target: ActionResult to merge into
            source: ActionResult containing matches to add

        Returns:
            A new ActionResult with merged matches
        """
        from .action_result import ActionResultBuilder

        builder = ActionResultBuilder(target.action_config)
        builder.with_success(target.success)

        # Add all matches from target
        for match in target.matches:
            builder.add_match(match)

        # Add all matches from source
        if source:
            for match in source.matches:
                builder.add_match(match)

        # Preserve other target fields
        if target.text:
            builder.add_text(target.text)
        if target.selected_text:
            builder.with_selected_text(target.selected_text)
        if target.action_description:
            builder.with_description(target.action_description)
        if target.output_text:
            builder.with_output_text(target.output_text)

        for region in target.defined_regions:
            builder.add_defined_region(region)
        for movement in target.movements:
            builder.add_movement(movement)
        for record in target.execution_history:
            builder.add_execution_record(record)

        builder.set_times_acted_on(target.times_acted_on)
        builder.with_timing(target.start_time, target.end_time, target.duration)

        return builder.build()

    @staticmethod
    def add_non_match_results(
        target: "ActionResult", source: "ActionResult"
    ) -> "ActionResult":
        """Merge non-match data from source into target.

        This includes text, states, regions, movements, and execution history.

        Args:
            target: ActionResult to merge into
            source: ActionResult containing data to merge

        Returns:
            A new ActionResult with merged non-match data
        """
        from .action_result import ActionResultBuilder

        builder = ActionResultBuilder(target.action_config)
        builder.with_success(target.success)

        # Preserve all matches from target
        for match in target.matches:
            builder.add_match(match)

        # Merge text fields (source overrides target if present)
        if source and source.text:
            builder.add_text(source.text)
        elif target.text:
            builder.add_text(target.text)

        if source and source.selected_text:
            builder.with_selected_text(source.selected_text)
        elif target.selected_text:
            builder.with_selected_text(target.selected_text)

        # Preserve target fields
        if target.action_description:
            builder.with_description(target.action_description)
        if target.output_text:
            builder.with_output_text(target.output_text)

        # Merge collections from target
        for region in target.defined_regions:
            builder.add_defined_region(region)
        for movement in target.movements:
            builder.add_movement(movement)
        for record in target.execution_history:
            builder.add_execution_record(record)

        # Merge collections from source
        if source:
            for region in source.defined_regions:
                builder.add_defined_region(region)
            for movement in source.movements:
                builder.add_movement(movement)
            for record in source.execution_history:
                builder.add_execution_record(record)

        builder.set_times_acted_on(target.times_acted_on)
        builder.with_timing(target.start_time, target.end_time, target.duration)

        return builder.build()

    @staticmethod
    def add_all_results(
        target: "ActionResult", source: "ActionResult"
    ) -> "ActionResult":
        """Merge all data from source into target.

        Combines both match objects and non-match data.

        Args:
            target: ActionResult to merge into
            source: ActionResult to merge completely

        Returns:
            A new ActionResult with all merged data
        """
        if not source:
            return target

        from .action_result import ActionResultBuilder

        builder = ActionResultBuilder(target.action_config)
        builder.with_success(target.success or source.success)

        # Add all matches from both
        for match in target.matches:
            builder.add_match(match)
        for match in source.matches:
            builder.add_match(match)

        # Merge text fields (source overrides target if present)
        if source.text:
            builder.add_text(source.text)
        elif target.text:
            builder.add_text(target.text)

        if source.selected_text:
            builder.with_selected_text(source.selected_text)
        elif target.selected_text:
            builder.with_selected_text(target.selected_text)

        # Preserve target fields
        if target.action_description:
            builder.with_description(target.action_description)
        if target.output_text:
            builder.with_output_text(target.output_text)

        # Merge collections from both
        for region in target.defined_regions:
            builder.add_defined_region(region)
        for region in source.defined_regions:
            builder.add_defined_region(region)

        for movement in target.movements:
            builder.add_movement(movement)
        for movement in source.movements:
            builder.add_movement(movement)

        for record in target.execution_history:
            builder.add_execution_record(record)
        for record in source.execution_history:
            builder.add_execution_record(record)

        builder.set_times_acted_on(max(target.times_acted_on, source.times_acted_on))
        builder.with_timing(
            target.start_time or source.start_time,
            target.end_time or source.end_time,
            target.duration or source.duration,
        )

        return builder.build()
