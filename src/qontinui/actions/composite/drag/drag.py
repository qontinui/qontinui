"""Drag composite action - ported from Qontinui framework.

Implements drag-and-drop functionality using action chaining.
"""

from ...action_chain_options import ActionChainOptions, ActionChainOptionsBuilder, ChainingStrategy
from ...action_config import ActionConfig
from ...action_interface import ActionInterface
from ...action_result import ActionResult
from ...action_type import ActionType
from ...basic.find.pattern_find_options import PatternFindOptions, PatternFindOptionsBuilder
from ...basic.mouse.mouse_down_options import MouseDownOptions, MouseDownOptionsBuilder
from ...basic.mouse.mouse_move_options import MouseMoveOptions, MouseMoveOptionsBuilder
from ...basic.mouse.mouse_press_options import MousePressOptions
from ...basic.mouse.mouse_up_options import MouseUpOptions, MouseUpOptionsBuilder
from ...internal.execution.action_chain_executor import ActionChainExecutor
from ...internal.service.action_service import ActionService
from ...object_collection import ObjectCollection
from .drag_options import DragOptions


class Drag(ActionInterface):
    """Composite action that drags from source to target location.

    Port of Drag from Qontinui framework action.

    Drag is a composite action that chains together multiple atomic actions
    to perform a drag-and-drop operation. The typical sequence is:
    1. Find the source location
    2. Find the target location
    3. Move mouse to source
    4. Press mouse button down
    5. Move mouse to target (while button held)
    6. Release mouse button

    This implementation uses the ActionChainExecutor with a fluent API
    to ensure proper sequencing and error handling across all steps.
    """

    def __init__(
        self,
        action_chain_executor: ActionChainExecutor | None = None,
        action_service: ActionService | None = None,
    ) -> None:
        """Initialize Drag action.

        Args:
            action_chain_executor: Executor for action chains
            action_service: Service for resolving actions
        """
        self.action_chain_executor = action_chain_executor or ActionChainExecutor()
        self.action_service = action_service
        if self.action_chain_executor.action_service is None:
            self.action_chain_executor.action_service = action_service

    def get_action_type(self) -> ActionType:
        """Return the action type.

        Returns:
            ActionType.DRAG
        """
        return ActionType.DRAG

    def perform(self, action_result: ActionResult, *object_collections: ObjectCollection) -> None:
        """Execute the drag operation using action chaining.

        Creates a chain of actions that:
        1. Finds the source location (using first ObjectCollection)
        2. Finds the target location (using second ObjectCollection)
        3. Moves to source
        4. Presses mouse down
        5. Moves to target
        6. Releases mouse

        Args:
            action_result: Result container that will be populated
            object_collections: Variable number of object collections.
                               First collection defines source, second defines target.
        """
        if not isinstance(action_result.action_config, DragOptions):
            object.__setattr__(action_result, "success", False)
            return

        drag_options = action_result.action_config

        # Ensure we have at least 2 object collections (source and target)
        if len(object_collections) < 2:
            object.__setattr__(action_result, "success", False)
            object.__setattr__(
                action_result,
                "output_text",
                "Drag requires at least 2 object collections (source and target)",
            )
            return

        source_collection = object_collections[0]
        target_collection = object_collections[1]

        # Build the action chain using fluent API
        chain_options = self._build_action_chain(drag_options, source_collection, target_collection)

        # Execute the chain
        chain_result = self.action_chain_executor.execute_chain(
            chain_options, action_result, source_collection, target_collection
        )

        # Copy results back to the provided action_result
        self._copy_results(chain_result, action_result)

    def _build_action_chain(
        self,
        drag_options: DragOptions,
        source_collection: ObjectCollection,
        target_collection: ObjectCollection,
    ) -> ActionChainOptions:
        """Build the 6-step action chain for drag operation.

        Args:
            drag_options: Configuration for the drag
            source_collection: Collection defining source location
            target_collection: Collection defining target location

        Returns:
            ActionChainOptions configured with the complete drag sequence
        """
        # Step 1: Find source location
        find_source_options = self._create_find_options(
            drag_options.get_find_source_options(), source_collection
        )

        # Step 2: Find target location
        find_target_options = self._create_find_options(
            drag_options.get_find_target_options(), target_collection
        )

        # Step 3: Move to source
        move_to_source_options = self._create_move_options(
            drag_options.get_move_to_source_options()
        )

        # Step 4: Mouse down
        mouse_down_options = self._create_mouse_down_options(drag_options.get_mouse_down_options())

        # Step 5: Move to target
        move_to_target_options = self._create_move_options(
            drag_options.get_move_to_target_options()
        )

        # Step 6: Mouse up
        mouse_up_options = self._create_mouse_up_options(drag_options.get_mouse_up_options())

        # Build the chain using fluent API
        chain_options = (
            ActionChainOptionsBuilder(find_source_options)
            .set_strategy(ChainingStrategy.NESTED)
            .then(find_target_options)
            .then(move_to_source_options)
            .then(mouse_down_options)
            .then(move_to_target_options)
            .then(mouse_up_options)
            .build()
        )

        return chain_options

    def _create_find_options(
        self, base_options: ActionConfig, collection: ObjectCollection
    ) -> PatternFindOptions:
        """Create find options for source or target.

        Args:
            base_options: Base configuration to use
            collection: Object collection to search in

        Returns:
            Configured PatternFindOptions
        """
        # If base_options is already PatternFindOptions, use it
        if isinstance(base_options, PatternFindOptions):
            return base_options

        # Create default find options
        from ...basic.find.do_on_each import DoOnEach

        return PatternFindOptionsBuilder().set_do_on_each(DoOnEach.FIRST).build()

    def _create_move_options(self, base_options: ActionConfig) -> MouseMoveOptions:
        """Create mouse move options.

        Args:
            base_options: Base configuration to use

        Returns:
            Configured MouseMoveOptions
        """
        # If base_options is already MouseMoveOptions, use it
        if isinstance(base_options, MouseMoveOptions):
            return base_options

        # Create default move options
        return MouseMoveOptionsBuilder().build()

    def _create_mouse_down_options(self, base_options: MousePressOptions) -> MouseDownOptions:
        """Create mouse down options.

        Args:
            base_options: Base configuration to use (MousePressOptions)

        Returns:
            Configured MouseDownOptions
        """
        # If base_options is already MouseDownOptions, use it
        if isinstance(base_options, MouseDownOptions):
            return base_options

        # Create default mouse down options
        return MouseDownOptionsBuilder().build()

    def _create_mouse_up_options(self, base_options: MousePressOptions) -> MouseUpOptions:
        """Create mouse up options.

        Args:
            base_options: Base configuration to use (MousePressOptions)

        Returns:
            Configured MouseUpOptions
        """
        # If base_options is already MouseUpOptions, use it
        if isinstance(base_options, MouseUpOptions):
            return base_options

        # Create default mouse up options
        return MouseUpOptionsBuilder().build()

    def _copy_results(self, source: ActionResult, target: ActionResult) -> None:
        """Copy results from chain execution to the target result.

        Args:
            source: Source result from chain execution
            target: Target result to populate
        """
        object.__setattr__(target, "success", source.is_success)
        object.__setattr__(target, "matches", source.matches)
        object.__setattr__(target, "duration", source.duration)
        object.__setattr__(target, "text", source.text)
        object.__setattr__(target, "active_states", source.active_states)

        # Copy movements
        for movement in source.movements:
            target.add_movement(movement)  # type: ignore[attr-defined]

        # Copy execution history
        for record in source.execution_history:
            target.add_execution_record(record)  # type: ignore[attr-defined]
