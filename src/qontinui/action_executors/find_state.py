"""Find State action executor.

This module provides the FindStateExecutor for checking which states
are currently active on screen by finding their associated images.
"""

import logging
from typing import Any

from qontinui import registry

from ..actions.find import FindAction
from ..actions.find.find_options_builder import CascadeContext, build_find_options
from ..config import FindStateActionConfig
from ..config.schema import Action
from ..config.settings import QontinuiSettings
from ..exceptions import ActionExecutionError
from ..model.element import Pattern
from .base import ActionExecutorBase
from .registry import register_executor

logger = logging.getLogger(__name__)


@register_executor
class FindStateExecutor(ActionExecutorBase):
    """Executor for FIND_STATE action.

    Checks which states are currently active on screen by searching for
    images associated with each state. Performs a single FIND ALL operation
    on all images from selected states, then maps found images back to
    their owning states.

    This action is useful for:
    - Detecting current application state before navigation
    - Verifying state after transitions
    - Implementing conditional logic based on active states

    Example:
        >>> context = ExecutionContext(...)
        >>> executor = FindStateExecutor(context)
        >>> action = Action(type="FIND_STATE", config={"stateIds": ["state1", "state2"]})
        >>> config = FindStateActionConfig(state_ids=["state1", "state2"])
        >>> executor.execute(action, config)
        True  # Returns True if at least one state is active
    """

    def get_supported_action_types(self) -> list[str]:
        """Get list of action types this executor handles.

        Returns:
            List containing ["FIND_STATE"]
        """
        return ["FIND_STATE"]

    async def execute(self, action: Action, typed_config: Any) -> bool:
        """Execute FIND_STATE action.

        Args:
            action: Pydantic Action model
            typed_config: Pre-validated FindStateActionConfig

        Returns:
            bool: True if at least one state is active, False if none found

        Raises:
            ActionExecutionError: If execution fails critically
        """
        config: FindStateActionConfig = typed_config
        logger.debug(f"Executing FIND_STATE action: {action.id}")

        state_ids = config.state_ids
        logger.debug(f"Checking {len(state_ids)} state(s) for visibility")

        if not state_ids:
            raise ActionExecutionError(
                action_type="FIND_STATE",
                reason="stateIds is required and must not be empty",
                action_id=action.id,
            )

        # Collect all images from all target states, tracking ownership
        # Structure: {image_id: state_id}
        image_to_state: dict[str, str] = {}

        # Get state_map from context.config
        state_map = getattr(self.context.config, "state_map", None)
        if not state_map:
            logger.warning("No state_map available in context.config")
            # Fall back to looking up images directly from registry if available
            # This allows the action to work even without full state model
            return self._execute_without_state_map(action, config)

        for state_id in state_ids:
            state = state_map.get(state_id)
            if not state:
                logger.warning(f"State '{state_id}' not found in state_map")
                continue

            # Get all StateImages for this state
            for state_image in state.state_images:
                # Try to get the image ID from the StateImage
                # StateImage may have multiple patterns (via registry)
                image_id = getattr(state_image, "id", None)
                if image_id:
                    # Check if this is a StateImage with multiple patterns
                    pattern_ids = registry.get_state_image_pattern_ids(image_id)
                    if pattern_ids:
                        for pattern_id in pattern_ids:
                            image_to_state[pattern_id] = state_id
                    else:
                        image_to_state[image_id] = state_id
                elif state_image.image:
                    # Use image name or generate ID
                    img_name = state_image.name or getattr(state_image.image, "name", None)
                    if img_name:
                        image_to_state[img_name] = state_id

        if not image_to_state:
            logger.warning("No images found for any of the specified states")
            self._emit_action_failure(
                action,
                "No images found for specified states",
                {"state_ids": state_ids},
            )
            return False

        logger.debug(f"Collected {len(image_to_state)} image(s) across {len(state_ids)} state(s)")

        # Perform FIND ALL operation on all images
        active_states = await self._find_all_images(image_to_state, config, action)

        # Store result
        if config.output_variable and active_states:
            self.context.variable_context.set_variable(config.output_variable, list(active_states))

        # Emit success event with results
        self._emit_action_success(
            action,
            {
                "active_states": list(active_states),
                "checked_states": state_ids,
                "images_searched": len(image_to_state),
            },
        )

        logger.info(f"FIND_STATE complete: {len(active_states)}/{len(state_ids)} states active")

        return len(active_states) > 0

    def _execute_without_state_map(self, action: Action, config: FindStateActionConfig) -> bool:
        """Execute FIND_STATE when state_map is not available.

        Falls back to searching for images registered under the state IDs.

        Args:
            action: The action being executed
            config: Action configuration

        Returns:
            bool: True if at least one state has visible images
        """
        logger.debug("Executing FIND_STATE without state_map (fallback mode)")

        # In fallback mode, we can't determine images from state model
        # This is a limitation - the action requires proper state configuration
        self._emit_action_failure(
            action,
            "FIND_STATE requires state configuration to be loaded",
            {"state_ids": config.state_ids},
        )
        return False

    async def _find_all_images(
        self,
        image_to_state: dict[str, str],
        config: FindStateActionConfig,
        action: Action,
    ) -> set[str]:
        """Find all images and return which states are active.

        Performs a FIND ALL operation on all images and maps found images
        back to their owning states.

        Args:
            image_to_state: Mapping of image_id to state_id
            config: Action configuration with search options
            action: The action being executed

        Returns:
            Set of active state IDs (states with at least one visible image)
        """
        active_states: set[str] = set()
        find_action = FindAction()

        # Get project config for cascade
        try:
            project_config = QontinuiSettings()
        except Exception:
            project_config = None
            logger.debug("Could not load project config, using defaults")

        # Get monitor from context if available
        monitor_index = getattr(self.context, "monitor_index", None)

        # Collect all patterns for parallel search
        patterns_with_info = []  # List of (pattern, image_id, state_id)

        for image_id, state_id in image_to_state.items():
            # Skip if state already marked as active (optimization)
            if state_id in active_states:
                continue

            # Get image from registry
            image = registry.get_image(image_id)
            if image is None:
                logger.debug(f"Image '{image_id}' not found in registry, skipping")
                continue

            metadata = registry.get_image_metadata(image_id)
            if metadata is None:
                logger.debug(f"Metadata not found for image '{image_id}', skipping")
                continue

            file_path = metadata.get("file_path")
            if not file_path:
                logger.debug(f"No file_path in metadata for image '{image_id}'")
                continue

            # Create pattern
            pattern = Pattern.from_file(
                img_path=file_path,
                name=metadata.get("name", image_id),
            )
            patterns_with_info.append((pattern, image_id, state_id))

        if not patterns_with_info:
            return active_states

        # Build find options with cascade (use first pattern for context)
        first_pattern = patterns_with_info[0][0]
        ctx = CascadeContext(
            search_options=config.search_options,
            pattern=first_pattern,
            state_image=None,
            project_config=project_config,
            monitor_index=monitor_index,
        )
        options = build_find_options(ctx)

        # Perform parallel find for all patterns
        patterns = [p[0] for p in patterns_with_info]
        results = await find_action.find(patterns, options)

        # Map results back to states
        for result, (_, image_id, state_id) in zip(results, patterns_with_info, strict=False):
            if result.found:
                logger.debug(f"Image '{image_id}' found -> state '{state_id}' is active")
                active_states.add(state_id)

        return active_states
