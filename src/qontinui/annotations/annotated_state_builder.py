"""Annotated state builder - ported from Qontinui framework.

Builds State objects from @state annotated classes.
"""

import logging
from typing import Any, cast

from ..model.state.state import State
from ..model.state.state_location import StateLocation
from ..model.state.state_region import StateRegion
from .state_component_extractor import StateComponentExtractor, StateComponents

logger = logging.getLogger(__name__)


class AnnotatedStateBuilder:
    """Builds Brobot State objects from @state annotated classes.

    Port of AnnotatedStateBuilder from Qontinui framework.

    This class is responsible solely for constructing State objects from
    annotated classes and their extracted components. It does not handle
    component extraction or state registration, following the Single
    Responsibility Principle.

    The builder creates a proper State object with:
    - A generated StateEnum based on the class name
    - All extracted StateImage, StateString, and StateObject components
    - Proper state configuration from the @state annotation
    """

    def __init__(self, component_extractor: StateComponentExtractor):
        """Initialize the builder.

        Args:
            component_extractor: Extractor for state components
        """
        self.component_extractor = component_extractor

    def build_state(self, state_instance: Any, state_metadata: dict[str, Any]) -> State:
        """Build a State object from an annotated state instance.

        Args:
            state_instance: The instance of a @state annotated class
            state_metadata: The metadata from the @state decorator

        Returns:
            A fully constructed State object ready for registration
        """
        state_class = state_instance.__class__
        state_name = state_metadata.get("name", "") or self._derive_state_name(state_class)

        logger.debug(f"Building state '{state_name}' from class {state_class.__name__}")

        # Extract components from the state instance
        components = self.component_extractor.extract_components(state_instance)

        # Build the State object using the StateBuilder constructor
        from ..model.state.state import StateBuilder

        state_builder = StateBuilder(state_name)

        # Add all extracted components
        if components.state_images:
            state_builder.with_images(*components.state_images)
            logger.debug(
                f"Added {len(components.state_images)} StateImages to state '{state_name}'"
            )

        if components.state_strings:
            state_builder.with_strings(*components.state_strings)
            logger.debug(
                f"Added {len(components.state_strings)} StateStrings to state '{state_name}'"
            )

        # Note: State.Builder doesn't support StateObjects directly in Brobot
        # They would need to be converted to StateImages or StateStrings
        if components.state_objects:
            logger.warning(
                f"Found {len(components.state_objects)} StateObjects in state '{state_name}' "
                "but State.Builder doesn't support them directly"
            )

        state = state_builder.build()
        logger.info(
            f"Built state '{state_name}' with {components.get_total_components()} total components"
        )

        # Set the owner state for all extracted components after State is built
        self._set_owner_state_for_components(components, state)

        return cast(State, state)

    def _set_owner_state_for_components(self, components: StateComponents, state: State) -> None:
        """Set the owner state for all components.

        This ensures that all StateObjects know which state they belong to,
        which is essential for features like cross-state search region resolution.

        Args:
            components: Extracted state components
            state: The owning State object
        """
        # Set owner state for all StateImages
        for state_image in components.state_images:
            previous_owner = state_image.owner_state_name if state_image.owner_state else None
            state_image.owner_state_name = state.name  # type: ignore[misc]
            logger.info(
                f"Set owner state '{state.name}' for StateImage '{state_image.name}' "
                f"(was: '{previous_owner}')"
            )

        # Set owner state for all StateStrings
        for state_string in components.state_strings:
            state_string.owner_state = state
            logger.debug(f"Set owner state '{state.name}' for StateString '{state_string.name}'")

        # Set owner state for all StateObjects (handle specific types)
        for state_object in components.state_objects:
            # StateObject is a base class - check for specific implementations
            if isinstance(state_object, StateLocation):
                state_object.owner_state = state
                logger.debug(
                    f"Set owner state '{state.name}' for StateLocation '{state_object.name}'"
                )
            elif isinstance(state_object, StateRegion):
                state_object.owner_state = state
                logger.debug(
                    f"Set owner state '{state.name}' for StateRegion '{state_object.name}'"
                )
            else:
                logger.debug(
                    f"StateObject '{state_object.get_name()}' type {state_object.__class__.__name__} "
                    "does not support owner state"
                )

        logger.debug(
            f"Set owner state '{state.name}' for {components.get_total_components()} components"
        )

    def _derive_state_name(self, state_class: type) -> str:
        """Derive the state name from the class.

        Args:
            state_class: State class

        Returns:
            Derived state name
        """
        class_name = state_class.__name__

        # Remove "State" suffix if present
        if class_name.endswith("State"):
            return class_name[:-5]

        return class_name
