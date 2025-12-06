"""State factory - ported from Qontinui framework.

Factory for creating states after framework initialization.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from ..model.state.state import State, StateBuilder

if TYPE_CHECKING:
    from ..model.state.state_image import StateImage
    from ..model.state.state_string import StateString

logger = logging.getLogger(__name__)


class StateFactory:
    """Factory for creating states after framework initialization.

    Port of StateFactory from Qontinui framework class.

    This factory ensures states are only created after the framework is ready.
    """

    def create_state(
        self, state_name: str, initializer: Callable[[StateBuilder], None] | None = None
    ) -> State:
        """Create a new State with the given name and optional initialization logic.

        This method can be safely called at any time as it doesn't perform any
        framework-dependent operations until the state is actually used.

        Args:
            state_name: The state name
            initializer: Optional function to initialize the state with components

        Returns:
            A new State instance
        """
        logger.debug(f"Creating state: {state_name}")

        builder = StateBuilder(state_name)

        if initializer:
            initializer(builder)

        return builder.build()

    def create_state_image(self, *image_names: str) -> StateImage:
        """Create a StateImage with the given names.

        This is a lightweight operation that doesn't load the actual image.

        Args:
            *image_names: Names of the images to include

        Returns:
            A new StateImage instance
        """
        from ..model.state.state_image import StateImageBuilder  # type: ignore[attr-defined]

        builder = StateImageBuilder()  # type: ignore[attr-defined]
        for image_name in image_names:
            builder.add_pattern(image_name)
        return cast("StateImage", builder.build())

    def create_state_string(
        self, value: str, name: str | None = None, owner_state_name: str | None = None
    ) -> StateString:
        """Create a StateString with the given string value.

        Args:
            value: The string value
            name: Optional name of the StateString
            owner_state_name: Optional name of the owner state

        Returns:
            A new StateString instance
        """
        from ..model.state.state_string import StateStringBuilder  # type: ignore[attr-defined]

        builder = StateStringBuilder()  # type: ignore[attr-defined]
        builder.set_string(value)

        if name:
            builder.set_name(name)
        if owner_state_name:
            builder.set_owner_state_name(owner_state_name)

        return cast("StateString", builder.build())
