"""Image collection builder for ObjectCollection.

Handles StateImages and Patterns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...model.state import State, StateImage


class ImageCollectionBuilder:
    """Builder for image-related objects in ObjectCollection.

    Handles:
    - StateImages
    - Patterns (converted to StateImages)
    - State images from States
    """

    def __init__(self) -> None:
        """Initialize builder with empty list."""
        self.state_images: list[StateImage] = []

    def with_images(self, *state_images) -> ImageCollectionBuilder:
        """Add state images to collection.

        Args:
            state_images: Variable number of StateImage objects or list

        Returns:
            This builder for method chaining
        """
        for item in state_images:
            if isinstance(item, list):
                self.state_images.extend(item)
            else:
                self.state_images.append(item)
        return self

    def set_images(self, state_images: list[StateImage]) -> ImageCollectionBuilder:
        """Set state images list.

        Args:
            state_images: List of StateImage objects

        Returns:
            This builder for method chaining
        """
        self.state_images = state_images
        return self

    def with_patterns(self, *patterns) -> ImageCollectionBuilder:
        """Add patterns as state images.

        Args:
            patterns: Variable number of Pattern objects or list

        Returns:
            This builder for method chaining
        """
        for item in patterns:
            if isinstance(item, list):
                for pattern in item:
                    self.state_images.append(pattern.in_null_state())
            else:
                self.state_images.append(item.in_null_state())
        return self

    def with_all_state_images(self, state: State | None) -> ImageCollectionBuilder:
        """Add all state images from a state.

        Args:
            state: State to get images from

        Returns:
            This builder for method chaining
        """
        if state is not None:
            self.state_images.extend(state.get_state_images())
        return self

    def with_non_shared_images(self, state: State | None) -> ImageCollectionBuilder:
        """Add non-shared state images from a state.

        Args:
            state: State to get images from

        Returns:
            This builder for method chaining
        """
        if state is not None:
            for state_image in state.get_state_images():
                if not state_image.is_shared:
                    self.state_images.append(state_image)
        return self

    def build(self) -> list[StateImage]:
        """Build and return the state images list.

        Returns:
            Copy of state images list
        """
        return self.state_images.copy()
