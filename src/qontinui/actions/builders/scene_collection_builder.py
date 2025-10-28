"""Scene collection builder for ObjectCollection.

Handles Scenes for offline processing.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...model.element import Pattern, Scene


class SceneCollectionBuilder:
    """Builder for scene-related objects in ObjectCollection.

    Handles:
    - Scenes
    - String filenames (converted to Scenes)
    - Patterns (converted to Scenes)
    """

    def __init__(self) -> None:
        """Initialize builder with empty list."""
        self.scenes: list["Scene"] = []

    def with_scenes(self, *scenes) -> "SceneCollectionBuilder":
        """Add scenes to collection.

        Args:
            scenes: Variable number of string, Pattern, Scene objects or list

        Returns:
            This builder for method chaining
        """
        from ...model.element import Pattern, Scene

        for item in scenes:
            if isinstance(item, str):
                self.scenes.append(Scene(filename=item))
            elif isinstance(item, Pattern):
                self.scenes.append(Scene(pattern=item))
            elif isinstance(item, Scene):
                self.scenes.append(item)
            elif isinstance(item, list):
                self.scenes.extend(item)
        return self

    def build(self) -> list["Scene"]:
        """Build and return the scenes list.

        Returns:
            Copy of scenes list
        """
        return self.scenes.copy()
