"""Scene class - ported from Qontinui framework.

Represents a captured screenshot or screen state as a searchable pattern.
"""

from dataclasses import dataclass

from .pattern import Pattern


@dataclass
class Scene:
    """Represents a captured screenshot or screen state as a searchable pattern.

    Port of Scene from Qontinui framework class.

    Scene encapsulates a full or partial screenshot that serves as a reference
    image for pattern matching operations. Unlike individual Pattern objects that
    typically represent specific UI elements, a Scene captures a broader view of
    the application state, providing context for finding multiple patterns within
    a single screen capture.

    Key characteristics:
    - Screenshot Storage: Holds a complete or partial screen capture
    - Pattern Container: Wraps the screenshot as a searchable Pattern
    - Persistent Identity: Maintains database ID for tracking and logging
    - Context Provider: Offers broader view than individual elements

    Common use cases:
    - Storing reference screenshots for state verification
    - Creating mock environments from captured scenes
    - Analyzing screen layouts and element positions
    - Debugging by comparing expected vs actual scenes
    - Building training data for pattern recognition
    """

    pattern: Pattern
    id: int = -1

    def __init__(self, pattern: Pattern | None = None, filename: str | None = None):
        """Initialize Scene with pattern or filename.

        Args:
            pattern: Pattern object containing the scene image
            filename: Path to scene image file
        """
        if pattern is not None:
            self.pattern = pattern
        elif filename is not None:
            from .image import Image

            image = Image(name=filename, path=filename)
            self.pattern = Pattern(image=image, name=filename)
        else:
            from .image import Image

            self.pattern = Pattern(image=Image())

        self.id = -1

    @property
    def name(self) -> str:
        """Get scene name from pattern.

        Returns:
            Pattern name or empty string
        """
        return self.pattern.name if self.pattern else ""

    @property
    def filename(self) -> str:
        """Get scene filename from pattern.

        Returns:
            Pattern path or empty string
        """
        return self.pattern.path if self.pattern else ""

    def has_pattern(self) -> bool:
        """Check if scene has a valid pattern.

        Returns:
            True if pattern exists and has content
        """
        return self.pattern is not None and self.pattern.has_image()

    def __str__(self) -> str:
        """String representation."""
        pattern_name = self.pattern.name if self.pattern else "null"
        return f"Scene{{id={self.id}, pattern={pattern_name}}}"

    def __repr__(self) -> str:
        """Developer representation."""
        return f"Scene(id={self.id}, pattern={self.pattern!r})"

    @classmethod
    def from_screenshot(cls, screenshot_path: str) -> "Scene":
        """Create Scene from screenshot file.

        Args:
            screenshot_path: Path to screenshot file

        Returns:
            New Scene instance
        """
        return cls(filename=screenshot_path)

    @classmethod
    def empty(cls) -> "Scene":
        """Create empty Scene.

        Returns:
            Empty Scene instance
        """
        return cls()
