"""Match serialization and conversion utilities.

Provides methods for converting Match objects to other formats.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from .match import Match
    from qontinui.model.state.state_image import StateImage


class MatchSerializer:
    """Serializes and converts Match objects."""

    @staticmethod
    def to_string(match: Match) -> str:
        """Convert match to string representation.

        Args:
            match: Match to convert

        Returns:
            String representation
        """
        parts = ["M["]

        if match.name:
            parts.append(f"#{match.name}# ")

        region = match.get_region()
        if region:
            parts.append(
                f"R[{region.x},{region.y} {region.width}x{region.height}] simScore:{match.score:.1f}"
            )
        else:
            parts.append(f"R[null] simScore:{match.score:.1f}")

        if match.ocr_text:
            parts.append(f" ocr_text:{match.ocr_text}")

        parts.append("]")
        return "".join(parts)

    @staticmethod
    def to_state_image(match: Match) -> StateImage:
        """Convert match to StateImage.

        If there is a StateObject, we try to recreate it as a StateImage.

        Args:
            match: Match to convert

        Returns:
            StateImage created from this match
        """
        from ..element.pattern import Pattern
        from ..state.state_image import StateImage

        # Create a Pattern from this match
        pattern = Pattern.from_match(match)
        state_image = StateImage(image=pattern, name=match.name)

        if match.metadata.state_object_data:
            state_image.owner_state_name = match.metadata.state_object_data.owner_state_name  # type: ignore[misc]
            if match.metadata.state_object_data.state_object_name:
                state_image.name = match.metadata.state_object_data.state_object_name

        return state_image

    @staticmethod
    def get_owner_state_name(match: Match) -> str:
        """Get the name of the owner state.

        Args:
            match: Match to extract owner state name from

        Returns:
            Owner state name or empty string
        """
        if match.metadata.state_object_data:
            return cast(str, match.metadata.state_object_data.owner_state_name)
        return ""
