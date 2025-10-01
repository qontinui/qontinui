"""SemanticObject - Represents a semantically identified object."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, cast

from ...model.element.region import Region
from .pixel_location import PixelLocation


class ObjectType(Enum):
    """Common semantic object types."""

    BUTTON = "button"
    TEXT = "text"
    IMAGE = "image"
    ICON = "icon"
    TEXT_FIELD = "text_field"
    CHECKBOX = "checkbox"
    RADIO_BUTTON = "radio_button"
    DROPDOWN = "dropdown"
    MENU = "menu"
    WINDOW = "window"
    DIALOG = "dialog"
    LIST_ITEM = "list_item"
    LINK = "link"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    TOOLBAR = "toolbar"
    SCROLLBAR = "scrollbar"
    UNKNOWN = "unknown"


@dataclass
class SemanticObject:
    """Represents a single identified object with its location and semantic description.

    Combines precise pixel-level location information with semantic understanding
    of what the object represents, enabling both visual and semantic querying.

    The distinction between fields:
    - description: Semantic description from visual models (e.g., "submit button", "navigation menu")
    - ocr_text: Actual text extracted via OCR from within the object's bounds

    For example, a button might have:
    - description: "blue rectangular button with white text"
    - ocr_text: "Submit Form"
    """

    location: PixelLocation
    """Precise pixel-level location of the object (from segmentation)."""

    description: str
    """Semantic description from visual model (SAM2, CLIP, etc.)."""

    id: str = field(default="")
    """Unique identifier for this object."""

    confidence: float = field(default=1.0)
    """Recognition confidence score (0.0 to 1.0)."""

    object_type: ObjectType = field(default=ObjectType.UNKNOWN)
    """Type classification of the object."""

    ocr_text: str | None = field(default=None)
    """Text extracted via OCR from this object's region."""

    attributes: dict[str, Any] = field(default_factory=dict)
    """Additional properties and metadata."""

    _is_interactable: bool | None = field(default=None, init=False)
    _dominant_color: tuple[Any, ...] | None = field(default=None, init=False)

    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            # Generate ID from location and description
            centroid = self.location.get_centroid()
            self.id = f"obj_{centroid.x}_{centroid.y}_{hash(self.description) % 10000}"

    def get_bounding_box(self) -> Region:
        """Get the bounding box of this object.

        Returns:
            Region that contains the entire object
        """
        return self.location.to_bounding_box()

    def add_attribute(self, key: str, value: Any) -> None:
        """Add or update an attribute.

        Args:
            key: Attribute name
            value: Attribute value
        """
        self.attributes[key] = value

    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Get an attribute value.

        Args:
            key: Attribute name
            default: Default value if attribute not found

        Returns:
            Attribute value or default
        """
        return self.attributes.get(key, default)

    def set_object_type(self, type_name: str) -> None:
        """Set the object type.

        Args:
            type_name: Type name (will be converted to ObjectType enum)
        """
        try:
            self.object_type = ObjectType(type_name.lower())
        except ValueError:
            self.object_type = ObjectType.UNKNOWN
            self.add_attribute("custom_type", type_name)

    def set_interactable(self, value: bool) -> None:
        """Set whether this object is interactable.

        Args:
            value: True if object can be interacted with
        """
        self._is_interactable = value
        self.add_attribute("interactable", value)

    def is_interactable(self) -> bool:
        """Check if this object is interactable.

        Returns:
            True if object can be interacted with
        """
        if self._is_interactable is not None:
            return self._is_interactable

        # Infer from type
        interactable_types = {
            ObjectType.BUTTON,
            ObjectType.LINK,
            ObjectType.TEXT_FIELD,
            ObjectType.CHECKBOX,
            ObjectType.RADIO_BUTTON,
            ObjectType.DROPDOWN,
            ObjectType.MENU,
            ObjectType.LIST_ITEM,
            ObjectType.ICON,
        }
        return self.object_type in interactable_types

    def set_color(self, color: tuple[Any, ...]) -> None:
        """Set the dominant color of the object.

        Args:
            color: RGB tuple (r, g, b)
        """
        self._dominant_color = color
        self.add_attribute("color", color)

    def get_color(self) -> tuple[Any, ...] | None:
        """Get the dominant color.

        Returns:
            RGB tuple or None
        """
        return self._dominant_color

    def set_text(self, text: str) -> None:
        """Set OCR-extracted text for this object.

        Args:
            text: OCR-extracted text content
        """
        self.ocr_text = text
        self.add_attribute("ocr_text", text)

    def get_text(self) -> str | None:
        """Get OCR-extracted text.

        Returns:
            OCR text content or None
        """
        return self.ocr_text

    def is_above(self, other: SemanticObject) -> bool:
        """Check if this object is above another.

        Args:
            other: Other object to compare

        Returns:
            True if this object is above the other
        """
        self_box = self.get_bounding_box()
        other_box = other.get_bounding_box()
        return self_box.y + self_box.height <= other_box.y

    def is_below(self, other: SemanticObject) -> bool:
        """Check if this object is below another.

        Args:
            other: Other object to compare

        Returns:
            True if this object is below the other
        """
        self_box = self.get_bounding_box()
        other_box = other.get_bounding_box()
        return self_box.y >= other_box.y + other_box.height

    def is_left_of(self, other: SemanticObject) -> bool:
        """Check if this object is to the left of another.

        Args:
            other: Other object to compare

        Returns:
            True if this object is to the left of the other
        """
        self_box = self.get_bounding_box()
        other_box = other.get_bounding_box()
        return self_box.x + self_box.width <= other_box.x

    def is_right_of(self, other: SemanticObject) -> bool:
        """Check if this object is to the right of another.

        Args:
            other: Other object to compare

        Returns:
            True if this object is to the right of the other
        """
        self_box = self.get_bounding_box()
        other_box = other.get_bounding_box()
        return self_box.x >= other_box.x + other_box.width

    def contains(self, other: SemanticObject) -> bool:
        """Check if this object contains another.

        Args:
            other: Other object to check

        Returns:
            True if this object fully contains the other
        """
        self_box = self.get_bounding_box()
        other_box = other.get_bounding_box()

        return (
            self_box.x <= other_box.x
            and self_box.y <= other_box.y
            and self_box.x + self_box.width >= other_box.x + other_box.width
            and self_box.y + self_box.height >= other_box.y + other_box.height
        )

    def distance_to(self, other: SemanticObject) -> float:
        """Calculate distance to another object.

        Uses centroid-to-centroid distance.

        Args:
            other: Other object

        Returns:
            Distance in pixels
        """
        self_center = self.location.get_centroid()
        other_center = other.location.get_centroid()

        dx = self_center.x - other_center.x
        dy = self_center.y - other_center.y
        return cast(float, (dx**2 + dy**2) ** 0.5)

    def overlaps(self, other: SemanticObject) -> bool:
        """Check if this object overlaps with another.

        Args:
            other: Other object

        Returns:
            True if objects overlap
        """
        return self.location.overlaps(other.location)

    def get_overlap_percentage(self, other: SemanticObject) -> float:
        """Calculate overlap percentage with another object.

        Args:
            other: Other object

        Returns:
            Percentage of this object that overlaps (0.0 to 1.0)
        """
        return self.location.get_overlap_percentage(other.location)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with object properties
        """
        return {
            "id": self.id,
            "description": self.description,
            "confidence": self.confidence,
            "type": self.object_type.value,
            "bounding_box": {
                "x": self.get_bounding_box().x,
                "y": self.get_bounding_box().y,
                "width": self.get_bounding_box().width,
                "height": self.get_bounding_box().height,
            },
            "centroid": {"x": self.location.get_centroid().x, "y": self.location.get_centroid().y},
            "area": self.location.get_area(),
            "attributes": self.attributes,
            "ocr_text": self.ocr_text,
            "interactable": self.is_interactable(),
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SemanticObject(id='{self.id}', type={self.object_type.value}, "
            f"description='{self.description[:30]}...', confidence={self.confidence:.2f})"
        )

    def __str__(self) -> str:
        """Human-readable string."""
        box = self.get_bounding_box()
        return (
            f"{self.object_type.value.title()}: '{self.description}' "
            f"at ({box.x}, {box.y}) [{box.width}x{box.height}]"
        )
