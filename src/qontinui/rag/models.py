"""RAG data models for GUI element chunking and retrieval."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ElementType(str, Enum):
    """Types of GUI elements that can be detected and stored."""

    # Button types
    BUTTON = "button"
    ICON_BUTTON = "icon_button"
    TOGGLE_BUTTON = "toggle_button"
    DROPDOWN_BUTTON = "dropdown_button"

    # Input types
    TEXT_INPUT = "text_input"
    SEARCH_INPUT = "search_input"
    PASSWORD_INPUT = "password_input"
    TEXTAREA = "textarea"

    # Selection types
    CHECKBOX = "checkbox"
    RADIO_BUTTON = "radio_button"
    DROPDOWN = "dropdown"
    COMBOBOX = "combobox"
    SLIDER = "slider"

    # Navigation types
    LINK = "link"
    TAB = "tab"
    MENU_ITEM = "menu_item"
    BREADCRUMB = "breadcrumb"

    # Container types
    MODAL = "modal"
    DIALOG = "dialog"
    PANEL = "panel"
    CARD = "card"

    # Display types
    ICON = "icon"
    IMAGE = "image"
    LABEL = "label"
    BADGE = "badge"
    TOOLTIP = "tooltip"

    # Data display types
    TABLE_CELL = "table_cell"
    TABLE_HEADER = "table_header"
    LIST_ITEM = "list_item"

    # Feedback types
    PROGRESS = "progress"
    SPINNER = "spinner"

    # Unknown
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Bounding box coordinates for a GUI element."""

    x: int
    y: int
    width: int
    height: int

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary format."""
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> "BoundingBox":
        """Create from dictionary format."""
        return cls(x=data["x"], y=data["y"], width=data["width"], height=data["height"])


@dataclass
class GUIElementChunk:
    """
    Complete representation of a GUI element chunk for RAG retrieval.

    This dataclass contains all information needed for storing, searching,
    and retrieving GUI elements from a vector database.
    """

    # ============================================================================
    # Identity
    # ============================================================================
    id: str  # Unique identifier (UUID)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # ============================================================================
    # Source Information
    # ============================================================================
    source_app: str = ""  # Application name/identifier
    source_state_id: str | None = None  # State machine state ID
    source_screenshot_id: str | None = None  # Screenshot identifier
    extraction_method: str = "manual"  # How element was extracted

    # ============================================================================
    # Geometry
    # ============================================================================
    bounding_box: BoundingBox | None = None
    width: int = 0
    height: int = 0
    aspect_ratio: float = 0.0
    area: int = 0
    position_quadrant: str = ""  # "top-left", "top-right", "bottom-left", "bottom-right", "center"

    # ============================================================================
    # Visual Features
    # ============================================================================
    dominant_colors: list[tuple[int, int, int]] = field(default_factory=list)
    color_histogram: list[int] = field(default_factory=list)
    average_brightness: float = 0.0
    contrast_ratio: float = 0.0
    edge_density: float = 0.0

    # ============================================================================
    # Text Content
    # ============================================================================
    has_text: bool = False
    ocr_text: str = ""
    ocr_confidence: float = 0.0
    text_length: int = 0

    # ============================================================================
    # Classification
    # ============================================================================
    element_type: ElementType = ElementType.UNKNOWN
    element_subtype: str = ""  # More specific classification
    is_interactive: bool = False
    interaction_type: str = ""  # "click", "type", "select", etc.

    # ============================================================================
    # State Indicators
    # ============================================================================
    visual_state: str = "normal"  # "normal", "hover", "pressed", "disabled"
    is_enabled: bool = True
    is_selected: bool = False
    is_focused: bool = False

    # ============================================================================
    # Context
    # ============================================================================
    parent_region: str | None = None  # Parent container/region identifier
    depth_in_hierarchy: int = 0
    sibling_count: int = 0

    # ============================================================================
    # Platform
    # ============================================================================
    platform: str = ""  # "windows", "macos", "linux", "web"

    # ============================================================================
    # Embeddings
    # ============================================================================
    text_embedding: list[float] | None = None  # Dense vector for text
    text_description: str = ""  # Human-readable description for embedding
    image_embedding: list[float] | None = None  # Dense vector for visual features

    # ============================================================================
    # State Machine Integration
    # ============================================================================
    state_id: str | None = None  # Associated state ID
    state_name: str = ""  # Human-readable state name
    is_defining_element: bool = False  # Is this element required for state identification?
    is_optional_element: bool = False  # Is this element optional in the state?
    similarity_threshold: float = 0.8  # Threshold for matching
    is_fixed_position: bool = False  # Does element stay in same position?
    is_shared: bool = False  # Is element shared across multiple states?
    probability: float = 1.0  # Probability of finding this element
    search_region_id: str | None = None  # Search region identifier

    # ============================================================================
    # Cross-Application Semantics
    # ============================================================================
    semantic_role: str = ""  # "save", "cancel", "submit", "close", etc.
    semantic_action: str = ""  # Expected action when interacted with
    style_family: str = ""  # UI toolkit/style (e.g., "material", "fluent", "gtk")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for serialization."""
        return {
            # Identity
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            # Source
            "source_app": self.source_app,
            "source_state_id": self.source_state_id,
            "source_screenshot_id": self.source_screenshot_id,
            "extraction_method": self.extraction_method,
            # Geometry
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "width": self.width,
            "height": self.height,
            "aspect_ratio": self.aspect_ratio,
            "area": self.area,
            "position_quadrant": self.position_quadrant,
            # Visual
            "dominant_colors": self.dominant_colors,
            "color_histogram": self.color_histogram,
            "average_brightness": self.average_brightness,
            "contrast_ratio": self.contrast_ratio,
            "edge_density": self.edge_density,
            # Text
            "has_text": self.has_text,
            "ocr_text": self.ocr_text,
            "ocr_confidence": self.ocr_confidence,
            "text_length": self.text_length,
            # Classification
            "element_type": self.element_type.value,
            "element_subtype": self.element_subtype,
            "is_interactive": self.is_interactive,
            "interaction_type": self.interaction_type,
            # State
            "visual_state": self.visual_state,
            "is_enabled": self.is_enabled,
            "is_selected": self.is_selected,
            "is_focused": self.is_focused,
            # Context
            "parent_region": self.parent_region,
            "depth_in_hierarchy": self.depth_in_hierarchy,
            "sibling_count": self.sibling_count,
            # Platform
            "platform": self.platform,
            # Embeddings
            "text_embedding": self.text_embedding,
            "text_description": self.text_description,
            "image_embedding": self.image_embedding,
            # State machine
            "state_id": self.state_id,
            "state_name": self.state_name,
            "is_defining_element": self.is_defining_element,
            "is_optional_element": self.is_optional_element,
            "similarity_threshold": self.similarity_threshold,
            "is_fixed_position": self.is_fixed_position,
            "is_shared": self.is_shared,
            "probability": self.probability,
            "search_region_id": self.search_region_id,
            # Semantics
            "semantic_role": self.semantic_role,
            "semantic_action": self.semantic_action,
            "style_family": self.style_family,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GUIElementChunk":
        """Create from dictionary format."""
        # Handle bounding box
        bbox = None
        if data.get("bounding_box"):
            bbox = BoundingBox.from_dict(data["bounding_box"])

        # Handle datetime fields
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now()

        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        elif updated_at is None:
            updated_at = datetime.now()

        # Handle element type
        element_type_val = data.get("element_type", "unknown")
        if isinstance(element_type_val, str):
            element_type = ElementType(element_type_val)
        else:
            element_type = element_type_val

        return cls(
            # Identity
            id=data["id"],
            created_at=created_at,
            updated_at=updated_at,
            # Source
            source_app=data.get("source_app", ""),
            source_state_id=data.get("source_state_id"),
            source_screenshot_id=data.get("source_screenshot_id"),
            extraction_method=data.get("extraction_method", "manual"),
            # Geometry
            bounding_box=bbox,
            width=data.get("width", 0),
            height=data.get("height", 0),
            aspect_ratio=data.get("aspect_ratio", 0.0),
            area=data.get("area", 0),
            position_quadrant=data.get("position_quadrant", ""),
            # Visual
            dominant_colors=data.get("dominant_colors", []),
            color_histogram=data.get("color_histogram", []),
            average_brightness=data.get("average_brightness", 0.0),
            contrast_ratio=data.get("contrast_ratio", 0.0),
            edge_density=data.get("edge_density", 0.0),
            # Text
            has_text=data.get("has_text", False),
            ocr_text=data.get("ocr_text", ""),
            ocr_confidence=data.get("ocr_confidence", 0.0),
            text_length=data.get("text_length", 0),
            # Classification
            element_type=element_type,
            element_subtype=data.get("element_subtype", ""),
            is_interactive=data.get("is_interactive", False),
            interaction_type=data.get("interaction_type", ""),
            # State
            visual_state=data.get("visual_state", "normal"),
            is_enabled=data.get("is_enabled", True),
            is_selected=data.get("is_selected", False),
            is_focused=data.get("is_focused", False),
            # Context
            parent_region=data.get("parent_region"),
            depth_in_hierarchy=data.get("depth_in_hierarchy", 0),
            sibling_count=data.get("sibling_count", 0),
            # Platform
            platform=data.get("platform", ""),
            # Embeddings
            text_embedding=data.get("text_embedding"),
            text_description=data.get("text_description", ""),
            image_embedding=data.get("image_embedding"),
            # State machine
            state_id=data.get("state_id"),
            state_name=data.get("state_name", ""),
            is_defining_element=data.get("is_defining_element", False),
            is_optional_element=data.get("is_optional_element", False),
            similarity_threshold=data.get("similarity_threshold", 0.8),
            is_fixed_position=data.get("is_fixed_position", False),
            is_shared=data.get("is_shared", False),
            probability=data.get("probability", 1.0),
            search_region_id=data.get("search_region_id"),
            # Semantics
            semantic_role=data.get("semantic_role", ""),
            semantic_action=data.get("semantic_action", ""),
            style_family=data.get("style_family", ""),
        )

    def to_qdrant_point(self) -> dict[str, Any]:
        """Convert to Qdrant point format for vector database storage.

        Returns:
            Dictionary with 'id', 'vector' (named vectors), and 'payload' keys
        """
        # Build named vectors dict
        vectors: dict[str, list[float]] = {}
        if self.text_embedding:
            vectors["text_embedding"] = self.text_embedding
        if self.image_embedding:
            vectors["clip_embedding"] = self.image_embedding

        # Convert to dict for payload (excluding embeddings which are in vectors)
        payload = self.to_dict()
        payload.pop("text_embedding", None)
        payload.pop("image_embedding", None)

        return {
            "id": self.id,
            "vector": vectors,
            "payload": payload,
        }

    @classmethod
    def from_qdrant_point(cls, point: Any) -> "GUIElementChunk":
        """Create from Qdrant point format.

        Args:
            point: Qdrant ScoredPoint or Record object

        Returns:
            GUIElementChunk instance
        """
        # Extract payload
        payload = dict(point.payload) if hasattr(point, "payload") else {}

        # Extract vectors if available
        vectors = {}
        if hasattr(point, "vector"):
            if isinstance(point.vector, dict):
                vectors = point.vector
            elif point.vector:
                # Single vector - assume it's text embedding
                vectors = {"text_embedding": point.vector}

        # Add embeddings back to payload
        if "text_embedding" in vectors:
            payload["text_embedding"] = vectors["text_embedding"]
        if "clip_embedding" in vectors:
            payload["image_embedding"] = vectors["clip_embedding"]
        if "dinov2_embedding" in vectors:
            # Store as image_embedding if clip not present
            if "image_embedding" not in payload:
                payload["image_embedding"] = vectors["dinov2_embedding"]

        return cls.from_dict(payload)


@dataclass
class EmbeddedElement:
    """
    Result of embedding a GUI element.

    Contains the original element plus computed embeddings.
    """

    element: GUIElementChunk
    text_embedding: list[float] | None = None
    image_embedding: list[float] | None = None
    embedding_model: str = ""  # Model used for embedding
    embedding_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SearchResult:
    """
    Result from a vector database search query.

    Contains the matched element and relevance scores.
    """

    element: GUIElementChunk
    score: float  # Similarity score (0-1)
    distance: float = 0.0  # Distance metric from query
    rank: int = 0  # Position in result list

    # Which embedding was matched
    matched_on: str = "text"  # "text", "image", or "hybrid"
    search_type: str = "text"  # Type of search performed (alias for matched_on)

    # Metadata
    query_text: str = ""
    query_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExportResult:
    """
    Result from the export pipeline.

    Tracks what was exported and any errors that occurred.
    """

    success: bool
    exported_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    export_timestamp: datetime = field(default_factory=datetime.now)
    export_path: str = ""
    format: str = "json"  # "json", "csv", "parquet", etc.
