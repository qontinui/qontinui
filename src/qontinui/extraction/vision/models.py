"""
Data models for vision-based extraction.

These models represent StateImage candidates discovered through computer vision
techniques. StateImage is the primary extractable element in Qontinui - element
types (button, input, etc.) are for description only and don't have functional
significance.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class DetectionTechnique(Enum):
    """Detection techniques used in vision extraction."""

    EDGE = "edge"  # Canny edge detection + contour analysis
    SAM3 = "sam3"  # Segment Anything Model 3
    OCR = "ocr"  # Text detection and recognition
    TEMPLATE = "template"  # Template matching
    CONTOUR = "contour"  # Direct contour detection


@dataclass
class BoundingBox:
    """Bounding box for an element or region."""

    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> "BoundingBox":
        return cls(x=data["x"], y=data["y"], width=data["width"], height=data["height"])

    def iou(self, other: "BoundingBox") -> float:
        """Calculate Intersection over Union with another bounding box."""
        xi = max(self.x, other.x)
        yi = max(self.y, other.y)
        wi = min(self.x2, other.x2) - xi
        hi = min(self.y2, other.y2) - yi

        if wi <= 0 or hi <= 0:
            return 0.0

        intersection = wi * hi
        union = self.area + other.area - intersection

        return intersection / union if union > 0 else 0.0

    def contains(self, other: "BoundingBox") -> bool:
        """Check if this bounding box fully contains another."""
        return (
            self.x <= other.x and self.y <= other.y and self.x2 >= other.x2 and self.y2 >= other.y2
        )


@dataclass
class ExtractedStateImageCandidate:
    """
    A candidate StateImage discovered by vision extraction.

    StateImage is the primary extractable element in Qontinui. Categories like
    button, input, link are for human description only - they don't have
    functional significance in the automation framework.
    """

    id: str
    bbox: BoundingBox
    confidence: float  # 0.0 - 1.0

    # Visual reference
    screenshot_id: str
    cropped_image_path: Path | None = None

    # Description (for human understanding, not functional)
    category: str | None = None  # "button", "input", "link", "icon", "text", etc.
    text: str | None = None  # OCR-detected text
    description: str | None = None  # Auto-generated description

    # Extraction metadata
    extraction_method: str = "vision"  # "vision", "dom", "accessibility"
    detection_technique: str = "unknown"  # "edge", "sam3", "ocr", "template"
    source_url: str | None = None

    # For transition detection
    is_clickable: bool = False  # Likely triggers a transition

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "bbox": self.bbox.to_dict(),
            "confidence": self.confidence,
            "screenshot_id": self.screenshot_id,
            "cropped_image_path": str(self.cropped_image_path) if self.cropped_image_path else None,
            "category": self.category,
            "text": self.text,
            "description": self.description,
            "extraction_method": self.extraction_method,
            "detection_technique": self.detection_technique,
            "source_url": self.source_url,
            "is_clickable": self.is_clickable,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractedStateImageCandidate":
        return cls(
            id=data["id"],
            bbox=BoundingBox.from_dict(data["bbox"]),
            confidence=data["confidence"],
            screenshot_id=data["screenshot_id"],
            cropped_image_path=(
                Path(data["cropped_image_path"]) if data.get("cropped_image_path") else None
            ),
            category=data.get("category"),
            text=data.get("text"),
            description=data.get("description"),
            extraction_method=data.get("extraction_method", "vision"),
            detection_technique=data.get("detection_technique", "unknown"),
            source_url=data.get("source_url"),
            is_clickable=data.get("is_clickable", False),
            metadata=data.get("metadata", {}),
        )


# -----------------------------------------------------------------------------
# Debug output models for individual detection techniques
# -----------------------------------------------------------------------------


@dataclass
class EdgeDetectionResult:
    """Result from edge detection technique."""

    id: str
    bbox: BoundingBox
    contour_area: float
    contour_perimeter: float
    vertex_count: int  # Number of vertices in approximated polygon
    aspect_ratio: float
    confidence: float

    # Debug visualization data
    contour_points: list[tuple[int, int]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "bbox": self.bbox.to_dict(),
            "contour_area": self.contour_area,
            "contour_perimeter": self.contour_perimeter,
            "vertex_count": self.vertex_count,
            "aspect_ratio": self.aspect_ratio,
            "confidence": self.confidence,
            "contour_points": self.contour_points,
        }


@dataclass
class SAM3SegmentResult:
    """Result from SAM3 segmentation technique."""

    id: str
    bbox: BoundingBox
    mask_area: int  # Pixel count
    stability_score: float
    predicted_iou: float
    confidence: float

    # Debug visualization data (mask encoded for transfer)
    mask_rle: dict[str, Any] | None = None  # Run-length encoding

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "bbox": self.bbox.to_dict(),
            "mask_area": self.mask_area,
            "stability_score": self.stability_score,
            "predicted_iou": self.predicted_iou,
            "confidence": self.confidence,
            "mask_rle": self.mask_rle,
        }


@dataclass
class OCRResult:
    """Result from OCR text detection technique."""

    id: str
    bbox: BoundingBox
    text: str
    confidence: float
    language: str = "en"

    # Word-level details
    word_boxes: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "bbox": self.bbox.to_dict(),
            "text": self.text,
            "confidence": self.confidence,
            "language": self.language,
            "word_boxes": self.word_boxes,
        }


@dataclass
class ContourResult:
    """Result from contour detection (used in edge detection)."""

    id: str
    bbox: BoundingBox
    area: float
    perimeter: float
    hierarchy_level: int  # 0 = outermost
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "bbox": self.bbox.to_dict(),
            "area": self.area,
            "perimeter": self.perimeter,
            "hierarchy_level": self.hierarchy_level,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
        }


@dataclass
class TemplateMatchResult:
    """Result from template matching technique."""

    id: str
    bbox: BoundingBox
    template_name: str
    match_score: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "bbox": self.bbox.to_dict(),
            "template_name": self.template_name,
            "match_score": self.match_score,
            "confidence": self.confidence,
        }


@dataclass
class ScreenshotInfo:
    """Information about a screenshot used in extraction."""

    id: str
    path: Path
    width: int
    height: int
    captured_at: datetime = field(default_factory=datetime.now)
    source_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": str(self.path),
            "width": self.width,
            "height": self.height,
            "captured_at": self.captured_at.isoformat(),
            "source_url": self.source_url,
        }


@dataclass
class VisionExtractionResult:
    """
    Result from vision extraction containing StateImage candidates.

    Includes debug outputs from each detection technique for visualization
    in qontinui-web.
    """

    extraction_id: str
    extraction_method: str = "vision"

    # Primary output
    candidates: list[ExtractedStateImageCandidate] = field(default_factory=list)

    # Debug outputs by technique (for visualization)
    edge_detection_results: list[EdgeDetectionResult] | None = None
    sam3_segments: list[SAM3SegmentResult] | None = None
    ocr_results: list[OCRResult] | None = None
    contour_results: list[ContourResult] | None = None
    template_matches: list[TemplateMatchResult] | None = None

    # Screenshots taken
    screenshots: list[ScreenshotInfo] = field(default_factory=list)

    # Raw debug images (base64 encoded for API transport)
    edge_overlay_image: str | None = None  # Edges overlaid on screenshot
    contour_overlay_image: str | None = None  # Contours highlighted
    sam3_mask_image: str | None = None  # SAM3 segments colored
    ocr_overlay_image: str | None = None  # OCR boxes with text

    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    duration_ms: float = 0.0

    # Metadata
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def complete(self) -> None:
        """Mark extraction as complete and calculate duration."""
        self.completed_at = datetime.now()
        self.duration_ms = (self.completed_at - self.started_at).total_seconds() * 1000

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)

    @property
    def is_successful(self) -> bool:
        """Check if extraction completed without errors."""
        return len(self.errors) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "extraction_id": self.extraction_id,
            "extraction_method": self.extraction_method,
            "candidates": [c.to_dict() for c in self.candidates],
            "edge_detection_results": (
                [e.to_dict() for e in self.edge_detection_results]
                if self.edge_detection_results
                else None
            ),
            "sam3_segments": (
                [s.to_dict() for s in self.sam3_segments] if self.sam3_segments else None
            ),
            "ocr_results": [o.to_dict() for o in self.ocr_results] if self.ocr_results else None,
            "contour_results": (
                [c.to_dict() for c in self.contour_results] if self.contour_results else None
            ),
            "template_matches": (
                [t.to_dict() for t in self.template_matches] if self.template_matches else None
            ),
            "screenshots": [s.to_dict() for s in self.screenshots],
            "edge_overlay_image": self.edge_overlay_image,
            "contour_overlay_image": self.contour_overlay_image,
            "sam3_mask_image": self.sam3_mask_image,
            "ocr_overlay_image": self.ocr_overlay_image,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "errors": self.errors,
            "warnings": self.warnings,
            "config": self.config,
        }


# -----------------------------------------------------------------------------
# Configuration models
# -----------------------------------------------------------------------------


@dataclass
class EdgeDetectionConfig:
    """Configuration for edge detection."""

    enabled: bool = True
    canny_low: int = 50
    canny_high: int = 150
    min_contour_area: int = 100
    max_contour_area: int = 500000  # Ignore very large contours
    aspect_ratio_min: float = 0.1
    aspect_ratio_max: float = 10.0
    approximation_epsilon: float = 0.02  # Contour approximation factor


@dataclass
class SAM3Config:
    """Configuration for SAM3 segmentation."""

    enabled: bool = True
    model_type: str = "vit_h"  # vit_h, vit_l, vit_b
    points_per_side: int = 32
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    min_mask_region_area: int = 100
    max_mask_region_area: int = 500000


@dataclass
class OCRConfig:
    """Configuration for OCR detection."""

    enabled: bool = True
    engine: str = "easyocr"  # easyocr, tesseract
    languages: list[str] = field(default_factory=lambda: ["en"])
    confidence_threshold: float = 0.6
    min_text_height: int = 8
    max_text_height: int = 200


@dataclass
class TemplateConfig:
    """Configuration for template matching."""

    enabled: bool = False  # Requires template library
    match_threshold: float = 0.8
    template_dir: Path | None = None


@dataclass
class FusionConfig:
    """Configuration for result fusion/deduplication."""

    iou_threshold: float = 0.5  # Above this, consider as same element
    prefer_higher_confidence: bool = True
    max_candidates: int = 500


@dataclass
class VisionExtractionConfig:
    """Complete configuration for vision extraction."""

    edge_detection: EdgeDetectionConfig = field(default_factory=EdgeDetectionConfig)
    sam3: SAM3Config = field(default_factory=SAM3Config)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    template: TemplateConfig = field(default_factory=TemplateConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)

    # Output options
    save_debug_images: bool = True
    save_cropped_candidates: bool = True
    output_dir: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_detection": {
                "enabled": self.edge_detection.enabled,
                "canny_low": self.edge_detection.canny_low,
                "canny_high": self.edge_detection.canny_high,
                "min_contour_area": self.edge_detection.min_contour_area,
                "max_contour_area": self.edge_detection.max_contour_area,
                "aspect_ratio_min": self.edge_detection.aspect_ratio_min,
                "aspect_ratio_max": self.edge_detection.aspect_ratio_max,
                "approximation_epsilon": self.edge_detection.approximation_epsilon,
            },
            "sam3": {
                "enabled": self.sam3.enabled,
                "model_type": self.sam3.model_type,
                "points_per_side": self.sam3.points_per_side,
                "pred_iou_thresh": self.sam3.pred_iou_thresh,
                "stability_score_thresh": self.sam3.stability_score_thresh,
                "min_mask_region_area": self.sam3.min_mask_region_area,
                "max_mask_region_area": self.sam3.max_mask_region_area,
            },
            "ocr": {
                "enabled": self.ocr.enabled,
                "engine": self.ocr.engine,
                "languages": self.ocr.languages,
                "confidence_threshold": self.ocr.confidence_threshold,
                "min_text_height": self.ocr.min_text_height,
                "max_text_height": self.ocr.max_text_height,
            },
            "template": {
                "enabled": self.template.enabled,
                "match_threshold": self.template.match_threshold,
                "template_dir": (
                    str(self.template.template_dir) if self.template.template_dir else None
                ),
            },
            "fusion": {
                "iou_threshold": self.fusion.iou_threshold,
                "prefer_higher_confidence": self.fusion.prefer_higher_confidence,
                "max_candidates": self.fusion.max_candidates,
            },
            "save_debug_images": self.save_debug_images,
            "save_cropped_candidates": self.save_cropped_candidates,
            "output_dir": str(self.output_dir) if self.output_dir else None,
        }
