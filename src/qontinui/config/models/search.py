"""
Search and pattern matching configuration models.

This module provides models for configuring target search behavior,
including image matching, polling, pattern options, and match adjustments.
"""

from typing import Literal

from pydantic import BaseModel, Field

from .base_types import SearchStrategy
from .geometry import Coordinates, Region


class PollingConfig(BaseModel):
    """Polling configuration for search operations."""

    interval: int | None = None
    max_attempts: int | None = Field(None, alias="maxAttempts")

    model_config = {"populate_by_name": True}


class PatternOptions(BaseModel):
    """Advanced pattern matching options."""

    match_method: (
        Literal[
            "CORRELATION", "CORRELATION_NORMED", "SQUARED_DIFFERENCE", "SQUARED_DIFFERENCE_NORMED"
        ]
        | None
    ) = Field(None, alias="matchMethod")
    scale_invariant: bool | None = Field(None, alias="scaleInvariant")
    rotation_invariant: bool | None = Field(None, alias="rotationInvariant")
    min_scale: float | None = Field(None, alias="minScale")
    max_scale: float | None = Field(None, alias="maxScale")
    scale_step: float | None = Field(None, alias="scaleStep")
    min_rotation: float | None = Field(None, alias="minRotation")
    max_rotation: float | None = Field(None, alias="maxRotation")
    rotation_step: float | None = Field(None, alias="rotationStep")
    use_grayscale: bool | None = Field(None, alias="useGrayscale")
    use_color_reduction: bool | None = Field(None, alias="useColorReduction")
    color_tolerance: float | None = Field(None, alias="colorTolerance")
    use_edges: bool | None = Field(None, alias="useEdges")
    edge_threshold1: float | None = Field(None, alias="edgeThreshold1")
    edge_threshold2: float | None = Field(None, alias="edgeThreshold2")
    non_max_suppression: bool | None = Field(None, alias="nonMaxSuppression")
    nms_threshold: float | None = Field(None, alias="nmsThreshold")
    min_distance_between_matches: float | None = Field(None, alias="minDistanceBetweenMatches")

    model_config = {"populate_by_name": True}


class MatchAdjustment(BaseModel):
    """Match adjustment - modify the matched region."""

    target_position: str | None = Field(None, alias="targetPosition")
    target_offset: Coordinates | None = Field(None, alias="targetOffset")
    add_w: int | None = Field(None, alias="addW")
    add_h: int | None = Field(None, alias="addH")
    absolute_w: int | None = Field(None, alias="absoluteW")
    absolute_h: int | None = Field(None, alias="absoluteH")
    add_x: int | None = Field(None, alias="addX")
    add_y: int | None = Field(None, alias="addY")

    model_config = {"populate_by_name": True}


class SearchOptions(BaseModel):
    """Search options for target finding."""

    similarity: float | None = None
    timeout: int | None = None
    search_regions: list[Region] | None = Field(None, alias="searchRegions")
    strategy: SearchStrategy | None = None
    use_defined_region: bool | None = Field(None, alias="useDefinedRegion")
    max_matches_to_act_on: int | None = Field(None, alias="maxMatchesToActOn")
    min_matches: int | None = Field(None, alias="minMatches")
    max_matches: int | None = Field(None, alias="maxMatches")
    polling: PollingConfig | None = None
    pattern: PatternOptions | None = None
    adjustment: MatchAdjustment | None = None
    capture_image: bool | None = Field(None, alias="captureImage")

    model_config = {"populate_by_name": True}


class TextSearchOptions(BaseModel):
    """Text search options for OCR-based finding."""

    ocr_engine: Literal["TESSERACT", "EASYOCR", "PADDLEOCR", "NATIVE"] | None = Field(
        None, alias="ocrEngine"
    )
    language: str | None = None
    whitelist_chars: str | None = Field(None, alias="whitelistChars")
    blacklist_chars: str | None = Field(None, alias="blacklistChars")
    match_type: (
        Literal["EXACT", "CONTAINS", "STARTS_WITH", "ENDS_WITH", "REGEX", "FUZZY"] | None
    ) = Field(None, alias="matchType")
    case_sensitive: bool | None = Field(None, alias="caseSensitive")
    ignore_whitespace: bool | None = Field(None, alias="ignoreWhitespace")
    normalize_unicode: bool | None = Field(None, alias="normalizeUnicode")
    fuzzy_threshold: float | None = Field(None, alias="fuzzyThreshold")
    edit_distance: int | None = Field(None, alias="editDistance")
    preprocessing: list[str] | None = None
    scale_factor: float | None = Field(None, alias="scaleFactor")
    psm_mode: int | None = Field(None, alias="psmMode")
    oem_mode: int | None = Field(None, alias="oemMode")
    confidence_threshold: float | None = Field(None, alias="confidenceThreshold")

    model_config = {"populate_by_name": True}
