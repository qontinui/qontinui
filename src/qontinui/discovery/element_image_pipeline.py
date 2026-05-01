"""Element-to-image pipeline for extracting GUI element images from screenshots.

This module bridges UI Bridge's semantic element data (positions, labels, types)
with screen capture to produce cropped element images suitable for visual GUI
automation configs.

Pipeline:
    1. Accept a UI Bridge snapshot (element rects) + a screenshot (PIL Image)
    2. Map viewport-relative element rects to screenshot coordinates
    3. Crop each element from the screenshot
    4. Encode as base64 PNG for inclusion in QontinuiConfig

The caller provides the window offset (distance from monitor origin to the
webview content area). This keeps the pipeline decoupled from window management.

Example:
    from qontinui.discovery.element_image_pipeline import ElementImagePipeline

    pipeline = ElementImagePipeline()

    # From UI Bridge GET /control/snapshot
    snapshot = {"elements": [...], "viewport": {"width": 1400, "height": 900}}

    # From IScreenCapture.capture_screen()
    screenshot = capture.capture_screen(monitor=0)

    # Window content area origin on the monitor
    window_offset = (100, 150)

    result = pipeline.extract(snapshot, screenshot, window_offset)
    for img in result.images:
        print(f"{img.element_id}: {img.width}x{img.height}")
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ElementRect:
    """Bounding rectangle for a UI element in viewport coordinates."""

    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @classmethod
    def from_snapshot_element(cls, state: dict[str, Any]) -> ElementRect | None:
        """Parse rect from a UI Bridge snapshot element's state dict."""
        rect = state.get("rect")
        if not rect:
            return None
        try:
            return cls(
                x=float(rect["x"]),
                y=float(rect["y"]),
                width=float(rect["width"]),
                height=float(rect["height"]),
            )
        except (KeyError, TypeError, ValueError):
            return None


@dataclass
class ExtractedElementImage:
    """A cropped element image with metadata."""

    element_id: str
    label: str
    element_type: str
    image: Image.Image
    bbox: tuple[int, int, int, int]  # (x, y, width, height) in screenshot coords
    base64_png: str  # base64-encoded PNG (no data: prefix)
    sha256: str
    viewport_rect: ElementRect  # Original viewport-relative rect

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height


@dataclass
class ExtractionResult:
    """Result of the element-to-image extraction pipeline."""

    images: list[ExtractedElementImage]
    screenshot_width: int
    screenshot_height: int
    viewport_width: int
    viewport_height: int
    window_offset: tuple[int, int]
    skipped: list[dict[str, str]] = field(default_factory=list)


@dataclass
class ExtractionConfig:
    """Configuration for the extraction pipeline."""

    min_element_size: int = 4
    """Minimum width/height in pixels to include an element."""

    padding: int = 0
    """Extra pixels around each element crop (clamped to screenshot bounds)."""

    include_invisible: bool = False
    """Include elements marked as not visible in the snapshot."""

    include_out_of_viewport: bool = False
    """Include elements outside the viewport."""

    category_filter: set[str] | None = None
    """If set, only include elements whose category is in this set.
    Common categories: 'interactive', 'content', 'media'."""

    type_filter: set[str] | None = None
    """If set, only include elements whose type is in this set.
    Common types: 'button', 'input', 'link', 'select', etc."""

    scale_factor: float = 1.0
    """DPI scale factor. If the screenshot is captured at 2x DPI,
    set this to 2.0 so viewport pixels map correctly to screenshot pixels."""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass
class _FilteredElement:
    """An element that passed all filter checks."""

    elem_id: str
    label: str
    elem_type: str
    rect: ElementRect


class ElementImagePipeline:
    """Extracts element images from screenshots using UI Bridge position data."""

    def __init__(self, config: ExtractionConfig | None = None) -> None:
        self.config = config or ExtractionConfig()

    def _filter_elements(
        self,
        elements: list[dict[str, Any]],
        skipped: list[dict[str, str]],
    ) -> list[_FilteredElement]:
        """Apply visibility, category, type, size, and rect filters to elements.

        Elements that fail a filter are appended to *skipped* with a reason.

        Returns:
            List of elements that passed all filters.
        """
        accepted: list[_FilteredElement] = []
        for elem in elements:
            elem_id = elem.get("id", "")
            label = elem.get("label", elem_id)
            elem_type = elem.get("type", "unknown")
            category = elem.get("category", "")
            state = elem.get("state", {})

            # Visibility filter
            if not self.config.include_invisible and not state.get("visible", True):
                skipped.append({"id": elem_id, "reason": "not visible"})
                continue

            if not self.config.include_out_of_viewport and not state.get(
                "inViewport", True
            ):
                skipped.append({"id": elem_id, "reason": "out of viewport"})
                continue

            # Category filter
            if (
                self.config.category_filter
                and category not in self.config.category_filter
            ):
                skipped.append({"id": elem_id, "reason": f"category={category}"})
                continue

            # Type filter
            if self.config.type_filter and elem_type not in self.config.type_filter:
                skipped.append({"id": elem_id, "reason": f"type={elem_type}"})
                continue

            # Parse rect
            rect = ElementRect.from_snapshot_element(state)
            if rect is None:
                skipped.append({"id": elem_id, "reason": "no rect"})
                continue

            # Size filter
            if (
                rect.width < self.config.min_element_size
                or rect.height < self.config.min_element_size
            ):
                skipped.append({"id": elem_id, "reason": "too small"})
                continue

            accepted.append(
                _FilteredElement(
                    elem_id=elem_id,
                    label=label,
                    elem_type=elem_type,
                    rect=rect,
                )
            )
        return accepted

    def extract(
        self,
        snapshot: dict[str, Any],
        screenshot: Image.Image,
        window_offset: tuple[int, int] = (0, 0),
    ) -> ExtractionResult:
        """Extract element images from a screenshot using UI Bridge snapshot data.

        Args:
            snapshot: UI Bridge control snapshot (from GET /control/snapshot).
                Must contain "elements" list and optionally "viewport".
            screenshot: Full screenshot as a PIL Image. Can be a full monitor
                capture or a window-only capture.
            window_offset: (x, y) offset from the screenshot origin to the
                webview content area's top-left corner. Use (0, 0) if the
                screenshot is already cropped to the content area.

        Returns:
            ExtractionResult with cropped element images and metadata.
        """
        elements = snapshot.get("elements", [])
        viewport = snapshot.get("viewport", {})
        viewport_w = int(viewport.get("width", 0))
        viewport_h = int(viewport.get("height", 0))

        result = ExtractionResult(
            images=[],
            screenshot_width=screenshot.width,
            screenshot_height=screenshot.height,
            viewport_width=viewport_w,
            viewport_height=viewport_h,
            window_offset=window_offset,
        )

        scale = self.config.scale_factor
        ox, oy = window_offset
        accepted = self._filter_elements(elements, result.skipped)

        for fe in accepted:
            rect = fe.rect

            # Map viewport coords to screenshot coords
            pad = self.config.padding
            sx = int(rect.x * scale + ox) - pad
            sy = int(rect.y * scale + oy) - pad
            sw = int(rect.width * scale) + 2 * pad
            sh = int(rect.height * scale) + 2 * pad

            # Clamp to screenshot bounds
            sx = max(0, sx)
            sy = max(0, sy)
            sw = min(sw, screenshot.width - sx)
            sh = min(sh, screenshot.height - sy)

            if sw <= 0 or sh <= 0:
                result.skipped.append({"id": fe.elem_id, "reason": "out of bounds"})
                continue

            # Crop
            cropped = screenshot.crop((sx, sy, sx + sw, sy + sh))

            # Encode to base64 PNG
            b64, sha = _encode_pil_to_base64(cropped)

            result.images.append(
                ExtractedElementImage(
                    element_id=fe.elem_id,
                    label=fe.label,
                    element_type=fe.elem_type,
                    image=cropped,
                    bbox=(sx, sy, sw, sh),
                    base64_png=b64,
                    sha256=sha,
                    viewport_rect=rect,
                )
            )

        logger.info(
            "Extracted %d element images, skipped %d",
            len(result.images),
            len(result.skipped),
        )
        return result

    def extract_from_captures(
        self,
        snapshot: dict[str, Any],
        captures: dict[str, dict[str, Any]],
    ) -> ExtractionResult:
        """Build extraction results from pre-captured element images.

        Instead of cropping from a screenshot, this accepts element images
        that were rendered directly from the DOM (e.g. via html2canvas in
        the UI Bridge). The same visibility/category/size filters are applied.

        Args:
            snapshot: UI Bridge control snapshot (from GET /control/snapshot).
            captures: Dict mapping element_id to capture data. Each value has:
                - base64_png: base64-encoded PNG (no data: prefix)
                - width: image width in pixels
                - height: image height in pixels

        Returns:
            ExtractionResult with element images and metadata.
        """
        elements = snapshot.get("elements", [])
        viewport = snapshot.get("viewport", {})
        viewport_w = int(viewport.get("width", 0))
        viewport_h = int(viewport.get("height", 0))

        result = ExtractionResult(
            images=[],
            screenshot_width=0,
            screenshot_height=0,
            viewport_width=viewport_w,
            viewport_height=viewport_h,
            window_offset=(0, 0),
        )

        accepted = self._filter_elements(elements, result.skipped)

        for fe in accepted:
            # Look up the pre-captured image
            capture = captures.get(fe.elem_id)
            if not capture:
                result.skipped.append({"id": fe.elem_id, "reason": "no capture"})
                continue

            b64_png = capture.get("base64_png", "")
            cap_w = int(capture.get("width", 0))
            cap_h = int(capture.get("height", 0))

            if not b64_png or cap_w <= 0 or cap_h <= 0:
                result.skipped.append({"id": fe.elem_id, "reason": "empty capture"})
                continue

            # Decode base64 to PIL Image (per-element error handling so one
            # corrupt capture doesn't crash the entire extraction)
            try:
                raw = base64.b64decode(b64_png)
                pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
                sha = hashlib.sha256(raw).hexdigest()
            except Exception as exc:
                logger.warning("Failed to decode capture for %s: %s", fe.elem_id, exc)
                result.skipped.append(
                    {"id": fe.elem_id, "reason": f"decode error: {exc}"}
                )
                continue

            result.images.append(
                ExtractedElementImage(
                    element_id=fe.elem_id,
                    label=fe.label,
                    element_type=fe.elem_type,
                    image=pil_image,
                    bbox=(int(fe.rect.x), int(fe.rect.y), cap_w, cap_h),
                    base64_png=b64_png,
                    sha256=sha,
                    viewport_rect=fe.rect,
                )
            )

        logger.info(
            "Extracted %d element images from captures, skipped %d",
            len(result.images),
            len(result.skipped),
        )
        return result

    def extract_for_states(
        self,
        snapshot: dict[str, Any],
        screenshot: Image.Image,
        states: list[dict[str, Any]],
        window_offset: tuple[int, int] = (0, 0),
    ) -> dict[str, list[ExtractedElementImage]]:
        """Extract images grouped by UI Bridge state machine states.

        Args:
            snapshot: UI Bridge control snapshot.
            screenshot: Full screenshot as PIL Image.
            states: List of state dicts, each with "state_id" and "element_ids".
            window_offset: Content area offset.

        Returns:
            Dict mapping state_id → list of extracted element images for that state.
        """
        # First extract all element images
        result = self.extract(snapshot, screenshot, window_offset)

        # Index by element ID
        image_by_id: dict[str, ExtractedElementImage] = {
            img.element_id: img for img in result.images
        }

        # Group by state
        state_images: dict[str, list[ExtractedElementImage]] = {}
        for state in states:
            state_id = state.get("state_id", state.get("id", ""))
            element_ids = state.get("element_ids", [])
            state_images[state_id] = [
                image_by_id[eid] for eid in element_ids if eid in image_by_id
            ]

        return state_images


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_pil_to_base64(image: Image.Image) -> tuple[str, str]:
    """Encode a PIL Image to base64 PNG and compute SHA-256.

    Returns:
        (base64_string, sha256_hex)
    """
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    raw = buf.getvalue()
    b64 = base64.b64encode(raw).decode("ascii")
    sha = hashlib.sha256(raw).hexdigest()
    return b64, sha


def generate_image_id() -> str:
    """Generate a unique image ID matching the runner's convention."""
    return f"img-{uuid.uuid4().hex[:12]}"


def generate_state_image_id() -> str:
    """Generate a unique state image ID matching the runner's convention."""
    ts = int(time.time() * 1000)
    suffix = uuid.uuid4().hex[:8]
    return f"stateimage-{ts}-{suffix}"


def generate_pattern_id() -> str:
    """Generate a unique pattern ID."""
    ts = int(time.time() * 1000)
    suffix = uuid.uuid4().hex[:8]
    return f"pattern-{ts}-{suffix}"
