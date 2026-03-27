"""Paired capture service — screenshot + accessibility tree in one atomic operation.

Inspired by screenpipe's paired capture pattern: captures a screenshot and walks
the OS accessibility tree simultaneously. Prefers accessibility text (fast, structured)
with OCR fallback when the a11y tree yields insufficient content.

This is the core capture primitive for black-box desktop automation. Results feed
into the activity timeline for searchable history.

Example:
    >>> from qontinui.hal import initialize_hal
    >>> from qontinui.hal.services import PairedCaptureService
    >>>
    >>> hal = initialize_hal()
    >>> service = PairedCaptureService(hal)
    >>> result = await service.capture()
    >>> print(f"Source: {result.source_type}, Elements: {result.element_count}")
    Source: accessibility, Elements: 42
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from qontinui.hal.container import HALContainer
    from qontinui_schemas.accessibility import AccessibilityNode

logger = logging.getLogger(__name__)


@dataclass
class PairedCaptureResult:
    """Result of an atomic paired capture (screenshot + text extraction)."""

    screenshot: Image.Image
    """The captured screenshot."""

    screenshot_path: str | None = None
    """Path where screenshot was saved (if save_screenshot=True)."""

    text_content: str = ""
    """Extracted text content from accessibility tree or OCR."""

    source_type: str = "ocr"
    """How text was extracted: 'accessibility' or 'ocr'."""

    content_hash: str = ""
    """SHA-256 of normalized text for deduplication."""

    element_count: int = 0
    """Number of accessibility nodes or OCR text regions found."""

    confidence: float = 0.0
    """Confidence score: a11y completeness ratio or average OCR confidence."""

    app_name: str | None = None
    """Active application name at capture time."""

    window_title: str | None = None
    """Active window title at capture time."""

    url: str | None = None
    """Page URL (for web targets via CDP)."""

    capture_duration_ms: float = 0.0
    """Time taken for the full paired capture."""

    metadata: dict = field(default_factory=dict)
    """Extensible metadata (backend name, interactive count, etc.)."""


class PairedCaptureService:
    """Atomically captures screenshot + accessibility tree text.

    Uses the HAL container's existing screen capture, accessibility capture,
    and OCR engine — no new backends are needed. The service coordinates
    these components and applies smart source selection:

    1. Capture screenshot via IScreenCapture
    2. Capture accessibility tree via IAccessibilityCapture
    3. Extract text from tree nodes (name, value, description)
    4. If too few elements found, fall back to OCR on the screenshot
    5. Compute content hash for deduplication

    Args:
        hal: HALContainer with screen_capture, accessibility_capture, and ocr_engine.
        min_a11y_elements: Minimum a11y elements required before OCR fallback triggers.
            Set lower for simpler apps, higher for rich UIs.
    """

    def __init__(self, hal: HALContainer, min_a11y_elements: int = 5) -> None:
        self._screen = hal.screen_capture
        self._a11y = hal.accessibility_capture
        self._ocr = hal.ocr_engine
        self._min_a11y_elements = min_a11y_elements

    async def capture(
        self,
        monitor: int | None = None,
        save_screenshot: bool = False,
        screenshot_dir: str | Path | None = None,
    ) -> PairedCaptureResult:
        """Perform a paired capture: screenshot + text extraction.

        Args:
            monitor: Monitor index to capture (None for primary).
            save_screenshot: Whether to save the screenshot to disk.
            screenshot_dir: Directory for saved screenshots (defaults to ~/.qontinui/captures/).

        Returns:
            PairedCaptureResult with screenshot, extracted text, and metadata.
        """
        start = time.monotonic()
        result = PairedCaptureResult(screenshot=Image.new("RGB", (1, 1)))

        # Step 1: Capture screenshot
        try:
            result.screenshot = self._screen.capture_screen(monitor)
        except Exception as e:
            logger.warning("Screenshot capture failed: %s", e)

        # Step 2: Try accessibility tree first (fast path)
        a11y_text, a11y_count, a11y_meta = await self._try_accessibility()

        if a11y_count >= self._min_a11y_elements:
            result.text_content = a11y_text
            result.source_type = "accessibility"
            result.element_count = a11y_count
            result.confidence = a11y_meta.get("completeness", 1.0)
            result.url = a11y_meta.get("url")
            result.window_title = a11y_meta.get("title")
            result.metadata = a11y_meta
        else:
            # Step 3: OCR fallback (slow path)
            ocr_text, ocr_count, ocr_confidence = self._try_ocr(result.screenshot)
            result.text_content = ocr_text
            result.source_type = "ocr"
            result.element_count = ocr_count
            result.confidence = ocr_confidence
            result.metadata = {
                "a11y_attempted": True,
                "a11y_element_count": a11y_count,
                "ocr_fallback_reason": (
                    "not_connected" if a11y_count == 0 and not (self._a11y and self._a11y.is_connected())
                    else f"insufficient_elements ({a11y_count} < {self._min_a11y_elements})"
                ),
            }

        # Step 4: Compute content hash
        result.content_hash = self._compute_hash(result.text_content)

        # Step 5: Optionally save screenshot
        if save_screenshot and result.screenshot.size != (1, 1):
            result.screenshot_path = self._save_screenshot(
                result.screenshot, screenshot_dir
            )

        result.capture_duration_ms = (time.monotonic() - start) * 1000
        return result

    async def capture_with_dedup(
        self,
        prev_hash: str | None = None,
        monitor: int | None = None,
    ) -> PairedCaptureResult | None:
        """Capture but return None if content hash matches previous.

        Useful for continuous background capture to skip identical frames.

        Args:
            prev_hash: Content hash from the previous capture.
            monitor: Monitor index to capture.

        Returns:
            PairedCaptureResult if content changed, None if identical.
        """
        result = await self.capture(monitor=monitor)
        if prev_hash and result.content_hash == prev_hash:
            return None
        return result

    async def _try_accessibility(self) -> tuple[str, int, dict]:
        """Attempt to capture and extract text from the accessibility tree.

        Returns:
            (text, element_count, metadata) — text may be empty if not connected.
        """
        if not self._a11y or not self._a11y.is_connected():
            return "", 0, {}

        try:
            snapshot = await self._a11y.capture_tree()
            text = self._extract_text_from_tree(snapshot.root)
            meta = {
                "backend": snapshot.backend.value if hasattr(snapshot.backend, "value") else str(snapshot.backend),
                "total_nodes": snapshot.total_nodes,
                "interactive_nodes": snapshot.interactive_nodes,
                "completeness": snapshot.interactive_nodes / max(snapshot.total_nodes, 1),
                "url": snapshot.url,
                "title": snapshot.title,
            }
            return text, snapshot.total_nodes, meta
        except Exception as e:
            logger.warning("Accessibility capture failed: %s", e)
            return "", 0, {"error": str(e)}

    def _try_ocr(self, screenshot: Image.Image) -> tuple[str, int, float]:
        """Extract text from screenshot via OCR engine.

        Returns:
            (text, region_count, average_confidence)
        """
        if not self._ocr:
            return "", 0, 0.0

        try:
            regions = self._ocr.get_text_regions(screenshot)
            if not regions:
                return "", 0, 0.0

            text_parts = [r.text for r in regions if r.text.strip()]
            avg_confidence = sum(r.confidence for r in regions) / len(regions)
            return " ".join(text_parts), len(regions), avg_confidence
        except Exception as e:
            logger.warning("OCR extraction failed: %s", e)
            return "", 0, 0.0

    def _extract_text_from_tree(self, node: AccessibilityNode, depth: int = 0) -> str:
        """Recursively extract text content from accessibility tree nodes.

        Collects name, value, and description fields from all nodes,
        producing a flattened text representation of the UI.
        """
        if depth > 200:
            return ""

        parts: list[str] = []

        # Collect text from this node
        if node.name:
            parts.append(node.name)
        if node.value and node.value != node.name:
            parts.append(node.value)
        if node.description and node.description not in (node.name, node.value):
            parts.append(node.description)

        # Recurse into children
        for child in node.children:
            child_text = self._extract_text_from_tree(child, depth + 1)
            if child_text:
                parts.append(child_text)

        return " ".join(parts)

    @staticmethod
    def _compute_hash(text: str) -> str:
        """Compute SHA-256 hash of normalized text for deduplication.

        Matches the normalization used by the Rust activity_timeline module:
        trim + lowercase + SHA-256.
        """
        normalized = text.strip().lower().encode("utf-8")
        return hashlib.sha256(normalized).hexdigest()

    @staticmethod
    def _save_screenshot(image: Image.Image, directory: str | Path | None) -> str:
        """Save screenshot to disk and return the file path."""
        if directory is None:
            directory = Path.home() / ".qontinui" / "captures"
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        filename = f"capture_{int(time.time() * 1000)}.jpg"
        path = directory / filename
        image.save(str(path), "JPEG", quality=85)
        return str(path)
