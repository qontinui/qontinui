"""CascadeDetector — unified detection fallback chain.

Wraps multiple DetectionBackends and implements graduated fallback:
try the cheapest backend first, fall through to more expensive backends
on failure. Short-circuits as soon as a backend returns results above
the confidence threshold.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from .base import DetectionBackend, DetectionResult

logger = logging.getLogger(__name__)


@dataclass
class MatchSettings:
    """Per-target override for cascade detection behaviour.

    Attach to a Pattern or pass via config to customise how the
    CascadeDetector searches for a specific target.

    Attributes:
        preferred_backend: Try this backend first regardless of cost ordering.
        min_confidence: Minimum confidence threshold (0.0-1.0).
        max_backends: Maximum number of backends to try before giving up.
        search_region: Optional (x, y, width, height) to restrict the search area.
    """

    preferred_backend: str | None = None
    min_confidence: float = 0.8
    max_backends: int = 5
    search_region: tuple[int, int, int, int] | None = None


class CascadeDetector(DetectionBackend):
    """Unified detection fallback chain.

    Wraps multiple ``DetectionBackend`` instances and tries them in order
    of ascending estimated cost. If a backend returns results above the
    confidence threshold, the cascade short-circuits and returns immediately.

    Args:
        backends: List of backends to use. If ``None``, default backends
                  are created (template, feature, and any available
                  optional backends like OmniParser).

    Example::

        detector = CascadeDetector()
        results = detector.find(
            needle=pattern,
            haystack=screenshot,
            config={"needle_type": "template", "min_confidence": 0.8},
        )
    """

    def __init__(
        self,
        backends: list[DetectionBackend] | None = None,
        accessibility_capture: Any | None = None,
        ocr_engine: Any | None = None,
        llm_client: Any | None = None,
    ) -> None:
        if backends is None:
            backends = self._default_backends(
                accessibility_capture=accessibility_capture,
                ocr_engine=ocr_engine,
                llm_client=llm_client,
            )
        self._backends = sorted(backends, key=lambda b: b.estimated_cost_ms())

    def add_backend(self, backend: DetectionBackend) -> None:
        """Add a backend and re-sort by cost."""
        self._backends.append(backend)
        self._backends.sort(key=lambda b: b.estimated_cost_ms())

    def remove_backend(self, name: str) -> None:
        """Remove a backend by name."""
        self._backends = [b for b in self._backends if b.name != name]

    def find(self, needle: Any, haystack: Any, config: dict[str, Any]) -> list[DetectionResult]:
        """Run the cascade: try backends cheapest-first until one succeeds.

        Config keys:
            needle_type (str): Type of needle — ``template``, ``text``,
                ``accessibility_id``, ``role``, ``label``, ``description``.
                Defaults to ``"template"``.
            min_confidence (float): Minimum confidence threshold. Default 0.8.
            max_backends (int): Maximum backends to try. Default is all.
            find_all (bool): Whether to find all matches or just the best.
            match_settings (MatchSettings): Optional per-target overrides.

        Returns:
            List of DetectionResult from the first successful backend,
            filtered by confidence. Empty list if all backends fail.
        """
        # Apply MatchSettings overrides if present
        match_settings: MatchSettings | None = config.get("match_settings")
        if match_settings is not None:
            config = {**config}  # shallow copy to avoid mutating caller's dict
            config.setdefault("min_confidence", match_settings.min_confidence)
            if match_settings.search_region is not None:
                config.setdefault("search_region", match_settings.search_region)

        needle_type = config.get("needle_type", "template")
        min_confidence = config.get("min_confidence", 0.8)

        # Determine max_backends: MatchSettings takes priority, then config, then all
        if match_settings is not None:
            max_backends = match_settings.max_backends
        else:
            max_backends = config.get("max_backends", len(self._backends))

        # Build ordered backend list, putting preferred backend first
        ordered = self._ordered_backends(
            match_settings.preferred_backend if match_settings else None
        )

        needle_label = getattr(needle, "name", None) or str(type(needle).__name__)
        cascade_t0 = time.perf_counter()
        self._emit_cascade_started(needle_label, needle_type, min_confidence)

        backends_tried = 0
        for backend in ordered:
            if backends_tried >= max_backends:
                break
            if not backend.supports(needle_type):
                continue
            if not backend.is_available():
                logger.debug("Backend %s is not available, skipping", backend.name)
                continue

            backends_tried += 1
            t0 = time.perf_counter()
            try:
                results = backend.find(needle, haystack, config)
            except Exception:
                logger.warning(
                    "Backend %s raised an exception, skipping",
                    backend.name,
                    exc_info=True,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000
                self._emit_backend_tried(
                    backend.name,
                    needle_label,
                    elapsed_ms,
                    0,
                    False,
                    "exception",
                )
                continue
            elapsed_ms = (time.perf_counter() - t0) * 1000

            results = [r for r in results if r.confidence >= min_confidence]
            if results:
                logger.info(
                    "Cascade hit on backend %s (%d results, %.0fms)",
                    backend.name,
                    len(results),
                    elapsed_ms,
                )
                self._emit_backend_tried(
                    backend.name,
                    needle_label,
                    elapsed_ms,
                    len(results),
                    True,
                    None,
                )
                total_ms = (time.perf_counter() - cascade_t0) * 1000
                self._emit_cascade_hit(
                    needle_label,
                    backend.name,
                    backends_tried,
                    results[0].confidence,
                    total_ms,
                )
                return results

            self._emit_backend_tried(
                backend.name,
                needle_label,
                elapsed_ms,
                0,
                False,
                "below_threshold",
            )
            logger.debug(
                "Backend %s returned no results above threshold (%.0fms)",
                backend.name,
                elapsed_ms,
            )

        total_ms = (time.perf_counter() - cascade_t0) * 1000
        self._emit_cascade_miss(needle_label, backends_tried, total_ms)
        return []

    def _ordered_backends(self, preferred: str | None) -> list[DetectionBackend]:
        """Return backends with the preferred one moved to front."""
        if preferred is None:
            return self._backends

        preferred_list: list[DetectionBackend] = []
        rest: list[DetectionBackend] = []
        for b in self._backends:
            if b.name == preferred:
                preferred_list.append(b)
            else:
                rest.append(b)

        return preferred_list + rest

    # ------------------------------------------------------------------
    # Event emission helpers (zero-overhead when no listeners)
    # ------------------------------------------------------------------

    @staticmethod
    def _emit_cascade_started(needle_label: str, needle_type: str, min_confidence: float) -> None:
        try:
            from ...reporting.events import EventType, emit_event

            emit_event(
                EventType.CASCADE_STARTED,
                data={
                    "needle": needle_label,
                    "needle_type": needle_type,
                    "min_confidence": min_confidence,
                    "timestamp": time.time(),
                },
            )
        except Exception:
            pass  # Never let event emission break detection

    @staticmethod
    def _emit_backend_tried(
        backend_name: str,
        needle_label: str,
        elapsed_ms: float,
        result_count: int,
        success: bool,
        reason: str | None,
    ) -> None:
        try:
            from ...reporting.events import EventType, emit_event

            emit_event(
                EventType.CASCADE_BACKEND_TRIED,
                data={
                    "backend": backend_name,
                    "needle": needle_label,
                    "duration_ms": round(elapsed_ms, 1),
                    "result_count": result_count,
                    "success": success,
                    "reason": reason,
                    "timestamp": time.time(),
                },
            )
        except Exception:
            pass

    @staticmethod
    def _emit_cascade_hit(
        needle_label: str,
        backend_name: str,
        backends_tried: int,
        confidence: float,
        total_ms: float,
    ) -> None:
        try:
            from ...reporting.events import EventType, emit_event

            emit_event(
                EventType.CASCADE_HIT,
                data={
                    "needle": needle_label,
                    "winning_backend": backend_name,
                    "backends_tried": backends_tried,
                    "confidence": confidence,
                    "total_duration_ms": round(total_ms, 1),
                    "timestamp": time.time(),
                },
            )
        except Exception:
            pass

    @staticmethod
    def _emit_cascade_miss(
        needle_label: str,
        backends_tried: int,
        total_ms: float,
    ) -> None:
        try:
            from ...reporting.events import EventType, emit_event

            emit_event(
                EventType.CASCADE_MISS,
                data={
                    "needle": needle_label,
                    "backends_tried": backends_tried,
                    "total_duration_ms": round(total_ms, 1),
                    "timestamp": time.time(),
                },
            )
        except Exception:
            pass

    @staticmethod
    def _default_backends(
        accessibility_capture: Any | None = None,
        ocr_engine: Any | None = None,
        llm_client: Any | None = None,
    ) -> list[DetectionBackend]:
        """Create the default set of backends.

        Uses lazy imports to avoid circular dependencies.
        Only creates backends whose dependencies are importable.

        Args:
            accessibility_capture: Optional IAccessibilityCapture instance.
                When provided, an AccessibilityBackend is included as the
                cheapest backend (~5ms) — enabling UIA-first detection on
                Windows and CDP-first for web targets.
            ocr_engine: Optional IOCREngine instance. When provided, an
                OCRBackend is included for text-based detection (~300ms).
            llm_client: Optional VisionLLMClient instance. When provided, a
                VisionLLMBackend is included as the most expensive fallback
                (~2000ms).
        """
        backends: list[DetectionBackend] = []

        # Accessibility tree (~5ms) — fastest backend, queries structured
        # data instead of doing vision processing. Added first when an
        # IAccessibilityCapture is available (UIA on Windows, CDP for web).
        if accessibility_capture is not None:
            try:
                from .accessibility_backend import AccessibilityBackend

                backends.append(AccessibilityBackend(accessibility_capture))
            except ImportError:
                logger.warning("CascadeDetector: AccessibilityBackend unavailable")

            # Semantic accessibility (~10ms) — fuzzy matching for natural-
            # language descriptions against the accessibility tree.
            try:
                from .semantic_accessibility_backend import (
                    SemanticAccessibilityBackend,
                )

                backends.append(SemanticAccessibilityBackend(accessibility_capture))
            except ImportError:
                logger.debug("CascadeDetector: SemanticAccessibilityBackend unavailable")

        # Template matching (always available — uses OpenCV)
        try:
            from .template_match_backend import TemplateMatchBackend

            backends.append(TemplateMatchBackend())
        except ImportError:
            logger.warning("CascadeDetector: TemplateMatchBackend unavailable")

        # Edge-aware template matching (~40ms) — robust to theme/color changes
        try:
            from .edge_template_backend import EdgeTemplateMatchBackend

            backends.append(EdgeTemplateMatchBackend())
        except ImportError:
            logger.warning("CascadeDetector: EdgeTemplateMatchBackend unavailable")

        # Feature matching (always available — uses OpenCV)
        try:
            from .feature_match_backend import FeatureMatchBackend

            backends.append(FeatureMatchBackend())
        except ImportError:
            logger.warning("CascadeDetector: FeatureMatchBackend unavailable")

        # Invariant template matching (~120ms) — scale/rotation-tolerant fallback.
        try:
            from .invariant_match_backend import InvariantMatchBackend

            backends.append(InvariantMatchBackend())
        except ImportError:
            logger.debug("CascadeDetector: InvariantMatchBackend unavailable")

        # QATM (~200ms) — quality-aware deep template matching with VGG-19.
        # Only included when QONTINUI_QATM_ENABLED=true.
        try:
            from .qatm_backend import QATMBackend

            backend = QATMBackend()
            backends.append(backend)  # is_available() checks enabled flag
        except ImportError:
            pass  # Optional dependency (torch), silent skip

        # OCR text detection (~300ms) — finds elements by text content.
        # Only included when an IOCREngine instance is provided.
        if ocr_engine is not None:
            try:
                from .ocr_backend import OCRBackend

                backends.append(OCRBackend(ocr_engine))
            except ImportError:
                logger.debug("CascadeDetector: OCRBackend unavailable")

        # OmniParser (~1500ms) — zero-shot element detection with semantic labels.
        # Only included when QONTINUI_OMNIPARSER_ENABLED=true.
        try:
            from .omniparser_backend import OmniParserBackend

            backend = OmniParserBackend()
            backends.append(backend)  # is_available() checks enabled flag
        except ImportError:
            pass  # Optional dependency, silent skip

        # Vision LLM (~2000ms) — sends screenshot to a VLM for element
        # location. Most expensive backend, used as last resort.
        # Only included when a VisionLLMClient instance is provided.
        if llm_client is not None:
            try:
                from .vision_llm_backend import VisionLLMBackend

                backends.append(VisionLLMBackend(llm_client))
            except ImportError:
                logger.debug("CascadeDetector: VisionLLMBackend unavailable")

        # OmniParser service (~2000ms) — remote HTTP endpoint fallback.
        try:
            from .omniparser_service_backend import OmniParserServiceBackend

            backend = OmniParserServiceBackend()
            backends.append(backend)  # is_available() checks enabled + provider
        except ImportError:
            pass

        return backends

    def supports(self, needle_type: str) -> bool:
        """True if any backend supports this needle type."""
        return any(b.supports(needle_type) for b in self._backends)

    def estimated_cost_ms(self) -> float:
        """Cost of the cheapest backend (best case)."""
        if not self._backends:
            return 0.0
        return self._backends[0].estimated_cost_ms()

    @property
    def name(self) -> str:
        return "cascade"

    @property
    def backends(self) -> list[DetectionBackend]:
        """The ordered list of backends."""
        return list(self._backends)
