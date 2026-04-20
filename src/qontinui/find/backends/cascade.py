"""CascadeDetector — unified detection fallback chain.

Wraps multiple DetectionBackends and implements graduated fallback:
try the cheapest backend first, fall through to more expensive backends
on failure. Short-circuits as soon as a backend returns results above
the confidence threshold.
"""

from __future__ import annotations

import logging
import os
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
    max_time_ms: float | None = None


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
        # Accessibility backend names known to the cascade. An "empty" result
        # from any of these signals broken accessibility → bypass mid-tier
        # vision backends and jump straight to the terminal fallback.
        self._accessibility_backend_names: set[str] = {
            "accessibility",
            "semantic_accessibility",
        }
        # Default terminal fallback: OmniParser (local or service). Registered
        # if available; callers may override via register_terminal_fallback().
        self._terminal_fallback: DetectionBackend | None = None
        # Pre-filter: wrap high-cost backends with InteractabilityFilter when
        # QONTINUI_OMNIPARSER_PREFILTER is set. Wrapping is done before
        # sorting so cost ordering is preserved.
        backends = self._maybe_wrap_prefilter(backends)
        self._backends = sorted(backends, key=lambda b: b.estimated_cost_ms())
        self._auto_register_terminal_fallback()

    def add_backend(self, backend: DetectionBackend) -> None:
        """Add a backend and re-sort by cost."""
        self._backends.append(backend)
        self._backends.sort(key=lambda b: b.estimated_cost_ms())

    def register_terminal_fallback(self, backend: DetectionBackend) -> None:
        """Register a backend as the terminal fallback.

        The terminal fallback is invoked when the accessibility tier
        (``AccessibilityBackend`` / ``SemanticAccessibilityBackend``) returns
        zero candidates. In that case the cascade bypasses the intermediate
        template/feature/OCR tiers — on broken-accessibility apps (legacy
        Win32, games, graphic canvases) those tiers will also fail and only
        waste latency — and jumps straight to the terminal fallback.

        Registering ``None`` clears the fallback.
        """
        self._terminal_fallback = backend

    @property
    def terminal_fallback(self) -> DetectionBackend | None:
        return self._terminal_fallback

    def _maybe_wrap_prefilter(self, backends: list[DetectionBackend]) -> list[DetectionBackend]:
        """Wrap high-cost backends with InteractabilityFilter if env flag on.

        Wrap threshold defaults to 1000ms: Template/Feature/Invariant/QATM/OCR
        tiers are untouched; OmniParser (~1500ms), VisionLLM (~2000ms), and
        OmniParser service (~2000ms) are wrapped. The OmniParser backend
        itself is also wrapped — pre-filter on its own output removes
        non-interactive captioned regions.
        """
        from .interactability_filter import InteractabilityFilter, prefilter_enabled

        if not prefilter_enabled():
            return backends

        try:
            threshold_ms = float(
                os.environ.get("QONTINUI_OMNIPARSER_PREFILTER_MIN_COST_MS", "1000")
            )
        except ValueError:
            threshold_ms = 1000.0

        wrapped: list[DetectionBackend] = []
        wrapped_names: list[str] = []
        for b in backends:
            if b.estimated_cost_ms() >= threshold_ms:
                wrapped.append(InteractabilityFilter(b))
                wrapped_names.append(b.name)
            else:
                wrapped.append(b)
        if wrapped_names:
            logger.info(
                "CascadeDetector: InteractabilityFilter wrapping %d backend(s) "
                "with cost >= %.0fms: %s",
                len(wrapped_names),
                threshold_ms,
                ", ".join(wrapped_names),
            )
        return wrapped

    def _auto_register_terminal_fallback(self) -> None:
        """Use OmniParser (local, then service) as the default terminal fallback.

        Picks the highest-cost omniparser backend already in the list — this
        is the one most likely to succeed on a broken-accessibility target.
        Silently no-ops if neither is available.
        """
        candidates: list[DetectionBackend] = []
        for b in self._backends:
            name = b.name
            # InteractabilityFilter-wrapped backend still reports the
            # underlying name, so this matches both raw and wrapped.
            if name in ("omniparser", "omniparser_service"):
                candidates.append(b)
        if not candidates:
            return
        # Prefer local ("omniparser") over remote service for lower latency.
        for b in candidates:
            if b.name == "omniparser":
                self._terminal_fallback = b
                return
        self._terminal_fallback = candidates[0]

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

        # Determine max_time_ms: MatchSettings takes priority, then config dict
        if match_settings is not None and match_settings.max_time_ms is not None:
            max_time_ms: float | None = match_settings.max_time_ms
        else:
            max_time_ms = config.get("max_time_ms")

        # Build ordered backend list, putting preferred backend first
        ordered = self._ordered_backends(
            match_settings.preferred_backend if match_settings else None
        )

        needle_label = getattr(needle, "name", None) or str(type(needle).__name__)
        cascade_t0 = time.perf_counter()
        self._emit_cascade_started(needle_label, needle_type, min_confidence)

        bypass_fallback = os.environ.get(
            "QONTINUI_CASCADE_DISABLE_A11Y_BYPASS", ""
        ).lower() not in ("1", "true", "yes", "on")

        accessibility_tried = False
        accessibility_all_empty = True
        already_ran: set[str] = set()
        backends_tried = 0
        for backend in ordered:
            if backends_tried >= max_backends:
                break
            if max_time_ms is not None:
                elapsed = (time.perf_counter() - cascade_t0) * 1000
                if elapsed >= max_time_ms:
                    logger.info(
                        "Cascade time budget exhausted (%.0fms >= %.0fms) after %d backends",
                        elapsed,
                        max_time_ms,
                        backends_tried,
                    )
                    self._emit_time_budget_exhausted(
                        needle_label, elapsed, max_time_ms, backends_tried
                    )
                    break
            if not backend.supports(needle_type):
                continue
            if not backend.is_available():
                logger.debug("Backend %s is not available, skipping", backend.name)
                continue

            backends_tried += 1
            already_ran.add(backend.name)
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
                # Treat an exception from an a11y backend the same as empty:
                # the accessibility tree is unusable for this target.
                if backend.name in self._accessibility_backend_names:
                    accessibility_tried = True
                continue
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if backend.name in self._accessibility_backend_names:
                accessibility_tried = True
                if results:
                    accessibility_all_empty = False

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
                # Normalize coordinates to 0.0-1.0 range if haystack dimensions available
                results = self._normalize_results(results, haystack)
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

            # Accessibility-empty bypass: once every a11y-tier backend has
            # run and returned empty, skip the template/feature/OCR tiers
            # and jump to the terminal fallback. Mid-tier vision-less
            # backends can't rescue a broken-accessibility target.
            if (
                bypass_fallback
                and accessibility_tried
                and accessibility_all_empty
                and self._terminal_fallback is not None
                and self._terminal_fallback.name not in already_ran
                and self._all_accessibility_done(ordered, backend)
            ):
                fb_results = self._invoke_terminal_fallback(
                    needle, haystack, config, needle_label, cascade_t0
                )
                if fb_results:
                    return fb_results
                break

        total_ms = (time.perf_counter() - cascade_t0) * 1000
        self._emit_cascade_miss(needle_label, backends_tried, total_ms)
        return []

    def _all_accessibility_done(
        self, ordered: list[DetectionBackend], current: DetectionBackend
    ) -> bool:
        """True if ``current`` is the last accessibility backend in ``ordered``.

        Used by the bypass path to wait until every a11y-tier backend has
        had a chance to run before giving up on accessibility.
        """
        a11y_in_order = [b for b in ordered if b.name in self._accessibility_backend_names]
        if not a11y_in_order:
            return False
        return current is a11y_in_order[-1]

    def _invoke_terminal_fallback(
        self,
        needle: Any,
        haystack: Any,
        config: dict[str, Any],
        needle_label: str,
        cascade_t0: float,
    ) -> list[DetectionResult]:
        """Run the registered terminal fallback backend directly.

        The fallback may already be present in ``self._backends`` — that's
        fine; running it here just short-circuits the mid-tier backends.
        Results go through the same confidence filter and normalisation
        as the main loop.
        """
        fb = self._terminal_fallback
        if fb is None:
            return []
        if not fb.is_available():
            logger.debug("CascadeDetector: terminal fallback %s not available", fb.name)
            return []
        needle_type = config.get("needle_type", "template")
        if not fb.supports(needle_type):
            logger.debug(
                "CascadeDetector: terminal fallback %s does not support " "needle_type=%s",
                fb.name,
                needle_type,
            )
            return []

        min_confidence = config.get("min_confidence", 0.8)
        logger.info(
            "CascadeDetector: accessibility tier empty, bypassing to terminal " "fallback %s",
            fb.name,
        )
        t0 = time.perf_counter()
        try:
            results = fb.find(needle, haystack, config)
        except Exception:
            logger.warning(
                "CascadeDetector: terminal fallback %s raised, cascade miss",
                fb.name,
                exc_info=True,
            )
            return []
        elapsed_ms = (time.perf_counter() - t0) * 1000
        results = [r for r in results if r.confidence >= min_confidence]
        self._emit_backend_tried(
            fb.name,
            needle_label,
            elapsed_ms,
            len(results),
            bool(results),
            "a11y_bypass_fallback",
        )
        if results:
            total_ms = (time.perf_counter() - cascade_t0) * 1000
            self._emit_cascade_hit(needle_label, fb.name, 1, results[0].confidence, total_ms)
            return self._normalize_results(results, haystack)
        return []

    def find_detections(self, needle: Any, haystack: Any, config: dict[str, Any]):
        """Run the cascade and return results as a Detections container.

        Same behaviour as ``find()`` but returns a ``Detections`` object
        instead of a plain list, enabling batch operations like filtering,
        merging, and NMS.

        Returns:
            A ``Detections`` container (possibly empty).
        """
        from ..detections import Detections

        results = self.find(needle, haystack, config)
        if not results:
            return Detections.empty()

        width = 0
        height = 0
        try:
            if hasattr(haystack, "size"):
                width, height = haystack.size
            elif hasattr(haystack, "shape"):
                height, width = haystack.shape[:2]
        except Exception:
            pass

        return Detections.from_detection_results(results, screen_width=width, screen_height=height)

    @staticmethod
    def _normalize_results(results: list[DetectionResult], haystack: Any) -> list[DetectionResult]:
        """Normalize result coordinates to 0.0-1.0 range using haystack dimensions.

        Tries to extract width/height from the haystack (PIL Image or numpy array).
        If dimensions can't be determined, returns results unchanged.
        """
        if not results:
            return results

        # Already normalized? Skip.
        if results[0].normalized_x is not None:
            return results

        width: int = 0
        height: int = 0
        try:
            # PIL Image
            if hasattr(haystack, "size"):
                width, height = haystack.size
            # numpy array (OpenCV format: height, width, channels)
            elif hasattr(haystack, "shape"):
                height, width = haystack.shape[:2]
        except Exception:
            pass

        if width <= 0 or height <= 0:
            return results

        return [r.normalize(width, height) for r in results]

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
    def _emit_time_budget_exhausted(
        needle_label: str,
        elapsed_ms: float,
        budget_ms: float,
        backends_tried: int,
    ) -> None:
        try:
            from ...reporting.events import EventType, emit_event

            emit_event(
                EventType.CASCADE_TIME_BUDGET_EXHAUSTED,
                data={
                    "needle": needle_label,
                    "elapsed_ms": round(elapsed_ms, 1),
                    "budget_ms": round(budget_ms, 1),
                    "backends_tried": backends_tried,
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
                from .semantic_accessibility_backend import SemanticAccessibilityBackend

                backends.append(
                    SemanticAccessibilityBackend(accessibility_capture, llm_client=llm_client)
                )
            except ImportError:
                logger.debug("CascadeDetector: SemanticAccessibilityBackend unavailable")

        # Template matching (always available — uses OpenCV)
        try:
            from .template_match_backend import TemplateMatchBackend

            backends.append(TemplateMatchBackend())
        except ImportError:
            logger.warning("CascadeDetector: TemplateMatchBackend unavailable")

        # Batch template matching (~25ms amortized) — matches multiple templates
        # against a single screenshot with cross-template NMS. Requires MTM.
        try:
            from .batch_template_match_backend import BatchTemplateMatchBackend

            backends.append(BatchTemplateMatchBackend())
        except ImportError:
            logger.debug("CascadeDetector: BatchTemplateMatchBackend unavailable")

        # Edge-aware template matching (~40ms) — robust to theme/color changes.
        # Reads DetectionConfig for enabled flag and Canny thresholds.
        try:
            from .edge_template_backend import EdgeTemplateMatchBackend

            edge_enabled = True
            edge_canny_low = 50
            edge_canny_high = 150
            try:
                from ...vision.verification.config import get_default_config

                det_cfg = get_default_config().detection
                edge_enabled = det_cfg.edge_template_enabled
                edge_canny_low = det_cfg.edge_canny_low
                edge_canny_high = det_cfg.edge_canny_high
            except Exception:
                pass  # Fall back to defaults if config unavailable

            backends.append(
                EdgeTemplateMatchBackend(
                    canny_low=edge_canny_low,
                    canny_high=edge_canny_high,
                    enabled=edge_enabled,
                )
            )
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

        # Scale-adaptive deep template matching (~140ms) — robust-TM
        # (kamata1729, APSIPA 2017) VGG-13 multi-scale correlation.
        # Only registered when QONTINUI_ENABLE_SCALE_ADAPTIVE_MATCH=1 so
        # the default cascade path is unaffected during rollout.
        try:
            from .scale_adaptive_backend import ScaleAdaptiveBackend, is_enabled

            if is_enabled():
                backends.append(ScaleAdaptiveBackend())
        except ImportError:
            pass  # Optional dependency (torch), silent skip

        # QATM (~200ms) — quality-aware deep template matching with VGG-19.
        # Only included when QONTINUI_QATM_ENABLED=true.
        try:
            from .qatm_backend import QATMBackend

            backends.append(QATMBackend())  # is_available() checks enabled flag
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

            backends.append(OmniParserBackend())  # is_available() checks enabled flag
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

            backends.append(OmniParserServiceBackend())  # is_available() checks enabled + provider
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
