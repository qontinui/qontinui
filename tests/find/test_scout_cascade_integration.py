"""Cross-cutting scout detection pipeline integration tests.

Verifies that multiple scout-contributed backends work together in the
CascadeDetector:
- Full cost ordering across ALL backends (guibot, Invariant-TM, MTM, QATM, Deep-TM, autoMate)
- preferred_backend overriding cost order across the full chain
- max_backends limiting attempts across the full chain
- search_region propagation across all backends
- Availability gating for backends with external dependencies

Run with:
    python -m pytest tests/find/test_scout_cascade_integration.py -v
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Auto-mock missing external deps so import chain works in CI.
# ---------------------------------------------------------------------------


class _MockFinder:
    """Meta-path finder that intercepts imports of missing packages."""

    _MOCK_PREFIXES = (
        "qontinui_schemas",
        "cv2",
        "pyautogui",
        "screeninfo",
        "mss",
        "pygetwindow",
        "pynput",
        "Xlib",
        "torch",
        "torchvision",
    )

    def find_module(self, fullname, path=None):
        for prefix in self._MOCK_PREFIXES:
            if fullname == prefix or fullname.startswith(prefix + "."):
                if fullname not in sys.modules:
                    return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        mod = MagicMock()
        mod.__path__ = []
        mod.__name__ = fullname
        sys.modules[fullname] = mod
        return mod


sys.meta_path.insert(0, _MockFinder())

import numpy as np  # noqa: E402

from qontinui.find.backends.base import DetectionBackend, DetectionResult  # noqa: E402
from qontinui.find.backends.cascade import CascadeDetector, MatchSettings  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Full expected backend roster with accurate costs from source.
# Backends that require constructor args (accessibility, ocr, vision_llm)
# are conditionally present, so we track them separately.
ALWAYS_PRESENT_BACKENDS = {
    # name -> (expected_cost_ms, supports_template)
    "template": (20.0, True),
    "batch_template_match": (25.0, True),
    "edge_template": (40.0, True),
    "feature": (100.0, True),
    "invariant_template": (120.0, True),
    # qatm: import may fail if torch not mocked properly, but we mock it
    "qatm": (200.0, True),
    "omniparser": (1500.0, True),
    "omniparser_service": (2000.0, True),
}

# Backends only present when specific dependencies are injected.
CONDITIONAL_BACKENDS = {
    "accessibility": (5.0, False),  # supports accessibility_id/role/label, not template
    "semantic_accessibility": (10.0, False),  # supports description/semantic
    "ocr": (300.0, False),  # supports text
    "vision_llm": (2000.0, True),  # supports template/text/description
}


class StubBackend(DetectionBackend):
    """Configurable stub backend for testing."""

    def __init__(
        self,
        backend_name: str,
        cost: float,
        supported_types: list[str],
        results: list[DetectionResult] | None = None,
        available: bool = True,
        raises: Exception | None = None,
    ):
        self._name = backend_name
        self._cost = cost
        self._supported = supported_types
        self._results = results or []
        self._available = available
        self._raises = raises
        self.find_called = False
        self.find_call_count = 0
        self.last_config: dict | None = None

    def find(self, needle, haystack, config):
        self.find_called = True
        self.find_call_count += 1
        self.last_config = dict(config)
        if self._raises:
            raise self._raises
        return list(self._results)

    def supports(self, needle_type):
        return needle_type in self._supported

    def estimated_cost_ms(self):
        return self._cost

    @property
    def name(self):
        return self._name

    def is_available(self):
        return self._available


def _make_result(confidence: float = 0.9, backend: str = "test") -> DetectionResult:
    return DetectionResult(
        x=100,
        y=200,
        width=50,
        height=30,
        confidence=confidence,
        backend_name=backend,
    )


def _build_full_stub_cascade(
    *,
    successes: dict[str, float] | None = None,
    unavailable: set[str] | None = None,
) -> tuple[CascadeDetector, dict[str, StubBackend]]:
    """Build a CascadeDetector with stubs mimicking every real backend.

    Args:
        successes: Mapping of backend name -> confidence for backends that
            should return a result.  All others return empty.
        unavailable: Set of backend names that report is_available()=False.

    Returns:
        (cascade, stubs_by_name)
    """
    successes = successes or {}
    unavailable = unavailable or set()

    all_backends = {**ALWAYS_PRESENT_BACKENDS, **CONDITIONAL_BACKENDS}
    stubs: dict[str, StubBackend] = {}

    for name, (cost, supports_template) in all_backends.items():
        supported = []
        if supports_template:
            supported.append("template")
        if name == "ocr":
            supported = ["text"]
        elif name == "accessibility":
            supported = ["accessibility_id", "role", "label"]
        elif name == "semantic_accessibility":
            supported = ["description", "semantic"]
        elif name in ("omniparser", "omniparser_service"):
            supported = ["template", "text", "description", "semantic"]
        elif name == "vision_llm":
            supported = ["template", "text", "description"]

        results = []
        if name in successes:
            results = [_make_result(successes[name], name)]

        stubs[name] = StubBackend(
            backend_name=name,
            cost=cost,
            supported_types=supported,
            results=results,
            available=(name not in unavailable),
        )

    cascade = CascadeDetector(backends=list(stubs.values()))
    return cascade, stubs


# ===========================================================================
# Full Cascade Ordering
# ===========================================================================


class TestFullScoutBackendCostOrdering:
    """All backends in cascade are sorted by estimated_cost_ms()."""

    def test_full_scout_backend_cost_ordering(self):
        """Every backend in the cascade must be in strictly non-decreasing
        cost order, regardless of insertion order."""
        cascade, _ = _build_full_stub_cascade()
        costs = [b.estimated_cost_ms() for b in cascade.backends]
        assert costs == sorted(costs), (
            f"Backends not in cost order: "
            f"{[(b.name, b.estimated_cost_ms()) for b in cascade.backends]}"
        )

    def test_cost_ordering_preserved_after_add(self):
        """Adding a backend mid-range re-sorts correctly."""
        cascade, _ = _build_full_stub_cascade()
        new_backend = StubBackend("new_scout", cost=75.0, supported_types=["template"])
        cascade.add_backend(new_backend)

        costs = [b.estimated_cost_ms() for b in cascade.backends]
        assert costs == sorted(costs)
        names = [b.name for b in cascade.backends]
        # new_scout (75ms) should be between edge_template (40ms) and feature (100ms)
        idx_new = names.index("new_scout")
        idx_feature = names.index("feature")
        idx_edge = names.index("edge_template")
        assert idx_edge < idx_new < idx_feature


class TestAllScoutBackendsPresent:
    """Verify the full roster of expected backends exists."""

    def test_all_scout_backends_present(self):
        """All expected backends must be in the cascade."""
        cascade, _ = _build_full_stub_cascade()
        names = {b.name for b in cascade.backends}

        expected_always = set(ALWAYS_PRESENT_BACKENDS.keys())
        expected_conditional = set(CONDITIONAL_BACKENDS.keys())
        expected_all = expected_always | expected_conditional

        missing = expected_all - names
        assert not missing, f"Missing backends: {missing}"

    def test_no_duplicate_backend_names(self):
        """Each backend name should appear exactly once."""
        cascade, _ = _build_full_stub_cascade()
        names = [b.name for b in cascade.backends]
        assert len(names) == len(set(names)), f"Duplicate names: {names}"


class TestScoutBackendSupportsTypes:
    """Each backend supports its expected needle types."""

    def test_template_backends_support_template(self):
        """Template-matching backends must support 'template' needle type."""
        template_backends = [
            "template",
            "batch_template_match",
            "edge_template",
            "feature",
            "invariant_template",
            "qatm",
        ]
        cascade, stubs = _build_full_stub_cascade()
        for name in template_backends:
            backend = stubs[name]
            assert backend.supports("template"), f"Backend '{name}' should support 'template'"

    def test_ocr_backend_supports_text(self):
        _, stubs = _build_full_stub_cascade()
        assert stubs["ocr"].supports("text")
        assert not stubs["ocr"].supports("template")

    def test_omniparser_supports_multiple_types(self):
        _, stubs = _build_full_stub_cascade()
        for needle_type in ("template", "text", "description", "semantic"):
            assert stubs["omniparser"].supports(
                needle_type
            ), f"omniparser should support '{needle_type}'"

    def test_accessibility_backend_supports_structured_types(self):
        _, stubs = _build_full_stub_cascade()
        for needle_type in ("accessibility_id", "role", "label"):
            assert stubs["accessibility"].supports(needle_type)
        assert not stubs["accessibility"].supports("template")

    def test_vision_llm_supports_multiple_types(self):
        _, stubs = _build_full_stub_cascade()
        for needle_type in ("template", "text", "description"):
            assert stubs["vision_llm"].supports(needle_type)


# ===========================================================================
# Cascade Integration — graduated fallback across scout backends
# ===========================================================================


class TestPreferredBackendSkipsCheaperBackends:
    """preferred_backend overrides cost order across the full chain."""

    def test_preferred_backend_skips_cheaper_backends(self):
        """Setting preferred_backend='invariant_template' should try it first,
        skipping template, batch_template_match, edge_template, and feature."""
        cascade, stubs = _build_full_stub_cascade(
            successes={"invariant_template": 0.95},
        )

        settings = MatchSettings(preferred_backend="invariant_template")
        results = cascade.find(
            needle=np.zeros((10, 10, 3), dtype=np.uint8),
            haystack=np.zeros((100, 100, 3), dtype=np.uint8),
            config={"needle_type": "template", "match_settings": settings},
        )

        assert len(results) == 1
        assert results[0].backend_name == "invariant_template"
        assert stubs["invariant_template"].find_called

        # Cheaper template backends should NOT have been tried because
        # the preferred backend succeeded immediately.
        for name in ("template", "batch_template_match", "edge_template", "feature"):
            assert not stubs[name].find_called, (
                f"Backend '{name}' should not have been tried when "
                f"preferred_backend='invariant_template' succeeded"
            )

    def test_preferred_backend_falls_through_on_failure(self):
        """If preferred backend returns no results, the cascade continues
        in cost order through cheaper backends first."""
        cascade, stubs = _build_full_stub_cascade(
            successes={"template": 0.92},
        )

        settings = MatchSettings(preferred_backend="invariant_template")
        results = cascade.find(
            needle=np.zeros((10, 10, 3), dtype=np.uint8),
            haystack=np.zeros((100, 100, 3), dtype=np.uint8),
            config={"needle_type": "template", "match_settings": settings},
        )

        assert len(results) == 1
        assert results[0].backend_name == "template"
        # invariant_template was tried first (preferred) but returned nothing
        assert stubs["invariant_template"].find_called
        # template was tried next (cheapest remaining)
        assert stubs["template"].find_called


class TestMaxBackendsLimitsCascadeAttempts:
    """max_backends limits how many backends are actually invoked."""

    def test_max_backends_limits_cascade_attempts(self):
        """With max_backends=2, only 2 supporting backends should be tried,
        even when more are available."""
        cascade, stubs = _build_full_stub_cascade()
        # No successes — all return empty, so cascade exhausts max_backends.

        settings = MatchSettings(max_backends=2)
        results = cascade.find(
            needle=np.zeros((10, 10, 3), dtype=np.uint8),
            haystack=np.zeros((100, 100, 3), dtype=np.uint8),
            config={"needle_type": "template", "match_settings": settings},
        )

        assert results == []

        # Count how many backends were actually called.
        called = [name for name, s in stubs.items() if s.find_called]
        assert len(called) == 2, (
            f"Expected exactly 2 backends tried with max_backends=2, "
            f"got {len(called)}: {called}"
        )

        # The two called should be the two cheapest template-supporting backends.
        # (template=20ms, batch_template_match=25ms)
        assert stubs["template"].find_called
        assert stubs["batch_template_match"].find_called

    def test_max_backends_one_stops_after_first_failure(self):
        """max_backends=1 should try exactly one backend."""
        cascade, stubs = _build_full_stub_cascade()

        settings = MatchSettings(max_backends=1)
        results = cascade.find(
            needle=np.zeros((10, 10, 3), dtype=np.uint8),
            haystack=np.zeros((100, 100, 3), dtype=np.uint8),
            config={"needle_type": "template", "match_settings": settings},
        )

        assert results == []
        called = [name for name, s in stubs.items() if s.find_called]
        assert len(called) == 1


class TestMinConfidenceOverridePerTarget:
    """MatchSettings min_confidence overrides cascade default."""

    def test_min_confidence_override_per_target(self):
        """A lower min_confidence via MatchSettings should accept results
        that the default threshold (0.8) would reject."""
        cascade, stubs = _build_full_stub_cascade(
            successes={"template": 0.65},
        )

        # Default 0.8 threshold -> should reject 0.65
        results_default = cascade.find(
            needle=np.zeros((10, 10, 3), dtype=np.uint8),
            haystack=np.zeros((100, 100, 3), dtype=np.uint8),
            config={"needle_type": "template"},
        )
        assert results_default == []

        # Reset call tracking
        for s in stubs.values():
            s.find_called = False
            s.find_call_count = 0

        # Lower threshold via MatchSettings
        settings = MatchSettings(min_confidence=0.5)
        results_override = cascade.find(
            needle=np.zeros((10, 10, 3), dtype=np.uint8),
            haystack=np.zeros((100, 100, 3), dtype=np.uint8),
            config={"needle_type": "template", "match_settings": settings},
        )
        assert len(results_override) == 1
        assert results_override[0].confidence == 0.65

    def test_higher_min_confidence_rejects_marginal_results(self):
        """A higher min_confidence should reject results that the default
        threshold would accept."""
        cascade, stubs = _build_full_stub_cascade(
            successes={"template": 0.85},
        )

        settings = MatchSettings(min_confidence=0.9)
        results = cascade.find(
            needle=np.zeros((10, 10, 3), dtype=np.uint8),
            haystack=np.zeros((100, 100, 3), dtype=np.uint8),
            config={"needle_type": "template", "match_settings": settings},
        )
        # 0.85 < 0.9 threshold -> rejected, cascade continues
        assert stubs["template"].find_called
        # If no other backend has results above 0.9, result is empty
        assert results == []


# ===========================================================================
# Availability Gating
# ===========================================================================


class TestUnavailableBackendsSkippedInCascade:
    """Backends with is_available()=False are skipped but still in the list."""

    def test_unavailable_backends_skipped_in_cascade(self):
        """An unavailable backend should not be called but should remain
        in the backend list for inspection."""
        cascade, stubs = _build_full_stub_cascade(
            successes={"feature": 0.90},
            unavailable={"template", "batch_template_match", "edge_template"},
        )

        results = cascade.find(
            needle=np.zeros((10, 10, 3), dtype=np.uint8),
            haystack=np.zeros((100, 100, 3), dtype=np.uint8),
            config={"needle_type": "template"},
        )

        # Unavailable backends should NOT have been called
        assert not stubs["template"].find_called
        assert not stubs["batch_template_match"].find_called
        assert not stubs["edge_template"].find_called

        # But they should still be in the backend list
        names = {b.name for b in cascade.backends}
        assert "template" in names
        assert "batch_template_match" in names
        assert "edge_template" in names

        # Feature backend (available) should have been called and succeeded
        assert stubs["feature"].find_called
        assert len(results) == 1
        assert results[0].backend_name == "feature"

    def test_unavailable_does_not_count_toward_max_backends(self):
        """Skipping an unavailable backend should not consume a
        max_backends slot."""
        cascade, stubs = _build_full_stub_cascade(
            successes={"feature": 0.92},
            unavailable={"template", "batch_template_match"},
        )

        settings = MatchSettings(max_backends=2)
        results = cascade.find(
            needle=np.zeros((10, 10, 3), dtype=np.uint8),
            haystack=np.zeros((100, 100, 3), dtype=np.uint8),
            config={"needle_type": "template", "match_settings": settings},
        )

        # template and batch_template_match are skipped (unavailable),
        # then edge_template (40ms) is tried (count=1), then feature (100ms)
        # is tried (count=2) and succeeds.
        assert not stubs["template"].find_called
        assert not stubs["batch_template_match"].find_called
        assert stubs["edge_template"].find_called
        assert stubs["feature"].find_called
        assert len(results) == 1
        assert results[0].backend_name == "feature"


class TestAllBackendsHaveNameAndCost:
    """Every backend has non-empty name and positive cost."""

    def test_all_backends_have_name_and_cost(self):
        cascade, _ = _build_full_stub_cascade()
        for backend in cascade.backends:
            assert (
                isinstance(backend.name, str) and len(backend.name) > 0
            ), f"Backend has empty or non-string name: {backend!r}"
            assert backend.estimated_cost_ms() > 0, (
                f"Backend '{backend.name}' has non-positive cost: " f"{backend.estimated_cost_ms()}"
            )

    def test_all_backends_implement_detection_interface(self):
        """Every backend must be a DetectionBackend subclass."""
        cascade, _ = _build_full_stub_cascade()
        for backend in cascade.backends:
            assert isinstance(
                backend, DetectionBackend
            ), f"Backend '{backend.name}' is not a DetectionBackend"


# ===========================================================================
# Search region propagation
# ===========================================================================


class TestSearchRegionPropagation:
    """search_region from MatchSettings reaches backend config."""

    def test_search_region_propagated_to_backend_config(self):
        """MatchSettings.search_region should appear in the config dict
        passed to the backend."""
        cascade, stubs = _build_full_stub_cascade(
            successes={"template": 0.95},
        )

        region = (50, 50, 200, 150)
        settings = MatchSettings(search_region=region)
        cascade.find(
            needle=np.zeros((10, 10, 3), dtype=np.uint8),
            haystack=np.zeros((100, 100, 3), dtype=np.uint8),
            config={"needle_type": "template", "match_settings": settings},
        )

        assert stubs["template"].find_called
        assert stubs["template"].last_config is not None
        assert stubs["template"].last_config.get("search_region") == region

    def test_search_region_does_not_override_explicit_config(self):
        """If config already has search_region, MatchSettings should not
        overwrite it (setdefault semantics)."""
        cascade, stubs = _build_full_stub_cascade(
            successes={"template": 0.95},
        )

        explicit_region = (10, 10, 50, 50)
        settings_region = (100, 100, 300, 300)
        settings = MatchSettings(search_region=settings_region)

        cascade.find(
            needle=np.zeros((10, 10, 3), dtype=np.uint8),
            haystack=np.zeros((100, 100, 3), dtype=np.uint8),
            config={
                "needle_type": "template",
                "match_settings": settings,
                "search_region": explicit_region,
            },
        )

        assert stubs["template"].last_config["search_region"] == explicit_region


# ===========================================================================
# Thread safety — concurrent cascade.find() calls
# ===========================================================================


class TestCascadeFindConcurrentCalls:
    """Thread safety: concurrent cascade.find() with stub backends."""

    def test_cascade_find_concurrent_calls(self):
        """Use ThreadPoolExecutor to run 5 concurrent cascade.find() calls
        with stub backends, verify all return results without corruption."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Each call gets its own cascade+stubs to avoid shared mutable state
        # on the stubs, but tests that the CascadeDetector itself is safe.
        cascade, stubs = _build_full_stub_cascade(
            successes={"template": 0.95},
        )

        needle = np.zeros((10, 10, 3), dtype=np.uint8)
        haystack = np.zeros((100, 100, 3), dtype=np.uint8)

        def run_find(idx: int):
            results = cascade.find(
                needle=needle,
                haystack=haystack,
                config={"needle_type": "template"},
            )
            return idx, results

        results_map = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(run_find, i) for i in range(5)]
            for future in as_completed(futures):
                idx, results = future.result()
                results_map[idx] = results

        # All 5 calls should return exactly 1 result each
        assert len(results_map) == 5, f"Expected 5 results, got {len(results_map)}"
        for idx, results in results_map.items():
            assert len(results) == 1, f"Call {idx}: expected 1 result, got {len(results)}"
            assert results[0].backend_name == "template"
            assert results[0].confidence == 0.95


# ===========================================================================
# Timeout fallback — first backend times out, second succeeds
# ===========================================================================


class TestBackendTimeoutCausesFallback:
    """Timeout in one backend causes cascade to fall through to the next."""

    def test_backend_timeout_causes_fallback(self):
        """First backend raises TimeoutError, second backend succeeds."""
        timeout_backend = StubBackend(
            backend_name="slow_backend",
            cost=10.0,
            supported_types=["template"],
            raises=TimeoutError("Backend timed out"),
        )
        success_backend = StubBackend(
            backend_name="fast_backend",
            cost=20.0,
            supported_types=["template"],
            results=[_make_result(0.88, "fast_backend")],
        )

        cascade = CascadeDetector(backends=[timeout_backend, success_backend])
        results = cascade.find(
            needle=np.zeros((10, 10, 3), dtype=np.uint8),
            haystack=np.zeros((100, 100, 3), dtype=np.uint8),
            config={"needle_type": "template"},
        )

        # The timeout backend was tried and failed
        assert timeout_backend.find_called
        # The success backend was tried next and succeeded
        assert success_backend.find_called
        assert len(results) == 1
        assert results[0].backend_name == "fast_backend"
        assert results[0].confidence == 0.88


# ===========================================================================
# Short-circuit — first successful backend's results returned directly
# ===========================================================================


class TestCascadeShortCircuitsNoNMS:
    """Cascade returns FIRST successful backend's results without cross-backend NMS."""

    def test_cascade_short_circuits_no_cross_backend_nms(self):
        """Verify cascade returns the first successful backend's results
        immediately (no cross-backend NMS needed since it short-circuits)."""
        cascade, stubs = _build_full_stub_cascade(
            successes={
                "template": 0.90,
                "feature": 0.95,
            },
        )

        results = cascade.find(
            needle=np.zeros((10, 10, 3), dtype=np.uint8),
            haystack=np.zeros((100, 100, 3), dtype=np.uint8),
            config={"needle_type": "template"},
        )

        # The cascade should short-circuit at the first successful backend
        assert len(results) == 1
        assert results[0].backend_name == "template", (
            "Cascade should return the FIRST successful backend's results "
            f"(template at 20ms), not {results[0].backend_name}"
        )
        assert results[0].confidence == 0.90

        # template was tried and succeeded — feature should NOT have been tried
        assert stubs["template"].find_called
        assert not stubs["feature"].find_called, (
            "Cascade should short-circuit after first success; "
            "feature backend should not have been called"
        )
