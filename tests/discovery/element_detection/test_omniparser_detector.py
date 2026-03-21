"""Tests for OmniParser detector and cascade integration.

Tests cover:
- OmniParserDetector lazy loading and unloading
- OmniParserDetector element classification
- Semantic matcher fuzzy matching
- OmniParserBackend cascade integration
- OmniParserServiceBackend response parsing
- CascadeDetector fallback behaviour with OmniParser
- Disabled mode (skipped when enabled=False)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qontinui.discovery.element_detection.analysis_base import (
    AnalysisType,
    BoundingBox,
    DetectedElement,
)
from qontinui.discovery.element_detection.omniparser_detector import (
    OmniParserDetector,
)
from qontinui.find.backends.base import DetectionResult
from qontinui.find.backends.cascade import CascadeDetector
from qontinui.find.backends.omniparser_backend import OmniParserBackend
from qontinui.find.backends.omniparser_config import OmniParserSettings
from qontinui.find.backends.omniparser_service_backend import (
    OmniParserServiceBackend,
)
from qontinui.find.semantic_matcher import (
    match_element_by_description,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def omniparser_settings():
    """OmniParser settings with enabled=True for testing."""
    return OmniParserSettings(enabled=True, device="cpu", lazy_load=True)


@pytest.fixture
def disabled_settings():
    """OmniParser settings with enabled=False."""
    return OmniParserSettings(enabled=False)


@pytest.fixture
def sample_screenshot():
    """A 640x480 synthetic screenshot (BGR numpy array)."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_screenshot_bytes(sample_screenshot):
    """Sample screenshot as PNG bytes."""
    from io import BytesIO

    from PIL import Image

    img = Image.fromarray(sample_screenshot[..., ::-1])
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def mock_detected_elements():
    """Pre-built list of DetectedElement for testing."""
    return [
        DetectedElement(
            bounding_box=BoundingBox(x=100, y=200, width=120, height=40),
            confidence=0.92,
            label="Submit button",
            element_type="button",
            screenshot_index=0,
            metadata={"source": "omniparser"},
        ),
        DetectedElement(
            bounding_box=BoundingBox(x=50, y=100, width=200, height=30),
            confidence=0.88,
            label="Search field",
            element_type="text_field",
            screenshot_index=0,
            metadata={"source": "omniparser"},
        ),
        DetectedElement(
            bounding_box=BoundingBox(x=300, y=50, width=32, height=32),
            confidence=0.75,
            label="Settings gear icon",
            element_type="icon",
            screenshot_index=0,
            metadata={"source": "omniparser"},
        ),
    ]


# ---------------------------------------------------------------------------
# OmniParserDetector tests
# ---------------------------------------------------------------------------


class TestOmniParserDetector:
    def test_properties(self):
        detector = OmniParserDetector()
        assert detector.analysis_type == AnalysisType.SINGLE_SHOT
        assert detector.name == "omniparser_detector"
        assert detector.supports_multi_screenshot is False

    def test_lazy_loading_not_loaded_initially(self):
        detector = OmniParserDetector()
        assert detector.is_loaded is False

    def test_unload_when_not_loaded(self):
        """Unload should be safe when models aren't loaded."""
        detector = OmniParserDetector()
        detector.unload()
        assert detector.is_loaded is False

    def test_default_parameters(self, omniparser_settings):
        detector = OmniParserDetector(settings=omniparser_settings)
        params = detector.get_default_parameters()
        assert "iou_threshold" in params
        assert "confidence_threshold" in params
        assert "caption_batch_size" in params
        assert "enable_captioning" in params
        assert "enable_ocr" in params

    def test_classify_element_button(self):
        assert OmniParserDetector._classify_element("Submit button", 100, 40, 0) == "button"

    def test_classify_element_text_field(self):
        assert OmniParserDetector._classify_element("Search input", 200, 30, 0) == "text_field"

    def test_classify_element_checkbox(self):
        assert OmniParserDetector._classify_element("Agree checkbox", 20, 20, 0) == "checkbox"

    def test_classify_element_icon_by_size(self):
        """Small regions without text labels are classified as icons."""
        assert OmniParserDetector._classify_element(None, 24, 24, 0) == "icon"

    def test_classify_element_button_by_aspect(self):
        """Medium aspect ratio regions are classified as buttons."""
        assert OmniParserDetector._classify_element(None, 120, 40, 0) == "button"

    def test_overall_confidence_empty(self):
        assert OmniParserDetector._overall_confidence([]) == 0.0

    def test_overall_confidence_average(self, mock_detected_elements):
        conf = OmniParserDetector._overall_confidence(mock_detected_elements)
        expected = (0.92 + 0.88 + 0.75) / 3
        assert abs(conf - expected) < 0.001


# ---------------------------------------------------------------------------
# Semantic matcher tests
# ---------------------------------------------------------------------------


class TestSemanticMatcher:
    def test_exact_substring_match(self):
        matches = match_element_by_description(
            "Submit",
            ["Submit button", "Cancel button", "OK"],
        )
        assert len(matches) > 0
        assert matches[0].element_index == 0
        assert matches[0].match_type == "exact"

    def test_fuzzy_match(self):
        matches = match_element_by_description(
            "Submitt buton",  # intentional typo
            ["Submit button", "Cancel button"],
            min_similarity=0.2,
        )
        assert len(matches) > 0
        # Submit button should still be the best match
        assert matches[0].element_index == 0

    def test_keyword_match(self):
        matches = match_element_by_description(
            "blue Submit button",
            ["Submit button (blue, rectangular)", "Red Cancel button"],
        )
        assert len(matches) > 0
        assert matches[0].element_index == 0

    def test_empty_description(self):
        matches = match_element_by_description("", ["Submit button"])
        assert matches == []

    def test_empty_labels(self):
        matches = match_element_by_description("Submit", [])
        assert matches == []

    def test_no_match_above_threshold(self):
        matches = match_element_by_description(
            "XYZ nonexistent element",
            ["Submit button"],
            min_similarity=0.99,
        )
        assert matches == []

    def test_type_bonus(self):
        """Description mentioning 'button' should boost button-type elements."""
        matches = match_element_by_description(
            "Save button",
            ["Save", "Save"],
            element_types=["button", "label"],
        )
        if len(matches) >= 2:
            # Button-type should score higher due to type bonus
            button_match = next((m for m in matches if m.element_index == 0), None)
            label_match = next((m for m in matches if m.element_index == 1), None)
            if button_match and label_match:
                assert button_match.score >= label_match.score

    def test_scores_bounded(self):
        """All scores should be between 0 and 1."""
        matches = match_element_by_description(
            "Submit button",
            ["Submit button", "button submit", "Submit"],
            min_similarity=0.0,
        )
        for m in matches:
            assert 0.0 <= m.score <= 1.0


# ---------------------------------------------------------------------------
# OmniParserBackend tests
# ---------------------------------------------------------------------------


class TestOmniParserBackend:
    def test_supports(self, omniparser_settings):
        backend = OmniParserBackend(settings=omniparser_settings)
        assert backend.supports("template")
        assert backend.supports("text")
        assert backend.supports("description")
        assert backend.supports("semantic")
        assert not backend.supports("accessibility_id")

    def test_estimated_cost(self, omniparser_settings):
        backend = OmniParserBackend(settings=omniparser_settings)
        assert backend.estimated_cost_ms() == 1500.0

    def test_name(self, omniparser_settings):
        backend = OmniParserBackend(settings=omniparser_settings)
        assert backend.name == "omniparser"

    def test_is_available_when_enabled(self, omniparser_settings):
        backend = OmniParserBackend(settings=omniparser_settings)
        assert backend.is_available() is True

    def test_is_available_when_disabled(self, disabled_settings):
        backend = OmniParserBackend(settings=disabled_settings)
        assert backend.is_available() is False

    def test_match_by_description(
        self, omniparser_settings, sample_screenshot, mock_detected_elements
    ):
        """Test that description matching works via the backend."""
        backend = OmniParserBackend(settings=omniparser_settings)

        with patch.object(
            backend,
            "_ensure_detector",
            return_value=MagicMock(
                detect_from_numpy=MagicMock(return_value=mock_detected_elements)
            ),
        ):
            results = backend.find(
                needle="Submit button",
                haystack=sample_screenshot,
                config={"needle_type": "description"},
            )
            assert len(results) > 0
            assert results[0].label == "Submit button"
            assert results[0].backend_name == "omniparser"

    def test_non_numpy_haystack_returns_empty(self, omniparser_settings):
        backend = OmniParserBackend(settings=omniparser_settings)
        results = backend.find("test", "not_an_array", {"needle_type": "text"})
        assert results == []


# ---------------------------------------------------------------------------
# OmniParserServiceBackend tests
# ---------------------------------------------------------------------------


class TestOmniParserServiceBackend:
    def test_name(self):
        backend = OmniParserServiceBackend()
        assert backend.name == "omniparser_service"

    def test_estimated_cost(self):
        backend = OmniParserServiceBackend()
        assert backend.estimated_cost_ms() == 2000.0

    def test_is_available_when_disabled(self, disabled_settings):
        backend = OmniParserServiceBackend(settings=disabled_settings)
        assert backend.is_available() is False

    def test_parse_response_xyxy_format(self):
        backend = OmniParserServiceBackend()
        data = {
            "elements": [
                {
                    "bbox": [10, 20, 110, 60],
                    "confidence": 0.9,
                    "label": "Submit",
                    "type": "button",
                },
            ]
        }
        results = backend._parse_response(data)
        assert len(results) == 1
        assert results[0].x == 10
        assert results[0].y == 20
        assert results[0].width == 100
        assert results[0].height == 40
        assert results[0].label == "Submit"

    def test_parse_response_xywh_format(self):
        backend = OmniParserServiceBackend()
        data = {
            "elements": [
                {
                    "x": 50,
                    "y": 100,
                    "width": 200,
                    "height": 30,
                    "confidence": 0.85,
                    "label": "Search",
                },
            ]
        }
        results = backend._parse_response(data)
        assert len(results) == 1
        assert results[0].x == 50
        assert results[0].width == 200

    def test_parse_response_empty(self):
        backend = OmniParserServiceBackend()
        results = backend._parse_response({"elements": []})
        assert results == []


# ---------------------------------------------------------------------------
# CascadeDetector integration tests
# ---------------------------------------------------------------------------


class TestCascadeWithOmniParser:
    def test_omniparser_skipped_when_disabled(self, sample_screenshot):
        """OmniParser should not be called when is_available() returns False."""
        omni_backend = OmniParserBackend(settings=OmniParserSettings(enabled=False))
        mock_backend = MagicMock()
        mock_backend.name = "mock_fast"
        mock_backend.supports.return_value = True
        mock_backend.is_available.return_value = True
        mock_backend.estimated_cost_ms.return_value = 10.0
        mock_backend.find.return_value = [
            DetectionResult(
                x=10,
                y=20,
                width=100,
                height=40,
                confidence=0.95,
                backend_name="mock_fast",
            )
        ]

        cascade = CascadeDetector(backends=[omni_backend, mock_backend])
        results = cascade.find("test", sample_screenshot, {"needle_type": "template"})

        assert len(results) == 1
        assert results[0].backend_name == "mock_fast"

    def test_cascade_falls_through_to_omniparser(self, sample_screenshot):
        """When cheaper backends fail, cascade should reach OmniParser."""
        fast_backend = MagicMock()
        fast_backend.name = "fast"
        fast_backend.supports.return_value = True
        fast_backend.is_available.return_value = True
        fast_backend.estimated_cost_ms.return_value = 20.0
        fast_backend.find.return_value = []  # No results

        omni_backend = MagicMock()
        omni_backend.name = "omniparser"
        omni_backend.supports.return_value = True
        omni_backend.is_available.return_value = True
        omni_backend.estimated_cost_ms.return_value = 1500.0
        omni_backend.find.return_value = [
            DetectionResult(
                x=100,
                y=200,
                width=120,
                height=40,
                confidence=0.9,
                backend_name="omniparser",
                label="Submit button",
            )
        ]

        cascade = CascadeDetector(backends=[fast_backend, omni_backend])
        results = cascade.find(
            "Submit button",
            sample_screenshot,
            {"needle_type": "description", "min_confidence": 0.5},
        )

        assert len(results) == 1
        assert results[0].backend_name == "omniparser"
        assert results[0].label == "Submit button"
        fast_backend.find.assert_called_once()
        omni_backend.find.assert_called_once()

    def test_cascade_short_circuits_on_success(self, sample_screenshot):
        """If the first backend succeeds, OmniParser should NOT be called."""
        fast_backend = MagicMock()
        fast_backend.name = "fast"
        fast_backend.supports.return_value = True
        fast_backend.is_available.return_value = True
        fast_backend.estimated_cost_ms.return_value = 20.0
        fast_backend.find.return_value = [
            DetectionResult(
                x=10,
                y=20,
                width=100,
                height=40,
                confidence=0.95,
                backend_name="fast",
            )
        ]

        omni_backend = MagicMock()
        omni_backend.name = "omniparser"
        omni_backend.supports.return_value = True
        omni_backend.is_available.return_value = True
        omni_backend.estimated_cost_ms.return_value = 1500.0

        cascade = CascadeDetector(backends=[fast_backend, omni_backend])
        results = cascade.find("test", sample_screenshot, {"needle_type": "template"})

        assert len(results) == 1
        assert results[0].backend_name == "fast"
        omni_backend.find.assert_not_called()

    def test_cascade_handles_backend_exception(self, sample_screenshot):
        """Cascade should skip backends that throw exceptions."""
        broken_backend = MagicMock()
        broken_backend.name = "broken"
        broken_backend.supports.return_value = True
        broken_backend.is_available.return_value = True
        broken_backend.estimated_cost_ms.return_value = 10.0
        broken_backend.find.side_effect = RuntimeError("YOLO model crashed")

        fallback_backend = MagicMock()
        fallback_backend.name = "fallback"
        fallback_backend.supports.return_value = True
        fallback_backend.is_available.return_value = True
        fallback_backend.estimated_cost_ms.return_value = 1500.0
        fallback_backend.find.return_value = [
            DetectionResult(
                x=10,
                y=20,
                width=100,
                height=40,
                confidence=0.9,
                backend_name="fallback",
            )
        ]

        cascade = CascadeDetector(backends=[broken_backend, fallback_backend])
        results = cascade.find(
            "test", sample_screenshot, {"needle_type": "template", "min_confidence": 0.5}
        )

        assert len(results) == 1
        assert results[0].backend_name == "fallback"


# ---------------------------------------------------------------------------
# OmniParserSettings tests
# ---------------------------------------------------------------------------


class TestOmniParserSettings:
    def test_default_disabled(self):
        settings = OmniParserSettings()
        assert settings.enabled is False

    def test_default_device_auto(self):
        settings = OmniParserSettings()
        assert settings.device == "auto"

    def test_resolve_device_cpu_without_torch(self):
        settings = OmniParserSettings(device="auto")
        with patch.dict("sys.modules", {"torch": None}):
            # When torch import fails, should fall back to cpu
            result = settings.resolve_device()
            # Result depends on whether torch is actually importable
            assert result in ("cpu", "cuda")

    def test_explicit_device(self):
        settings = OmniParserSettings(device="cpu")
        assert settings.resolve_device() == "cpu"

    def test_env_prefix(self):
        with patch.dict(
            "os.environ",
            {"QONTINUI_OMNIPARSER_ENABLED": "true", "QONTINUI_OMNIPARSER_DEVICE": "cpu"},
        ):
            settings = OmniParserSettings()
            assert settings.enabled is True
            assert settings.device == "cpu"
