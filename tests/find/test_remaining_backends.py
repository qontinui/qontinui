"""Tests for backends requiring injected dependencies.

Verifies AccessibilityBackend, SemanticAccessibilityBackend, OCRBackend,
VisionLLMBackend with mocked dependencies. Also tests OmniParser auto-unload.
"""

from __future__ import annotations

import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Auto-mock missing external deps
# ---------------------------------------------------------------------------


class _MockFinder:
    _MOCK_PREFIXES = (
        "qontinui_schemas",
        "cv2",
        "pyautogui",
        "screeninfo",
        "mss",
        "pygetwindow",
        "pynput",
        "Xlib",
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


# ===========================================================================
# AccessibilityBackend tests
# ===========================================================================


class TestAccessibilityBackend:
    def _make_backend(self, nodes=None):
        from qontinui.find.backends.accessibility_backend import AccessibilityBackend

        capture = MagicMock()
        capture.is_connected.return_value = True

        async def fake_find_nodes(selector):
            return nodes or []

        capture.find_nodes = fake_find_nodes
        return AccessibilityBackend(capture)

    def test_properties(self):
        backend = self._make_backend()
        assert backend.name == "accessibility"
        assert backend.estimated_cost_ms() == 5.0
        assert backend.supports("accessibility_id")
        assert backend.supports("role")
        assert backend.supports("label")
        assert not backend.supports("template")
        assert not backend.supports("description")

    def test_is_available(self):
        backend = self._make_backend()
        assert backend.is_available() is True

    def test_is_available_disconnected(self):
        from qontinui.find.backends.accessibility_backend import AccessibilityBackend

        capture = MagicMock()
        capture.is_connected.return_value = False
        backend = AccessibilityBackend(capture)
        assert backend.is_available() is False

    def test_find_by_accessibility_id(self):
        node = SimpleNamespace(
            bounds=(100, 200, 50, 30),
            name="Submit",
            role="button",
            ref="btn1",
        )
        backend = self._make_backend(nodes=[node])
        results = backend.find(
            "submitBtn",
            None,
            {"needle_type": "accessibility_id"},
        )
        assert len(results) == 1
        assert results[0].x == 100
        assert results[0].y == 200
        assert results[0].width == 50
        assert results[0].height == 30
        assert results[0].confidence == 1.0
        assert results[0].label == "Submit"
        assert results[0].backend_name == "accessibility"

    def test_find_by_role(self):
        node = SimpleNamespace(bounds=(10, 20, 100, 40), name="Search", role="textbox")
        backend = self._make_backend(nodes=[node])
        results = backend.find("textbox", None, {"needle_type": "role"})
        assert len(results) == 1

    def test_find_empty_results(self):
        backend = self._make_backend(nodes=[])
        results = backend.find("missing", None, {"needle_type": "label"})
        assert results == []

    def test_find_non_string_needle_returns_empty(self):
        backend = self._make_backend()
        results = backend.find(123, None, {"needle_type": "label"})
        assert results == []

    def test_find_unsupported_needle_type_returns_empty(self):
        backend = self._make_backend()
        results = backend.find("test", None, {"needle_type": "template"})
        assert results == []

    def test_node_without_bounds_skipped(self):
        node = SimpleNamespace(name="NoBounds", role="button")
        backend = self._make_backend(nodes=[node])
        results = backend.find("test", None, {"needle_type": "label"})
        assert results == []


# ===========================================================================
# SemanticAccessibilityBackend tests
# ===========================================================================


class TestSemanticAccessibilityBackend:
    def test_properties(self):
        from qontinui.find.backends.semantic_accessibility_backend import (
            SemanticAccessibilityBackend,
        )

        capture = MagicMock()
        capture.is_connected.return_value = True
        backend = SemanticAccessibilityBackend(capture)
        assert backend.name == "semantic_accessibility"
        assert backend.estimated_cost_ms() == 10.0
        assert backend.supports("description")
        assert backend.supports("semantic")
        assert not backend.supports("template")
        assert not backend.supports("accessibility_id")


# ===========================================================================
# OCRBackend tests
# ===========================================================================


class TestOCRBackend:
    def _make_backend(self, find_text_result=None, find_all_result=None):
        from qontinui.find.backends.ocr_backend import OCRBackend

        engine = MagicMock()

        if find_text_result is not None:
            engine.find_text.return_value = find_text_result
        else:
            engine.find_text.return_value = None

        if find_all_result is not None:
            engine.find_all_text.return_value = find_all_result
        else:
            engine.find_all_text.return_value = []

        return OCRBackend(engine)

    def test_properties(self):
        backend = self._make_backend()
        assert backend.name == "ocr"
        assert backend.estimated_cost_ms() == 300.0
        assert backend.supports("text")
        assert not backend.supports("template")
        assert not backend.supports("description")

    def test_find_text_single(self):
        match = SimpleNamespace(
            region=SimpleNamespace(x=50, y=100, width=200, height=30),
            text="Submit",
            similarity=0.95,
        )
        backend = self._make_backend(find_text_result=match)

        from PIL import Image

        haystack = Image.new("RGB", (640, 480))
        results = backend.find("Submit", haystack, {"needle_type": "text", "min_confidence": 0.8})
        assert len(results) == 1
        assert results[0].x == 50
        assert results[0].width == 200
        assert results[0].confidence == 0.95
        assert results[0].label == "Submit"
        assert results[0].backend_name == "ocr"

    def test_find_text_all(self):
        matches = [
            SimpleNamespace(
                region=SimpleNamespace(x=50, y=100, width=100, height=20),
                text="Save",
                similarity=0.9,
            ),
            SimpleNamespace(
                region=SimpleNamespace(x=200, y=100, width=100, height=20),
                text="Save As",
                similarity=0.85,
            ),
        ]
        backend = self._make_backend(find_all_result=matches)

        from PIL import Image

        haystack = Image.new("RGB", (640, 480))
        results = backend.find(
            "Save",
            haystack,
            {"needle_type": "text", "find_all": True, "min_confidence": 0.5},
        )
        assert len(results) == 2
        # Should be sorted by confidence
        assert results[0].confidence >= results[1].confidence

    def test_find_non_string_needle_returns_empty(self):
        backend = self._make_backend()
        results = backend.find(123, None, {"needle_type": "text"})
        assert results == []

    def test_find_none_haystack_returns_empty(self):
        backend = self._make_backend()
        results = backend.find("text", None, {"needle_type": "text"})
        assert results == []


# ===========================================================================
# VisionLLMBackend tests
# ===========================================================================


class TestVisionLLMBackend:
    def _make_backend(self, location=None, available=True):
        from qontinui.find.backends.vision_llm_backend import VisionLLMBackend

        client = MagicMock()
        client.is_available = available
        client.find_element.return_value = location
        return VisionLLMBackend(client)

    def test_properties(self):
        backend = self._make_backend()
        assert backend.name == "vision_llm"
        assert backend.estimated_cost_ms() == 2000.0
        assert backend.supports("template")
        assert backend.supports("text")
        assert backend.supports("description")
        assert not backend.supports("accessibility_id")

    def test_is_available(self):
        backend = self._make_backend(available=True)
        assert backend.is_available() is True

    def test_is_available_when_unavailable(self):
        backend = self._make_backend(available=False)
        assert backend.is_available() is False

    def test_find_with_description(self):
        location = SimpleNamespace(
            x=150,
            y=250,
            confidence=0.85,
            region=(130, 230, 40, 40),
            description="Submit button",
        )
        backend = self._make_backend(location=location)

        from PIL import Image

        haystack = Image.new("RGB", (640, 480))
        results = backend.find(
            "Submit button",
            haystack,
            {"needle_type": "description", "min_confidence": 0.5},
        )
        assert len(results) == 1
        assert results[0].x == 130
        assert results[0].y == 230
        assert results[0].width == 40
        assert results[0].confidence == 0.85
        assert results[0].backend_name == "vision_llm"

    def test_find_returns_empty_on_no_match(self):
        backend = self._make_backend(location=None)

        from PIL import Image

        haystack = Image.new("RGB", (640, 480))
        results = backend.find("nonexistent element", haystack, {"needle_type": "description"})
        assert results == []

    def test_find_with_pattern_needle(self):
        """Pattern.name should be used as description."""
        location = SimpleNamespace(
            x=100,
            y=200,
            confidence=0.9,
            region=(90, 190, 20, 20),
            description="icon",
        )
        backend = self._make_backend(location=location)

        pattern = SimpleNamespace(name="settings_icon")
        from PIL import Image

        haystack = Image.new("RGB", (640, 480))
        results = backend.find(
            pattern, haystack, {"needle_type": "template", "min_confidence": 0.5}
        )
        assert len(results) == 1

    def test_find_below_confidence_returns_empty(self):
        location = SimpleNamespace(
            x=100,
            y=200,
            confidence=0.3,
            region=(90, 190, 20, 20),
            description="low confidence",
        )
        backend = self._make_backend(location=location)

        from PIL import Image

        haystack = Image.new("RGB", (640, 480))
        results = backend.find("test", haystack, {"needle_type": "text", "min_confidence": 0.8})
        assert results == []


# ===========================================================================
# OmniParser auto-unload timer tests
# ===========================================================================


class TestOmniParserAutoUnload:
    def test_unload_after_idle(self):
        """Models should be unloaded after unload_after_seconds of inactivity."""
        from qontinui.discovery.element_detection.omniparser_detector import (
            OmniParserDetector,
        )
        from qontinui.find.backends.omniparser_config import OmniParserSettings

        settings = OmniParserSettings(enabled=True, unload_after_seconds=0.1)
        detector = OmniParserDetector(settings=settings)

        # Simulate loaded state
        detector._yolo_model = MagicMock()
        detector._caption_model = MagicMock()
        detector._caption_processor = MagicMock()
        detector._ocr_reader = MagicMock()
        detector._device = "cpu"
        detector._last_used = time.perf_counter() - 1.0  # 1 second ago

        assert detector.is_loaded is True

        # This should trigger unload since idle > 0.1s
        detector._maybe_unload_idle()

        assert detector.is_loaded is False

    def test_no_unload_when_recently_used(self):
        """Models should NOT be unloaded if recently used."""
        from qontinui.discovery.element_detection.omniparser_detector import (
            OmniParserDetector,
        )
        from qontinui.find.backends.omniparser_config import OmniParserSettings

        settings = OmniParserSettings(enabled=True, unload_after_seconds=60.0)
        detector = OmniParserDetector(settings=settings)

        detector._yolo_model = MagicMock()
        detector._device = "cpu"
        detector._last_used = time.perf_counter()  # Just now

        detector._maybe_unload_idle()

        assert detector.is_loaded is True

    def test_no_unload_when_disabled(self):
        """Auto-unload should not fire when unload_after_seconds=0."""
        from qontinui.discovery.element_detection.omniparser_detector import (
            OmniParserDetector,
        )
        from qontinui.find.backends.omniparser_config import OmniParserSettings

        settings = OmniParserSettings(enabled=True, unload_after_seconds=0.0)
        detector = OmniParserDetector(settings=settings)

        detector._yolo_model = MagicMock()
        detector._device = "cpu"
        detector._last_used = time.perf_counter() - 3600  # 1 hour ago

        detector._maybe_unload_idle()

        assert detector.is_loaded is True

    def test_no_unload_when_not_loaded(self):
        """Auto-unload should be a no-op when models aren't loaded."""
        from qontinui.discovery.element_detection.omniparser_detector import (
            OmniParserDetector,
        )
        from qontinui.find.backends.omniparser_config import OmniParserSettings

        settings = OmniParserSettings(enabled=True, unload_after_seconds=0.1)
        detector = OmniParserDetector(settings=settings)

        assert detector.is_loaded is False
        detector._maybe_unload_idle()  # Should not crash
        assert detector.is_loaded is False
