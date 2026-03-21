"""Tests for detection backend properties and cascade integration.

Verifies:
- All concrete backends can be instantiated
- supports(), estimated_cost_ms(), name are correct
- Backend ordering in cascade is correct
- FindExecutor.with_cascade() factory works
- Semantic matcher handles realistic Florence-2 captions
"""

import sys
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

from qontinui.find.backends.cascade import CascadeDetector
from qontinui.find.semantic_matcher import (
    match_element_by_description,
)

# ===========================================================================
# Task 4: Backend property verification
# ===========================================================================


class TestTemplateMatchBackend:
    def test_instantiation_and_properties(self):
        from qontinui.find.backends.template_match_backend import TemplateMatchBackend

        b = TemplateMatchBackend()
        assert b.name == "template"
        assert b.estimated_cost_ms() == 20.0
        assert b.supports("template")
        assert not b.supports("text")
        assert b.is_available()


class TestEdgeTemplateMatchBackend:
    def test_instantiation_and_properties(self):
        from qontinui.find.backends.edge_template_backend import EdgeTemplateMatchBackend

        b = EdgeTemplateMatchBackend()
        assert b.name == "edge_template"
        assert 30 <= b.estimated_cost_ms() <= 50
        assert b.supports("template")
        assert not b.supports("description")
        assert b.is_available()


class TestFeatureMatchBackend:
    def test_instantiation_and_properties(self):
        from qontinui.find.backends.feature_match_backend import FeatureMatchBackend

        b = FeatureMatchBackend()
        assert b.name == "feature"
        assert b.estimated_cost_ms() == 100.0
        assert b.supports("template")
        assert not b.supports("text")
        assert b.is_available()


class TestInvariantMatchBackend:
    def test_instantiation_and_properties(self):
        from qontinui.find.backends.invariant_match_backend import InvariantMatchBackend

        b = InvariantMatchBackend()
        assert b.name == "invariant_template"
        assert 100 <= b.estimated_cost_ms() <= 200
        assert b.supports("template")
        assert b.is_available()


class TestOmniParserBackendProperties:
    def test_properties(self):
        from qontinui.find.backends.omniparser_backend import OmniParserBackend

        b = OmniParserBackend()
        assert b.name == "omniparser"
        assert b.estimated_cost_ms() == 1500.0
        assert b.supports("template")
        assert b.supports("text")
        assert b.supports("description")
        assert b.supports("semantic")
        assert not b.supports("accessibility_id")
        # Disabled by default
        assert b.is_available() is False

    def test_enabled_via_settings(self):
        from qontinui.find.backends.omniparser_backend import OmniParserBackend
        from qontinui.find.backends.omniparser_config import OmniParserSettings

        settings = OmniParserSettings(enabled=True)
        b = OmniParserBackend(settings=settings)
        assert b.is_available() is True


class TestOmniParserServiceBackendProperties:
    def test_properties(self):
        from qontinui.find.backends.omniparser_service_backend import OmniParserServiceBackend

        b = OmniParserServiceBackend()
        assert b.name == "omniparser_service"
        assert b.estimated_cost_ms() == 2000.0
        assert b.supports("description")
        assert b.supports("semantic")


# ===========================================================================
# Task 4: Cascade ordering verification
# ===========================================================================


class TestCascadeDefaultOrdering:
    def test_default_backends_ordered_by_cost(self):
        """Default backends should be sorted cheapest-first."""
        cascade = CascadeDetector()
        costs = [b.estimated_cost_ms() for b in cascade.backends]
        assert costs == sorted(
            costs
        ), f"Backends not sorted by cost: {[(b.name, b.estimated_cost_ms()) for b in cascade.backends]}"

    def test_default_includes_core_backends(self):
        """At minimum, template + feature backends should be present."""
        cascade = CascadeDetector()
        names = {b.name for b in cascade.backends}
        assert "template" in names, f"Missing template backend. Got: {names}"
        assert "feature" in names, f"Missing feature backend. Got: {names}"

    def test_omniparser_in_chain_but_unavailable(self):
        """OmniParser should be in the chain but is_available=False by default."""
        cascade = CascadeDetector()
        omni_backends = [b for b in cascade.backends if "omniparser" in b.name]
        # OmniParser backends are present in the chain
        assert len(omni_backends) >= 1, "OmniParser backends should be in chain"
        # But unavailable by default
        for b in omni_backends:
            assert b.is_available() is False


# ===========================================================================
# Task 5: FindExecutor.with_cascade() factory
# ===========================================================================


class TestFindExecutorFactory:
    def test_with_cascade_creates_cascade(self):
        from qontinui.find.find_executor import FindExecutor

        mock_provider = MagicMock()
        executor = FindExecutor.with_cascade(screenshot_provider=mock_provider)

        assert executor.cascade_detector is not None
        assert isinstance(executor.cascade_detector, CascadeDetector)
        assert executor.matcher is not None

    def test_with_cascade_default_matcher(self):
        """When no matcher is passed, TemplateMatcher is used."""
        from qontinui.find.find_executor import FindExecutor
        from qontinui.find.matchers.template_matcher import TemplateMatcher

        mock_provider = MagicMock()
        executor = FindExecutor.with_cascade(screenshot_provider=mock_provider)

        assert isinstance(executor.matcher, TemplateMatcher)

    def test_with_cascade_custom_matcher(self):
        from qontinui.find.find_executor import FindExecutor

        mock_provider = MagicMock()
        mock_matcher = MagicMock()
        executor = FindExecutor.with_cascade(
            screenshot_provider=mock_provider,
            matcher=mock_matcher,
        )

        assert executor.matcher is mock_matcher

    def test_with_cascade_passes_filters(self):
        from qontinui.find.find_executor import FindExecutor

        mock_provider = MagicMock()
        mock_filter = MagicMock()
        executor = FindExecutor.with_cascade(
            screenshot_provider=mock_provider,
            filters=[mock_filter],
        )

        assert mock_filter in executor.filters


# ===========================================================================
# Task 6: Semantic matching with realistic Florence-2 captions
# ===========================================================================


class TestSemanticMatcherFlorence2:
    """Test semantic matching against realistic Florence-2 output."""

    # Realistic Florence-2 captions
    FLORENCE2_CAPTIONS = [
        "a blue rectangular button with text 'Submit'",
        "a magnifying glass search icon",
        "a text input field with placeholder 'Search...'",
        "a red circular close button with X",
        "a navigation menu bar with Home, About, Contact links",
        "a checkbox with label 'I agree to the terms'",
        "a dropdown selector showing 'English'",
        "a small gear settings icon",
        "a green 'Save' button",
        "a text area for entering comments",
    ]

    def test_submit_button_match(self):
        matches = match_element_by_description("Submit button", self.FLORENCE2_CAPTIONS)
        assert len(matches) > 0
        assert matches[0].element_index == 0  # "blue rectangular button with text Submit"
        assert matches[0].score >= 0.5

    def test_search_field_match(self):
        matches = match_element_by_description("Search field", self.FLORENCE2_CAPTIONS)
        assert len(matches) > 0
        # Should match "text input field with placeholder 'Search...'"
        best = matches[0]
        assert best.element_index == 2

    def test_close_button_match(self):
        matches = match_element_by_description("close button", self.FLORENCE2_CAPTIONS)
        assert len(matches) > 0
        assert matches[0].element_index == 3  # "red circular close button with X"

    def test_settings_icon_match(self):
        matches = match_element_by_description("settings icon", self.FLORENCE2_CAPTIONS)
        assert len(matches) > 0
        assert matches[0].element_index == 7  # "small gear settings icon"

    def test_save_button_match(self):
        matches = match_element_by_description("Save button", self.FLORENCE2_CAPTIONS)
        assert len(matches) > 0
        assert matches[0].element_index == 8  # "green 'Save' button"

    def test_checkbox_agree_match(self):
        matches = match_element_by_description("agree checkbox", self.FLORENCE2_CAPTIONS)
        assert len(matches) > 0
        assert matches[0].element_index == 5  # "checkbox with label 'I agree'"

    def test_dropdown_match(self):
        matches = match_element_by_description(
            "language dropdown",
            self.FLORENCE2_CAPTIONS,
            min_similarity=0.3,
        )
        assert len(matches) > 0
        # Dropdown element should be among the matches
        dropdown_indices = [m.element_index for m in matches]
        assert 6 in dropdown_indices

    def test_short_description_vs_long_caption(self):
        """Short user descriptions should match verbose Florence-2 captions."""
        matches = match_element_by_description(
            "Submit",
            ["a blue rectangular button with text 'Submit'"],
        )
        assert len(matches) > 0
        assert matches[0].score >= 0.5

    def test_type_bonus_with_florence2(self):
        """Element type synonyms should boost matching."""
        matches = match_element_by_description(
            "Save button",
            ["Save", "Save"],
            element_types=["button", "label"],
        )
        if len(matches) >= 2:
            button_match = next(m for m in matches if m.element_index == 0)
            label_match = next(m for m in matches if m.element_index == 1)
            assert button_match.score >= label_match.score

    def test_keyword_coverage_handles_asymmetry(self):
        """Coverage score should be high when short desc keywords are all in label."""
        matches = match_element_by_description(
            "gear icon",
            ["a small gear-shaped settings icon in the top-right corner"],
            min_similarity=0.3,
        )
        assert len(matches) > 0
        assert matches[0].score >= 0.4

    def test_no_false_positives(self):
        """Unrelated descriptions should not match."""
        matches = match_element_by_description(
            "password field",
            self.FLORENCE2_CAPTIONS,
            min_similarity=0.6,
        )
        # No element in FLORENCE2_CAPTIONS is a password field
        assert len(matches) == 0


class TestKeywordCoverage:
    """Test the new _keyword_coverage function directly."""

    def test_full_coverage(self):
        from qontinui.find.semantic_matcher import _keyword_coverage

        desc = {"submit", "button"}
        label = {"blue", "rectangular", "button", "text", "submit"}
        assert _keyword_coverage(desc, label) == 1.0

    def test_partial_coverage(self):
        from qontinui.find.semantic_matcher import _keyword_coverage

        desc = {"submit", "button", "blue"}
        label = {"submit", "button"}
        assert abs(_keyword_coverage(desc, label) - 2.0 / 3) < 0.01

    def test_no_coverage(self):
        from qontinui.find.semantic_matcher import _keyword_coverage

        desc = {"submit", "button"}
        label = {"search", "icon"}
        assert _keyword_coverage(desc, label) == 0.0

    def test_empty_desc(self):
        from qontinui.find.semantic_matcher import _keyword_coverage

        assert _keyword_coverage(set(), {"submit"}) == 0.0

    def test_empty_label(self):
        from qontinui.find.semantic_matcher import _keyword_coverage

        assert _keyword_coverage({"submit"}, set()) == 0.0
