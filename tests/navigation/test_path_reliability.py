"""Tests for TransitionReliability."""

import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path for direct import
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from qontinui.navigation.path_reliability import (
    TransitionReliability,
    get_transition_reliability,
    set_transition_reliability,
)


class TestTransitionReliability:
    """Tests for TransitionReliability class."""

    def test_initial_reliability(self):
        """Test reliability score for unknown transition."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            reliability = TransitionReliability(persistence_path=f.name)

            # Unknown transitions should return neutral score
            score = reliability.get_reliability("A", "B")
            assert score == 0.5  # Unknown returns neutral

    def test_record_success(self):
        """Test recording a successful transition."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            reliability = TransitionReliability(persistence_path=f.name)

            reliability.record_attempt("A", "B", success=True)
            score = reliability.get_reliability("A", "B")

            assert score == 1.0

    def test_record_failure(self):
        """Test recording a failed transition."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            reliability = TransitionReliability(persistence_path=f.name)

            reliability.record_attempt("A", "B", success=False)
            score = reliability.get_reliability("A", "B")

            assert score == 0.0

    def test_mixed_results(self):
        """Test reliability with mixed success/failure."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            reliability = TransitionReliability(persistence_path=f.name)

            # Record 3 successes and 1 failure
            reliability.record_attempt("A", "B", success=True)
            reliability.record_attempt("A", "B", success=True)
            reliability.record_attempt("A", "B", success=True)
            reliability.record_attempt("A", "B", success=False)

            score = reliability.get_reliability("A", "B", use_recency_weighting=False)

            # Without recency: 3/4 = 0.75
            assert 0.74 <= score <= 0.76

    def test_recency_weighting(self):
        """Test that recent results are weighted more heavily."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            reliability = TransitionReliability(persistence_path=f.name, recency_decay=0.9)

            # Record older successes, then a recent failure
            reliability.record_attempt("A", "B", success=True)
            reliability.record_attempt("A", "B", success=True)
            reliability.record_attempt("A", "B", success=True)
            reliability.record_attempt("A", "B", success=False)  # Most recent

            score_weighted = reliability.get_reliability("A", "B", use_recency_weighting=True)
            score_unweighted = reliability.get_reliability("A", "B", use_recency_weighting=False)

            # With recency weighting, recent failure should have more impact
            assert score_weighted < score_unweighted

    def test_cost_multiplier_reliable(self):
        """Test cost multiplier for reliable transition."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            reliability = TransitionReliability(persistence_path=f.name)

            # All successes
            for _ in range(10):
                reliability.record_attempt("A", "B", success=True)

            multiplier = reliability.get_cost_multiplier("A", "B")

            # Should be close to min_multiplier (1.0)
            assert multiplier <= 1.5

    def test_cost_multiplier_unreliable(self):
        """Test cost multiplier for unreliable transition."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            reliability = TransitionReliability(persistence_path=f.name)

            # All failures
            for _ in range(10):
                reliability.record_attempt("A", "B", success=False)

            multiplier = reliability.get_cost_multiplier("A", "B")

            # Should be at max_multiplier (10.0 by default)
            assert multiplier >= 9.0

    def test_custom_multiplier_range(self):
        """Test custom min/max multiplier range."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            reliability = TransitionReliability(persistence_path=f.name)

            # All failures
            for _ in range(10):
                reliability.record_attempt("A", "B", success=False)

            multiplier = reliability.get_cost_multiplier(
                "A", "B", min_multiplier=2.0, max_multiplier=5.0
            )

            # Should be at custom max
            assert 4.5 <= multiplier <= 5.0

    def test_get_failing_transitions(self):
        """Test getting list of failing transitions."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            reliability = TransitionReliability(persistence_path=f.name)

            # Create a transition that only fails (no successes, so is_failing=True)
            reliability.record_attempt("X", "Y", success=False)
            reliability.record_attempt("X", "Y", success=False)

            # Create a reliable transition (only successes)
            reliability.record_attempt("C", "D", success=True)
            reliability.record_attempt("C", "D", success=True)

            failing = reliability.get_failing_transitions(min_failures=2)

            assert len(failing) == 1
            assert failing[0].from_state == "X"
            assert failing[0].to_state == "Y"

    def test_record_duration(self):
        """Test recording transition duration."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            reliability = TransitionReliability(persistence_path=f.name)

            reliability.record_attempt("A", "B", success=True, duration_ms=100.0)
            reliability.record_attempt("A", "B", success=True, duration_ms=200.0)

            stats = reliability.get_stats("A", "B")
            assert stats is not None
            assert stats.avg_duration_ms == 150.0

    def test_record_failure_reason(self):
        """Test recording failure reason."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            reliability = TransitionReliability(persistence_path=f.name)

            reliability.record_attempt(
                "A", "B", success=False, failure_reason="Element not found"
            )

            stats = reliability.get_stats("A", "B")
            assert stats is not None
            assert stats.failures == 1

    def test_persistence(self):
        """Test that data persists across instances."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        # First instance
        reliability1 = TransitionReliability(persistence_path=path)
        reliability1.record_attempt("A", "B", success=True)
        reliability1.record_attempt("A", "B", success=False)
        reliability1.save()

        # Second instance
        reliability2 = TransitionReliability(persistence_path=path)
        score = reliability2.get_reliability("A", "B", use_recency_weighting=False)

        assert 0.4 <= score <= 0.6  # 1/2 = 0.5

    def test_clear_history(self):
        """Test clearing all reliability data."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            reliability = TransitionReliability(persistence_path=f.name)

            reliability.record_attempt("A", "B", success=True)
            reliability.record_attempt("C", "D", success=False)

            reliability.clear_history()

            # Should return neutral for unknown
            assert reliability.get_reliability("A", "B") == 0.5
            assert reliability.get_reliability("C", "D") == 0.5

    def test_multiple_transitions(self):
        """Test tracking multiple independent transitions."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            reliability = TransitionReliability(persistence_path=f.name)

            # A->B always succeeds
            for _ in range(5):
                reliability.record_attempt("A", "B", success=True)

            # B->C always fails
            for _ in range(5):
                reliability.record_attempt("B", "C", success=False)

            assert reliability.get_reliability("A", "B") == 1.0
            assert reliability.get_reliability("B", "C") == 0.0


class TestGlobalReliability:
    """Tests for global reliability functions."""

    def test_get_transition_reliability(self):
        """Test getting global reliability instance."""
        reliability = get_transition_reliability()
        assert reliability is not None
        assert isinstance(reliability, TransitionReliability)

    def test_set_transition_reliability(self):
        """Test setting global reliability instance."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            custom = TransitionReliability(persistence_path=f.name)
            set_transition_reliability(custom)

            reliability = get_transition_reliability()
            assert reliability is custom
