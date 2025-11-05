"""Tests for BreakpointManager."""

import pytest

from qontinui.debugging import BreakpointManager, BreakpointType


class TestBreakpointManager:
    """Test suite for BreakpointManager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = BreakpointManager()

    def test_add_action_breakpoint(self):
        """Test adding action ID breakpoints."""
        bp_id = self.manager.add_action_breakpoint("action_123")
        assert bp_id is not None

        bp = self.manager.get_breakpoint(bp_id)
        assert bp is not None
        assert bp.type == BreakpointType.ACTION_ID
        assert bp.action_id == "action_123"
        assert bp.enabled is True
        assert bp.hit_count == 0

    def test_add_type_breakpoint(self):
        """Test adding action type breakpoints."""
        bp_id = self.manager.add_type_breakpoint("Click")
        assert bp_id is not None

        bp = self.manager.get_breakpoint(bp_id)
        assert bp is not None
        assert bp.type == BreakpointType.ACTION_TYPE
        assert bp.action_type == "Click"

    def test_add_conditional_breakpoint(self):
        """Test adding conditional breakpoints."""

        def condition(ctx):
            return ctx.get("match_count", 0) > 5

        bp_id = self.manager.add_conditional_breakpoint(condition, condition_str="match_count > 5")

        bp = self.manager.get_breakpoint(bp_id)
        assert bp is not None
        assert bp.type == BreakpointType.CONDITIONAL
        assert bp.condition is not None
        assert bp.condition_str == "match_count > 5"

    def test_add_error_breakpoint(self):
        """Test adding error breakpoints."""
        bp_id = self.manager.add_error_breakpoint()
        bp = self.manager.get_breakpoint(bp_id)
        assert bp is not None
        assert bp.type == BreakpointType.ERROR

    def test_add_breakpoint_validation(self):
        """Test validation when adding breakpoints."""
        # Missing action_id for ACTION_ID type
        with pytest.raises(ValueError):
            self.manager.add_breakpoint(BreakpointType.ACTION_ID)

        # Missing action_type for ACTION_TYPE type
        with pytest.raises(ValueError):
            self.manager.add_breakpoint(BreakpointType.ACTION_TYPE)

        # Missing condition for CONDITIONAL type
        with pytest.raises(ValueError):
            self.manager.add_breakpoint(BreakpointType.CONDITIONAL)

    def test_remove_breakpoint(self):
        """Test removing breakpoints."""
        bp_id = self.manager.add_action_breakpoint("action_1")
        assert self.manager.get_breakpoint(bp_id) is not None

        removed = self.manager.remove_breakpoint(bp_id)
        assert removed is True
        assert self.manager.get_breakpoint(bp_id) is None

        # Test removing non-existent breakpoint
        removed = self.manager.remove_breakpoint("nonexistent")
        assert removed is False

    def test_enable_disable_breakpoint(self):
        """Test enabling and disabling breakpoints."""
        bp_id = self.manager.add_action_breakpoint("action_1")
        bp = self.manager.get_breakpoint(bp_id)
        assert bp.enabled is True

        # Disable
        result = self.manager.disable_breakpoint(bp_id)
        assert result is True
        assert bp.enabled is False

        # Enable
        result = self.manager.enable_breakpoint(bp_id)
        assert result is True
        assert bp.enabled is True

        # Test with non-existent breakpoint
        assert self.manager.enable_breakpoint("nonexistent") is False
        assert self.manager.disable_breakpoint("nonexistent") is False

    def test_list_breakpoints(self):
        """Test listing all breakpoints."""
        assert len(self.manager.list_breakpoints()) == 0

        bp1 = self.manager.add_action_breakpoint("action_1")
        bp2 = self.manager.add_type_breakpoint("Click")
        bp3 = self.manager.add_error_breakpoint()

        breakpoints = self.manager.list_breakpoints()
        assert len(breakpoints) == 3

        bp_ids = [bp.id for bp in breakpoints]
        assert bp1 in bp_ids
        assert bp2 in bp_ids
        assert bp3 in bp_ids

    def test_clear_all(self):
        """Test clearing all breakpoints."""
        self.manager.add_action_breakpoint("action_1")
        self.manager.add_type_breakpoint("Click")
        self.manager.add_error_breakpoint()

        assert len(self.manager.list_breakpoints()) == 3

        count = self.manager.clear_all()
        assert count == 3
        assert len(self.manager.list_breakpoints()) == 0

    def test_check_action_id_breakpoint(self):
        """Test checking action ID breakpoints."""
        self.manager.add_action_breakpoint("action_123")

        # Should trigger
        should_break, triggered = self.manager.check_breakpoint({"action_id": "action_123"})
        assert should_break is True
        assert len(triggered) == 1
        assert triggered[0].hit_count == 1

        # Should not trigger
        should_break, triggered = self.manager.check_breakpoint({"action_id": "other_action"})
        assert should_break is False
        assert len(triggered) == 0

    def test_check_type_breakpoint(self):
        """Test checking action type breakpoints."""
        self.manager.add_type_breakpoint("Click")

        # Should trigger
        should_break, triggered = self.manager.check_breakpoint({"action_type": "Click"})
        assert should_break is True
        assert len(triggered) == 1

        # Should not trigger
        should_break, triggered = self.manager.check_breakpoint({"action_type": "Find"})
        assert should_break is False

    def test_check_conditional_breakpoint(self):
        """Test checking conditional breakpoints."""

        def condition(ctx):
            return ctx.get("match_count", 0) > 5

        self.manager.add_conditional_breakpoint(condition)

        # Should trigger
        should_break, triggered = self.manager.check_breakpoint({"match_count": 10})
        assert should_break is True

        # Should not trigger
        should_break, triggered = self.manager.check_breakpoint({"match_count": 3})
        assert should_break is False

    def test_check_error_breakpoint(self):
        """Test checking error breakpoints."""
        self.manager.add_error_breakpoint()

        # Should trigger
        should_break, triggered = self.manager.check_breakpoint({"has_error": True})
        assert should_break is True

        # Should not trigger
        should_break, triggered = self.manager.check_breakpoint({"has_error": False})
        assert should_break is False

    def test_disabled_breakpoint_not_triggered(self):
        """Test that disabled breakpoints don't trigger."""
        bp_id = self.manager.add_action_breakpoint("action_1")
        self.manager.disable_breakpoint(bp_id)

        should_break, triggered = self.manager.check_breakpoint({"action_id": "action_1"})
        assert should_break is False
        assert len(triggered) == 0

    def test_hit_count(self):
        """Test breakpoint hit counting."""
        bp_id = self.manager.add_action_breakpoint("action_1")
        bp = self.manager.get_breakpoint(bp_id)

        assert bp.hit_count == 0

        # Hit it multiple times
        for _ in range(5):
            self.manager.check_breakpoint({"action_id": "action_1"})

        assert bp.hit_count == 5

    def test_multiple_breakpoints_trigger(self):
        """Test that multiple breakpoints can trigger simultaneously."""
        self.manager.add_action_breakpoint("action_1")
        self.manager.add_type_breakpoint("Click")

        context = {"action_id": "action_1", "action_type": "Click"}
        should_break, triggered = self.manager.check_breakpoint(context)

        assert should_break is True
        assert len(triggered) == 2

    def test_statistics(self):
        """Test breakpoint statistics."""
        self.manager.add_action_breakpoint("action_1")
        bp2_id = self.manager.add_type_breakpoint("Click")
        self.manager.disable_breakpoint(bp2_id)
        self.manager.add_error_breakpoint()

        # Trigger some breakpoints
        self.manager.check_breakpoint({"action_id": "action_1"})
        self.manager.check_breakpoint({"action_id": "action_1"})
        self.manager.check_breakpoint({"has_error": True})

        stats = self.manager.get_statistics()

        assert stats["total_breakpoints"] == 3
        assert stats["enabled_breakpoints"] == 2
        assert stats["disabled_breakpoints"] == 1

        assert "action_id" in stats["by_type"]
        assert stats["by_type"]["action_id"]["count"] == 1
        assert stats["by_type"]["action_id"]["hits"] == 2

        assert "error" in stats["by_type"]
        assert stats["by_type"]["error"]["hits"] == 1

    def test_format_breakpoint(self):
        """Test breakpoint formatting."""
        bp_id = self.manager.add_action_breakpoint("action_123")
        bp = self.manager.get_breakpoint(bp_id)

        formatted = self.manager.format_breakpoint(bp)

        assert "action_id" in formatted
        assert "action_123" in formatted
        assert "enabled" in formatted
        assert "hits=0" in formatted

    def test_repr(self):
        """Test string representation."""
        self.manager.add_action_breakpoint("action_1")
        self.manager.add_type_breakpoint("Click")

        repr_str = repr(self.manager)
        assert "BreakpointManager" in repr_str
        assert "breakpoints=2" in repr_str
