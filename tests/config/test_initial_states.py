"""Tests for resolve_initial_states function."""

import pytest

from qontinui.config.workflow_utils import (
    ResolvedInitialStates,
    get_initial_states_source,
    resolve_initial_states,
)


class TestResolveInitialStates:
    """Tests for the resolve_initial_states function."""

    @pytest.fixture
    def sample_config(self) -> dict:
        """Create a sample configuration for testing."""
        return {
            "workflows": [
                {
                    "id": "wf-with-initial",
                    "name": "Workflow With Initial States",
                    "initialStateIds": ["state-a", "state-b"],
                },
                {
                    "id": "wf-without-initial",
                    "name": "Workflow Without Initial States",
                },
                {
                    "id": "wf-snake-case",
                    "name": "Workflow with snake_case",
                    "initial_state_ids": ["state-c"],
                },
            ],
            "states": [
                {"id": "state-a", "name": "State A"},
                {"id": "state-b", "name": "State B"},
                {"id": "state-c", "name": "State C", "isInitial": True},
                {"id": "state-d", "name": "State D", "is_initial": True},
                {"id": "state-e", "name": "State E"},
            ],
        }

    def test_override_takes_priority(self, sample_config: dict) -> None:
        """Override IDs should take priority over everything else."""
        result = resolve_initial_states(
            sample_config,
            "wf-with-initial",
            override_ids=["override-state-1", "override-state-2"],
        )

        assert result.state_ids == ["override-state-1", "override-state-2"]
        assert result.source == "override"

    def test_workflow_initial_state_ids(self, sample_config: dict) -> None:
        """Workflow's initialStateIds should be used when no override."""
        result = resolve_initial_states(sample_config, "wf-with-initial")

        assert result.state_ids == ["state-a", "state-b"]
        assert result.source == "workflow"
        assert len(result.states) == 2
        assert result.states[0]["id"] == "state-a"
        assert result.states[0]["name"] == "State A"

    def test_workflow_initial_state_ids_snake_case(self, sample_config: dict) -> None:
        """Should support snake_case initial_state_ids."""
        result = resolve_initial_states(sample_config, "wf-snake-case")

        assert result.state_ids == ["state-c"]
        assert result.source == "workflow"

    def test_falls_back_to_defaults(self, sample_config: dict) -> None:
        """Falls back to states with isInitial=true when workflow has none."""
        result = resolve_initial_states(sample_config, "wf-without-initial")

        assert result.source == "defaults"
        # Should include both state-c (isInitial) and state-d (is_initial)
        assert "state-c" in result.state_ids
        assert "state-d" in result.state_ids
        assert len(result.state_ids) == 2

    def test_nonexistent_workflow(self, sample_config: dict) -> None:
        """Nonexistent workflow should fall back to defaults."""
        result = resolve_initial_states(sample_config, "nonexistent-workflow")

        assert result.source == "defaults"
        assert "state-c" in result.state_ids

    def test_empty_override_is_ignored(self, sample_config: dict) -> None:
        """Empty override list should be ignored."""
        result = resolve_initial_states(
            sample_config,
            "wf-with-initial",
            override_ids=[],
        )

        assert result.state_ids == ["state-a", "state-b"]
        assert result.source == "workflow"

    def test_state_names_are_resolved(self, sample_config: dict) -> None:
        """State names should be resolved from the states list."""
        result = resolve_initial_states(sample_config, "wf-with-initial")

        assert result.states[0]["name"] == "State A"
        assert result.states[1]["name"] == "State B"

    def test_unknown_state_id_uses_id_as_name(self, sample_config: dict) -> None:
        """Unknown state IDs should use the ID as the name."""
        result = resolve_initial_states(
            sample_config,
            "wf-with-initial",
            override_ids=["unknown-state"],
        )

        assert result.states[0]["id"] == "unknown-state"
        assert result.states[0]["name"] == "unknown-state"

    def test_to_dict_serialization(self, sample_config: dict) -> None:
        """to_dict should return a JSON-serializable dict."""
        result = resolve_initial_states(sample_config, "wf-with-initial")
        d = result.to_dict()

        assert d["stateIds"] == ["state-a", "state-b"]
        assert d["source"] == "workflow"
        assert len(d["states"]) == 2

    def test_empty_config(self) -> None:
        """Empty config should return empty defaults."""
        result = resolve_initial_states({}, "any-workflow")

        assert result.state_ids == []
        assert result.source == "defaults"
        assert result.states == []


class TestGetInitialStatesSource:
    """Tests for the get_initial_states_source convenience function."""

    @pytest.fixture
    def sample_config(self) -> dict:
        """Create a sample configuration."""
        return {
            "workflows": [
                {"id": "wf-1", "initialStateIds": ["state-a"]},
                {"id": "wf-2"},
            ],
            "states": [{"id": "state-a", "isInitial": True}],
        }

    def test_override_source(self, sample_config: dict) -> None:
        """Returns 'override' when override_ids provided."""
        source = get_initial_states_source(sample_config, "wf-1", override_ids=["x"])
        assert source == "override"

    def test_workflow_source(self, sample_config: dict) -> None:
        """Returns 'workflow' when workflow has initialStateIds."""
        source = get_initial_states_source(sample_config, "wf-1")
        assert source == "workflow"

    def test_defaults_source(self, sample_config: dict) -> None:
        """Returns 'defaults' when falling back to isInitial states."""
        source = get_initial_states_source(sample_config, "wf-2")
        assert source == "defaults"


class TestResolvedInitialStates:
    """Tests for the ResolvedInitialStates class."""

    def test_default_states_from_ids(self) -> None:
        """states should default to id=name when not provided."""
        result = ResolvedInitialStates(
            state_ids=["s1", "s2"],
            source="workflow",
        )

        assert result.states == [
            {"id": "s1", "name": "s1"},
            {"id": "s2", "name": "s2"},
        ]

    def test_custom_states(self) -> None:
        """Custom states should be preserved."""
        custom_states = [{"id": "s1", "name": "Custom Name"}]
        result = ResolvedInitialStates(
            state_ids=["s1"],
            source="workflow",
            states=custom_states,
        )

        assert result.states == custom_states
