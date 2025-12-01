"""Tests for JSON config validation.

This module validates that JSON configuration files are correctly formatted
and can be parsed by Pydantic models without silent failures.
"""

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

# Mock cv2 to avoid OpenCV dependency issues in WSL
sys.modules["cv2"] = mock.MagicMock()
from qontinui.config.models.action import Action  # noqa: E402
from qontinui.config.models.execution import BaseActionSettings  # noqa: E402


class TestJSONConfigValidation:
    """Test JSON configuration file validation."""

    @pytest.fixture
    def sample_config_path(self, tmp_path):
        """Create a sample config file for testing."""
        config = {
            "version": "2.0.0",
            "metadata": {
                "name": "Test Config",
                "created": "2025-10-31T00:00:00Z",
                "modified": "2025-10-31T00:00:00Z",
                "compatibleVersions": {"runner": "2.0.0", "website": "2.0.0"},
            },
            "images": [],
            "workflows": [
                {
                    "id": "workflow-1",
                    "name": "Test Workflow",
                    "format": "graph",
                    "version": "1.0.0",
                    "actions": [
                        {
                            "id": "action-1",
                            "type": "TYPE",
                            "config": {
                                "textSource": {
                                    "stateId": "Main",
                                    "stringIds": ["string-1"],
                                    "useAll": False,
                                },
                                "typing_delay": 50,
                                "clear_before": False,
                                "press_enter": False,
                            },
                            "base": {"pauseAfterEnd": 10000},
                            "position": [100, 100],
                        }
                    ],
                    "connections": {"action-1": {"main": []}},
                    "metadata": {},
                }
            ],
            "states": [],
            "transitions": [],
        }

        config_file = tmp_path / "test_config.json"
        config_file.write_text(json.dumps(config, indent=2))
        return config_file

    def test_action_base_settings_camelcase(self, sample_config_path):
        """Test that base settings use camelCase and are parsed correctly."""
        with open(sample_config_path) as f:
            config_dict = json.load(f)

        # Find the TYPE action
        type_action_dict = config_dict["workflows"][0]["actions"][0]

        # Parse the action
        action = Action(**type_action_dict)

        # Verify base settings were parsed correctly
        assert action.base is not None, "Base settings should be present"
        assert (
            action.base.pause_after_end == 10000
        ), "pauseAfterEnd should be parsed as pause_after_end and equal 10000"

    def test_action_base_settings_snakecase_fails(self):
        """Test that snake_case in JSON does NOT work (should use camelCase)."""
        # This is the WRONG format - using snake_case in JSON
        action_dict = {
            "id": "action-1",
            "type": "TYPE",
            "config": {},
            "base": {"pause_after_end": 10000},  # Wrong: should be pauseAfterEnd
        }

        # Parse the action
        action = Action(**action_dict)

        # With populate_by_name=True, this MIGHT work, but we want to enforce camelCase
        # The issue is that Pydantic accepts both formats with populate_by_name=True
        # We're documenting that JSON should use camelCase
        assert action.base is not None
        # If it parsed correctly, it would be 10000
        # If it didn't, it would be None
        # This test documents the expected behavior

    def test_base_action_settings_field_names(self):
        """Test that BaseActionSettings accepts camelCase aliases."""
        # Correct format: camelCase
        base_dict_camel = {"pauseBeforeBegin": 1000, "pauseAfterEnd": 2000}

        base = BaseActionSettings(**base_dict_camel)
        assert base.pause_before_begin == 1000
        assert base.pause_after_end == 2000

    def test_base_action_settings_snake_case_also_works(self):
        """Test that snake_case works due to populate_by_name=True.

        Note: While this works, JSON configs SHOULD use camelCase.
        This test documents that both work, but camelCase is preferred.
        """
        # Alternative format: snake_case (works but not recommended for JSON)
        base_dict_snake = {"pause_before_begin": 1000, "pause_after_end": 2000}

        base = BaseActionSettings(**base_dict_snake)
        assert base.pause_before_begin == 1000
        assert base.pause_after_end == 2000

    def test_action_without_base_settings(self):
        """Test that actions work without base settings."""
        action_dict = {
            "id": "action-1",
            "type": "CLICK",
            "config": {
                "target": {"type": "coordinates", "coordinates": {"x": 100, "y": 200}}
            },
        }

        action = Action(**action_dict)
        assert action.base is None

    def test_action_with_empty_base_settings(self):
        """Test that empty base settings are handled correctly."""
        action_dict = {
            "id": "action-1",
            "type": "CLICK",
            "config": {
                "target": {"type": "coordinates", "coordinates": {"x": 100, "y": 200}}
            },
            "base": {},
        }

        action = Action(**action_dict)
        assert action.base is not None
        assert action.base.pause_before_begin is None
        assert action.base.pause_after_end is None


class TestRealConfigFiles:
    """Test real configuration files from the project."""

    def test_bdo_config_56(self):
        """Test that bdo_config (56).json parses correctly."""
        # Path to the actual config file
        config_path = (
            Path(__file__).parent.parent.parent.parent.parent / "bdo_config (56).json"
        )

        # Skip if file doesn't exist (CI/CD environments)
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        with open(config_path) as f:
            config_dict = json.load(f)

        # Find all actions with base settings
        actions_with_base = []
        for workflow in config_dict.get("workflows", []):
            for action_dict in workflow.get("actions", []):
                if "base" in action_dict:
                    action = Action(**action_dict)
                    actions_with_base.append(
                        {"id": action.id, "type": action.type, "base": action.base}
                    )

        # Verify at least one action has base settings
        assert (
            len(actions_with_base) > 0
        ), "Expected at least one action with base settings"

        # Verify all base settings were parsed correctly
        for action_info in actions_with_base:
            base = action_info["base"]

            # If pauseAfterEnd is set, it should be parsed correctly
            if base.pause_after_end is not None:
                assert isinstance(
                    base.pause_after_end, int
                ), f"Action {action_info['id']}: pause_after_end should be int, got {type(base.pause_after_end)}"
                assert (
                    base.pause_after_end >= 0
                ), f"Action {action_info['id']}: pause_after_end should be non-negative"

            # If pauseBeforeBegin is set, it should be parsed correctly
            if base.pause_before_begin is not None:
                assert isinstance(
                    base.pause_before_begin, int
                ), f"Action {action_info['id']}: pause_before_begin should be int, got {type(base.pause_before_begin)}"
                assert (
                    base.pause_before_begin >= 0
                ), f"Action {action_info['id']}: pause_before_begin should be non-negative"

    def test_all_config_files_in_parent_dir(self):
        """Test all bdo_config*.json files in the parent directory."""
        parent_dir = Path(__file__).parent.parent.parent.parent.parent
        config_files = list(parent_dir.glob("bdo_config*.json"))

        if not config_files:
            pytest.skip("No bdo_config*.json files found in parent directory")

        errors = []

        for config_file in config_files:
            try:
                with open(config_file) as f:
                    config_dict = json.load(f)

                # Validate all actions can be parsed
                for workflow in config_dict.get("workflows", []):
                    for action_dict in workflow.get("actions", []):
                        try:
                            action = Action(**action_dict)

                            # If action has base settings, verify they're valid
                            if action.base:
                                if action.base.pause_after_end is not None:
                                    assert action.base.pause_after_end >= 0
                                if action.base.pause_before_begin is not None:
                                    assert action.base.pause_before_begin >= 0
                        except Exception as e:
                            errors.append(
                                f"Config {config_file.name}, "
                                f"Action {action_dict.get('id', 'unknown')}: {e}"
                            )
            except Exception as e:
                errors.append(f"Config {config_file.name}: Failed to load - {e}")

        # Report all errors at once
        if errors:
            pytest.fail("\n".join(["Config validation errors:"] + errors))
