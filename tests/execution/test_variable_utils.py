"""Tests for variable utility functions."""

import tempfile
from pathlib import Path

import pytest

from qontinui.execution.variable_utils import (
    create_variable_snapshot,
    filter_variables_by_prefix,
    get_nested_variable,
    interpolate_variables,
    is_json_serializable,
    load_variables_from_json,
    merge_variable_scopes,
    resolve_variable_reference,
    restore_variable_snapshot,
    sanitize_for_persistence,
    save_variables_to_json,
    set_nested_variable,
    validate_variable_name,
)


class TestLoadSave:
    """Test loading and saving variables from/to JSON files."""

    def test_save_and_load_json(self):
        """Test saving and loading variables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "vars.json"

            vars_to_save = {"x": 1, "y": "text", "z": [1, 2, 3]}

            save_variables_to_json(vars_to_save, file_path)
            loaded_vars = load_variables_from_json(file_path)

            assert loaded_vars == vars_to_save

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_variables_from_json("/nonexistent/path/vars.json")

    def test_save_creates_parent_directory(self):
        """Test that save creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "nested" / "dir" / "vars.json"

            save_variables_to_json({"x": 1}, file_path)

            assert file_path.exists()
            assert load_variables_from_json(file_path) == {"x": 1}


class TestMergeScopes:
    """Test merging variable scopes."""

    def test_merge_with_last_precedence(self):
        """Test merging with later scopes overriding earlier."""
        scope1 = {"x": 1, "y": 2}
        scope2 = {"y": 20, "z": 3}
        scope3 = {"z": 30}

        merged = merge_variable_scopes(scope1, scope2, scope3)

        assert merged == {"x": 1, "y": 20, "z": 30}

    def test_merge_with_first_precedence(self):
        """Test merging with earlier scopes taking precedence."""
        scope1 = {"x": 1, "y": 2}
        scope2 = {"y": 20, "z": 3}
        scope3 = {"z": 30}

        merged = merge_variable_scopes(scope1, scope2, scope3, precedence="first")

        assert merged == {"x": 1, "y": 2, "z": 3}

    def test_merge_empty_scopes(self):
        """Test merging with empty scopes."""
        merged = merge_variable_scopes({}, {"x": 1}, {})
        assert merged == {"x": 1}


class TestSerialization:
    """Test JSON serialization validation."""

    def test_is_json_serializable(self):
        """Test checking JSON serializability."""
        assert is_json_serializable({"x": 1, "y": "text"})
        assert is_json_serializable([1, 2, 3])
        assert is_json_serializable("string")
        assert is_json_serializable(42)
        assert is_json_serializable(None)
        assert is_json_serializable(True)

        # Not serializable
        assert not is_json_serializable(lambda x: x)
        assert not is_json_serializable({1, 2, 3})


class TestVariableNameValidation:
    """Test variable name validation."""

    def test_valid_names(self):
        """Test valid variable names."""
        assert validate_variable_name("x")
        assert validate_variable_name("my_var")
        assert validate_variable_name("_private")
        assert validate_variable_name("var123")
        assert validate_variable_name("CamelCase")

    def test_invalid_names(self):
        """Test invalid variable names."""
        assert not validate_variable_name("")
        assert not validate_variable_name("123invalid")
        assert not validate_variable_name("my-var")
        assert not validate_variable_name("my var")
        assert not validate_variable_name("my.var")


class TestInterpolation:
    """Test variable interpolation in text."""

    def test_basic_interpolation(self):
        """Test basic variable interpolation."""
        variables = {"user": "alice", "count": 42}
        text = "Hello ${user}, count=${count}"

        result = interpolate_variables(text, variables)

        assert result == "Hello alice, count=42"

    def test_missing_variable(self):
        """Test interpolation with missing variable keeps placeholder."""
        variables = {"user": "alice"}
        text = "Hello ${user}, missing=${missing}"

        result = interpolate_variables(text, variables)

        assert result == "Hello alice, missing=${missing}"

    def test_no_variables_in_text(self):
        """Test text with no variables."""
        variables = {"x": 1}
        text = "No variables here"

        result = interpolate_variables(text, variables)

        assert result == "No variables here"


class TestVariableReference:
    """Test resolving variable references."""

    def test_resolve_reference(self):
        """Test resolving variable reference string."""
        variables = {"user_id": "12345", "name": "alice"}

        assert resolve_variable_reference("${user_id}", variables) == "12345"
        assert resolve_variable_reference("${name}", variables) == "alice"

    def test_resolve_non_reference(self):
        """Test that non-reference values pass through."""
        variables = {"x": 1}

        assert resolve_variable_reference("literal", variables) == "literal"
        assert resolve_variable_reference(42, variables) == 42
        assert resolve_variable_reference([1, 2, 3], variables) == [1, 2, 3]

    def test_resolve_missing_reference(self):
        """Test resolving missing variable keeps reference."""
        variables = {"x": 1}

        # Missing variable should return the original reference string
        assert resolve_variable_reference("${missing}", variables) == "${missing}"


class TestNestedVariables:
    """Test nested variable access."""

    def test_get_nested_variable(self):
        """Test getting nested variable with dot notation."""
        variables = {"user": {"name": "alice", "address": {"city": "NYC", "zip": "10001"}}}

        assert get_nested_variable(variables, "user.name") == "alice"
        assert get_nested_variable(variables, "user.address.city") == "NYC"
        assert get_nested_variable(variables, "user.address.zip") == "10001"

    def test_get_nested_missing(self):
        """Test getting missing nested variable returns default."""
        variables = {"user": {"name": "alice"}}

        assert get_nested_variable(variables, "user.missing") is None
        assert get_nested_variable(variables, "user.missing", default="N/A") == "N/A"

    def test_set_nested_variable(self):
        """Test setting nested variable with dot notation."""
        variables = {}

        assert set_nested_variable(variables, "user.name", "alice") is True
        assert variables == {"user": {"name": "alice"}}

        assert set_nested_variable(variables, "user.address.city", "NYC") is True
        assert variables == {"user": {"name": "alice", "address": {"city": "NYC"}}}

    def test_set_nested_without_creating(self):
        """Test setting nested variable without creating missing paths."""
        variables = {}

        assert set_nested_variable(variables, "user.name", "alice", create_missing=False) is False
        assert variables == {}


class TestFilterByPrefix:
    """Test filtering variables by prefix."""

    def test_filter_with_prefix(self):
        """Test filtering variables by prefix."""
        variables = {
            "app_name": "test",
            "app_version": "1.0",
            "user_id": "123",
            "user_name": "alice",
        }

        app_vars = filter_variables_by_prefix(variables, "app_")
        assert app_vars == {"app_name": "test", "app_version": "1.0"}

        user_vars = filter_variables_by_prefix(variables, "user_")
        assert user_vars == {"user_id": "123", "user_name": "alice"}

    def test_filter_with_strip_prefix(self):
        """Test filtering with prefix stripping."""
        variables = {"app_name": "test", "app_version": "1.0"}

        filtered = filter_variables_by_prefix(variables, "app_", strip_prefix=True)
        assert filtered == {"name": "test", "version": "1.0"}

    def test_filter_no_matches(self):
        """Test filtering with no matches."""
        variables = {"x": 1, "y": 2}

        filtered = filter_variables_by_prefix(variables, "app_")
        assert filtered == {}


class TestSanitization:
    """Test sanitizing variables for persistence."""

    def test_sanitize_valid_variables(self):
        """Test sanitizing valid variables."""
        variables = {"x": 1, "y": "text", "z": [1, 2, 3]}

        sanitized = sanitize_for_persistence(variables)

        assert sanitized == variables

    def test_sanitize_skip_non_serializable(self):
        """Test sanitizing skips non-serializable values."""
        variables = {
            "valid": "text",
            "func": lambda x: x,
            "set": {1, 2, 3},
        }

        sanitized = sanitize_for_persistence(variables, skip_non_serializable=True)

        assert sanitized == {"valid": "text"}
        assert "func" not in sanitized
        assert "set" not in sanitized

    def test_sanitize_convert_non_serializable(self):
        """Test sanitizing converts non-serializable to string."""
        variables = {"func": lambda x: x}

        sanitized = sanitize_for_persistence(variables, skip_non_serializable=False)

        assert "func" in sanitized
        assert isinstance(sanitized["func"], str)

    def test_sanitize_truncate_large_values(self):
        """Test sanitizing truncates large string values."""
        variables = {"large": "x" * 10000, "small": "y" * 10}

        sanitized = sanitize_for_persistence(variables, max_value_size=100)

        assert len(sanitized["large"]) == 100
        assert len(sanitized["small"]) == 10


class TestSnapshot:
    """Test variable snapshot creation and restoration."""

    def test_create_snapshot(self):
        """Test creating variable snapshot."""
        exec_vars = {"temp": 1}
        workflow_vars = {"user_id": "123"}
        global_vars = {"api_key": "secret"}

        snapshot = create_variable_snapshot(exec_vars, workflow_vars, global_vars)

        assert "timestamp" in snapshot
        assert "scopes" in snapshot
        assert "merged" in snapshot

        assert snapshot["scopes"]["execution"] == exec_vars
        assert snapshot["scopes"]["workflow"] == workflow_vars
        assert snapshot["scopes"]["global"] == global_vars

        # Merged should have all variables with proper precedence
        assert snapshot["merged"] == {
            "temp": 1,
            "user_id": "123",
            "api_key": "secret",
        }

    def test_restore_snapshot(self):
        """Test restoring variables from snapshot."""
        snapshot = {
            "timestamp": 123456789,
            "scopes": {
                "execution": {"temp": 1},
                "workflow": {"user_id": "123"},
                "global": {"api_key": "secret"},
            },
            "merged": {"temp": 1, "user_id": "123", "api_key": "secret"},
        }

        exec_vars, workflow_vars, global_vars = restore_variable_snapshot(snapshot)

        assert exec_vars == {"temp": 1}
        assert workflow_vars == {"user_id": "123"}
        assert global_vars == {"api_key": "secret"}

    def test_restore_invalid_snapshot(self):
        """Test restoring invalid snapshot raises error."""
        with pytest.raises(ValueError, match="Invalid snapshot"):
            restore_variable_snapshot({"invalid": "data"})

    def test_snapshot_roundtrip(self):
        """Test creating and restoring snapshot preserves data."""
        original_exec = {"a": 1}
        original_workflow = {"b": 2}
        original_global = {"c": 3}

        snapshot = create_variable_snapshot(original_exec, original_workflow, original_global)

        restored_exec, restored_workflow, restored_global = restore_variable_snapshot(snapshot)

        assert restored_exec == original_exec
        assert restored_workflow == original_workflow
        assert restored_global == original_global


class TestComplexScenarios:
    """Test complex real-world scenarios."""

    def test_config_management_scenario(self):
        """Test typical configuration management scenario."""
        # Load default config
        default_config = {
            "app_name": "MyApp",
            "app_version": "1.0.0",
            "app_debug": False,
        }

        # Load user overrides
        user_config = {"app_debug": True, "app_log_level": "DEBUG"}

        # Merge with user taking precedence
        merged = merge_variable_scopes(default_config, user_config)

        assert merged["app_name"] == "MyApp"  # From default
        assert merged["app_debug"] is True  # User override
        assert merged["app_log_level"] == "DEBUG"  # User addition

    def test_template_rendering_scenario(self):
        """Test template rendering with variable interpolation."""
        variables = {
            "user_name": "Alice",
            "order_id": "ORD-12345",
            "total": "99.99",
        }

        template = """
        Hello ${user_name},
        Your order ${order_id} for $${total} has been confirmed.
        """

        result = interpolate_variables(template, variables)

        assert "Alice" in result
        assert "ORD-12345" in result
        assert "$99.99" in result

    def test_hierarchical_config_scenario(self):
        """Test hierarchical configuration with nested values."""
        config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "credentials": {"username": "admin", "password": "secret"},
            }
        }

        # Access nested values
        assert get_nested_variable(config, "database.host") == "localhost"
        assert get_nested_variable(config, "database.port") == 5432
        assert get_nested_variable(config, "database.credentials.username") == "admin"

        # Update nested value
        set_nested_variable(config, "database.port", 3306)
        assert config["database"]["port"] == 3306
