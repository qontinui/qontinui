"""Tests for EnhancedVariableContext with three-tier storage."""

import json
import tempfile
import threading
import time
from pathlib import Path

import pytest

from qontinui.execution.enhanced_variable_context import EnhancedVariableContext


class TestBasicOperations:
    """Test basic get/set/delete operations."""

    def test_set_and_get_execution_scope(self):
        """Test setting and getting execution-scoped variables."""
        context = EnhancedVariableContext()
        context.set("temp_var", 42, scope="execution")

        assert context.get("temp_var") == 42
        assert context.get("temp_var", scope="execution") == 42
        assert context.exists("temp_var", scope="execution")

    def test_set_and_get_workflow_scope(self):
        """Test setting and getting workflow-scoped variables."""
        context = EnhancedVariableContext()
        context.set("user_id", "12345", scope="workflow")

        assert context.get("user_id") == "12345"
        assert context.get("user_id", scope="workflow") == "12345"
        assert context.exists("user_id", scope="workflow")

    def test_set_and_get_global_scope(self):
        """Test setting and getting global-scoped variables."""
        context = EnhancedVariableContext()
        context.set("api_key", "secret", scope="global")

        assert context.get("api_key") == "secret"
        assert context.get("api_key", scope="global") == "secret"
        assert context.exists("api_key", scope="global")

    def test_scope_precedence(self):
        """Test that execution > workflow > global precedence works."""
        context = EnhancedVariableContext()

        # Set same variable in all scopes
        context.set("x", 1, scope="global")
        context.set("x", 10, scope="workflow")
        context.set("x", 100, scope="execution")

        # Without scope, should get execution value (highest precedence)
        assert context.get("x") == 100

        # With specific scope, should get that scope's value
        assert context.get("x", scope="execution") == 100
        assert context.get("x", scope="workflow") == 10
        assert context.get("x", scope="global") == 1

    def test_scope_fallback(self):
        """Test fallback to lower precedence scopes."""
        context = EnhancedVariableContext()

        context.set("x", 1, scope="global")
        context.set("y", 2, scope="workflow")
        context.set("z", 3, scope="execution")

        # Should find variables in their respective scopes
        assert context.get("x") == 1  # Only in global
        assert context.get("y") == 2  # Only in workflow
        assert context.get("z") == 3  # Only in execution

    def test_default_value(self):
        """Test default value when variable not found."""
        context = EnhancedVariableContext()

        assert context.get("missing") is None
        assert context.get("missing", default=42) == 42
        assert context.get("missing", default="N/A") == "N/A"

    def test_delete_from_scope(self):
        """Test deleting variables from specific scope."""
        context = EnhancedVariableContext()

        context.set("x", 1, scope="execution")
        context.set("x", 10, scope="workflow")

        # Delete from execution scope
        assert context.delete("x", scope="execution") is True
        assert not context.exists("x", scope="execution")
        assert context.exists("x", scope="workflow")

        # Should now get workflow value
        assert context.get("x") == 10

    def test_delete_from_all_scopes(self):
        """Test deleting variable from all scopes."""
        context = EnhancedVariableContext()

        context.set("x", 1, scope="global")
        context.set("x", 10, scope="workflow")
        context.set("x", 100, scope="execution")

        # Delete from all scopes
        assert context.delete("x") is True
        assert not context.exists("x")

    def test_clear_scope(self):
        """Test clearing all variables in a scope."""
        context = EnhancedVariableContext()

        context.set("a", 1, scope="execution")
        context.set("b", 2, scope="execution")
        context.set("c", 3, scope="workflow")

        # Clear execution scope
        count = context.clear("execution")
        assert count == 2
        assert not context.exists("a")
        assert not context.exists("b")
        assert context.exists("c")  # Workflow var should remain


class TestGetAll:
    """Test get_all method for retrieving all variables."""

    def test_get_all_merged(self):
        """Test get_all with no scope returns merged dict."""
        context = EnhancedVariableContext()

        context.set("x", 1, scope="global")
        context.set("y", 2, scope="workflow")
        context.set("z", 3, scope="execution")
        context.set("x", 10, scope="execution")  # Override global

        merged = context.get_all()

        assert merged == {"x": 10, "y": 2, "z": 3}

    def test_get_all_by_scope(self):
        """Test get_all with specific scope."""
        context = EnhancedVariableContext()

        context.set("a", 1, scope="execution")
        context.set("b", 2, scope="workflow")
        context.set("c", 3, scope="global")

        assert context.get_all(scope="execution") == {"a": 1}
        assert context.get_all(scope="workflow") == {"b": 2}
        assert context.get_all(scope="global") == {"c": 3}


class TestPersistence:
    """Test file-based persistence for workflow and global scopes."""

    def test_save_and_load_workflow_vars(self):
        """Test saving and loading workflow variables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workflow_file = Path(tmpdir) / "workflow_vars.json"

            # Create context and set variables
            context1 = EnhancedVariableContext(workflow_file=workflow_file)
            context1.set("user_id", "12345", scope="workflow")
            context1.set("session_id", "abc123", scope="workflow")
            context1.save_to_file("workflow")

            # Create new context and load
            context2 = EnhancedVariableContext(workflow_file=workflow_file)
            count = context2.load_from_file("workflow")

            assert count == 2
            assert context2.get("user_id", scope="workflow") == "12345"
            assert context2.get("session_id", scope="workflow") == "abc123"

    def test_save_and_load_global_vars(self):
        """Test saving and loading global variables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            global_file = Path(tmpdir) / "global_vars.json"

            # Create context and set variables
            context1 = EnhancedVariableContext(global_file=global_file)
            context1.set("api_key", "secret", scope="global")
            context1.set("api_url", "https://api.example.com", scope="global")
            context1.save_to_file("global")

            # Create new context and load
            context2 = EnhancedVariableContext(global_file=global_file)
            count = context2.load_from_file("global")

            assert count == 2
            assert context2.get("api_key", scope="global") == "secret"
            assert context2.get("api_url", scope="global") == "https://api.example.com"

    def test_auto_load_on_init(self):
        """Test that existing files are auto-loaded on init."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workflow_file = Path(tmpdir) / "workflow_vars.json"

            # Create and save
            context1 = EnhancedVariableContext(workflow_file=workflow_file)
            context1.set("loaded", True, scope="workflow")
            context1.save_to_file("workflow")

            # Create new context - should auto-load
            context2 = EnhancedVariableContext(workflow_file=workflow_file)
            assert context2.get("loaded", scope="workflow") is True

    def test_auto_save(self):
        """Test auto-save on variable modifications."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workflow_file = Path(tmpdir) / "workflow_vars.json"

            # Create context with auto_save enabled
            context = EnhancedVariableContext(workflow_file=workflow_file, auto_save=True)

            context.set("auto_saved", True, scope="workflow")

            # File should exist and contain the variable
            assert workflow_file.exists()
            with open(workflow_file) as f:
                data = json.load(f)
            assert data["auto_saved"] is True

    def test_execution_vars_not_persisted(self):
        """Test that execution variables are not persisted to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workflow_file = Path(tmpdir) / "workflow_vars.json"

            context = EnhancedVariableContext(workflow_file=workflow_file)
            context.set("temp", "ephemeral", scope="execution")

            # Should raise error - execution scope doesn't support persistence
            with pytest.raises(ValueError, match="Only workflow and global support"):
                context.save_to_file("execution")  # type: ignore


class TestChangeTracking:
    """Test variable change tracking and history."""

    def test_change_tracking(self):
        """Test that changes are tracked in history."""
        context = EnhancedVariableContext()

        context.set("x", 1, scope="execution")
        context.set("x", 2, scope="execution")
        context.delete("x", scope="execution")

        history = context.get_change_history()

        assert len(history) == 3
        assert history[0]["operation"] == "set"
        assert history[0]["scope"] == "execution"
        assert history[0]["name"] == "x"
        assert history[0]["value"] == 1

        assert history[1]["operation"] == "set"
        assert history[1]["value"] == 2

        assert history[2]["operation"] == "delete"

    def test_clear_change_history(self):
        """Test clearing change history."""
        context = EnhancedVariableContext()

        context.set("x", 1)
        context.set("y", 2)

        assert len(context.get_change_history()) == 2

        context.clear_change_history()
        assert len(context.get_change_history()) == 0

    def test_change_callback(self):
        """Test change callback is invoked."""
        changes = []

        def callback(scope, name, value):
            changes.append((scope, name, value))

        context = EnhancedVariableContext(change_callback=callback)

        context.set("x", 1, scope="workflow")
        context.set("y", 2, scope="global")

        assert len(changes) == 2
        assert changes[0] == ("workflow", "x", 1)
        assert changes[1] == ("global", "y", 2)


class TestThreadSafety:
    """Test thread-safe operations."""

    def test_concurrent_modifications(self):
        """Test concurrent modifications from multiple threads."""
        context = EnhancedVariableContext()

        def worker(thread_id, iterations):
            for i in range(iterations):
                context.set(f"thread_{thread_id}_var_{i}", i, scope="execution")
                value = context.get(f"thread_{thread_id}_var_{i}")
                assert value == i

        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i, 100))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All variables should be set
        all_vars = context.get_all(scope="execution")
        assert len(all_vars) == 1000


class TestBackwardCompatibility:
    """Test backward compatibility with old VariableContext interface."""

    def test_variables_property(self):
        """Test 'variables' property returns merged dict."""
        context = EnhancedVariableContext()

        context.set("x", 1, scope="global")
        context.set("y", 2, scope="workflow")

        # Old interface used .variables property
        assert context.variables == {"x": 1, "y": 2}

    def test_get_all_variables(self):
        """Test get_all_variables method (old interface)."""
        context = EnhancedVariableContext()

        context.set("a", 1, scope="execution")
        context.set("b", 2, scope="workflow")

        # Old interface had get_all_variables() method
        assert context.get_all_variables() == {"a": 1, "b": 2}

    def test_clear_scope_old_names(self):
        """Test clear_scope with old scope names (local, process, global)."""
        context = EnhancedVariableContext()

        context.set("x", 1, scope="execution")
        context.set("y", 2, scope="workflow")

        # Old interface used "local" and "process"
        context.clear_scope("local")  # Should map to "execution"
        assert not context.exists("x")

        context.clear_scope("process")  # Should map to "workflow"
        assert not context.exists("y")


class TestValidation:
    """Test input validation and error handling."""

    def test_empty_variable_name(self):
        """Test that empty variable name raises error."""
        context = EnhancedVariableContext()

        with pytest.raises(ValueError, match="Variable name cannot be empty"):
            context.set("", 42)

    def test_invalid_scope(self):
        """Test that invalid scope raises error."""
        context = EnhancedVariableContext()

        with pytest.raises(ValueError, match="Invalid scope"):
            context.set("x", 1, scope="invalid")  # type: ignore

    def test_no_file_configured(self):
        """Test that save fails if no file configured."""
        context = EnhancedVariableContext()

        with pytest.raises(ValueError, match="No file configured"):
            context.save_to_file("workflow")


class TestComplexValues:
    """Test handling of complex variable values."""

    def test_dict_values(self):
        """Test storing dictionary values."""
        context = EnhancedVariableContext()

        data = {"name": "alice", "age": 30, "tags": ["python", "testing"]}
        context.set("user_data", data, scope="workflow")

        retrieved = context.get("user_data")
        assert retrieved == data
        assert retrieved["name"] == "alice"

    def test_list_values(self):
        """Test storing list values."""
        context = EnhancedVariableContext()

        data = [1, 2, 3, {"nested": True}]
        context.set("items", data, scope="execution")

        assert context.get("items") == data

    def test_persistence_of_complex_values(self):
        """Test that complex values persist correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workflow_file = Path(tmpdir) / "workflow_vars.json"

            context1 = EnhancedVariableContext(workflow_file=workflow_file)
            data = {
                "config": {
                    "timeout": 30,
                    "retry": True,
                    "endpoints": ["api1", "api2"],
                }
            }
            context1.set("app_config", data, scope="workflow")
            context1.save_to_file("workflow")

            context2 = EnhancedVariableContext(workflow_file=workflow_file)
            context2.load_from_file("workflow")

            assert context2.get("app_config", scope="workflow") == data


class TestUsageScenarios:
    """Test real-world usage scenarios."""

    def test_workflow_execution_scenario(self):
        """Test typical workflow execution with all three scopes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workflow_file = Path(tmpdir) / "workflow_vars.json"
            global_file = Path(tmpdir) / "global_vars.json"

            # Initialize with global config
            context = EnhancedVariableContext(workflow_file=workflow_file, global_file=global_file)

            # Set global configuration (shared across workflows)
            context.set("api_url", "https://api.example.com", scope="global")
            context.set("api_timeout", 30, scope="global")
            context.save_to_file("global")

            # Set workflow-specific variables
            context.set("user_id", "12345", scope="workflow")
            context.set("session_start", time.time(), scope="workflow")

            # Set temporary execution variables
            context.set("current_step", 1, scope="execution")
            context.set("temp_result", {"status": "pending"}, scope="execution")

            # Verify all are accessible
            assert context.get("api_url") == "https://api.example.com"
            assert context.get("user_id") == "12345"
            assert context.get("current_step") == 1

            # Clear execution vars after step completes
            context.clear("execution")
            assert context.get("current_step") is None
            assert context.get("user_id") == "12345"  # Workflow vars remain

    def test_multi_workflow_shared_globals(self):
        """Test multiple workflows sharing global variables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            global_file = Path(tmpdir) / "global_vars.json"

            # Workflow 1 sets global config
            context1 = EnhancedVariableContext(global_file=global_file)
            context1.set("shared_config", {"mode": "production"}, scope="global")
            context1.save_to_file("global")

            # Workflow 2 reads same global config
            context2 = EnhancedVariableContext(global_file=global_file)
            context2.load_from_file("global")

            assert context2.get("shared_config", scope="global") == {"mode": "production"}
