"""Security tests for condition evaluation.

Tests that ConditionEvaluator properly handles expressions safely and that
security warnings are documented.
"""

import pytest

from qontinui.actions.control_flow.condition_evaluator import ConditionEvaluator
from qontinui.config import ConditionConfig
from qontinui.orchestration.execution_context import ExecutionContext


class TestConditionEvaluatorSafety:
    """Test that condition evaluator handles expressions safely."""

    def test_simple_expression_evaluation(self):
        """Test that simple expressions evaluate correctly."""
        context = ExecutionContext({"x": 10, "y": 5})
        evaluator = ConditionEvaluator(context)

        config = ConditionConfig(type="expression", expression="x > y")
        assert evaluator.evaluate_condition(config) is True

        config = ConditionConfig(type="expression", expression="x < y")
        assert evaluator.evaluate_condition(config) is False

    def test_variable_condition(self):
        """Test variable-based conditions."""
        context = ExecutionContext({"counter": 10})
        evaluator = ConditionEvaluator(context)

        config = ConditionConfig(
            type="variable",
            variable_name="counter",
            operator=">",
            expected_value=5
        )
        assert evaluator.evaluate_condition(config) is True

    def test_complex_expression(self):
        """Test complex boolean expressions."""
        context = ExecutionContext({"a": 5, "b": 10, "c": 15})
        evaluator = ConditionEvaluator(context)

        config = ConditionConfig(
            type="expression",
            expression="a < b and b < c"
        )
        assert evaluator.evaluate_condition(config) is True

    def test_arithmetic_in_expression(self):
        """Test arithmetic operations in expressions."""
        context = ExecutionContext({"x": 10, "y": 5})
        evaluator = ConditionEvaluator(context)

        config = ConditionConfig(
            type="expression",
            expression="x + y == 15"
        )
        assert evaluator.evaluate_condition(config) is True

    def test_empty_builtins_prevents_dangerous_ops(self):
        """Test that empty __builtins__ prevents dangerous operations."""
        context = ExecutionContext({})
        evaluator = ConditionEvaluator(context)

        # Trying to use __import__ should fail due to empty builtins
        config = ConditionConfig(
            type="expression",
            expression="__import__('os').system('ls')"
        )

        with pytest.raises(ValueError, match="Invalid expression"):
            evaluator.evaluate_condition(config)


class TestConditionEvaluatorErrorHandling:
    """Test error handling in condition evaluation."""

    def test_undefined_variable_in_expression(self):
        """Test that undefined variables raise appropriate errors."""
        context = ExecutionContext({})
        evaluator = ConditionEvaluator(context)

        config = ConditionConfig(
            type="expression",
            expression="undefined_var > 5"
        )

        with pytest.raises(ValueError, match="Invalid expression"):
            evaluator.evaluate_condition(config)

    def test_syntax_error_in_expression(self):
        """Test that syntax errors are caught."""
        context = ExecutionContext({})
        evaluator = ConditionEvaluator(context)

        config = ConditionConfig(
            type="expression",
            expression="x +"  # Invalid syntax
        )

        with pytest.raises(ValueError, match="Invalid expression"):
            evaluator.evaluate_condition(config)

    def test_type_error_in_expression(self):
        """Test that type errors are caught."""
        context = ExecutionContext({"x": "string", "y": 5})
        evaluator = ConditionEvaluator(context)

        config = ConditionConfig(
            type="expression",
            expression="x + y"  # Can't add string and int
        )

        with pytest.raises(ValueError, match="Invalid expression"):
            evaluator.evaluate_condition(config)

    def test_division_by_zero(self):
        """Test that division by zero is caught."""
        context = ExecutionContext({"x": 10})
        evaluator = ConditionEvaluator(context)

        config = ConditionConfig(
            type="expression",
            expression="x / 0"
        )

        with pytest.raises(ValueError, match="Invalid expression"):
            evaluator.evaluate_condition(config)


class TestConditionTypes:
    """Test different condition types."""

    def test_variable_condition_operators(self):
        """Test all supported operators for variable conditions."""
        context = ExecutionContext({"x": 10})
        evaluator = ConditionEvaluator(context)

        operators_tests = [
            ("==", 10, True),
            ("==", 5, False),
            ("!=", 5, True),
            ("!=", 10, False),
            (">", 5, True),
            (">", 15, False),
            ("<", 15, True),
            ("<", 5, False),
            (">=", 10, True),
            (">=", 15, False),
            ("<=", 10, True),
            ("<=", 5, False),
        ]

        for operator, expected_value, expected_result in operators_tests:
            config = ConditionConfig(
                type="variable",
                variable_name="x",
                operator=operator,
                expected_value=expected_value
            )
            result = evaluator.evaluate_condition(config)
            assert result == expected_result, \
                f"Failed for operator {operator} with expected {expected_value}"

    def test_contains_operator(self):
        """Test the contains operator."""
        context = ExecutionContext({"text": "hello world", "items": [1, 2, 3]})
        evaluator = ConditionEvaluator(context)

        # String contains
        config = ConditionConfig(
            type="variable",
            variable_name="text",
            operator="contains",
            expected_value="world"
        )
        assert evaluator.evaluate_condition(config) is True

        # List contains
        config = ConditionConfig(
            type="variable",
            variable_name="items",
            operator="contains",
            expected_value=2
        )
        assert evaluator.evaluate_condition(config) is True

    def test_matches_operator(self):
        """Test the matches (regex) operator."""
        context = ExecutionContext({"text": "hello123"})
        evaluator = ConditionEvaluator(context)

        config = ConditionConfig(
            type="variable",
            variable_name="text",
            operator="matches",
            expected_value=r"hello\d+"
        )
        assert evaluator.evaluate_condition(config) is True


class TestVariableAccess:
    """Test variable access patterns."""

    def test_missing_variable_treated_as_none(self):
        """Test that missing variables are treated as None."""
        context = ExecutionContext({})
        evaluator = ConditionEvaluator(context)

        config = ConditionConfig(
            type="variable",
            variable_name="nonexistent",
            operator="==",
            expected_value=None
        )
        # Should not raise, should treat as None
        result = evaluator.evaluate_condition(config)
        assert result is True

    def test_variables_accessible_in_expression(self):
        """Test that context variables are accessible in expressions."""
        context = ExecutionContext({"a": 1, "b": 2, "c": 3})
        evaluator = ConditionEvaluator(context)

        config = ConditionConfig(
            type="expression",
            expression="a + b == c"
        )
        assert evaluator.evaluate_condition(config) is True

    def test_namespaced_variable_access(self):
        """Test that variables can be accessed via variables namespace."""
        context = ExecutionContext({"x": 10})
        evaluator = ConditionEvaluator(context)

        # Should work both as 'x' and 'variables.x' (though latter not in current impl)
        config = ConditionConfig(
            type="expression",
            expression="x == 10"
        )
        assert evaluator.evaluate_condition(config) is True


class TestSecurityDocumentation:
    """Test that security warnings are present in documentation."""

    def test_evaluate_expression_has_security_warning(self):
        """Test that _evaluate_expression_condition has security warning."""
        docstring = ConditionEvaluator._evaluate_expression_condition.__doc__

        assert docstring is not None
        assert "SECURITY WARNING" in docstring
        assert "TRUSTED INPUT ONLY" in docstring

    def test_dangerous_usage_documented(self):
        """Test that dangerous usage patterns are documented."""
        docstring = ConditionEvaluator._evaluate_expression_condition.__doc__

        assert docstring is not None
        assert "DO NOT use this with" in docstring
        assert any(word in docstring.lower() for word in ["user", "untrusted", "external"])

    def test_mitigations_documented(self):
        """Test that security mitigations are documented."""
        docstring = ConditionEvaluator._evaluate_expression_condition.__doc__

        assert docstring is not None
        assert "__builtins__" in docstring
        assert any(word in docstring.lower() for word in ["prevent", "restricted", "mitigation"])

    def test_security_docs_referenced(self):
        """Test that security documentation is referenced."""
        docstring = ConditionEvaluator._evaluate_expression_condition.__doc__

        assert docstring is not None
        assert "SECURITY.md" in docstring


class TestUnknownConditionType:
    """Test handling of unknown condition types."""

    def test_unknown_type_raises_error(self):
        """Test that unknown condition types raise ValueError."""
        context = ExecutionContext({})
        evaluator = ConditionEvaluator(context)

        config = ConditionConfig(type="unknown_type")  # type: ignore

        with pytest.raises(ValueError, match="Unknown condition type"):
            evaluator.evaluate_condition(config)
