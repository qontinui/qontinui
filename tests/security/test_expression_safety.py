"""Security tests for expression evaluation.

Tests that SafeEvaluator properly blocks dangerous operations while allowing
safe mathematical and logical expressions.
"""

import pytest

from qontinui.actions.data_operations.evaluator import SafeEvaluator


class TestSafeExpressionEvaluation:
    """Test that safe expressions evaluate correctly."""

    def test_basic_arithmetic(self):
        """Test basic arithmetic operations."""
        evaluator = SafeEvaluator()

        assert evaluator.safe_eval("2 + 2", {}) == 4
        assert evaluator.safe_eval("10 - 3", {}) == 7
        assert evaluator.safe_eval("5 * 6", {}) == 30
        assert evaluator.safe_eval("15 / 3", {}) == 5
        assert evaluator.safe_eval("17 // 3", {}) == 5
        assert evaluator.safe_eval("17 % 3", {}) == 2
        assert evaluator.safe_eval("2 ** 8", {}) == 256

    def test_comparisons(self):
        """Test comparison operations."""
        evaluator = SafeEvaluator()

        assert evaluator.safe_eval("5 > 3", {}) is True
        assert evaluator.safe_eval("5 < 3", {}) is False
        assert evaluator.safe_eval("5 >= 5", {}) is True
        assert evaluator.safe_eval("5 <= 4", {}) is False
        assert evaluator.safe_eval("5 == 5", {}) is True
        assert evaluator.safe_eval("5 != 3", {}) is True

    def test_logical_operations(self):
        """Test logical operations."""
        evaluator = SafeEvaluator()

        assert evaluator.safe_eval("True and True", {}) is True
        assert evaluator.safe_eval("True and False", {}) is False
        assert evaluator.safe_eval("True or False", {}) is True
        assert evaluator.safe_eval("not False", {}) is True
        assert evaluator.safe_eval("5 > 3 and 10 < 20", {}) is True

    def test_safe_functions(self):
        """Test whitelisted safe functions."""
        evaluator = SafeEvaluator()

        assert evaluator.safe_eval("abs(-5)", {}) == 5
        assert evaluator.safe_eval("max([1, 5, 3])", {}) == 5
        assert evaluator.safe_eval("min([1, 5, 3])", {}) == 1
        assert evaluator.safe_eval("len([1, 2, 3])", {}) == 3
        assert evaluator.safe_eval("sum([1, 2, 3])", {}) == 6
        assert evaluator.safe_eval("round(3.7)", {}) == 4

    def test_data_structures(self):
        """Test list, dict, and tuple operations."""
        evaluator = SafeEvaluator()

        assert evaluator.safe_eval("[1, 2, 3]", {}) == [1, 2, 3]
        assert evaluator.safe_eval("(1, 2, 3)", {}) == (1, 2, 3)
        assert evaluator.safe_eval("{'a': 1, 'b': 2}", {}) == {"a": 1, "b": 2}
        assert evaluator.safe_eval("{1, 2, 3}", {}) == {1, 2, 3}

    def test_list_comprehensions(self):
        """Test list comprehensions work correctly."""
        evaluator = SafeEvaluator()

        assert evaluator.safe_eval("[x*2 for x in range(5)]", {}) == [0, 2, 4, 6, 8]
        assert evaluator.safe_eval("[x for x in range(10) if x % 2 == 0]", {}) == [0, 2, 4, 6, 8]

    def test_variable_access(self):
        """Test that variables in context are accessible."""
        evaluator = SafeEvaluator()
        context = {"x": 10, "y": 5, "name": "test"}

        assert evaluator.safe_eval("x + y", context) == 15
        assert evaluator.safe_eval("x > y", context) is True
        assert evaluator.safe_eval("name == 'test'", context) is True

    def test_nested_expressions(self):
        """Test nested/complex expressions."""
        evaluator = SafeEvaluator()

        assert evaluator.safe_eval("(2 + 3) * 4", {}) == 20
        assert evaluator.safe_eval("max([1, 2, 3]) + min([4, 5, 6])", {}) == 7
        assert evaluator.safe_eval("True if 5 > 3 else False", {}) is True


class TestDangerousOperationsBlocked:
    """Test that dangerous operations are properly blocked."""

    def test_import_blocked(self):
        """Test that import statements are blocked."""
        evaluator = SafeEvaluator()

        with pytest.raises(SyntaxError):
            evaluator.safe_eval("import os", {})

    def test_dunder_import_blocked(self):
        """Test that __import__ is blocked."""
        evaluator = SafeEvaluator()

        with pytest.raises(ValueError, match="Unsafe function"):
            evaluator.safe_eval("__import__('os')", {})

    def test_open_blocked(self):
        """Test that open() is blocked."""
        evaluator = SafeEvaluator()

        with pytest.raises(ValueError, match="Unsafe function"):
            evaluator.safe_eval("open('/etc/passwd')", {})

    def test_exec_blocked(self):
        """Test that exec is blocked."""
        evaluator = SafeEvaluator()

        with pytest.raises(ValueError, match="Unsafe function"):
            evaluator.safe_eval("exec('print(1)')", {})

    def test_eval_blocked(self):
        """Test that eval is blocked."""
        evaluator = SafeEvaluator()

        with pytest.raises(ValueError, match="Unsafe function"):
            evaluator.safe_eval("eval('2+2')", {})

    def test_compile_blocked(self):
        """Test that compile is blocked."""
        evaluator = SafeEvaluator()

        with pytest.raises(ValueError, match="Unsafe function"):
            evaluator.safe_eval("compile('2+2', '<string>', 'eval')", {})

    def test_getattr_blocked(self):
        """Test that getattr with dangerous targets is blocked."""
        evaluator = SafeEvaluator()

        # getattr is not in SAFE_FUNCTIONS, so it should be blocked
        with pytest.raises(ValueError, match="Unsafe function"):
            evaluator.safe_eval("getattr(__builtins__, 'open')", {})

    def test_delattr_blocked(self):
        """Test that delattr is blocked."""
        evaluator = SafeEvaluator()

        with pytest.raises(ValueError, match="Unsafe function"):
            evaluator.safe_eval("delattr(obj, 'attr')", {"obj": object()})

    def test_setattr_blocked(self):
        """Test that setattr is blocked."""
        evaluator = SafeEvaluator()

        with pytest.raises(ValueError, match="Unsafe function"):
            evaluator.safe_eval("setattr(obj, 'attr', 1)", {"obj": object()})


class TestErrorHandling:
    """Test that errors are handled gracefully."""

    def test_syntax_error_handling(self):
        """Test that syntax errors are caught and reported."""
        evaluator = SafeEvaluator()

        with pytest.raises(SyntaxError):
            evaluator.safe_eval("2 +", {})

        with pytest.raises(SyntaxError):
            evaluator.safe_eval("if True:", {})

    def test_name_error_handling(self):
        """Test that undefined variables raise errors."""
        evaluator = SafeEvaluator()

        with pytest.raises(ValueError, match="Failed to evaluate"):
            evaluator.safe_eval("undefined_variable", {})

    def test_type_error_handling(self):
        """Test that type errors are caught."""
        evaluator = SafeEvaluator()

        with pytest.raises(ValueError, match="Failed to evaluate"):
            evaluator.safe_eval("'string' + 5", {})

    def test_zero_division_handling(self):
        """Test that division by zero is caught."""
        evaluator = SafeEvaluator()

        with pytest.raises(ValueError, match="Failed to evaluate"):
            evaluator.safe_eval("5 / 0", {})

    def test_empty_expression(self):
        """Test that empty expressions are rejected."""
        evaluator = SafeEvaluator()

        with pytest.raises(ValueError, match="must be a non-empty string"):
            evaluator.safe_eval("", {})

        with pytest.raises(ValueError, match="must be a non-empty string"):
            evaluator.safe_eval("   ", {})


class TestIsSafeExpression:
    """Test the is_safe_expression validation method."""

    def test_safe_expressions_pass(self):
        """Test that safe expressions are recognized."""
        assert SafeEvaluator.is_safe_expression("2 + 2") is True
        assert SafeEvaluator.is_safe_expression("x > 5") is True
        assert SafeEvaluator.is_safe_expression("max([1, 2, 3])") is True
        assert SafeEvaluator.is_safe_expression("[x*2 for x in range(5)]") is True

    def test_unsafe_expressions_fail(self):
        """Test that unsafe expressions are recognized."""
        assert SafeEvaluator.is_safe_expression("import os") is False
        assert SafeEvaluator.is_safe_expression("__import__('os')") is False
        assert SafeEvaluator.is_safe_expression("open('file')") is False

    def test_invalid_syntax_fails(self):
        """Test that invalid syntax is recognized."""
        assert SafeEvaluator.is_safe_expression("2 +") is False
        assert SafeEvaluator.is_safe_expression("if True:") is False


class TestSecurityDocumentation:
    """Test that security warnings are present in documentation."""

    def test_safe_eval_has_security_warning(self):
        """Test that safe_eval docstring contains security warning."""
        docstring = SafeEvaluator.safe_eval.__doc__

        assert docstring is not None
        assert "SECURITY WARNING" in docstring
        assert "TRUSTED INPUT ONLY" in docstring
        assert "DO NOT use this with" in docstring

    def test_security_documentation_reference(self):
        """Test that security docs are referenced."""
        docstring = SafeEvaluator.safe_eval.__doc__

        assert docstring is not None
        assert "SECURITY.md" in docstring
