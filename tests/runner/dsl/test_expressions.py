"""Comprehensive tests for DSL expression classes.

This module tests all expression types including:
- LiteralExpression: all literal types
- VariableExpression: lookup, errors for undefined variables
- BinaryOperationExpression: all operators (arithmetic, comparison, logical)
- MethodCallExpression: method invocation
- BuilderExpression: fluent API pattern
"""

import pytest

from qontinui.runner.dsl.expressions.binary_operation_expression import BinaryOperationExpression
from qontinui.runner.dsl.expressions.builder_expression import (
    BuilderExpression,
    BuilderMethodCall,
)
from qontinui.runner.dsl.expressions.expression import Expression
from qontinui.runner.dsl.expressions.literal_expression import LiteralExpression
from qontinui.runner.dsl.expressions.method_call_expression import MethodCallExpression
from qontinui.runner.dsl.expressions.variable_expression import VariableExpression


class TestLiteralExpression:
    """Test LiteralExpression functionality."""

    def test_create_integer_literal(self):
        """Test creating an integer literal."""
        expr = LiteralExpression(value_type="integer", value=42)

        assert expr.expression_type == "literal"
        assert expr.value_type == "integer"
        assert expr.value == 42

    def test_create_string_literal(self):
        """Test creating a string literal."""
        expr = LiteralExpression(value_type="string", value="hello world")

        assert expr.value_type == "string"
        assert expr.value == "hello world"

    def test_create_boolean_literal(self):
        """Test creating a boolean literal."""
        expr = LiteralExpression(value_type="boolean", value=True)

        assert expr.value_type == "boolean"
        assert expr.value is True

    def test_create_double_literal(self):
        """Test creating a double literal."""
        expr = LiteralExpression(value_type="double", value=3.14159)

        assert expr.value_type == "double"
        assert expr.value == 3.14159

    def test_create_null_literal(self):
        """Test creating a null literal."""
        expr = LiteralExpression(value_type="null", value=None)

        assert expr.value_type == "null"
        assert expr.value is None

    def test_evaluate_returns_value(self, sample_context):
        """Test that evaluate returns the literal value."""
        expr = LiteralExpression(value_type="integer", value=100)

        result = expr.evaluate(sample_context)

        assert result == 100

    def test_evaluate_ignores_context(self):
        """Test that evaluate works with empty context."""
        expr = LiteralExpression(value_type="string", value="test")

        result = expr.evaluate({})

        assert result == "test"

    def test_to_dict(self):
        """Test converting to dictionary."""
        expr = LiteralExpression(value_type="boolean", value=False)

        result = expr.to_dict()

        assert result["expressionType"] == "literal"
        assert result["valueType"] == "boolean"
        assert result["value"] is False

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {"expressionType": "literal", "valueType": "double", "value": 2.718}

        expr = LiteralExpression.from_dict(data)

        assert expr.value_type == "double"
        assert expr.value == 2.718

    def test_round_trip_serialization(self):
        """Test round-trip serialization."""
        original = LiteralExpression(value_type="string", value="test value")

        serialized = original.to_dict()
        restored = LiteralExpression.from_dict(serialized)

        assert restored.value_type == original.value_type
        assert restored.value == original.value


class TestVariableExpression:
    """Test VariableExpression functionality."""

    def test_create_variable_expression(self):
        """Test creating a variable expression."""
        expr = VariableExpression(name="myVar")

        assert expr.expression_type == "variable"
        assert expr.name == "myVar"

    def test_evaluate_existing_variable(self, sample_context):
        """Test evaluating a variable that exists in context."""
        expr = VariableExpression(name="x")

        result = expr.evaluate(sample_context)

        assert result == 10

    def test_evaluate_string_variable(self, sample_context):
        """Test evaluating a string variable."""
        expr = VariableExpression(name="name")

        result = expr.evaluate(sample_context)

        assert result == "test"

    def test_evaluate_boolean_variable(self, sample_context):
        """Test evaluating a boolean variable."""
        expr = VariableExpression(name="enabled")

        result = expr.evaluate(sample_context)

        assert result is True

    def test_evaluate_list_variable(self, sample_context):
        """Test evaluating a list variable."""
        expr = VariableExpression(name="items")

        result = expr.evaluate(sample_context)

        assert result == [1, 2, 3, 4, 5]

    def test_evaluate_undefined_variable_raises_error(self, sample_context):
        """Test that evaluating undefined variable raises KeyError."""
        expr = VariableExpression(name="undefined_var")

        with pytest.raises(KeyError, match="Variable 'undefined_var' not found"):
            expr.evaluate(sample_context)

    def test_evaluate_empty_context_raises_error(self):
        """Test that evaluating with empty context raises error."""
        expr = VariableExpression(name="x")

        with pytest.raises(KeyError, match="Variable 'x' not found"):
            expr.evaluate({})

    def test_to_dict(self):
        """Test converting to dictionary."""
        expr = VariableExpression(name="counter")

        result = expr.to_dict()

        assert result["expressionType"] == "variable"
        assert result["name"] == "counter"

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {"expressionType": "variable", "name": "value"}

        expr = VariableExpression.from_dict(data)

        assert expr.name == "value"

    def test_round_trip_serialization(self):
        """Test round-trip serialization."""
        original = VariableExpression(name="testVar")

        serialized = original.to_dict()
        restored = VariableExpression.from_dict(serialized)

        assert restored.name == original.name


class TestBinaryOperationExpression:
    """Test BinaryOperationExpression functionality."""

    # Arithmetic Operations

    def test_addition(self, sample_context):
        """Test addition operation."""
        expr = BinaryOperationExpression(
            operator="+",
            left=VariableExpression(name="x"),
            right=LiteralExpression(value_type="integer", value=5),
        )

        result = expr.evaluate(sample_context)

        assert result == 15  # 10 + 5

    def test_subtraction(self, sample_context):
        """Test subtraction operation."""
        expr = BinaryOperationExpression(
            operator="-",
            left=VariableExpression(name="x"),
            right=VariableExpression(name="y"),
        )

        result = expr.evaluate(sample_context)

        assert result == 5  # 10 - 5

    def test_multiplication(self, sample_context):
        """Test multiplication operation."""
        expr = BinaryOperationExpression(
            operator="*",
            left=VariableExpression(name="x"),
            right=LiteralExpression(value_type="integer", value=3),
        )

        result = expr.evaluate(sample_context)

        assert result == 30  # 10 * 3

    def test_division(self, sample_context):
        """Test division operation."""
        expr = BinaryOperationExpression(
            operator="/",
            left=VariableExpression(name="x"),
            right=VariableExpression(name="y"),
        )

        result = expr.evaluate(sample_context)

        assert result == 2.0  # 10 / 5

    def test_modulo(self, sample_context):
        """Test modulo operation."""
        expr = BinaryOperationExpression(
            operator="%",
            left=VariableExpression(name="x"),
            right=LiteralExpression(value_type="integer", value=3),
        )

        result = expr.evaluate(sample_context)

        assert result == 1  # 10 % 3

    # Comparison Operations

    def test_equality(self, sample_context):
        """Test equality comparison."""
        expr = BinaryOperationExpression(
            operator="==",
            left=VariableExpression(name="x"),
            right=LiteralExpression(value_type="integer", value=10),
        )

        result = expr.evaluate(sample_context)

        assert result is True

    def test_inequality(self, sample_context):
        """Test inequality comparison."""
        expr = BinaryOperationExpression(
            operator="!=",
            left=VariableExpression(name="x"),
            right=VariableExpression(name="y"),
        )

        result = expr.evaluate(sample_context)

        assert result is True  # 10 != 5

    def test_less_than(self, sample_context):
        """Test less than comparison."""
        expr = BinaryOperationExpression(
            operator="<",
            left=VariableExpression(name="y"),
            right=VariableExpression(name="x"),
        )

        result = expr.evaluate(sample_context)

        assert result is True  # 5 < 10

    def test_greater_than(self, sample_context):
        """Test greater than comparison."""
        expr = BinaryOperationExpression(
            operator=">",
            left=VariableExpression(name="x"),
            right=VariableExpression(name="y"),
        )

        result = expr.evaluate(sample_context)

        assert result is True  # 10 > 5

    def test_less_than_or_equal(self, sample_context):
        """Test less than or equal comparison."""
        expr = BinaryOperationExpression(
            operator="<=",
            left=VariableExpression(name="y"),
            right=LiteralExpression(value_type="integer", value=5),
        )

        result = expr.evaluate(sample_context)

        assert result is True  # 5 <= 5

    def test_greater_than_or_equal(self, sample_context):
        """Test greater than or equal comparison."""
        expr = BinaryOperationExpression(
            operator=">=",
            left=VariableExpression(name="x"),
            right=LiteralExpression(value_type="integer", value=10),
        )

        result = expr.evaluate(sample_context)

        assert result is True  # 10 >= 10

    # Logical Operations

    def test_logical_and_both_true(self, sample_context):
        """Test logical AND with both operands true."""
        expr = BinaryOperationExpression(
            operator="&&",
            left=VariableExpression(name="enabled"),
            right=LiteralExpression(value_type="boolean", value=True),
        )

        result = expr.evaluate(sample_context)

        assert result is True

    def test_logical_and_one_false(self, sample_context):
        """Test logical AND with one operand false."""
        expr = BinaryOperationExpression(
            operator="&&",
            left=VariableExpression(name="enabled"),
            right=LiteralExpression(value_type="boolean", value=False),
        )

        result = expr.evaluate(sample_context)

        assert result is False

    def test_logical_or_both_true(self, sample_context):
        """Test logical OR with both operands true."""
        expr = BinaryOperationExpression(
            operator="||",
            left=VariableExpression(name="enabled"),
            right=LiteralExpression(value_type="boolean", value=True),
        )

        result = expr.evaluate(sample_context)

        assert result is True

    def test_logical_or_one_true(self, sample_context):
        """Test logical OR with one operand true."""
        expr = BinaryOperationExpression(
            operator="||",
            left=VariableExpression(name="enabled"),
            right=LiteralExpression(value_type="boolean", value=False),
        )

        result = expr.evaluate(sample_context)

        assert result is True

    # Nested Operations

    def test_nested_operations(self, sample_context):
        """Test nested binary operations."""
        # (x + y) * 2
        inner = BinaryOperationExpression(
            operator="+",
            left=VariableExpression(name="x"),
            right=VariableExpression(name="y"),
        )
        outer = BinaryOperationExpression(
            operator="*",
            left=inner,
            right=LiteralExpression(value_type="integer", value=2),
        )

        result = outer.evaluate(sample_context)

        assert result == 30  # (10 + 5) * 2

    def test_complex_logical_expression(self, sample_context):
        """Test complex logical expression."""
        # (x > 5) && (y < 10)
        left_comparison = BinaryOperationExpression(
            operator=">",
            left=VariableExpression(name="x"),
            right=LiteralExpression(value_type="integer", value=5),
        )
        right_comparison = BinaryOperationExpression(
            operator="<",
            left=VariableExpression(name="y"),
            right=LiteralExpression(value_type="integer", value=10),
        )
        expr = BinaryOperationExpression(
            operator="&&",
            left=left_comparison,
            right=right_comparison,
        )

        result = expr.evaluate(sample_context)

        assert result is True

    # Error Cases

    def test_missing_left_operand_raises_error(self, sample_context):
        """Test that missing left operand raises ValueError."""
        expr = BinaryOperationExpression(
            operator="+",
            left=None,
            right=LiteralExpression(value_type="integer", value=5),
        )

        with pytest.raises(ValueError, match="missing operands"):
            expr.evaluate(sample_context)

    def test_missing_right_operand_raises_error(self, sample_context):
        """Test that missing right operand raises ValueError."""
        expr = BinaryOperationExpression(
            operator="+",
            left=LiteralExpression(value_type="integer", value=5),
            right=None,
        )

        with pytest.raises(ValueError, match="missing operands"):
            expr.evaluate(sample_context)

    def test_unknown_operator_raises_error(self, sample_context):
        """Test that unknown operator raises ValueError."""
        expr = BinaryOperationExpression(
            operator="??",
            left=LiteralExpression(value_type="integer", value=5),
            right=LiteralExpression(value_type="integer", value=3),
        )

        with pytest.raises(ValueError, match="Unknown operator"):
            expr.evaluate(sample_context)

    # Serialization

    def test_to_dict(self):
        """Test converting to dictionary."""
        expr = BinaryOperationExpression(
            operator="+",
            left=VariableExpression(name="a"),
            right=LiteralExpression(value_type="integer", value=10),
        )

        result = expr.to_dict()

        assert result["expressionType"] == "binaryOperation"
        assert result["operator"] == "+"
        assert "left" in result
        assert "right" in result

    def test_from_dict(self, sample_json_binary_expr):
        """Test creating from dictionary."""
        expr = BinaryOperationExpression.from_dict(sample_json_binary_expr)

        assert expr.operator == "+"
        assert expr.left is not None
        assert expr.right is not None

    def test_round_trip_serialization(self):
        """Test round-trip serialization."""
        original = BinaryOperationExpression(
            operator="*",
            left=VariableExpression(name="x"),
            right=LiteralExpression(value_type="integer", value=5),
        )

        serialized = original.to_dict()
        restored = BinaryOperationExpression.from_dict(serialized)

        assert restored.operator == original.operator
        assert restored.left is not None
        assert restored.right is not None


class TestMethodCallExpression:
    """Test MethodCallExpression functionality."""

    def test_create_method_call_with_object(self):
        """Test creating a method call with an object."""
        expr = MethodCallExpression(
            object="calculator",
            method="add",
            arguments=[
                LiteralExpression(value_type="integer", value=5),
                LiteralExpression(value_type="integer", value=3),
            ],
        )

        assert expr.expression_type == "methodCall"
        assert expr.object == "calculator"
        assert expr.method == "add"
        assert len(expr.arguments) == 2

    def test_create_method_call_without_object(self):
        """Test creating a method call without an object (global function)."""
        expr = MethodCallExpression(
            method="sqrt",
            arguments=[LiteralExpression(value_type="integer", value=16)],
        )

        assert expr.object is None
        assert expr.method == "sqrt"
        assert len(expr.arguments) == 1

    def test_method_call_no_arguments(self):
        """Test method call with no arguments."""
        expr = MethodCallExpression(object="service", method="getStatus")

        assert len(expr.arguments) == 0

    def test_evaluate_returns_placeholder(self, sample_context):
        """Test that evaluate returns placeholder string."""
        expr = MethodCallExpression(
            object="math",
            method="multiply",
            arguments=[
                LiteralExpression(value_type="integer", value=3),
                LiteralExpression(value_type="integer", value=7),
            ],
        )

        result = expr.evaluate(sample_context)

        assert "MethodCall" in result
        assert "math.multiply" in result

    def test_evaluate_with_variable_arguments(self, sample_context):
        """Test evaluating with variable arguments."""
        expr = MethodCallExpression(
            object="processor",
            method="process",
            arguments=[VariableExpression(name="x"), VariableExpression(name="y")],
        )

        result = expr.evaluate(sample_context)

        assert "processor.process" in result
        assert "[10, 5]" in result

    def test_to_dict(self):
        """Test converting to dictionary."""
        expr = MethodCallExpression(
            object="api",
            method="fetch",
            arguments=[LiteralExpression(value_type="string", value="url")],
        )

        result = expr.to_dict()

        assert result["expressionType"] == "methodCall"
        assert result["object"] == "api"
        assert result["method"] == "fetch"
        assert len(result["arguments"]) == 1

    def test_to_dict_without_object(self):
        """Test to_dict without object."""
        expr = MethodCallExpression(
            method="globalFunc",
            arguments=[],
        )

        result = expr.to_dict()

        assert "object" not in result
        assert result["method"] == "globalFunc"

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "expressionType": "methodCall",
            "object": "service",
            "method": "execute",
            "arguments": [{"expressionType": "literal", "valueType": "integer", "value": 42}],
        }

        expr = MethodCallExpression.from_dict(data)

        assert expr.object == "service"
        assert expr.method == "execute"
        assert len(expr.arguments) == 1

    def test_round_trip_serialization(self):
        """Test round-trip serialization."""
        original = MethodCallExpression(
            object="calculator",
            method="divide",
            arguments=[
                LiteralExpression(value_type="integer", value=100),
                LiteralExpression(value_type="integer", value=5),
            ],
        )

        serialized = original.to_dict()
        restored = MethodCallExpression.from_dict(serialized)

        assert restored.object == original.object
        assert restored.method == original.method
        assert len(restored.arguments) == len(original.arguments)


class TestBuilderExpression:
    """Test BuilderExpression functionality."""

    def test_create_builder_expression(self):
        """Test creating a builder expression."""
        expr = BuilderExpression(
            builder_type="ObjectCollection.Builder",
            method_calls=[
                BuilderMethodCall(
                    method="withImages",
                    arguments=[VariableExpression(name="image")],
                ),
                BuilderMethodCall(method="build", arguments=[]),
            ],
        )

        assert expr.expression_type == "builder"
        assert expr.builder_type == "ObjectCollection.Builder"
        assert len(expr.method_calls) == 2

    def test_builder_method_call_creation(self):
        """Test creating a BuilderMethodCall."""
        method_call = BuilderMethodCall(
            method="withTimeout",
            arguments=[LiteralExpression(value_type="integer", value=5000)],
        )

        assert method_call.method == "withTimeout"
        assert len(method_call.arguments) == 1

    def test_evaluate_returns_placeholder(self, sample_context):
        """Test that evaluate returns placeholder string."""
        expr = BuilderExpression(
            builder_type="Config.Builder",
            method_calls=[BuilderMethodCall(method="build", arguments=[])],
        )

        result = expr.evaluate(sample_context)

        assert "Builder" in result
        assert "Config.Builder" in result

    def test_to_dict(self):
        """Test converting to dictionary."""
        expr = BuilderExpression(
            builder_type="Query.Builder",
            method_calls=[
                BuilderMethodCall(
                    method="where",
                    arguments=[LiteralExpression(value_type="string", value="field")],
                ),
                BuilderMethodCall(method="execute", arguments=[]),
            ],
        )

        result = expr.to_dict()

        assert result["expressionType"] == "builder"
        assert result["builderType"] == "Query.Builder"
        assert len(result["methodCalls"]) == 2

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "expressionType": "builder",
            "builderType": "Request.Builder",
            "methodCalls": [
                {
                    "method": "withHeader",
                    "arguments": [
                        {"expressionType": "literal", "valueType": "string", "value": "key"}
                    ],
                },
                {"method": "build", "arguments": []},
            ],
        }

        expr = BuilderExpression.from_dict(data)

        assert expr.builder_type == "Request.Builder"
        assert len(expr.method_calls) == 2

    def test_builder_method_call_to_dict(self):
        """Test BuilderMethodCall to_dict."""
        method_call = BuilderMethodCall(
            method="addOption",
            arguments=[LiteralExpression(value_type="string", value="value")],
        )

        result = method_call.to_dict()

        assert result["method"] == "addOption"
        assert len(result["arguments"]) == 1

    def test_builder_method_call_from_dict(self):
        """Test BuilderMethodCall from_dict."""
        data = {
            "method": "setProperty",
            "arguments": [{"expressionType": "literal", "valueType": "integer", "value": 100}],
        }

        method_call = BuilderMethodCall.from_dict(data)

        assert method_call.method == "setProperty"
        assert len(method_call.arguments) == 1

    def test_round_trip_serialization(self):
        """Test round-trip serialization."""
        original = BuilderExpression(
            builder_type="Action.Builder",
            method_calls=[
                BuilderMethodCall(
                    method="withTarget",
                    arguments=[VariableExpression(name="target")],
                ),
                BuilderMethodCall(
                    method="withRetries",
                    arguments=[LiteralExpression(value_type="integer", value=3)],
                ),
                BuilderMethodCall(method="build", arguments=[]),
            ],
        )

        serialized = original.to_dict()
        restored = BuilderExpression.from_dict(serialized)

        assert restored.builder_type == original.builder_type
        assert len(restored.method_calls) == len(original.method_calls)


class TestExpressionPolymorphism:
    """Test Expression base class polymorphic deserialization."""

    def test_from_dict_literal(self):
        """Test creating LiteralExpression via Expression.from_dict."""
        data = {"expressionType": "literal", "valueType": "integer", "value": 42}

        expr = Expression.from_dict(data)

        assert isinstance(expr, LiteralExpression)
        assert expr.value == 42

    def test_from_dict_variable(self):
        """Test creating VariableExpression via Expression.from_dict."""
        data = {"expressionType": "variable", "name": "myVar"}

        expr = Expression.from_dict(data)

        assert isinstance(expr, VariableExpression)
        assert expr.name == "myVar"

    def test_from_dict_binary_operation(self):
        """Test creating BinaryOperationExpression via Expression.from_dict."""
        data = {
            "expressionType": "binaryOperation",
            "operator": "+",
            "left": {"expressionType": "literal", "valueType": "integer", "value": 1},
            "right": {"expressionType": "literal", "valueType": "integer", "value": 2},
        }

        expr = Expression.from_dict(data)

        assert isinstance(expr, BinaryOperationExpression)
        assert expr.operator == "+"

    def test_from_dict_method_call(self):
        """Test creating MethodCallExpression via Expression.from_dict."""
        data = {
            "expressionType": "methodCall",
            "object": "obj",
            "method": "method",
            "arguments": [],
        }

        expr = Expression.from_dict(data)

        assert isinstance(expr, MethodCallExpression)

    def test_from_dict_builder(self):
        """Test creating BuilderExpression via Expression.from_dict."""
        data = {
            "expressionType": "builder",
            "builderType": "Builder",
            "methodCalls": [],
        }

        expr = Expression.from_dict(data)

        assert isinstance(expr, BuilderExpression)

    def test_from_dict_unknown_type(self):
        """Test that unknown expression type raises ValueError."""
        data = {"expressionType": "unknownType"}

        with pytest.raises(ValueError, match="Unknown expression type"):
            Expression.from_dict(data)

    def test_complex_expression_hierarchy(self, sample_context):
        """Test deserializing and evaluating complex expression structure."""
        data = {
            "expressionType": "binaryOperation",
            "operator": "&&",
            "left": {
                "expressionType": "binaryOperation",
                "operator": ">",
                "left": {"expressionType": "variable", "name": "x"},
                "right": {"expressionType": "literal", "valueType": "integer", "value": 5},
            },
            "right": {
                "expressionType": "binaryOperation",
                "operator": "<=",
                "left": {"expressionType": "variable", "name": "y"},
                "right": {"expressionType": "literal", "valueType": "integer", "value": 10},
            },
        }

        expr = Expression.from_dict(data)
        result = expr.evaluate(sample_context)

        assert result is True  # (10 > 5) && (5 <= 10)
