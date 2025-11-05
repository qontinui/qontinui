"""Comprehensive tests for DSL statement classes.

This module tests all statement types including:
- IfStatement: conditional execution, nested ifs
- ForEachStatement: iteration, loop variable scope
- AssignmentStatement: variable updates
- ReturnStatement: return values, early return
- VariableDeclarationStatement: initialization
- MethodCallStatement: method invocation
"""

import pytest

from qontinui.runner.dsl.expressions.binary_operation_expression import BinaryOperationExpression
from qontinui.runner.dsl.expressions.literal_expression import LiteralExpression
from qontinui.runner.dsl.expressions.variable_expression import VariableExpression
from qontinui.runner.dsl.statements.assignment_statement import AssignmentStatement
from qontinui.runner.dsl.statements.for_each_statement import ForEachStatement
from qontinui.runner.dsl.statements.if_statement import IfStatement
from qontinui.runner.dsl.statements.method_call_statement import MethodCallStatement
from qontinui.runner.dsl.statements.return_statement import ReturnStatement
from qontinui.runner.dsl.statements.statement import Statement
from qontinui.runner.dsl.statements.variable_declaration_statement import (
    VariableDeclarationStatement,
)


class TestVariableDeclarationStatement:
    """Test VariableDeclarationStatement functionality."""

    def test_create_with_initial_value(self):
        """Test creating a variable declaration with initial value."""
        stmt = VariableDeclarationStatement(
            variable_name="count",
            variable_type="integer",
            initial_value=LiteralExpression(value_type="integer", value=42),
        )

        assert stmt.statement_type == "variableDeclaration"
        assert stmt.variable_name == "count"
        assert stmt.variable_type == "integer"
        assert stmt.initial_value is not None
        assert stmt.initial_value.value == 42

    def test_create_without_initial_value(self):
        """Test creating a variable declaration without initial value."""
        stmt = VariableDeclarationStatement(
            variable_name="result",
            variable_type="string",
        )

        assert stmt.statement_type == "variableDeclaration"
        assert stmt.variable_name == "result"
        assert stmt.variable_type == "string"
        assert stmt.initial_value is None

    def test_to_dict(self):
        """Test converting to dictionary representation."""
        stmt = VariableDeclarationStatement(
            variable_name="enabled",
            variable_type="boolean",
            initial_value=LiteralExpression(value_type="boolean", value=True),
        )

        result = stmt.to_dict()

        assert result["statementType"] == "variableDeclaration"
        assert result["variableName"] == "enabled"
        assert result["variableType"] == "boolean"
        assert "initialValue" in result
        assert result["initialValue"]["value"] is True

    def test_to_dict_without_initial_value(self):
        """Test to_dict without initial value."""
        stmt = VariableDeclarationStatement(
            variable_name="temp",
            variable_type="double",
        )

        result = stmt.to_dict()

        assert result["statementType"] == "variableDeclaration"
        assert result["variableName"] == "temp"
        assert result["variableType"] == "double"
        assert "initialValue" not in result

    def test_from_dict(self):
        """Test creating from dictionary representation."""
        data = {
            "statementType": "variableDeclaration",
            "variableName": "counter",
            "variableType": "integer",
            "initialValue": {"expressionType": "literal", "valueType": "integer", "value": 0},
        }

        stmt = VariableDeclarationStatement.from_dict(data)

        assert stmt.variable_name == "counter"
        assert stmt.variable_type == "integer"
        assert stmt.initial_value is not None
        assert stmt.initial_value.value == 0

    def test_from_dict_without_initial_value(self):
        """Test from_dict without initial value."""
        data = {
            "statementType": "variableDeclaration",
            "variableName": "temp",
            "variableType": "string",
        }

        stmt = VariableDeclarationStatement.from_dict(data)

        assert stmt.variable_name == "temp"
        assert stmt.variable_type == "string"
        assert stmt.initial_value is None

    def test_round_trip_serialization(self):
        """Test that to_dict and from_dict are inverse operations."""
        original = VariableDeclarationStatement(
            variable_name="value",
            variable_type="double",
            initial_value=LiteralExpression(value_type="double", value=3.14),
        )

        serialized = original.to_dict()
        restored = VariableDeclarationStatement.from_dict(serialized)

        assert restored.variable_name == original.variable_name
        assert restored.variable_type == original.variable_type
        assert restored.initial_value.value == original.initial_value.value


class TestAssignmentStatement:
    """Test AssignmentStatement functionality."""

    def test_create_assignment(self):
        """Test creating an assignment statement."""
        stmt = AssignmentStatement(
            variable_name="x",
            value=LiteralExpression(value_type="integer", value=100),
        )

        assert stmt.statement_type == "assignment"
        assert stmt.variable_name == "x"
        assert stmt.value.value == 100

    def test_assignment_with_expression(self):
        """Test assignment with complex expression."""
        expr = BinaryOperationExpression(
            operator="+",
            left=VariableExpression(name="a"),
            right=VariableExpression(name="b"),
        )
        stmt = AssignmentStatement(variable_name="sum", value=expr)

        assert stmt.variable_name == "sum"
        assert isinstance(stmt.value, BinaryOperationExpression)

    def test_to_dict(self):
        """Test converting to dictionary."""
        stmt = AssignmentStatement(
            variable_name="result",
            value=LiteralExpression(value_type="string", value="done"),
        )

        result = stmt.to_dict()

        assert result["statementType"] == "assignment"
        assert result["variableName"] == "result"
        assert result["value"]["value"] == "done"

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "statementType": "assignment",
            "variableName": "count",
            "value": {"expressionType": "literal", "valueType": "integer", "value": 5},
        }

        stmt = AssignmentStatement.from_dict(data)

        assert stmt.variable_name == "count"
        assert stmt.value.value == 5

    def test_round_trip_serialization(self):
        """Test round-trip serialization."""
        original = AssignmentStatement(
            variable_name="flag",
            value=LiteralExpression(value_type="boolean", value=False),
        )

        serialized = original.to_dict()
        restored = AssignmentStatement.from_dict(serialized)

        assert restored.variable_name == original.variable_name
        assert restored.value.value == original.value.value


class TestIfStatement:
    """Test IfStatement functionality."""

    def test_create_if_with_else(self):
        """Test creating an if statement with else branch."""
        condition = BinaryOperationExpression(
            operator=">",
            left=VariableExpression(name="x"),
            right=LiteralExpression(value_type="integer", value=0),
        )
        then_stmts = [
            AssignmentStatement(
                variable_name="result",
                value=LiteralExpression(value_type="string", value="positive"),
            )
        ]
        else_stmts = [
            AssignmentStatement(
                variable_name="result",
                value=LiteralExpression(value_type="string", value="non-positive"),
            )
        ]

        stmt = IfStatement(
            condition=condition,
            then_statements=then_stmts,
            else_statements=else_stmts,
        )

        assert stmt.statement_type == "if"
        assert stmt.condition is not None
        assert len(stmt.then_statements) == 1
        assert len(stmt.else_statements) == 1

    def test_create_if_without_else(self):
        """Test creating an if statement without else branch."""
        condition = VariableExpression(name="enabled")
        then_stmts = [
            AssignmentStatement(
                variable_name="count",
                value=LiteralExpression(value_type="integer", value=1),
            )
        ]

        stmt = IfStatement(condition=condition, then_statements=then_stmts)

        assert stmt.statement_type == "if"
        assert len(stmt.then_statements) == 1
        assert len(stmt.else_statements) == 0

    def test_nested_if_statements(self):
        """Test nested if statements."""
        inner_if = IfStatement(
            condition=BinaryOperationExpression(
                operator="<",
                left=VariableExpression(name="y"),
                right=LiteralExpression(value_type="integer", value=5),
            ),
            then_statements=[
                AssignmentStatement(
                    variable_name="result",
                    value=LiteralExpression(value_type="string", value="inner"),
                )
            ],
        )

        outer_if = IfStatement(
            condition=VariableExpression(name="enabled"),
            then_statements=[inner_if],
        )

        assert len(outer_if.then_statements) == 1
        assert isinstance(outer_if.then_statements[0], IfStatement)

    def test_to_dict(self):
        """Test converting to dictionary."""
        stmt = IfStatement(
            condition=VariableExpression(name="flag"),
            then_statements=[
                AssignmentStatement(
                    variable_name="x",
                    value=LiteralExpression(value_type="integer", value=1),
                )
            ],
            else_statements=[
                AssignmentStatement(
                    variable_name="x",
                    value=LiteralExpression(value_type="integer", value=0),
                )
            ],
        )

        result = stmt.to_dict()

        assert result["statementType"] == "if"
        assert "condition" in result
        assert "thenStatements" in result
        assert "elseStatements" in result
        assert len(result["thenStatements"]) == 1
        assert len(result["elseStatements"]) == 1

    def test_from_dict(self, sample_json_if_stmt):
        """Test creating from dictionary."""
        stmt = IfStatement.from_dict(sample_json_if_stmt)

        assert stmt.condition is not None
        assert len(stmt.then_statements) == 1
        assert len(stmt.else_statements) == 1

    def test_round_trip_serialization(self):
        """Test round-trip serialization."""
        original = IfStatement(
            condition=BinaryOperationExpression(
                operator="==",
                left=VariableExpression(name="status"),
                right=LiteralExpression(value_type="string", value="ready"),
            ),
            then_statements=[
                AssignmentStatement(
                    variable_name="proceed",
                    value=LiteralExpression(value_type="boolean", value=True),
                )
            ],
        )

        serialized = original.to_dict()
        restored = IfStatement.from_dict(serialized)

        assert restored.condition is not None
        assert len(restored.then_statements) == len(original.then_statements)


class TestForEachStatement:
    """Test ForEachStatement functionality."""

    def test_create_for_each(self):
        """Test creating a forEach statement."""
        stmt = ForEachStatement(
            variable_name="item",
            collection=VariableExpression(name="items"),
            statements=[
                AssignmentStatement(
                    variable_name="count",
                    value=BinaryOperationExpression(
                        operator="+",
                        left=VariableExpression(name="count"),
                        right=LiteralExpression(value_type="integer", value=1),
                    ),
                )
            ],
        )

        assert stmt.statement_type == "forEach"
        assert stmt.variable_name == "item"
        assert stmt.collection is not None
        assert len(stmt.statements) == 1

    def test_nested_for_each(self):
        """Test nested forEach loops."""
        inner_loop = ForEachStatement(
            variable_name="j",
            collection=VariableExpression(name="inner_items"),
            statements=[
                AssignmentStatement(
                    variable_name="sum",
                    value=BinaryOperationExpression(
                        operator="+",
                        left=VariableExpression(name="sum"),
                        right=VariableExpression(name="j"),
                    ),
                )
            ],
        )

        outer_loop = ForEachStatement(
            variable_name="i",
            collection=VariableExpression(name="outer_items"),
            statements=[inner_loop],
        )

        assert len(outer_loop.statements) == 1
        assert isinstance(outer_loop.statements[0], ForEachStatement)

    def test_to_dict(self):
        """Test converting to dictionary."""
        stmt = ForEachStatement(
            variable_name="element",
            collection=VariableExpression(name="collection"),
            statements=[
                MethodCallStatement(
                    object="processor",
                    method="process",
                    arguments=[VariableExpression(name="element")],
                )
            ],
        )

        result = stmt.to_dict()

        assert result["statementType"] == "forEach"
        assert result["variableName"] == "element"
        assert "collection" in result
        assert "statements" in result
        assert len(result["statements"]) == 1

    def test_from_dict(self, sample_json_for_each_stmt):
        """Test creating from dictionary."""
        stmt = ForEachStatement.from_dict(sample_json_for_each_stmt)

        assert stmt.variable_name == "item"
        assert stmt.collection is not None
        assert len(stmt.statements) == 1

    def test_round_trip_serialization(self):
        """Test round-trip serialization."""
        original = ForEachStatement(
            variable_name="n",
            collection=VariableExpression(name="numbers"),
            statements=[
                AssignmentStatement(
                    variable_name="total",
                    value=BinaryOperationExpression(
                        operator="+",
                        left=VariableExpression(name="total"),
                        right=VariableExpression(name="n"),
                    ),
                )
            ],
        )

        serialized = original.to_dict()
        restored = ForEachStatement.from_dict(serialized)

        assert restored.variable_name == original.variable_name
        assert len(restored.statements) == len(original.statements)


class TestReturnStatement:
    """Test ReturnStatement functionality."""

    def test_create_return_with_value(self):
        """Test creating a return statement with a value."""
        stmt = ReturnStatement(value=LiteralExpression(value_type="integer", value=42))

        assert stmt.statement_type == "return"
        assert stmt.value is not None
        assert stmt.value.value == 42

    def test_create_return_without_value(self):
        """Test creating a return statement without a value (void)."""
        stmt = ReturnStatement()

        assert stmt.statement_type == "return"
        assert stmt.value is None

    def test_return_with_variable(self):
        """Test return with variable expression."""
        stmt = ReturnStatement(value=VariableExpression(name="result"))

        assert stmt.value is not None
        assert isinstance(stmt.value, VariableExpression)

    def test_to_dict_with_value(self):
        """Test converting to dictionary with value."""
        stmt = ReturnStatement(value=LiteralExpression(value_type="string", value="success"))

        result = stmt.to_dict()

        assert result["statementType"] == "return"
        assert "value" in result
        assert result["value"]["value"] == "success"

    def test_to_dict_without_value(self):
        """Test converting to dictionary without value."""
        stmt = ReturnStatement()

        result = stmt.to_dict()

        assert result["statementType"] == "return"
        assert "value" not in result

    def test_from_dict_with_value(self):
        """Test creating from dictionary with value."""
        data = {
            "statementType": "return",
            "value": {"expressionType": "literal", "valueType": "boolean", "value": True},
        }

        stmt = ReturnStatement.from_dict(data)

        assert stmt.value is not None
        assert stmt.value.value is True

    def test_from_dict_without_value(self):
        """Test creating from dictionary without value."""
        data = {"statementType": "return"}

        stmt = ReturnStatement.from_dict(data)

        assert stmt.value is None

    def test_round_trip_serialization(self):
        """Test round-trip serialization."""
        original = ReturnStatement(value=LiteralExpression(value_type="double", value=3.14))

        serialized = original.to_dict()
        restored = ReturnStatement.from_dict(serialized)

        assert restored.value is not None
        assert restored.value.value == original.value.value


class TestMethodCallStatement:
    """Test MethodCallStatement functionality."""

    def test_create_method_call_with_object(self):
        """Test creating a method call with an object."""
        stmt = MethodCallStatement(
            object="logger",
            method="log",
            arguments=[LiteralExpression(value_type="string", value="message")],
        )

        assert stmt.statement_type == "methodCall"
        assert stmt.object == "logger"
        assert stmt.method == "log"
        assert len(stmt.arguments) == 1

    def test_create_method_call_without_object(self):
        """Test creating a method call without an object (global function)."""
        stmt = MethodCallStatement(
            method="print",
            arguments=[LiteralExpression(value_type="string", value="hello")],
        )

        assert stmt.object is None
        assert stmt.method == "print"
        assert len(stmt.arguments) == 1

    def test_method_call_no_arguments(self):
        """Test method call with no arguments."""
        stmt = MethodCallStatement(object="service", method="start")

        assert len(stmt.arguments) == 0

    def test_method_call_multiple_arguments(self):
        """Test method call with multiple arguments."""
        stmt = MethodCallStatement(
            object="calculator",
            method="add",
            arguments=[
                LiteralExpression(value_type="integer", value=5),
                LiteralExpression(value_type="integer", value=10),
                VariableExpression(name="offset"),
            ],
        )

        assert len(stmt.arguments) == 3

    def test_to_dict(self):
        """Test converting to dictionary."""
        stmt = MethodCallStatement(
            object="browser",
            method="click",
            arguments=[LiteralExpression(value_type="string", value="#button")],
        )

        result = stmt.to_dict()

        assert result["statementType"] == "methodCall"
        assert result["object"] == "browser"
        assert result["method"] == "click"
        assert len(result["arguments"]) == 1

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "statementType": "methodCall",
            "object": "api",
            "method": "call",
            "arguments": [
                {"expressionType": "literal", "valueType": "string", "value": "endpoint"}
            ],
        }

        stmt = MethodCallStatement.from_dict(data)

        assert stmt.object == "api"
        assert stmt.method == "call"
        assert len(stmt.arguments) == 1

    def test_round_trip_serialization(self):
        """Test round-trip serialization."""
        original = MethodCallStatement(
            object="service",
            method="execute",
            arguments=[
                VariableExpression(name="param1"),
                LiteralExpression(value_type="integer", value=100),
            ],
        )

        serialized = original.to_dict()
        restored = MethodCallStatement.from_dict(serialized)

        assert restored.object == original.object
        assert restored.method == original.method
        assert len(restored.arguments) == len(original.arguments)


class TestStatementPolymorphism:
    """Test Statement base class polymorphic deserialization."""

    def test_from_dict_variable_declaration(self):
        """Test creating VariableDeclarationStatement via Statement.from_dict."""
        data = {
            "statementType": "variableDeclaration",
            "variableName": "x",
            "variableType": "integer",
        }

        stmt = Statement.from_dict(data)

        assert isinstance(stmt, VariableDeclarationStatement)
        assert stmt.variable_name == "x"

    def test_from_dict_assignment(self):
        """Test creating AssignmentStatement via Statement.from_dict."""
        data = {
            "statementType": "assignment",
            "variableName": "y",
            "value": {"expressionType": "literal", "valueType": "integer", "value": 10},
        }

        stmt = Statement.from_dict(data)

        assert isinstance(stmt, AssignmentStatement)
        assert stmt.variable_name == "y"

    def test_from_dict_if(self):
        """Test creating IfStatement via Statement.from_dict."""
        data = {
            "statementType": "if",
            "condition": {"expressionType": "variable", "name": "flag"},
            "thenStatements": [],
        }

        stmt = Statement.from_dict(data)

        assert isinstance(stmt, IfStatement)

    def test_from_dict_for_each(self):
        """Test creating ForEachStatement via Statement.from_dict."""
        data = {
            "statementType": "forEach",
            "variableName": "item",
            "collection": {"expressionType": "variable", "name": "items"},
            "statements": [],
        }

        stmt = Statement.from_dict(data)

        assert isinstance(stmt, ForEachStatement)

    def test_from_dict_return(self):
        """Test creating ReturnStatement via Statement.from_dict."""
        data = {"statementType": "return"}

        stmt = Statement.from_dict(data)

        assert isinstance(stmt, ReturnStatement)

    def test_from_dict_method_call(self):
        """Test creating MethodCallStatement via Statement.from_dict."""
        data = {
            "statementType": "methodCall",
            "object": "obj",
            "method": "doSomething",
            "arguments": [],
        }

        stmt = Statement.from_dict(data)

        assert isinstance(stmt, MethodCallStatement)

    def test_from_dict_unknown_type(self):
        """Test that unknown statement type raises ValueError."""
        data = {"statementType": "unknownType"}

        with pytest.raises(ValueError, match="Unknown statement type"):
            Statement.from_dict(data)

    def test_complex_statement_hierarchy(self):
        """Test deserializing complex nested statement structure."""
        data = {
            "statementType": "if",
            "condition": {
                "expressionType": "binaryOperation",
                "operator": ">",
                "left": {"expressionType": "variable", "name": "x"},
                "right": {"expressionType": "literal", "valueType": "integer", "value": 0},
            },
            "thenStatements": [
                {
                    "statementType": "forEach",
                    "variableName": "i",
                    "collection": {"expressionType": "variable", "name": "items"},
                    "statements": [
                        {
                            "statementType": "assignment",
                            "variableName": "sum",
                            "value": {
                                "expressionType": "binaryOperation",
                                "operator": "+",
                                "left": {"expressionType": "variable", "name": "sum"},
                                "right": {"expressionType": "variable", "name": "i"},
                            },
                        }
                    ],
                }
            ],
            "elseStatements": [
                {
                    "statementType": "return",
                    "value": {"expressionType": "literal", "valueType": "integer", "value": 0},
                }
            ],
        }

        stmt = Statement.from_dict(data)

        assert isinstance(stmt, IfStatement)
        assert len(stmt.then_statements) == 1
        assert isinstance(stmt.then_statements[0], ForEachStatement)
        assert len(stmt.else_statements) == 1
        assert isinstance(stmt.else_statements[0], ReturnStatement)
