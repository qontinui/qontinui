"""Pytest fixtures for DSL tests."""

import pytest

from qontinui.runner.dsl.expressions.binary_operation_expression import (
    BinaryOperationExpression,
)
from qontinui.runner.dsl.expressions.builder_expression import (
    BuilderExpression,
    BuilderMethodCall,
)
from qontinui.runner.dsl.expressions.literal_expression import LiteralExpression
from qontinui.runner.dsl.expressions.method_call_expression import MethodCallExpression
from qontinui.runner.dsl.expressions.variable_expression import VariableExpression
from qontinui.runner.dsl.statements.assignment_statement import AssignmentStatement
from qontinui.runner.dsl.statements.for_each_statement import ForEachStatement
from qontinui.runner.dsl.statements.if_statement import IfStatement
from qontinui.runner.dsl.statements.method_call_statement import MethodCallStatement
from qontinui.runner.dsl.statements.return_statement import ReturnStatement
from qontinui.runner.dsl.statements.variable_declaration_statement import (
    VariableDeclarationStatement,
)


@pytest.fixture
def sample_context():
    """Provide a sample execution context with variables."""
    return {
        "x": 10,
        "y": 5,
        "name": "test",
        "enabled": True,
        "items": [1, 2, 3, 4, 5],
        "count": 0,
        "result": None,
    }


@pytest.fixture
def literal_int():
    """Create a literal integer expression."""
    return LiteralExpression(value_type="integer", value=42)


@pytest.fixture
def literal_string():
    """Create a literal string expression."""
    return LiteralExpression(value_type="string", value="hello")


@pytest.fixture
def literal_bool():
    """Create a literal boolean expression."""
    return LiteralExpression(value_type="boolean", value=True)


@pytest.fixture
def variable_expr():
    """Create a variable expression."""
    return VariableExpression(name="x")


@pytest.fixture
def binary_add_expr():
    """Create a binary addition expression: x + 5."""
    return BinaryOperationExpression(
        operator="+",
        left=VariableExpression(name="x"),
        right=LiteralExpression(value_type="integer", value=5),
    )


@pytest.fixture
def binary_comparison_expr():
    """Create a binary comparison expression: x > 5."""
    return BinaryOperationExpression(
        operator=">",
        left=VariableExpression(name="x"),
        right=LiteralExpression(value_type="integer", value=5),
    )


@pytest.fixture
def method_call_expr():
    """Create a method call expression."""
    return MethodCallExpression(
        object="calculator",
        method="add",
        arguments=[
            LiteralExpression(value_type="integer", value=3),
            LiteralExpression(value_type="integer", value=7),
        ],
    )


@pytest.fixture
def builder_expr():
    """Create a builder expression."""
    return BuilderExpression(
        builder_type="ObjectCollection.Builder",
        method_calls=[
            BuilderMethodCall(
                method="withImages",
                arguments=[VariableExpression(name="targetImage")],
            ),
            BuilderMethodCall(method="build", arguments=[]),
        ],
    )


@pytest.fixture
def var_decl_stmt():
    """Create a variable declaration statement."""
    return VariableDeclarationStatement(
        variable_name="newVar",
        variable_type="integer",
        initial_value=LiteralExpression(value_type="integer", value=100),
    )


@pytest.fixture
def assignment_stmt():
    """Create an assignment statement."""
    return AssignmentStatement(
        variable_name="x",
        value=LiteralExpression(value_type="integer", value=20),
    )


@pytest.fixture
def if_stmt():
    """Create an if statement."""
    return IfStatement(
        condition=BinaryOperationExpression(
            operator=">",
            left=VariableExpression(name="x"),
            right=LiteralExpression(value_type="integer", value=5),
        ),
        then_statements=[
            AssignmentStatement(
                variable_name="result",
                value=LiteralExpression(value_type="string", value="high"),
            )
        ],
        else_statements=[
            AssignmentStatement(
                variable_name="result",
                value=LiteralExpression(value_type="string", value="low"),
            )
        ],
    )


@pytest.fixture
def for_each_stmt():
    """Create a forEach statement."""
    return ForEachStatement(
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


@pytest.fixture
def return_stmt():
    """Create a return statement."""
    return ReturnStatement(value=VariableExpression(name="result"))


@pytest.fixture
def method_call_stmt():
    """Create a method call statement."""
    return MethodCallStatement(
        object="logger",
        method="log",
        arguments=[LiteralExpression(value_type="string", value="test message")],
    )


@pytest.fixture
def complex_nested_if():
    """Create a complex nested if statement for testing."""
    return IfStatement(
        condition=BinaryOperationExpression(
            operator=">",
            left=VariableExpression(name="x"),
            right=LiteralExpression(value_type="integer", value=10),
        ),
        then_statements=[
            IfStatement(
                condition=BinaryOperationExpression(
                    operator="<",
                    left=VariableExpression(name="y"),
                    right=LiteralExpression(value_type="integer", value=5),
                ),
                then_statements=[
                    AssignmentStatement(
                        variable_name="result",
                        value=LiteralExpression(
                            value_type="string", value="x>10 and y<5"
                        ),
                    )
                ],
                else_statements=[
                    AssignmentStatement(
                        variable_name="result",
                        value=LiteralExpression(
                            value_type="string", value="x>10 and y>=5"
                        ),
                    )
                ],
            )
        ],
        else_statements=[
            AssignmentStatement(
                variable_name="result",
                value=LiteralExpression(value_type="string", value="x<=10"),
            )
        ],
    )


@pytest.fixture
def sample_json_literal_expr():
    """Sample JSON for literal expression."""
    return {"expressionType": "literal", "valueType": "string", "value": "test"}


@pytest.fixture
def sample_json_variable_expr():
    """Sample JSON for variable expression."""
    return {"expressionType": "variable", "name": "myVar"}


@pytest.fixture
def sample_json_binary_expr():
    """Sample JSON for binary operation expression."""
    return {
        "expressionType": "binaryOperation",
        "operator": "+",
        "left": {"expressionType": "literal", "valueType": "integer", "value": 5},
        "right": {"expressionType": "literal", "valueType": "integer", "value": 3},
    }


@pytest.fixture
def sample_json_if_stmt():
    """Sample JSON for if statement."""
    return {
        "statementType": "if",
        "condition": {
            "expressionType": "binaryOperation",
            "operator": ">",
            "left": {"expressionType": "variable", "name": "x"},
            "right": {"expressionType": "literal", "valueType": "integer", "value": 0},
        },
        "thenStatements": [
            {
                "statementType": "assignment",
                "variableName": "result",
                "value": {
                    "expressionType": "literal",
                    "valueType": "string",
                    "value": "positive",
                },
            }
        ],
        "elseStatements": [
            {
                "statementType": "assignment",
                "variableName": "result",
                "value": {
                    "expressionType": "literal",
                    "valueType": "string",
                    "value": "negative",
                },
            }
        ],
    }


@pytest.fixture
def sample_json_for_each_stmt():
    """Sample JSON for forEach statement."""
    return {
        "statementType": "forEach",
        "variableName": "item",
        "collection": {"expressionType": "variable", "name": "items"},
        "statements": [
            {
                "statementType": "methodCall",
                "object": "processor",
                "method": "process",
                "arguments": [{"expressionType": "variable", "name": "item"}],
            }
        ],
    }
