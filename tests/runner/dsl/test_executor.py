"""Comprehensive tests for DSL statement execution.

This module tests the execution of DSL statements including:
- Variable scoping in nested contexts
- Flow control (break, continue, return)
- Error propagation
- Complex scenarios (nested if/loops, early returns)
- Integration between statements and expressions
"""

import pytest

from qontinui.runner.dsl.executor.flow_control import (
    BreakException,
    ContinueException,
    ExecutionError,
    ReturnException,
)
from qontinui.runner.dsl.expressions.binary_operation_expression import BinaryOperationExpression
from qontinui.runner.dsl.expressions.literal_expression import LiteralExpression
from qontinui.runner.dsl.expressions.variable_expression import VariableExpression
from qontinui.runner.dsl.statements.assignment_statement import AssignmentStatement
from qontinui.runner.dsl.statements.for_each_statement import ForEachStatement
from qontinui.runner.dsl.statements.if_statement import IfStatement
from qontinui.runner.dsl.statements.return_statement import ReturnStatement
from qontinui.runner.dsl.statements.variable_declaration_statement import (
    VariableDeclarationStatement,
)


class SimpleStatementExecutor:
    """Simple statement executor for testing.

    This is a minimal executor implementation for testing purposes.
    It executes statements and maintains a variable context.
    """

    def __init__(self, initial_context: dict | None = None):
        """Initialize the executor.

        Args:
            initial_context: Initial variable context
        """
        self.context = initial_context.copy() if initial_context else {}

    def execute_statement(self, statement):
        """Execute a single statement.

        Args:
            statement: Statement to execute

        Returns:
            None, or raises ReturnException for return statements

        Raises:
            ReturnException: When a return statement is executed
            ExecutionError: When execution fails
        """
        stmt_type = statement.statement_type

        if stmt_type == "variableDeclaration":
            self._execute_variable_declaration(statement)
        elif stmt_type == "assignment":
            self._execute_assignment(statement)
        elif stmt_type == "if":
            self._execute_if(statement)
        elif stmt_type == "forEach":
            self._execute_for_each(statement)
        elif stmt_type == "return":
            self._execute_return(statement)
        elif stmt_type == "methodCall":
            # For testing, we just log method calls
            pass
        else:
            raise ExecutionError(f"Unknown statement type: {stmt_type}", statement_type=stmt_type)

    def execute_statements(self, statements):
        """Execute a list of statements.

        Args:
            statements: List of statements to execute

        Returns:
            None, or raises ReturnException

        Raises:
            ReturnException: When a return statement is executed
        """
        for stmt in statements:
            self.execute_statement(stmt)

    def _execute_variable_declaration(self, statement):
        """Execute a variable declaration statement."""
        var_name = statement.variable_name
        if statement.initial_value:
            value = statement.initial_value.evaluate(self.context)
        else:
            value = None
        self.context[var_name] = value

    def _execute_assignment(self, statement):
        """Execute an assignment statement."""
        var_name = statement.variable_name
        if var_name not in self.context:
            raise ExecutionError(
                f"Variable '{var_name}' not declared",
                statement_type="assignment",
            )
        value = statement.value.evaluate(self.context)
        self.context[var_name] = value

    def _execute_if(self, statement):
        """Execute an if statement."""
        condition_result = statement.condition.evaluate(self.context)
        if condition_result:
            self.execute_statements(statement.then_statements)
        else:
            self.execute_statements(statement.else_statements)

    def _execute_for_each(self, statement):
        """Execute a forEach statement."""
        collection = statement.collection.evaluate(self.context)
        var_name = statement.variable_name

        # Save the original value if it exists
        original_value = self.context.get(var_name)
        had_original = var_name in self.context

        try:
            for item in collection:
                # Set the loop variable
                self.context[var_name] = item

                try:
                    self.execute_statements(statement.statements)
                except ContinueException:
                    # Skip to next iteration
                    continue
                except BreakException:
                    # Exit the loop
                    break
        finally:
            # Restore the original value or remove the loop variable
            if had_original:
                self.context[var_name] = original_value
            else:
                self.context.pop(var_name, None)

    def _execute_return(self, statement):
        """Execute a return statement."""
        if statement.value:
            value = statement.value.evaluate(self.context)
        else:
            value = None
        raise ReturnException(value)


class TestFlowControlExceptions:
    """Test flow control exception classes."""

    def test_break_exception(self):
        """Test BreakException creation."""
        exc = BreakException()

        assert isinstance(exc, Exception)
        assert "Break" in str(exc)

    def test_continue_exception(self):
        """Test ContinueException creation."""
        exc = ContinueException()

        assert isinstance(exc, Exception)
        assert "Continue" in str(exc)

    def test_return_exception_with_value(self):
        """Test ReturnException with value."""
        exc = ReturnException(42)

        assert exc.value == 42
        assert "42" in str(exc)

    def test_return_exception_without_value(self):
        """Test ReturnException without value (void return)."""
        exc = ReturnException()

        assert exc.value is None

    def test_execution_error(self):
        """Test ExecutionError creation."""
        exc = ExecutionError(
            "Test error",
            statement_type="testStatement",
            context={"var": "value"},
        )

        assert exc.message == "Test error"
        assert exc.statement_type == "testStatement"
        assert exc.context == {"var": "value"}


class TestVariableDeclarationExecution:
    """Test executing variable declaration statements."""

    def test_declare_with_initial_value(self):
        """Test declaring a variable with initial value."""
        executor = SimpleStatementExecutor()
        stmt = VariableDeclarationStatement(
            variable_name="x",
            variable_type="integer",
            initial_value=LiteralExpression(value_type="integer", value=42),
        )

        executor.execute_statement(stmt)

        assert "x" in executor.context
        assert executor.context["x"] == 42

    def test_declare_without_initial_value(self):
        """Test declaring a variable without initial value."""
        executor = SimpleStatementExecutor()
        stmt = VariableDeclarationStatement(
            variable_name="y",
            variable_type="string",
        )

        executor.execute_statement(stmt)

        assert "y" in executor.context
        assert executor.context["y"] is None

    def test_declare_with_expression(self):
        """Test declaring with expression initial value."""
        executor = SimpleStatementExecutor({"a": 10, "b": 5})
        stmt = VariableDeclarationStatement(
            variable_name="sum",
            variable_type="integer",
            initial_value=BinaryOperationExpression(
                operator="+",
                left=VariableExpression(name="a"),
                right=VariableExpression(name="b"),
            ),
        )

        executor.execute_statement(stmt)

        assert executor.context["sum"] == 15


class TestAssignmentExecution:
    """Test executing assignment statements."""

    def test_assign_to_existing_variable(self):
        """Test assigning to an existing variable."""
        executor = SimpleStatementExecutor({"x": 10})
        stmt = AssignmentStatement(
            variable_name="x",
            value=LiteralExpression(value_type="integer", value=20),
        )

        executor.execute_statement(stmt)

        assert executor.context["x"] == 20

    def test_assign_expression_result(self):
        """Test assigning the result of an expression."""
        executor = SimpleStatementExecutor({"x": 10, "y": 5})
        stmt = AssignmentStatement(
            variable_name="x",
            value=BinaryOperationExpression(
                operator="*",
                left=VariableExpression(name="x"),
                right=VariableExpression(name="y"),
            ),
        )

        executor.execute_statement(stmt)

        assert executor.context["x"] == 50  # 10 * 5

    def test_assign_to_undeclared_variable_raises_error(self):
        """Test that assigning to undeclared variable raises error."""
        executor = SimpleStatementExecutor()
        stmt = AssignmentStatement(
            variable_name="undefined",
            value=LiteralExpression(value_type="integer", value=42),
        )

        with pytest.raises(ExecutionError, match="not declared"):
            executor.execute_statement(stmt)


class TestIfStatementExecution:
    """Test executing if statements."""

    def test_if_true_branch_executes(self):
        """Test that true branch executes when condition is true."""
        executor = SimpleStatementExecutor({"x": 10, "result": None})
        stmt = IfStatement(
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

        executor.execute_statement(stmt)

        assert executor.context["result"] == "high"

    def test_if_false_branch_executes(self):
        """Test that else branch executes when condition is false."""
        executor = SimpleStatementExecutor({"x": 3, "result": None})
        stmt = IfStatement(
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

        executor.execute_statement(stmt)

        assert executor.context["result"] == "low"

    def test_if_without_else(self):
        """Test if statement without else branch."""
        executor = SimpleStatementExecutor({"x": 3, "result": "initial"})
        stmt = IfStatement(
            condition=BinaryOperationExpression(
                operator=">",
                left=VariableExpression(name="x"),
                right=LiteralExpression(value_type="integer", value=5),
            ),
            then_statements=[
                AssignmentStatement(
                    variable_name="result",
                    value=LiteralExpression(value_type="string", value="changed"),
                )
            ],
        )

        executor.execute_statement(stmt)

        # Result should not change since condition is false and no else
        assert executor.context["result"] == "initial"

    def test_nested_if_statements(self):
        """Test nested if statements."""
        executor = SimpleStatementExecutor({"x": 15, "y": 3, "result": None})
        stmt = IfStatement(
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
                            value=LiteralExpression(value_type="string", value="x>10 and y<5"),
                        )
                    ],
                )
            ],
        )

        executor.execute_statement(stmt)

        assert executor.context["result"] == "x>10 and y<5"


class TestForEachExecution:
    """Test executing forEach statements."""

    def test_iterate_over_list(self):
        """Test iterating over a list."""
        executor = SimpleStatementExecutor({"items": [1, 2, 3], "sum": 0})
        stmt = ForEachStatement(
            variable_name="item",
            collection=VariableExpression(name="items"),
            statements=[
                AssignmentStatement(
                    variable_name="sum",
                    value=BinaryOperationExpression(
                        operator="+",
                        left=VariableExpression(name="sum"),
                        right=VariableExpression(name="item"),
                    ),
                )
            ],
        )

        executor.execute_statement(stmt)

        assert executor.context["sum"] == 6  # 1 + 2 + 3

    def test_loop_variable_scoping(self):
        """Test that loop variable is scoped to the loop."""
        executor = SimpleStatementExecutor({"items": [1, 2, 3]})
        stmt = ForEachStatement(
            variable_name="item",
            collection=VariableExpression(name="items"),
            statements=[
                # Empty body - just testing variable scoping
            ],
        )

        executor.execute_statement(stmt)

        # Loop variable should not exist after loop completes
        assert "item" not in executor.context

    def test_loop_variable_restores_original(self):
        """Test that loop variable restores original value after loop."""
        executor = SimpleStatementExecutor({"items": [1, 2, 3], "item": "original"})
        stmt = ForEachStatement(
            variable_name="item",
            collection=VariableExpression(name="items"),
            statements=[],
        )

        executor.execute_statement(stmt)

        # Original value should be restored
        assert executor.context["item"] == "original"

    def test_nested_loops(self):
        """Test nested forEach loops."""
        executor = SimpleStatementExecutor({"outer": [1, 2], "inner": [10, 20], "sum": 0})
        stmt = ForEachStatement(
            variable_name="i",
            collection=VariableExpression(name="outer"),
            statements=[
                ForEachStatement(
                    variable_name="j",
                    collection=VariableExpression(name="inner"),
                    statements=[
                        AssignmentStatement(
                            variable_name="sum",
                            value=BinaryOperationExpression(
                                operator="+",
                                left=VariableExpression(name="sum"),
                                right=BinaryOperationExpression(
                                    operator="*",
                                    left=VariableExpression(name="i"),
                                    right=VariableExpression(name="j"),
                                ),
                            ),
                        )
                    ],
                )
            ],
        )

        executor.execute_statement(stmt)

        # (1*10 + 1*20) + (2*10 + 2*20) = 30 + 60 = 90
        assert executor.context["sum"] == 90

    def test_empty_collection(self):
        """Test forEach with empty collection."""
        executor = SimpleStatementExecutor({"items": [], "count": 0})
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

        executor.execute_statement(stmt)

        # Count should remain 0 since loop body never executes
        assert executor.context["count"] == 0


class TestReturnExecution:
    """Test executing return statements."""

    def test_return_with_value(self):
        """Test return statement with value."""
        executor = SimpleStatementExecutor({"x": 42})
        stmt = ReturnStatement(value=VariableExpression(name="x"))

        with pytest.raises(ReturnException) as exc_info:
            executor.execute_statement(stmt)

        assert exc_info.value.value == 42

    def test_return_without_value(self):
        """Test return statement without value (void)."""
        executor = SimpleStatementExecutor()
        stmt = ReturnStatement()

        with pytest.raises(ReturnException) as exc_info:
            executor.execute_statement(stmt)

        assert exc_info.value.value is None

    def test_early_return_in_if(self):
        """Test early return within if statement."""
        executor = SimpleStatementExecutor({"x": 10, "result": "initial"})
        statements = [
            IfStatement(
                condition=BinaryOperationExpression(
                    operator=">",
                    left=VariableExpression(name="x"),
                    right=LiteralExpression(value_type="integer", value=5),
                ),
                then_statements=[
                    ReturnStatement(
                        value=LiteralExpression(value_type="string", value="early return")
                    )
                ],
            ),
            # This should not execute
            AssignmentStatement(
                variable_name="result",
                value=LiteralExpression(value_type="string", value="should not execute"),
            ),
        ]

        with pytest.raises(ReturnException) as exc_info:
            executor.execute_statements(statements)

        assert exc_info.value.value == "early return"
        # Verify the assignment didn't happen
        assert executor.context["result"] == "initial"


class TestComplexExecutionScenarios:
    """Test complex execution scenarios."""

    def test_nested_if_with_loops(self):
        """Test nested if statements containing loops."""
        executor = SimpleStatementExecutor(
            {"x": 10, "items": [1, 2, 3], "sum": 0, "processed": False}
        )
        stmt = IfStatement(
            condition=BinaryOperationExpression(
                operator=">",
                left=VariableExpression(name="x"),
                right=LiteralExpression(value_type="integer", value=5),
            ),
            then_statements=[
                ForEachStatement(
                    variable_name="item",
                    collection=VariableExpression(name="items"),
                    statements=[
                        AssignmentStatement(
                            variable_name="sum",
                            value=BinaryOperationExpression(
                                operator="+",
                                left=VariableExpression(name="sum"),
                                right=VariableExpression(name="item"),
                            ),
                        )
                    ],
                ),
                AssignmentStatement(
                    variable_name="processed",
                    value=LiteralExpression(value_type="boolean", value=True),
                ),
            ],
        )

        executor.execute_statement(stmt)

        assert executor.context["sum"] == 6
        assert executor.context["processed"] is True

    def test_loop_with_nested_if(self):
        """Test loop containing nested if statements."""
        executor = SimpleStatementExecutor({"items": [1, 2, 3, 4, 5], "even_sum": 0, "odd_sum": 0})
        stmt = ForEachStatement(
            variable_name="item",
            collection=VariableExpression(name="items"),
            statements=[
                IfStatement(
                    condition=BinaryOperationExpression(
                        operator="==",
                        left=BinaryOperationExpression(
                            operator="%",
                            left=VariableExpression(name="item"),
                            right=LiteralExpression(value_type="integer", value=2),
                        ),
                        right=LiteralExpression(value_type="integer", value=0),
                    ),
                    then_statements=[
                        AssignmentStatement(
                            variable_name="even_sum",
                            value=BinaryOperationExpression(
                                operator="+",
                                left=VariableExpression(name="even_sum"),
                                right=VariableExpression(name="item"),
                            ),
                        )
                    ],
                    else_statements=[
                        AssignmentStatement(
                            variable_name="odd_sum",
                            value=BinaryOperationExpression(
                                operator="+",
                                left=VariableExpression(name="odd_sum"),
                                right=VariableExpression(name="item"),
                            ),
                        )
                    ],
                )
            ],
        )

        executor.execute_statement(stmt)

        assert executor.context["even_sum"] == 6  # 2 + 4
        assert executor.context["odd_sum"] == 9  # 1 + 3 + 5

    def test_multiple_nested_loops(self):
        """Test multiple levels of nested loops."""
        executor = SimpleStatementExecutor(
            {"outer": [1, 2], "middle": [10, 20], "inner": [100, 200], "sum": 0}
        )
        stmt = ForEachStatement(
            variable_name="i",
            collection=VariableExpression(name="outer"),
            statements=[
                ForEachStatement(
                    variable_name="j",
                    collection=VariableExpression(name="middle"),
                    statements=[
                        ForEachStatement(
                            variable_name="k",
                            collection=VariableExpression(name="inner"),
                            statements=[
                                AssignmentStatement(
                                    variable_name="sum",
                                    value=BinaryOperationExpression(
                                        operator="+",
                                        left=VariableExpression(name="sum"),
                                        right=BinaryOperationExpression(
                                            operator="+",
                                            left=BinaryOperationExpression(
                                                operator="+",
                                                left=VariableExpression(name="i"),
                                                right=VariableExpression(name="j"),
                                            ),
                                            right=VariableExpression(name="k"),
                                        ),
                                    ),
                                )
                            ],
                        )
                    ],
                )
            ],
        )

        executor.execute_statement(stmt)

        # Each iteration: (i + j + k)
        # Total: 2 outer * 2 middle * 2 inner = 8 iterations
        # Sum of all (i+j+k) combinations
        assert executor.context["sum"] > 0

    def test_complex_conditional_logic(self):
        """Test complex conditional logic with multiple branches."""
        executor = SimpleStatementExecutor({"x": 15, "y": 8, "result": "none"})
        stmt = IfStatement(
            condition=BinaryOperationExpression(
                operator=">",
                left=VariableExpression(name="x"),
                right=LiteralExpression(value_type="integer", value=10),
            ),
            then_statements=[
                IfStatement(
                    condition=BinaryOperationExpression(
                        operator=">",
                        left=VariableExpression(name="y"),
                        right=LiteralExpression(value_type="integer", value=5),
                    ),
                    then_statements=[
                        IfStatement(
                            condition=BinaryOperationExpression(
                                operator="<",
                                left=VariableExpression(name="y"),
                                right=LiteralExpression(value_type="integer", value=10),
                            ),
                            then_statements=[
                                AssignmentStatement(
                                    variable_name="result",
                                    value=LiteralExpression(
                                        value_type="string", value="x>10, 5<y<10"
                                    ),
                                )
                            ],
                        )
                    ],
                )
            ],
        )

        executor.execute_statement(stmt)

        assert executor.context["result"] == "x>10, 5<y<10"
