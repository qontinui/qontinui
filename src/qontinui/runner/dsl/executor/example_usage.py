"""Example usage of the DSL statement executor.

This module demonstrates how to use the StatementExecutor to run DSL statements
with proper flow control, variable scoping, and expression evaluation.
"""

from qontinui.runner.dsl.executor import (
    ExecutionContext,
    ReturnException,
    StatementExecutor,
)
from qontinui.runner.dsl.expressions import (
    BinaryOperationExpression,
    LiteralExpression,
    VariableExpression,
)
from qontinui.runner.dsl.statements import (
    AssignmentStatement,
    ForEachStatement,
    IfStatement,
    ReturnStatement,
    VariableDeclarationStatement,
)


def example_1_basic_variables():
    """Example 1: Basic variable declaration and assignment."""
    print("\n=== Example 1: Basic Variables ===")

    executor = StatementExecutor()

    # Declare a variable: let count: integer = 10
    stmt1 = VariableDeclarationStatement(
        variable_name="count",
        variable_type="integer",
        initial_value=LiteralExpression(value_type="integer", value=10),
    )
    executor.execute(stmt1)
    print(f"After declaration: count = {executor.get_variable('count')}")

    # Assign a new value: count = 20
    stmt2 = AssignmentStatement(
        variable_name="count", value=LiteralExpression(value_type="integer", value=20)
    )
    executor.execute(stmt2)
    print(f"After assignment: count = {executor.get_variable('count')}")

    # Increment: count = count + 5
    stmt3 = AssignmentStatement(
        variable_name="count",
        value=BinaryOperationExpression(
            operator="+",
            left=VariableExpression(name="count"),
            right=LiteralExpression(value_type="integer", value=5),
        ),
    )
    executor.execute(stmt3)
    print(f"After increment: count = {executor.get_variable('count')}")


def example_2_if_statement():
    """Example 2: If statement with condition."""
    print("\n=== Example 2: If Statement ===")

    executor = StatementExecutor()

    # Declare: let x: integer = 15
    executor.execute(
        VariableDeclarationStatement(
            variable_name="x",
            variable_type="integer",
            initial_value=LiteralExpression(value_type="integer", value=15),
        )
    )

    # Declare: let result: string
    executor.execute(
        VariableDeclarationStatement(
            variable_name="result", variable_type="string", initial_value=None
        )
    )

    # if (x > 10) { result = "big" } else { result = "small" }
    if_stmt = IfStatement(
        condition=BinaryOperationExpression(
            operator=">",
            left=VariableExpression(name="x"),
            right=LiteralExpression(value_type="integer", value=10),
        ),
        then_statements=[
            AssignmentStatement(
                variable_name="result",
                value=LiteralExpression(value_type="string", value="big"),
            )
        ],
        else_statements=[
            AssignmentStatement(
                variable_name="result",
                value=LiteralExpression(value_type="string", value="small"),
            )
        ],
    )

    executor.execute(if_stmt)
    print(f"x = {executor.get_variable('x')}")
    print(f"result = {executor.get_variable('result')}")


def example_3_foreach_loop():
    """Example 3: ForEach loop with iteration."""
    print("\n=== Example 3: ForEach Loop ===")

    executor = StatementExecutor()

    # Declare: let numbers: array = [1, 2, 3, 4, 5]
    executor.execute(
        VariableDeclarationStatement(
            variable_name="numbers",
            variable_type="array",
            initial_value=LiteralExpression(value_type="array", value=[1, 2, 3, 4, 5]),
        )
    )

    # Declare: let sum: integer = 0
    executor.execute(
        VariableDeclarationStatement(
            variable_name="sum",
            variable_type="integer",
            initial_value=LiteralExpression(value_type="integer", value=0),
        )
    )

    # forEach (num in numbers) { sum = sum + num }
    foreach_stmt = ForEachStatement(
        variable_name="num",
        collection=VariableExpression(name="numbers"),
        statements=[
            AssignmentStatement(
                variable_name="sum",
                value=BinaryOperationExpression(
                    operator="+",
                    left=VariableExpression(name="sum"),
                    right=VariableExpression(name="num"),
                ),
            )
        ],
    )

    executor.execute(foreach_stmt)
    print(f"numbers = {executor.get_variable('numbers')}")
    print(f"sum = {executor.get_variable('sum')}")


def example_4_nested_scopes():
    """Example 4: Nested scopes with forEach."""
    print("\n=== Example 4: Nested Scopes ===")

    executor = StatementExecutor()

    # Declare: let outer: string = "outer value"
    executor.execute(
        VariableDeclarationStatement(
            variable_name="outer",
            variable_type="string",
            initial_value=LiteralExpression(value_type="string", value="outer value"),
        )
    )

    # Declare: let items: array = ["a", "b", "c"]
    executor.execute(
        VariableDeclarationStatement(
            variable_name="items",
            variable_type="array",
            initial_value=LiteralExpression(value_type="array", value=["a", "b", "c"]),
        )
    )

    print(f"Before loop: outer = {executor.get_variable('outer')}")

    # forEach (item in items) {
    #   let inner: string = item
    #   // inner is scoped to the loop
    # }
    foreach_stmt = ForEachStatement(
        variable_name="item",
        collection=VariableExpression(name="items"),
        statements=[
            VariableDeclarationStatement(
                variable_name="inner",
                variable_type="string",
                initial_value=VariableExpression(name="item"),
            )
        ],
    )

    executor.execute(foreach_stmt)

    print(f"After loop: outer = {executor.get_variable('outer')}")

    # Try to access 'inner' - should fail
    try:
        executor.get_variable("inner")
        print("ERROR: inner should not be accessible!")
    except KeyError:
        print("Correct: 'inner' is not accessible outside loop scope")


def example_5_return_statement():
    """Example 5: Return statement with exception handling."""
    print("\n=== Example 5: Return Statement ===")

    executor = StatementExecutor()

    # Declare: let value: integer = 42
    executor.execute(
        VariableDeclarationStatement(
            variable_name="value",
            variable_type="integer",
            initial_value=LiteralExpression(value_type="integer", value=42),
        )
    )

    # return value * 2
    return_stmt = ReturnStatement(
        value=BinaryOperationExpression(
            operator="*",
            left=VariableExpression(name="value"),
            right=LiteralExpression(value_type="integer", value=2),
        )
    )

    try:
        executor.execute(return_stmt)
        print("ERROR: Should have raised ReturnException!")
    except ReturnException as e:
        print(f"Caught ReturnException with value: {e.value}")


def example_6_complex_expression():
    """Example 6: Complex expressions with multiple operators."""
    print("\n=== Example 6: Complex Expressions ===")

    executor = StatementExecutor()

    # Declare variables
    executor.execute(
        VariableDeclarationStatement(
            variable_name="a",
            variable_type="integer",
            initial_value=LiteralExpression(value_type="integer", value=10),
        )
    )
    executor.execute(
        VariableDeclarationStatement(
            variable_name="b",
            variable_type="integer",
            initial_value=LiteralExpression(value_type="integer", value=5),
        )
    )
    executor.execute(
        VariableDeclarationStatement(
            variable_name="c",
            variable_type="integer",
            initial_value=LiteralExpression(value_type="integer", value=3),
        )
    )

    # result = (a + b) * c
    # This requires nested binary operations
    executor.execute(
        VariableDeclarationStatement(
            variable_name="result",
            variable_type="integer",
            initial_value=BinaryOperationExpression(
                operator="*",
                left=BinaryOperationExpression(
                    operator="+",
                    left=VariableExpression(name="a"),
                    right=VariableExpression(name="b"),
                ),
                right=VariableExpression(name="c"),
            ),
        )
    )

    print(f"a = {executor.get_variable('a')}")
    print(f"b = {executor.get_variable('b')}")
    print(f"c = {executor.get_variable('c')}")
    print(f"result = (a + b) * c = {executor.get_variable('result')}")


def example_7_external_context():
    """Example 7: Using external context for method calls."""
    print("\n=== Example 7: External Context ===")

    # Create a mock logger object
    class MockLogger:
        def __init__(self):
            self.logs = []

        def log(self, message):
            self.logs.append(message)
            print(f"[LOG] {message}")

        def get_logs(self):
            return self.logs

    logger = MockLogger()

    # Create executor with external context
    context = ExecutionContext({"logger": logger})

    # Note: This example shows the setup, but MethodCallStatement would need
    # the external objects to actually work. The executor is ready for it.
    print("External context setup complete")
    print(f"Logger available in context: {context.get_external_object('logger')}")


def example_8_foreach_with_break():
    """Example 8: ForEach with break (conceptual - would need break statement)."""
    print("\n=== Example 8: ForEach Loop Behavior ===")

    executor = StatementExecutor()

    # Declare array
    executor.execute(
        VariableDeclarationStatement(
            variable_name="numbers",
            variable_type="array",
            initial_value=LiteralExpression(value_type="array", value=[1, 2, 3, 4, 5]),
        )
    )

    # Declare counter
    executor.execute(
        VariableDeclarationStatement(
            variable_name="counter",
            variable_type="integer",
            initial_value=LiteralExpression(value_type="integer", value=0),
        )
    )

    # forEach with increment
    foreach_stmt = ForEachStatement(
        variable_name="num",
        collection=VariableExpression(name="numbers"),
        statements=[
            AssignmentStatement(
                variable_name="counter",
                value=BinaryOperationExpression(
                    operator="+",
                    left=VariableExpression(name="counter"),
                    right=LiteralExpression(value_type="integer", value=1),
                ),
            )
        ],
    )

    executor.execute(foreach_stmt)
    print(f"Iterated over {executor.get_variable('counter')} items")


def main():
    """Run all examples."""
    print("=" * 60)
    print("DSL Statement Executor Examples")
    print("=" * 60)

    example_1_basic_variables()
    example_2_if_statement()
    example_3_foreach_loop()
    example_4_nested_scopes()
    example_5_return_statement()
    example_6_complex_expression()
    example_7_external_context()
    example_8_foreach_with_break()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
