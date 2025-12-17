"""Standalone test for the DSL statement executor.

This test can be run directly without installing the package.
"""

import sys
from pathlib import Path

# Add the src directory to the path
src_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(src_dir))

# Now we can import using the package structure
from execution_context import ExecutionContext  # noqa: E402
from flow_control import ReturnException  # noqa: E402
from statement_executor import StatementExecutor  # noqa: E402

# Import expressions and statements using relative paths
sys.path.insert(0, str(Path(__file__).parent.parent / "expressions"))
sys.path.insert(0, str(Path(__file__).parent.parent / "statements"))

from assignment_statement import AssignmentStatement  # noqa: E402
from binary_operation_expression import BinaryOperationExpression  # noqa: E402
from for_each_statement import ForEachStatement  # noqa: E402
from if_statement import IfStatement  # noqa: E402
from literal_expression import LiteralExpression  # noqa: E402
from return_statement import ReturnStatement  # noqa: E402
from variable_declaration_statement import VariableDeclarationStatement  # noqa: E402
from variable_expression import VariableExpression  # noqa: E402


def test_basic_variables():
    """Test basic variable declaration and assignment."""
    print("\n=== Test 1: Basic Variables ===")

    executor = StatementExecutor()

    # Declare a variable: let count: integer = 10
    stmt1 = VariableDeclarationStatement(
        variable_name="count",
        variable_type="integer",
        initial_value=LiteralExpression(value_type="integer", value=10),
    )
    executor.execute(stmt1)
    assert executor.get_variable("count") == 10
    print(f"✓ After declaration: count = {executor.get_variable('count')}")

    # Assign a new value: count = 20
    stmt2 = AssignmentStatement(
        variable_name="count", value=LiteralExpression(value_type="integer", value=20)
    )
    executor.execute(stmt2)
    assert executor.get_variable("count") == 20
    print(f"✓ After assignment: count = {executor.get_variable('count')}")

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
    assert executor.get_variable("count") == 25
    print(f"✓ After increment: count = {executor.get_variable('count')}")


def test_if_statement():
    """Test if statement with condition."""
    print("\n=== Test 2: If Statement ===")

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
    assert executor.get_variable("result") == "big"
    print(
        f"✓ x = {executor.get_variable('x')}, result = {executor.get_variable('result')}"
    )


def test_foreach_loop():
    """Test ForEach loop with iteration."""
    print("\n=== Test 3: ForEach Loop ===")

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
    assert executor.get_variable("sum") == 15
    print(
        f"✓ Sum of {executor.get_variable('numbers')} = {executor.get_variable('sum')}"
    )


def test_nested_scopes():
    """Test nested scopes with forEach."""
    print("\n=== Test 4: Nested Scopes ===")

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

    # forEach (item in items) { let inner: string = item }
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

    # 'outer' should still be accessible
    assert executor.get_variable("outer") == "outer value"
    print(f"✓ After loop: outer = {executor.get_variable('outer')}")

    # 'inner' should not be accessible (out of scope)
    try:
        executor.get_variable("inner")
        raise AssertionError("inner should not be accessible!")
    except KeyError:
        print("✓ Correct: 'inner' is not accessible outside loop scope")


def test_return_statement():
    """Test return statement with exception handling."""
    print("\n=== Test 5: Return Statement ===")

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
        raise AssertionError("Should have raised ReturnException!")
    except ReturnException as e:
        assert e.value == 84
        print(f"✓ Caught ReturnException with value: {e.value}")


def test_complex_expression():
    """Test complex expressions with multiple operators."""
    print("\n=== Test 6: Complex Expressions ===")

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

    assert executor.get_variable("result") == 45
    print(f"✓ (10 + 5) * 3 = {executor.get_variable('result')}")


def test_execution_context():
    """Test ExecutionContext independently."""
    print("\n=== Test 7: ExecutionContext ===")

    context = ExecutionContext()

    # Set variable in global scope
    context.set_variable("x", 10)
    assert context.get_variable("x") == 10
    print(f"✓ Global scope: x = {context.get_variable('x')}")

    # Push new scope
    context.push_scope()
    context.set_variable("y", 20)
    context.set_variable("x", 15)  # Shadow outer x
    assert context.get_variable("x") == 15
    assert context.get_variable("y") == 20
    print(
        f"✓ Inner scope: x = {context.get_variable('x')}, y = {context.get_variable('y')}"
    )

    # Pop scope
    context.pop_scope()
    assert context.get_variable("x") == 10
    print(f"✓ After pop: x = {context.get_variable('x')}")

    # y should not be accessible
    try:
        context.get_variable("y")
        raise AssertionError("y should not be accessible!")
    except KeyError:
        print("✓ y is correctly out of scope")


def test_external_context():
    """Test external context for method calls."""
    print("\n=== Test 8: External Context ===")

    class MockLogger:
        def __init__(self):
            self.logs = []

        def log(self, message):
            self.logs.append(message)

    logger = MockLogger()
    context = ExecutionContext({"logger": logger})

    # Test external object access
    retrieved_logger = context.get_external_object("logger")
    assert retrieved_logger is logger
    print("✓ External object retrieved successfully")

    # Test that we can call methods on it
    retrieved_logger.log("Test message")
    assert len(logger.logs) == 1
    assert logger.logs[0] == "Test message"
    print("✓ External object methods work correctly")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("DSL Statement Executor Tests")
    print("=" * 60)

    tests = [
        test_basic_variables,
        test_if_statement,
        test_foreach_loop,
        test_nested_scopes,
        test_return_statement,
        test_complex_expression,
        test_execution_context,
        test_external_context,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n✗ FAILED: {test.__name__}")
            print(f"  Error: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
