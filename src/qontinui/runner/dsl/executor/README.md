# DSL Statement Executor

This package provides the runtime execution engine for the Qontinui DSL (Domain-Specific Language). It executes DSL statements with proper flow control, variable scoping, and expression evaluation.

## Overview

The executor consists of three main components:

1. **ExecutionContext**: Manages variable scopes and external objects
2. **FlowControl**: Exception-based control flow (break, continue, return)
3. **StatementExecutor**: Main execution engine that runs DSL statements

## Architecture

### ExecutionContext

The `ExecutionContext` class manages variable scopes using a stack-based approach:

- **Scope Stack**: Variables are stored in nested scopes (global → function → loop)
- **Variable Shadowing**: Inner scopes can shadow outer scope variables
- **External Context**: Stores objects that DSL code can interact with (e.g., logger, database)

**Key Methods:**
- `push_scope()` / `pop_scope()`: Manage scope stack for blocks and loops
- `set_variable(name, value)`: Declare variable in current scope
- `update_variable(name, value)`: Update variable in any scope
- `get_variable(name)`: Retrieve variable value (searches all scopes)
- `get_external_object(name)`: Access external objects for method calls

### Flow Control Exceptions

Flow control is implemented using exceptions for clean, predictable behavior:

- **BreakException**: Exit from a forEach loop
- **ContinueException**: Skip to next iteration of a forEach loop
- **ReturnException**: Exit from a function with a value
- **ExecutionError**: Wrapper for execution errors with context

These exceptions allow proper handling of control flow even in deeply nested statements.

### StatementExecutor

The `StatementExecutor` is the main execution engine:

**Supported Statements:**
- `VariableDeclarationStatement`: Declare and initialize variables
- `AssignmentStatement`: Assign values to existing variables
- `IfStatement`: Conditional execution (if/else)
- `ForEachStatement`: Iterate over collections
- `ReturnStatement`: Return from functions
- `MethodCallStatement`: Call methods on external objects

**Supported Expressions:**
- `LiteralExpression`: Constant values (numbers, strings, booleans)
- `VariableExpression`: Variable references
- `BinaryOperationExpression`: Arithmetic and logical operations
- `MethodCallExpression`: Method calls that return values
- `BuilderExpression`: Fluent API pattern for object construction

## Usage Examples

### Basic Variable Declaration and Assignment

```python
from qontinui.runner.dsl.executor import StatementExecutor
from qontinui.runner.dsl.statements import VariableDeclarationStatement, AssignmentStatement
from qontinui.runner.dsl.expressions import LiteralExpression, VariableExpression, BinaryOperationExpression

# Create executor
executor = StatementExecutor()

# Declare: let count: integer = 10
stmt1 = VariableDeclarationStatement(
    variable_name="count",
    variable_type="integer",
    initial_value=LiteralExpression(value_type="integer", value=10)
)
executor.execute(stmt1)

# Assign: count = count + 5
stmt2 = AssignmentStatement(
    variable_name="count",
    value=BinaryOperationExpression(
        operator="+",
        left=VariableExpression(name="count"),
        right=LiteralExpression(value_type="integer", value=5)
    )
)
executor.execute(stmt2)

print(executor.get_variable("count"))  # 15
```

### If Statement

```python
from qontinui.runner.dsl.statements import IfStatement

# if (count > 10) { result = "big" } else { result = "small" }
if_stmt = IfStatement(
    condition=BinaryOperationExpression(
        operator=">",
        left=VariableExpression(name="count"),
        right=LiteralExpression(value_type="integer", value=10)
    ),
    then_statements=[
        AssignmentStatement(
            variable_name="result",
            value=LiteralExpression(value_type="string", value="big")
        )
    ],
    else_statements=[
        AssignmentStatement(
            variable_name="result",
            value=LiteralExpression(value_type="string", value="small")
        )
    ]
)

executor.execute(if_stmt)
```

### ForEach Loop

```python
from qontinui.runner.dsl.statements import ForEachStatement

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
                right=VariableExpression(name="num")
            )
        )
    ]
)

executor.execute(foreach_stmt)
```

### Using External Context

```python
from qontinui.runner.dsl.executor import ExecutionContext

# Create a logger service
class Logger:
    def log(self, message):
        print(f"[LOG] {message}")

logger = Logger()

# Create executor with external context
context = ExecutionContext({"logger": logger})
executor = StatementExecutor(context)

# Now DSL code can call logger.log()
# (via MethodCallStatement)
```

### Handling Return Statements

```python
from qontinui.runner.dsl.executor import ReturnException
from qontinui.runner.dsl.statements import ReturnStatement

# return count * 2
return_stmt = ReturnStatement(
    value=BinaryOperationExpression(
        operator="*",
        left=VariableExpression(name="count"),
        right=LiteralExpression(value_type="integer", value=2)
    )
)

try:
    executor.execute(return_stmt)
except ReturnException as e:
    print(f"Function returned: {e.value}")
```

## File Structure

```
executor/
├── __init__.py                # Package exports
├── execution_context.py       # Variable scoping and management
├── flow_control.py           # Flow control exceptions
├── statement_executor.py     # Main execution engine
├── example_usage.py          # Comprehensive examples
├── test_executor.py          # Test suite
└── README.md                 # This file
```

## Integration Points

### With DSL Statements

The executor integrates with all statement types defined in `qontinui.runner.dsl.statements`:

- Statements implement the structure/data model
- Executor provides the runtime behavior
- Clean separation of concerns

### With DSL Expressions

Expressions use the `evaluate(context)` method:

```python
class LiteralExpression:
    def evaluate(self, context):
        return self.value
```

The executor calls `evaluate()` with a merged context containing:
- All variables from all scopes
- All external objects

### With External Systems

External objects (services, APIs, UI automation) are accessed via:

1. Add to external context: `context.set_external_object("name", obj)`
2. DSL calls methods: `MethodCallStatement(object="name", method="method_name")`
3. Executor invokes: `obj.method_name(*args)`

## Design Decisions

### Exception-Based Control Flow

We use exceptions for break/continue/return because:
- Clean propagation through nested structures
- No need for special return value checking
- Standard Python idiom
- Easy to understand and debug

### Stack-Based Scoping

Variables use a scope stack because:
- Proper variable shadowing
- Automatic cleanup on scope exit (finally blocks)
- Efficient scope management
- Matches common programming language semantics

### Separation of Concerns

- **Statements**: Define structure and data (what to execute)
- **Executor**: Define behavior (how to execute)
- **Context**: Manage state (where variables live)

This separation makes the code:
- Easier to test
- Easier to extend
- Easier to understand
- More maintainable

## Error Handling

The executor wraps execution errors in `ExecutionError` with context:

```python
raise ExecutionError(
    f"Error executing statement: {e}",
    statement_type="assignment",
    context={"variable": "count"}
)
```

This provides clear error messages with context for debugging.

## Future Enhancements

Potential improvements:

1. **Break/Continue Statements**: Add dedicated statement types (currently use exceptions directly)
2. **Function Calls**: Support user-defined functions with parameters
3. **Advanced Scoping**: Support block-level scoping for if statements
4. **Type Checking**: Runtime type validation for variables
5. **Debugging Support**: Add breakpoint and step execution capabilities
6. **Performance Optimization**: Cache compiled expressions, optimize variable lookups

## Testing

The executor includes comprehensive tests in `test_executor.py`:

- Basic variable operations
- If statement evaluation
- ForEach loops with iteration
- Nested scopes and shadowing
- Return statement handling
- Complex expressions
- External context integration

## Contributing

When adding new statement types:

1. Create the statement class in `statements/`
2. Add execution logic to `StatementExecutor._execute_<type>()`
3. Add tests to `test_executor.py`
4. Update this README with usage examples

## License

Part of the Qontinui project.
