"""Statement executor for DSL execution.

Executes DSL statements with proper flow control and context management.
"""

from typing import TYPE_CHECKING, Any

from .execution_context import ExecutionContext
from .flow_control import (
    BreakException,
    ContinueException,
    ExecutionError,
    ReturnException,
)

if TYPE_CHECKING:
    from ..statements.assignment_statement import AssignmentStatement
    from ..statements.for_each_statement import ForEachStatement
    from ..statements.if_statement import IfStatement
    from ..statements.method_call_statement import MethodCallStatement
    from ..statements.return_statement import ReturnStatement
    from ..statements.variable_declaration_statement import (
        VariableDeclarationStatement,
    )


class StatementExecutor:
    """Executes DSL statements with context and flow control.

    The StatementExecutor is the runtime engine for the DSL. It takes DSL statements
    and executes them within an execution context, handling variable scoping, control
    flow (if, forEach, return), and expression evaluation.

    The executor uses exception-based control flow for break, continue, and return
    statements, which allows proper handling of these constructs even in nested
    control structures.

    Example:
        ```python
        from qontinui.runner.dsl.statements import VariableDeclarationStatement
        from qontinui.runner.dsl.expressions import LiteralExpression

        executor = StatementExecutor()

        # Create and execute a variable declaration
        stmt = VariableDeclarationStatement(
            variable_name="count",
            variable_type="integer",
            initial_value=LiteralExpression(value_type="integer", value=42)
        )

        executor.execute(stmt)
        print(executor.context.get_variable("count"))  # 42
        ```

    Attributes:
        context: The execution context managing variable scopes
    """

    def __init__(self, context: ExecutionContext | None = None):
        """Initialize statement executor.

        Args:
            context: Optional execution context. If not provided, a new context
                    will be created.
        """
        self.context = context or ExecutionContext()

    def execute(self, statement: Any) -> Any:
        """Execute a statement.

        Dispatches to the appropriate execution method based on statement type.

        Args:
            statement: Statement to execute

        Returns:
            Return value if statement is a return statement, None otherwise

        Raises:
            ExecutionError: If statement execution fails
            ReturnException: If a return statement is executed (propagates up)

        Example:
            ```python
            executor = StatementExecutor()
            stmt = VariableDeclarationStatement(...)
            executor.execute(stmt)
            ```
        """
        from ..statements.assignment_statement import AssignmentStatement
        from ..statements.for_each_statement import ForEachStatement
        from ..statements.if_statement import IfStatement
        from ..statements.method_call_statement import MethodCallStatement
        from ..statements.return_statement import ReturnStatement
        from ..statements.variable_declaration_statement import (
            VariableDeclarationStatement,
        )

        try:
            if isinstance(statement, VariableDeclarationStatement):
                return self._execute_variable_declaration(statement)
            elif isinstance(statement, AssignmentStatement):
                return self._execute_assignment(statement)
            elif isinstance(statement, IfStatement):
                return self._execute_if(statement)
            elif isinstance(statement, ForEachStatement):
                return self._execute_for_each(statement)
            elif isinstance(statement, ReturnStatement):
                return self._execute_return(statement)
            elif isinstance(statement, MethodCallStatement):
                return self._execute_method_call(statement)
            else:
                raise ExecutionError(
                    f"Unknown statement type: {type(statement).__name__}",
                    statement_type=getattr(statement, "statement_type", None),
                )
        except (BreakException, ContinueException, ReturnException):
            # Propagate flow control exceptions
            raise
        except ExecutionError:
            # Propagate execution errors
            raise
        except Exception as e:
            # Wrap other exceptions in ExecutionError
            raise ExecutionError(
                f"Error executing statement: {e}",
                statement_type=getattr(statement, "statement_type", None),
            ) from e

    def execute_statements(self, statements: list[Any]) -> Any:
        """Execute a list of statements.

        Executes statements sequentially until completion or until a flow control
        exception (break, continue, return) is raised.

        Args:
            statements: List of statements to execute

        Returns:
            Return value if a return statement is executed, None otherwise

        Raises:
            BreakException: If a break statement is executed
            ContinueException: If a continue statement is executed
            ReturnException: If a return statement is executed

        Example:
            ```python
            executor = StatementExecutor()
            statements = [stmt1, stmt2, stmt3]
            executor.execute_statements(statements)
            ```
        """
        for statement in statements:
            self.execute(statement)
        return None

    def _execute_variable_declaration(self, statement: "VariableDeclarationStatement") -> None:
        """Execute a variable declaration statement.

        Declares a new variable in the current scope and optionally initializes
        it with a value from an expression.

        Args:
            statement: Variable declaration statement

        Example:
            ```python
            # let count: integer = 42
            stmt = VariableDeclarationStatement(
                variable_name="count",
                variable_type="integer",
                initial_value=LiteralExpression(value_type="integer", value=42)
            )
            ```
        """
        from ..statements.variable_declaration_statement import (
            VariableDeclarationStatement,
        )

        if not isinstance(statement, VariableDeclarationStatement):
            raise ExecutionError("Expected VariableDeclarationStatement")

        value = None
        if statement.initial_value:
            value = self._evaluate_expression(statement.initial_value)

        self.context.set_variable(statement.variable_name, value)

    def _execute_assignment(self, statement: "AssignmentStatement") -> None:
        """Execute an assignment statement.

        Assigns a value to an existing variable. The variable must have been
        previously declared in the current or an outer scope.

        Args:
            statement: Assignment statement

        Raises:
            ExecutionError: If variable not found

        Example:
            ```python
            # count = count + 1
            stmt = AssignmentStatement(
                variable_name="count",
                value=BinaryOperationExpression(
                    operator="+",
                    left=VariableExpression(name="count"),
                    right=LiteralExpression(value_type="integer", value=1)
                )
            )
            ```
        """
        from ..statements.assignment_statement import AssignmentStatement

        if not isinstance(statement, AssignmentStatement):
            raise ExecutionError("Expected AssignmentStatement")

        if not statement.value:
            raise ExecutionError(f"Assignment to '{statement.variable_name}' has no value")

        value = self._evaluate_expression(statement.value)

        try:
            self.context.update_variable(statement.variable_name, value)
        except KeyError as e:
            raise ExecutionError(
                f"Cannot assign to undeclared variable '{statement.variable_name}'",
                statement_type="assignment",
            ) from e

    def _execute_if(self, statement: "IfStatement") -> None:
        """Execute an if statement.

        Evaluates the condition and executes either the then-branch or else-branch
        based on the result.

        Args:
            statement: If statement

        Example:
            ```python
            # if (count > 0) { ... } else { ... }
            stmt = IfStatement(
                condition=BinaryOperationExpression(...),
                then_statements=[...],
                else_statements=[...]
            )
            ```
        """
        from ..statements.if_statement import IfStatement

        if not isinstance(statement, IfStatement):
            raise ExecutionError("Expected IfStatement")

        if not statement.condition:
            raise ExecutionError("If statement missing condition")

        condition_value = self._evaluate_expression(statement.condition)

        if condition_value:
            # Execute then branch
            if statement.then_statements:
                self.execute_statements(statement.then_statements)
        else:
            # Execute else branch
            if statement.else_statements:
                self.execute_statements(statement.else_statements)

    def _execute_for_each(self, statement: "ForEachStatement") -> None:
        """Execute a forEach loop statement.

        Iterates over a collection, executing the loop body for each element.
        The loop variable is created in a new scope for each iteration.

        Handles break and continue flow control:
        - Break: Exits the loop immediately
        - Continue: Skips to the next iteration

        Args:
            statement: ForEach statement

        Example:
            ```python
            # forEach (item in items) { ... }
            stmt = ForEachStatement(
                variable_name="item",
                collection=VariableExpression(name="items"),
                statements=[...]
            )
            ```
        """
        from ..statements.for_each_statement import ForEachStatement

        if not isinstance(statement, ForEachStatement):
            raise ExecutionError("Expected ForEachStatement")

        if not statement.collection:
            raise ExecutionError("ForEach statement missing collection")

        # Evaluate collection expression
        collection = self._evaluate_expression(statement.collection)

        # Ensure collection is iterable
        if not hasattr(collection, "__iter__"):
            raise ExecutionError(
                f"ForEach collection must be iterable, got {type(collection).__name__}",
                statement_type="forEach",
            )

        # Iterate over collection
        for item in collection:
            # Create new scope for loop variable
            self.context.push_scope()
            try:
                # Set loop variable
                self.context.set_variable(statement.variable_name, item)

                # Execute loop body
                try:
                    self.execute_statements(statement.statements)
                except ContinueException:
                    # Continue to next iteration
                    continue
                except BreakException:
                    # Exit loop
                    break
            finally:
                # Always pop scope, even if exception occurred
                self.context.pop_scope()

    def _execute_return(self, statement: "ReturnStatement") -> None:
        """Execute a return statement.

        Evaluates the return value expression (if any) and raises a ReturnException
        to exit the current function.

        Args:
            statement: Return statement

        Raises:
            ReturnException: Always raised to implement return behavior

        Example:
            ```python
            # return count * 2
            stmt = ReturnStatement(
                value=BinaryOperationExpression(...)
            )
            ```
        """
        from ..statements.return_statement import ReturnStatement

        if not isinstance(statement, ReturnStatement):
            raise ExecutionError("Expected ReturnStatement")

        value = None
        if statement.value:
            value = self._evaluate_expression(statement.value)

        raise ReturnException(value)

    def _execute_method_call(self, statement: "MethodCallStatement") -> Any:
        """Execute a method call statement.

        Calls a method on an object from the external context. The object must
        exist in the external context, and the method must be callable.

        Args:
            statement: Method call statement

        Returns:
            The return value of the method call

        Raises:
            ExecutionError: If object not found or method not callable

        Example:
            ```python
            # logger.log("Hello")
            stmt = MethodCallStatement(
                object="logger",
                method="log",
                arguments=[LiteralExpression(value_type="string", value="Hello")]
            )
            ```
        """
        from ..statements.method_call_statement import MethodCallStatement

        if not isinstance(statement, MethodCallStatement):
            raise ExecutionError("Expected MethodCallStatement")

        # Get object from external context
        if statement.object:
            try:
                obj = self.context.get_external_object(statement.object)
            except KeyError as e:
                raise ExecutionError(
                    f"Object '{statement.object}' not found in external context",
                    statement_type="methodCall",
                ) from e

            # Get method
            if not hasattr(obj, statement.method):
                raise ExecutionError(
                    f"Object '{statement.object}' has no method '{statement.method}'",
                    statement_type="methodCall",
                )

            method = getattr(obj, statement.method)
        else:
            # No object specified - this is not supported yet
            raise ExecutionError(
                "Method call without object not yet supported",
                statement_type="methodCall",
            )

        if not callable(method):
            raise ExecutionError(
                f"'{statement.object}.{statement.method}' is not callable",
                statement_type="methodCall",
            )

        # Evaluate arguments
        args = []
        if statement.arguments:
            for arg_expr in statement.arguments:
                args.append(self._evaluate_expression(arg_expr))

        # Call method
        try:
            return method(*args)
        except Exception as e:
            raise ExecutionError(
                f"Error calling {statement.object}.{statement.method}: {e}",
                statement_type="methodCall",
            ) from e

    def _evaluate_expression(self, expression: Any) -> Any:
        """Evaluate an expression.

        Evaluates an expression in the current context. For expressions that have
        an evaluate() method, calls it with the merged context (variables + external).
        For other objects, returns them as-is.

        Args:
            expression: Expression to evaluate

        Returns:
            The evaluated value

        Raises:
            ExecutionError: If expression evaluation fails

        Example:
            ```python
            expr = VariableExpression(name="count")
            value = executor._evaluate_expression(expr)
            ```
        """
        if not expression:
            return None

        try:
            # Check if expression has evaluate method
            if hasattr(expression, "evaluate"):
                # Create merged context for evaluation
                # This includes both variables and external objects
                eval_context = {}
                eval_context.update(self.context.get_all_variables())
                eval_context.update(self.context.get_external_context())
                return expression.evaluate(eval_context)
            else:
                # For non-expression objects, return as-is
                return expression
        except ExecutionError:
            # Propagate execution errors
            raise
        except Exception as e:
            # Wrap other exceptions
            raise ExecutionError(
                f"Error evaluating expression: {e}",
                expression_type=getattr(expression, "expression_type", None),
            ) from e

    def reset(self) -> None:
        """Reset the executor to a clean state.

        Creates a new execution context, clearing all variables and scopes.
        The external context is preserved.

        Example:
            ```python
            executor.execute(stmt1)
            executor.reset()  # Clear all variables
            executor.execute(stmt2)  # Fresh start
            ```
        """
        external_context = self.context.get_external_context()
        self.context = ExecutionContext(external_context)

    def get_variable(self, name: str) -> Any:
        """Get a variable value from the context.

        Convenience method to access variables without directly using the context.

        Args:
            name: Variable name

        Returns:
            Variable value

        Raises:
            KeyError: If variable not found
        """
        return self.context.get_variable(name)

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable in the current scope.

        Convenience method to set variables without directly using the context.

        Args:
            name: Variable name
            value: Variable value
        """
        self.context.set_variable(name, value)

    def __repr__(self) -> str:
        """Return string representation of executor.

        Returns:
            String representation
        """
        return f"StatementExecutor(context={self.context})"
