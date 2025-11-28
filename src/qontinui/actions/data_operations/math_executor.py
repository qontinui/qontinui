"""Mathematical operation executor for data operations.

This module provides the MathExecutor class for performing mathematical
operations on operands with variable resolution support.
"""

import logging
import math
from typing import Any, cast

from .context import VariableContext
from .evaluator import SafeEvaluator

logger = logging.getLogger(__name__)


class MathExecutor:
    """Executor for mathematical operations on numeric values.

    Handles various mathematical operations including basic arithmetic,
    advanced math functions, and custom expression-based calculations.

    Supports variable resolution for operands through VariableContext.

    Operations:
        - ADD: Sum of all operands
        - SUBTRACT: Sequential subtraction from first operand
        - MULTIPLY: Product of all operands
        - DIVIDE: Sequential division from first operand
        - MODULO: Modulo operation (requires exactly 2 operands)
        - POWER: Power operation (requires exactly 2 operands)
        - SQRT: Square root (requires exactly 1 operand)
        - ABS: Absolute value (requires exactly 1 operand)
        - ROUND: Round to specified decimals (1 or 2 operands)
        - CUSTOM: Custom expression evaluation

    Example:
        >>> context = VariableContext()
        >>> context.set("x", 10)
        >>> context.set("y", 5)
        >>> evaluator = SafeEvaluator()
        >>> executor = MathExecutor(context, evaluator)
        >>> executor.execute("ADD", [5, 10, 15])
        30
        >>> executor.execute("DIVIDE", [100, 5])
        20.0
        >>> executor.execute("CUSTOM", [10, 5], "op0 ** 2 + op1")
        105
    """

    def __init__(
        self,
        variable_context: VariableContext,
        evaluator: SafeEvaluator,
    ) -> None:
        """Initialize the math executor.

        Args:
            variable_context: Context for resolving variable references
            evaluator: Safe evaluator for custom expression evaluation
        """
        self.variable_context = variable_context
        self.evaluator = evaluator
        logger.debug("Initialized MathExecutor")

    def execute(
        self,
        operation: str,
        operands: list[Any],
        custom_expression: str | None = None,
    ) -> float:
        """Execute a mathematical operation on operands.

        Args:
            operation: Operation type (ADD, SUBTRACT, MULTIPLY, DIVIDE, MODULO,
                      POWER, SQRT, ABS, ROUND, CUSTOM)
            operands: List of operands (can be values or variable references)
            custom_expression: Custom expression for CUSTOM operation (optional)

        Returns:
            Result of the mathematical operation

        Raises:
            ValueError: If operation is invalid, operand count is wrong, or
                       operation fails (e.g., division by zero)

        Example:
            >>> executor.execute("ADD", [1, 2, 3])
            6
            >>> executor.execute("SQRT", [16])
            4.0
            >>> executor.execute("DIVIDE", [10, 0])
            Traceback (most recent call last):
                ...
            ValueError: Division by zero
        """
        # Resolve operands from variables if needed
        resolved_operands = self._resolve_operands(operands)

        # Perform the operation
        result = self._perform_operation(operation, resolved_operands, custom_expression)

        logger.debug(f"Math operation {operation} executed: {resolved_operands} -> {result}")
        return result

    def _resolve_operands(self, operands: list[Any]) -> list[float]:
        """Resolve operands by converting values and variable references to floats.

        Args:
            operands: List of operands (can be numbers, variable references,
                     or convertible strings)

        Returns:
            List of resolved numeric values as floats

        Raises:
            ValueError: If operand cannot be resolved or converted to float

        Example:
            >>> context = VariableContext()
            >>> context.set("x", 10)
            >>> executor = MathExecutor(context, SafeEvaluator())
            >>> executor._resolve_operands([5, {"variableName": "x"}, "3.14"])
            [5.0, 10.0, 3.14]
        """
        resolved = []

        for i, operand in enumerate(operands):
            try:
                if isinstance(operand, int | float):
                    # Direct numeric value
                    resolved.append(float(operand))
                elif isinstance(operand, dict):
                    # Variable reference
                    var_name = operand.get("variableName")
                    if not var_name:
                        raise ValueError(f"Operand {i}: Variable reference missing 'variableName'")

                    value = self.variable_context.get(var_name)
                    if value is None:
                        raise ValueError(f"Operand {i}: Variable '{var_name}' not found")

                    resolved.append(float(value))
                else:
                    # Try to convert to float
                    resolved.append(float(operand))
            except (ValueError, TypeError) as e:
                raise ValueError(f"Operand {i}: Cannot convert to number: {operand}") from e

        return resolved

    def _perform_operation(
        self,
        operation: str,
        operands: list[float],
        custom_expression: str | None,
    ) -> float:
        """Perform a mathematical operation on resolved operands.

        Args:
            operation: Operation type
            operands: List of numeric operands
            custom_expression: Custom expression for CUSTOM operation

        Returns:
            Result value

        Raises:
            ValueError: If operation is invalid, operand count is wrong,
                       or operation fails
        """
        if not operands:
            raise ValueError("Math operation requires at least one operand")

        operation = operation.upper()

        if operation == "ADD":
            return self._add(operands)
        elif operation == "SUBTRACT":
            return self._subtract(operands)
        elif operation == "MULTIPLY":
            return self._multiply(operands)
        elif operation == "DIVIDE":
            return self._divide(operands)
        elif operation == "MODULO":
            return self._modulo(operands)
        elif operation == "POWER":
            return self._power(operands)
        elif operation == "SQRT":
            return self._sqrt(operands)
        elif operation == "ABS":
            return self._abs(operands)
        elif operation == "ROUND":
            return self._round(operands)
        elif operation == "CUSTOM":
            return self._custom(operands, custom_expression)
        else:
            raise ValueError(f"Unknown math operation: {operation}")

    def _add(self, operands: list[float]) -> float:
        """Add all operands together.

        Args:
            operands: List of numbers to add

        Returns:
            Sum of all operands
        """
        return sum(operands)

    def _subtract(self, operands: list[float]) -> float:
        """Subtract subsequent operands from the first.

        Args:
            operands: List of numbers (requires at least 2)

        Returns:
            Result of sequential subtraction

        Raises:
            ValueError: If less than 2 operands provided
        """
        if len(operands) < 2:
            raise ValueError("SUBTRACT requires at least 2 operands")

        result = operands[0]
        for val in operands[1:]:
            result -= val
        return result

    def _multiply(self, operands: list[float]) -> float:
        """Multiply all operands together.

        Args:
            operands: List of numbers to multiply

        Returns:
            Product of all operands
        """
        result = 1.0
        for val in operands:
            result *= val
        return result

    def _divide(self, operands: list[float]) -> float:
        """Divide first operand by subsequent operands sequentially.

        Args:
            operands: List of numbers (requires at least 2)

        Returns:
            Result of sequential division

        Raises:
            ValueError: If less than 2 operands or division by zero
        """
        if len(operands) < 2:
            raise ValueError("DIVIDE requires at least 2 operands")

        result = operands[0]
        for val in operands[1:]:
            if val == 0:
                raise ValueError("Division by zero")
            result /= val
        return result

    def _modulo(self, operands: list[float]) -> float:
        """Calculate modulo of two operands.

        Args:
            operands: List of exactly 2 numbers

        Returns:
            Remainder of first operand divided by second

        Raises:
            ValueError: If not exactly 2 operands or modulo by zero
        """
        if len(operands) != 2:
            raise ValueError("MODULO requires exactly 2 operands")

        if operands[1] == 0:
            raise ValueError("Modulo by zero")

        return operands[0] % operands[1]

    def _power(self, operands: list[float]) -> float:
        """Raise first operand to the power of second operand.

        Args:
            operands: List of exactly 2 numbers [base, exponent]

        Returns:
            Base raised to the power of exponent

        Raises:
            ValueError: If not exactly 2 operands
        """
        if len(operands) != 2:
            raise ValueError("POWER requires exactly 2 operands")

        return cast(float, operands[0] ** operands[1])

    def _sqrt(self, operands: list[float]) -> float:
        """Calculate square root of operand.

        Args:
            operands: List of exactly 1 number

        Returns:
            Square root of the operand

        Raises:
            ValueError: If not exactly 1 operand or negative number
        """
        if len(operands) != 1:
            raise ValueError("SQRT requires exactly 1 operand")

        if operands[0] < 0:
            raise ValueError("Cannot calculate square root of negative number")

        return math.sqrt(operands[0])

    def _abs(self, operands: list[float]) -> float:
        """Calculate absolute value of operand.

        Args:
            operands: List of exactly 1 number

        Returns:
            Absolute value of the operand

        Raises:
            ValueError: If not exactly 1 operand
        """
        if len(operands) != 1:
            raise ValueError("ABS requires exactly 1 operand")

        return abs(operands[0])

    def _round(self, operands: list[float]) -> float:
        """Round operand to specified decimal places.

        Args:
            operands: List of 1 or 2 numbers [value, decimals]
                     If decimals not provided, rounds to nearest integer

        Returns:
            Rounded value

        Raises:
            ValueError: If not 1 or 2 operands
        """
        if len(operands) < 1 or len(operands) > 2:
            raise ValueError("ROUND requires 1 or 2 operands")

        if len(operands) == 1:
            return round(operands[0])
        else:
            return round(operands[0], int(operands[1]))

    def _custom(
        self,
        operands: list[float],
        custom_expression: str | None,
    ) -> float:
        """Evaluate custom mathematical expression.

        The expression has access to:
        - operands: List of all operands
        - op0, op1, op2, ...: Individual operands by index
        - All variables from the variable context

        Args:
            operands: List of operands
            custom_expression: Python expression to evaluate

        Returns:
            Result of expression evaluation

        Raises:
            ValueError: If expression is missing or evaluation fails

        Example:
            >>> executor._custom([5, 10], "op0 * 2 + op1")
            20.0
            >>> executor._custom([3, 4], "sqrt(op0**2 + op1**2)")  # Pythagorean
            5.0
        """
        if not custom_expression:
            raise ValueError("CUSTOM operation requires 'customExpression'")

        # Create evaluation context with operands
        eval_context = {
            "operands": operands,
            **{f"op{i}": val for i, val in enumerate(operands)},
            "sqrt": math.sqrt,  # Make math functions available
            "pow": pow,
            "abs": abs,
            **self.variable_context.get_all_variables(),
        }

        try:
            result = self.evaluator.safe_eval(custom_expression, eval_context)
            return float(result)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Custom expression evaluation failed: {e}") from e
