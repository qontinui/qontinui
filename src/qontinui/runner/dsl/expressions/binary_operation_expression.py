"""Binary operation expression - ported from Qontinui framework.

Represents a binary operation expression in the DSL.
"""

from dataclasses import dataclass
from typing import Any

from .expression import Expression


@dataclass
class BinaryOperationExpression(Expression):
    """Represents a binary operation expression in the DSL.

    Port of BinaryOperationExpression from Qontinui framework class.

    A binary operation expression performs an operation between two operands.
    Supports arithmetic operations (+, -, *, /, %), comparison operations
    (==, !=, <, >, <=, >=), and logical operations (&&, ||).

    Example in JSON:
        {
            "expressionType": "binaryOperation",
            "operator": "+",
            "left": {"expressionType": "variable", "name": "x"},
            "right": {"expressionType": "literal", "valueType": "integer", "value": 5}
        }

        {
            "expressionType": "binaryOperation",
            "operator": "&&",
            "left": {"expressionType": "variable", "name": "isReady"},
            "right": {"expressionType": "binaryOperation", "operator": ">",
                     "left": {"expressionType": "variable", "name": "count"},
                     "right": {"expressionType": "literal", "valueType": "integer", "value": 0}}
        }
    """

    operator: str = ""
    """The binary operator.
    Arithmetic: +, -, *, /, %
    Comparison: ==, !=, <, >, <=, >=
    Logical: &&, ||"""

    left: Expression | None = None
    """The left operand expression."""

    right: Expression | None = None
    """The right operand expression."""

    def __init__(
        self,
        operator: str = "",
        left: Expression | None = None,
        right: Expression | None = None,
    ):
        """Initialize binary operation expression.

        Args:
            operator: The operator
            left: Left operand
            right: Right operand
        """
        super().__init__("binaryOperation")
        self.operator = operator
        self.left = left
        self.right = right

    @classmethod
    def from_dict(cls, data: dict) -> "BinaryOperationExpression":
        """Create BinaryOperationExpression from dictionary.

        Args:
            data: Dictionary with expression data

        Returns:
            BinaryOperationExpression instance
        """
        left = None
        if "left" in data:
            left = Expression.from_dict(data["left"])

        right = None
        if "right" in data:
            right = Expression.from_dict(data["right"])

        return cls(operator=data.get("operator", ""), left=left, right=right)

    def to_dict(self) -> dict:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation
        """
        result = super().to_dict()
        result["operator"] = self.operator
        if self.left:
            result["left"] = self.left.to_dict()
        if self.right:
            result["right"] = self.right.to_dict()
        return result

    def evaluate(self, context: dict) -> Any:
        """Evaluate the binary operation.

        Args:
            context: Variable context for evaluation

        Returns:
            The result of the operation
        """
        if not self.left or not self.right:
            raise ValueError(f"Binary operation '{self.operator}' missing operands")

        left_val = self.left.evaluate(context)
        right_val = self.right.evaluate(context)

        # Arithmetic operations
        if self.operator == "+":
            return left_val + right_val
        elif self.operator == "-":
            return left_val - right_val
        elif self.operator == "*":
            return left_val * right_val
        elif self.operator == "/":
            return left_val / right_val
        elif self.operator == "%":
            return left_val % right_val

        # Comparison operations
        elif self.operator == "==":
            return left_val == right_val
        elif self.operator == "!=":
            return left_val != right_val
        elif self.operator == "<":
            return left_val < right_val
        elif self.operator == ">":
            return left_val > right_val
        elif self.operator == "<=":
            return left_val <= right_val
        elif self.operator == ">=":
            return left_val >= right_val

        # Logical operations
        elif self.operator == "&&":
            return left_val and right_val
        elif self.operator == "||":
            return left_val or right_val

        else:
            raise ValueError(f"Unknown operator: {self.operator}")
