"""Expression base class - ported from Qontinui framework.

Abstract base class for all expressions in the DSL.
"""

from abc import ABC, abstractmethod
from typing import Any


class Expression(ABC):
    """Abstract base class for all expressions in the DSL.

    Port of Expression from Qontinui framework class.

    An expression represents a value-producing computation in the DSL. Expressions can be:
    - Literals (constant values like "hello" or 42)
    - Variables (references to previously declared values)
    - Method calls (invocations that return values)
    - Binary operations (arithmetic or logical operations between two expressions)
    - Builder expressions (fluent API pattern for constructing complex objects)

    This class uses polymorphic deserialization to support parsing different
    expression types from JSON based on the "expressionType" discriminator field.
    """

    def __init__(self, expression_type: str):
        """Initialize expression with its type.

        Args:
            expression_type: The discriminator field used to determine the concrete type
                           Valid values: "literal", "variable", "methodCall",
                           "binaryOperation", "builder"
        """
        self.expression_type = expression_type

    @classmethod
    def from_dict(cls, data: dict) -> "Expression":
        """Create Expression from dictionary representation.

        Args:
            data: Dictionary with expression data

        Returns:
            Appropriate Expression subclass instance
        """
        expression_type = data.get("expressionType", "")

        if expression_type == "literal":
            from .literal_expression import LiteralExpression

            return LiteralExpression.from_dict(data)
        elif expression_type == "variable":
            from .variable_expression import VariableExpression

            return VariableExpression.from_dict(data)
        elif expression_type == "methodCall":
            from .method_call_expression import MethodCallExpression

            return MethodCallExpression.from_dict(data)
        elif expression_type == "binaryOperation":
            from .binary_operation_expression import BinaryOperationExpression

            return BinaryOperationExpression.from_dict(data)
        elif expression_type == "builder":
            from .builder_expression import BuilderExpression

            return BuilderExpression.from_dict(data)
        else:
            raise ValueError(f"Unknown expression type: {expression_type}")

    def to_dict(self) -> dict:
        """Convert Expression to dictionary representation.

        Returns:
            Dictionary representation
        """
        return {"expressionType": self.expression_type}

    @abstractmethod
    def evaluate(self, context: dict) -> Any:
        """Evaluate the expression in the given context.

        Args:
            context: Variable context for evaluation

        Returns:
            The computed value
        """
        pass
