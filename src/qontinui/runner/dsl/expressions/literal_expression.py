"""Literal expression - ported from Qontinui framework.

Represents a literal value expression in the DSL.
"""

from dataclasses import dataclass
from typing import Any

from .expression import Expression


@dataclass
class LiteralExpression(Expression):
    """Represents a literal value expression in the DSL.

    Port of LiteralExpression from Qontinui framework class.

    A literal expression represents a constant value like a string, number,
    or boolean. The value is stored directly and returned when evaluated.

    Example in JSON:
        {"expressionType": "literal", "valueType": "string", "value": "Hello World"}
        {"expressionType": "literal", "valueType": "integer", "value": 42}
        {"expressionType": "literal", "valueType": "boolean", "value": true}
        {"expressionType": "literal", "valueType": "double", "value": 3.14}
    """

    value_type: str = ""
    """The data type of the literal value.
    Common types: "boolean", "string", "integer", "double", "null"."""

    value: Any = None
    """The actual literal value."""

    def __init__(self, value_type: str = "", value: Any = None):
        """Initialize literal expression.

        Args:
            value_type: Type of the literal
            value: The literal value
        """
        super().__init__("literal")
        self.value_type = value_type
        self.value = value

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LiteralExpression":
        """Create LiteralExpression from dictionary.

        Args:
            data: Dictionary with expression data

        Returns:
            LiteralExpression instance
        """
        return cls(value_type=data.get("valueType", ""), value=data.get("value"))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation
        """
        result = super().to_dict()
        result["valueType"] = self.value_type
        result["value"] = self.value
        return result

    def evaluate(self, context: dict[str, Any]) -> Any:
        """Evaluate the expression (returns the literal value).

        Args:
            context: Variable context (unused for literals)

        Returns:
            The literal value
        """
        return self.value
