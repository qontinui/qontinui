"""Return statement - ported from Qontinui framework.

Represents a return statement in the DSL.
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .statement import Statement

if TYPE_CHECKING:
    from ..expressions.expression import Expression


@dataclass
class ReturnStatement(Statement):
    """Represents a return statement in the DSL.

    Port of ReturnStatement from Qontinui framework class.

    This statement returns a value from a function and terminates its execution.
    The type of the returned value should match the function's declared return type.

    Example in JSON:
        {
            "statementType": "return",
            "value": {"expressionType": "variable", "name": "result"}
        }

    Or for void return:
        {
            "statementType": "return"
        }
    """

    value: Expression | None = None
    """The expression whose value to return.
    None for void functions that don't return a value."""

    def __init__(self, value: Expression | None = None) -> None:
        """Initialize return statement.

        Args:
            value: Expression to evaluate and return
        """
        super().__init__("return")
        self.value = value

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReturnStatement:
        """Create ReturnStatement from dictionary.

        Args:
            data: Dictionary with statement data

        Returns:
            ReturnStatement instance
        """
        value = None
        if "value" in data:
            from ..expressions.expression import Expression

            value = Expression.from_dict(data["value"])

        return cls(value=value)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation
        """
        result = super().to_dict()
        if self.value:
            result["value"] = self.value.to_dict()
        return result
