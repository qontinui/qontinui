"""Assignment statement - ported from Qontinui framework.

Represents an assignment statement in the DSL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .statement import Statement

if TYPE_CHECKING:
    from ..expressions.expression import Expression


@dataclass
class AssignmentStatement(Statement):
    """Represents an assignment statement in the DSL.

    Port of AssignmentStatement from Qontinui framework class.

    This statement assigns a value to an existing variable. The variable must
    have been previously declared in the current or an enclosing scope.

    Example in JSON:
        {
            "statementType": "assignment",
            "variableName": "result",
            "value": {"expressionType": "methodCall", "object": "calculator", "method": "add",
                     "arguments": [{"expressionType": "literal", "valueType": "integer", "value": 5},
                                  {"expressionType": "literal", "valueType": "integer", "value": 3}]}
        }
    """

    variable_name: str = ""
    """The name of the variable to assign to.
    Must reference an existing variable in scope."""

    value: Expression | None = None
    """The expression whose value will be assigned to the variable."""

    def __init__(self, variable_name: str = "", value: Expression | None = None):
        """Initialize assignment statement.

        Args:
            variable_name: Name of variable to assign to
            value: Expression to evaluate and assign
        """
        super().__init__("assignment")
        self.variable_name = variable_name
        self.value = value

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AssignmentStatement:
        """Create AssignmentStatement from dictionary.

        Args:
            data: Dictionary with statement data

        Returns:
            AssignmentStatement instance
        """
        from ..expressions.expression import Expression

        value = None
        if "value" in data:
            value = Expression.from_dict(data["value"])

        return cls(variable_name=data.get("variableName", ""), value=value)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation
        """
        result = super().to_dict()
        result["variableName"] = self.variable_name
        if self.value:
            result["value"] = self.value.to_dict()
        return result
