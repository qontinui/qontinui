"""Variable declaration statement - ported from Qontinui framework.

Represents a variable declaration statement in the DSL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .statement import Statement

if TYPE_CHECKING:
    from ..expressions.expression import Expression


@dataclass
class VariableDeclarationStatement(Statement):
    """Represents a variable declaration statement in the DSL.

    Port of VariableDeclarationStatement from Qontinui framework class.

    This statement declares a new variable in the current scope and optionally
    initializes it with a value. The variable can then be referenced by name
    in subsequent statements and expressions within its scope.

    Example in JSON:
        {
            "statementType": "variableDeclaration",
            "variableName": "elementId",
            "variableType": "string",
            "initialValue": {"expressionType": "literal", "valueType": "string", "value": "#submit-button"}
        }
    """

    variable_name: str = ""
    """The name of the variable being declared.
    Must be unique within the current scope."""

    variable_type: str = ""
    """The data type of the variable.
    Common types: "boolean", "string", "integer", "double", "object", "array"."""

    initial_value: Expression | None = None
    """Optional expression providing the initial value for the variable.
    If None, the variable is declared but not initialized."""

    def __init__(
        self,
        variable_name: str = "",
        variable_type: str = "",
        initial_value: Expression | None = None,
    ) -> None:
        """Initialize variable declaration statement.

        Args:
            variable_name: Name of variable
            variable_type: Type of variable
            initial_value: Initial value expression
        """
        super().__init__("variableDeclaration")
        self.variable_name = variable_name
        self.variable_type = variable_type
        self.initial_value = initial_value

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VariableDeclarationStatement:
        """Create VariableDeclarationStatement from dictionary.

        Args:
            data: Dictionary with statement data

        Returns:
            VariableDeclarationStatement instance
        """
        initial_value = None
        if "initialValue" in data:
            from ..expressions.expression import Expression

            initial_value = Expression.from_dict(data["initialValue"])

        return cls(
            variable_name=data.get("variableName", ""),
            variable_type=data.get("variableType", ""),
            initial_value=initial_value,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation
        """
        result = super().to_dict()
        result["variableName"] = self.variable_name
        result["variableType"] = self.variable_type
        if self.initial_value:
            result["initialValue"] = self.initial_value.to_dict()
        return result
