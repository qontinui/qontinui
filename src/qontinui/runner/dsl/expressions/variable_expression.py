"""Variable expression - ported from Qontinui framework.

Represents a variable reference expression in the DSL.
"""

from dataclasses import dataclass
from typing import Any

from .expression import Expression


@dataclass
class VariableExpression(Expression):
    """Represents a variable reference expression in the DSL.

    Port of VariableExpression from Qontinui framework class.

    A variable expression references a previously declared variable by name.
    When evaluated, it returns the current value of that variable from the context.

    Example in JSON:
        {"expressionType": "variable", "name": "userName"}
        {"expressionType": "variable", "name": "count"}
    """

    name: str = ""
    """The name of the variable to reference.
    Must correspond to a variable in the current scope."""

    def __init__(self, name: str = ""):
        """Initialize variable expression.

        Args:
            name: Name of the variable
        """
        super().__init__("variable")
        self.name = name

    @classmethod
    def from_dict(cls, data: dict) -> "VariableExpression":
        """Create VariableExpression from dictionary.

        Args:
            data: Dictionary with expression data

        Returns:
            VariableExpression instance
        """
        return cls(name=data.get("name", ""))

    def to_dict(self) -> dict:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation
        """
        result = super().to_dict()
        result["name"] = self.name
        return result

    def evaluate(self, context: dict) -> Any:
        """Evaluate the expression (returns the variable value).

        Args:
            context: Variable context containing variable values

        Returns:
            The value of the variable

        Raises:
            KeyError: If variable not found in context
        """
        if self.name not in context:
            raise KeyError(f"Variable '{self.name}' not found in context")
        return context[self.name]
