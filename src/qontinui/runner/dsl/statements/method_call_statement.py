"""Method call statement - ported from Qontinui framework.

Represents a method call statement in the DSL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .statement import Statement

if TYPE_CHECKING:
    from ..expressions.expression import Expression


@dataclass
class MethodCallStatement(Statement):
    """Represents a method call statement in the DSL.

    Port of MethodCallStatement from Qontinui framework class.

    This statement invokes a method for its side effects rather than its return value.
    While MethodCallExpression is used when the return value is needed, this statement
    is used when only the method's side effects are desired (e.g., logging, UI
    interactions, state modifications).

    The method can be called on an object instance or as a static/global function.

    Example in JSON:
        {
            "statementType": "methodCall",
            "object": "browser",
            "method": "click",
            "arguments": [{"expressionType": "literal", "valueType": "string", "value": "#submit-button"}]
        }
    """

    object: str | None = None
    """The name of the object on which to invoke the method.
    Can be None for static methods or global function calls.
    When non-None, this should reference a variable in the current scope."""

    method: str = ""
    """The name of the method or function to invoke.
    Must match an available method on the target object or a global function.
    The method is called for its side effects; any return value is discarded."""

    arguments: list[Expression] = field(default_factory=list)
    """List of argument expressions to pass to the method.
    Each expression is evaluated before the method call, and the resulting
    values are passed as arguments in the order specified."""

    def __init__(
        self,
        object: str | None = None,
        method: str = "",
        arguments: list[Expression] | None = None,
    ) -> None:
        """Initialize method call statement.

        Args:
            object: Object to call method on
            method: Method name
            arguments: Method arguments
        """
        super().__init__("methodCall")
        self.object = object
        self.method = method
        self.arguments = arguments or []

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MethodCallStatement:
        """Create MethodCallStatement from dictionary.

        Args:
            data: Dictionary with statement data

        Returns:
            MethodCallStatement instance
        """
        from ..expressions.expression import Expression

        arguments = []
        if "arguments" in data:
            arguments = [Expression.from_dict(arg) for arg in data["arguments"]]

        return cls(object=data.get("object"), method=data.get("method", ""), arguments=arguments)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation
        """
        result = super().to_dict()
        if self.object:
            result["object"] = self.object
        result["method"] = self.method
        if self.arguments:
            result["arguments"] = [arg.to_dict() for arg in self.arguments]
        return result
