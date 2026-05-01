"""If statement - ported from Qontinui framework.

Represents an if statement in the DSL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .statement import Statement

if TYPE_CHECKING:
    from ..expressions.expression import Expression


@dataclass
class IfStatement(Statement):
    """Represents an if statement in the DSL.

    Port of IfStatement from Qontinui framework class.

    This statement provides conditional execution of statements based on a
    boolean condition. If the condition evaluates to true, the thenStatements
    are executed; otherwise, the elseStatements (if any) are executed.

    Example in JSON:
        {
            "statementType": "if",
            "condition": {"expressionType": "binaryOperation", "operator": ">",
                         "left": {"expressionType": "variable", "name": "count"},
                         "right": {"expressionType": "literal", "valueType": "integer", "value": 0}},
            "thenStatements": [
                {"statementType": "methodCall", "object": "logger", "method": "log",
                 "arguments": [{"expressionType": "literal", "valueType": "string", "value": "Count is positive"}]}
            ],
            "elseStatements": [
                {"statementType": "methodCall", "object": "logger", "method": "log",
                 "arguments": [{"expressionType": "literal", "valueType": "string", "value": "Count is zero or negative"}]}
            ]
        }
    """

    condition: Expression | None = None
    """The boolean expression to evaluate.
    Must evaluate to a boolean value."""

    then_statements: list[Statement] = field(default_factory=list)
    """Statements to execute if the condition is true."""

    else_statements: list[Statement] = field(default_factory=list)
    """Statements to execute if the condition is false.
    Can be empty for if-without-else."""

    def __init__(
        self,
        condition: Expression | None = None,
        then_statements: list[Statement] | None = None,
        else_statements: list[Statement] | None = None,
    ) -> None:
        """Initialize if statement.

        Args:
            condition: Boolean condition expression
            then_statements: Statements for true branch
            else_statements: Statements for false branch
        """
        super().__init__("if")
        self.condition = condition
        self.then_statements = then_statements or []
        self.else_statements = else_statements or []

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IfStatement:
        """Create IfStatement from dictionary.

        Args:
            data: Dictionary with statement data

        Returns:
            IfStatement instance
        """
        from ..expressions.expression import Expression

        condition = None
        if "condition" in data:
            condition = Expression.from_dict(data["condition"])

        then_statements = []
        if "thenStatements" in data:
            then_statements = [
                Statement.from_dict(stmt) for stmt in data["thenStatements"]
            ]

        else_statements = []
        if "elseStatements" in data:
            else_statements = [
                Statement.from_dict(stmt) for stmt in data["elseStatements"]
            ]

        return cls(
            condition=condition,
            then_statements=then_statements,
            else_statements=else_statements,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation
        """
        result = super().to_dict()
        if self.condition:
            result["condition"] = self.condition.to_dict()
        if self.then_statements:
            result["thenStatements"] = [stmt.to_dict() for stmt in self.then_statements]
        if self.else_statements:
            result["elseStatements"] = [stmt.to_dict() for stmt in self.else_statements]
        return result
