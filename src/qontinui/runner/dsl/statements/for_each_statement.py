"""ForEach statement - ported from Qontinui framework.

Represents a forEach loop statement in the DSL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .statement import Statement

if TYPE_CHECKING:
    from ..expressions.expression import Expression


@dataclass
class ForEachStatement(Statement):
    """Represents a forEach loop statement in the DSL.

    Port of ForEachStatement from Qontinui framework class.

    This statement iterates over a collection, executing a block of statements
    for each element. The current element is available through the specified
    variable name within the loop body.

    Example in JSON:
        {
            "statementType": "forEach",
            "variableName": "item",
            "collection": {"expressionType": "variable", "name": "items"},
            "statements": [
                {"statementType": "methodCall", "object": "processor", "method": "process",
                 "arguments": [{"expressionType": "variable", "name": "item"}]}
            ]
        }
    """

    variable_name: str = ""
    """The name of the loop variable that holds the current element.
    This variable is scoped to the loop body."""

    collection: Expression | None = None
    """Expression that evaluates to a collection to iterate over.
    Must evaluate to an array or iterable object."""

    statements: list[Statement] = field(default_factory=list)
    """Statements to execute for each element in the collection."""

    def __init__(
        self,
        variable_name: str = "",
        collection: Expression | None = None,
        statements: list[Statement] | None = None,
    ):
        """Initialize forEach statement.

        Args:
            variable_name: Name of loop variable
            collection: Collection expression to iterate
            statements: Loop body statements
        """
        super().__init__("forEach")
        self.variable_name = variable_name
        self.collection = collection
        self.statements = statements or []

    @classmethod
    def from_dict(cls, data: dict) -> ForEachStatement:
        """Create ForEachStatement from dictionary.

        Args:
            data: Dictionary with statement data

        Returns:
            ForEachStatement instance
        """
        from ..expressions.expression import Expression

        collection = None
        if "collection" in data:
            collection = Expression.from_dict(data["collection"])

        statements = []
        if "statements" in data:
            statements = [Statement.from_dict(stmt) for stmt in data["statements"]]

        return cls(
            variable_name=data.get("variableName", ""), collection=collection, statements=statements
        )

    def to_dict(self) -> dict:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation
        """
        result = super().to_dict()
        result["variableName"] = self.variable_name
        if self.collection:
            result["collection"] = self.collection.to_dict()
        if self.statements:
            result["statements"] = [stmt.to_dict() for stmt in self.statements]
        return result
