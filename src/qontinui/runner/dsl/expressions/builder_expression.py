"""Builder expression - ported from Qontinui framework.

Represents a builder pattern expression in the DSL.
"""

from dataclasses import dataclass, field
from typing import Any

from .expression import Expression


@dataclass
class BuilderExpression(Expression):
    """Represents a builder pattern expression in the DSL.

    Port of BuilderExpression from Qontinui framework class.

    A builder expression constructs complex objects using the builder pattern,
    where methods are chained together to configure the object before building it.
    This is commonly used for creating action configurations, object collections,
    and other complex objects in Brobot.

    Example in JSON:
        {
            "expressionType": "builder",
            "builderType": "ObjectCollection.Builder",
            "methodCalls": [
                {
                    "method": "withImages",
                    "arguments": [{"expressionType": "variable", "name": "targetImage"}]
                },
                {
                    "method": "withSearchRegions",
                    "arguments": [{"expressionType": "variable", "name": "searchArea"}]
                },
                {
                    "method": "build",
                    "arguments": []
                }
            ]
        }
    """

    builder_type: str = ""
    """The type of builder being used (e.g., "ObjectCollection.Builder")."""

    method_calls: list["BuilderMethodCall"] = field(default_factory=list)
    """Sequence of method calls to configure and build the object."""

    def __init__(
        self, builder_type: str = "", method_calls: list["BuilderMethodCall"] | None = None
    ):
        """Initialize builder expression.

        Args:
            builder_type: Type of builder
            method_calls: Builder method calls
        """
        super().__init__("builder")
        self.builder_type = builder_type
        self.method_calls = method_calls or []

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BuilderExpression":
        """Create BuilderExpression from dictionary.

        Args:
            data: Dictionary with expression data

        Returns:
            BuilderExpression instance
        """
        method_calls = []
        if "methodCalls" in data:
            method_calls = [BuilderMethodCall.from_dict(mc) for mc in data["methodCalls"]]

        return cls(builder_type=data.get("builderType", ""), method_calls=method_calls)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation
        """
        result = super().to_dict()
        result["builderType"] = self.builder_type
        if self.method_calls:
            result["methodCalls"] = [mc.to_dict() for mc in self.method_calls]
        return result

    def evaluate(self, context: dict[str, Any]) -> Any:
        """Evaluate the builder expression.

        Args:
            context: Variable context for evaluation

        Returns:
            The built object
        """
        # In a real implementation, this would:
        # 1. Create an instance of the builder
        # 2. Call each method in sequence
        # 3. Return the final built object
        return f"Builder({self.builder_type})"


@dataclass
class BuilderMethodCall:
    """Represents a single method call in a builder chain.

    Port of BuilderMethodCall from Qontinui framework nested class.
    """

    method: str = ""
    """The method name to call on the builder."""

    arguments: list[Expression] = field(default_factory=list)
    """Arguments to pass to the method."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BuilderMethodCall":
        """Create BuilderMethodCall from dictionary.

        Args:
            data: Dictionary with method call data

        Returns:
            BuilderMethodCall instance
        """
        arguments = []
        if "arguments" in data:
            arguments = [Expression.from_dict(arg) for arg in data["arguments"]]

        return cls(method=data.get("method", ""), arguments=arguments)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation
        """
        result: dict[str, Any] = {"method": self.method}
        if self.arguments:
            result["arguments"] = [arg.to_dict() for arg in self.arguments]
        return result
