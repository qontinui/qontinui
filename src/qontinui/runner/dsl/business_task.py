"""Business task - ported from Qontinui framework.

Represents a single automation function in the DSL.
"""


from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model.parameter import Parameter
    from .statements.statement import Statement


@dataclass
class BusinessTask:
    """Represents a single automation function in the DSL.

    Port of BusinessTask from Qontinui framework class.

    An automation function is a reusable unit of automation logic that can:
    - Accept parameters for customization
    - Execute a series of statements to perform automation tasks
    - Return a value to the caller
    - Call other automation functions

    Functions are typically defined in JSON and parsed at runtime, allowing for
    dynamic automation script creation without recompilation.
    """

    id: int | None = None
    """Unique identifier for this function within the automation context.
    Used for referencing and debugging purposes."""

    name: str = ""
    """The name of this function, used when calling it from other functions
    or from the main automation script."""

    description: str = ""
    """Human-readable description of what this function does.
    This helps maintainers understand the function's purpose."""

    return_type: str = "void"
    """The data type that this function returns (e.g., "string", "boolean", "void").
    Used for type checking and validation during DSL parsing."""

    parameters: list[Parameter] = field(default_factory=list)
    """List of parameters that this function accepts.
    Parameters allow the function to be customized for different use cases."""

    statements: list[Statement] = field(default_factory=list)
    """The ordered list of statements that make up this function's body.
    These statements are executed sequentially when the function is called."""
