"""Parameter model - ported from Qontinui framework.

Represents a parameter definition for automation functions in the DSL.
"""

from dataclasses import dataclass


@dataclass
class Parameter:
    """Represents a parameter definition for automation functions.

    Port of Parameter from Qontinui framework class.

    Parameters allow functions to accept input values, making them reusable
    and configurable. Each parameter has a name and type, which are used for
    validation and type checking when the function is called.

    When a function is invoked, arguments are matched to parameters by position,
    and type compatibility is verified to ensure correct execution.

    Example in JSON (as part of a function definition):
        "parameters": [
            {"name": "elementId", "type": "string"},
            {"name": "timeout", "type": "integer"},
            {"name": "retry", "type": "boolean"}
        ]
    """

    name: str = ""
    """The name of the parameter.
    Used to reference the parameter value within the function body
    and for documentation purposes."""

    type: str = ""
    """The data type of the parameter.
    Common types include: "boolean", "string", "integer", "double", "object", "array"
    Used for type checking when arguments are passed to the function."""
