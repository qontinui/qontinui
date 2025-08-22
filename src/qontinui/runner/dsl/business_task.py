"""Business task - ported from Qontinui framework.

Represents a single automation function in the DSL.
"""

from typing import List, Optional
from dataclasses import dataclass, field


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
    
    id: Optional[int] = None
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
    
    parameters: List['Parameter'] = field(default_factory=list)
    """List of parameters that this function accepts.
    Parameters allow the function to be customized for different use cases."""
    
    statements: List['Statement'] = field(default_factory=list)
    """The ordered list of statements that make up this function's body.
    These statements are executed sequentially when the function is called."""


# Forward references - will be imported when implementing
class Parameter:
    """Placeholder for Parameter class."""
    pass


class Statement:
    """Placeholder for Statement class."""
    pass