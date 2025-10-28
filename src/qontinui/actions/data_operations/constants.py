"""Constants and enumerations for data operations.

This module defines shared enums used across data operation components:
- VariableScope: Hierarchical variable storage levels
- ComparatorType: Types of comparators for sorting operations
- ComparisonOperator: Comparison operators for filtering
"""

from enum import Enum


class VariableScope(str, Enum):
    """Variable scope levels for hierarchical variable storage.

    Variables are resolved in order of priority:
    - LOCAL: Action-level variables (highest priority)
    - PROCESS: Process-level variables (medium priority)
    - GLOBAL: Application-level variables (lowest priority)
    """

    LOCAL = "local"
    GLOBAL = "global"
    PROCESS = "process"


class ComparatorType(str, Enum):
    """Types of comparators for sorting collections.

    Attributes:
        NUMERIC: Numeric comparison (converts to numbers)
        ALPHABETIC: Alphabetic comparison (converts to strings)
        DATE: Date comparison (parses ISO date strings)
        CUSTOM: Custom comparison using expression
    """

    NUMERIC = "NUMERIC"
    ALPHABETIC = "ALPHABETIC"
    DATE = "DATE"
    CUSTOM = "CUSTOM"


class ComparisonOperator(str, Enum):
    """Comparison operators for filtering collections.

    Attributes:
        EQUAL: Equality comparison (==)
        NOT_EQUAL: Inequality comparison (!=)
        GREATER: Greater than comparison (>)
        LESS: Less than comparison (<)
        GREATER_EQUAL: Greater than or equal comparison (>=)
        LESS_EQUAL: Less than or equal comparison (<=)
        CONTAINS: Containment check (in)
        MATCHES: Regular expression match
    """

    EQUAL = "=="
    NOT_EQUAL = "!="
    GREATER = ">"
    LESS = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    CONTAINS = "contains"
    MATCHES = "matches"
