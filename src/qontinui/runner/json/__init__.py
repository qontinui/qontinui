"""JSON infrastructure package - ported from Qontinui framework.

Handles JSON parsing, serialization, and validation for the DSL.
"""

from .dsl_parser import DSLParser, DSLValidator

__all__ = [
    'DSLParser',
    'DSLValidator',
]