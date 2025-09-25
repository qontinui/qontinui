"""Runner package - ported from Qontinui framework.

The runner package provides the Domain Specific Language (DSL) and JSON
infrastructure for defining automation functions declaratively.

Key components:
- DSL: Domain-specific language for automation definitions
- JSON: Parsing and validation of JSON-based automation scripts
"""

from .dsl import BusinessTask, InstructionSet, Parameter
from .json.dsl_parser import DSLParser, DSLValidator

__all__ = [
    "BusinessTask",
    "InstructionSet",
    "Parameter",
    "DSLParser",
    "DSLValidator",
]
