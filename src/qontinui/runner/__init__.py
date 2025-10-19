"""Runner package - ported from Qontinui framework.

The runner package provides the Domain Specific Language (DSL) and JSON
infrastructure for defining automation functions declaratively.

Key components:
- DSL: Domain-specific language for automation definitions
- JSON: Parsing and validation of JSON-based automation scripts
- Executor: Execution engine for DSL statements and expressions
"""

from .dsl import BusinessTask, InstructionSet, Parameter

__all__ = [
    "BusinessTask",
    "InstructionSet",
    "Parameter",
]
