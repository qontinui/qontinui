"""DSL package - ported from Qontinui framework.

Domain Specific Language for defining automation functions in JSON.
"""

from .business_task import BusinessTask
from .instruction_set import InstructionSet
from .model.parameter import Parameter

__all__ = [
    "BusinessTask",
    "InstructionSet",
    "Parameter",
]
