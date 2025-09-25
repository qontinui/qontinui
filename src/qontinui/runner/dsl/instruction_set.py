"""Instruction set - ported from Qontinui framework.

Root data structure for the Domain-Specific Language (DSL) automation definitions.
"""

from dataclasses import dataclass, field

from .business_task import BusinessTask


@dataclass
class InstructionSet:
    """Root data structure for DSL automation definitions.

    Port of InstructionSet from Qontinui framework class.

    This class serves as the top-level container for automation functions defined in JSON format.
    It allows users to define reusable automation functions that can be executed by Qontinui.
    Each function represents a discrete automation task with its own parameters and logic.

    The DSL supports parsing JSON files that contain automation function definitions,
    enabling declarative automation script creation without direct Python programming.
    """

    automation_functions: list[BusinessTask] = field(default_factory=list)
    """List of automation functions defined in this DSL instance.
    Each function can be independently executed and may call other functions."""
