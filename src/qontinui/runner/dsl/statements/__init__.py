"""DSL statements package - ported from Qontinui framework.

Statement types for the Domain Specific Language.
"""

from .statement import Statement
from .assignment_statement import AssignmentStatement
from .for_each_statement import ForEachStatement
from .if_statement import IfStatement
from .method_call_statement import MethodCallStatement
from .return_statement import ReturnStatement
from .variable_declaration_statement import VariableDeclarationStatement

__all__ = [
    'Statement',
    'AssignmentStatement',
    'ForEachStatement',
    'IfStatement',
    'MethodCallStatement',
    'ReturnStatement',
    'VariableDeclarationStatement',
]