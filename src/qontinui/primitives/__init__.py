"""Primitives package - ported from Qontinui framework.

Contains the most basic, primitive actions that form the foundation
of all higher-level actions. Each primitive does exactly one thing.
"""

from .mouse import MouseMove, MouseClick, MouseDrag, MouseWheel, MouseDown, MouseUp
from .keyboard import KeyPress, KeyDown, KeyUp, TypeText

__all__ = [
    # Mouse primitives
    "MouseMove",
    "MouseClick",
    "MouseDrag",
    "MouseWheel",
    "MouseDown",
    "MouseUp",
    
    # Keyboard primitives
    "KeyPress",
    "KeyDown",
    "KeyUp",
    "TypeText",
]