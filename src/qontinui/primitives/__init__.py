"""Primitives package - ported from Qontinui framework.

Contains the most basic, primitive actions that form the foundation
of all higher-level actions. Each primitive does exactly one thing.
"""

from .keyboard import KeyDown, KeyPress, KeyUp, TypeText
from .mouse import MouseClick, MouseDown, MouseDrag, MouseMove, MouseUp, MouseWheel

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
