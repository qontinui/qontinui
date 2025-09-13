#!/usr/bin/env python3
"""Test the refactored model package structure."""

import sys
sys.path.insert(0, 'src')

# Test model.element imports
from qontinui.model.element import Location, Region, Image, Pattern, RGB, HSV
print("✓ model.element imports successful")

# Test model.state imports  
from qontinui.model.state import State, StateImage, StateObject, StateLocation
print("✓ model.state imports successful")

# Test model.transition imports
from qontinui.model.transition import StateTransition, StateTransitions, TransitionType
print("✓ model.transition imports successful")

# Test model.match imports
from qontinui.model.match import MatchObject
print("✓ model.match imports successful")

# Test find package imports
from qontinui.find import Find, FindImage, Match, Matches, FindResults
print("✓ find package imports successful")

# Test actions package imports
from qontinui.actions import Action, ActionConfig, ClickOptions, DragOptions
print("✓ actions package imports successful")

# Test primitives package imports
from qontinui.primitives import MouseClick, MouseMove, TypeText
print("✓ primitives package imports successful")

print("\n✅ All imports successful! Refactoring complete.")