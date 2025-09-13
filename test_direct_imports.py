#!/usr/bin/env python3
"""Test direct imports of refactored model package."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test model.element imports directly
import qontinui.model.element.location as loc
import qontinui.model.element.region as reg
print("✓ model.element direct imports successful")

# Test model.state imports directly  
import qontinui.model.state.state as st
import qontinui.model.state.state_image as si
print("✓ model.state direct imports successful")

# Test model.transition imports directly
import qontinui.model.transition.state_transition as trans
print("✓ model.transition direct imports successful")

# Test model.match imports directly
import qontinui.model.match.match as m
print("✓ model.match direct imports successful")

# Test find package imports directly
import qontinui.find.find as f
import qontinui.find.find_image as fi
print("✓ find package direct imports successful")

# Test actions package imports directly
import qontinui.actions.action as a
import qontinui.actions.action_config as ac
print("✓ actions package direct imports successful")

# Test primitives package imports directly
import qontinui.primitives.mouse as mo
import qontinui.primitives.keyboard as kb
print("✓ primitives package direct imports successful")

print("\n✅ All direct imports successful! Refactoring complete.")