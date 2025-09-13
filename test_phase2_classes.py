#!/usr/bin/env python3
"""Test Phase 2 newly implemented model classes."""

import sys
sys.path.insert(0, 'src')

# Test Movement class
from qontinui.model.element import Movement, Location
print("Testing Movement class...")
move = Movement(Location(10, 20), Location(100, 150))
print(f"✓ Movement: delta=({move.delta_x}, {move.delta_y}), distance={move.distance:.1f}")
reversed_move = move.reverse()
print(f"✓ Reversed movement: start={reversed_move.start_location}, end={reversed_move.end_location}")

# Test special states
from qontinui.model.state import NullState, UnknownState
print("\nTesting special states...")
null_state = NullState()
print(f"✓ NullState: {null_state}, is_null={null_state.is_null()}")
unknown_state = UnknownState.instance()
print(f"✓ UnknownState: {unknown_state}, is_unknown={unknown_state.is_unknown()}")

# Test MouseButton enum
from qontinui.model.action import MouseButton
print("\nTesting MouseButton enum...")
left_button = MouseButton.LEFT
print(f"✓ MouseButton: {left_button}, is_primary={left_button.is_primary()}")
right_button = MouseButton.from_string("right")
print(f"✓ MouseButton from string: {right_button}, pyautogui='{right_button.to_pyautogui()}'")

# Test ActionRecord and ActionHistory
from qontinui.model.action import ActionRecord, ActionRecordBuilder, ActionHistory
print("\nTesting ActionRecord and ActionHistory...")
record = ActionRecordBuilder() \
    .set_state("MainMenu", 1) \
    .set_action_success(True) \
    .set_duration(0.5) \
    .set_text("Button clicked") \
    .build()
print(f"✓ ActionRecord: {record}")

history = ActionHistory()
history.add_record(record)
print(f"✓ ActionHistory: {history}, success_rate={history.get_success_rate():.0%}")

# Test EmptyMatch
from qontinui.model.match import EmptyMatch
print("\nTesting EmptyMatch...")
empty = EmptyMatch.builder() \
    .set_name("search failed") \
    .build()
print(f"✓ EmptyMatch: {empty}, exists={empty.exists()}, is_empty={empty.is_empty()}")

print("\n✅ All Phase 2 classes working correctly!")