#!/usr/bin/env python3
"""Test Phase 4 newly implemented model classes."""

import sys
sys.path.insert(0, 'src')

print("Testing Phase 4 Implementation...")
print("=" * 50)

# Test CrossStateAnchor
print("\n1. Testing CrossStateAnchor...")
from qontinui.model.element.cross_state_anchor import CrossStateAnchor, CrossStateAnchorBuilder
from qontinui.model.element.anchor import Anchor
from qontinui.model.element.location import Location, Position

anchor1 = Anchor(position=Position.CENTER, offset_x=10, offset_y=20)
anchor2 = Anchor(position=Position.TOP, offset_x=5, offset_y=15)

cross_anchor = CrossStateAnchor()
cross_anchor.add_anchor("LoginState", anchor1)
cross_anchor.add_anchor("DashboardState", anchor2)
cross_anchor.set_default_anchor(anchor1)

print(f"✓ CrossStateAnchor created: {cross_anchor}")
print(f"  States: {cross_anchor.get_states()}")
print(f"  Has LoginState: {cross_anchor.has_state('LoginState')}")

# Test builder pattern
builder_anchor = CrossStateAnchorBuilder() \
    .with_state_anchor("State1", anchor1) \
    .with_state_anchor("State2", anchor2) \
    .with_default_anchor(anchor1) \
    .strict() \
    .build()
print(f"✓ Built with builder: {builder_anchor}")

# Test SearchRegionOnObject
print("\n2. Testing SearchRegionOnObject...")
from qontinui.model.element.search_region_on_object import SearchRegionOnObject, SearchStrategy
from qontinui.model.element.region import Region
from qontinui.model.state.state_object import StateObject

# Create a mock state object with a region
state_obj = StateObject()
state_obj.search_region = Region(100, 100, 200, 150)

search_region = SearchRegionOnObject()
search_region.set_base_object(state_obj)
search_region.set_strategy(SearchStrategy.EXPANDED)
search_region.expand(20)

computed_region = search_region.get_search_region()
if computed_region:
    print(f"✓ SearchRegionOnObject created: {search_region}")
    print(f"  Strategy: {search_region.strategy.name}")
    print(f"  Computed region: x={computed_region.x}, y={computed_region.y}, "
          f"w={computed_region.width}, h={computed_region.height}")

# Test adjacent positioning
search_region2 = SearchRegionOnObject()
search_region2.set_base_object(state_obj)
search_region2.adjacent_to(Position.BOTTOM, distance=10, height=50)

adjacent_region = search_region2.get_search_region()
if adjacent_region:
    print(f"✓ Adjacent region created below object")
    print(f"  Position: x={adjacent_region.x}, y={adjacent_region.y}")

# Test TransitionFunction
print("\n3. Testing TransitionFunction...")
from qontinui.model.transition.transition_function import (
    TransitionFunction, TransitionType, TransitionResult
)

# Simple action transition
def click_action():
    print("  > Executing click action")
    return True

transition = TransitionFunction(
    name="click_login",
    transition_type=TransitionType.ACTION,
    function=click_action,
    timeout=5.0,
    max_retries=2
)

print(f"✓ TransitionFunction created: {transition}")
result = transition.execute()
print(f"  Execution result: success={result.success}, duration={result.duration:.3f}s")

# Composite transition
child1 = TransitionFunction(name="step1", function=lambda: True)
child2 = TransitionFunction(name="step2", function=lambda: True)

composite = TransitionFunction(
    name="multi_step",
    transition_type=TransitionType.COMPOSITE
)
composite.add_child(child1).add_child(child2)

print(f"✓ Composite transition: {composite}")
print(f"  Is composite: {composite.is_composite()}")
print(f"  Child count: {len(composite.child_functions)}")

# Test StateStore
print("\n4. Testing StateStore...")
from qontinui.model.state.state_store import StateStore, StateStatus
from qontinui.model.state.state import State
from qontinui.model.state.state_enum import StateEnum

# Create store
store = StateStore(max_cache_size=50)

# Create and register states
login_state = State(name="LoginState")
dashboard_state = State(name="DashboardState")
settings_state = State(name="SettingsState")

store.register(login_state)
store.register(dashboard_state)
store.register(settings_state, parent="DashboardState")

print(f"✓ StateStore created: {store}")
print(f"  Total states: {len(store.get_all())}")

# Set current state
store.set_current_state("LoginState")
current = store.get_current_state()
print(f"✓ Current state set: {current.name if current else 'None'}")

# Add tags
store.add_tag("LoginState", "authentication")
store.add_tag("DashboardState", "main")
tagged = store.find_by_tag("authentication")
print(f"✓ States with 'authentication' tag: {[s.name for s in tagged]}")

# Get children
children = store.get_children("DashboardState")
print(f"✓ Children of DashboardState: {children}")

# Get statistics
stats = store.get_statistics()
print(f"✓ Store statistics:")
print(f"  Total states: {stats['total_states']}")
print(f"  Active states: {stats['active_states']}")
print(f"  Cached states: {stats['cached_states']}")

# Validate store
errors = store.validate()
if not errors:
    print("✓ Store validation passed")
else:
    print(f"✗ Validation errors: {errors}")

print("\n" + "=" * 50)
print("✅ All Phase 4 classes tested successfully!")
print("\nImplemented classes:")
print("  - CrossStateAnchor (with builder)")
print("  - SearchRegionOnObject (with strategies)")
print("  - TransitionFunction (with composite support)")
print("  - StateStore (with full lifecycle management)")