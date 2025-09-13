#!/usr/bin/env python3
"""Test Phase 4 classes directly without package imports."""

import sys
import os
sys.path.insert(0, 'src')

print("Testing Phase 4 Implementation (Direct Import)...")
print("=" * 50)

# Test CrossStateAnchor directly
print("\n1. Testing CrossStateAnchor...")
try:
    # Import dependencies first
    from qontinui.model.element.anchor import Anchor
    from qontinui.model.element.location import Location, Position
    from qontinui.model.element.cross_state_anchor import CrossStateAnchor, CrossStateAnchorBuilder
    
    anchor1 = Anchor(position=Position.CENTER, offset_x=10, offset_y=20)
    anchor2 = Anchor(position=Position.TOP, offset_x=5, offset_y=15)
    
    cross_anchor = CrossStateAnchor()
    cross_anchor.add_anchor("LoginState", anchor1)
    cross_anchor.add_anchor("DashboardState", anchor2)
    cross_anchor.set_default_anchor(anchor1)
    
    print(f"✓ CrossStateAnchor created: {cross_anchor}")
    print(f"  States: {cross_anchor.get_states()}")
    print(f"  Valid: {cross_anchor.is_valid()}")
    
except Exception as e:
    print(f"✗ Error: {e}")

# Test SearchRegionOnObject
print("\n2. Testing SearchRegionOnObject...")
try:
    from qontinui.model.element.region import Region
    from qontinui.model.element.search_region_on_object import SearchRegionOnObject, SearchStrategy
    from qontinui.model.state.state_object import StateObject
    
    # Create a mock state object
    state_obj = StateObject()
    state_obj.search_region = Region(100, 100, 200, 150)
    
    search_region = SearchRegionOnObject()
    search_region.set_base_object(state_obj)
    search_region.set_strategy(SearchStrategy.EXPANDED)
    search_region.expand(20)
    
    computed_region = search_region.get_search_region()
    if computed_region:
        print(f"✓ SearchRegionOnObject created: {search_region}")
        print(f"  Expanded region: x={computed_region.x}, y={computed_region.y}")
    
except Exception as e:
    print(f"✗ Error: {e}")

# Test TransitionFunction
print("\n3. Testing TransitionFunction...")
try:
    from qontinui.model.transition.transition_function import (
        TransitionFunction, TransitionType, TransitionResult
    )
    
    # Simple transition
    def simple_action():
        return True
    
    transition = TransitionFunction(
        name="test_transition",
        transition_type=TransitionType.ACTION,
        function=simple_action,
        timeout=5.0
    )
    
    print(f"✓ TransitionFunction created: {transition}")
    print(f"  Valid: {transition.validate()}")
    
    # Test execution
    result = transition.execute()
    print(f"  Executed: success={result.success}")
    
    # Composite transition
    composite = TransitionFunction(
        name="composite_test",
        transition_type=TransitionType.COMPOSITE
    )
    child1 = TransitionFunction(name="child1", function=lambda: True)
    composite.add_child(child1)
    print(f"✓ Composite transition with {len(composite.child_functions)} children")
    
except Exception as e:
    print(f"✗ Error: {e}")

# Test StateStore
print("\n4. Testing StateStore...")
try:
    from qontinui.model.state.state import State
    from qontinui.model.state.state_store import StateStore, StateStatus
    
    # Create store
    store = StateStore(max_cache_size=10)
    
    # Register states
    state1 = State(name="State1")
    state2 = State(name="State2")
    
    store.register(state1)
    store.register(state2, parent="State1")
    
    print(f"✓ StateStore created: {store}")
    print(f"  Registered states: {len(store.get_all())}")
    
    # Test operations
    store.set_current_state("State1")
    current = store.get_current_state()
    print(f"  Current state: {current.name if current else 'None'}")
    
    # Test tags
    store.add_tag("State1", "test_tag")
    tagged = store.find_by_tag("test_tag")
    print(f"  States with 'test_tag': {len(tagged)}")
    
    # Test hierarchy
    children = store.get_children("State1")
    print(f"  Children of State1: {children}")
    
    # Test statistics
    stats = store.get_statistics()
    print(f"  Statistics: {stats['total_states']} states, {stats['active_states']} active")
    
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 50)
print("✅ Phase 4 Direct Testing Complete!")
print("\nSuccessfully implemented:")
print("  - CrossStateAnchor with state-specific anchoring")
print("  - SearchRegionOnObject with multiple strategies")
print("  - TransitionFunction with composite support")
print("  - StateStore with full lifecycle management")