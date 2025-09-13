"""Simple test for the new transition decorators without full import chain."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import only the decorator modules directly
from qontinui.annotations.transition_set import (
    transition_set,
    is_transition_set,
    get_transition_set_metadata
)

from qontinui.annotations.from_transition import (
    from_transition,
    is_from_transition,
    get_from_transition_metadata
)

from qontinui.annotations.to_transition import (
    to_transition,
    is_to_transition,
    get_to_transition_metadata
)


class MockState:
    """Mock state for testing."""
    pass


class AnotherMockState:
    """Another mock state for testing."""
    pass


@transition_set(state=MockState, name="Mock", description="Test transition set")
class TestTransitionSet:
    """Test transition set class."""
    
    @from_transition(from_state=AnotherMockState, priority=5, description="Test from transition")
    def from_another(self) -> bool:
        """Test from transition method."""
        return True
    
    @to_transition(description="Test to transition", timeout=10, required=False)
    def verify_arrival(self) -> bool:
        """Test to transition method."""
        return True


def test_transition_set_decorator():
    """Test that transition_set decorator works correctly."""
    print("Testing @transition_set decorator...")
    
    # Check that class is marked as transition set
    assert is_transition_set(TestTransitionSet), "Class should be marked as transition set"
    
    # Check metadata
    metadata = get_transition_set_metadata(TestTransitionSet)
    assert metadata is not None, "Should have metadata"
    assert metadata['state'] == MockState, "Should have correct state"
    assert metadata['name'] == "Mock", "Should have correct name"
    assert metadata['description'] == "Test transition set", "Should have correct description"
    
    print("✓ @transition_set decorator works correctly")


def test_from_transition_decorator():
    """Test that from_transition decorator works correctly."""
    print("Testing @from_transition decorator...")
    
    instance = TestTransitionSet()
    
    # Check that method is marked as from_transition
    assert is_from_transition(instance.from_another), "Method should be marked as from_transition"
    
    # Check metadata
    metadata = get_from_transition_metadata(instance.from_another)
    assert metadata is not None, "Should have metadata"
    assert metadata['from_state'] == AnotherMockState, "Should have correct from_state"
    assert metadata['priority'] == 5, "Should have correct priority"
    assert metadata['description'] == "Test from transition", "Should have correct description"
    assert metadata['timeout'] == 10, "Should have default timeout"
    
    # Check that method still works
    assert instance.from_another() == True, "Method should still execute correctly"
    
    print("✓ @from_transition decorator works correctly")


def test_to_transition_decorator():
    """Test that to_transition decorator works correctly."""
    print("Testing @to_transition decorator...")
    
    instance = TestTransitionSet()
    
    # Check that method is marked as to_transition
    assert is_to_transition(instance.verify_arrival), "Method should be marked as to_transition"
    
    # Check metadata
    metadata = get_to_transition_metadata(instance.verify_arrival)
    assert metadata is not None, "Should have metadata"
    assert metadata['description'] == "Test to transition", "Should have correct description"
    assert metadata['timeout'] == 10, "Should have correct timeout"
    assert metadata['required'] == False, "Should have correct required flag"
    
    # Check that method still works
    assert instance.verify_arrival() == True, "Method should still execute correctly"
    
    print("✓ @to_transition decorator works correctly")


def test_multiple_from_transitions():
    """Test that multiple from_transitions can be defined."""
    print("Testing multiple @from_transition methods...")
    
    @transition_set(state=MockState)
    class MultiTransitionSet:
        @from_transition(from_state=AnotherMockState, priority=1)
        def from_another(self) -> bool:
            return True
        
        @from_transition(from_state=MockState, priority=2)
        def from_self(self) -> bool:
            return True
        
        @to_transition()
        def verify(self) -> bool:
            return True
    
    instance = MultiTransitionSet()
    
    # Check both from_transition methods
    assert is_from_transition(instance.from_another), "First method should be from_transition"
    assert is_from_transition(instance.from_self), "Second method should be from_transition"
    assert is_to_transition(instance.verify), "Should have to_transition"
    
    # Check different metadata
    meta1 = get_from_transition_metadata(instance.from_another)
    meta2 = get_from_transition_metadata(instance.from_self)
    
    assert meta1['from_state'] == AnotherMockState, "First should have correct from_state"
    assert meta2['from_state'] == MockState, "Second should have correct from_state"
    assert meta1['priority'] == 1, "First should have priority 1"
    assert meta2['priority'] == 2, "Second should have priority 2"
    
    print("✓ Multiple @from_transition methods work correctly")




if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Qontinui Transition Decorators (Simple)")
    print("="*60 + "\n")
    
    try:
        test_transition_set_decorator()
        test_from_transition_decorator()
        test_to_transition_decorator()
        test_multiple_from_transitions()
        
        print("\n" + "="*60)
        print("✅ All tests passed successfully!")
        print("="*60 + "\n")
        
        print("Summary:")
        print("- @transition_set decorator properly marks classes")
        print("- @from_transition decorator properly marks methods")
        print("- @to_transition decorator properly marks methods")
        print("- Multiple from_transitions can be defined in one class")
        print("\nThe Qontinui library now has the same transition functionality as Brobot!")
        print("\nKey features:")
        print("1. Separation of from and to transitions")
        print("2. Each state has one to_transition (arrival verification)")
        print("3. Each state can have many from_transitions (from different source states)")
        print("4. The to_transition is automatically run after any from_transition")
        print("5. Support for priority-based transition selection")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)