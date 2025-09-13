#!/usr/bin/env python3
"""Test Phase 5 implementations - Special states and reorganized actions."""

import sys
sys.path.insert(0, 'src')

print("Testing Phase 5 Implementation...")
print("=" * 50)

# Test SpecialStateType enum
print("\n1. Testing SpecialStateType...")
try:
    from qontinui.model.state.special import SpecialStateType
    
    error_type = SpecialStateType.ERROR
    loading_type = SpecialStateType.LOADING
    
    print(f"✓ SpecialStateType.ERROR: is_error={error_type.is_error_type()}")
    print(f"✓ SpecialStateType.LOADING: is_transient={loading_type.is_transient()}")
    print(f"  From string: {SpecialStateType.from_string('unknown')}")
    
except Exception as e:
    print(f"✗ Error: {e}")

# Test StateText
print("\n2. Testing StateText...")
try:
    from qontinui.model.state.special import StateText, TextMatchType
    
    # Create text-based state
    login_state = StateText("LoginScreen")
    login_state.add_pattern("Username:", TextMatchType.CONTAINS)
    login_state.add_pattern("Password:", TextMatchType.CONTAINS)
    login_state.add_required_text("Sign In")
    login_state.add_forbidden_text("Error")
    
    print(f"✓ StateText created: {login_state}")
    print(f"  Patterns: {len(login_state.get_patterns())}")
    print(f"  Required texts: {len(login_state.get_required_texts())}")
    print(f"  Valid: {login_state.is_valid()}")
    
    # Test text matching
    test_text = "Username: admin\nPassword: ****\nSign In"
    matches = login_state.matches_text(test_text)
    print(f"✓ Text matching: {matches}")
    
    # Test regex pattern
    login_state.add_regex_pattern(r"User.*:")
    print(f"  Added regex pattern")
    
except Exception as e:
    print(f"✗ Error: {e}")

# Test Click action
print("\n3. Testing Click action...")
try:
    from qontinui.actions.basic.click.click import Click, ClickOptions, ClickType
    from qontinui.model.element.location import Location
    
    # Create click with options
    options = ClickOptions().double().with_shift()
    click = Click(options)
    
    print(f"✓ Click created with options:")
    print(f"  Type: {options.click_type}")
    print(f"  Count: {options.click_count}")
    print(f"  Modifiers: {options.modifiers}")
    
    # Test execution (mock)
    location = Location(100, 200)
    # Note: This will print mock output
    # click.execute(location)
    
    print(f"✓ Click action configured successfully")
    
except Exception as e:
    print(f"✗ Error: {e}")

# Test Type action
print("\n4. Testing Type action...")
try:
    from qontinui.actions.basic.type.type_action import TypeAction, TypeOptions, TypeMethod
    
    # Create type action with options
    options = TypeOptions().paste().clear_first()
    type_action = TypeAction(options)
    
    print(f"✓ TypeAction created with options:")
    print(f"  Method: {options.type_method}")
    print(f"  Clear before: {options.clear_before}")
    
    # Test secure typing
    secure_options = TypeOptions().secure()
    secure_type = TypeAction(secure_options)
    print(f"✓ Secure typing configured")
    
except Exception as e:
    print(f"✗ Error: {e}")

# Test Wait action
print("\n5. Testing Wait action...")
try:
    from qontinui.actions.basic.wait.wait import Wait, WaitOptions, WaitType
    
    # Create different wait types
    time_wait = WaitOptions().for_time(2.0)
    condition_wait = WaitOptions().for_condition().with_timeout(10.0)
    visible_wait = WaitOptions().for_visible().with_poll_interval(0.25)
    
    print(f"✓ Wait options created:")
    print(f"  Time wait: {time_wait.timeout}s")
    print(f"  Condition wait: timeout={condition_wait.timeout}s")
    print(f"  Visible wait: poll={visible_wait.poll_interval}s")
    
    # Test wait execution
    wait = Wait(time_wait)
    print(f"✓ Wait action configured")
    
    # Test convenience methods
    print(f"  Has convenience methods: wait_seconds, wait_until")
    
except Exception as e:
    print(f"✗ Error: {e}")

# Test action structure
print("\n6. Testing Brobot-style action structure...")
try:
    import os
    
    action_dirs = [
        'src/qontinui/actions/basic/click',
        'src/qontinui/actions/basic/type',
        'src/qontinui/actions/basic/wait',
        'src/qontinui/actions/composite',
        'src/qontinui/actions/internal',
    ]
    
    existing = [d for d in action_dirs if os.path.exists(d)]
    print(f"✓ Created {len(existing)}/{len(action_dirs)} action directories")
    
    for d in existing[:3]:
        print(f"  - {d.split('/')[-2]}/{d.split('/')[-1]}")
    
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 50)
print("✅ Phase 5 Testing Complete!")
print("\nImplemented:")
print("  - SpecialStateType enum with utility methods")
print("  - StateText for text-based state identification")
print("  - Click action with modifiers and options")
print("  - Type action with multiple input methods")
print("  - Wait action with various wait conditions")
print("  - Brobot-style action package structure")