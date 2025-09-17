"""Test for Key enum and related classes."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qontinui.actions import Key, KeyCombo, KeyCombos, KeyModifier


def test_key_enum():
    """Test basic Key enum functionality."""
    print("Testing Key enum...")
    
    # Test string values
    assert Key.ENTER.value == "\n"
    assert Key.TAB.value == "\t"
    assert Key.ESCAPE.value == "escape"
    assert Key.F1.value == "f1"
    assert Key.UP.value == "up"
    
    # Test string conversion
    assert str(Key.ENTER) == "\n"
    assert str(Key.ESCAPE) == "escape"
    
    print("✓ Key enum basic values work correctly")


def test_key_classification():
    """Test key classification methods."""
    print("\nTesting key classification...")
    
    # Test modifier detection
    assert Key.is_modifier(Key.SHIFT) == True
    assert Key.is_modifier(Key.CTRL) == True
    assert Key.is_modifier(Key.ALT) == True
    assert Key.is_modifier(Key.ENTER) == False
    assert Key.is_modifier(Key.F1) == False
    
    # Test function key detection
    assert Key.is_function_key(Key.F1) == True
    assert Key.is_function_key(Key.F12) == True
    assert Key.is_function_key(Key.ENTER) == False
    assert Key.is_function_key(Key.SHIFT) == False
    
    # Test navigation key detection
    assert Key.is_navigation_key(Key.UP) == True
    assert Key.is_navigation_key(Key.DOWN) == True
    assert Key.is_navigation_key(Key.HOME) == True
    assert Key.is_navigation_key(Key.ENTER) == False
    
    print("✓ Key classification methods work correctly")


def test_key_from_string():
    """Test converting strings to Key enums."""
    print("\nTesting Key.from_string()...")
    
    # Test by enum name
    assert Key.from_string("ENTER") == Key.ENTER
    assert Key.from_string("enter") == Key.ENTER
    assert Key.from_string("ESCAPE") == Key.ESCAPE
    assert Key.from_string("F1") == Key.F1
    
    # Test by value
    assert Key.from_string("up") == Key.UP
    assert Key.from_string("escape") == Key.ESCAPE
    
    # Test error case
    try:
        Key.from_string("invalid_key")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "No Key enum found" in str(e)
    
    print("✓ Key.from_string() works correctly")


def test_key_combo():
    """Test KeyCombo class."""
    print("\nTesting KeyCombo...")
    
    # Test simple combo
    combo = KeyCombo(Key.CTRL, 'a')
    assert Key.CTRL in combo.modifiers
    assert combo.main_key == 'a'
    assert str(combo) == "ctrl+a"
    
    # Test multiple modifiers
    combo = KeyCombo(Key.CTRL, Key.SHIFT, 's')
    assert Key.CTRL in combo.modifiers
    assert Key.SHIFT in combo.modifiers
    assert combo.main_key == 's'
    
    # Test static methods
    combo = KeyCombo.ctrl('c')
    assert Key.CTRL in combo.modifiers
    assert combo.main_key == 'c'
    
    combo = KeyCombo.alt(Key.F4)
    assert Key.ALT in combo.modifiers
    assert combo.main_key == Key.F4
    
    print("✓ KeyCombo works correctly")


def test_key_combos_constants():
    """Test predefined key combinations."""
    print("\nTesting KeyCombos constants...")
    
    # Test common combos
    assert Key.CTRL in KeyCombos.COPY.modifiers
    assert KeyCombos.COPY.main_key == 'c'
    
    assert Key.CTRL in KeyCombos.PASTE.modifiers
    assert KeyCombos.PASTE.main_key == 'v'
    
    assert Key.ALT in KeyCombos.ALT_F4.modifiers
    assert KeyCombos.ALT_F4.main_key == Key.F4
    
    # Test Mac combos
    assert Key.META in KeyCombos.MAC_COPY.modifiers
    assert KeyCombos.MAC_COPY.main_key == 'c'
    
    print("✓ KeyCombos constants work correctly")


def test_key_modifier_compatibility():
    """Test that KeyModifier enum is compatible with Key enum."""
    print("\nTesting KeyModifier compatibility...")
    
    # KeyModifier should be importable
    assert KeyModifier.CTRL.value == "ctrl"
    assert KeyModifier.ALT.value == "alt"
    assert KeyModifier.SHIFT.value == "shift"
    
    # Can convert KeyModifier to Key
    combo = KeyCombo(KeyModifier.CTRL, 's')
    assert len(combo.modifiers) == 1
    # The modifier should be converted to a Key
    assert combo.modifiers[0] == Key.CTRL
    
    print("✓ KeyModifier is compatible with Key enum")


def test_special_characters():
    """Test special character representations."""
    print("\nTesting special characters...")
    
    # Test special string values
    assert Key.ENTER.value == "\n"
    assert Key.RETURN.value == "\r"
    assert Key.TAB.value == "\t"
    assert Key.SPACE.value == " "
    assert Key.BACKSPACE.value == "\b"
    
    # Test that string values are usable
    enter_str = str(Key.ENTER)
    assert enter_str == "\n"
    assert len(enter_str) == 1
    
    tab_str = str(Key.TAB)
    assert tab_str == "\t"
    assert len(tab_str) == 1
    
    print("✓ Special character representations work correctly")


def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing Key enum implementation for Qontinui")
    print("=" * 50)
    
    try:
        test_key_enum()
        test_key_classification()
        test_key_from_string()
        test_key_combo()
        test_key_combos_constants()
        test_key_modifier_compatibility()
        test_special_characters()
        
        print("\n" + "=" * 50)
        print("✅ All tests passed successfully!")
        print("=" * 50)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())