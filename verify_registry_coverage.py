#!/usr/bin/env python3
"""Verification script to check action executor registry coverage.

This script:
1. Imports action_executors package to trigger all registrations
2. Gets all registered action types via get_registered_action_types()
3. Compares against the expected 30 action types
4. Reports coverage and missing action types
"""

import sys
from pathlib import Path

# Add src directory to path to enable imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def main():
    """Main verification function."""
    print("=" * 80)
    print("ACTION EXECUTOR REGISTRY COVERAGE VERIFICATION")
    print("=" * 80)
    print()

    # Expected 30 action types from the analysis
    expected_action_types = {
        # Mouse (9)
        "MOUSE_MOVE",
        "MOUSE_DOWN",
        "MOUSE_UP",
        "MOUSE_SCROLL",
        "SCROLL",
        "CLICK",
        "DOUBLE_CLICK",
        "RIGHT_CLICK",
        "DRAG",
        # Keyboard (4)
        "KEY_DOWN",
        "KEY_UP",
        "KEY_PRESS",
        "TYPE",
        # Vision (3)
        "FIND",
        "EXISTS",
        "VANISH",
        # Navigation (2)
        "GO_TO_STATE",
        "RUN_WORKFLOW",
        # Utility (2)
        "WAIT",
        "SCREENSHOT",
        # Control Flow (4)
        "LOOP",
        "IF",
        "BREAK",
        "CONTINUE",
        # Data Operations (8)
        "SET_VARIABLE",
        "GET_VARIABLE",
        "MAP",
        "REDUCE",
        "SORT",
        "FILTER",
        "STRING_OPERATION",
        "MATH_OPERATION",
    }

    print(f"Expected action types: {len(expected_action_types)}")
    print()

    # Import action_executors to trigger all @register_executor decorators
    print("Importing action_executors package to trigger registrations...")
    try:
        from qontinui.action_executors import (
            get_executor_class,
            get_registered_action_types,
        )

        print("✓ Import successful")
        print()
    except Exception as e:
        print(f"✗ Failed to import action_executors: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Get all registered action types
    print("Getting registered action types from registry...")
    registered_types = set(get_registered_action_types())
    print(f"✓ Found {len(registered_types)} registered action types")
    print()

    # Calculate coverage
    coverage_count = len(expected_action_types & registered_types)
    coverage_percent = (coverage_count / len(expected_action_types)) * 100

    print("=" * 80)
    print("COVERAGE SUMMARY")
    print("=" * 80)
    print(f"Total expected: {len(expected_action_types)}")
    print(f"Total registered: {len(registered_types)}")
    print(f"Coverage: {coverage_count}/{len(expected_action_types)} ({coverage_percent:.1f}%)")
    print()

    # Check for missing action types
    missing_types = expected_action_types - registered_types
    extra_types = registered_types - expected_action_types

    if missing_types:
        print("=" * 80)
        print(f"MISSING ACTION TYPES ({len(missing_types)})")
        print("=" * 80)

        # Categorize missing types
        categories = {
            "Mouse": [
                "MOUSE_MOVE",
                "MOUSE_DOWN",
                "MOUSE_UP",
                "MOUSE_SCROLL",
                "SCROLL",
                "CLICK",
                "DOUBLE_CLICK",
                "RIGHT_CLICK",
                "DRAG",
            ],
            "Keyboard": ["KEY_DOWN", "KEY_UP", "KEY_PRESS", "TYPE"],
            "Vision": ["FIND", "EXISTS", "VANISH"],
            "Navigation": ["GO_TO_STATE", "RUN_WORKFLOW"],
            "Utility": ["WAIT", "SCREENSHOT"],
            "Control Flow": ["LOOP", "IF", "BREAK", "CONTINUE"],
            "Data Operations": [
                "SET_VARIABLE",
                "GET_VARIABLE",
                "MAP",
                "REDUCE",
                "SORT",
                "FILTER",
                "STRING_OPERATION",
                "MATH_OPERATION",
            ],
        }

        for category, category_types in categories.items():
            missing_in_category = [t for t in category_types if t in missing_types]
            if missing_in_category:
                print(f"\n{category}:")
                for action_type in missing_in_category:
                    print(f"  ✗ {action_type}")
    else:
        print("✓ All expected action types are registered!")

    if extra_types:
        print()
        print("=" * 80)
        print(f"EXTRA REGISTERED TYPES (not in expected list): {len(extra_types)}")
        print("=" * 80)
        for action_type in sorted(extra_types):
            executor_class = get_executor_class(action_type)
            print(
                f"  + {action_type} -> {executor_class.__name__ if executor_class else 'Unknown'}"
            )

    # Report which executor handles each registered action type
    print()
    print("=" * 80)
    print("REGISTERED ACTION TYPES BY EXECUTOR")
    print("=" * 80)

    # Group by executor class
    executor_map = {}
    for action_type in sorted(registered_types):
        executor_class = get_executor_class(action_type)
        executor_name = executor_class.__name__ if executor_class else "Unknown"

        if executor_name not in executor_map:
            executor_map[executor_name] = []
        executor_map[executor_name].append(action_type)

    for executor_name, action_types in sorted(executor_map.items()):
        print(f"\n{executor_name} ({len(action_types)} action types):")
        for action_type in action_types:
            # Check if this is an expected type
            status = "✓" if action_type in expected_action_types else "+"
            print(f"  {status} {action_type}")

    print()
    print("=" * 80)
    print("LEGEND")
    print("=" * 80)
    print("✓ = Expected action type (covered)")
    print("✗ = Expected action type (missing)")
    print("+ = Extra action type (not in expected list)")
    print()

    # Return exit code based on coverage
    if coverage_percent == 100.0:
        print("SUCCESS: All expected action types are registered! 🎉")
        return 0
    else:
        print(f"INCOMPLETE: {len(missing_types)} action types are missing.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
