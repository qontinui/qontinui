"""
Test for co-occurrence clustering algorithm.

This tests the core logic that:
- States are collections of elements that appear together across screens
- Elements with the same visibility pattern form a cluster (state)
"""

import numpy as np


def cluster_elements_by_cooccurrence(
    screens: dict[str, set[str]],
) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    """
    Cluster elements by their co-occurrence pattern across screens.

    Args:
        screens: Mapping of screen_id -> set of element_ids visible on that screen

    Returns:
        Tuple of:
        - states: Mapping of state_id -> set of element_ids in that state
        - state_screens: Mapping of state_id -> list of screen_ids where state is active
    """
    # Get all unique elements and screens
    all_elements = sorted({e for elems in screens.values() for e in elems})
    screen_ids = sorted(screens.keys())

    if not all_elements or not screen_ids:
        return {}, {}

    # Build visibility matrix: rows = screens, columns = elements
    elem_to_idx = {e: i for i, e in enumerate(all_elements)}
    visibility_matrix = np.zeros((len(screen_ids), len(all_elements)))

    for screen_idx, screen_id in enumerate(screen_ids):
        for elem in screens[screen_id]:
            visibility_matrix[screen_idx, elem_to_idx[elem]] = 1.0

    # Group elements by their visibility pattern
    # Elements with identical patterns form a cluster (state)
    pattern_to_elements: dict[tuple, list[str]] = {}
    for elem_idx, elem_id in enumerate(all_elements):
        pattern = tuple(visibility_matrix[:, elem_idx])
        if pattern not in pattern_to_elements:
            pattern_to_elements[pattern] = []
        pattern_to_elements[pattern].append(elem_id)

    # Create states from clusters
    states: dict[str, set[str]] = {}
    state_screens: dict[str, list[str]] = {}
    state_idx = 0

    for pattern, cluster_elements in pattern_to_elements.items():
        if sum(pattern) == 0:
            continue  # Skip elements that don't appear on any screen

        state_idx += 1
        state_id = f"state{state_idx}"

        states[state_id] = set(cluster_elements)

        # Determine which screens this state is active on
        active_screens = [screen_ids[i] for i, visible in enumerate(pattern) if visible > 0]
        state_screens[state_id] = active_screens

    return states, state_screens


def get_active_states_for_screen(
    screen_id: str,
    state_screens: dict[str, list[str]],
) -> set[str]:
    """Get all states that are active on a given screen."""
    return {state_id for state_id, screens in state_screens.items() if screen_id in screens}


class TestCooccurrenceClustering:
    """Test the co-occurrence clustering algorithm."""

    def test_basic_clustering(self):
        """
        Test the example from the user:

        Screen  Images   States
        1       a,b,c,d  state1, state2
        2       a,b,c,d  state1, state2
        3       a,b,e    state1, state3

        Expected:
        States  Images  Screens
        state1  a,b     1,2,3
        state2  c,d     1,2
        state3  e       3
        """
        # Input: which elements appear on which screen
        screens = {
            "screen1": {"a", "b", "c", "d"},
            "screen2": {"a", "b", "c", "d"},
            "screen3": {"a", "b", "e"},
        }

        # Run clustering
        states, state_screens = cluster_elements_by_cooccurrence(screens)

        # Verify we got 3 states
        assert len(states) == 3, f"Expected 3 states, got {len(states)}: {states}"

        # Find which state has which elements
        state_with_ab = None
        state_with_cd = None
        state_with_e = None

        for state_id, elements in states.items():
            if elements == {"a", "b"}:
                state_with_ab = state_id
            elif elements == {"c", "d"}:
                state_with_cd = state_id
            elif elements == {"e"}:
                state_with_e = state_id

        # Verify state compositions
        assert state_with_ab is not None, f"No state with {{a, b}} found. States: {states}"
        assert state_with_cd is not None, f"No state with {{c, d}} found. States: {states}"
        assert state_with_e is not None, f"No state with {{e}} found. States: {states}"

        # Verify which screens each state is active on
        assert set(state_screens[state_with_ab]) == {
            "screen1",
            "screen2",
            "screen3",
        }, f"State with {{a,b}} should be active on all screens, got {state_screens[state_with_ab]}"

        assert set(state_screens[state_with_cd]) == {
            "screen1",
            "screen2",
        }, f"State with {{c,d}} should be active on screens 1,2, got {state_screens[state_with_cd]}"

        assert set(state_screens[state_with_e]) == {
            "screen3"
        }, f"State with {{e}} should be active only on screen 3, got {state_screens[state_with_e]}"

        # Verify active states per screen
        screen1_states = get_active_states_for_screen("screen1", state_screens)
        screen2_states = get_active_states_for_screen("screen2", state_screens)
        screen3_states = get_active_states_for_screen("screen3", state_screens)

        assert screen1_states == {
            state_with_ab,
            state_with_cd,
        }, f"Screen 1 should have state1 and state2, got {screen1_states}"

        assert screen2_states == {
            state_with_ab,
            state_with_cd,
        }, f"Screen 2 should have state1 and state2, got {screen2_states}"

        assert screen3_states == {
            state_with_ab,
            state_with_e,
        }, f"Screen 3 should have state1 and state3, got {screen3_states}"

        print("\n" + "=" * 60)
        print("CO-OCCURRENCE CLUSTERING TEST RESULTS")
        print("=" * 60)
        print("\nInput (Screens -> Elements):")
        for screen_id, elements in sorted(screens.items()):
            print(f"  {screen_id}: {{{', '.join(sorted(elements))}}}")

        print("\nOutput (States -> Elements):")
        for state_id in sorted(states.keys()):
            elements = states[state_id]
            active_on = state_screens[state_id]
            print(f"  {state_id}: {{{', '.join(sorted(elements))}}} - active on {active_on}")

        print("\nVerification (Screens -> Active States):")
        for screen_id in sorted(screens.keys()):
            active = get_active_states_for_screen(screen_id, state_screens)
            print(f"  {screen_id}: {{{', '.join(sorted(active))}}}")

        print("\n" + "=" * 60)
        print("ALL ASSERTIONS PASSED!")
        print("=" * 60)

    def test_single_element_states(self):
        """Test that single elements that appear alone form their own state."""
        screens = {
            "screen1": {"a"},
            "screen2": {"b"},
            "screen3": {"a", "b"},
        }

        states, state_screens = cluster_elements_by_cooccurrence(screens)

        # a appears on screens 1,3; b appears on screens 2,3
        # They have different patterns, so they should be different states
        assert len(states) == 2

        state_with_a = None
        state_with_b = None
        for state_id, elements in states.items():
            if elements == {"a"}:
                state_with_a = state_id
            elif elements == {"b"}:
                state_with_b = state_id

        assert state_with_a is not None
        assert state_with_b is not None
        assert set(state_screens[state_with_a]) == {"screen1", "screen3"}
        assert set(state_screens[state_with_b]) == {"screen2", "screen3"}

    def test_all_elements_together(self):
        """Test that elements always appearing together form one state."""
        screens = {
            "screen1": {"a", "b", "c"},
            "screen2": {"a", "b", "c"},
            "screen3": {"a", "b", "c"},
        }

        states, state_screens = cluster_elements_by_cooccurrence(screens)

        # All elements appear on all screens, so they form one state
        assert len(states) == 1
        state_id = list(states.keys())[0]
        assert states[state_id] == {"a", "b", "c"}
        assert set(state_screens[state_id]) == {"screen1", "screen2", "screen3"}

    def test_empty_screens(self):
        """Test handling of empty input."""
        states, state_screens = cluster_elements_by_cooccurrence({})
        assert states == {}
        assert state_screens == {}

    def test_complex_pattern(self):
        """Test a more complex visibility pattern."""
        screens = {
            "s1": {"header", "nav", "content", "footer"},
            "s2": {"header", "nav", "content", "footer"},
            "s3": {"header", "nav", "modal"},
            "s4": {"header", "nav", "sidebar", "content"},
        }

        states, state_screens = cluster_elements_by_cooccurrence(screens)

        # Expected clusters:
        # {header, nav} - appears on s1, s2, s3, s4 (always together)
        # {content} - appears on s1, s2, s4 (different from header/nav)
        # {footer} - appears on s1, s2
        # {modal} - appears on s3
        # {sidebar} - appears on s4

        # Actually, header and nav should form state1 only if they ALWAYS appear together
        # and content appears on same screens minus s3
        # Let's verify:

        # header: s1, s2, s3, s4
        # nav: s1, s2, s3, s4
        # content: s1, s2, s4
        # footer: s1, s2
        # modal: s3
        # sidebar: s4

        # header and nav have same pattern -> one state
        # content has different pattern -> separate state
        # footer has different pattern -> separate state
        # modal has different pattern -> separate state
        # sidebar has different pattern -> separate state

        assert len(states) == 5, f"Expected 5 states, got {len(states)}: {states}"

        # Find the state with header and nav
        header_nav_state = None
        for state_id, elements in states.items():
            if "header" in elements and "nav" in elements:
                header_nav_state = state_id
                assert elements == {
                    "header",
                    "nav",
                }, f"Expected header+nav state to have only header and nav, got {elements}"
                break

        assert header_nav_state is not None
        assert set(state_screens[header_nav_state]) == {"s1", "s2", "s3", "s4"}

        print("\nComplex pattern test passed!")
        print(f"States: {states}")


if __name__ == "__main__":
    # Run tests directly
    test = TestCooccurrenceClustering()
    test.test_basic_clustering()
    test.test_single_element_states()
    test.test_all_elements_together()
    test.test_empty_screens()
    test.test_complex_pattern()
    print("\nAll tests passed!")
