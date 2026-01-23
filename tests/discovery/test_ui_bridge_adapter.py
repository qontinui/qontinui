"""Tests for UI Bridge adapter state discovery."""

from qontinui.discovery.ui_bridge_adapter import (
    UIBridgeElement,
    discover_states_from_renders,
    extract_elements_from_render,
    get_active_states_for_render,
    get_elements_by_render,
    get_state_elements,
)


class TestExtractElementsFromRender:
    """Tests for extract_elements_from_render function."""

    def test_extracts_registered_elements(self) -> None:
        """Should extract registered elements with reg: prefix (simple format)."""
        render = {
            "id": "render_1",
            "elements": [
                {"id": "nav-menu"},
                {"id": "sidebar"},
            ],
        }

        elements = extract_elements_from_render(render)

        assert "reg:nav-menu" in elements
        assert "reg:sidebar" in elements

    def test_extracts_from_dom_snapshot_format(self) -> None:
        """Should extract elements from DomSnapshotRenderLogEntry format."""
        render = {
            "id": "render_1",
            "type": "dom_snapshot",
            "page_url": "/dashboard",
            "snapshot": {
                "root": {
                    "tag": "div",
                    "attributes": {"data-testid": "dashboard-container"},
                    "children": [
                        {
                            "tag": "nav",
                            "attributes": {"data-ui-id": "main-nav"},
                            "children": [],
                        },
                        {
                            "tag": "aside",
                            "attributes": {"data-testid": "sidebar"},
                            "children": [],
                        },
                    ],
                },
                "total_elements": 3,
            },
        }

        elements = extract_elements_from_render(render)

        assert "testid:dashboard-container" in elements
        assert "ui:main-nav" in elements
        assert "testid:sidebar" in elements

    def test_extracts_html_ids_when_enabled(self) -> None:
        """Should extract HTML id attributes when include_html_ids=True."""
        render = {
            "id": "render_1",
            "type": "dom_snapshot",
            "snapshot": {
                "root": {
                    "tag": "div",
                    "id": "app-root",
                    "attributes": {"data-testid": "container"},
                    "children": [],
                },
            },
        }

        # Without include_html_ids
        elements = extract_elements_from_render(render, include_html_ids=False)
        assert "testid:container" in elements
        assert "html:app-root" not in elements

        # With include_html_ids
        elements = extract_elements_from_render(render, include_html_ids=True)
        assert "testid:container" in elements
        assert "html:app-root" in elements

    def test_extracts_testid_from_component_tree(self) -> None:
        """Should extract data-testid attributes from component tree."""
        render = {
            "id": "render_1",
            "componentTree": {
                "attributes": {"data-testid": "container"},
                "children": [
                    {"attributes": {"data-testid": "header"}},
                    {"attributes": {"data-testid": "footer"}},
                ],
            },
        }

        elements = extract_elements_from_render(render)

        assert "testid:container" in elements
        assert "testid:header" in elements
        assert "testid:footer" in elements

    def test_extracts_testid_from_tree_key(self) -> None:
        """Should also check 'tree' key for component tree."""
        render = {
            "id": "render_1",
            "tree": {
                "attributes": {"data-testid": "main"},
            },
        }

        elements = extract_elements_from_render(render)

        assert "testid:main" in elements

    def test_extracts_testid_from_props(self) -> None:
        """Should extract data-testid from props as well as attributes."""
        render = {
            "id": "render_1",
            "componentTree": {
                "props": {"data-testid": "from-props"},
            },
        }

        elements = extract_elements_from_render(render)

        assert "testid:from-props" in elements

    def test_handles_nested_children(self) -> None:
        """Should recursively extract from deeply nested children."""
        render = {
            "id": "render_1",
            "componentTree": {
                "attributes": {"data-testid": "level-0"},
                "children": [
                    {
                        "attributes": {"data-testid": "level-1"},
                        "children": [
                            {
                                "attributes": {"data-testid": "level-2"},
                                "children": [
                                    {"attributes": {"data-testid": "level-3"}},
                                ],
                            },
                        ],
                    },
                ],
            },
        }

        elements = extract_elements_from_render(render)

        assert "testid:level-0" in elements
        assert "testid:level-1" in elements
        assert "testid:level-2" in elements
        assert "testid:level-3" in elements

    def test_handles_empty_render(self) -> None:
        """Should return empty list for render with no elements."""
        render: dict = {"id": "render_1"}

        elements = extract_elements_from_render(render)

        assert elements == []

    def test_skips_elements_without_id(self) -> None:
        """Should skip elements that don't have an id."""
        render = {
            "id": "render_1",
            "elements": [
                {"id": "valid"},
                {"name": "no-id"},
                {},
            ],
        }

        elements = extract_elements_from_render(render)

        assert "reg:valid" in elements
        assert len([e for e in elements if e.startswith("reg:")]) == 1

    def test_combines_registered_and_testid(self) -> None:
        """Should extract both registered elements and testids."""
        render = {
            "id": "render_1",
            "elements": [{"id": "registered-elem"}],
            "componentTree": {
                "attributes": {"data-testid": "testid-elem"},
            },
        }

        elements = extract_elements_from_render(render)

        assert "reg:registered-elem" in elements
        assert "testid:testid-elem" in elements


class TestDiscoverStatesFromRenders:
    """Tests for discover_states_from_renders function."""

    def test_groups_elements_by_cooccurrence(self) -> None:
        """Elements appearing in same renders should be grouped into a state."""
        renders = [
            {
                "id": "render_1",
                "elements": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            },
            {
                "id": "render_2",
                "elements": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            },
            {
                "id": "render_3",
                "elements": [{"id": "d"}, {"id": "e"}],
            },
        ]

        result = discover_states_from_renders(renders)

        # a, b, c appear in renders 1,2 - should be grouped
        # d, e appear in render 3 - should be grouped
        assert len(result.states) == 2

        state_element_sets = [set(s.state_image_ids) for s in result.states]
        assert {"reg:a", "reg:b", "reg:c"} in state_element_sets
        assert {"reg:d", "reg:e"} in state_element_sets

    def test_example_from_docstring(self) -> None:
        """Test the example from the module docstring."""
        renders = [
            {
                "id": "render_1",
                "elements": [{"id": "nav-menu"}, {"id": "project-list"}],
                "componentTree": {
                    "attributes": {"data-testid": "dashboard-container"},
                    "children": [
                        {"attributes": {"data-testid": "sidebar"}},
                        {"attributes": {"data-testid": "main-content"}},
                    ],
                },
            },
            {
                "id": "render_2",
                "elements": [{"id": "nav-menu"}, {"id": "project-list"}],
                "componentTree": {
                    "attributes": {"data-testid": "dashboard-container"},
                    "children": [
                        {"attributes": {"data-testid": "sidebar"}},
                        {"attributes": {"data-testid": "main-content"}},
                        {"attributes": {"data-testid": "modal-overlay"}},
                    ],
                },
            },
            {
                "id": "render_3",
                "elements": [{"id": "nav-menu"}, {"id": "settings-form"}],
                "componentTree": {
                    "attributes": {"data-testid": "settings-container"},
                    "children": [
                        {"attributes": {"data-testid": "sidebar"}},
                    ],
                },
            },
        ]

        result = discover_states_from_renders(renders)

        # nav-menu and sidebar appear in all 3 renders
        # project-list, dashboard-container, main-content appear in renders 1,2
        # modal-overlay appears only in render 2
        # settings-form, settings-container appear only in render 3

        assert result.render_count == 3
        assert result.unique_element_count > 0

        # Find the state containing nav-menu and sidebar (all renders)
        nav_sidebar_state = None
        for state in result.states:
            if "reg:nav-menu" in state.state_image_ids:
                nav_sidebar_state = state
                break

        assert nav_sidebar_state is not None
        assert "testid:sidebar" in nav_sidebar_state.state_image_ids
        assert set(nav_sidebar_state.screenshot_ids) == {
            "render_1",
            "render_2",
            "render_3",
        }

    def test_single_element_per_render(self) -> None:
        """Elements appearing in only one render each form singleton states."""
        renders = [
            {"id": "r1", "elements": [{"id": "a"}]},
            {"id": "r2", "elements": [{"id": "b"}]},
            {"id": "r3", "elements": [{"id": "c"}]},
        ]

        result = discover_states_from_renders(renders)

        # Each element appears in different renders, so 3 singleton states
        assert len(result.states) == 3
        for state in result.states:
            assert len(state.state_image_ids) == 1

    def test_empty_renders_list(self) -> None:
        """Should handle empty renders list gracefully."""
        result = discover_states_from_renders([])

        assert result.states == []
        assert result.elements == []
        assert result.render_count == 0
        assert result.unique_element_count == 0

    def test_renders_with_no_elements(self) -> None:
        """Should handle renders that have no extractable elements."""
        renders = [
            {"id": "r1"},
            {"id": "r2", "elements": []},
        ]

        result = discover_states_from_renders(renders)

        assert result.states == []
        assert result.render_count == 2
        assert result.unique_element_count == 0

    def test_result_includes_element_details(self) -> None:
        """Result should include detailed element information."""
        renders = [
            {"id": "r1", "elements": [{"id": "elem1"}]},
        ]

        result = discover_states_from_renders(renders)

        assert len(result.elements) == 1
        elem = result.elements[0]
        assert elem.id == "reg:elem1"
        assert elem.name == "elem1"
        assert elem.type == "registered"
        assert elem.render_ids == ["r1"]

    def test_dom_snapshot_format_state_discovery(self) -> None:
        """Should discover states from DomSnapshotRenderLogEntry format."""
        renders = [
            {
                "id": "r1",
                "type": "dom_snapshot",
                "page_url": "/dashboard",
                "snapshot": {
                    "root": {
                        "tag": "div",
                        "attributes": {"data-testid": "app"},
                        "children": [
                            {"tag": "nav", "attributes": {"data-ui-id": "nav"}, "children": []},
                            {
                                "tag": "main",
                                "attributes": {"data-testid": "dashboard"},
                                "children": [],
                            },
                        ],
                    },
                },
            },
            {
                "id": "r2",
                "type": "dom_snapshot",
                "page_url": "/dashboard",
                "snapshot": {
                    "root": {
                        "tag": "div",
                        "attributes": {"data-testid": "app"},
                        "children": [
                            {"tag": "nav", "attributes": {"data-ui-id": "nav"}, "children": []},
                            {
                                "tag": "main",
                                "attributes": {"data-testid": "dashboard"},
                                "children": [],
                            },
                            {"tag": "div", "attributes": {"data-testid": "modal"}, "children": []},
                        ],
                    },
                },
            },
            {
                "id": "r3",
                "type": "dom_snapshot",
                "page_url": "/settings",
                "snapshot": {
                    "root": {
                        "tag": "div",
                        "attributes": {"data-testid": "app"},
                        "children": [
                            {"tag": "nav", "attributes": {"data-ui-id": "nav"}, "children": []},
                            {
                                "tag": "main",
                                "attributes": {"data-testid": "settings"},
                                "children": [],
                            },
                        ],
                    },
                },
            },
        ]

        result = discover_states_from_renders(renders)

        # app and nav appear in all 3 renders - should be grouped
        # dashboard appears in r1, r2 - should be grouped
        # modal appears only in r2 - singleton
        # settings appears only in r3 - singleton

        assert result.render_count == 3
        assert result.unique_element_count == 5

        # Find the state containing nav (all renders)
        nav_state = None
        for state in result.states:
            if "ui:nav" in state.state_image_ids:
                nav_state = state
                break

        assert nav_state is not None
        assert "testid:app" in nav_state.state_image_ids
        assert set(nav_state.screenshot_ids) == {"r1", "r2", "r3"}

    def test_element_to_renders_mapping(self) -> None:
        """Result should include element to renders mapping."""
        renders = [
            {"id": "r1", "elements": [{"id": "a"}, {"id": "b"}]},
            {"id": "r2", "elements": [{"id": "a"}]},
        ]

        result = discover_states_from_renders(renders)

        assert result.element_to_renders["reg:a"] == ["r1", "r2"]
        assert result.element_to_renders["reg:b"] == ["r1"]


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_state_elements(self) -> None:
        """Should return elements belonging to a state."""
        renders = [
            {"id": "r1", "elements": [{"id": "a"}, {"id": "b"}]},
            {"id": "r2", "elements": [{"id": "a"}, {"id": "b"}]},
        ]

        result = discover_states_from_renders(renders)

        # There should be one state with a and b
        state = result.states[0]
        state_elements = get_state_elements(state, result.elements)

        assert len(state_elements) == 2
        element_ids = {e.id for e in state_elements}
        assert element_ids == {"reg:a", "reg:b"}

    def test_get_elements_by_render(self) -> None:
        """Should return elements present in a specific render."""
        renders = [
            {"id": "r1", "elements": [{"id": "a"}, {"id": "b"}]},
            {"id": "r2", "elements": [{"id": "a"}, {"id": "c"}]},
        ]

        result = discover_states_from_renders(renders)

        r1_elements = get_elements_by_render("r1", result.element_to_renders, result.elements)
        r2_elements = get_elements_by_render("r2", result.element_to_renders, result.elements)

        r1_ids = {e.id for e in r1_elements}
        r2_ids = {e.id for e in r2_elements}

        assert r1_ids == {"reg:a", "reg:b"}
        assert r2_ids == {"reg:a", "reg:c"}

    def test_get_active_states_for_render(self) -> None:
        """Should return states active in a specific render."""
        renders = [
            {"id": "r1", "elements": [{"id": "a"}]},
            {"id": "r2", "elements": [{"id": "b"}]},
            {"id": "r3", "elements": [{"id": "a"}, {"id": "b"}]},
        ]

        result = discover_states_from_renders(renders)

        r1_states = get_active_states_for_render("r1", result.states)
        r2_states = get_active_states_for_render("r2", result.states)
        r3_states = get_active_states_for_render("r3", result.states)

        # r1 has only element a
        # r2 has only element b
        # r3 has both a and b
        # Since a appears in r1,r3 and b appears in r2,r3, they form different states

        assert len(r3_states) == 2  # Both states active in r3
        assert len(r1_states) == 1  # Only state with 'a'
        assert len(r2_states) == 1  # Only state with 'b'


class TestSerialization:
    """Tests for JSON serialization."""

    def test_result_to_dict(self) -> None:
        """Result should serialize to dict properly."""
        renders = [
            {"id": "r1", "elements": [{"id": "a"}]},
        ]

        result = discover_states_from_renders(renders)
        result_dict = result.to_dict()

        assert "states" in result_dict
        assert "elements" in result_dict
        assert "elementToRenders" in result_dict
        assert result_dict["renderCount"] == 1
        assert result_dict["uniqueElementCount"] == 1

    def test_element_to_dict(self) -> None:
        """UIBridgeElement should serialize properly."""
        elem = UIBridgeElement(
            id="reg:test",
            name="test",
            type="registered",
            render_ids=["r1", "r2"],
            tag_name="button",
            text_content="Click me",
        )

        elem_dict = elem.to_dict()

        assert elem_dict["id"] == "reg:test"
        assert elem_dict["name"] == "test"
        assert elem_dict["type"] == "registered"
        assert elem_dict["renderIds"] == ["r1", "r2"]
        assert elem_dict["tagName"] == "button"
        assert elem_dict["textContent"] == "Click me"
