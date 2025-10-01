"""Tests for DSL parser."""

import pytest

from qontinui.dsl.parser import EXAMPLE_DSL, QontinuiDSLParser


class TestQontinuiDSLParser:
    """Test QontinuiDSLParser class."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return QontinuiDSLParser()

    def test_parse_state_definition(self, parser):
        """Test parsing state definitions."""
        script = """
        state LoginPage {
            elements: [
                {id: "username", type: input},
                {id: "password", type: input}
            ]
            min_elements: 2
        }
        """

        result = parser.parse(script)
        assert result is not None
        assert "states" in result
        assert "LoginPage" in result["states"]

        state = result["states"]["LoginPage"]
        assert state.name == "LoginPage"
        assert state.min_elements == 2
        assert len(state.elements) == 2

    def test_parse_transition_definition(self, parser):
        """Test parsing transition definitions."""
        script = """
        transition login_transition {
            from: LoginPage
            to: HomePage
            action: click
            trigger: login_button
            probability: 0.95
        }
        """

        result = parser.parse(script)
        assert result is not None
        assert "transitions" in result
        assert "login_transition" in result["transitions"]

        transition = result["transitions"]["login_transition"]
        assert transition.name == "login_transition"
        assert transition.from_state == "LoginPage"
        assert transition.to_state == "HomePage"
        assert transition.action == "click"
        assert transition.trigger == "login_button"
        assert transition.probability == 0.95

    def test_parse_action_commands(self, parser):
        """Test parsing action commands."""
        script = """
        click(element=button1);
        type(text="hello world");
        wait(duration=2);
        """

        result = parser.parse(script)
        assert result is not None
        assert "actions" in result
        assert len(result["actions"]) == 3

        actions = result["actions"]
        assert actions[0].action_type == "click"
        assert actions[1].action_type == "type"
        assert actions[2].action_type == "wait"

    def test_parse_element_definition(self, parser):
        """Test parsing element definitions."""
        script = """
        element submit_button {
            id: "submit",
            type: button,
            text: "Submit",
            bbox: [100, 200, 150, 50]
        };
        """

        result = parser.parse(script)
        assert result is not None
        assert "elements" in result
        assert "submit_button" in result["elements"]

        element = result["elements"]["submit_button"]
        assert element["id"] == "submit"
        assert element["type"] == "button"
        assert element["text"] == "Submit"
        assert element["bbox"] == [100, 200, 150, 50]

    def test_parse_variable_definition(self, parser):
        """Test parsing variable definitions."""
        script = """
        var username = "test_user";
        var count = 5;
        var enabled = true;
        """

        result = parser.parse(script)
        assert result is not None
        assert "variables" in result
        assert result["variables"]["username"] == "test_user"
        assert result["variables"]["count"] == 5
        assert result["variables"]["enabled"]

    def test_parse_loop_constructs(self, parser):
        """Test parsing loop constructs."""
        script = """
        repeat 3 {
            click(element=button);
        }

        while count > 0 {
            type(text="test");
        }

        for item in items {
            click(element=item);
        }
        """

        # Basic parsing test - full loop implementation would require more complex grammar
        result = parser.parse(script)
        assert result is not None

    def test_parse_conditional_constructs(self, parser):
        """Test parsing conditional constructs."""
        script = """
        if state == LoginPage {
            click(element=login_button);
        } else {
            click(element=logout_button);
        }
        """

        # Basic parsing test - full conditional implementation would require more complex grammar
        result = parser.parse(script)
        assert result is not None

    def test_parse_assertions(self, parser):
        """Test parsing assertions."""
        script = """
        assert current_state == HomePage;
        assert element_visible == true;
        """

        result = parser.parse(script)
        assert result is not None

    def test_parse_comments(self, parser):
        """Test parsing with comments."""
        script = """
        // This is a line comment
        state TestState {
            elements: []  // Empty elements
            min_elements: 0
        }

        /* This is a
           multi-line comment */
        click(element=button);
        """

        result = parser.parse(script)
        assert result is not None
        assert "TestState" in result["states"]

    def test_parse_example_dsl(self, parser):
        """Test parsing the example DSL script."""
        result = parser.parse(EXAMPLE_DSL)

        assert result is not None
        assert "states" in result
        assert "transitions" in result
        assert "actions" in result
        assert "variables" in result

        # Check states were parsed
        assert "LoginPage" in result["states"]
        assert "HomePage" in result["states"]

        # Check transition was parsed
        assert "login_to_home" in result["transitions"]

        # Check variables were parsed
        assert result["variables"]["username"] == "test_user"
        assert result["variables"]["password"] == "test_pass"

    def test_validate_valid_script(self, parser):
        """Test validation of valid script."""
        script = """
        state ValidState {
            elements: []
            min_elements: 0
        }
        """

        assert parser.validate(script)

    def test_validate_invalid_script(self, parser):
        """Test validation of invalid script."""
        script = """
        state InvalidState {
            elements: [  // Missing closing bracket
            min_elements: 0
        }
        """

        assert not parser.validate(script)

    def test_to_python_generation(self, parser):
        """Test Python code generation from parsed DSL."""
        script = """
        state TestState {
            elements: []
            min_elements: 1
        }

        transition test_transition {
            from: TestState
            to: TestState
            action: click
            probability: 0.9
        }
        """

        parsed = parser.parse(script)
        python_code = parser.to_python(parsed)

        assert isinstance(python_code, str)
        assert "from qontinui import" in python_code
        assert "TestState_state = State(" in python_code
        assert "test_transition_transition = Transition(" in python_code
        assert 'name="TestState"' in python_code
        assert "min_elements=1" in python_code
        assert "probability=0.9" in python_code
