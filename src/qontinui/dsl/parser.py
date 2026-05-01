"""DSL parser for Qontinui automation scripts."""

import logging
from dataclasses import dataclass
from typing import Any, cast

from lark import Lark, Token, Transformer
from lark.exceptions import LarkError

logger = logging.getLogger(__name__)


# Qontinui DSL Grammar
QONTINUI_GRAMMAR = r"""
    ?start: program

    program: statement+

    ?statement: state_def
              | transition_def
              | action_def
              | element_def
              | assertion
              | loop
              | conditional
              | variable_def

    state_def: "state" IDENTIFIER "{" state_body "}"
    state_body: state_property*
    state_property: elements_prop
                  | min_elements_prop
                  | parent_prop
                  | metadata_prop

    elements_prop: "elements" ":" element_list
    min_elements_prop: "min_elements" ":" NUMBER
    parent_prop: "parent" ":" IDENTIFIER
    metadata_prop: "metadata" ":" json_value

    element_list: "[" [element ("," element)*] "]"
    element: IDENTIFIER | element_inline

    element_inline: "{" element_props "}"
    element_props: element_prop ("," element_prop)*
    element_prop: id_prop
                | type_prop
                | bbox_prop
                | text_prop
                | description_prop

    id_prop: "id" ":" STRING
    type_prop: "type" ":" element_type
    bbox_prop: "bbox" ":" bbox
    text_prop: "text" ":" STRING
    description_prop: "description" ":" STRING

    element_type: ELEMENT_TYPE_TOKEN

    ELEMENT_TYPE_TOKEN: "button" | "text" | "input" | "image" | "icon" | "link" | "checkbox"

    bbox: "[" NUMBER "," NUMBER "," NUMBER "," NUMBER "]"

    transition_def: "transition" IDENTIFIER "{" transition_body "}"
    transition_body: transition_property*
    transition_property: from_prop
                       | to_prop
                       | action_prop
                       | trigger_prop
                       | probability_prop
                       | conditions_prop

    from_prop: "from" ":" IDENTIFIER
    to_prop: "to" ":" IDENTIFIER
    action_prop: "action" ":" action_type
    trigger_prop: "trigger" ":" IDENTIFIER
    probability_prop: "probability" ":" NUMBER
    conditions_prop: "conditions" ":" condition_list

    action_type: ACTION_TYPE_TOKEN

    ACTION_TYPE_TOKEN: "click" | "type" | "hover" | "drag" | "scroll" | "key_press" | "wait"

    condition_list: "[" [condition ("," condition)*] "]"
    condition: expression

    action_def: action_type "(" [action_args] ")" ";"
    action_args: action_arg ("," action_arg)*
    action_arg: IDENTIFIER "=" value
              | value

    element_def: "element" IDENTIFIER element_inline ";"

    assertion: "assert" expression ";"

    loop: "repeat" NUMBER "{" statement* "}"
        | "while" expression "{" statement* "}"
        | "for" IDENTIFIER "in" iterable "{" statement* "}"

    conditional: "if" expression "{" statement* "}" ["else" "{" statement* "}"]

    variable_def: "var" IDENTIFIER "=" value ";"

    expression: comparison
    comparison: term (COMP_OP term)*
    term: factor (ADD_OP factor)*
    factor: atom (MUL_OP atom)*
    atom: NUMBER
        | STRING
        | IDENTIFIER
        | BOOLEAN
        | "(" expression ")"
        | function_call

    BOOLEAN.2: "true" | "false"

    function_call: IDENTIFIER "(" [args] ")"
    args: value ("," value)*

    iterable: IDENTIFIER
            | "[" [value ("," value)*] "]"
            | function_call

    value: NUMBER
         | STRING
         | IDENTIFIER
         | json_value
         | BOOLEAN
         | "null"

    json_value: json_object | json_array
    json_object: "{" [json_pair ("," json_pair)*] "}"
    json_pair: STRING ":" json_element
    json_array: "[" [json_element ("," json_element)*] "]"
    json_element: STRING | NUMBER | "true" | "false" | "null" | json_object | json_array

    COMMENT: "//" /[^\n]*/
           | "/*" /(.|\n)*?/ "*/"

    COMP_OP: "==" | "!=" | "<" | ">" | "<=" | ">="
    ADD_OP: "+" | "-"
    MUL_OP: "*" | "/" | "%"

    %import common.NUMBER
    %import common.ESCAPED_STRING -> STRING
    %import common.WS
    %import common.NEWLINE
    %import common.CNAME -> IDENTIFIER
    %ignore WS
    %ignore COMMENT
"""


@dataclass
class ParsedState:
    """Parsed state definition."""

    name: str
    elements: list[dict[str, Any]]
    min_elements: int
    parent: str | None
    metadata: dict[str, Any]


@dataclass
class ParsedTransition:
    """Parsed transition definition."""

    name: str
    from_state: str | None
    to_state: str | None
    action: str
    trigger: str | None
    probability: float
    conditions: list[str]


@dataclass
class ParsedAction:
    """Parsed action command."""

    action_type: str
    arguments: dict[str, Any]


class QontinuiTransformer(Transformer[Any, Any]):
    """Transform parsed DSL into Python objects."""

    def __init__(self) -> None:
        super().__init__()
        self.states: dict[str, ParsedState] = {}
        self.transitions: dict[str, ParsedTransition] = {}
        self.elements: dict[str, dict[str, Any]] = {}
        self.variables: dict[str, Any] = {}
        self.actions: list[ParsedAction] = []

    def program(self, items: list[Any]) -> dict[str, Any]:
        """Process program items and return all parsed entities.

        Args:
            items: List of parsed items from the DSL program

        Returns:
            Dictionary containing states, transitions, elements, variables, and actions
        """
        # Process all items to populate dictionaries
        for _item in items:
            pass  # Items are processed as side effects

        return {
            "states": self.states,
            "transitions": self.transitions,
            "elements": self.elements,
            "variables": self.variables,
            "actions": self.actions,
        }

    def state_def(self, items: list[Any]) -> ParsedState:
        """Parse state definition.

        Args:
            items: List containing state name and properties

        Returns:
            ParsedState object
        """
        name = str(items[0])
        properties: dict[str, Any] = {}

        # Collect properties from state_body
        if len(items) > 1 and isinstance(items[1], dict):
            properties = items[1]

        state = ParsedState(
            name=name,
            elements=properties.get("elements", []),
            min_elements=properties.get("min_elements", 1),
            parent=properties.get("parent"),
            metadata=properties.get("metadata", {}),
        )

        self.states[name] = state
        return state

    def state_property(self, items: list[Any]) -> Any:
        """Pass through the property tuple from child rules.

        Args:
            items: List of property items

        Returns:
            First item or None
        """
        return items[0] if items else None

    def state_body(self, items: list[Any]) -> dict[str, Any]:
        """Parse state body and collect properties.

        Args:
            items: List of state properties

        Returns:
            Dictionary of state properties
        """
        properties: dict[str, Any] = {}
        for item in items:
            if isinstance(item, tuple) and len(item) == 2:
                properties[item[0]] = item[1]
            elif isinstance(item, dict):
                properties.update(item)
        return properties

    def elements_prop(self, items: list[Any]) -> tuple[str, list[Any]]:
        """Handle elements property.

        Args:
            items: List containing elements

        Returns:
            Tuple of property name and element list
        """
        return ("elements", items[0] if items else [])

    def min_elements_prop(self, items: list[Any]) -> tuple[str, int]:
        """Handle min_elements property.

        Args:
            items: List containing min_elements value

        Returns:
            Tuple of property name and integer value
        """
        return ("min_elements", int(items[0]) if items else 1)

    def parent_prop(self, items: list[Any]) -> tuple[str, str | None]:
        """Handle parent property.

        Args:
            items: List containing parent state name

        Returns:
            Tuple of property name and parent state name
        """
        return ("parent", str(items[0]) if items else None)

    def metadata_prop(self, items: list[Any]) -> tuple[str, dict[str, Any]]:
        """Handle metadata property.

        Args:
            items: List containing metadata dictionary

        Returns:
            Tuple of property name and metadata dictionary
        """
        return ("metadata", items[0] if items else {})

    def element(self, items: list[Any]) -> Any:
        """Handle element - can be IDENTIFIER or element_inline.

        Args:
            items: List containing element identifier or inline definition

        Returns:
            Element identifier or inline definition
        """
        return items[0] if items else None

    def element_list(self, items: list[Any]) -> list[Any]:
        """Return list of elements.

        Args:
            items: List of elements

        Returns:
            List of elements
        """
        return items

    def element_inline(self, items: list[Any]) -> dict[str, Any]:
        """Handle inline element definition.

        Args:
            items: List containing element properties dictionary

        Returns:
            Element properties dictionary
        """
        # items[0] should be the element_props dict
        if items:
            result = items[0]
            return result if isinstance(result, dict) else {}
        return {}

    def element_type(self, items: list[Any]) -> str:
        """Handle element type - returns the element type string.

        Args:
            items: List containing element type token

        Returns:
            Element type string (defaults to 'button')
        """
        if items:
            return str(items[0])
        return "button"  # default element type

    def id_prop(self, items: list[Any]) -> dict[str, str]:
        """Handle id property.

        Args:
            items: List containing element id

        Returns:
            Dictionary with id property
        """
        return {"id": items[0] if items else ""}

    def type_prop(self, items: list[Any]) -> dict[str, str]:
        """Handle type property.

        Args:
            items: List containing element type

        Returns:
            Dictionary with type property
        """
        return {"type": items[0] if items else "button"}

    def bbox_prop(self, items: list[Any]) -> dict[str, list[int]]:
        """Handle bbox property.

        Args:
            items: List containing bounding box coordinates

        Returns:
            Dictionary with bbox property
        """
        return {"bbox": items[0] if items else [0, 0, 0, 0]}

    def text_prop(self, items: list[Any]) -> dict[str, str]:
        """Handle text property.

        Args:
            items: List containing text value

        Returns:
            Dictionary with text property
        """
        return {"text": items[0] if items else ""}

    def description_prop(self, items: list[Any]) -> dict[str, str]:
        """Handle description property.

        Args:
            items: List containing description value

        Returns:
            Dictionary with description property
        """
        return {"description": items[0] if items else ""}

    def element_prop(self, items: list[Any]) -> dict[str, Any]:
        """Pass through the property dict from child rules.

        Args:
            items: List containing element property dictionary

        Returns:
            Element property dictionary
        """
        if items:
            result = items[0]
            return result if isinstance(result, dict) else {}
        return {}

    def element_props(self, items: list[Any]) -> dict[str, Any]:
        """Combine all element properties into a single dict.

        Args:
            items: List of element property dictionaries

        Returns:
            Combined element properties dictionary
        """
        props: dict[str, Any] = {}
        for item in items:
            if isinstance(item, dict):
                props.update(item)
        return props

    def bbox(self, items: list[Any]) -> list[int]:
        """Parse bounding box coordinates.

        Args:
            items: List of coordinate values

        Returns:
            List of integer coordinates [x, y, width, height]
        """
        return [int(item) for item in items]

    def transition_def(self, items: list[Any]) -> ParsedTransition:
        """Parse transition definition.

        Args:
            items: List containing transition name and properties

        Returns:
            ParsedTransition object
        """
        name = str(items[0])
        properties: dict[str, Any] = {}

        if len(items) > 1 and isinstance(items[1], dict):
            properties = items[1]

        transition = ParsedTransition(
            name=name,
            from_state=properties.get("from"),
            to_state=properties.get("to"),
            action=properties.get("action", "click"),
            trigger=properties.get("trigger"),
            probability=properties.get("probability", 1.0),
            conditions=properties.get("conditions", []),
        )

        self.transitions[name] = transition
        return transition

    def transition_property(self, items: list[Any]) -> Any:
        """Pass through the property tuple from child rules.

        Args:
            items: List of property items

        Returns:
            First item or None
        """
        return items[0] if items else None

    def transition_body(self, items: list[Any]) -> dict[str, Any]:
        """Parse transition body and collect properties.

        Args:
            items: List of transition properties

        Returns:
            Dictionary of transition properties
        """
        properties: dict[str, Any] = {}
        for item in items:
            if isinstance(item, tuple) and len(item) == 2:
                properties[item[0]] = item[1]
            elif isinstance(item, dict):
                properties.update(item)
        return properties

    def from_prop(self, items: list[Any]) -> tuple[str, str | None]:
        """Handle from property.

        Args:
            items: List containing from state name

        Returns:
            Tuple of property name and from state name
        """
        return ("from", str(items[0]) if items else None)

    def to_prop(self, items: list[Any]) -> tuple[str, str | None]:
        """Handle to property.

        Args:
            items: List containing to state name

        Returns:
            Tuple of property name and to state name
        """
        return ("to", str(items[0]) if items else None)

    def action_prop(self, items: list[Any]) -> tuple[str, str]:
        """Handle action property.

        Args:
            items: List containing action type

        Returns:
            Tuple of property name and action type
        """
        return ("action", str(items[0]) if items else "click")

    def trigger_prop(self, items: list[Any]) -> tuple[str, str | None]:
        """Handle trigger property.

        Args:
            items: List containing trigger element name

        Returns:
            Tuple of property name and trigger element name
        """
        return ("trigger", str(items[0]) if items else None)

    def probability_prop(self, items: list[Any]) -> tuple[str, float]:
        """Handle probability property.

        Args:
            items: List containing probability value

        Returns:
            Tuple of property name and probability value
        """
        return ("probability", float(items[0]) if items else 1.0)

    def conditions_prop(self, items: list[Any]) -> tuple[str, list[str]]:
        """Handle conditions property.

        Args:
            items: List containing conditions

        Returns:
            Tuple of property name and conditions list
        """
        return ("conditions", items[0] if items else [])

    def action_def(self, items: list[Any]) -> ParsedAction:
        """Parse action definition.

        Args:
            items: List containing action type and arguments

        Returns:
            ParsedAction object
        """
        action_type = str(items[0])
        args: dict[str, Any] = {}

        if len(items) > 1 and isinstance(items[1], dict):
            args = items[1]

        action = ParsedAction(action_type=action_type, arguments=args)

        self.actions.append(action)
        return action

    def action_type(self, items: list[Any]) -> str:
        """Handle action type - returns the action type string.

        Args:
            items: List containing action type token

        Returns:
            Action type string (defaults to 'click')
        """
        if items:
            return str(items[0])
        return "click"  # default action type

    def action_args(self, items: list[Any]) -> dict[str, Any]:
        """Parse action arguments.

        Args:
            items: List of action arguments

        Returns:
            Dictionary of action arguments
        """
        args: dict[str, Any] = {}
        for item in items:
            if isinstance(item, dict):
                args.update(item)
            elif isinstance(item, tuple):
                args[item[0]] = item[1]
        return args

    def element_def(self, items: list[Any]) -> dict[str, dict[str, Any]]:
        """Parse element definition.

        Args:
            items: List containing element name and properties

        Returns:
            Dictionary mapping element name to properties
        """
        name = str(items[0])
        props: dict[str, Any] = items[1] if len(items) > 1 else {}
        self.elements[name] = props
        return {name: props}

    def variable_def(self, items: list[Any]) -> dict[str, Any]:
        """Parse variable definition.

        Args:
            items: List containing variable name and value

        Returns:
            Dictionary mapping variable name to value
        """
        name = str(items[0])
        value: Any = items[1] if len(items) > 1 else None

        # Extract actual value if it's a Tree
        if hasattr(value, "data"):
            if value.data == "value":
                value = value.children[0] if value.children else None

        self.variables[name] = value
        return {name: value}

    def value(self, items: list[Any]) -> Any:
        """Return the actual value from the parse tree.

        Args:
            items: List containing value

        Returns:
            Value or None
        """
        if items:
            return items[0]
        return None

    def NUMBER(self, token: Token) -> float:
        """Parse number token.

        Args:
            token: Number token

        Returns:
            Float value
        """
        return float(token)

    def STRING(self, token: Token) -> str:
        """Parse string token.

        Args:
            token: String token

        Returns:
            String value with quotes removed
        """
        return str(token)[1:-1]  # Remove quotes

    def IDENTIFIER(self, token: Token) -> str:
        """Parse identifier token.

        Args:
            token: Identifier token

        Returns:
            Identifier string
        """
        return str(token)

    def ELEMENT_TYPE_TOKEN(self, token: Token) -> str:
        """Parse element type token.

        Args:
            token: Element type token

        Returns:
            Element type string
        """
        return str(token)

    def ACTION_TYPE_TOKEN(self, token: Token) -> str:
        """Parse action type token.

        Args:
            token: Action type token

        Returns:
            Action type string
        """
        return str(token)

    def BOOLEAN(self, token: Token) -> bool:
        """Parse boolean token.

        Args:
            token: Boolean token

        Returns:
            Boolean value
        """
        return str(token) == "true"

    def json_object(self, items: list[Any]) -> dict[str, Any]:
        """Parse JSON object.

        Args:
            items: List of key-value tuples

        Returns:
            Dictionary representing JSON object
        """
        obj: dict[str, Any] = {}
        for item in items:
            if isinstance(item, tuple):
                obj[item[0]] = item[1]
        return obj

    def json_pair(self, items: list[Any]) -> tuple[str, Any]:
        """Parse JSON key-value pair.

        Args:
            items: List containing key and value

        Returns:
            Tuple of key and value
        """
        return (items[0], items[1])

    def json_array(self, items: list[Any]) -> list[Any]:
        """Parse JSON array.

        Args:
            items: List of array elements

        Returns:
            List representing JSON array
        """
        return list(items)


class QontinuiDSLParser:
    """Parser for Qontinui DSL scripts."""

    def __init__(self) -> None:
        """Initialize the DSL parser."""
        self.parser: Lark = Lark(QONTINUI_GRAMMAR, parser="lalr")

    def parse(self, script: str) -> dict[str, Any]:
        """Parse a Qontinui DSL script.

        Args:
            script: DSL script content

        Returns:
            Parsed representation
        """
        try:
            tree = self.parser.parse(script)
            transformer = QontinuiTransformer()
            result = transformer.transform(tree)
            return cast(dict[str, Any], result)
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            # Catch parsing errors from Lark parser and transformer
            logger.error(f"Failed to parse DSL script: {e}")
            raise

    def parse_file(self, filepath: str) -> dict[str, Any]:
        """Parse a Qontinui DSL script from file.

        Args:
            filepath: Path to DSL script file

        Returns:
            Parsed representation
        """
        with open(filepath) as f:
            script = f.read()
        return self.parse(script)

    def validate(self, script: str) -> bool:
        """Validate a DSL script.

        Args:
            script: DSL script content

        Returns:
            True if valid, False otherwise
        """
        try:
            self.parse(script)
            return True
        except (ValueError, TypeError, AttributeError, RuntimeError, LarkError):
            # Catch parsing errors to indicate invalid DSL syntax
            return False

    def to_python(self, parsed: dict[str, Any]) -> str:
        """Convert parsed DSL to Python code.

        Args:
            parsed: Parsed DSL representation

        Returns:
            Generated Python code
        """
        lines = []
        lines.append("# Generated from Qontinui DSL")
        lines.append(
            "from qontinui import QontinuiStateManager, State, Element, Transition"
        )
        lines.append("")

        # Generate state definitions
        for state_name, state in parsed.get("states", {}).items():
            lines.append(f"# State: {state_name}")
            lines.append(f"{state_name}_state = State(")
            lines.append(f'    name="{state.name}",')
            lines.append("    elements=[],")
            lines.append(f"    min_elements={state.min_elements},")
            if state.parent:
                lines.append(f'    parent_state="{state.parent}",')
            lines.append(")")
            lines.append("")

        # Generate transition definitions
        for trans_name, trans in parsed.get("transitions", {}).items():
            lines.append(f"# Transition: {trans_name}")
            lines.append(f"{trans_name}_transition = Transition(")
            lines.append(f'    from_state="{trans.from_state}",')
            lines.append(f'    to_state="{trans.to_state}",')
            lines.append(f"    action_type=TransitionType.{trans.action.upper()},")
            if trans.trigger:
                lines.append(f'    trigger_element="{trans.trigger}",')
            lines.append(f"    probability={trans.probability},")
            lines.append(")")
            lines.append("")

        # Generate action calls
        lines.append("# Actions")
        for action in parsed.get("actions", []):
            args_str = ", ".join(f"{k}={v}" for k, v in action.arguments.items())
            lines.append(f"actions.{action.action_type}({args_str})")

        return "\n".join(lines)


# Example DSL script
EXAMPLE_DSL = """
// Define login state
state LoginPage {
    elements: [
        {id: "username_field", type: input, text: "Username"},
        {id: "password_field", type: input, text: "Password"},
        {id: "login_button", type: button, text: "Login"}
    ]
    min_elements: 3
}

// Define home state
state HomePage {
    elements: [
        {id: "welcome_text", type: text},
        {id: "menu_button", type: button},
        {id: "logout_link", type: link, text: "Logout"}
    ]
    min_elements: 2
}

// Define transition from login to home
transition login_to_home {
    from: LoginPage
    to: HomePage
    action: click
    trigger: login_button
    probability: 0.95
}

// Example automation script
var username = "test_user";
var password = "test_pass";

click(element=username_field);
type(text=username);
click(element=password_field);
type(text=password);
click(element=login_button);

assert current_state == HomePage;
"""
