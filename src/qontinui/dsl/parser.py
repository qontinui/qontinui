"""DSL parser for Qontinui automation scripts."""

from typing import Dict, List, Any, Optional
from lark import Lark, Transformer, Tree
from dataclasses import dataclass
import json
import logging


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
    state_property: "elements" ":" element_list
                  | "min_elements" ":" NUMBER
                  | "parent" ":" IDENTIFIER
                  | "metadata" ":" json_value

    element_list: "[" [element ("," element)*] "]"
    element: IDENTIFIER | element_inline

    element_inline: "{" element_props "}"
    element_props: element_prop ("," element_prop)*
    element_prop: "id" ":" STRING
                | "type" ":" element_type
                | "bbox" ":" bbox
                | "text" ":" STRING
                | "description" ":" STRING

    element_type: "button" | "text" | "input" | "image" | "icon" | "link" | "checkbox"

    bbox: "[" NUMBER "," NUMBER "," NUMBER "," NUMBER "]"

    transition_def: "transition" IDENTIFIER "{" transition_body "}"
    transition_body: transition_property*
    transition_property: "from" ":" IDENTIFIER
                       | "to" ":" IDENTIFIER
                       | "action" ":" action_type
                       | "trigger" ":" IDENTIFIER
                       | "probability" ":" NUMBER
                       | "conditions" ":" condition_list

    action_type: "click" | "type" | "hover" | "drag" | "scroll" | "key_press" | "wait"

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
        | "true" | "false"
        | "(" expression ")"
        | function_call

    function_call: IDENTIFIER "(" [args] ")"
    args: value ("," value)*

    iterable: IDENTIFIER
            | "[" [value ("," value)*] "]"
            | function_call

    value: NUMBER
         | STRING
         | IDENTIFIER
         | json_value
         | "true" | "false"
         | "null"

    json_value: json_object | json_array
    json_object: "{" [json_pair ("," json_pair)*] "}"
    json_pair: STRING ":" json_element
    json_array: "[" [json_element ("," json_element)*] "]"
    json_element: STRING | NUMBER | "true" | "false" | "null" | json_object | json_array

    COMMENT: "//" /[^\n]*/ NEWLINE
           | "/*" /.*?/ "*/"

    COMP_OP: "==" | "!=" | "<" | ">" | "<=" | ">="
    ADD_OP: "+" | "-"
    MUL_OP: "*" | "/" | "%"

    %import common.CNAME -> IDENTIFIER
    %import common.NUMBER
    %import common.ESCAPED_STRING -> STRING
    %import common.WS
    %import common.NEWLINE
    %ignore WS
    %ignore COMMENT
"""


@dataclass
class ParsedState:
    """Parsed state definition."""
    name: str
    elements: List[Dict[str, Any]]
    min_elements: int
    parent: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class ParsedTransition:
    """Parsed transition definition."""
    name: str
    from_state: str
    to_state: str
    action: str
    trigger: Optional[str]
    probability: float
    conditions: List[str]


@dataclass
class ParsedAction:
    """Parsed action command."""
    action_type: str
    arguments: Dict[str, Any]


class QontinuiTransformer(Transformer):
    """Transform parsed DSL into Python objects."""
    
    def __init__(self):
        super().__init__()
        self.states = {}
        self.transitions = {}
        self.elements = {}
        self.variables = {}
        self.actions = []
    
    def program(self, items):
        # Process all items to populate dictionaries
        for item in items:
            pass  # Items are processed as side effects
        
        return {
            "states": self.states,
            "transitions": self.transitions,
            "elements": self.elements,
            "variables": self.variables,
            "actions": self.actions,
        }
    
    def state_def(self, items):
        name = str(items[0])
        properties = {}
        
        # Collect properties from state_body
        if len(items) > 1 and isinstance(items[1], dict):
            properties = items[1]
        
        state = ParsedState(
            name=name,
            elements=properties.get("elements", []),
            min_elements=properties.get("min_elements", 1),
            parent=properties.get("parent"),
            metadata=properties.get("metadata", {})
        )
        
        self.states[name] = state
        return state
    
    def state_body(self, items):
        properties = {}
        for item in items:
            if isinstance(item, tuple) and len(item) == 2:
                properties[item[0]] = item[1]
            elif isinstance(item, dict):
                properties.update(item)
        return properties
    
    def state_property(self, items):
        # Return tuple of (key, value) for proper handling
        if items[0] == "elements":
            return ("elements", items[1])
        elif items[0] == "min_elements":
            return ("min_elements", items[1])
        elif items[0] == "parent":
            return ("parent", str(items[1]))
        elif items[0] == "metadata":
            return ("metadata", items[1])
        return {}
    
    def element_list(self, items):
        return items
    
    def element_inline(self, items):
        props = {}
        for item in items:
            if isinstance(item, dict):
                props.update(item)
        return props
    
    def element_prop(self, items):
        if len(items) >= 2:
            key = str(items[0])
            value = items[1]
            return {key: value}
        return {}
    
    def bbox(self, items):
        return [int(item) for item in items]
    
    def transition_def(self, items):
        name = str(items[0])
        properties = {}
        
        if len(items) > 1 and isinstance(items[1], dict):
            properties = items[1]
        
        transition = ParsedTransition(
            name=name,
            from_state=properties.get("from"),
            to_state=properties.get("to"),
            action=properties.get("action", "click"),
            trigger=properties.get("trigger"),
            probability=properties.get("probability", 1.0),
            conditions=properties.get("conditions", [])
        )
        
        self.transitions[name] = transition
        return transition
    
    def transition_body(self, items):
        properties = {}
        for item in items:
            if isinstance(item, tuple) and len(item) == 2:
                properties[item[0]] = item[1]
            elif isinstance(item, dict):
                properties.update(item)
        return properties
    
    def transition_property(self, items):
        if items[0] == "from":
            return ("from", str(items[1]))
        elif items[0] == "to":
            return ("to", str(items[1]))
        elif items[0] == "action":
            return ("action", str(items[1]))
        elif items[0] == "trigger":
            return ("trigger", str(items[1]))
        elif items[0] == "probability":
            return ("probability", float(items[1]))
        elif items[0] == "conditions":
            return ("conditions", items[1])
        return {}
    
    def action_def(self, items):
        action_type = str(items[0])
        args = {}
        
        if len(items) > 1 and isinstance(items[1], dict):
            args = items[1]
        
        action = ParsedAction(
            action_type=action_type,
            arguments=args
        )
        
        self.actions.append(action)
        return action
    
    def action_type(self, items):
        return str(items[0])
    
    def action_args(self, items):
        args = {}
        for item in items:
            if isinstance(item, dict):
                args.update(item)
            elif isinstance(item, tuple):
                args[item[0]] = item[1]
        return args
    
    def element_def(self, items):
        name = str(items[0])
        props = items[1] if len(items) > 1 else {}
        self.elements[name] = props
        return {name: props}
    
    def variable_def(self, items):
        name = str(items[0])
        value = items[1] if len(items) > 1 else None
        
        # Extract actual value if it's a Tree
        if hasattr(value, 'data'):
            if value.data == 'value':
                value = value.children[0] if value.children else None
        
        self.variables[name] = value
        return {name: value}
    
    def value(self, items):
        # Return the actual value from the parse tree
        if items:
            return items[0]
        return None
    
    def NUMBER(self, token):
        return float(token)
    
    def STRING(self, token):
        return str(token)[1:-1]  # Remove quotes
    
    def IDENTIFIER(self, token):
        return str(token)
    
    def json_object(self, items):
        obj = {}
        for item in items:
            if isinstance(item, tuple):
                obj[item[0]] = item[1]
        return obj
    
    def json_pair(self, items):
        return (items[0], items[1])
    
    def json_array(self, items):
        return list(items)


class QontinuiDSLParser:
    """Parser for Qontinui DSL scripts."""
    
    def __init__(self):
        """Initialize the DSL parser."""
        self.parser = Lark(QONTINUI_GRAMMAR, parser='lalr', transformer=QontinuiTransformer())
        self.transformer = QontinuiTransformer()
    
    def parse(self, script: str) -> Dict[str, Any]:
        """Parse a Qontinui DSL script.
        
        Args:
            script: DSL script content
            
        Returns:
            Parsed representation
        """
        try:
            tree = self.parser.parse(script)
            return tree
        except Exception as e:
            logger.error(f"Failed to parse DSL script: {e}")
            raise
    
    def parse_file(self, filepath: str) -> Dict[str, Any]:
        """Parse a Qontinui DSL script from file.
        
        Args:
            filepath: Path to DSL script file
            
        Returns:
            Parsed representation
        """
        with open(filepath, 'r') as f:
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
        except:
            return False
    
    def to_python(self, parsed: Dict[str, Any]) -> str:
        """Convert parsed DSL to Python code.
        
        Args:
            parsed: Parsed DSL representation
            
        Returns:
            Generated Python code
        """
        lines = []
        lines.append("# Generated from Qontinui DSL")
        lines.append("from qontinui import QontinuiStateManager, State, Element, Transition")
        lines.append("")
        
        # Generate state definitions
        for state_name, state in parsed.get("states", {}).items():
            lines.append(f"# State: {state_name}")
            lines.append(f"{state_name}_state = State(")
            lines.append(f'    name="{state.name}",')
            lines.append(f"    elements=[],")
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
            lines.append(f'    action_type=TransitionType.{trans.action.upper()},')
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