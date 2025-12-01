# QontinUI TypeScript/JavaScript Parser

A robust TypeScript and JavaScript parser that extracts React component structures, state management patterns, conditional rendering, event handlers, and import/export relationships for static analysis.

## Features

- **Component Detection**: Identifies function, arrow function, and class components
- **React Hooks Analysis**: Extracts `useState`, `useReducer`, `useContext`, `useRef`, `useMemo`, `useCallback`
- **Conditional Rendering**: Detects logical AND (`&&`), ternary operators, and early returns
- **Event Handlers**: Identifies onClick, onChange, and other event handlers with state changes
- **Import/Export Tracking**: Maps module dependencies
- **JSX Hierarchy**: Extracts component composition and parent-child relationships
- **Error Resilient**: Handles syntax errors gracefully, supports partial parses

## Installation

First, install the Node.js dependencies:

```bash
cd qontinui/src/qontinui/extraction/static/typescript
npm install
```

## Usage

### Python API

```python
from pathlib import Path
from qontinui.extraction.static.typescript import TypeScriptParser

# Create parser instance
parser = TypeScriptParser()

# Parse specific files
files = [
    Path("src/App.tsx"),
    Path("src/components/Modal.tsx"),
]
result = parser.parse_files_sync(files)

# Access parsed data
for file_path, file_result in result.files.items():
    print(f"\nFile: {file_path}")

    # Components
    for component in file_result.components:
        print(f"  Component: {component.name} ({component.type})")
        print(f"    Props: {[p.name for p in component.props]}")
        print(f"    Children: {component.children}")

    # State variables
    for state in file_result.state_variables:
        print(f"  State: {state.name} = {state.hook}({state.initial_value})")
        if state.setter:
            print(f"    Setter: {state.setter}")

    # Conditional rendering
    for cond in file_result.conditional_renders:
        print(f"  Conditional: {cond.pattern} at line {cond.line}")
        print(f"    Condition: {cond.condition}")
        print(f"    Renders: {cond.renders or cond.renders_true}")

    # Event handlers
    for handler in file_result.event_handlers:
        print(f"  Handler: {handler.name} ({handler.event})")
        print(f"    State changes: {handler.state_changes}")

# Parse entire directory
result = parser.parse_directory_sync(
    directory=Path("src"),
    patterns=["*.tsx", "*.ts"],
    exclude=["*.test.tsx", "node_modules/**"],
)
```

### Async API

```python
import asyncio
from pathlib import Path
from qontinui.extraction.static.typescript import TypeScriptParser

async def analyze_code():
    parser = TypeScriptParser()

    # Parse files asynchronously
    result = await parser.parse_files([
        Path("src/App.tsx"),
        Path("src/components/Modal.tsx"),
    ])

    return result

# Run async
result = asyncio.run(analyze_code())
```

### Direct Node.js Usage

You can also call the parser directly from the command line:

```bash
# Create input configuration
echo '{
  "files": ["src/App.tsx", "src/components/Modal.tsx"],
  "extract": ["components", "state", "conditionals", "handlers", "imports"]
}' | node parser.js
```

## Configuration Options

### Extract Types

Specify which information to extract:

- `components`: Component definitions (function, class, arrow function)
- `state`: React hooks and state variables
- `conditionals`: Conditional rendering patterns
- `handlers`: Event handlers
- `imports`: Import and export statements

### File Patterns

When using `parse_directory`, you can specify:

```python
result = parser.parse_directory_sync(
    directory=Path("src"),
    patterns=["*.tsx", "*.ts", "*.jsx", "*.js"],  # Files to include
    exclude=[
        "node_modules/**",
        "dist/**",
        "*.test.ts",
        "*.spec.tsx",
    ],  # Files to exclude
    extract=["components", "state"],  # What to extract
)
```

## Output Format

### Component Information

```python
ComponentInfo(
    name="App",
    type="function",  # or "arrow_function", "class"
    line=10,
    props=[
        PropInfo(name="title", default=None),
        PropInfo(name="isOpen", default="false"),
    ],
    children=["Header", "Modal", "Footer"],
    returns_jsx=True,
    extends=None,  # For class components: "React.Component"
)
```

### State Variables

```python
StateVariableInfo(
    name="isOpen",
    setter="setIsOpen",
    hook="useState",
    line=12,
    initial_value="false",
    type="boolean",
)
```

### Conditional Rendering

```python
ConditionalRenderInfo(
    condition="isOpen",
    line=25,
    pattern="AND",  # or "TERNARY", "EARLY_RETURN"
    renders=["Modal"],
    renders_true=[],  # For ternary
    renders_false=[],  # For ternary
)
```

### Event Handlers

```python
EventHandlerInfo(
    event="click",
    line=18,
    name="handleClick",
    state_changes=["setIsOpen"],
)
```

### Imports/Exports

```python
ImportInfo(
    source="react",
    specifiers=[
        {"type": "named", "name": "useState", "imported": "useState"},
        {"type": "named", "name": "useEffect", "imported": "useEffect"},
    ],
    line=1,
)

ExportInfo(
    type="default",  # or "named"
    name="App",
    line=50,
)
```

## Example: Complete Analysis

```python
from pathlib import Path
from qontinui.extraction.static.typescript import create_parser

# Create parser
parser = create_parser()

# Parse a React component
result = parser.parse_files_sync([Path("src/components/TodoList.tsx")])

# Get the file result
file_result = result.files["src/components/TodoList.tsx"]

# Analyze component structure
print("Components found:")
for comp in file_result.components:
    print(f"  {comp.name}:")
    print(f"    Type: {comp.type}")
    print(f"    Props: {', '.join(p.name for p in comp.props)}")
    print(f"    Renders: {', '.join(comp.children)}")

# Analyze state management
print("\nState management:")
for state in file_result.state_variables:
    print(f"  {state.name} ({state.type})")
    print(f"    Hook: {state.hook}")
    print(f"    Initial: {state.initial_value}")
    print(f"    Setter: {state.setter}")

# Analyze user interactions
print("\nEvent handlers:")
for handler in file_result.event_handlers:
    print(f"  {handler.name} (on{handler.event.title()})")
    if handler.state_changes:
        print(f"    Updates: {', '.join(handler.state_changes)}")

# Analyze conditional logic
print("\nConditional rendering:")
for cond in file_result.conditional_renders:
    print(f"  {cond.pattern} at line {cond.line}")
    print(f"    Condition: {cond.condition}")
    if cond.pattern == "TERNARY":
        print(f"    True: {', '.join(cond.renders_true)}")
        print(f"    False: {', '.join(cond.renders_false)}")
    else:
        print(f"    Renders: {', '.join(cond.renders)}")
```

## Error Handling

The parser handles errors gracefully:

```python
result = parser.parse_files_sync([Path("src/BrokenComponent.tsx")])

# Check for errors
if result.errors:
    print("Parsing errors occurred:")
    for error in result.errors:
        print(f"  {error}")

# Individual file errors
for file_path, file_result in result.files.items():
    if file_result.error:
        print(f"Error in {file_path}: {file_result.error}")
    else:
        # File parsed successfully
        print(f"{file_path}: {len(file_result.components)} components found")
```

## Supported Patterns

### Component Patterns

```typescript
// Function declaration
function App() {
  return <div>Hello</div>;
}

// Arrow function
const App = () => <div>Hello</div>;

// Arrow function with body
const App = () => {
  return <div>Hello</div>;
};

// Class component
class App extends React.Component {
  render() {
    return <div>Hello</div>;
  }
}
```

### State Patterns

```typescript
// useState
const [count, setCount] = useState(0);

// useReducer
const [state, dispatch] = useReducer(reducer, initialState);

// useContext
const value = useContext(MyContext);

// useRef
const ref = useRef(null);

// useMemo
const memoized = useMemo(() => expensive(), [dep]);

// useCallback
const callback = useCallback(() => {}, [dep]);
```

### Conditional Patterns

```typescript
// Logical AND
{isOpen && <Modal />}

// Ternary operator
{isLoading ? <Spinner /> : <Content />}

// Early return
if (isError) return <ErrorPage />;
```

### Event Handler Patterns

```typescript
// Inline arrow function
<button onClick={() => setCount(count + 1)}>Click</button>

// Named function reference
<button onClick={handleClick}>Click</button>

// Inline function
<button onClick={function() { setCount(count + 1); }}>Click</button>
```

## Performance Considerations

- The parser processes files in parallel when possible
- Large codebases can be analyzed incrementally by directory
- Use the `exclude` parameter to skip unnecessary files (tests, node_modules, etc.)
- Consider parsing only specific extract types when full analysis isn't needed

## Troubleshooting

### Parser Script Not Found

```python
FileNotFoundError: Parser script not found at .../parser.js
```

**Solution**: Ensure `parser.js` is in the same directory as `parser.py`.

### Node.js Not Found

```python
RuntimeError: Failed to run Node.js parser
```

**Solution**: Install Node.js or specify the path:

```python
parser = TypeScriptParser(node_path="/usr/local/bin/node")
```

### Missing Dependencies

```bash
Error: Cannot find module '@babel/parser'
```

**Solution**: Install npm dependencies:

```bash
cd qontinui/src/qontinui/extraction/static/typescript
npm install
```

### Parse Errors

If a file has syntax errors, the parser will:
1. Report the error in `file_result.error`
2. Add the error to `result.errors`
3. Return empty arrays for extracted data
4. Continue parsing other files

## Integration with QontinUI

This parser is part of the QontinUI static analysis pipeline and works alongside:

- Python AST parser for Python/Django code
- CSS/Tailwind parser for styling analysis
- Template parser for HTML/JSX analysis

Together, these provide comprehensive code understanding for the QontinUI system.
