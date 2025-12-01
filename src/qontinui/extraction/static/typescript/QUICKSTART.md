# Quick Start Guide

Get up and running with the QontinUI TypeScript Parser in 5 minutes.

## Prerequisites

- Node.js 18+ installed
- Python 3.10+ installed
- QontinUI package installed

## Installation

```bash
# Navigate to the parser directory
cd qontinui/src/qontinui/extraction/static/typescript

# Install Node.js dependencies
./install.sh
```

## Basic Usage

### 1. Parse a Single File

```python
from pathlib import Path
from qontinui.extraction.static.typescript import TypeScriptParser

# Create parser
parser = TypeScriptParser()

# Parse a file
result = parser.parse_files_sync([Path("src/App.tsx")])

# Get the file result
file_result = result.files["src/App.tsx"]

# Print components
for component in file_result.components:
    print(f"Component: {component.name}")
    print(f"  Type: {component.type}")
    print(f"  Props: {[p.name for p in component.props]}")
    print(f"  Children: {component.children}")
```

### 2. Parse a Directory

```python
from pathlib import Path
from qontinui.extraction.static.typescript import TypeScriptParser

parser = TypeScriptParser()

# Parse all TypeScript files in src/
result = parser.parse_directory_sync(
    directory=Path("src"),
    patterns=["*.tsx", "*.ts"],
)

# Print summary
print(f"Parsed {len(result.files)} files")
for file_path, file_result in result.files.items():
    print(f"{file_path}: {len(file_result.components)} components")
```

### 3. Analyze State Management

```python
from pathlib import Path
from qontinui.extraction.static.typescript import TypeScriptParser

parser = TypeScriptParser()
result = parser.parse_files_sync([Path("src/TodoApp.tsx")])

file_result = result.files["src/TodoApp.tsx"]

# Print state variables
print("State Variables:")
for state in file_result.state_variables:
    print(f"  {state.name}: {state.hook}({state.initial_value})")
    print(f"    Setter: {state.setter}")
    print(f"    Type: {state.type}")
```

### 4. Find Event Handlers

```python
from pathlib import Path
from qontinui.extraction.static.typescript import TypeScriptParser

parser = TypeScriptParser()
result = parser.parse_files_sync([Path("src/Form.tsx")])

file_result = result.files["src/Form.tsx"]

# Print event handlers
print("Event Handlers:")
for handler in file_result.event_handlers:
    print(f"  {handler.name} (on{handler.event})")
    if handler.state_changes:
        print(f"    State changes: {', '.join(handler.state_changes)}")
```

### 5. Analyze Conditional Rendering

```python
from pathlib import Path
from qontinui.extraction.static.typescript import TypeScriptParser

parser = TypeScriptParser()
result = parser.parse_files_sync([Path("src/Modal.tsx")])

file_result = result.files["src/Modal.tsx"]

# Print conditionals
print("Conditional Rendering:")
for cond in file_result.conditional_renders:
    print(f"  Pattern: {cond.pattern}")
    print(f"  Condition: {cond.condition}")
    print(f"  Renders: {cond.renders or cond.renders_true}")
```

### 6. Complete Project Analysis

```python
from pathlib import Path
from qontinui.extraction.static.typescript.integration_example import (
    analyze_typescript_project
)

# Analyze entire project
report = analyze_typescript_project(Path("./my-react-app"))

# Get summary
summary = report.generate_summary()
print(f"Total Components: {summary['total_components']}")
print(f"Total State Variables: {summary['total_state_variables']}")
print(f"Total Event Handlers: {summary['total_event_handlers']}")

# Get component hierarchy
hierarchy = report.get_component_hierarchy()
print("\nComponent Hierarchy:")
for component, children in hierarchy.items():
    if children:
        print(f"  {component} -> {', '.join(children)}")

# Get state graph
state_graph = report.get_state_graph()
print("\nState Variables:")
for state_name, info in state_graph.items():
    print(f"  {state_name}: {info['hook']}({info['initial']})")

# Get interactions
interactions = report.get_interaction_map()
print("\nUser Interactions:")
for event, handlers in interactions.items():
    print(f"  on{event.title()}: {len(handlers)} handler(s)")
```

## Common Patterns

### Extract Specific Information

```python
# Only extract components and state
result = parser.parse_files_sync(
    files=[Path("src/App.tsx")],
    extract=["components", "state"]
)

# Only extract conditionals and handlers
result = parser.parse_files_sync(
    files=[Path("src/App.tsx")],
    extract=["conditionals", "handlers"]
)
```

Available extract types:
- `components`: Component definitions
- `state`: React hooks and state
- `conditionals`: Conditional rendering
- `handlers`: Event handlers
- `imports`: Import/export statements

### Async Usage

```python
import asyncio
from pathlib import Path
from qontinui.extraction.static.typescript import TypeScriptParser

async def analyze():
    parser = TypeScriptParser()
    result = await parser.parse_files([Path("src/App.tsx")])
    return result

result = asyncio.run(analyze())
```

### Error Handling

```python
from pathlib import Path
from qontinui.extraction.static.typescript import TypeScriptParser

parser = TypeScriptParser()
result = parser.parse_files_sync([Path("src/BrokenFile.tsx")])

# Check for errors
if result.errors:
    print("Errors occurred:")
    for error in result.errors:
        print(f"  {error}")

# Check individual file errors
for file_path, file_result in result.files.items():
    if file_result.error:
        print(f"Error in {file_path}: {file_result.error}")
    else:
        print(f"{file_path}: OK")
```

### Filter Files

```python
from pathlib import Path
from qontinui.extraction.static.typescript import TypeScriptParser

parser = TypeScriptParser()

# Parse only .tsx files, exclude tests
result = parser.parse_directory_sync(
    directory=Path("src"),
    patterns=["*.tsx"],  # Only .tsx files
    exclude=[
        "*.test.tsx",
        "*.spec.tsx",
        "node_modules/**",
        "dist/**",
    ],
)
```

## Example Output

```python
# Component Information
ComponentInfo(
    name='TodoList',
    type='function',
    line=15,
    props=[
        PropInfo(name='initialTodos', default='[]'),
        PropInfo(name='title', default='"My Todos"')
    ],
    children=['Modal', 'TodoItem'],
    returns_jsx=True
)

# State Variable Information
StateVariableInfo(
    name='todos',
    setter='setTodos',
    hook='useState',
    line=16,
    initial_value='initialTodos',
    type='array'
)

# Conditional Rendering
ConditionalRenderInfo(
    condition='isModalOpen',
    line=45,
    pattern='AND',
    renders=['Modal']
)

# Event Handler
EventHandlerInfo(
    event='click',
    line=32,
    name='handleAddTodo',
    state_changes=['setTodos', 'setNewTodo']
)
```

## Run Tests

```bash
# Run the example test
cd qontinui/src/qontinui/extraction/static/typescript
python example_test.py

# Analyze a real project
python integration_example.py /path/to/your/react/project
```

## Next Steps

1. Read the full [README.md](README.md) for detailed documentation
2. Check [integration_example.py](integration_example.py) for advanced usage
3. See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for architecture details
4. Integrate with your QontinUI workflows

## Troubleshooting

**Problem**: `FileNotFoundError: Parser script not found`
**Solution**: Make sure you're in the correct directory and parser.js exists

**Problem**: `RuntimeError: Failed to run Node.js parser`
**Solution**: Install Node.js or specify the path:
```python
parser = TypeScriptParser(node_path="/usr/local/bin/node")
```

**Problem**: `Error: Cannot find module '@babel/parser'`
**Solution**: Run the installation script:
```bash
./install.sh
```

## Help

For more information:
- Full documentation: [README.md](README.md)
- Implementation details: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- Example code: [example_test.py](example_test.py)
- Integration guide: [integration_example.py](integration_example.py)
