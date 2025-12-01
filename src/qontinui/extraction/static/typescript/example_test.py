"""
Example usage and test of the TypeScript parser.

This file demonstrates how to use the TypeScript parser to analyze
React components and extract useful information.
"""

import asyncio
from pathlib import Path
from tempfile import NamedTemporaryFile

from .parser import TypeScriptParser


def create_example_component() -> Path:
    """Create an example TypeScript React component for testing."""
    code = """
import React, { useState, useEffect } from 'react';
import { Modal } from './Modal';

interface TodoItem {
  id: number;
  text: string;
  completed: boolean;
}

interface TodoListProps {
  initialTodos?: TodoItem[];
  title?: string;
}

export const TodoList: React.FC<TodoListProps> = ({ initialTodos = [], title = "My Todos" }) => {
  const [todos, setTodos] = useState<TodoItem[]>(initialTodos);
  const [newTodo, setNewTodo] = useState('');
  const [filter, setFilter] = useState<'all' | 'active' | 'completed'>('all');
  const [isModalOpen, setIsModalOpen] = useState(false);

  useEffect(() => {
    const savedTodos = localStorage.getItem('todos');
    if (savedTodos) {
      setTodos(JSON.parse(savedTodos));
    }
  }, []);

  const handleAddTodo = () => {
    if (newTodo.trim()) {
      setTodos([...todos, { id: Date.now(), text: newTodo, completed: false }]);
      setNewTodo('');
    }
  };

  const handleToggleTodo = (id: number) => {
    setTodos(todos.map(todo =>
      todo.id === id ? { ...todo, completed: !todo.completed } : todo
    ));
  };

  const handleDeleteTodo = (id: number) => {
    setTodos(todos.filter(todo => todo.id !== id));
  };

  const filteredTodos = todos.filter(todo => {
    if (filter === 'active') return !todo.completed;
    if (filter === 'completed') return todo.completed;
    return true;
  });

  return (
    <div className="todo-list">
      <h1>{title}</h1>

      <div className="add-todo">
        <input
          type="text"
          value={newTodo}
          onChange={(e) => setNewTodo(e.target.value)}
          placeholder="Add a new todo"
        />
        <button onClick={handleAddTodo}>Add</button>
      </div>

      <div className="filters">
        <button onClick={() => setFilter('all')}>All</button>
        <button onClick={() => setFilter('active')}>Active</button>
        <button onClick={() => setFilter('completed')}>Completed</button>
      </div>

      {filteredTodos.length === 0 && (
        <p className="empty">No todos found</p>
      )}

      <ul className="todo-items">
        {filteredTodos.map(todo => (
          <li key={todo.id} className={todo.completed ? 'completed' : ''}>
            <input
              type="checkbox"
              checked={todo.completed}
              onChange={() => handleToggleTodo(todo.id)}
            />
            <span>{todo.text}</span>
            <button onClick={() => handleDeleteTodo(todo.id)}>Delete</button>
          </li>
        ))}
      </ul>

      <button onClick={() => setIsModalOpen(true)}>Settings</button>

      {isModalOpen && (
        <Modal onClose={() => setIsModalOpen(false)}>
          <h2>Settings</h2>
          <p>Todo list settings go here</p>
        </Modal>
      )}
    </div>
  );
};

export default TodoList;
"""

    # Create temporary file
    temp = NamedTemporaryFile(mode="w", suffix=".tsx", delete=False)
    temp.write(code)
    temp.close()
    return Path(temp.name)


async def test_parser():
    """Test the TypeScript parser with an example component."""
    print("QontinUI TypeScript Parser - Example Test\n")
    print("=" * 60)

    # Create example component
    component_file = create_example_component()
    print(f"\nCreated example component: {component_file}\n")

    try:
        # Initialize parser
        parser = TypeScriptParser()

        # Parse the file
        print("Parsing component...")
        result = await parser.parse_files([component_file])

        # Get the file result
        file_path = str(component_file)
        file_result = result.files.get(file_path)

        if not file_result:
            print("ERROR: No parse result found")
            return

        if file_result.error:
            print(f"ERROR: {file_result.error}")
            return

        print("\n" + "=" * 60)
        print("PARSE RESULTS")
        print("=" * 60)

        # Display components
        print("\n1. COMPONENTS FOUND:")
        print("-" * 60)
        for comp in file_result.components:
            print(f"\n  Name: {comp.name}")
            print(f"  Type: {comp.type}")
            print(f"  Line: {comp.line}")
            print(
                f"  Props: {[f'{p.name}={p.default}' if p.default else p.name for p in comp.props]}"
            )
            print(f"  Children: {comp.children}")
            print(f"  Returns JSX: {comp.returns_jsx}")

        # Display state variables
        print("\n2. STATE VARIABLES:")
        print("-" * 60)
        for state in file_result.state_variables:
            print(f"\n  Name: {state.name}")
            print(f"  Hook: {state.hook}")
            print(f"  Setter: {state.setter}")
            print(f"  Initial Value: {state.initial_value}")
            print(f"  Type: {state.type}")
            print(f"  Line: {state.line}")

        # Display conditional renders
        print("\n3. CONDITIONAL RENDERING:")
        print("-" * 60)
        for cond in file_result.conditional_renders:
            print(f"\n  Pattern: {cond.pattern}")
            print(f"  Condition: {cond.condition}")
            print(f"  Line: {cond.line}")
            if cond.renders:
                print(f"  Renders: {cond.renders}")
            if cond.renders_true:
                print(f"  Renders (true): {cond.renders_true}")
            if cond.renders_false:
                print(f"  Renders (false): {cond.renders_false}")

        # Display event handlers
        print("\n4. EVENT HANDLERS:")
        print("-" * 60)
        for handler in file_result.event_handlers:
            print(f"\n  Name: {handler.name}")
            print(f"  Event: {handler.event}")
            print(f"  Line: {handler.line}")
            print(f"  State Changes: {handler.state_changes}")

        # Display imports
        print("\n5. IMPORTS:")
        print("-" * 60)
        for imp in file_result.imports:
            print(f"\n  Source: {imp.source}")
            print(f"  Line: {imp.line}")
            for spec in imp.specifiers:
                if spec["type"] == "named":
                    print(f"    - {spec['name']} (named)")
                elif spec["type"] == "default":
                    print(f"    - {spec['name']} (default)")

        # Display exports
        print("\n6. EXPORTS:")
        print("-" * 60)
        for exp in file_result.exports:
            print(f"\n  Name: {exp.name}")
            print(f"  Type: {exp.type}")
            print(f"  Line: {exp.line}")

        # Display JSX elements
        print("\n7. JSX ELEMENTS:")
        print("-" * 60)
        jsx_counts = {}
        for jsx in file_result.jsx_elements:
            jsx_counts[jsx.name] = jsx_counts.get(jsx.name, 0) + 1

        for name, count in sorted(jsx_counts.items()):
            print(f"  {name}: {count} occurrence(s)")

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Components: {len(file_result.components)}")
        print(f"  State Variables: {len(file_result.state_variables)}")
        print(f"  Conditional Renders: {len(file_result.conditional_renders)}")
        print(f"  Event Handlers: {len(file_result.event_handlers)}")
        print(f"  Imports: {len(file_result.imports)}")
        print(f"  Exports: {len(file_result.exports)}")
        print(f"  Unique JSX Elements: {len(jsx_counts)}")
        print("=" * 60 + "\n")

    finally:
        # Clean up temporary file
        component_file.unlink()
        print(f"Cleaned up temporary file: {component_file}")


def test_parser_sync():
    """Synchronous test wrapper."""
    asyncio.run(test_parser())


if __name__ == "__main__":
    test_parser_sync()
