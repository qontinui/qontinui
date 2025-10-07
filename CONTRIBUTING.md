# Contributing to Qontinui

Thank you for your interest in contributing to Qontinui! This document provides guidelines for contributing to the project.

## Code of Conduct

Be respectful, constructive, and collaborative. We're all here to build something useful together.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/yourusername/qontinui/issues)
2. If not, create a new issue with:
   - Clear title describing the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version and OS
   - Code sample or JSON configuration if applicable

### Suggesting Features

1. Check existing [Issues](https://github.com/yourusername/qontinui/issues)
2. Create a new issue describing:
   - The problem you're trying to solve
   - Your proposed solution
   - Example use cases
   - How it relates to Brobot (if applicable)

### Pull Requests

1. **Fork the repository** and create a branch from `main`
2. **Install development dependencies:**
   ```bash
   poetry install
   ```

3. **Make your changes:**
   - Follow the Brobot migration approach (see CLAUDE.md)
   - Write clear, documented code
   - Follow existing code style (black, ruff)
   - Add tests for new functionality
   - Update documentation if needed

4. **Run tests and linting:**
   ```bash
   poetry run pytest
   poetry run black .
   poetry run ruff check --fix .
   poetry run mypy src/
   ```

5. **Commit your changes:**
   - Use clear commit messages
   - Reference issues when applicable

6. **Push to your fork** and submit a pull request

7. **Address review feedback** if requested

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/qontinui.git
cd qontinui

# Install dependencies (including multistate)
poetry install

# Run tests
poetry run pytest

# Run linting
poetry run black .
poetry run ruff check --fix .

# Run type checking
poetry run mypy src/
```

## Migration from Brobot

Qontinui is a Python port of [Brobot](https://github.com/jspinak/brobot). When contributing:

### DO:
- Preserve Brobot's architecture and behavior
- Port classes faithfully from Java to Python
- Use Python idioms while maintaining the original design
- Check BROBOT_MIGRATION_TRACKER.md for migration status
- Ask for Brobot source code when porting a class

### DON'T:
- Create new architectures or abstractions
- Skip ActionConfig/Options classes
- Add "helper" methods that don't exist in Brobot
- **NEVER add backward compatibility code** - This is active development

See [CLAUDE.md](CLAUDE.md) for detailed migration guidelines.

## Code Style

- **Python**: Follow PEP 8, enforced by `black` and `ruff`
- **Type hints**: Required for all public APIs
- **Docstrings**: Google style for all public functions/classes
- **Line length**: 88 characters (black default)
- **Naming**: Follow Brobot naming conventions where applicable

## Testing

- Write tests for all new features
- Port corresponding Brobot tests when migrating classes
- Maintain or improve code coverage
- Use `pytest` for unit tests
- Test with actual GUI automation when applicable

## Documentation

- Update docstrings for any API changes
- Add examples for new features
- Update README.md if needed
- Document any deviations from Brobot behavior

## Project Structure

```
qontinui/
├── src/qontinui/
│   ├── actions/             # Core action system (from Brobot)
│   ├── state_management/    # State management (Brobot + MultiState)
│   ├── find/                # Image finding operations
│   ├── json_executor/       # JSON configuration execution
│   ├── hal/                 # Hardware Abstraction Layer
│   └── model/               # Data models (State, Transition, etc.)
├── tests/                   # Test suite
└── examples/                # Example automation scripts
```

## Areas for Contribution

### Good First Issues
- Documentation improvements
- Example automation projects
- Bug fixes in JSON executor
- HAL improvements for Linux/Mac

### Migration Tasks
Check BROBOT_MIGRATION_TRACKER.md for:
- Classes that need porting from Brobot
- Incomplete implementations
- Missing tests

### Advanced Contributions
- Performance optimizations
- New HAL implementations (wayland, mobile)
- Advanced MultiState integration
- GUI automation best practices

## Dependencies

Qontinui depends on:
- **[MultiState](https://github.com/jspinak/multistate)** - Multi-state state management
- **OpenCV** - Image template matching
- **PyAutoGUI/pynput** - Input control

## Questions?

- Check [CLAUDE.md](CLAUDE.md) for AI assistant guidelines
- Open an issue for questions
- Ask about Brobot behavior when porting classes

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing! 🎉
