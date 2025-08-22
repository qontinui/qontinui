# Brobot Test Migration System

This system provides tools for migrating Java unit and integration tests from the Brobot library to equivalent Python tests in the Qontinui project.

## Overview

The Brobot Test Migration System addresses the unique challenges of migrating model-based GUI automation tests, including Brobot's mocking capabilities for GUI environment simulation. It follows a modular architecture with clear separation between test discovery, migration logic, execution, and validation components.

## Directory Structure

```
test_migration/
├── __init__.py                 # Main package initialization
├── README.md                   # This documentation
├── config.py                   # Configuration management
├── pytest.ini                  # Pytest configuration
├── core/                       # Core interfaces and data models
│   ├── __init__.py
│   ├── models.py              # Data models (TestFile, MockUsage, etc.)
│   └── interfaces.py          # Abstract interfaces
├── discovery/                  # Test discovery and classification
│   └── __init__.py
├── translation/               # Java-to-Python translation engine
│   └── __init__.py
├── mocks/                     # Mock migration system
│   └── __init__.py
├── execution/                 # Test execution and result collection
│   └── __init__.py
├── validation/                # Test validation and diagnostics
│   └── __init__.py
└── tests/                     # Test suite for the migration system
    ├── __init__.py
    ├── conftest.py           # Pytest fixtures and configuration
    ├── test_core_models.py   # Tests for core data models
    └── test_config.py        # Tests for configuration module
```

## Core Components

### Data Models (`core/models.py`)

- **TestFile**: Represents a Java test file to be migrated
- **TestMethod**: Represents individual test methods
- **MockUsage**: Represents mock usage patterns in tests
- **GuiModel**: Represents GUI models used in Brobot mocks
- **Dependency**: Represents Java-to-Python dependency mappings
- **MigrationConfig**: Configuration for the migration process
- **TestFailure/FailureAnalysis**: Test failure analysis models

### Interfaces (`core/interfaces.py`)

Abstract interfaces for all major components:
- **TestScanner**: Test discovery and classification
- **TestTranslator**: Java-to-Python translation
- **MockAnalyzer/MockGenerator**: Mock migration
- **TestRunner**: Test execution
- **FailureAnalyzer**: Test failure analysis
- **BehaviorComparator**: Test behavior comparison
- **DiagnosticReporter**: Diagnostic reporting
- **MigrationOrchestrator**: Complete migration orchestration

### Configuration (`config.py`)

Provides default configurations and mappings:
- Java-to-Python dependency mappings
- Brobot-to-Qontinui mock mappings
- Default test patterns and exclusions
- Pytest markers for different test types

## Key Features

### 1. Test Discovery and Classification
- Automatically discovers Java test files in Brobot directories
- Classifies tests as unit vs integration tests
- Identifies mock usage patterns and dependencies

### 2. Java-to-Python Translation
- Converts Java test syntax to Python/pytest equivalents
- Maps JUnit assertions to pytest assertions
- Handles SpringBoot test patterns and annotations

### 3. Mock Migration
- Preserves Brobot's model-based GUI automation approach
- Maps Brobot mocks to equivalent Qontinui mocks
- Maintains GUI state simulation capabilities

### 4. Test Execution and Validation
- Runs migrated Python tests using pytest
- Compares behavior between Java and Python tests
- Provides diagnostic analysis for test failures

### 5. Failure Analysis
- Distinguishes between test migration errors and code migration errors
- Provides confidence scoring for failure analysis
- Suggests fixes for common migration issues

## Usage

### Basic Configuration

```python
from qontinui.test_migration.config import TestMigrationConfig
from pathlib import Path

# Create default configuration
config = TestMigrationConfig.create_default_config(
    source_directories=[Path("brobot/library/src/test")],
    target_directory=Path("tests/migrated")
)

# Or create from environment variables
config = TestMigrationConfig.from_environment()
```

### Running Tests

The migration system includes its own test suite:

```bash
# Run the setup verification
python test_migration_setup.py

# Run specific tests (when pytest is available)
pytest src/qontinui/test_migration/tests/ -v
```

## Testing

The system includes comprehensive tests:
- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test component interactions
- **Migration tests**: Test the migration process itself

### Test Markers

- `@pytest.mark.unit`: Unit tests for individual components
- `@pytest.mark.integration`: Integration tests for component interaction
- `@pytest.mark.migration`: Tests for the migration process itself
- `@pytest.mark.slow`: Tests that take a long time to run
- `@pytest.mark.requires_java`: Tests that require Java environment
- `@pytest.mark.requires_brobot`: Tests that require Brobot test files

## Requirements

### Core Requirements (Requirement 7.1, 7.4)
- ✅ Directory structure for test migration components
- ✅ Core interfaces and data models for test discovery, migration, and validation
- ✅ Pytest configuration for the migration test suite

### Dependencies

The test migration system is designed to be independent of the main Qontinui package to avoid dependency conflicts during migration. It only requires:
- Python 3.12+
- Standard library modules (dataclasses, enum, pathlib, abc)
- pytest (for running tests)

## Next Steps

This completes Task 1 of the implementation plan. The next tasks will implement:

1. **Test Discovery System** (Task 2): BrobotTestScanner and TestClassifier
2. **Translation Engine** (Task 3): JavaToPythonTranslator and AssertionConverter
3. **Test Generation** (Task 4): PythonTestGenerator and PytestRunner
4. **Mock Migration** (Task 5): BrobotMockAnalyzer and QontinuiMockGenerator
5. **Validation System** (Tasks 6-9): Failure analysis and diagnostic tools
6. **Complete Workflow** (Task 10): MigrationOrchestrator and CLI interface

## Architecture Principles

The system follows these key principles:
1. **Modular Design**: Clear separation of concerns between components
2. **Independence**: Minimal dependencies to avoid conflicts
3. **Extensibility**: Plugin-based architecture for custom patterns
4. **Testability**: Comprehensive test coverage with clear interfaces
5. **Brobot Compatibility**: Preserves Brobot's model-based approach