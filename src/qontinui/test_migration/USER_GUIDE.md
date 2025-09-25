# Qontinui Test Migration System - User Guide

## Overview

The Qontinui Test Migration System helps you migrate Java test suites from Brobot to equivalent Python tests in Qontinui. This system preserves the model-based GUI automation approach while adapting tests to Python syntax and testing frameworks.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [CLI Commands](#cli-commands)
4. [Configuration](#configuration)
5. [Migration Workflow](#migration-workflow)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Usage](#advanced-usage)

## Installation

### Prerequisites

- Python 3.8 or higher
- pytest (for running migrated tests)
- Access to Brobot Java test source code
- Qontinui project environment

### Setup

1. **Navigate to the test migration directory:**
   ```bash
   cd qontinui/src/qontinui/test_migration
   ```

2. **Set up the CLI (optional but recommended):**
   ```bash
   python setup_cli.py
   ```
   This creates launcher scripts for easier CLI access.

3. **Verify installation:**
   ```bash
   python cli.py --help
   ```

### Dependencies

The system uses existing Qontinui dependencies. If you encounter import issues, ensure these are available:
- `pathlib` (built-in)
- `argparse` (built-in)
- `json` (built-in)
- `logging` (built-in)
- `subprocess` (built-in)

## Quick Start

### 1. Basic Migration

Migrate Brobot tests to Qontinui tests:

```bash
python cli.py migrate /path/to/brobot/tests /path/to/qontinui/tests
```

### 2. Preview Migration (Dry Run)

See what would be migrated without making changes:

```bash
python cli.py migrate /path/to/brobot/tests /path/to/qontinui/tests --dry-run
```

### 3. Validate Migrated Tests

Run and validate migrated tests:

```bash
python cli.py validate /path/to/qontinui/tests
```

### 4. Generate Report

Create a comprehensive migration report:

```bash
python cli.py report /path/to/qontinui/tests --format html --output migration_report.html
```

## CLI Commands

### `migrate` - Migrate Java Tests to Python

**Syntax:**
```bash
python cli.py migrate <source_dir> <target_dir> [options]
```

**Options:**
- `--preserve-structure` / `--no-preserve-structure` - Keep directory structure (default: preserve)
- `--enable-mocks` / `--no-mocks` - Enable mock migration (default: enabled)
- `--parallel` / `--no-parallel` - Enable parallel execution (default: enabled)
- `--dry-run` - Preview migration without making changes
- `--output-format {json,yaml,text}` - Output format (default: text)
- `--report-file <path>` - Save migration report to file

**Examples:**
```bash
# Basic migration
python cli.py migrate brobot/library/src/test qontinui/tests/migrated

# Dry run with detailed output
python cli.py migrate brobot/library/src/test qontinui/tests/migrated --dry-run -vv

# Migration with flat structure and report
python cli.py migrate brobot/library/src/test qontinui/tests/migrated \
  --no-preserve-structure --report-file migration_report.json
```

### `validate` - Validate Migrated Tests

**Syntax:**
```bash
python cli.py validate <test_directory> [options]
```

**Options:**
- `--compare-with <path>` - Original Java test directory for comparison
- `--output-format {json,yaml,text}` - Output format (default: text)
- `--report-file <path>` - Save validation report to file

**Examples:**
```bash
# Basic validation
python cli.py validate qontinui/tests/migrated

# Validation with comparison and report
python cli.py validate qontinui/tests/migrated \
  --compare-with brobot/library/src/test \
  --report-file validation_report.json
```

### `report` - Generate Migration Reports

**Syntax:**
```bash
python cli.py report <test_directory> [options]
```

**Options:**
- `--format {html,json,yaml,text,pdf}` - Report format (default: html)
- `--output <path>` - Output file for the report
- `--include-coverage` - Include test coverage information
- `--include-diagnostics` - Include diagnostic information
- `--template <path>` - Custom report template file

**Examples:**
```bash
# HTML report with all information
python cli.py report qontinui/tests/migrated \
  --format html --output migration_report.html \
  --include-coverage --include-diagnostics

# JSON report for programmatic access
python cli.py report qontinui/tests/migrated \
  --format json --output migration_data.json
```

### `config` - Manage Configuration

**Syntax:**
```bash
python cli.py config [options]
```

**Options:**
- `--create` - Create a new configuration file
- `--validate` - Validate an existing configuration file
- `--output <path>` - Output file for configuration
- `--input <path>` - Input configuration file to validate

**Examples:**
```bash
# Create default configuration
python cli.py config --create --output migration_config.json

# Validate configuration
python cli.py config --validate --input migration_config.json
```

## Configuration

### Configuration File Format

Create a JSON configuration file to customize migration behavior:

```json
{
  "source_directories": [
    "/path/to/brobot/library/src/test",
    "/path/to/brobot/library-test/src/test/java"
  ],
  "target_directory": "/path/to/qontinui/tests/migrated",
  "preserve_structure": true,
  "enable_mock_migration": true,
  "diagnostic_level": "detailed",
  "parallel_execution": true,
  "comparison_mode": "behavioral",
  "java_test_patterns": [
    "*Test.java",
    "*Tests.java",
    "Test*.java"
  ],
  "exclude_patterns": [
    "*/target/*",
    "*/build/*",
    "*/.git/*"
  ]
}
```

### Configuration Options

- **`source_directories`**: List of directories containing Java tests
- **`target_directory`**: Directory where Python tests will be created
- **`preserve_structure`**: Whether to maintain directory structure
- **`enable_mock_migration`**: Enable Brobot to Qontinui mock conversion
- **`diagnostic_level`**: Logging detail level (`minimal`, `normal`, `detailed`)
- **`parallel_execution`**: Enable parallel processing
- **`comparison_mode`**: How to compare test behavior (`behavioral`, `output`, `both`)
- **`java_test_patterns`**: File patterns to identify Java tests
- **`exclude_patterns`**: Patterns to exclude from migration

### Using Configuration Files

```bash
# Use configuration file
python cli.py migrate --config migration_config.json

# Override config with command line options
python cli.py migrate --config migration_config.json --no-parallel
```

## Migration Workflow

### Phase 1: Discovery and Analysis

1. **Scan Source Directory**: Identifies Java test files
2. **Classify Tests**: Categorizes as unit vs integration tests
3. **Analyze Dependencies**: Maps Java imports to Python equivalents
4. **Detect Mock Usage**: Identifies Brobot and Spring mocks

### Phase 2: Translation and Generation

1. **Syntax Translation**: Converts Java syntax to Python
2. **Assertion Conversion**: Maps JUnit assertions to pytest
3. **Mock Migration**: Converts Brobot mocks to Qontinui equivalents
4. **Spring Integration**: Handles SpringBoot test patterns

### Phase 3: Execution and Validation

1. **Test Execution**: Runs migrated tests with pytest
2. **Result Collection**: Gathers execution results and metrics
3. **Failure Analysis**: Categorizes failures as migration vs code issues
4. **Diagnostic Reporting**: Provides detailed analysis and suggestions

### Typical Workflow Commands

```bash
# 1. Preview what will be migrated
python cli.py migrate brobot/library/src/test qontinui/tests/migrated --dry-run

# 2. Perform the migration
python cli.py migrate brobot/library/src/test qontinui/tests/migrated

# 3. Validate the migrated tests
python cli.py validate qontinui/tests/migrated

# 4. Generate comprehensive report
python cli.py report qontinui/tests/migrated --format html --output report.html
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors During Migration

**Problem**: `ModuleNotFoundError` or import-related errors

**Solution**:
```bash
# Run from the correct directory
cd qontinui/src/qontinui/test_migration

# Use absolute paths
python cli.py migrate /absolute/path/to/brobot/tests /absolute/path/to/qontinui/tests
```

#### 2. No Tests Discovered

**Problem**: Migration reports 0 tests found

**Solutions**:
- Verify source directory contains Java test files
- Check file patterns match your test naming convention
- Use custom patterns in configuration:

```json
{
  "java_test_patterns": [
    "*Test.java",
    "*Tests.java",
    "*IT.java",
    "Test*.java"
  ]
}
```

#### 3. Translation Failures

**Problem**: Tests fail to translate properly

**Solutions**:
- Check the migration report for specific error messages
- Use dry-run mode to identify problematic tests
- Review complex tests manually for custom handling

#### 4. Test Execution Failures

**Problem**: Migrated tests fail when executed

**Solutions**:
- Check if pytest is installed and accessible
- Verify Qontinui dependencies are available
- Review failure analysis in validation report

### Debugging Tips

1. **Use Verbose Mode**:
   ```bash
   python cli.py migrate source target -vv
   ```

2. **Check Logs**:
   The system provides detailed logging. Look for specific error messages.

3. **Dry Run First**:
   Always use `--dry-run` to preview migrations before executing.

4. **Incremental Migration**:
   Migrate small batches of tests to identify issues early.

## Advanced Usage

### Custom Migration Patterns

For specific Brobot patterns not handled automatically, you can:

1. **Extend Configuration**: Add custom dependency mappings
2. **Manual Post-Processing**: Edit generated tests for complex cases
3. **Iterative Refinement**: Use feedback from validation to improve translations

### Integration with Development Workflow

#### 1. Automated Migration in CI/CD

```bash
# In your CI/CD pipeline
python cli.py migrate brobot/tests qontinui/tests/migrated --output-format json > migration_results.json
python cli.py validate qontinui/tests/migrated --output-format json > validation_results.json
```

#### 2. Regular Migration Updates

```bash
#!/bin/bash
# update_tests.sh - Script to update migrated tests

echo "Updating migrated tests..."
python cli.py migrate brobot/library/src/test qontinui/tests/migrated
python cli.py validate qontinui/tests/migrated --report-file validation_$(date +%Y%m%d).json
echo "Migration complete. Check validation report for issues."
```

#### 3. Selective Migration

```bash
# Migrate only specific test files
python cli.py migrate brobot/library/src/test qontinui/tests/migrated \
  --java-test-patterns "*SpecificTest.java"
```

### Performance Optimization

For large test suites:

1. **Enable Parallel Processing**:
   ```bash
   python cli.py migrate source target --parallel
   ```

2. **Use Incremental Migration**:
   Migrate in batches to avoid memory issues with very large test suites.

3. **Exclude Unnecessary Files**:
   ```json
   {
     "exclude_patterns": [
       "*/target/*",
       "*/build/*",
       "**/generated-test-sources/**"
     ]
   }
   ```

## Best Practices

### 1. Migration Strategy

- **Start Small**: Begin with simple unit tests
- **Validate Early**: Run validation after each migration batch
- **Review Results**: Manually review complex test migrations
- **Iterate**: Use feedback to improve subsequent migrations

### 2. Quality Assurance

- **Compare Behavior**: Ensure migrated tests verify the same functionality
- **Test Coverage**: Maintain or improve test coverage during migration
- **Documentation**: Update test documentation to reflect Python patterns

### 3. Maintenance

- **Regular Updates**: Re-run migration when Brobot tests change
- **Monitor Failures**: Track and address common migration issues
- **Feedback Loop**: Use validation results to improve the migration system

## Support and Troubleshooting

If you encounter issues:

1. **Check the logs** for detailed error messages
2. **Use dry-run mode** to preview migrations
3. **Review the validation report** for failure analysis
4. **Start with simple tests** to verify the system works
5. **Check file paths** and permissions

The migration system is designed to handle most common Brobot test patterns automatically, but complex or custom patterns may require manual review and adjustment of the generated Python tests.
