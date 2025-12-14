# Qontinui CLI Tool - Implementation Summary

## Overview

A comprehensive command-line interface for running Qontinui visual automation workflows in CI/CD pipelines and headless environments. The CLI provides production-ready tools for testing, validation, and result reporting.

## Files Created

### Core CLI Module (`src/qontinui/cli/`)

1. **`__init__.py`** - Package initialization
2. **`main.py`** - CLI entry point with Click framework
3. **`exit_codes.py`** - Standard exit codes for CI/CD integration
4. **`utils.py`** - Utility functions (logging, colored output)
5. **`formatters.py`** - Result formatters (JSON, JUnit XML, TAP)
6. **`result_streamer.py`** - Stream results to remote servers

### Commands (`src/qontinui/cli/commands/`)

1. **`run.py`** - Execute workflows
2. **`test.py`** - Run workflows in test mode with reporting
3. **`validate.py`** - Validate configuration files

### Documentation

1. **`README.md`** - Complete CLI documentation
2. **`examples/github-actions.yml`** - GitHub Actions workflow example
3. **`examples/gitlab-ci.yml`** - GitLab CI configuration example
4. **`examples/Jenkinsfile`** - Jenkins pipeline example

### Tests (`tests/cli/`)

1. **`test_cli_basic.py`** - Basic CLI functionality tests
2. **`test_formatters.py`** - Result formatter tests

## Commands

### `qontinui run`

Execute a specific workflow from a JSON configuration.

```bash
qontinui run config.json --workflow "Login Workflow" --monitor 0 --timeout 300
```

**Options:**
- `--workflow, -w`: Workflow name or ID
- `--monitor, -m`: Monitor index (0-based)
- `--timeout, -t`: Maximum execution time in seconds
- `--verbose, -v`: Enable verbose logging
- `--headless`: Run without interactive prompts

### `qontinui test`

Run workflows in test mode with detailed reporting.

```bash
qontinui test config.json --format junit --output ./test-results/ --continue-on-failure
```

**Options:**
- `--workflow, -w`: Specific workflow to test (or all if omitted)
- `--format, -f`: Output format (json, junit, tap)
- `--output, -o`: Directory or file for results
- `--stream-to`: URL to stream results to
- `--monitor, -m`: Monitor index
- `--timeout, -t`: Maximum time per workflow
- `--verbose, -v`: Verbose output
- `--headless`: Headless mode for CI
- `--continue-on-failure`: Continue even if tests fail

### `qontinui validate`

Validate a configuration file without executing.

```bash
qontinui validate config.json --verbose
```

**Options:**
- `--verbose, -v`: Verbose validation output

## Output Formats

### JSON Format
- Detailed machine-readable results
- Full test information and metadata
- Suitable for custom processing

### JUnit XML Format
- Standard CI/CD format
- Compatible with Jenkins, GitLab CI, GitHub Actions
- Test result integration with CI systems

### TAP Format
- Test Anything Protocol
- Perl-compatible test runners
- Diagnostic information included

## Exit Codes

- `0`: Success - All tests passed
- `1`: Test Failure - One or more tests failed
- `2`: Configuration Error - Invalid configuration
- `3`: Execution Error - Runtime error (timeout, exception)

## Features

### CI/CD Integration
- Standard exit codes for pipeline control
- Multiple output formats
- Headless execution support
- Timeout handling
- Continue-on-failure option

### Result Streaming
- Real-time result streaming to remote servers
- Integration with qontinui-web
- Monitoring long-running test suites

### Robust Error Handling
- Detailed error messages
- Validation before execution
- Graceful timeout handling
- Exception tracebacks in verbose mode

### Multi-Monitor Support
- Run tests on specific monitors
- Monitor index configuration
- Coordinate offset handling

## Usage Examples

### Basic Execution

```bash
# Run first workflow in config
qontinui run automation.json

# Run specific workflow
qontinui run automation.json --workflow "Login Test"

# Run with timeout
qontinui run automation.json --timeout 120 --verbose
```

### Testing

```bash
# Run all workflows in test mode
qontinui test automation.json

# Save JUnit results
qontinui test automation.json --format junit --output ./results/

# Continue on failure
qontinui test automation.json --continue-on-failure
```

### CI/CD Pipelines

```bash
# Validate before testing
qontinui validate automation.json || exit 1

# Run tests with timeout and reporting
qontinui test automation.json \
  --format junit \
  --output ./test-results/ \
  --timeout 300 \
  --headless \
  --verbose
```

## CI/CD Examples

### GitHub Actions

```yaml
- name: Run Qontinui tests
  run: |
    poetry run qontinui test automation.json \
      --format junit \
      --output ./test-results/ \
      --headless

- name: Publish test results
  uses: EnricoMi/publish-unit-test-result-action@v2
  with:
    files: ./test-results/*.xml
```

### GitLab CI

```yaml
qontinui_tests:
  script:
    - qontinui test automation.json --format junit --output ./test-results/
  artifacts:
    reports:
      junit: test-results/*.xml
```

### Jenkins

```groovy
stage('Test') {
    steps {
        sh 'qontinui test automation.json --format junit --output ./test-results/'
    }
    post {
        always {
            junit 'test-results/*.xml'
        }
    }
}
```

## Installation

```bash
cd qontinui
poetry install
```

The `qontinui` command will be available after installation.

## Testing

```bash
# Run CLI tests
poetry run pytest tests/cli/ -v

# Run formatter tests
poetry run pytest tests/cli/test_formatters.py -v

# Run all tests
poetry run pytest tests/cli/
```

## Architecture

The CLI is built on Click, a Python framework for command-line interfaces:

- **Main entry point**: `qontinui.cli:main`
- **Commands**: Registered as Click command groups
- **Formatters**: Pluggable result formatters
- **Streamer**: HTTP-based result streaming
- **Exit codes**: Standard codes for CI/CD integration

## Implementation Notes

### Thread Safety
- Timeout implementation uses threading
- Graceful shutdown with runner.request_stop()
- 5-second grace period for cleanup

### Error Handling
- Try/except blocks with specific error types
- Verbose mode shows full tracebacks
- Colored output for errors/warnings/success

### Configuration Validation
- Pydantic-based validation via JSONRunner
- v2.0.0 format requirements enforced
- Detailed validation error messages

### Result Formatters
- XML generation with proper indentation
- TAP version 13 compliance
- JSON with ISO timestamps

## Future Enhancements

Potential improvements for future versions:

1. **Parallel Execution**: Run multiple workflows in parallel
2. **Result Caching**: Cache and compare results across runs
3. **Interactive Mode**: TUI for workflow selection
4. **Screenshot Diffing**: Visual regression testing
5. **Custom Reporters**: Plugin system for formatters
6. **Retry Logic**: Automatic retry on transient failures
7. **Result Database**: Store results in SQLite/PostgreSQL
8. **Slack/Discord Integration**: Post results to chat platforms

## Dependencies

- `click>=8.0`: CLI framework
- `qontinui`: Core automation library
- Standard library: `json`, `xml.etree.ElementTree`, `threading`, `time`

Optional for streaming:
- `requests`: HTTP requests for result streaming

## License

MIT License - same as Qontinui core library

## Author

Joshua Spinak

## Contributing

This is part of the Qontinui project. See the main Qontinui repository for contribution guidelines.
