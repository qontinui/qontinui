# Qontinui CLI - Quick Start Guide

## Installation

```bash
cd qontinui
poetry install
```

After installation, the `qontinui` command will be available.

## Verify Installation

```bash
qontinui --version
qontinui --help
```

## Basic Usage

### 1. Validate Your Configuration

Before running tests, validate your configuration file:

```bash
qontinui validate automation.json
```

With verbose output:

```bash
qontinui validate automation.json --verbose
```

### 2. Run a Workflow

Execute a specific workflow:

```bash
qontinui run automation.json --workflow "My Workflow"
```

Run the first workflow (default):

```bash
qontinui run automation.json
```

With timeout:

```bash
qontinui run automation.json --workflow "Login Test" --timeout 120
```

### 3. Test Mode

Run all workflows and generate a test report:

```bash
qontinui test automation.json
```

Generate JUnit XML for CI/CD:

```bash
qontinui test automation.json --format junit --output ./test-results/
```

Continue even if some tests fail:

```bash
qontinui test automation.json --continue-on-failure
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Qontinui Tests

on: [push]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install poetry
          poetry install

      - name: Run tests
        run: |
          xvfb-run poetry run qontinui test automation.json \
            --format junit \
            --output ./test-results/ \
            --headless

      - name: Publish results
        uses: EnricoMi/publish-unit-test-result-action@v2
        if: always()
        with:
          files: ./test-results/*.xml
```

### GitLab CI Example

```yaml
qontinui_tests:
  stage: test
  image: python:3.12

  before_script:
    - apt-get update && apt-get install -y xvfb
    - pip install poetry
    - poetry install

  script:
    - xvfb-run poetry run qontinui test automation.json \
        --format junit \
        --output ./test-results/ \
        --headless

  artifacts:
    reports:
      junit: test-results/*.xml
```

## Output Formats

### JSON (Default)
```bash
qontinui test automation.json --format json --output results.json
```

Produces detailed machine-readable results with full test information.

### JUnit XML
```bash
qontinui test automation.json --format junit --output test-results/
```

Compatible with Jenkins, GitLab CI, GitHub Actions, and other CI/CD systems.

### TAP (Test Anything Protocol)
```bash
qontinui test automation.json --format tap --output results.tap
```

Compatible with Perl-style test runners.

## Exit Codes

The CLI uses standard exit codes:

- `0` - Success (all tests passed)
- `1` - Test failure (one or more tests failed)
- `2` - Configuration error (invalid config file)
- `3` - Execution error (timeout, exception)

Use these in your CI/CD scripts:

```bash
qontinui validate automation.json || exit 2
qontinui test automation.json || exit 1
```

## Common Options

### Headless Mode

For CI/CD environments:

```bash
qontinui run automation.json --headless
```

Disables interactive prompts and visual output.

### Verbose Output

For debugging:

```bash
qontinui test automation.json --verbose
```

Shows detailed logs and execution information.

### Timeout

Prevent hanging builds:

```bash
qontinui run automation.json --timeout 300
```

Stops execution after 300 seconds.

### Monitor Selection

Run on a specific monitor:

```bash
qontinui run automation.json --monitor 1
```

Monitor index is 0-based (0 = primary monitor).

## Tips & Tricks

### Combine with Make

Create a Makefile:

```makefile
.PHONY: validate test ci

validate:
    poetry run qontinui validate automation.json --verbose

test: validate
    poetry run qontinui test automation.json \
        --format junit \
        --output ./test-results/ \
        --timeout 600 \
        --continue-on-failure

ci: test
    # Upload results or notify
```

Run with:

```bash
make test
```

### Pre-commit Hook

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
poetry run qontinui validate automation.json || {
    echo "Configuration validation failed"
    exit 1
}
```

### Parallel Execution

Run different workflows in parallel:

```bash
qontinui run config1.json --workflow "Test A" &
qontinui run config2.json --workflow "Test B" &
wait
```

## Troubleshooting

### Config Validation Fails

Check that your config:
1. Is valid JSON
2. Uses v2.0.0 format
3. Has workflows with "connections" field
4. Has workflow "version" field
5. Has at least one state

### Tests Hang in CI

Use `--headless` and `--timeout`:

```bash
qontinui test automation.json --headless --timeout 300
```

### Monitor Not Found

Check available monitors:

```bash
qontinui validate automation.json --verbose
# Shows detected monitors
```

### Import Errors

Ensure Qontinui is installed:

```bash
poetry install
poetry run qontinui --version
```

## More Information

- Full CLI documentation: `src/qontinui/cli/README.md`
- CI/CD examples: `src/qontinui/cli/examples/`
- Qontinui documentation: https://qontinui.github.io

## Support

For issues and questions:
- GitHub Issues: https://github.com/qontinui/qontinui/issues
- Documentation: https://qontinui.github.io/qontinui
