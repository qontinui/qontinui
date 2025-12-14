# Qontinui CLI

Command-line interface for running Qontinui visual automation workflows in CI/CD pipelines and headless environments.

## Installation

```bash
# Install from source
cd qontinui
poetry install

# The CLI will be available as 'qontinui'
qontinui --help
```

## Commands

### `qontinui run`

Run a specific workflow from a JSON configuration file.

```bash
# Run the first workflow in the config
qontinui run automation.json

# Run a specific workflow by name
qontinui run automation.json --workflow "Login Workflow"

# Run on a specific monitor with timeout
qontinui run automation.json --workflow "Test" --monitor 1 --timeout 300

# Verbose headless execution for CI
qontinui run automation.json --verbose --headless
```

**Options:**
- `--workflow, -w`: Workflow name or ID to execute
- `--monitor, -m`: Monitor index (0-based, default: 0)
- `--timeout, -t`: Maximum execution time in seconds
- `--verbose, -v`: Enable verbose logging
- `--headless`: Run in headless mode (no interactive prompts)

### `qontinui test`

Run workflows in test mode with detailed result reporting.

```bash
# Run all workflows and output JSON results
qontinui test automation.json

# Run specific workflow with JUnit output
qontinui test automation.json --workflow "Login Test" --format junit

# Save results to file
qontinui test automation.json --format junit --output ./test-results/

# Stream results to qontinui-web
qontinui test automation.json --stream-to http://localhost:8000/api/v1/results

# Run all workflows with timeout, continue on failure
qontinui test automation.json --timeout 60 --continue-on-failure
```

**Options:**
- `--workflow, -w`: Workflow to test (runs all if not specified)
- `--format, -f`: Output format (json, junit, tap)
- `--output, -o`: Directory or file path for results
- `--stream-to`: URL to stream results to
- `--monitor, -m`: Monitor index (0-based)
- `--timeout, -t`: Maximum execution time per workflow
- `--verbose, -v`: Enable verbose output
- `--headless`: Run in headless mode
- `--continue-on-failure`: Continue even if a workflow fails

### `qontinui validate`

Validate a JSON configuration file without executing it.

```bash
# Validate a configuration
qontinui validate automation.json

# Verbose validation with detailed output
qontinui validate automation.json --verbose
```

**Options:**
- `--verbose, -v`: Enable verbose validation output

## Output Formats

### JSON Format

Detailed machine-readable results with full test information:

```json
{
  "summary": {
    "total_tests": 3,
    "passed": 2,
    "failed": 1,
    "total_duration": 45.67,
    "config_file": "automation.json",
    "timestamp": 1703001234.567
  },
  "tests": [
    {
      "workflow_id": "wf-123",
      "workflow_name": "Login Test",
      "success": true,
      "duration": 12.34,
      "error": null,
      "start_time": 1703001234.567
    }
  ],
  "timestamp_iso": "2023-12-19T12:34:56.789"
}
```

### JUnit XML Format

Standard format compatible with CI/CD systems (Jenkins, GitLab CI, GitHub Actions):

```xml
<?xml version='1.0' encoding='UTF-8'?>
<testsuites name="Qontinui Test Suite" tests="3" failures="1" time="45.67">
  <testsuite name="Qontinui Workflows" tests="3" failures="1" time="45.67">
    <testcase name="Login Test" classname="qontinui.wf-123" time="12.34"/>
    <testcase name="Failed Test" classname="qontinui.wf-456" time="5.67">
      <failure message="Workflow execution failed" type="WorkflowFailure">
        Workflow did not complete successfully
      </failure>
    </testcase>
  </testsuite>
</testsuites>
```

### TAP Format

Test Anything Protocol for Perl-compatible test runners:

```tap
TAP version 13
1..3
ok 1 - Login Test
  ---
  duration_ms: 12340
  workflow_id: wf-123
  ...
not ok 2 - Failed Test
  ---
  duration_ms: 5670
  workflow_id: wf-456
  error: Workflow execution failed
  ...

# Test Summary
# Total: 3
# Passed: 2
# Failed: 1
# Duration: 45.67s
```

## Exit Codes

The CLI uses standard exit codes for CI/CD integration:

- `0`: Success - All tests passed
- `1`: Test Failure - One or more tests failed
- `2`: Configuration Error - Invalid or missing configuration
- `3`: Execution Error - Runtime error (timeout, exception)

## CI/CD Integration Examples

### GitHub Actions

```yaml
name: Qontinui Tests

on: [push, pull_request]

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

      - name: Run Qontinui tests
        run: |
          poetry run qontinui test automation.json \
            --format junit \
            --output ./test-results/ \
            --headless \
            --verbose

      - name: Publish test results
        uses: EnricoMi/publish-unit-test-result-action@v2
        if: always()
        with:
          files: ./test-results/*.xml
```

### GitLab CI

```yaml
qontinui_tests:
  stage: test
  image: python:3.12

  before_script:
    - pip install poetry
    - poetry install

  script:
    - poetry run qontinui test automation.json --format junit --output ./test-results/ --headless

  artifacts:
    when: always
    reports:
      junit: ./test-results/*.xml
```

### Jenkins

```groovy
pipeline {
    agent any

    stages {
        stage('Test') {
            steps {
                sh '''
                    pip install poetry
                    poetry install
                    poetry run qontinui test automation.json \
                        --format junit \
                        --output ./test-results/ \
                        --headless \
                        --verbose
                '''
            }
        }
    }

    post {
        always {
            junit 'test-results/*.xml'
        }
    }
}
```

### CircleCI

```yaml
version: 2.1

jobs:
  test:
    docker:
      - image: python:3.12

    steps:
      - checkout

      - run:
          name: Install dependencies
          command: |
            pip install poetry
            poetry install

      - run:
          name: Run tests
          command: |
            poetry run qontinui test automation.json \
              --format junit \
              --output ./test-results/ \
              --headless

      - store_test_results:
          path: ./test-results
```

## Result Streaming

Stream test results to qontinui-web in real-time for monitoring:

```bash
qontinui test automation.json \
  --stream-to http://your-server:8000/api/v1/test-results \
  --format json
```

This is useful for:
- Real-time monitoring of long-running test suites
- Centralized test result storage
- Integration with qontinui-web dashboard

## Headless Mode

For CI/CD environments, use `--headless` to:
- Disable interactive prompts
- Suppress visual output
- Ensure non-blocking execution

```bash
qontinui test automation.json --headless --verbose
```

## Best Practices

### Timeouts

Always set timeouts in CI/CD to prevent hanging builds:

```bash
qontinui test automation.json --timeout 300  # 5 minutes per workflow
```

### Continue on Failure

For test suites, use `--continue-on-failure` to run all tests:

```bash
qontinui test automation.json --continue-on-failure
```

### Verbose Output

Enable verbose logging for debugging CI failures:

```bash
qontinui test automation.json --verbose
```

### Multi-Monitor Testing

Test workflows on different monitors:

```bash
# Test on primary monitor (0)
qontinui test automation.json --monitor 0

# Test on secondary monitor (1)
qontinui test automation.json --monitor 1
```

## Troubleshooting

### Configuration Validation

Always validate configs before running:

```bash
qontinui validate automation.json --verbose
```

### Debugging Failed Tests

Run individual workflows with verbose output:

```bash
qontinui run automation.json --workflow "Problematic Test" --verbose
```

### CI/CD Hanging

If tests hang in CI:
1. Add `--timeout` to limit execution time
2. Use `--headless` to avoid interactive prompts
3. Check logs with `--verbose`
4. Verify display server is available (Xvfb on Linux)

## Advanced Usage

### Custom Test Reports

Combine formats for different audiences:

```bash
# JSON for processing
qontinui test automation.json --format json --output ./results.json

# JUnit for CI system
qontinui test automation.json --format junit --output ./junit.xml

# TAP for test runners
qontinui test automation.json --format tap --output ./results.tap
```

### Parallel Execution

Run different workflows in parallel (requires separate configs):

```bash
# Terminal 1
qontinui run config1.json --workflow "Test A" --monitor 0 &

# Terminal 2
qontinui run config2.json --workflow "Test B" --monitor 1 &

wait  # Wait for both to complete
```

### Integration with Make

```makefile
.PHONY: test validate

validate:
	poetry run qontinui validate automation.json --verbose

test: validate
	poetry run qontinui test automation.json \
		--format junit \
		--output ./test-results/ \
		--timeout 600 \
		--continue-on-failure \
		--headless \
		--verbose

ci-test: test
	# Upload results to server
	curl -X POST http://server/results -F "file=@test-results/*.xml"
```
