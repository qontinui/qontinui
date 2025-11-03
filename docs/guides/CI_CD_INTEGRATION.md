# CI/CD Integration Guide

This document describes the comprehensive CI/CD integration for the qontinui project, including automated quality checks, pre-commit hooks, and continuous monitoring.

## Table of Contents

- [Overview](#overview)
- [Quality Gates](#quality-gates)
- [GitHub Actions Workflows](#github-actions-workflows)
- [Pre-Commit Hooks](#pre-commit-hooks)
- [Local Development](#local-development)
- [Interpreting Reports](#interpreting-reports)
- [Fixing Common Issues](#fixing-common-issues)
- [Updating Baselines](#updating-baselines)
- [Configuration](#configuration)

## Overview

The qontinui project uses `qontinui-devtools` to enforce code quality standards through automated checks at multiple stages:

1. **Pre-commit**: Local checks before committing code
2. **Pull Request**: Comprehensive checks on every PR
3. **Weekly Analysis**: Full codebase analysis with trend tracking

### Current Baselines (as of analysis)

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Circular Dependencies | 0 | 0 | ✅ Zero tolerance |
| Critical God Classes | 43 | 43 | ⚠️ Do not increase |
| Critical Security Issues | 20 | 20 | ⚠️ Do not increase |
| Type Coverage | 89.8% | 85% | ✅ Above target |
| Critical Race Conditions | 474 | 474 | ⚠️ Do not increase |

## Quality Gates

### 1. Circular Dependencies

**Threshold**: 0 (zero tolerance)

Circular dependencies have been eliminated from the codebase. Any new circular dependencies will fail the build.

**What it checks**:
- Import cycles between modules
- Direct and transitive dependencies

**Why it matters**:
- Makes code harder to understand and maintain
- Can cause initialization issues
- Indicates poor module separation

### 2. God Classes

**Threshold**: 43 critical god classes

A "god class" is a class that does too much. Critical god classes have:
- More than 500 lines of code, OR
- More than 30 methods, OR
- LCOM (Lack of Cohesion in Methods) > 0.9

**What it checks**:
- Class size (lines of code)
- Number of methods
- Cohesion metrics (LCOM)

**Why it matters**:
- Violates Single Responsibility Principle
- Harder to test and maintain
- Increases coupling

### 3. Security Vulnerabilities

**Threshold**: 20 critical issues

Security scan identifies potential vulnerabilities including:
- Hardcoded secrets
- Unsafe deserialization
- SQL injection risks
- Path traversal vulnerabilities
- Insecure random number generation

**What it checks**:
- Static analysis for common security patterns
- High and critical severity issues

**Why it matters**:
- Protects against security breaches
- Ensures safe handling of user data
- Prevents common attack vectors

### 4. Type Coverage

**Threshold**: 85% minimum

Type hints improve code quality and enable better IDE support and static analysis.

**What it checks**:
- Percentage of functions with type hints
- Parameter and return type annotations
- Quality of type hints

**Why it matters**:
- Catches bugs before runtime
- Improves code documentation
- Enables better IDE autocomplete

### 5. Race Conditions

**Threshold**: 474 critical race conditions

Race conditions occur when multiple threads or async operations access shared state without proper synchronization.

**What it checks**:
- Shared state access patterns
- Unprotected modifications
- Async/thread race conditions

**Why it matters**:
- Can cause intermittent bugs
- Hard to reproduce and debug
- Can lead to data corruption

## GitHub Actions Workflows

### Quality Checks Workflow

**File**: `.github/workflows/quality-checks.yml`

**Triggers**:
- Push to `main` or `develop`
- Pull requests to `main` or `develop`

**What it does**:
1. Runs all quality gate checks
2. Generates JSON reports
3. Uploads artifacts
4. Comments on PR with summary

**Artifacts**: Quality reports are saved for 30 days

### Comprehensive Analysis Workflow

**File**: `.github/workflows/comprehensive-analysis.yml`

**Triggers**:
- Every Sunday at midnight UTC (scheduled)
- Manual trigger via GitHub Actions UI

**What it does**:
1. Runs comprehensive analysis of entire codebase
2. Generates HTML and JSON reports
3. Tracks trends over time
4. Creates GitHub issue with summary
5. (Optional) Posts to Slack/Discord

**Artifacts**: Weekly reports are saved for 90 days for trend analysis

## Pre-Commit Hooks

### Setup

The project includes two pre-commit configurations:

1. **Standard** (`.pre-commit-config.yaml`): Basic formatting and linting
2. **Quality Gates** (`.pre-commit-config-quality.yaml`): Full quality checks

#### Option 1: Standard Pre-commit (Recommended for most developers)

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install
```

This runs:
- Black (formatter)
- Ruff (linter)
- Basic checks (trailing whitespace, YAML syntax, etc.)

#### Option 2: Full Quality Gates (Recommended for maintainers)

```bash
# Install qontinui-devtools first
pip install qontinui-devtools

# Install pre-commit
pip install pre-commit

# Use quality gates configuration
cp .pre-commit-config-quality.yaml .pre-commit-config.yaml
pre-commit install
```

This runs all standard checks PLUS:
- Circular dependency detection
- God class analysis
- Security scan
- Type coverage check
- Race condition detection

**Note**: Full quality gates can take 30-60 seconds. Use for final commits or when changing architecture.

### Skipping Pre-commit Hooks

If you need to skip pre-commit hooks (not recommended):

```bash
git commit --no-verify -m "Your message"
```

## Local Development

### Installing qontinui-devtools

```bash
# From PyPI (once published)
pip install qontinui-devtools

# Or from source
cd qontinui_parent/qontinui-devtools
pip install -e .
```

### Running Checks Locally

#### Check Everything

```bash
# Run quality gates script
python scripts/quality_gates.py --verbose

# Or use qontinui-devtools directly
qontinui-devtools report src/ --output report.html
```

#### Check Specific Issues

```bash
# Circular dependencies
qontinui-devtools import check src/

# God classes
qontinui-devtools architecture god-classes src/ --threshold 0.9

# Security
qontinui-devtools security scan src/ --severity critical

# Type coverage
qontinui-devtools types coverage src/ --min-coverage 85

# Race conditions
qontinui-devtools concurrency detect src/
```

#### Generate Reports

```bash
# HTML report
qontinui-devtools report src/ --output report.html

# JSON reports for automation
qontinui-devtools import check src/ --format json --output circular.json
qontinui-devtools architecture god-classes src/ --format json --output gods.json
```

## Interpreting Reports

### Circular Dependency Report

```json
{
  "cycles": [
    {
      "cycle": ["module_a", "module_b", "module_a"],
      "length": 2
    }
  ]
}
```

**How to fix**:
1. Identify the import causing the cycle
2. Move shared code to a new module
3. Use dependency injection
4. Apply lazy imports (use with caution)

### God Class Report

```json
{
  "god_classes": [
    {
      "class_name": "ActionExecutor",
      "file": "src/qontinui/actions/executor.py",
      "num_lines": 650,
      "num_methods": 35,
      "lcom": 0.92,
      "severity": "critical"
    }
  ]
}
```

**How to fix**:
1. Extract cohesive groups of methods into separate classes
2. Use composition instead of inheritance
3. Apply Strategy or Command patterns
4. Split into multiple smaller classes

### Security Report

```json
{
  "vulnerabilities": [
    {
      "file": "src/qontinui/config.py",
      "line": 42,
      "severity": "critical",
      "message": "Hardcoded password detected",
      "recommendation": "Use environment variables or secrets manager"
    }
  ]
}
```

**How to fix**:
1. Move secrets to environment variables
2. Use proper cryptographic functions
3. Validate and sanitize inputs
4. Use parameterized queries

### Type Coverage Report

```json
{
  "overall_coverage": {
    "coverage_percentage": 89.8,
    "total_functions": 523,
    "functions_with_hints": 470,
    "functions_without_hints": 53
  },
  "files_needing_hints": [
    {
      "file": "src/qontinui/utils/helpers.py",
      "coverage": 60.0,
      "missing_hints": 12
    }
  ]
}
```

**How to fix**:
1. Add type hints to function parameters
2. Add return type annotations
3. Use `typing` module for complex types
4. Run `mypy` to validate

### Race Condition Report

```json
{
  "race_conditions": [
    {
      "file": "src/qontinui/state/manager.py",
      "line": 85,
      "type": "shared_state_access",
      "severity": "critical",
      "details": "Shared state accessed without lock"
    }
  ]
}
```

**How to fix**:
1. Use locks (`threading.Lock`, `asyncio.Lock`)
2. Use thread-safe data structures
3. Apply immutable data patterns
4. Use async context managers

## Fixing Common Issues

### "New circular dependencies detected"

1. **Identify the cycle**:
   ```bash
   qontinui-devtools import check src/ --verbose
   ```

2. **Break the cycle**:
   - Extract shared code to a new module
   - Use dependency injection
   - Move imports inside functions (temporary)
   - Restructure module hierarchy

3. **Verify fix**:
   ```bash
   qontinui-devtools import check src/
   ```

### "New critical god classes detected"

1. **Identify the class**:
   ```bash
   qontinui-devtools architecture god-classes src/ --threshold 0.9
   ```

2. **Refactor strategies**:
   - Extract related methods into new classes
   - Use composition over inheritance
   - Apply design patterns (Strategy, Command, etc.)
   - Split into domain-specific classes

3. **Monitor metrics**:
   ```bash
   qontinui-devtools architecture god-classes src/ --class "YourClass"
   ```

### "New critical security vulnerabilities"

1. **Review the issue**:
   ```bash
   qontinui-devtools security scan src/ --severity critical --verbose
   ```

2. **Common fixes**:
   - Move secrets to environment variables
   - Use `secrets` module for random generation
   - Validate and sanitize all inputs
   - Use parameterized SQL queries
   - Avoid `eval()` and `exec()`

3. **Verify fix**:
   ```bash
   qontinui-devtools security scan src/
   ```

### "Type coverage dropped below 85%"

1. **Find files needing hints**:
   ```bash
   qontinui-devtools types coverage src/ --show-missing
   ```

2. **Add type hints**:
   ```python
   # Before
   def process_data(data, timeout):
       return data.process()

   # After
   def process_data(data: Dict[str, Any], timeout: float) -> Result:
       return data.process()
   ```

3. **Verify**:
   ```bash
   mypy src/
   qontinui-devtools types coverage src/
   ```

### "New critical race conditions detected"

1. **Identify the issue**:
   ```bash
   qontinui-devtools concurrency detect src/ --verbose
   ```

2. **Common fixes**:
   ```python
   # Before
   class StateManager:
       def update(self):
           self.state = new_value  # Not thread-safe

   # After
   class StateManager:
       def __init__(self):
           self._lock = Lock()

       def update(self):
           with self._lock:
               self.state = new_value
   ```

3. **Verify**:
   ```bash
   qontinui-devtools concurrency detect src/
   ```

## Updating Baselines

As you intentionally refactor and improve code quality, you may need to update baselines.

### When to Update Baselines

✅ **Do update** when:
- You've refactored code and reduced issues
- You've split god classes into smaller classes
- You've fixed security vulnerabilities
- You've improved type coverage

❌ **Don't update** when:
- Adding new features increased issues
- You want to "make CI pass"
- Issues increased without good reason

### How to Update Baselines

1. **Verify improvements**:
   ```bash
   # Run all checks
   python scripts/quality_gates.py --verbose
   ```

2. **Update `pyproject.toml`**:
   ```toml
   [tool.qontinui-devtools]
   # Update these values to new (lower) numbers
   baseline_god_classes_critical = 35  # Was 43
   baseline_race_conditions_critical = 420  # Was 474
   baseline_security_critical = 15  # Was 20
   ```

3. **Update GitHub workflows**:
   - Edit `.github/workflows/quality-checks.yml`
   - Update threshold values in each check step

4. **Document the change**:
   ```bash
   git commit -m "chore: update quality baselines after refactoring

   - Reduced critical god classes from 43 to 35
   - Fixed 54 race conditions
   - Resolved 5 security issues"
   ```

### Baseline Update Checklist

- [ ] Run full quality check locally
- [ ] Update `pyproject.toml` thresholds
- [ ] Update GitHub workflow thresholds
- [ ] Update this documentation
- [ ] Document changes in commit message
- [ ] Create PR with baseline updates

## Configuration

### pyproject.toml

Main configuration file for quality thresholds:

```toml
[tool.qontinui-devtools]
max_god_class_lines = 500
max_god_class_methods = 30
max_lcom = 0.9
min_type_coverage = 85.0
allow_circular_dependencies = false

baseline_god_classes_critical = 43
baseline_race_conditions_critical = 474
baseline_security_critical = 20

[tool.qontinui-devtools.security]
fail_on_critical = true
fail_on_high = false
ignore_patterns = ["*/tests/*", "*/examples/*"]

[tool.qontinui-devtools.race-conditions]
fail_on_new_critical = true
baseline_critical_count = 474
```

### Environment Variables

For Slack integration (optional):

```bash
# Add to GitHub repository secrets
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

### GitHub Actions Permissions

Required permissions for workflows:
- `contents: read` - Read repository contents
- `pull-requests: write` - Comment on PRs
- `issues: write` - Create weekly summary issues

## Best Practices

### For Developers

1. **Run checks before pushing**:
   ```bash
   python scripts/quality_gates.py
   ```

2. **Fix issues incrementally**: Don't try to fix everything at once

3. **Use verbose mode**: Understand what's being flagged
   ```bash
   qontinui-devtools security scan src/ --verbose
   ```

4. **Ask for help**: Quality issues can indicate design problems

### For Maintainers

1. **Monitor trends**: Review weekly reports for patterns

2. **Update baselines carefully**: Only when code quality genuinely improves

3. **Document exceptions**: If you skip a check, document why

4. **Enforce on critical paths**: Require quality checks on main/develop branches

### For CI/CD

1. **Cache dependencies**: Speed up workflows with caching

2. **Fail fast**: Stop on critical issues to save resources

3. **Provide clear feedback**: Help developers understand failures

4. **Track trends**: Use weekly reports to identify improvements

## Troubleshooting

### Pre-commit hooks are slow

Use the standard configuration for regular commits:
```bash
cp .pre-commit-config.yaml.bak .pre-commit-config.yaml
pre-commit install --force
```

Run full quality checks manually before pushing:
```bash
python scripts/quality_gates.py
```

### False positives in security scan

Add patterns to ignore in `pyproject.toml`:
```toml
[tool.qontinui-devtools.security]
ignore_patterns = [
    "*/tests/*",
    "*/examples/*",
    "**/specific_file.py"
]
```

### Type coverage calculation seems wrong

Ensure mypy is configured correctly:
```bash
mypy --config-file=pyproject.toml src/
```

### Workflow fails on dependencies

Check that `qontinui-devtools` is properly installed:
```yaml
- name: Install qontinui-devtools
  run: pip install qontinui-devtools
```

## Resources

- [qontinui-devtools Documentation](../qontinui-devtools/README.md)
- [Pre-commit Documentation](https://pre-commit.com/)
- [GitHub Actions Documentation](https://docs.github.com/actions)
- [Type Hints PEP 484](https://peps.python.org/pep-0484/)

## Support

For issues or questions:
1. Check this documentation
2. Review qontinui-devtools README
3. Open an issue on GitHub
4. Contact the maintainers

---

**Last Updated**: 2025-01-28
**Version**: 1.0
