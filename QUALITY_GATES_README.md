# Quality Gates Quick Reference

This is a quick reference guide for the qontinui quality gates system. For detailed documentation, see [docs/CI_CD_INTEGRATION.md](docs/CI_CD_INTEGRATION.md).

## Quick Setup

```bash
# Run the setup script
bash scripts/setup_quality_checks.sh
```

This will:
1. Install `qontinui-devtools`
2. Install `pre-commit`
3. Set up pre-commit hooks
4. Run initial checks

## Current Quality Baselines

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Circular Dependencies | 0 | 0 | ✅ Zero tolerance |
| Critical God Classes | 43 | ≤43 | ⚠️ Do not increase |
| Critical Security Issues | 20 | ≤20 | ⚠️ Do not increase |
| Type Coverage | 89.8% | ≥85% | ✅ Above target |
| Critical Race Conditions | 474 | ≤474 | ⚠️ Do not increase |

## Quick Commands

### Run All Quality Checks

```bash
python scripts/quality_gates.py --verbose
```

### Run Specific Checks

```bash
# Circular dependencies
qontinui-devtools import check src/

# God classes
qontinui-devtools architecture god-classes src/ --threshold 0.9

# Security scan
qontinui-devtools security scan src/ --severity critical

# Type coverage
qontinui-devtools types coverage src/ --min-coverage 85

# Race conditions
qontinui-devtools concurrency detect src/
```

### Generate Reports

```bash
# HTML report (comprehensive)
qontinui-devtools report src/ --output report.html

# JSON reports (for automation)
qontinui-devtools import check src/ --format json --output circular.json
```

### Pre-commit

```bash
# Run pre-commit manually
pre-commit run --all-files

# Skip pre-commit (not recommended)
git commit --no-verify -m "Your message"

# Switch to standard config (faster)
git checkout .pre-commit-config.yaml
pre-commit install --force
```

## CI/CD Workflows

### Quality Checks (On every PR/push)

**File**: `.github/workflows/quality-checks.yml`

**What it does**:
- Checks all quality gates
- Comments on PR with results
- Uploads reports as artifacts

### Comprehensive Analysis (Weekly)

**File**: `.github/workflows/comprehensive-analysis.yml`

**What it does**:
- Full codebase analysis
- Trend tracking
- Creates GitHub issue with summary
- Optional Slack notifications

## Common Issues

### "New circular dependencies detected"

```bash
# Identify the cycle
qontinui-devtools import check src/ --verbose

# Fix: Extract shared code to a new module or use dependency injection
```

### "New critical god classes detected"

```bash
# Identify the class
qontinui-devtools architecture god-classes src/ --threshold 0.9

# Fix: Split into smaller, focused classes
```

### "New critical security vulnerabilities"

```bash
# Review issues
qontinui-devtools security scan src/ --severity critical --verbose

# Fix: Move secrets to env vars, validate inputs, use safe functions
```

### "Type coverage dropped below 85%"

```bash
# Find files needing hints
qontinui-devtools types coverage src/ --show-missing

# Fix: Add type hints to functions
```

### "New critical race conditions detected"

```bash
# Identify issues
qontinui-devtools concurrency detect src/ --verbose

# Fix: Add locks, use thread-safe data structures
```

## Quality Gate Definitions

### Circular Dependencies (0 allowed)
- **What**: Import cycles between modules
- **Why**: Makes code harder to understand, can cause initialization issues
- **Fix**: Extract shared code, use dependency injection

### God Classes (≤43 critical)
A class with:
- \>500 lines of code, OR
- \>30 methods, OR
- LCOM > 0.9

- **What**: Classes that do too much
- **Why**: Violates Single Responsibility Principle, hard to test
- **Fix**: Split into smaller, focused classes

### Security (≤20 critical)
- **What**: Potential vulnerabilities (hardcoded secrets, unsafe functions)
- **Why**: Protects against security breaches
- **Fix**: Use env vars, validate inputs, use safe APIs

### Type Coverage (≥85%)
- **What**: Percentage of functions with type hints
- **Why**: Catches bugs early, improves documentation
- **Fix**: Add type hints to parameters and returns

### Race Conditions (≤474 critical)
- **What**: Shared state access without synchronization
- **Why**: Can cause intermittent bugs and data corruption
- **Fix**: Use locks, thread-safe data structures, immutable patterns

## Configuration Files

- **`.github/workflows/quality-checks.yml`**: PR quality checks
- **`.github/workflows/comprehensive-analysis.yml`**: Weekly analysis
- **`.pre-commit-config.yaml`**: Standard pre-commit (fast)
- **`.pre-commit-config-quality.yaml`**: Full quality gates (slow)
- **`pyproject.toml`**: Quality thresholds and configuration
- **`scripts/quality_gates.py`**: Quality gate enforcement script

## Resources

- **Full Documentation**: [docs/CI_CD_INTEGRATION.md](docs/CI_CD_INTEGRATION.md)
- **qontinui-devtools**: [../qontinui-devtools/README.md](../qontinui-devtools/README.md)
- **Pre-commit**: https://pre-commit.com/
- **GitHub Actions**: https://docs.github.com/actions

## Support

1. Check [docs/CI_CD_INTEGRATION.md](docs/CI_CD_INTEGRATION.md)
2. Review qontinui-devtools documentation
3. Open an issue on GitHub
4. Contact maintainers

---

**Pro Tips:**

- Run `python scripts/quality_gates.py` before pushing
- Use verbose mode to understand issues: `--verbose`
- Generate HTML reports for detailed analysis
- Update baselines only when quality genuinely improves
- Use standard pre-commit for regular commits, full quality gates for major changes

---

**Last Updated**: 2025-01-28
