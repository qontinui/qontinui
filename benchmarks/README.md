# Qontinui Performance Benchmarks

This directory contains performance benchmarking tools for the Qontinui automation system.

## Overview

The benchmarks measure five key areas of system performance:

1. **Configuration Loading** - JSON parsing and validation performance
2. **Workflow Execution** - Action execution overhead and orchestration
3. **Image Finding** - Template matching and pattern recognition speed
4. **Export/Import** - Configuration serialization/deserialization
5. **Threading** - Thread safety overhead and lock contention

## Quick Start

### Run All Benchmarks

```bash
cd /mnt/c/Users/jspin/Documents/qontinui_parent/qontinui
python -m benchmarks.run_all_benchmarks
```

This will:
- Run all benchmark suites
- Generate a JSON report at `/tmp/benchmark_results.json`
- Generate an HTML report at `/tmp/benchmark_report.html`

### Run Individual Benchmarks

```bash
# Configuration loading
python -m benchmarks.benchmark_config_loading

# Workflow execution
python -m benchmarks.benchmark_workflow_execution

# Image finding
python -m benchmarks.benchmark_image_finding

# Export/Import
python -m benchmarks.benchmark_export_import

# Threading
python -m benchmarks.benchmark_threading
```

## Benchmark Scripts

### 1. benchmark_config_loading.py

Measures configuration loading performance with various config sizes.

**Tests:**
- Small config: 10 workflows, 50 actions
- Medium config: 50 workflows, 250 actions
- Large config: 100 workflows, 1000 actions
- Very large config: 500 workflows, 5000 actions

**Metrics:**
- Parse time (ms)
- Memory usage (MB)
- Actions per second

**Target:** < 1 second for typical configs

### 2. benchmark_workflow_execution.py

Measures workflow execution overhead.

**Tests:**
- Simple workflow: 5 actions
- Medium workflow: 20 actions
- Complex workflow: 50 actions
- Parallel workflows: 10 concurrent workflows

**Metrics:**
- Execution time (ms)
- Overhead per action (ms)
- Parallel speedup

**Target:** < 100ms overhead per action

### 3. benchmark_image_finding.py

Measures image template matching performance.

**Tests:**
- Best case: Image present, quick find
- Average case: Image present, multiple patterns
- Worst case: Image not present, timeout
- Multiple images: Sequential finds
- Concurrent finds: Parallel find operations

**Metrics:**
- Find time (ms)
- Pattern matching FPS
- Memory usage

**Target:** < 500ms per find operation

### 4. benchmark_export_import.py

Measures configuration serialization performance.

**Tests:**
- Export small config to JSON
- Export large config to JSON
- Import small config from JSON
- Import large config from JSON
- Round-trip export/import

**Metrics:**
- Serialization time (ms)
- Deserialization time (ms)
- File size (KB)
- Memory usage (MB)

**Target:** < 2 seconds for large configs

### 5. benchmark_threading.py

Measures thread safety overhead.

**Tests:**
- Lock overhead vs no lock
- Lock contention with 2, 10, 50 threads
- Lock acquisition time
- Scalability with varying thread counts

**Metrics:**
- Threading overhead (%)
- Lock acquisition time (us)
- Scalability efficiency

**Target:** < 10% overhead vs single-threaded

## Performance Grading

Results are graded on the following scale:

- **Excellent**: Within target performance goals
- **Good**: Within 1.5x target
- **Fair**: Within 2x target
- **Poor**: Within 5x target
- **Critical**: Exceeds 5x target

## Understanding Results

### Configuration Loading

```
Configuration Loading:
  Small config:    87ms   [Excellent]
  Medium config:   245ms  [Excellent]
  Large config:    892ms  [Excellent]
```

- Fast parsing indicates efficient JSON handling
- Memory usage should scale linearly with config size
- Look for O(n²) behavior if performance degrades quadratically

### Workflow Execution

```
Workflow Execution:
  Simple workflow:   412ms  [Excellent]
  Action overhead:   73ms   [Excellent]
```

- Low overhead per action is critical for responsiveness
- Parallel workflows should show good speedup

### Image Finding

```
Image Finding:
  Best case:        156ms  [Excellent]
  Pattern FPS:      15.2   [Excellent]
```

- Fast find times indicate efficient template matching
- Higher FPS means better real-time performance

### Export/Import

```
Export/Import:
  Export large:     1432ms [Excellent]
  Import large:     1876ms [Excellent]
```

- Fast serialization enables quick saves
- Fast deserialization enables quick loads

### Threading

```
Threading:
  10 threads:       8.7%   [Excellent]
  Lock time:        0.3ms  [Excellent]
```

- Low overhead indicates efficient lock implementation
- Good scalability shows minimal contention

## Troubleshooting

### Benchmarks Fail with Import Errors

Make sure you're running from the correct directory:

```bash
cd /mnt/c/Users/jspin/Documents/qontinui_parent/qontinui
python -m benchmarks.run_all_benchmarks
```

### Mock Classes Not Working

The benchmarks use mock classes to simulate real operations. They don't require actual Qontinui components to run.

### Results Vary Between Runs

Some variation is normal due to:
- System load
- CPU throttling
- Disk I/O
- Background processes

Run benchmarks multiple times and average the results for more stable measurements.

## Integration with CI/CD

To add benchmarks to CI/CD pipeline:

```yaml
# .github/workflows/benchmarks.yml
name: Performance Benchmarks

on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Run benchmarks
        run: |
          cd qontinui
          python -m benchmarks.run_all_benchmarks
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: benchmark-results
          path: /tmp/benchmark_*.json
```

## Related Documentation

- `/mnt/c/Users/jspin/Documents/qontinui_parent/PERFORMANCE_BENCHMARKS.md` - Full benchmarking documentation
- `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/tests/integration/test_performance_regression.py` - Performance regression tests

## Version History

- 2025-10-29: Initial benchmark suite created
