# Qontinui CI/CD Templates

This directory contains ready-to-use GitHub Actions workflow templates for integrating Qontinui tests into your CI/CD pipeline.

## Quick Start

1. **Copy the setup action to your repository:**
   ```bash
   mkdir -p .github/actions
   cp -r ci-templates/github-actions/setup-qontinui .github/actions/
   ```

2. **Choose and copy a workflow template:**
   ```bash
   cp ci-templates/github-actions/qontinui-test.yml .github/workflows/
   ```

3. **Create a Qontinui test configuration:**
   ```json
   {
     "workflows": [
       {
         "name": "Login Test",
         "actions": [
           {
             "type": "CLICK",
             "target": "login_button"
           }
         ]
       }
     ]
   }
   ```
   Save this as `tests/ci-test-config.json` in your repository.

4. **Commit and push** - the workflow will run automatically on your next PR!

## Available Templates

### 1. `qontinui-test.yml` - Basic Test Workflow

**Use when:** You want simple, fast CI tests on every push/PR.

**Features:**
- Runs on push to main/develop and on pull requests
- Installs Qontinui with all dependencies
- Runs tests from configuration file
- Uploads test results as artifacts
- Posts test summary as PR comment
- Fails CI if tests fail

**Configuration:**
```yaml
# No secrets required
# Adjust test config path in the workflow file
```

**Example PR comment:**
```
## ✅ Qontinui Test Results

All tests passed!

| Metric | Count |
|--------|-------|
| Total | 15 |
| Passed | ✅ 15 |
| Failed | ❌ 0 |
```

---

### 2. `qontinui-test-with-streaming.yml` - Tests with Real-Time Streaming

**Use when:** You want live test results streamed to the Qontinui dashboard for monitoring.

**Features:**
- Everything from basic test workflow, plus:
- Creates test run in Qontinui dashboard
- Streams results in real-time during execution
- Provides live monitoring link
- Better debugging with centralized results

**Required Secrets:**
```yaml
QONTINUI_API_KEY: "your-api-key-here"
```

**Optional Variables:**
```yaml
QONTINUI_STREAM_URL: "https://api.qontinui.io/v1/test-runs"
```

**Setup:**
1. Get your API key from [qontinui.io/settings/api-keys](https://qontinui.io/settings/api-keys)
2. Add it to your repository secrets:
   - Go to Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `QONTINUI_API_KEY`
   - Value: Your API key

**Example PR comment:**
```
## Qontinui Test Results

All tests passed!

| Metric | Count |
|--------|-------|
| Total | 15 |
| Passed | ✅ 15 |
| Failed | ❌ 0 |

### 🔗 Links

- View live results on Qontinui Dashboard
- GitHub Actions workflow
```

---

### 3. `qontinui-nightly.yml` - Comprehensive Nightly Tests

**Use when:** You want thorough daily testing with coverage reporting.

**Features:**
- Runs daily at 2 AM UTC (configurable)
- Extended timeout (2 hours)
- Full test suite with coverage reporting
- Performance metrics collection
- Slack notifications on failure
- Manual trigger support

**Required Secrets (for notifications):**
```yaml
SLACK_WEBHOOK_URL: "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
```

**Manual Trigger:**
```
# From GitHub UI:
Actions → Qontinui Nightly Tests → Run workflow
Options:
- Test suite: full, smoke, or regression
- Send notification: yes/no
```

**Schedule Configuration:**
```yaml
on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM UTC
    # Or customize:
    # - cron: '0 0 * * 1'  # Weekly on Monday
    # - cron: '0 0 1 * *'  # Monthly on 1st
```

---

### 4. `qontinui-visual-regression.yml` - Visual Regression Testing

**Use when:** You want to detect unintended UI changes automatically.

**Features:**
- Compares screenshots against approved baselines
- Highlights visual differences
- Blocks PR if changes detected (unless approved)
- Auto-updates baselines when approved
- Detailed diff images for review

**Workflow:**
1. PR is opened with UI changes
2. Workflow captures screenshots
3. Compares with baseline screenshots from main branch
4. If differences detected:
   - Posts PR comment with details
   - Uploads diff images
   - Requests review
   - Blocks merge
5. Reviewer examines diffs
6. If intentional: Re-run workflow with "Update baselines" option
7. If unintentional: Fix the code

**Example PR comment:**
```
## ⚠️ Visual Regression Test Results

| Status | Count |
|--------|-------|
| Total Screenshots | 25 |
| ✅ Identical | 23 |
| 🔄 Changed | 2 |
| ➕ New | 0 |
| ❌ Missing | 0 |

### ⚠️ Visual Changes Detected

This PR introduces visual changes. Please review the screenshots carefully.

#### Changes:

- 🔄 login-page.png (92.3% similar)
- 🔄 dashboard.png (88.7% similar)

### 📸 Review Instructions

1. Download the visual-regression-results artifact
2. Review screenshots in the diff/ folder
3. If changes are intentional, approve and update baselines
4. If changes are unintentional, fix the code before merging
```

---

## Setup Action Reference

The `setup-qontinui` action handles all environment setup. It's used by all workflow templates.

**Inputs:**

| Input | Description | Default | Required |
|-------|-------------|---------|----------|
| `python-version` | Python version to install | `3.12` | No |
| `cache-key-suffix` | Additional cache key suffix | `""` | No |
| `install-dev-dependencies` | Install dev dependencies | `true` | No |
| `install-system-dependencies` | Install system deps (Linux) | `true` | No |
| `poetry-version` | Poetry version to install | `1.8.0` | No |

**Outputs:**

| Output | Description |
|--------|-------------|
| `python-version` | Installed Python version |
| `cache-hit` | Whether cache was hit |

**Example usage:**
```yaml
- name: Setup Qontinui
  uses: ./.github/actions/setup-qontinui
  with:
    python-version: '3.12'
    install-dev-dependencies: 'true'
```

---

## Test Configuration Files

Qontinui workflows require JSON configuration files defining which tests to run.

### Basic Configuration

**File:** `tests/ci-test-config.json`

```json
{
  "name": "CI Test Suite",
  "description": "Fast tests for CI pipeline",
  "timeout": 600,
  "workflows": [
    {
      "name": "Login Test",
      "description": "Verify user can log in",
      "actions": [
        {
          "type": "CLICK",
          "target": "login_button",
          "wait": 1000
        },
        {
          "type": "TYPE",
          "target": "username_field",
          "text": "testuser"
        },
        {
          "type": "TYPE",
          "target": "password_field",
          "text": "testpass123"
        },
        {
          "type": "CLICK",
          "target": "submit_button"
        },
        {
          "type": "WAIT_FOR",
          "target": "dashboard",
          "timeout": 5000
        }
      ],
      "success_criteria": {
        "required_states": ["logged_in", "dashboard_visible"]
      }
    }
  ]
}
```

### Nightly Configuration

**File:** `tests/nightly-full-config.json`

```json
{
  "name": "Nightly Full Test Suite",
  "description": "Comprehensive tests run nightly",
  "timeout": 3600,
  "workflows": [
    {
      "name": "User Registration Flow",
      "actions": [...]
    },
    {
      "name": "Complete Purchase Flow",
      "actions": [...]
    },
    {
      "name": "Admin Panel Access",
      "actions": [...]
    }
  ],
  "coverage": {
    "enabled": true,
    "threshold": 80
  },
  "performance": {
    "enabled": true,
    "max_action_time": 1000,
    "max_vision_time": 500
  }
}
```

### Visual Regression Configuration

**File:** `tests/visual-regression-config.json`

```json
{
  "name": "Visual Regression Tests",
  "description": "Capture screenshots for visual comparison",
  "workflows": [
    {
      "name": "Capture Login Page",
      "actions": [
        {
          "type": "NAVIGATE",
          "url": "https://myapp.com/login"
        },
        {
          "type": "SCREENSHOT",
          "name": "login-page",
          "full_page": true
        }
      ]
    },
    {
      "name": "Capture Dashboard",
      "actions": [
        {
          "type": "NAVIGATE",
          "url": "https://myapp.com/dashboard"
        },
        {
          "type": "SCREENSHOT",
          "name": "dashboard",
          "full_page": true
        }
      ]
    }
  ],
  "screenshot_options": {
    "format": "png",
    "quality": 100,
    "full_page": true
  }
}
```

---

## Customization Guide

### Changing Test Triggers

**Run on specific paths only:**
```yaml
on:
  push:
    paths:
      - 'src/**'
      - 'tests/**'
      - 'qontinui-config.json'
```

**Run on specific branches:**
```yaml
on:
  push:
    branches: [ main, staging, develop ]
```

**Disable PR comments:**
```yaml
# Remove or comment out the "Comment PR" step
```

### Adjusting Timeouts

**Workflow timeout:**
```yaml
jobs:
  qontinui-test:
    timeout-minutes: 30  # Adjust as needed
```

**Test timeout:**
```bash
poetry run python -m qontinui.cli run \
  --timeout 1200  # Adjust as needed (seconds)
```

### Multiple Python Versions

```yaml
jobs:
  test:
    strategy:
      matrix:
        python-version: ['3.12', '3.13']
    steps:
      - name: Setup Qontinui
        with:
          python-version: ${{ matrix.python-version }}
```

### Multiple Operating Systems

```yaml
jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    runs-on: ${{ matrix.os }}
```

**Note:** Virtual display setup differs by OS:

**Linux (Xvfb):**
```bash
export DISPLAY=:99
Xvfb :99 -screen 0 1920x1080x24 &
```

**macOS:** No virtual display needed for headless mode

**Windows:** Use `--headless` flag

---

## Required Secrets

Add these secrets to your repository settings (Settings → Secrets and variables → Actions):

### For Streaming Workflow

| Secret | Description | How to Get |
|--------|-------------|------------|
| `QONTINUI_API_KEY` | Qontinui API key | [qontinui.io/settings/api-keys](https://qontinui.io/settings/api-keys) |

### For Nightly Workflow (Optional)

| Secret | Description | How to Get |
|--------|-------------|------------|
| `SLACK_WEBHOOK_URL` | Slack incoming webhook URL | [Slack App Settings → Incoming Webhooks](https://api.slack.com/messaging/webhooks) |

### For Coverage Reporting (Optional)

| Secret | Description | How to Get |
|--------|-------------|------------|
| `CODECOV_TOKEN` | Codecov upload token | [codecov.io](https://codecov.io) |

---

## Troubleshooting

### Tests timeout in CI

**Cause:** Tests take longer in CI than locally due to slower machines.

**Solution:**
- Increase workflow timeout: `timeout-minutes: 60`
- Increase test timeout: `--timeout 1800`
- Reduce test suite size for CI

### Screenshots are blank

**Cause:** Virtual display not properly configured.

**Solution:**
```bash
# Ensure Xvfb is running before tests
export DISPLAY=:99
Xvfb :99 -screen 0 1920x1080x24 > /dev/null 2>&1 &
sleep 3  # Wait for Xvfb to start
```

### Pattern matching fails in CI

**Cause:** Different screen resolution or DPI.

**Solution:**
- Use `--headless` mode with fixed resolution
- Ensure Xvfb uses same resolution as test patterns: `1920x1080x24`
- Adjust pattern matching thresholds for CI

### Cache not working

**Cause:** Cache key mismatch or cache full.

**Solution:**
- Check `poetry.lock` is committed
- Bump cache version: `cache-key-suffix: 'v2'`
- Clear cache from Actions settings

### Visual regression has many false positives

**Cause:** Anti-aliasing or slight rendering differences.

**Solution:**
- Adjust similarity threshold in comparison script: `similarity < 0.95` → `similarity < 0.90`
- Use higher quality screenshots: `quality: 100`
- Ensure consistent fonts and rendering

---

## Best Practices

### 1. Keep CI Tests Fast

- Run full suite nightly, subset in CI
- Use `--headless` mode
- Cache dependencies aggressively
- Parallelize independent tests

### 2. Use Appropriate Timeouts

- CI tests: 5-10 minutes per workflow
- Nightly tests: Up to 1 hour
- Individual actions: 5-30 seconds

### 3. Handle Flaky Tests

- Add retries for unstable actions:
  ```json
  {
    "type": "CLICK",
    "target": "submit_button",
    "retry": 3,
    "retry_delay": 1000
  }
  ```
- Use explicit waits instead of sleeps
- Increase timeouts for slow operations

### 4. Organize Test Configurations

```
tests/
├── ci-test-config.json          # Fast CI tests
├── nightly-full-config.json     # Comprehensive nightly
├── smoke-test-config.json       # Critical path only
├── regression-test-config.json  # Regression suite
└── visual-regression-config.json # Visual tests
```

### 5. Monitor Test Performance

- Track test duration trends
- Set performance budgets
- Alert on slow tests

### 6. Maintain Visual Baselines

- Update baselines when UI intentionally changes
- Store baselines in version control
- Review all visual changes before approving

---

## Examples

### Example 1: Simple Web App Testing

**Scenario:** Test login and basic navigation on every PR.

**Setup:**
1. Copy `qontinui-test.yml` to `.github/workflows/`
2. Create `tests/ci-test-config.json`:
   ```json
   {
     "workflows": [
       {"name": "Login", "actions": [...]},
       {"name": "Navigate Home", "actions": [...]}
     ]
   }
   ```
3. Push and create PR - tests run automatically

---

### Example 2: E-commerce Checkout Flow

**Scenario:** Test complete purchase flow nightly with Slack alerts.

**Setup:**
1. Copy `qontinui-nightly.yml` to `.github/workflows/`
2. Add `SLACK_WEBHOOK_URL` secret
3. Create `tests/nightly-full-config.json` with checkout workflow
4. Tests run daily at 2 AM, Slack notification on failure

---

### Example 3: UI Component Library Testing

**Scenario:** Ensure component rendering doesn't break with visual regression.

**Setup:**
1. Copy `qontinui-visual-regression.yml`
2. Create `tests/visual-regression-config.json`:
   ```json
   {
     "workflows": [
       {"name": "Button Variants", "actions": [...]},
       {"name": "Form Components", "actions": [...]},
       {"name": "Modal Dialogs", "actions": [...]}
     ]
   }
   ```
3. Commit baseline screenshots
4. PR automatically blocked if visual changes detected

---

## Advanced Configuration

### Custom Result Parsing

Add a step to parse custom test results:

```yaml
- name: Custom result parsing
  run: |
    poetry run python scripts/parse_qontinui_results.py \
      --input .qontinui/test-results/results.json \
      --output custom-report.html
```

### Integration with Other Tools

**Send to custom dashboard:**
```yaml
- name: Send to dashboard
  run: |
    curl -X POST https://my-dashboard.com/api/results \
      -H "Authorization: Bearer ${{ secrets.DASHBOARD_TOKEN }}" \
      -d @.qontinui/test-results/results.json
```

**Convert to JUnit format:**
```yaml
- name: Convert to JUnit
  run: |
    poetry run python -m qontinui.cli convert \
      --input .qontinui/test-results/results.json \
      --output junit.xml \
      --format junit
```

---

## Support

- **Documentation:** [qontinui.github.io](https://qontinui.github.io)
- **Issues:** [github.com/qontinui/qontinui/issues](https://github.com/qontinui/qontinui/issues)
- **Discussions:** [github.com/qontinui/qontinui/discussions](https://github.com/qontinui/qontinui/discussions)
- **Discord:** [discord.gg/qontinui](https://discord.gg/qontinui)

---

## License

These templates are provided under the MIT License, same as Qontinui itself.
