# Migration Guide: Integrating Qontinui into Existing CI/CD

This guide helps you add Qontinui tests to your existing GitHub Actions workflows.

## Table of Contents

1. [Adding Qontinui to Existing Test Workflows](#adding-qontinui-to-existing-test-workflows)
2. [Parallel Testing Strategy](#parallel-testing-strategy)
3. [Sequential Testing Strategy](#sequential-testing-strategy)
4. [Migrating from Other Test Frameworks](#migrating-from-other-test-frameworks)
5. [Optimizing Build Times](#optimizing-build-times)
6. [Advanced Integration Patterns](#advanced-integration-patterns)

---

## Adding Qontinui to Existing Test Workflows

If you already have a GitHub Actions workflow for testing, you can add Qontinui as an additional step.

### Example: Existing Pytest + Jest Workflow

**Before:**
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -r requirements.txt
      - run: pytest
      - run: npm test
```

**After (Adding Qontinui):**
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # Existing Python tests
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -r requirements.txt
      - run: pytest

      # Existing JavaScript tests
      - run: npm test

      # Add Qontinui E2E tests
      - name: Setup Qontinui
        uses: ./.github/actions/setup-qontinui
        with:
          python-version: '3.12'

      - name: Setup virtual display
        run: |
          export DISPLAY=:99
          Xvfb :99 -screen 0 1920x1080x24 > /dev/null 2>&1 &
          sleep 3

      - name: Run Qontinui E2E tests
        run: |
          poetry run python -m qontinui.cli run \
            --config ./tests/e2e-config.json \
            --headless \
            --output-dir .qontinui/test-results
        env:
          DISPLAY: :99

      - name: Upload Qontinui results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: qontinui-results
          path: .qontinui/test-results/
```

---

## Parallel Testing Strategy

Run Qontinui tests in parallel with other test suites to save time.

### Example: Parallel Unit, Integration, and E2E Tests

```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -r requirements.txt
      - run: pytest tests/unit/

  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -r requirements.txt
      - run: pytest tests/integration/

  e2e-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Qontinui
        uses: ./.github/actions/setup-qontinui
      - name: Setup virtual display
        run: |
          export DISPLAY=:99
          Xvfb :99 -screen 0 1920x1080x24 &
          sleep 3
      - name: Run E2E tests
        run: |
          poetry run python -m qontinui.cli run \
            --config tests/e2e-config.json \
            --headless
        env:
          DISPLAY: :99
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: e2e-results
          path: .qontinui/

  # All tests must pass
  all-tests:
    needs: [unit-tests, integration-tests, e2e-tests]
    runs-on: ubuntu-latest
    steps:
      - run: echo "All tests passed!"
```

**Benefits:**
- Tests run simultaneously (faster CI)
- Failures are isolated (easier debugging)
- Can use different runners/resources per job

---

## Sequential Testing Strategy

Run Qontinui tests only after other tests pass to save resources.

### Example: Sequential Test Pipeline

```yaml
name: Test Pipeline

on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install ruff black
      - run: ruff check .
      - run: black --check .

  unit-tests:
    needs: lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pytest tests/unit/

  integration-tests:
    needs: unit-tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pytest tests/integration/

  e2e-tests:
    needs: integration-tests  # Only run if integration tests pass
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Qontinui
        uses: ./.github/actions/setup-qontinui
      - name: Run E2E tests
        run: |
          export DISPLAY=:99
          Xvfb :99 -screen 0 1920x1080x24 &
          sleep 3
          poetry run python -m qontinui.cli run \
            --config tests/e2e-config.json \
            --headless
        env:
          DISPLAY: :99
```

**Benefits:**
- Don't waste time on E2E if unit tests fail
- Faster feedback on simple failures
- Save CI minutes

---

## Migrating from Other Test Frameworks

### From Selenium

**Old Selenium test:**
```python
from selenium import webdriver

driver = webdriver.Chrome()
driver.get("https://myapp.com")
driver.find_element("id", "login-button").click()
driver.find_element("name", "username").send_keys("testuser")
driver.quit()
```

**Equivalent Qontinui config:**
```json
{
  "workflows": [
    {
      "name": "Login Test",
      "actions": [
        {
          "type": "NAVIGATE",
          "url": "https://myapp.com"
        },
        {
          "type": "CLICK",
          "target": {"type": "OCR", "text": "Login"}
        },
        {
          "type": "TYPE",
          "target": {"type": "OCR", "text": "Username"},
          "text": "testuser"
        }
      ]
    }
  ]
}
```

**Migration strategy:**
1. Keep existing Selenium tests initially
2. Add Qontinui tests in parallel
3. Gradually migrate tests to Qontinui
4. Remove Selenium when coverage is complete

**Workflow integration:**
```yaml
jobs:
  selenium-tests:
    runs-on: ubuntu-latest
    steps:
      - run: pytest tests/selenium/  # Old tests

  qontinui-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: ./.github/actions/setup-qontinui
      - run: poetry run qontinui run --config tests/qontinui-config.json
```

---

### From Playwright

**Old Playwright test:**
```javascript
const { test, expect } = require('@playwright/test');

test('login', async ({ page }) => {
  await page.goto('https://myapp.com');
  await page.click('text=Login');
  await page.fill('[name="username"]', 'testuser');
});
```

**Equivalent Qontinui config:**
```json
{
  "workflows": [
    {
      "name": "Login Test",
      "actions": [
        {
          "type": "NAVIGATE",
          "url": "https://myapp.com"
        },
        {
          "type": "CLICK",
          "target": {"type": "OCR", "text": "Login"}
        },
        {
          "type": "TYPE",
          "target": {"type": "OCR", "text": "Username"},
          "text": "testuser"
        }
      ]
    }
  ]
}
```

**Workflow integration:**
```yaml
jobs:
  playwright-tests:
    runs-on: ubuntu-latest
    steps:
      - run: npx playwright test

  qontinui-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: ./.github/actions/setup-qontinui
      - run: poetry run qontinui run --config tests/qontinui-config.json
```

---

### From Cypress

**Old Cypress test:**
```javascript
describe('Login', () => {
  it('should log in', () => {
    cy.visit('https://myapp.com')
    cy.contains('Login').click()
    cy.get('[name="username"]').type('testuser')
  })
})
```

**Equivalent Qontinui config:**
```json
{
  "workflows": [
    {
      "name": "Login Test",
      "actions": [
        {
          "type": "NAVIGATE",
          "url": "https://myapp.com"
        },
        {
          "type": "CLICK",
          "target": {"type": "OCR", "text": "Login"}
        },
        {
          "type": "TYPE",
          "target": {"type": "OCR", "text": "Username"},
          "text": "testuser"
        }
      ]
    }
  ]
}
```

---

## Optimizing Build Times

### 1. Cache Dependencies Aggressively

```yaml
- name: Cache Poetry
  uses: actions/cache@v4
  with:
    path: |
      ~/.cache/pypoetry
      ~/.local/share/pypoetry
    key: ${{ runner.os }}-poetry-${{ hashFiles('**/poetry.lock') }}

- name: Cache Qontinui models
  uses: actions/cache@v4
  with:
    path: ~/.cache/qontinui
    key: ${{ runner.os }}-qontinui-models-v1
```

### 2. Run Fast Tests First

```yaml
jobs:
  smoke-tests:
    runs-on: ubuntu-latest
    steps:
      - run: poetry run qontinui run --config tests/smoke.json  # Fast

  full-tests:
    needs: smoke-tests  # Only run if smoke tests pass
    runs-on: ubuntu-latest
    steps:
      - run: poetry run qontinui run --config tests/full.json  # Slow
```

### 3. Use Matrix Builds for Parallelization

```yaml
jobs:
  qontinui-tests:
    strategy:
      matrix:
        test-suite: [auth, checkout, search, admin]
    runs-on: ubuntu-latest
    steps:
      - uses: ./.github/actions/setup-qontinui
      - run: |
          poetry run qontinui run \
            --config tests/${{ matrix.test-suite }}-config.json
```

### 4. Skip Tests on Documentation Changes

```yaml
on:
  push:
    paths-ignore:
      - '**.md'
      - 'docs/**'
```

### 5. Use Concurrency to Cancel Outdated Runs

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

---

## Advanced Integration Patterns

### 1. Conditional E2E Tests

Run E2E tests only on main branch or when specific files change:

```yaml
jobs:
  qontinui-tests:
    if: |
      github.ref == 'refs/heads/main' ||
      contains(github.event.head_commit.modified, 'frontend/')
    runs-on: ubuntu-latest
    steps:
      - uses: ./.github/actions/setup-qontinui
      - run: poetry run qontinui run --config tests/e2e.json
```

### 2. Deployment-Triggered Tests

Run tests after deployment to verify production:

```yaml
name: Post-Deployment Tests

on:
  deployment_status:

jobs:
  verify-deployment:
    if: github.event.deployment_status.state == 'success'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: ./.github/actions/setup-qontinui
      - run: |
          poetry run qontinui run \
            --config tests/production-smoke.json \
            --base-url ${{ github.event.deployment.payload.web_url }}
```

### 3. Multi-Environment Testing

Test against multiple environments (staging, production):

```yaml
jobs:
  test-environments:
    strategy:
      matrix:
        environment:
          - name: staging
            url: https://staging.myapp.com
          - name: production
            url: https://myapp.com
    runs-on: ubuntu-latest
    steps:
      - uses: ./.github/actions/setup-qontinui
      - run: |
          poetry run qontinui run \
            --config tests/smoke.json \
            --base-url ${{ matrix.environment.url }}
```

### 4. Scheduled Monitoring

Run tests periodically to monitor production health:

```yaml
name: Production Monitoring

on:
  schedule:
    - cron: '*/30 * * * *'  # Every 30 minutes

jobs:
  monitor:
    runs-on: ubuntu-latest
    steps:
      - uses: ./.github/actions/setup-qontinui
      - run: |
          poetry run qontinui run \
            --config tests/health-check.json \
            --base-url https://myapp.com

      - name: Alert on failure
        if: failure()
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "🚨 Production health check failed!"
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

### 5. Canary Testing

Test new deployments with small percentage of traffic:

```yaml
name: Canary Tests

on:
  workflow_dispatch:
    inputs:
      canary_url:
        description: 'Canary deployment URL'
        required: true

jobs:
  test-canary:
    runs-on: ubuntu-latest
    steps:
      - uses: ./.github/actions/setup-qontinui
      - name: Run canary tests
        run: |
          poetry run qontinui run \
            --config tests/canary.json \
            --base-url ${{ github.event.inputs.canary_url }}

      - name: Compare with production
        run: |
          # Compare canary results with production baseline
          poetry run python scripts/compare_results.py \
            --canary .qontinui/test-results/ \
            --baseline .qontinui/production-baseline/
```

---

## Common Migration Scenarios

### Scenario 1: Existing Jenkins Pipeline

**Old Jenkinsfile:**
```groovy
pipeline {
  stages {
    stage('Test') {
      steps {
        sh 'pytest'
      }
    }
  }
}
```

**Migration:** Create GitHub Actions workflow with same steps
```yaml
name: Tests
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: pytest
      - uses: ./.github/actions/setup-qontinui
      - run: poetry run qontinui run --config tests/e2e.json
```

### Scenario 2: Existing Travis CI

**Old .travis.yml:**
```yaml
language: python
script:
  - pytest
```

**Migration:** Equivalent GitHub Actions
```yaml
name: Tests
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/setup-python@v5
      - run: pytest
      - uses: ./.github/actions/setup-qontinui
      - run: poetry run qontinui run --config tests/e2e.json
```

### Scenario 3: Existing CircleCI

**Old .circleci/config.yml:**
```yaml
jobs:
  test:
    steps:
      - run: pytest
```

**Migration:** GitHub Actions equivalent
```yaml
name: Tests
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: pytest
      - uses: ./.github/actions/setup-qontinui
      - run: poetry run qontinui run --config tests/e2e.json
```

---

## Troubleshooting Migration Issues

### Issue: Different environment variables

**Solution:** Map old env vars to new ones
```yaml
env:
  # Old CI
  OLD_VAR: ${{ secrets.OLD_SECRET }}
  # New Qontinui
  QONTINUI_API_KEY: ${{ secrets.QONTINUI_API_KEY }}
```

### Issue: Different artifact storage

**Solution:** Upload to both old and new locations
```yaml
- uses: actions/upload-artifact@v4
  with:
    name: test-results
    path: |
      old-results/
      .qontinui/test-results/
```

### Issue: Different notification systems

**Solution:** Send notifications to multiple channels
```yaml
- name: Notify Slack
  if: failure()
  run: # Send to Slack

- name: Notify Email
  if: failure()
  run: # Send email

- name: Notify PagerDuty
  if: failure()
  run: # Send to PagerDuty
```

---

## Best Practices for Migration

1. **Incremental migration** - Don't migrate everything at once
2. **Run both pipelines** - Keep old CI while adding new
3. **Compare results** - Ensure new tests catch same issues
4. **Update documentation** - Document new CI setup
5. **Train team** - Ensure team understands new workflow
6. **Monitor closely** - Watch for regressions during migration
7. **Rollback plan** - Have plan to revert if needed

---

## Need Help?

- **Documentation:** [README.md](README.md)
- **Issues:** [github.com/qontinui/qontinui/issues](https://github.com/qontinui/qontinui/issues)
- **Discussions:** [github.com/qontinui/qontinui/discussions](https://github.com/qontinui/qontinui/discussions)
