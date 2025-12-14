# Quick Start Guide: Qontinui CI/CD Integration

Get Qontinui tests running in your GitHub Actions CI/CD pipeline in under 5 minutes.

## Step 1: Copy Setup Action (30 seconds)

```bash
# From your repository root
mkdir -p .github/actions
cp -r ci-templates/github-actions/setup-qontinui .github/actions/
```

Or if you're using Qontinui as a dependency:

```yaml
# Use the remote action directly in your workflow
uses: qontinui/qontinui/ci-templates/github-actions/setup-qontinui@main
```

## Step 2: Choose a Workflow Template (1 minute)

Pick one based on your needs:

### For basic CI testing (recommended to start):
```bash
cp ci-templates/github-actions/qontinui-test.yml .github/workflows/
```

### For real-time result streaming:
```bash
cp ci-templates/github-actions/qontinui-test-with-streaming.yml .github/workflows/
```

### For nightly comprehensive tests:
```bash
cp ci-templates/github-actions/qontinui-nightly.yml .github/workflows/
```

### For visual regression testing:
```bash
cp ci-templates/github-actions/qontinui-visual-regression.yml .github/workflows/
```

## Step 3: Create Test Configuration (2 minutes)

Create `tests/ci-test-config.json`:

```json
{
  "name": "My CI Tests",
  "workflows": [
    {
      "name": "Quick Smoke Test",
      "actions": [
        {
          "type": "NAVIGATE",
          "url": "https://myapp.com"
        },
        {
          "type": "WAIT_FOR",
          "target": {
            "type": "OCR",
            "text": "Welcome"
          },
          "timeout": 5000
        }
      ]
    }
  ]
}
```

**Or copy an example:**
```bash
mkdir -p tests
cp ci-templates/examples/ci-test-config.json tests/
# Edit tests/ci-test-config.json with your URLs and actions
```

## Step 4: Commit and Push (1 minute)

```bash
git add .github/ tests/
git commit -m "Add Qontinui CI/CD integration"
git push
```

## Step 5: Create a Pull Request

Create a PR and watch your tests run automatically!

---

## What Happens Next?

1. **GitHub Actions triggers** the workflow on your PR
2. **Setup action** installs Python, Poetry, and Qontinui dependencies
3. **Virtual display** is configured (Xvfb on Linux)
4. **Qontinui runs** your test configuration
5. **Results are uploaded** as artifacts
6. **PR comment** is posted with test summary
7. **CI passes/fails** based on test results

---

## Example PR Comment

After your tests run, you'll see a comment like this:

```
## ✅ Qontinui Test Results

All tests passed!

| Metric | Count |
|--------|-------|
| Total | 5 |
| Passed | ✅ 5 |
| Failed | ❌ 0 |

View full results
```

---

## Next Steps

### Add More Tests

Edit `tests/ci-test-config.json` to add more workflows:

```json
{
  "workflows": [
    {
      "name": "Login Test",
      "actions": [...]
    },
    {
      "name": "Checkout Test",
      "actions": [...]
    },
    {
      "name": "Search Test",
      "actions": [...]
    }
  ]
}
```

### Enable Nightly Tests

```bash
cp ci-templates/github-actions/qontinui-nightly.yml .github/workflows/
cp ci-templates/examples/nightly-full-config.json tests/
```

Add `SLACK_WEBHOOK_URL` secret for notifications:
1. Settings → Secrets and variables → Actions
2. New repository secret
3. Name: `SLACK_WEBHOOK_URL`
4. Value: Your webhook URL from Slack

### Enable Visual Regression

```bash
cp ci-templates/github-actions/qontinui-visual-regression.yml .github/workflows/
cp ci-templates/examples/visual-regression-config.json tests/
```

Commit baseline screenshots:
```bash
mkdir -p .qontinui/screenshots/baseline
# Add your baseline screenshots to this directory
git add .qontinui/screenshots/baseline
git commit -m "Add visual regression baselines"
```

### Enable Result Streaming

Add `QONTINUI_API_KEY` secret:
1. Get API key from [qontinui.io/settings/api-keys](https://qontinui.io/settings/api-keys)
2. Settings → Secrets and variables → Actions
3. New repository secret
4. Name: `QONTINUI_API_KEY`
5. Value: Your API key

Then use the streaming workflow:
```bash
cp ci-templates/github-actions/qontinui-test-with-streaming.yml .github/workflows/
```

---

## Troubleshooting

### Tests timeout
- Increase workflow timeout: `timeout-minutes: 60`
- Increase test timeout in config: `"timeout": 1800`

### Screenshots are blank
- Check Xvfb is running (setup action handles this)
- Ensure `--headless` flag is used in workflow

### Pattern matching fails
- Use same resolution in CI as locally: `1920x1080x24`
- Adjust pattern matching thresholds
- Use OCR for text-based targets (more reliable)

### Dependencies installation fails
- Check `poetry.lock` is committed
- Clear cache: Settings → Actions → Caches → Delete all
- Verify Python version matches: `python-version: '3.12'`

---

## Common Customizations

### Run on specific branches only
```yaml
on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
```

### Run on specific paths
```yaml
on:
  push:
    paths:
      - 'src/**'
      - 'tests/**'
```

### Run multiple Python versions
```yaml
strategy:
  matrix:
    python-version: ['3.12', '3.13']
```

### Disable PR comments
```yaml
# Remove or comment out the "Comment PR" step
```

---

## Getting Help

- **Full Documentation:** [README.md](README.md)
- **Issues:** [github.com/qontinui/qontinui/issues](https://github.com/qontinui/qontinui/issues)
- **Discussions:** [github.com/qontinui/qontinui/discussions](https://github.com/qontinui/qontinui/discussions)
- **Discord:** [discord.gg/qontinui](https://discord.gg/qontinui)

---

## What's in the Box?

This quick start gives you:

- ✅ Automated testing on every push/PR
- ✅ Test result artifacts for debugging
- ✅ PR comments with test summary
- ✅ Dependency caching for fast builds
- ✅ Virtual display for headless testing
- ✅ Coverage reporting (nightly)
- ✅ Slack notifications (nightly)
- ✅ Visual regression testing (optional)
- ✅ Result streaming (optional)

You can enable more features by copying additional workflow templates!
