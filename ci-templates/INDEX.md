# Qontinui CI/CD Templates - File Index

Complete reference for all files in the CI/CD templates directory.

## Quick Navigation

- **New to Qontinui CI?** → Start with [QUICKSTART.md](QUICKSTART.md)
- **Have existing CI?** → Read [MIGRATION.md](MIGRATION.md)
- **Need full details?** → See [README.md](README.md)

---

## Directory Structure

```
ci-templates/
├── README.md                                    # Complete documentation
├── QUICKSTART.md                                # 5-minute quick start guide
├── MIGRATION.md                                 # Existing CI integration guide
├── INDEX.md                                     # This file
├── github-actions/                              # GitHub Actions workflows
│   ├── setup-qontinui/
│   │   └── action.yml                          # Reusable setup action
│   ├── qontinui-test.yml                       # Basic test workflow
│   ├── qontinui-test-with-streaming.yml        # Tests with result streaming
│   ├── qontinui-nightly.yml                    # Nightly comprehensive tests
│   └── qontinui-visual-regression.yml          # Visual regression testing
└── examples/                                    # Example test configurations
    ├── ci-test-config.json                     # Fast CI smoke tests
    ├── nightly-full-config.json                # Comprehensive nightly tests
    └── visual-regression-config.json           # Visual comparison tests
```

---

## Workflow Templates

### 1. `github-actions/qontinui-test.yml`

**Purpose:** Basic automated testing on every push/PR

**Key Features:**
- Fast CI tests (5-10 minutes)
- Test result artifacts
- PR comments with summary
- Fails CI on test failure

**When to use:**
- You want simple, reliable CI testing
- You're just getting started with Qontinui
- You don't need advanced features

**Required setup:**
- Copy `setup-qontinui/action.yml` to `.github/actions/`
- Create test config at `tests/ci-test-config.json`

**Secrets needed:** None

---

### 2. `github-actions/qontinui-test-with-streaming.yml`

**Purpose:** Real-time test result streaming to Qontinui dashboard

**Key Features:**
- Everything from basic test workflow
- Live test monitoring
- Centralized result storage
- Dashboard integration

**When to use:**
- You want to monitor tests in real-time
- You need centralized test history
- You're using qontinui.io dashboard

**Required setup:**
- Copy `setup-qontinui/action.yml` to `.github/actions/`
- Create test config at `tests/ci-test-config.json`
- Add `QONTINUI_API_KEY` secret

**Secrets needed:**
- `QONTINUI_API_KEY` - Get from [qontinui.io/settings/api-keys](https://qontinui.io/settings/api-keys)

---

### 3. `github-actions/qontinui-nightly.yml`

**Purpose:** Comprehensive testing with coverage and performance metrics

**Key Features:**
- Runs daily at 2 AM UTC
- Extended timeout (2 hours)
- Coverage reporting
- Performance tracking
- Slack notifications

**When to use:**
- You want thorough daily testing
- You need coverage reports
- You want to track performance trends
- You want alerts on failures

**Required setup:**
- Copy `setup-qontinui/action.yml` to `.github/actions/`
- Create test config at `tests/nightly-full-config.json`
- (Optional) Add `SLACK_WEBHOOK_URL` for notifications

**Secrets needed:**
- `SLACK_WEBHOOK_URL` (optional) - For failure notifications

---

### 4. `github-actions/qontinui-visual-regression.yml`

**Purpose:** Detect unintended UI changes through screenshot comparison

**Key Features:**
- Compares screenshots with baselines
- Blocks PRs with visual changes
- Generates diff images
- Auto-update baselines when approved

**When to use:**
- You want to catch unexpected UI changes
- You're working on a design system
- You need visual QA automation

**Required setup:**
- Copy `setup-qontinui/action.yml` to `.github/actions/`
- Create test config at `tests/visual-regression-config.json`
- Commit baseline screenshots

**Secrets needed:** None

---

## Setup Action

### `github-actions/setup-qontinui/action.yml`

**Purpose:** Reusable action to set up Qontinui environment

**What it does:**
1. Installs Python
2. Installs system dependencies (Linux)
3. Installs Poetry
4. Caches dependencies
5. Installs Qontinui
6. Verifies installation

**Inputs:**
- `python-version` (default: `3.12`)
- `cache-key-suffix` (default: `""`)
- `install-dev-dependencies` (default: `true`)
- `install-system-dependencies` (default: `true`)
- `poetry-version` (default: `1.8.0`)

**Outputs:**
- `python-version` - Installed Python version
- `cache-hit` - Whether cache was hit

**Usage:**
```yaml
- name: Setup Qontinui
  uses: ./.github/actions/setup-qontinui
  with:
    python-version: '3.12'
```

---

## Example Configurations

### 1. `examples/ci-test-config.json`

**Purpose:** Fast smoke tests for CI

**Contents:**
- Login test workflow
- Navigation test workflow

**Estimated runtime:** 2-3 minutes

**Use for:**
- Every push/PR
- Quick feedback
- Smoke testing

---

### 2. `examples/nightly-full-config.json`

**Purpose:** Comprehensive test suite

**Contents:**
- User registration flow
- Complete purchase flow
- Admin panel access
- Search and filter
- Settings and profile update

**Estimated runtime:** 15-30 minutes

**Use for:**
- Nightly runs
- Full regression testing
- Pre-release verification

---

### 3. `examples/visual-regression-config.json`

**Purpose:** Screenshot capture for visual comparison

**Contents:**
- Login page screenshots (multiple states)
- Dashboard screenshots
- Product list screenshots
- Shopping cart screenshots
- Checkout page screenshots
- Settings page screenshots
- Modal dialog screenshots
- Responsive viewport screenshots

**Estimated runtime:** 10-15 minutes

**Use for:**
- Visual regression testing
- UI component verification
- Design system testing

---

## Documentation Files

### README.md

**Complete documentation** covering:
- All workflow templates in detail
- Setup action reference
- Test configuration format
- Customization guide
- Required secrets
- Troubleshooting
- Best practices
- Examples

**Start here if:** You want comprehensive understanding

---

### QUICKSTART.md

**5-minute setup guide** with:
- Copy-paste commands
- Minimal configuration
- Quick examples
- Common customizations

**Start here if:** You want to get running fast

---

### MIGRATION.md

**Integration guide** covering:
- Adding Qontinui to existing workflows
- Parallel vs sequential strategies
- Migrating from Selenium/Playwright/Cypress
- Build time optimization
- Advanced patterns

**Start here if:** You have existing CI/CD

---

### INDEX.md (this file)

**File reference** listing:
- All files with descriptions
- When to use each template
- Required setup per template
- Quick navigation

**Start here if:** You want an overview

---

## Recommended Workflow

### For New Projects

1. **Start with basic testing:**
   ```bash
   cp ci-templates/github-actions/qontinui-test.yml .github/workflows/
   cp ci-templates/examples/ci-test-config.json tests/
   ```

2. **Add nightly comprehensive tests:**
   ```bash
   cp ci-templates/github-actions/qontinui-nightly.yml .github/workflows/
   cp ci-templates/examples/nightly-full-config.json tests/
   ```

3. **Add visual regression (optional):**
   ```bash
   cp ci-templates/github-actions/qontinui-visual-regression.yml .github/workflows/
   cp ci-templates/examples/visual-regression-config.json tests/
   ```

### For Existing Projects

1. **Read migration guide:**
   ```bash
   cat ci-templates/MIGRATION.md
   ```

2. **Add Qontinui to existing workflow:**
   - Copy setup action
   - Add Qontinui steps to existing workflow
   - Run in parallel with existing tests

3. **Gradually migrate tests:**
   - Start with smoke tests
   - Add more comprehensive tests
   - Eventually replace old E2E framework

---

## File Size Reference

| File | Size | Complexity |
|------|------|------------|
| `qontinui-test.yml` | ~150 lines | Simple |
| `qontinui-test-with-streaming.yml` | ~200 lines | Moderate |
| `qontinui-nightly.yml` | ~250 lines | Moderate |
| `qontinui-visual-regression.yml` | ~350 lines | Complex |
| `setup-qontinui/action.yml` | ~100 lines | Simple |
| `ci-test-config.json` | ~60 lines | Simple |
| `nightly-full-config.json` | ~250 lines | Moderate |
| `visual-regression-config.json` | ~350 lines | Moderate |

---

## Common Use Cases

### Use Case 1: Startup MVP Testing

**Goal:** Fast, reliable testing for small team

**Recommended:**
- `qontinui-test.yml` - Basic CI
- `ci-test-config.json` - Minimal tests

**Rationale:** Simple, fast, low maintenance

---

### Use Case 2: E-commerce Platform

**Goal:** Comprehensive testing with visual QA

**Recommended:**
- `qontinui-test.yml` - Fast CI smoke tests
- `qontinui-nightly.yml` - Full regression nightly
- `qontinui-visual-regression.yml` - Catch UI bugs

**Rationale:** Multi-layered testing strategy

---

### Use Case 3: SaaS Product

**Goal:** Continuous monitoring with alerts

**Recommended:**
- `qontinui-test-with-streaming.yml` - CI with dashboard
- `qontinui-nightly.yml` - Comprehensive nightly
- Custom scheduled workflow - Hourly production checks

**Rationale:** Real-time visibility and monitoring

---

### Use Case 4: Design System

**Goal:** Ensure components render correctly

**Recommended:**
- `qontinui-visual-regression.yml` - Visual testing
- `visual-regression-config.json` - Component screenshots

**Rationale:** Visual changes must be approved

---

## Getting Started Checklist

- [ ] Read [QUICKSTART.md](QUICKSTART.md)
- [ ] Copy `setup-qontinui` action to `.github/actions/`
- [ ] Copy a workflow template to `.github/workflows/`
- [ ] Copy an example config to `tests/`
- [ ] Customize config with your URLs and actions
- [ ] Add required secrets (if using streaming/notifications)
- [ ] Commit and push
- [ ] Create PR and verify tests run
- [ ] Review test results
- [ ] Iterate on test configuration

---

## Support Resources

- **Full Documentation:** [README.md](README.md)
- **Quick Start:** [QUICKSTART.md](QUICKSTART.md)
- **Migration Guide:** [MIGRATION.md](MIGRATION.md)
- **GitHub Issues:** [github.com/qontinui/qontinui/issues](https://github.com/qontinui/qontinui/issues)
- **Discussions:** [github.com/qontinui/qontinui/discussions](https://github.com/qontinui/qontinui/discussions)
- **Discord:** [discord.gg/qontinui](https://discord.gg/qontinui)

---

## Contributing

Found a bug or have a suggestion? Please open an issue or PR!

See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines.
