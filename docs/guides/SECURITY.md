# Qontinui Security Guide

## Executive Summary

Qontinui is a GUI automation framework designed for **trusted automation environments**. This document outlines the security model, threat model, security assumptions, and best practices for using Qontinui safely.

**Key Security Principles:**
- Designed for trusted developer-written scripts, not untrusted user input
- Runs in controlled, single-user environments
- Uses dynamic code evaluation for flexible automation DSL
- Employs pickle serialization for state persistence
- Path operations assume trusted script sources

## Table of Contents

1. [Security Model](#security-model)
2. [Threat Model](#threat-model)
3. [Security Assumptions](#security-assumptions)
4. [Security Features](#security-features)
5. [Known Security Considerations](#known-security-considerations)
6. [Best Practices](#best-practices)
7. [Security Roadmap](#security-roadmap)
8. [Vulnerability Disclosure](#vulnerability-disclosure)

---

## Security Model

### Design Philosophy

Qontinui is designed as a **developer tool for automation**, not a web application or multi-tenant service. The security model reflects this use case:

**Trust Boundaries:**
- **TRUSTED**: Automation scripts written by developers
- **TRUSTED**: Configuration files in the project
- **TRUSTED**: State files saved by Qontinui
- **SEMI-TRUSTED**: Input images and screenshots
- **UNTRUSTED**: External network resources (future feature)

**Execution Context:**
- Single-user workstation or CI/CD environment
- Local file system access required for automation
- No network exposure by default
- No multi-tenant isolation needed

### Use Cases

**Intended Use:**
- Desktop GUI test automation
- Workflow automation for trusted applications
- CI/CD pipeline integration
- Developer productivity tools
- QA testing frameworks

**NOT Intended For:**
- Processing untrusted user input
- Multi-tenant SaaS environments
- Public-facing web services
- Sandboxed execution of arbitrary code
- Security-critical applications without additional hardening

---

## Threat Model

### In Scope Threats

Qontinui considers and mitigates these threat vectors:

#### 1. Malicious Input Files
**Threat:** Compromised image files or state files could exploit vulnerabilities
**Mitigation:**
- File extension validation for images (.png, .jpg, .jpeg)
- Controlled pickle loading from trusted directories only
- Future: HMAC signature verification for state files

#### 2. Compromised Dependencies
**Threat:** Malicious or vulnerable third-party libraries
**Mitigation:**
- Regular dependency updates via Dependabot
- Pinned versions in poetry.lock
- Security scanning in CI/CD pipeline
- Minimal dependency footprint

#### 3. Unintentional Code Injection
**Threat:** Developer error leading to unsafe expression evaluation
**Mitigation:**
- Restricted AST parsing for expressions
- Safe function whitelist
- No dangerous builtins in eval context
- Clear security documentation

#### 4. Path Traversal
**Threat:** Malicious file paths accessing unauthorized directories
**Mitigation:**
- Path resolution and validation
- Extension whitelisting
- Base directory constraints (where applicable)

### Out of Scope Threats

The following threats are **outside** Qontinui's security scope:

#### 1. Untrusted User Input
**Rationale:** Qontinui scripts are written by developers, not end users. The framework assumes script contents are trusted.

**If you need to process untrusted input:**
- Run Qontinui in isolated containers/VMs
- Implement additional input validation layers
- Use restricted execution environments
- Consider alternative sandboxed frameworks

#### 2. Multi-Tenant Isolation
**Rationale:** Qontinui is a single-user tool. No isolation between concurrent executions is provided.

**For multi-tenant scenarios:**
- Use separate OS-level users
- Container/VM isolation
- Separate Qontinui installations per tenant

#### 3. Network Attacks
**Rationale:** Qontinui runs locally with no network services exposed.

**Note:** Future network features (remote control, cloud storage) will require security reassessment.

#### 4. Physical Access
**Rationale:** Local physical access to a machine running Qontinui implies full compromise.

---

## Security Assumptions

### Critical Security Assumptions

Qontinui's design makes several security assumptions that users must be aware of:

### 1. Dynamic Code Evaluation

#### 1.1 Expression Evaluation (SafeEvaluator)

**Location:** `/src/qontinui/actions/data_operations/evaluator.py`

**Purpose:** Evaluate Python expressions in automation scripts for data operations and calculations.

**Security Implementation:**
```python
# Restricted AST parsing - only whitelisted nodes allowed
ALLOWED_NODES = {
    ast.Expression, ast.Constant, ast.Name, ast.BinOp,
    ast.Compare, ast.BoolOp, ast.UnaryOp, ...
}

# Safe function whitelist
SAFE_FUNCTIONS = {
    "abs", "bool", "float", "int", "len", "max", "min",
    "range", "round", "sorted", "str", "sum", ...
}

# Evaluation with restricted builtins
result = eval(compile(tree, "<string>", "eval"),
              {"__builtins__": SAFE_FUNCTIONS})
```

**Assumptions:**
- Expressions come from trusted automation scripts
- Developers understand expression syntax is limited
- No import statements or file I/O in expressions
- AST validation prevents dangerous operations

**Risk Level:** **LOW** (when used as designed)

**Risk Escalation:** **HIGH** if expressions contain untrusted user input

**Mitigation:**
- Static AST analysis before evaluation
- Whitelist-based approach (deny by default)
- No access to `__import__`, `open`, `exec`, `eval`
- Clear documentation of allowed operations

**Example Safe Usage:**
```python
# SAFE: Trusted developer-written expression
evaluator.safe_eval("x + y * 2", {"x": 10, "y": 5})

# SAFE: Mathematical operations
evaluator.safe_eval("max(numbers) if numbers else 0", {"numbers": [1, 2, 3]})
```

**Example Unsafe Usage:**
```python
# UNSAFE: User input in expression
user_input = request.get("expression")  # From web form
evaluator.safe_eval(user_input, context)  # DON'T DO THIS

# Unsafe user input example:
# "__import__('os').system('rm -rf /')"
# This would be blocked by AST validation, but still never use untrusted input
```

#### 1.2 Condition Evaluation

**Location:** `/src/qontinui/actions/control_flow/condition_evaluator.py`

**Purpose:** Evaluate conditional expressions for IF/WHILE/BREAK/CONTINUE control flow.

**Security Implementation:**
```python
# Restricted builtins
eval_context = {"variables": variables, **variables}
result = eval(expression, {"__builtins__": {}}, eval_context)
```

**Assumptions:**
- Condition expressions are part of trusted workflow definitions
- No untrusted user input in conditions
- Developers are responsible for expression safety

**Risk Level:** **LOW** (trusted input only)

**Mitigation:**
- Empty `__builtins__` prevents dangerous operations
- Expressions are limited to variable comparisons
- No file I/O or network access possible
- Clear error messages for invalid expressions

**Example Safe Usage:**
```python
# SAFE: Trusted workflow condition
condition = ConditionConfig(
    type="expression",
    expression="counter > 10 and status == 'ready'"
)
```

### 2. Pickle Deserialization

**Locations:**
- `/src/qontinui/persistence/serializers.py` - State persistence
- `/src/qontinui/perception/matching.py` - Element matcher state
- `/src/qontinui/perception/vector_store.py` - Vector database state

**Purpose:** Serialize and deserialize Python objects for state persistence, ML models, and cached data.

**Pickle Security Concerns:**

Pickle is inherently insecure when loading untrusted data. Malicious pickle files can execute arbitrary code during deserialization.

**Assumptions:**
- Pickle files are generated by Qontinui itself
- Files are stored in trusted locations controlled by the user
- File system permissions protect state files
- No pickle files from external/untrusted sources

**Risk Level:** **LOW** (internal files only)

**Risk Escalation:** **HIGH** if loading pickle files from:
- Network sources
- User-uploaded files
- Shared/world-writable directories
- Untrusted USB drives or external media

**Current Mitigations:**
- Files use specific extensions (.pkl, .qontinui, .faiss)
- Stored in user-controlled project directories
- Documentation warns against loading external files
- No network loading of state files

**Future Mitigations (Phase 2):**
- HMAC signature verification
- Encrypted state files
- Version tagging to detect tampering
- Optional JSON-only mode for maximum security

**Example Safe Usage:**
```python
# SAFE: Loading state saved by Qontinui
vector_store = VectorStore()
vector_store.load("./project/state/vectors")  # Trusted local directory

# SAFE: Saving state for later use
serializer = PickleSerializer()
serializer.serialize(data, Path("./project/state/matches.pkl"))
```

**Example Unsafe Usage:**
```python
# UNSAFE: Loading pickle from untrusted source
import pickle
with open("/tmp/untrusted_file.pkl", "rb") as f:
    data = pickle.load(f)  # DON'T DO THIS

# UNSAFE: Loading user-uploaded pickle
user_file = request.files["upload"]
user_file.save("/tmp/upload.pkl")
with open("/tmp/upload.pkl", "rb") as f:
    pickle.load(f)  # NEVER DO THIS
```

**Recommended Safe Alternative:**

For scenarios requiring untrusted data exchange, use JSON serialization:

```python
# SAFE: JSON for untrusted/external data
from qontinui.persistence.serializers import JsonSerializer

serializer = JsonSerializer()
serializer.serialize(data, Path("shared_data.json"))
```

### 3. Path Operations

**Location:** `/src/qontinui/perception/matching.py`

**Purpose:** Load image files for template matching and pattern recognition.

**Security Implementation:**
```python
# Path validation and sanitization
def load_pattern_image(image_path: str | Path) -> Image:
    path = Path(image_path).resolve()

    # Extension validation
    if path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
        raise ValueError(f"Unsupported image format: {path.suffix}")

    # Path exists check
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    return Image.open(path)
```

**Assumptions:**
- Image paths come from automation scripts (trusted source)
- Scripts are written by developers, not end users
- File system permissions protect sensitive directories
- Image files are in project-controlled locations

**Risk Level:** **LOW** (trusted script sources)

**Risk Escalation:** **MEDIUM** if:
- Paths constructed from external input
- Symlinks to sensitive files
- World-writable directories
- No base directory constraints

**Mitigations:**
- File extension validation (.png, .jpg, .jpeg only)
- Path resolution to detect symlinks
- Existence checks before operations
- Future: Base directory restrictions

**Example Safe Usage:**
```python
# SAFE: Hard-coded path in script
matcher.load_pattern_image("./assets/button.png")

# SAFE: Path from project config
config = load_config("automation.yaml")
matcher.load_pattern_image(config["image_path"])
```

**Example Unsafe Usage:**
```python
# UNSAFE: User-provided path without validation
user_path = input("Enter image path: ")
matcher.load_pattern_image(user_path)  # Path traversal risk

# UNSAFE: Allowing arbitrary extensions
matcher.load_pattern_image("/etc/passwd")  # Blocked by extension check
```

### 4. File System Access

**Assumption:** Qontinui requires and assumes full file system access for automation tasks.

**Scope:**
- Read access to image assets, configuration files
- Write access to state files, logs, screenshots
- Execute access for launching applications (via HAL)

**Mitigation:**
- Run Qontinui with principle of least privilege
- Use dedicated user accounts for automation
- Implement OS-level access controls
- Container/VM isolation for untrusted scenarios

---

## Security Features

### Input Validation

#### File Extension Whitelisting
```python
# Image files: .png, .jpg, .jpeg only
ALLOWED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

# State files: .pkl, .qontinui
ALLOWED_STATE_EXTENSIONS = {'.pkl', '.qontinui'}
```

#### Path Sanitization
```python
# Resolve symlinks and relative paths
path = Path(user_input).resolve()

# Validate extension
if path.suffix not in ALLOWED_EXTENSIONS:
    raise ValueError("Invalid file type")
```

### Expression Safety

#### AST-Based Validation
```python
# Parse expression into Abstract Syntax Tree
tree = ast.parse(expression, mode="eval")

# Validate all nodes are in whitelist
for node in ast.walk(tree):
    if type(node) not in ALLOWED_NODES:
        raise ValueError(f"Unsafe operation: {type(node).__name__}")
```

#### Function Call Restrictions
```python
# Only whitelisted functions allowed
if isinstance(node, ast.Call):
    if isinstance(node.func, ast.Name):
        if node.func.id not in SAFE_FUNCTIONS:
            raise ValueError(f"Unsafe function: {node.func.id}")
```

### Dependency Management

- **Poetry lock file** with pinned versions
- **Dependabot** automated security updates
- **CI/CD scanning** for known vulnerabilities
- **Minimal dependencies** to reduce attack surface

---

## Known Security Considerations

### 1. Eval/Compile Usage (8 instances)

**Files:**
- `condition_evaluator.py` - Control flow conditions
- `evaluator.py` - Data operation expressions

**Status:** Accepted risk with mitigations

**Justification:**
- Required for flexible automation DSL
- Restricted to trusted developer scripts
- AST validation prevents dangerous operations
- Clear documentation of security model

**Additional Hardening:**
- Consider compiling expressions once and reusing
- Add optional "strict mode" with even more restrictions
- Implement expression caching with signature validation

### 2. Pickle Usage (4 instances)

**Files:**
- `serializers.py` - State persistence
- `matching.py` - Matcher state
- `vector_store.py` - Vector DB

**Status:** Accepted risk with future improvements planned

**Justification:**
- Only used for internally-generated state files
- Files stored in trusted project directories
- No network sources or user uploads
- Required for complex object serialization

**Phase 2 Improvements:**
- HMAC signature verification
- Encrypted state files
- Alternative JSON mode for simple data
- File integrity checking

### 3. Path Traversal (2 instances)

**Files:**
- `matching.py` - Image loading

**Status:** Mitigated

**Implemented Protections:**
- Extension validation
- Path resolution
- Existence checks

**Phase 2 Improvements:**
- Base directory validation
- Chroot-like restrictions
- Comprehensive path testing

### 4. False Positives (6 instances in test code)

**Status:** Documented and accepted

**Examples:**
- Test code using `exec()` for test case generation
- Mock objects in test fixtures
- Development utilities

**Justification:**
- Limited to test/development code
- Not included in production builds
- Clearly marked in security audit

---

## Best Practices

### For Developers (Writing Qontinui Scripts)

#### 1. Never Use Untrusted Input in Expressions

```python
# GOOD: Hard-coded expression
condition = "counter > 10"
evaluator.evaluate_condition(condition)

# BAD: User input in expression
user_input = get_user_input()
evaluator.evaluate_condition(user_input)  # NEVER DO THIS
```

#### 2. Validate File Paths

```python
# GOOD: Validate before use
from pathlib import Path

image_path = Path(config["image"])
if not image_path.suffix in [".png", ".jpg"]:
    raise ValueError("Invalid image format")
matcher.load_pattern_image(image_path)

# BAD: No validation
matcher.load_pattern_image(user_provided_path)
```

#### 3. Use JSON for Shared Data

```python
# GOOD: JSON for data exchange
with open("shared_data.json") as f:
    data = json.load(f)

# BAD: Pickle for shared data
with open("shared_data.pkl", "rb") as f:
    data = pickle.load(f)  # Only if YOU created this file
```

#### 4. Principle of Least Privilege

```bash
# GOOD: Dedicated user with minimal permissions
sudo -u automation python automation_script.py

# BAD: Running as root
sudo python automation_script.py  # Avoid if possible
```

#### 5. Isolate Untrusted Scenarios

```bash
# GOOD: Container isolation for untrusted workflows
docker run --rm -v $(pwd):/work qontinui:latest python workflow.py

# GOOD: VM isolation
vagrant up && vagrant ssh -c "python workflow.py"
```

### For Users (Running Qontinui)

#### 1. Only Run Trusted Scripts

```bash
# GOOD: Scripts from your team repository
git clone https://github.com/yourteam/automation.git
cd automation
python workflow.py

# BAD: Random scripts from internet
curl https://random-site.com/script.py | python  # NEVER DO THIS
```

#### 2. Keep Qontinui Updated

```bash
# GOOD: Regular updates
poetry update qontinui

# Check for security advisories
poetry show qontinui
```

#### 3. Secure State File Storage

```bash
# GOOD: Protected directory
mkdir -p ~/.qontinui/state
chmod 700 ~/.qontinui/state

# BAD: World-writable directory
chmod 777 ./state  # NEVER DO THIS
```

#### 4. Review Automation Scripts

```python
# Before running new scripts, review for:
# - External network connections
# - Unexpected file operations
# - Suspicious eval/exec usage
# - Pickle loading from untrusted sources
```

#### 5. Use Virtual Environments

```bash
# GOOD: Isolated environment
python -m venv venv
source venv/bin/activate
pip install qontinui

# Prevents dependency conflicts and limits scope
```

### For System Administrators

#### 1. Container/VM Isolation

```yaml
# Docker Compose example
services:
  automation:
    image: qontinui:latest
    read_only: true
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - NET_ADMIN  # Only if needed
```

#### 2. File System Permissions

```bash
# Dedicated automation user
sudo useradd -r -s /bin/false qontinui-bot

# Restrict state directory
sudo mkdir /var/lib/qontinui
sudo chown qontinui-bot:qontinui-bot /var/lib/qontinui
sudo chmod 700 /var/lib/qontinui
```

#### 3. Network Segmentation

```bash
# If network features added, isolate automation network
# Use firewall rules to restrict access
sudo iptables -A OUTPUT -m owner --uid-owner qontinui-bot -d 10.0.0.0/8 -j ACCEPT
sudo iptables -A OUTPUT -m owner --uid-owner qontinui-bot -j DROP
```

#### 4. Audit Logging

```python
# Enable comprehensive logging
import logging
logging.basicConfig(level=logging.INFO)

# Monitor for suspicious activity
# - Unexpected file access
# - Failed expression evaluations
# - Path traversal attempts
```

---

## Security Roadmap

### Phase 1 (Current)

- [x] AST-based expression validation
- [x] File extension whitelisting
- [x] Path resolution and validation
- [x] Security documentation
- [x] Clear security assumptions documented

### Phase 2 (Planned - Next 6 Months)

- [ ] **HMAC Signature Verification for Pickle**
  - Sign all saved state files
  - Verify signatures on load
  - Detect tampering

- [ ] **Encrypted State Files**
  - Optional encryption for sensitive state
  - Key management integration
  - Transparent encryption/decryption

- [ ] **Base Directory Validation**
  - Restrict file operations to project directory
  - Chroot-like constraints
  - Configurable allow-list

- [ ] **Expression Sandboxing**
  - More restrictive execution environment
  - Resource limits (CPU, memory)
  - Timeout enforcement

- [ ] **Security Audit Mode**
  - Log all security-relevant operations
  - Detect anomalies
  - Integration with SIEM systems

### Phase 3 (Future)

- [ ] **JSON-Only Mode**
  - Alternative to pickle for maximum security
  - Performance trade-offs documented
  - Gradual migration path

- [ ] **Network Security**
  - If remote features added
  - TLS/mTLS for all network communication
  - Certificate pinning

- [ ] **Code Signing**
  - Verify automation script integrity
  - Trust chain for script distribution
  - Revocation mechanism

- [ ] **Sandboxed Execution**
  - Optional container-based isolation
  - Seccomp/AppArmor profiles
  - Resource quotas

---

## Vulnerability Disclosure

### Reporting Security Issues

If you discover a security vulnerability in Qontinui, please report it responsibly:

**Email:** security@qontinui.dev

**Please Include:**
- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Suggested mitigation (if any)

**What to Expect:**
- Acknowledgment within 48 hours
- Status updates every 7 days
- Coordinated disclosure timeline
- Credit in security advisory (if desired)

### Security Advisory Process

1. **Receipt:** We acknowledge your report within 48 hours
2. **Validation:** We reproduce and validate the issue (1-7 days)
3. **Severity Assessment:** We assign a severity level using CVSS
4. **Fix Development:** We develop and test a fix (timeline varies)
5. **Disclosure:** We coordinate disclosure with you
6. **Release:** We release patched version with security advisory
7. **Credit:** We publicly credit you (unless you prefer anonymity)

### PGP Key

```
-----BEGIN PGP PUBLIC KEY BLOCK-----
[PGP key for encrypted vulnerability reports will be provided]
-----END PGP PUBLIC KEY BLOCK-----
```

### Hall of Fame

We recognize security researchers who help improve Qontinui's security:

- *[Your name could be here]*

---

## Security Testing

### Testing Security Features

Qontinui includes security-focused tests:

```bash
# Run security validation tests
pytest tests/security/ -v

# Test expression safety
pytest tests/security/test_safe_evaluator.py

# Test path validation
pytest tests/security/test_path_validation.py

# Test pickle safety
pytest tests/security/test_serialization_security.py
```

### Security Test Coverage

Current security test coverage includes:

- Expression AST validation
- Path traversal prevention
- File extension validation
- Pickle signature verification (Phase 2)
- Resource limit enforcement (Phase 2)

---

## Frequently Asked Questions

### Q: Is Qontinui safe to use in production?

**A:** Yes, when used as designed for its intended purpose: automating trusted applications with developer-written scripts. It is NOT safe for processing untrusted user input or multi-tenant environments without additional hardening.

### Q: Can I use Qontinui with untrusted input?

**A:** Not recommended. If you must, use container/VM isolation, implement additional input validation, and use JSON serialization instead of pickle.

### Q: Why use pickle if it's insecure?

**A:** Pickle is used for internal state files only, generated and consumed by Qontinui itself. It enables rich object serialization needed for complex state. Future versions will add signature verification and optional JSON mode.

### Q: Should I run security scans on Qontinui?

**A:** Yes! We encourage security testing. Please report any findings responsibly.

### Q: How often are dependencies updated?

**A:** Dependencies are reviewed and updated monthly. Critical security updates are applied within 48 hours of disclosure.

### Q: Can I disable eval/pickle for maximum security?

**A:** Currently, no. These features are core to Qontinui's design. Phase 3 will add a JSON-only mode for users who require maximum security at the cost of some functionality.

### Q: Is Qontinui suitable for security-critical applications?

**A:** Only with additional hardening: container isolation, input validation, audit logging, and regular security reviews. Consult a security professional for your specific use case.

---

## Conclusion

Qontinui prioritizes **usability for automation** while maintaining **reasonable security** for its intended use case. By understanding the security model, threat model, and best practices, you can use Qontinui safely and effectively.

**Remember:**
- Qontinui is a **developer tool**, not a web application
- Scripts are **trusted**, user input is **untrusted**
- Use **isolation** when processing anything untrusted
- Keep Qontinui and dependencies **updated**
- Report security issues **responsibly**

For questions or concerns, contact: security@qontinui.dev

---

**Document Version:** 1.0
**Last Updated:** 2025-10-28
**Next Review:** 2025-04-28
