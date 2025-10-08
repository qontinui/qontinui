# Instructions for AI Assistants (Claude/Other LLMs)

## CRITICAL: No Backward Compatibility Policy

**Qontinui is in active development. DO NOT ADD BACKWARD COMPATIBILITY CODE.**

- This project does NOT maintain backward compatibility between development versions
- Never add "legacy" support, compatibility aliases, or migration helpers
- Never keep old method/class names alongside new ones
- Never add comments about "backward compatibility" or "migration"
- If an API changes, update ALL code to use the new API immediately
- Remove old code completely when replacing it with new implementations

This is a development project, not a released library. Clean code is more important than compatibility with intermediate development stages.

## Project Overview
You are working on migrating Brobot (Java, ~1M LOC) to Qontinui (Python). This is a systematic, class-by-class migration that must preserve Brobot's architecture and behavior.

## Key Documents
- **[unified_plan.md](../unified_plan.md)** - High-level migration strategy and architecture
- **[migration_notes.md](../migration_notes.md)** - Detailed migration notes and important information
- **[BROBOT_MIGRATION_TRACKER.md](BROBOT_MIGRATION_TRACKER.md)** - Class-by-class migration status

## Migration Approach

### DO:
1. **Preserve Brobot's architecture** - Don't reinvent, translate faithfully
2. **Port class-by-class** - Complete one class fully before moving to next
3. **Maintain exact behavior** - Brobot tests should pass with minimal changes
4. **Use Python idioms** - While preserving architecture, use Pythonic patterns
5. **Document mappings** - Note any necessary changes in behavior
6. **Fix migration mistakes immediately** - When you find errors, fix them properly
7. **Use the latest patterns** - Always use the most current API design

### DON'T:
1. **Don't create new architectures** - This is a port, not a rewrite
2. **Don't skip ActionConfig/Options** - These control timing and behavior
3. **Don't combine atomic actions** - Keep Brobot's action separation
4. **Don't guess at implementation** - Ask for Brobot source when needed
5. **Don't create helper methods that don't exist in Brobot** - Stay faithful to the original
6. **NEVER add backward compatibility** - No legacy code, no compatibility layers, no migration helpers
7. **NEVER keep old APIs alongside new ones** - Remove old code completely when replacing

## Working on a Class

When migrating a Brobot class:

1. **Check the tracker** - See current status in BROBOT_MIGRATION_TRACKER.md
2. **Request the Java source** - Ask user for the specific Brobot class
3. **Create Python equivalent** - In the appropriate package
4. **Port all methods** - Including private/protected helpers
5. **Port inner classes** - Maintain the same structure
6. **Update the tracker** - Mark class as ✅ when complete

## Example Migration Pattern

```python
# Java (Brobot)
public class ActionOptions extends ActionConfig {
    private double pauseBefore = 0.0;
    private double pauseAfter = 0.0;

    public ActionOptions pauseBefore(double seconds) {
        this.pauseBefore = seconds;
        return this;
    }
}

# Python (Qontinui)
class ActionOptions(ActionConfig):
    """Direct port of Brobot's ActionOptions."""

    def __init__(self):
        super().__init__()
        self._pause_before = 0.0
        self._pause_after = 0.0

    def pause_before(self, seconds: float) -> 'ActionOptions':
        """Fluent setter for pause_before."""
        self._pause_before = seconds
        return self
```

## Package Mapping

| Brobot Package | Qontinui Package | Notes |
|---------------|------------------|-------|
| io.github.jspinak.brobot.actions | qontinui.actions | Core action system |
| io.github.jspinak.brobot.state | qontinui.state_management | State management |
| io.github.jspinak.brobot.find | qontinui.find | Find operations |
| io.github.jspinak.brobot.imageUtils | qontinui.image_utils | Image utilities |
| io.github.jspinak.brobot.datatypes | qontinui.datatypes | Core data types |

## Testing Requirements

For each migrated class:
1. Port the corresponding Brobot unit tests
2. Ensure behavioral compatibility
3. Add Python-specific tests for edge cases
4. Document any unavoidable differences

## Common Patterns to Preserve

### 1. Fluent Interfaces
```python
# Preserve Brobot's fluent API
result = (action
    .click(location)
    .pause_after(0.5)
    .then()
    .type_text("hello"))
```

### 2. Options Classes
```python
# Every action has an Options class
class ClickOptions(ActionOptions):
    def __init__(self):
        super().__init__()
        self.click_type = ClickType.LEFT
```

### 3. Find Operations
```python
# Preserve Find's builder pattern
matches = (Find(image)
    .similarity(0.95)
    .search_region(region)
    .find_all())
```

## Current Focus Areas

1. **Actions Package** - Core action system with Options
2. **State Management** - State, StateTransition, StateManager
3. **Find System** - Image matching and location
4. **DataTypes** - Region, Location, Image, etc.

## Questions to Ask

When you need clarification:
1. "Can you provide the Brobot source for [ClassName]?"
2. "What is the expected behavior for [method] in [scenario]?"
3. "Are there Brobot tests for [ClassName] I should reference?"
4. "Should [Java pattern] be adapted to Python idiom or preserved?"

## Common Migration Mistakes to Avoid

1. **Adding execute() when Brobot uses perform()** - Use exact method names
2. **Creating resolve/helper methods** - Only port methods that exist in Brobot
3. **Adding "backward compatibility"** - This is a fresh port, not maintaining old code
4. **Simplifying complex patterns** - Keep Brobot's architecture even if complex
5. **Mixing responsibilities** - Keep separation of concerns from Brobot

## Validation Checklist

Before marking a class as complete:
- [ ] All public methods ported with EXACT names
- [ ] All private/protected methods ported
- [ ] Inner classes ported
- [ ] Options/Config classes ported
- [ ] Tests ported and passing
- [ ] Behavior validated against Brobot
- [ ] Python docstrings added
- [ ] Type hints complete
- [ ] NO extra helper methods added
- [ ] NO "legacy" or "compatibility" code
- [ ] Updated BROBOT_MIGRATION_TRACKER.md

## Git Commit Messages

**Joshua Spinak is the sole contributor to this project.**

- DO NOT add "Co-Authored-By: Claude" or similar lines
- DO NOT add "Generated with Claude Code" or similar attribution
- Keep commit messages professional and focused on the changes
- Use conventional commit format (e.g., "feat:", "fix:", "docs:", "refactor:")

## Remember

This is a **faithful port**, not a rewrite. When in doubt, preserve Brobot's design decisions. The goal is behavioral compatibility with Brobot.

**Never add backward compatibility code.** Qontinui is in active development - clean code and correct implementation are the priorities.
