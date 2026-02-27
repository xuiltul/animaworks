# Engineer Specialty Guidelines

## Coding Principles

### Principle of Minimal Change
- Keep changes to existing code to the minimum necessary
- Perform large refactors only when explicitly instructed
- "Fix it while you're at it" is prohibited. Out-of-scope fixes should be logged as separate tasks

### Avoid Over-Engineering
- Follow YAGNI (You Aren't Gonna Need It) strictly
- Do not complicate code for hypothetical future extensions
- Consider abstraction only after the same pattern appears 3 times (Rule of Three)
- If a simple implementation meets requirements, it is best

### Security (OWASP Top 10 Awareness)
- Always validate user input
- Use parameter binding for SQL queries (no string concatenation)
- Never hardcode secrets (API keys, passwords) in code
- Use `pathlib.Path` for file path construction to prevent path traversal
- Avoid `shell=True` in subprocess calls; use argument lists

```python
# BAD
subprocess.run(f"ls {user_input}", shell=True)

# GOOD
subprocess.run(["ls", user_input], shell=False)
```

## Tool Usage Rules

### Prefer Dedicated Tools for File Operations
- File read: Use `Read` (not `cat`, `head`, `tail`)
- File edit: Use `Edit` (not `sed`, `awk`)
- File write: Use `Write` (not `echo >`, `cat <<EOF`)
- File search: Use `Glob` (not `find`, `ls`)
- Content search: Use `Grep` (not `grep`, `rg`)
- Use `Bash` only when no dedicated tool exists: Git, package management, build, test execution

### File Operation Best Practices
- Prefer editing existing files; avoid creating unnecessary files
- Check for existing suitable files before creating new ones
- Create documentation files (*.md, README) only when explicitly instructed

## Code Quality Standards

### Type Hints Required
```python
from __future__ import annotations

def process_item(name: str, count: int = 0) -> dict[str, int]:
    ...
```

- Use `str | None` (not `Optional[str]`)
- Always add type hints to function parameters and return values
- Use `TypeAlias` for complex types

### Path Operations
```python
from pathlib import Path

# BAD
import os
path = os.path.join(base_dir, "subdir", "file.txt")

# GOOD
path = Path(base_dir) / "subdir" / "file.txt"
```

### Docstrings (Google Style)
```python
def calculate_score(items: list[Item], weight: float = 1.0) -> float:
    """Calculate score.

    Args:
        items: List of items to evaluate.
        weight: Score weight coefficient.

    Returns:
        Calculated score value.

    Raises:
        ValueError: When items is empty.
    """
```

### Logging
```python
import logging
logger = logging.getLogger(__name__)

# Use logger, not print()
logger.info("Processing %d items", len(items))
```

### Data Models
- Use Pydantic Model or dataclass for data structure definitions
- Prefer structured models over raw dict manipulation

## Commit Convention

Use semantic commit format:
- `feat:` — New feature
- `fix:` — Bug fix
- `refactor:` — Refactoring (no functional change)
- `docs:` — Documentation only
- `test:` — Test addition or update
- `chore:` — Build config, dependencies, misc

```
feat: Add OAuth2 flow for user authentication
fix: Fix memory leak on session timeout
refactor: Consolidate database connection pool into singleton
```

## Test Guidelines

- After changing code, verify related tests pass
- Add unit tests for new functions and methods
- Place tests in `tests/` with the same structure as the target module
- Run tests with `pytest`, specifying tests related to the change

```bash
# Run specific test file
pytest tests/test_target_module.py -v

# Run only tests related to the change
pytest tests/test_target_module.py::TestClassName::test_method -v
```

## Error Handling

- Avoid bare `except:` or `except Exception:`; catch specific exceptions
- Include information needed to identify the issue in error messages
- Use exponential backoff for retry logic

```python
# BAD
try:
    result = api_call()
except:
    pass

# GOOD
try:
    result = api_call()
except ConnectionError as e:
    logger.warning("API connection failed (attempt %d/%d): %s", attempt, max_retries, e)
    raise
```

## Async Processing

- Use `async/await`; avoid blocking calls
- Use `asyncio.Lock()` for shared state
- Offload long CPU-bound work to `asyncio.to_thread()`
