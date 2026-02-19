# Contributing to AnimaWorks

Thank you for your interest in contributing to AnimaWorks.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all-tools,test]"
```

## Code Style

- `from __future__ import annotations` at the top of every file
- Type hints required (`str | None` style)
- Google-style docstrings
- `logger = logging.getLogger(__name__)`
- Semantic commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`

## License Headers

All source files must include the Apache 2.0 header:

```python
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
```

## Pull Request Process

1. Create a feature branch from `main`.
2. Add tests for new functionality.
3. Run `pytest` and ensure all tests pass.
4. Submit a pull request with a clear description.
