# Contributing to AnimaWorks

Thank you for your interest in contributing to AnimaWorks! This guide will help you get started quickly.

## Development Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Clone and Install

```bash
git clone https://github.com/xuiltul/animaworks.git
cd animaworks

# Using uv (recommended)
uv sync

# Or using pip
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all-tools,test]"
```

### Start the Server

```bash
python main.py start --host 0.0.0.0 --port 18500
```

## Running Tests

```bash
# Run unit tests (fast, no external services)
pytest tests/unit/ -x -q -m "not live and not azure and not ollama"

# Run full test suite
pytest

# Run with coverage
pytest --cov=core --cov=server
```

## Linting

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Check for lint errors
ruff check core/ cli/ server/

# Auto-fix lint errors
ruff check core/ cli/ server/ --fix

# Format code
ruff format core/ cli/ server/
```

Configuration: `pyproject.toml` — line length 120, target Python 3.12.

## Code Style

- `from __future__ import annotations` at the top of every file
- Type hints required (`str | None` style)
- Google-style docstrings
- `logger = logging.getLogger(__name__)`

## Finding Issues

Look for issues labeled [`good first issue`](https://github.com/xuiltul/animaworks/labels/good%20first%20issue) — these are great starting points for new contributors.

## Third-Party Service Integrations

We don't accept pull requests that add a backend, provider, or preset for a hosted gateway, aggregator, or reseller of AI models (LLM, image, video, or audio). This includes optional ones that are off by default. Each one is code we would maintain against an API we don't use, and anyone who turns it on sends their Animas' prompts, memories, or images to that service. We close these PRs without a code review (see #438 and #445).

If AnimaWorks is missing something you need from a model or service, open an issue that describes the problem first.

If you work for or are paid by a service your change touches, say so in the PR description.

## Pull Request Process

### Branch Naming

Use descriptive branch names:
- `fix/issue-42-fix-theme-toggle`
- `feat/add-health-endpoint`
- `docs/improve-contributing`

### Commit Messages

Use semantic commit prefixes:
- `feat:` — new feature
- `fix:` — bug fix
- `refactor:` — code restructuring
- `docs:` — documentation changes
- `test:` — adding or updating tests

### PR Checklist

1. Create a feature branch from `main`
2. Add tests for new functionality
3. Run `pytest` and ensure all tests pass
4. Run `ruff check` and `ruff format` — no lint errors
5. Submit a PR with a clear description of what changed and why
6. Reference the issue: `Fixes #N` or `Closes #N`

## License Headers

All source files must include the Apache 2.0 header:

```python
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
```
