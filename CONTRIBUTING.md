# Contributing to DeepArticle

First off, thank you for considering contributing! 🎉 This project is open to
everyone, and contributions of all kinds are welcome — bug reports, feature
ideas, documentation, and code.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)

## Code of Conduct

This project follows our [Code of Conduct](CODE_OF_CONDUCT.md). By
participating, you are expected to uphold it.

## How Can I Contribute?

### 🐛 Reporting Bugs

Open an issue using the **Bug Report** template. Include:
- A clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Your environment (OS, Python version, LLM provider)

### 💡 Suggesting Features

Open an issue using the **Feature Request** template. Explain the use case and
why it would be valuable.

### 🔧 Contributing Code

Good first contributions:
- Adding a new paper source (e.g. OpenAlex, DBLP, CORE)
- Improving error handling and logging
- Adding caching for API calls
- Writing tests
- Improving documentation

## Development Setup

```bash
# 1. Fork & clone the repo
git clone https://github.com/<your-username>/DeepArticle.git
cd DeepArticle

# 2. Create a virtual environment
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 3. Install in editable mode with dev tools
pip install -e ".[dev]"
# or
pip install -r requirements-dev.txt

# 4. Copy the env file and add at least one LLM API key
cp .env.example .env

# 5. Run the tests
pytest tests/ -v
```

## Pull Request Process

1. Create a branch from `main`: `git checkout -b feature/my-feature`
2. Make your changes and **add tests** for new behavior.
3. Make sure all tests pass: `pytest tests/`
4. Run the linter: `ruff check .`
5. Commit with a clear message (see below).
6. Push and open a Pull Request against `main`, filling in the PR template.

### Commit Messages

Use clear, imperative messages:
- `Add OpenAlex paper source`
- `Fix rate-limit handling in Semantic Scholar tool`
- `Update README installation steps`

## Style Guidelines

- Follow [PEP 8](https://peps.python.org/pep-0008/); the project uses `ruff`.
- Add type hints to new functions.
- Keep functions small and focused; match the surrounding code style.
- Document public functions with docstrings.

Thank you for helping make DeepArticle better! 🙏
