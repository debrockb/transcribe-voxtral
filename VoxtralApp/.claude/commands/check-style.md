---
description: Run code style and linting checks
---

Run code quality checks using flake8, black, and isort:

1. Run flake8 linter:
```bash
test_venv/bin/flake8 . --config=.flake8
```

2. Check code formatting with black:
```bash
test_venv/bin/black --check app.py config_manager.py transcribe_voxtral.py transcription_engine.py tests/*.py
```

3. Check import sorting with isort:
```bash
test_venv/bin/isort --check-only app.py config_manager.py transcribe_voxtral.py transcription_engine.py tests/*.py --skip test_venv --skip voxtral_env
```

If any issues are found, ask the user if they want to auto-fix them using:
- `test_venv/bin/black <files>`
- `test_venv/bin/isort <files>`

Provide a summary of:
- Style violations found
- Formatting issues
- Suggestions for fixes
