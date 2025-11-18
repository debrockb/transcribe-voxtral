---
description: Run the test suite for Voxtral application
---

Run the complete test suite excluding model-dependent tests using pytest. Use the test_venv virtual environment and generate a coverage report.

Execute:
```bash
test_venv/bin/pytest tests/ -v --tb=short -m "not requires_model and not requires_gpu and not slow"
```

After running tests, provide a summary of:
- Number of tests passed/failed/skipped
- Any failing tests with error details
- Test coverage percentage
- Recommendations for fixing any failures
