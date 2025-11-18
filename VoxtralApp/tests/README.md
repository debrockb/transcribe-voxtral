# Voxtral Test Suite

This directory contains the comprehensive test suite for the Voxtral transcription application.

## Test Structure

```
tests/
├── __init__.py                # Test package initialization
├── conftest.py               # Pytest fixtures and configuration
├── test_transcription_engine.py  # Unit tests for transcription engine
├── test_api.py               # API endpoint tests
├── test_integration.py       # Integration tests
└── test_history.py           # History management tests
```

## Running Tests

### Run All Tests

```bash
cd VoxtralApp
pytest
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest -m unit

# API tests only
pytest -m api

# Integration tests only
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

### Run Specific Test Files

```bash
pytest tests/test_api.py
pytest tests/test_transcription_engine.py
```

### Run with Coverage

```bash
pytest --cov=. --cov-report=html
# Open htmlcov/index.html to view coverage report
```

### Run with Verbose Output

```bash
pytest -v
pytest -vv  # Extra verbose
```

## Test Categories

Tests are marked with the following pytest markers:

- `@pytest.mark.unit` - Unit tests for individual components
- `@pytest.mark.api` - API endpoint tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Tests that take longer to run

## Writing New Tests

### Test Naming Convention

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Using Fixtures

Common fixtures available in `conftest.py`:

```python
def test_example(client, temp_dir, sample_audio_file):
    # client: Flask test client
    # temp_dir: Temporary directory for test files
    # sample_audio_file: Sample audio file for testing
    pass
```

### Example Test

```python
import pytest

@pytest.mark.unit
class TestMyFeature:
    def test_something(self, client):
        response = client.get('/api/endpoint')
        assert response.status_code == 200
```

## Continuous Integration

Tests are automatically run on GitHub Actions for:
- Multiple Python versions (3.9, 3.10, 3.11)
- Multiple operating systems (Ubuntu, macOS, Windows)
- On every push and pull request

## Code Coverage

We aim for:
- **Minimum**: 70% code coverage
- **Target**: 85%+ code coverage
- **Critical paths**: 95%+ coverage

View coverage report:
```bash
pytest --cov=. --cov-report=html
open htmlcov/index.html  # macOS
```

## Troubleshooting

### Import Errors

Make sure you're in the VoxtralApp directory:
```bash
cd VoxtralApp
export PYTHONPATH=$PYTHONPATH:$(pwd)
pytest
```

### Missing Dependencies

Install test dependencies:
```bash
pip install -r requirements.txt
# Or for all dev dependencies:
pip install -r requirements-dev.txt
```

### Slow Tests

Skip slow tests during development:
```bash
pytest -m "not slow"
```

## Best Practices

1. **Isolation**: Each test should be independent
2. **Cleanup**: Use fixtures for setup/teardown
3. **Mocking**: Mock external dependencies (models, APIs)
4. **Assertions**: Use clear, specific assertions
5. **Documentation**: Add docstrings to test classes and complex tests

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Flask Testing](https://flask.palletsprojects.com/en/latest/testing/)
- [Coverage.py](https://coverage.readthedocs.io/)
