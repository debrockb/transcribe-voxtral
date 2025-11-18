# Voxtral Test Suite

This directory contains the comprehensive test suite for the Voxtral transcription application.

## Test Structure

```
tests/
├── __init__.py                      # Test package initialization
├── conftest.py                      # Pytest fixtures and configuration
├── test_transcription_engine.py     # Unit tests for transcription engine
├── test_api.py                      # API endpoint tests
├── test_integration.py              # Integration tests
├── test_history.py                  # History management tests
├── test_platform_compatibility.py   # Cross-platform tests
├── README.md                        # This file
├── CI_CD_TESTING.md                # CI/CD documentation
└── PLATFORM_TESTING.md             # Platform testing guide
```

## Running Tests

### Prerequisites

**Important:** Tests require the `TESTING=1` environment variable to enable mocking and the `PYTHONPATH` to be set correctly.

```bash
cd transcribe-voxtral-main/VoxtralApp

# Set environment variables
export TESTING=1
export PYTHONPATH=/Users/ddbco/Desktop/Voxtral/transcribe-voxtral-main/VoxtralApp:$PYTHONPATH
```

**Note:** Adjust the `PYTHONPATH` to match your actual project location.

### Run All Tests

**Recommended (excludes model/GPU tests):**
```bash
test_venv/bin/pytest tests/ -v --tb=line -m "not requires_model and not requires_gpu and not slow"
```

**All tests (may require GPU/model):**
```bash
test_venv/bin/pytest tests/ -v
```

### Run Specific Test Categories

```bash
# Unit tests only
test_venv/bin/pytest -m unit -v

# API tests only
test_venv/bin/pytest -m api -v
test_venv/bin/pytest tests/test_api.py -v

# Integration tests only
test_venv/bin/pytest -m integration -v
test_venv/bin/pytest tests/test_integration.py -v

# Skip slow tests
test_venv/bin/pytest -m "not slow" -v

# Platform-specific tests
test_venv/bin/pytest tests/test_platform_compatibility.py -v

# Cross-platform tests
test_venv/bin/pytest -m cross_platform -v
```

### Run Specific Test Files

```bash
test_venv/bin/pytest tests/test_api.py -v
test_venv/bin/pytest tests/test_transcription_engine.py -v
test_venv/bin/pytest tests/test_integration.py -v
test_venv/bin/pytest tests/test_history.py -v
```

### Run Specific Test Classes or Methods

```bash
# Run specific test class
test_venv/bin/pytest tests/test_api.py::TestHealthEndpoint -v

# Run specific test method
test_venv/bin/pytest tests/test_api.py::TestHealthEndpoint::test_health_check -v
```

### Run with Coverage

```bash
test_venv/bin/pytest tests/ --cov=. --cov-report=html --cov-report=term-missing
# Open htmlcov/index.html to view coverage report
```

### Run with Different Output Modes

```bash
# Verbose output
test_venv/bin/pytest -v

# Extra verbose
test_venv/bin/pytest -vv

# Short traceback
test_venv/bin/pytest --tb=short

# Line-only traceback (recommended)
test_venv/bin/pytest --tb=line

# No traceback
test_venv/bin/pytest --tb=no
```

## Test Categories

Tests are marked with the following pytest markers (defined in `pytest.ini`):

- `@pytest.mark.unit` - Unit tests for individual components
- `@pytest.mark.api` - API endpoint tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Tests that take longer to run
- `@pytest.mark.windows` - Windows-specific tests
- `@pytest.mark.macos` - macOS-specific tests
- `@pytest.mark.linux` - Linux-specific tests
- `@pytest.mark.cross_platform` - Cross-platform compatibility tests
- `@pytest.mark.requires_model` - Tests requiring the actual ML model (skipped in CI)
- `@pytest.mark.requires_gpu` - Tests requiring GPU hardware (skipped in CI)

## Writing New Tests

### Test Naming Convention

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Using Fixtures

Common fixtures available in `conftest.py`:

```python
def test_example(app, client, socketio_client, temp_dir, sample_audio_file):
    """
    Available fixtures:
    - app: Flask test application instance
    - client: Flask HTTP test client
    - socketio_client: SocketIO test client for WebSocket testing
    - temp_dir: Temporary directory for test files (auto-cleanup)
    - sample_audio_file: Mock audio file for testing
    """
    response = client.get('/api/health')
    assert response.status_code == 200
```

### Example Test

```python
import pytest

@pytest.mark.unit
@pytest.mark.api
class TestMyFeature:
    """Test suite for my feature."""

    def test_endpoint(self, client):
        """Test a specific endpoint."""
        response = client.get('/api/endpoint')
        assert response.status_code == 200
        assert 'expected_key' in response.json

    def test_with_socketio(self, socketio_client):
        """Test WebSocket functionality."""
        socketio_client.emit('test_event', {'data': 'test'})
        received = socketio_client.get_received()
        assert len(received) > 0
```

### Adding Test Markers

```python
import pytest

@pytest.mark.slow
@pytest.mark.requires_model
def test_actual_transcription():
    """This test requires the actual model and takes time."""
    pass

@pytest.mark.cross_platform
def test_path_handling():
    """This test verifies cross-platform path handling."""
    pass
```

## Mock Strategy

The test suite uses extensive mocking to avoid requiring heavy dependencies in CI/CD:

### Mocked Dependencies

Defined in `conftest.py`:

- `torch` - PyTorch deep learning framework
- `transformers` - HuggingFace transformers
- `librosa` - Audio processing library
- `soundfile` - Audio file I/O
- `accelerate` - Model acceleration

### When Mocks Are Active

Mocks are automatically activated when `TESTING=1` environment variable is set. This allows:

- Tests to run without GPU
- Tests to run without downloading the 20GB model
- Fast test execution in CI/CD pipelines
- Tests on any hardware (including GitHub Actions runners)

### Testing with Real Dependencies

To test with real dependencies (requires model and GPU):

```bash
unset TESTING  # Disable mocks
test_venv/bin/pytest tests/ -v -m requires_model
```

## Continuous Integration

Tests are designed to run in CI/CD environments:

### GitHub Actions

Tests automatically run on:
- Multiple Python versions (3.9, 3.10, 3.11, 3.12)
- Multiple operating systems (Ubuntu, macOS, Windows)
- On every push and pull request

### CI Test Configuration

```bash
# Standard CI test run
TESTING=1 pytest tests/ -v -m "not requires_model and not requires_gpu and not slow"
```

This skips tests that require:
- Large model downloads
- GPU hardware
- Long execution time

For more details, see [CI_CD_TESTING.md](CI_CD_TESTING.md)

## Code Coverage

### Coverage Goals

- **Minimum**: 70% code coverage
- **Target**: 85%+ code coverage
- **Critical paths**: 95%+ coverage (API endpoints, transcription engine)

### Generate Coverage Report

```bash
test_venv/bin/pytest tests/ --cov=. --cov-report=html --cov-report=term-missing
```

### View Coverage Report

```bash
# macOS
open htmlcov/index.html

# Linux
xdg-open htmlcov/index.html

# Windows
start htmlcov/index.html
```

### Coverage Configuration

Coverage settings are defined in:
- `pytest.ini` - Pytest coverage options
- `pyproject.toml` - Coverage.py configuration

Files excluded from coverage:
- `tests/*` - Test files themselves
- `voxtral_env/*` - Virtual environment
- `*/__pycache__/*` - Python cache files

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError` when running tests

**Solutions:**
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH=/path/to/transcribe-voxtral-main/VoxtralApp:$PYTHONPATH

# Or run from VoxtralApp directory
cd transcribe-voxtral-main/VoxtralApp
test_venv/bin/pytest tests/
```

### Missing Dependencies

**Problem:** Import errors for pytest or test dependencies

**Solutions:**
```bash
# Install all development dependencies
test_venv/bin/pip install -r requirements-dev.txt

# Or install specific test dependencies
test_venv/bin/pip install pytest pytest-cov pytest-flask pytest-mock
```

### Mock Errors

**Problem:** Tests fail with "Mock object has no attribute..."

**Solutions:**
1. Ensure `TESTING=1` environment variable is set
2. Check that mocks are properly configured in `conftest.py`
3. For tests requiring real dependencies, unset `TESTING` and install full requirements

### Slow Tests

**Problem:** Tests take too long during development

**Solutions:**
```bash
# Skip slow tests
test_venv/bin/pytest -m "not slow" -v

# Run only fast unit tests
test_venv/bin/pytest -m "unit and not slow" -v

# Run specific test file
test_venv/bin/pytest tests/test_api.py -v
```

### Fixture Errors

**Problem:** Fixture not found or not working

**Solutions:**
1. Ensure you're in the correct directory
2. Check that `conftest.py` is in the tests directory
3. Verify fixture is properly defined with `@pytest.fixture` decorator
4. Check fixture scope (function, class, module, session)

## Best Practices

1. **Isolation**: Each test should be independent and not rely on other tests
2. **Cleanup**: Use fixtures for setup/teardown to ensure proper cleanup
3. **Mocking**: Mock external dependencies (models, APIs, file I/O when appropriate)
4. **Assertions**: Use clear, specific assertions with helpful error messages
5. **Documentation**: Add docstrings to test classes and complex tests
6. **Markers**: Use appropriate markers to categorize tests
7. **Naming**: Use descriptive test names that explain what is being tested
8. **Coverage**: Aim for high coverage but prioritize critical paths

### Example Best Practice Test

```python
import pytest
from pathlib import Path

@pytest.mark.unit
@pytest.mark.api
class TestFileUpload:
    """Test suite for file upload functionality."""

    def test_upload_valid_audio_file(self, client, sample_audio_file):
        """Test uploading a valid audio file succeeds."""
        with open(sample_audio_file, 'rb') as f:
            data = {'file': (f, 'test.mp3', 'audio/mpeg')}
            response = client.post('/api/upload', data=data, content_type='multipart/form-data')

        assert response.status_code == 200, "Upload should succeed"
        json_data = response.json
        assert 'file_id' in json_data, "Response should include file_id"
        assert json_data['filename'] == 'test.mp3', "Filename should be preserved"

    def test_upload_invalid_file_type(self, client, temp_dir):
        """Test uploading an invalid file type returns 400."""
        invalid_file = temp_dir / 'test.txt'
        invalid_file.write_text('Not an audio file')

        with open(invalid_file, 'rb') as f:
            data = {'file': (f, 'test.txt', 'text/plain')}
            response = client.post('/api/upload', data=data, content_type='multipart/form-data')

        assert response.status_code == 400, "Should reject invalid file type"
        assert 'error' in response.json, "Response should include error message"
```

## Platform-Specific Testing

For platform-specific tests and cross-platform compatibility testing, see:

- [PLATFORM_TESTING.md](PLATFORM_TESTING.md) - Platform testing guide
- Tests marked with `@pytest.mark.windows`, `@pytest.mark.macos`, or `@pytest.mark.linux`

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Flask Testing](https://flask.palletsprojects.com/en/latest/testing/)
- [Coverage.py](https://coverage.readthedocs.io/)
- [Flask-SocketIO Testing](https://flask-socketio.readthedocs.io/en/latest/testing.html)
- [pytest-mock](https://pytest-mock.readthedocs.io/)

## Getting Help

For test-related issues:

1. Check this README for common solutions
2. Review [CI_CD_TESTING.md](CI_CD_TESTING.md) for CI-specific issues
3. Check [PLATFORM_TESTING.md](PLATFORM_TESTING.md) for platform issues
4. Review test output carefully - pytest provides detailed error messages
5. Check `conftest.py` for available fixtures and mock configuration

---

Happy Testing! 🧪
