# CI/CD Testing Strategy

## Problem: GPU and Model Requirements

The Voxtral application uses a large ML model (~20GB) that requires:
- **Model Download**: 10-60 minutes initial download
- **GPU Memory**: Significant VRAM for CUDA/MPS
- **System RAM**: Multiple GB for model loading

**GitHub Actions runners do NOT have:**
- GPU hardware (no CUDA, no MPS)
- Sufficient resources to download/load 20GB models
- Extended time limits for model downloads

## Solution: Mocking Strategy

### Automatic Mocking

All tests automatically mock heavy dependencies via `conftest.py`:

```python
@pytest.fixture(autouse=True)
def mock_heavy_dependencies():
    """
    Automatically mock heavy ML dependencies.
    Prevents model loading in CI/CD environments.
    """
    # Mocks: torch, transformers, model loading
```

### Test Environment Variable

Tests set `TESTING=1` environment variable:

```python
# In conftest.py
os.environ['TESTING'] = '1'

# In app.py
if os.environ.get('TESTING') == '1':
    # Skip model initialization
    transcription_engine = MagicMock()
```

## Test Markers for GPU/Model Tests

### `@pytest.mark.requires_model`
Tests that need the actual ML model loaded:
```python
@pytest.mark.requires_model
def test_real_transcription():
    # Skipped in CI/CD
    pass
```

### `@pytest.mark.requires_gpu`
Tests that need GPU hardware:
```python
@pytest.mark.requires_gpu
def test_cuda_performance():
    # Skipped in CI/CD
    pass
```

## CI/CD Test Execution

GitHub Actions runs tests with:
```bash
pytest -m "not slow and not requires_model and not requires_gpu"
```

This skips:
- ❌ Slow tests
- ❌ Tests requiring actual model
- ❌ Tests requiring GPU hardware

## What Gets Tested in CI/CD

✅ **API Endpoints** - All REST endpoints
✅ **File Operations** - Upload, download, delete
✅ **Path Handling** - Cross-platform paths
✅ **History Management** - CRUD operations
✅ **Platform Compatibility** - Windows/macOS/Linux
✅ **Configuration** - Settings and environment
✅ **Error Handling** - Exception handling
✅ **WebSocket** - Connection and events
✅ **Mocked Transcription** - Using MagicMock

## What Requires Manual Testing

⚠️ **Real Model Loading** - Requires 20GB download
⚠️ **GPU Acceleration** - Requires CUDA/MPS hardware
⚠️ **Actual Transcription** - End-to-end with real audio
⚠️ **Performance Tests** - Under real load
⚠️ **Large File Processing** - Multi-GB audio files

## Running Tests Locally

### Without Model (like CI/CD)
```bash
export TESTING=1
pytest -m "not requires_model and not requires_gpu"
```

### With Model (full test suite)
```bash
unset TESTING
pytest  # Runs ALL tests including model-dependent ones
```

### Specific Categories
```bash
# Only API tests (no model needed)
pytest -m api

# Only platform tests (no model needed)
pytest -m cross_platform

# Only unit tests (mocked)
pytest -m unit
```

## CI/CD Pipeline Stages

### 1. Test Stage
- **Platforms**: Ubuntu, macOS, Windows
- **Python**: 3.9, 3.10, 3.11
- **Mocking**: Enabled (`TESTING=1`)
- **Skips**: `requires_model`, `requires_gpu`, `slow`

### 2. Lint Stage
- **Platform**: Ubuntu only
- **Checks**: Black, isort, flake8
- **No model needed**

### 3. Build Stage
- **Platform**: Ubuntu only
- **Verifies**: Imports, dependencies
- **Mocking**: Enabled

### 4. Integration Test Stage
- **Platform**: Ubuntu only
- **Tests**: Integration workflows
- **Mocking**: Enabled
- **Skips**: Model-dependent tests

## Mock Behavior

### Mocked Transcription Engine
```python
# Returns mocked data instead of real transcription
transcription_engine.transcribe_file(audio, output, "en")
# Writes: "Mocked transcription" to output file
```

### Mocked Device Detection
```python
# Always returns CPU in test mode
engine.device  # Returns: "cpu"
```

### Mocked Progress Callbacks
```python
# Callbacks are called but with fake progress
callback({"status": "processing", "progress": 50})
```

## Benefits of This Approach

✅ **Fast CI/CD** - Tests run in <5 minutes instead of hours
✅ **No GPU Needed** - Runs on standard GitHub runners
✅ **Cross-Platform** - Tests on all three platforms
✅ **Reliable** - No dependency on external model downloads
✅ **Cost Effective** - Uses free GitHub Actions tier
✅ **Early Detection** - Catches bugs in API, logic, paths

## Limitations

⚠️ **Not Testing Actual Transcription** - Mocked results only
⚠️ **Not Testing GPU Code Paths** - CUDA/MPS untested
⚠️ **Not Testing Model Loading** - Transformers mocked
⚠️ **Not Testing Real Performance** - Speed/memory untested

These require manual testing with actual hardware and model.

## Manual Testing Checklist

For release testing, verify manually:

- [ ] Model downloads successfully
- [ ] GPU detection works (CUDA/MPS/CPU)
- [ ] Real audio transcription produces good results
- [ ] Large files (>1GB) process correctly
- [ ] Memory usage is acceptable
- [ ] Performance meets expectations
- [ ] All supported languages work
- [ ] Video-to-audio conversion works (if ffmpeg installed)

## Future Improvements

Potential enhancements:
- [ ] Self-hosted runner with GPU for monthly full tests
- [ ] Smaller test model for actual transcription tests
- [ ] Performance benchmarks on dedicated hardware
- [ ] Integration with cloud GPU providers for testing
- [ ] Caching of model weights in CI (if feasible)

## Troubleshooting

### "Model download failed" in CI/CD
**Expected** - Model downloads are skipped in CI/CD. Tests should be mocked.

### "CUDA not available" in CI/CD
**Expected** - GitHub runners don't have GPUs. Tests use CPU mocking.

### "Tests pass in CI but fail locally"
Check if `TESTING=1` is set. Unset it for local model testing:
```bash
unset TESTING
pytest
```

### "Import errors in tests"
Ensure `PYTHONPATH` is set:
```bash
export PYTHONPATH=/path/to/VoxtralApp
pytest
```

## Summary

The test suite uses **intelligent mocking** to enable fast, reliable CI/CD testing without requiring expensive GPU hardware or large model downloads. This allows us to catch bugs early while maintaining reasonable CI/CD costs and execution times.

Manual testing with real hardware and models is still required before releases to verify end-to-end functionality.
