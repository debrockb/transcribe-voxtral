# Platform-Specific Testing Guide

This document describes the platform-specific testing strategy for the Voxtral application.

## Overview

The Voxtral application is designed to run on:
- **Windows** (Windows 10/11)
- **macOS** (macOS 10.15+, including Apple Silicon)
- **Linux** (Ubuntu, Debian, Fedora, etc.)

## Test Categories

### Cross-Platform Tests (`@pytest.mark.cross_platform`)

Tests that verify functionality works identically across all platforms:
- Path handling (forward slashes vs backslashes)
- File operations (read/write, binary/text)
- Environment variables
- API endpoints
- Database operations

**Example:**
```python
@pytest.mark.cross_platform
def test_file_upload_works_on_all_platforms(client, sample_audio_file):
    # This test runs on all platforms and should behave identically
    pass
```

### Platform-Specific Tests

#### Windows Tests (`@pytest.mark.windows`)
- Windows batch file validation
- Drive letter handling (C:, D:, etc.)
- CRLF line ending handling
- Windows-specific path formats
- Backslash path separators

**Example:**
```python
@pytest.mark.windows
@pytest.mark.skipif(not IS_WINDOWS, reason="Windows-specific test")
def test_batch_file_execution():
    # Only runs on Windows
    pass
```

#### macOS Tests (`@pytest.mark.macos`)
- Shell script validation (.sh, .command files)
- Executable permissions
- .DS_Store handling
- MPS (Metal Performance Shaders) device detection
- macOS-specific file attributes

**Example:**
```python
@pytest.mark.macos
@pytest.mark.skipif(not IS_MACOS, reason="macOS-specific test")
def test_shell_script_executable():
    # Only runs on macOS
    pass
```

#### Linux Tests (`@pytest.mark.linux`)
- Shell script validation
- Case-sensitive filesystem handling
- Linux-specific permissions
- Package management compatibility

**Example:**
```python
@pytest.mark.linux
@pytest.mark.skipif(not IS_LINUX, reason="Linux-specific test")
def test_case_sensitive_files():
    # Only runs on Linux
    pass
```

## Running Platform-Specific Tests

### Run All Cross-Platform Tests
```bash
pytest -m cross_platform
```

### Run Windows-Specific Tests
```bash
# Only runs on Windows
pytest -m windows
```

### Run macOS-Specific Tests
```bash
# Only runs on macOS
pytest -m macos
```

### Run Linux-Specific Tests
```bash
# Only runs on Linux
pytest -m linux
```

### Run Tests for Current Platform
```bash
# Automatically runs appropriate tests for your platform
pytest tests/test_platform_compatibility.py
```

## CI/CD Platform Testing

The GitHub Actions CI/CD pipeline tests on all three platforms:

### Test Matrix
- **Ubuntu Latest** (Linux)
  - Python 3.9, 3.10, 3.11
  - Runs: linux + cross_platform tests

- **macOS Latest**
  - Python 3.9, 3.10, 3.11
  - Runs: macos + cross_platform tests
  - Tests Apple Silicon (MPS) compatibility

- **Windows Latest**
  - Python 3.9, 3.10, 3.11
  - Runs: windows + cross_platform tests

### Viewing Platform Results

Check the GitHub Actions tab to see results for each platform:
```
https://github.com/debrockb/transcribe-voxtral/actions
```

Each platform's results are shown separately in the test matrix.

## Platform-Specific Features

### File Paths

**Windows:**
```python
# Use pathlib.Path for cross-platform compatibility
from pathlib import Path
path = Path("C:/Users/User/file.txt")  # pathlib handles this correctly
```

**macOS/Linux:**
```python
path = Path("/Users/user/file.txt")
```

**Cross-Platform:**
```python
# Always use pathlib.Path
path = Path.home() / "Documents" / "file.txt"
```

### Startup Scripts

**Windows:** `start_web.bat`, `run_tests.bat`
- Use `\` for paths
- Use `set` for variables
- Line endings: CRLF (`\r\n`)

**macOS/Linux:** `start_web.sh`, `run_tests.sh`
- Use `/` for paths
- Shebang: `#!/bin/bash`
- Executable permission required: `chmod +x script.sh`
- Line endings: LF (`\n`)

### Environment Activation

**Windows:**
```batch
voxtral_env\Scripts\activate.bat
voxtral_env\Scripts\python.exe
```

**macOS/Linux:**
```bash
source voxtral_env/bin/activate
voxtral_env/bin/python3
```

## Testing Checklist

When adding new features, ensure:

- [ ] Feature works on all three platforms
- [ ] File paths use `pathlib.Path`
- [ ] No hardcoded path separators (`/` or `\`)
- [ ] Cross-platform tests cover the feature
- [ ] Platform-specific code is marked with appropriate tests
- [ ] Startup scripts updated for all platforms
- [ ] Documentation mentions platform requirements

## Common Platform Issues

### Issue: Tests fail on Windows due to file locks
**Solution:** Ensure files are properly closed after use:
```python
with open(file_path, 'r') as f:
    content = f.read()
# File is automatically closed
```

### Issue: Shell scripts don't run on macOS/Linux
**Solution:** Check executable permissions:
```bash
chmod +x start_web.sh
chmod +x run_tests.sh
```

### Issue: Path separators cause failures
**Solution:** Always use `pathlib.Path`:
```python
# Bad
path = "folder/subfolder/file.txt"  # Fails on Windows

# Good
from pathlib import Path
path = Path("folder") / "subfolder" / "file.txt"  # Works everywhere
```

### Issue: Line endings cause script failures
**Solution:** Configure git to handle line endings:
```bash
# .gitattributes
*.sh text eol=lf
*.bat text eol=crlf
```

## Device Detection by Platform

### CUDA (NVIDIA GPUs)
- Available on: Windows, Linux
- Rarely on: macOS (deprecated)

### MPS (Apple Silicon)
- Available on: macOS (M1, M2, M3 chips)
- Not available on: Windows, Linux

### CPU
- Available on: All platforms
- Fallback device

**Test Coverage:**
```python
@pytest.mark.cross_platform
def test_device_detection():
    # Tests CPU, CUDA, and MPS detection
    # Mocks different scenarios for each platform
    pass
```

## Continuous Improvement

Platform testing is continuously improved by:
1. Monitoring CI/CD failures across platforms
2. Adding tests for platform-specific bugs
3. Updating this documentation with new findings
4. Testing on actual hardware when possible

## Resources

- [Python pathlib documentation](https://docs.python.org/3/library/pathlib.html)
- [pytest platform markers](https://docs.pytest.org/en/latest/how-to/mark.html)
- [GitHub Actions matrix testing](https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs)
