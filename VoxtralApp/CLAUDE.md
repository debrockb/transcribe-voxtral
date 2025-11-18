# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Voxtral is a Flask-based web application for audio/video transcription using the Mistral AI Voxtral-Mini-3B model. The application runs entirely locally (no cloud/API costs), automatically detects the best available hardware (MPS for Apple Silicon, CUDA for NVIDIA GPUs, or CPU), and provides a real-time web interface for transcription.

**Key directories:**
- `transcribe-voxtral-main/VoxtralApp/` - Main application code
- `transcribe-voxtral-main/` - Contains original CLI script and documentation

## Architecture

### Core Components

**TranscriptionEngine** (`transcription_engine.py`)
- Encapsulates all Voxtral model interactions
- Handles device detection (MPS/CUDA/CPU) and model loading
- Processes audio in 2-minute chunks to manage memory
- Supports progress callbacks for real-time UI updates
- Cross-platform compatible (Windows & macOS)

**Flask Application** (`app.py`)
- REST API endpoints for file upload, transcription jobs, and status
- WebSocket (SocketIO) for real-time progress streaming
- Background threading for non-blocking transcription
- Job tracking with in-memory storage (dict-based)
- Video-to-audio conversion support (moviepy)

**Processing Pipeline:**
```
Upload File → Validate → (Convert if video) → Create Job →
Background Thread → Load into 2min chunks → Process each chunk →
Emit progress via WebSocket → Save transcript → Return result
```

### Important Design Patterns

1. **Progress Callbacks**: TranscriptionEngine uses callbacks to report progress without coupling to Flask/SocketIO
2. **Background Processing**: Long-running transcriptions run in threads to avoid blocking the web server
3. **Chunk-based Processing**: Large audio files are split into 2-minute chunks to prevent memory overflow
4. **Device Abstraction**: Device detection is centralized and automatic (no manual configuration needed)

### Cross-Platform Considerations

- Use `pathlib.Path` for all file paths (never hardcode `/` or `\`)
- Both `.sh` (Unix) and `.bat` (Windows) scripts provided for startup
- Line endings handled by Git (CRLF conversion)
- FFmpeg required for video conversion (platform-specific installation)

## Development Commands

### Environment Setup

```bash
# The virtual environment is in voxtral_env/ (not test_venv/)
source transcribe-voxtral-main/VoxtralApp/voxtral_env/bin/activate  # macOS/Linux
# OR
transcribe-voxtral-main\VoxtralApp\voxtral_env\Scripts\activate.bat  # Windows
```

### Running the Web Application

```bash
cd transcribe-voxtral-main/VoxtralApp
python app.py
# Access at http://localhost:5000
```

Or use startup scripts:
```bash
./start_web.sh  # macOS/Linux
start_web.bat   # Windows
```

### Testing

**Run all tests (excluding model/GPU tests):**
```bash
cd transcribe-voxtral-main/VoxtralApp
export TESTING=1  # Important: enables test mocking
export PYTHONPATH=/Users/ddbco/Desktop/Voxtral/transcribe-voxtral-main/VoxtralApp:$PYTHONPATH
test_venv/bin/pytest tests/ -v --tb=line -m "not requires_model and not requires_gpu and not slow"
```

**Run specific test types:**
```bash
# API tests only
test_venv/bin/pytest tests/test_api.py -v

# Integration tests
test_venv/bin/pytest tests/ -v -m integration

# Platform compatibility tests
test_venv/bin/pytest tests/test_platform_compatibility.py -v
```

**Test markers available:**
- `unit` - Unit tests for individual components
- `integration` - Integration tests
- `api` - API endpoint tests
- `slow` - Long-running tests
- `requires_model` - Tests needing the actual ML model (skipped in CI)
- `requires_gpu` - Tests requiring GPU hardware (skipped in CI)
- `cross_platform` - Platform compatibility tests

### Code Quality

**Linting:**
```bash
cd transcribe-voxtral-main/VoxtralApp
test_venv/bin/flake8 . --config=.flake8
```

**Formatting:**
```bash
# Check formatting
test_venv/bin/black --check .

# Auto-format
test_venv/bin/black app.py transcribe_voxtral.py transcription_engine.py tests/*.py

# Sort imports
test_venv/bin/isort app.py transcribe_voxtral.py transcription_engine.py tests/*.py --skip test_venv
```

**Configuration:**
- Line length: 127 characters (black, flake8, isort all aligned)
- Style: black-compatible
- Config files: `.flake8`, `pyproject.toml`, `pytest.ini`

## Key Implementation Details

### Model Loading

- Model: `mistralai/Voxtral-Mini-3B-2507` (~20GB download on first run)
- Cached in: `~/.cache/huggingface/hub/`
- Loaded ONCE at app startup and kept in memory (not per-request)
- Uses `bfloat16` precision for GPU (MPS/CUDA), `float32` for CPU

### File Processing

**Supported formats:**
- Audio: WAV, MP3, FLAC, M4A
- Video: MP4, AVI, MOV (converted to audio via moviepy/ffmpeg)

**File size limits:**
- Max upload: 500MB
- Chunking: 2-minute segments (configurable in transcription_engine.py)
- Temporary files auto-deleted after processing

### API Endpoints

**REST API:**
- `POST /api/upload` - Upload file, returns file_id
- `POST /api/transcribe` - Start transcription job, returns job_id
- `GET /api/status/<job_id>` - Check job status
- `GET /api/transcript/<job_id>` - Retrieve completed transcript
- `DELETE /api/file/<file_id>` - Delete uploaded file

**WebSocket Events:**
- `transcription_progress` - Real-time progress updates
- `transcription_complete` - Final result
- `transcription_error` - Error notifications

### Testing Architecture

**Mock Strategy:**
- Heavy dependencies (torch, transformers, librosa) are mocked in `conftest.py`
- `TESTING=1` environment variable triggers mock mode
- Allows tests to run without GPU or 20GB model download
- Platform-specific tests use markers to skip on incompatible systems

**Fixtures:**
- `app` - Flask test app
- `client` - HTTP test client
- `socketio_client` - WebSocket test client
- `temp_dir` - Temporary directory for test files
- `sample_audio_file` - Mock audio file

## Common Workflows

### Adding a New API Endpoint

1. Define route in `app.py`
2. Add corresponding test in `tests/test_api.py`
3. Update API documentation (if exists)
4. Test with `pytest tests/test_api.py -v`

### Modifying Transcription Logic

1. Update `TranscriptionEngine` in `transcription_engine.py`
2. Add/update tests in `tests/test_transcription_engine.py`
3. Test with `pytest tests/test_transcription_engine.py -v`
4. Run integration tests to ensure end-to-end flow works

### Adding Language Support

1. Verify language code in Voxtral model documentation
2. Add to `SUPPORTED_LANGUAGES` in frontend (static/js/app.js or templates/index.html)
3. Test with a sample file in that language

## Troubleshooting

### Test Environment

If tests fail with import errors:
```bash
export TESTING=1
export PYTHONPATH=/Users/ddbco/Desktop/Voxtral/transcribe-voxtral-main/VoxtralApp:$PYTHONPATH
```

### Model Loading Issues

- First run downloads ~20GB, ensure adequate disk space
- Model loading can take 10-30 seconds on startup
- Check `~/.cache/huggingface/` for cached model

### Memory Issues

- Reduce chunk size in `transcription_engine.py` (default: 2 minutes)
- Limit concurrent transcriptions in `app.py`
- Monitor with Activity Monitor/Task Manager

### Platform-Specific Issues

**MPS (Apple Silicon):**
- Requires macOS with M1/M2/M3 chip
- Auto-detected via `torch.backends.mps.is_available()`

**CUDA (NVIDIA GPU):**
- Requires CUDA toolkit installed
- Auto-detected via `torch.cuda.is_available()`

**CPU Fallback:**
- Used when no GPU available
- Significantly slower but always works

## Dependencies

**Core:**
- Python 3.11+ required
- torch, transformers, librosa, soundfile - ML/audio processing
- Flask, Flask-SocketIO - Web framework and real-time comms
- moviepy (optional) - Video conversion (requires ffmpeg)

**Dev:**
- pytest, pytest-cov, pytest-flask - Testing framework
- black, isort, flake8 - Code quality
- See `requirements.txt` and `requirements-dev.txt`

## Important Notes

- **Virtual Environment**: Use `voxtral_env/` (not `test_venv/`) for development
- **Test Virtual Environment**: `test_venv/` is used for running tests with dev dependencies
- **Git**: This is NOT currently a git repository (working directory shows "Is directory a git repo: No")
- **Privacy**: All processing is local - no data sent to cloud services
- **Model Persistence**: Model stays loaded between requests (don't reload per-request)
- **Job Storage**: Jobs are stored in-memory (lost on restart) - consider persistence for production

## File Structure Reference

```
transcribe-voxtral-main/VoxtralApp/
├── app.py                      # Flask web application
├── transcription_engine.py     # Core transcription logic
├── transcribe_voxtral.py       # CLI script (original)
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
├── pyproject.toml             # Tool configurations (black, isort, etc)
├── pytest.ini                 # Pytest configuration
├── .flake8                    # Flake8 linting rules
├── static/                    # Frontend assets (CSS, JS)
├── templates/                 # HTML templates
├── tests/                     # Test suite
│   ├── conftest.py           # Shared fixtures and mocks
│   ├── test_api.py           # API endpoint tests
│   ├── test_transcription_engine.py
│   ├── test_integration.py
│   └── test_platform_compatibility.py
├── uploads/                   # Temporary upload directory
├── transcriptions_voxtral_final/  # Output directory
└── voxtral_env/              # Virtual environment (production)
```
