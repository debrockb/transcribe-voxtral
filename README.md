# Voxtral Transcription Application

A Flask-based web application for transcribing audio and video files using the Mistral AI Voxtral-Mini-3B model. Features a modern web interface with real-time progress updates, REST API, and WebSocket support, while leveraging local AI processing for privacy-focused, cost-free transcription.

![Transcribe Voxtral hero banner](VoxtralApp/assets/voxtral_banner.svg)

## Features

### Web Application
- 🌐 **Modern Web Interface** - User-friendly drag-and-drop file upload
- ⚡ **Real-Time Progress** - Live updates via WebSocket during transcription
- 📊 **Progress Tracking** - Visual progress bar with chunk-by-chunk updates
- 📋 **Easy Export** - Copy to clipboard or download as text file
- 🎨 **Responsive Design** - Works on desktop and mobile browsers

### Transcription
- 🎵 **Audio & Video Support** - WAV, MP3, FLAC, M4A, MP4, AVI, MOV
- 🎬 **Auto Video Conversion** - Automatically extracts audio from video files
- 🌍 **30+ Languages** - Multilingual support (English, French, Spanish, and more)
- 🔄 **Chunked Processing** - Handles large files efficiently with 2-minute chunks
- 🎯 **High Accuracy** - Powered by Mistral AI Voxtral-Mini-3B model

### Performance & Privacy
- 🚀 **Device Auto-Detection** - MPS (Apple Silicon), CUDA (NVIDIA GPU), or CPU
- 🔒 **Privacy-Focused** - All processing on local hardware, no cloud uploads
- 💰 **No API Costs** - Completely free to use
- ⚙️ **Efficient Memory** - Smart chunking prevents memory overflow

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Web Browser                        │
│          (HTML/CSS/JavaScript + Socket.IO)          │
└──────────────────┬──────────────────────────────────┘
                   │ HTTP/WebSocket
┌──────────────────▼──────────────────────────────────┐
│              Flask Application                       │
│  ┌─────────────────────────────────────────────┐   │
│  │      REST API + WebSocket (app.py)          │   │
│  └──────────┬──────────────────────────────────┘   │
│             │                                        │
│  ┌──────────▼──────────────────────────────────┐   │
│  │   TranscriptionEngine (transcription_       │   │
│  │        engine.py)                            │   │
│  │  • Model loading & device detection          │   │
│  │  • Audio chunking & processing               │   │
│  │  • Progress callbacks                        │   │
│  └──────────┬──────────────────────────────────┘   │
└─────────────┼──────────────────────────────────────┘
              │
┌─────────────▼──────────────────────────────────────┐
│       Mistral AI Voxtral-Mini-3B Model             │
│            (~20GB, cached locally)                  │
└────────────────────────────────────────────────────┘
```

## Requirements

- **Python 3.11 or later**
- **Operating System:** macOS, Linux, or Windows
- **Disk Space:** 20GB+ for initial model download
- **RAM:** 8GB+ (16GB+ recommended for optimal performance)
- **Internet:** Required for initial model download only

### Optional
- **FFmpeg** - For video conversion (MP4, AVI, MOV files)
  - macOS: `brew install ffmpeg`
  - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## Quick Start

### 1. Setup

Navigate to the VoxtralApp directory:

```bash
cd transcribe-voxtral-main/VoxtralApp
```

**macOS/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows:**
```cmd
python -m venv voxtral_env
voxtral_env\Scripts\activate.bat
pip install -r requirements.txt
```

### 2. Start the Web Application

**macOS/Linux:**
```bash
./start_web.sh
```

**Windows:**
```cmd
start_web.bat
```

**Or manually:**
```bash
source voxtral_env/bin/activate  # macOS/Linux
# OR
voxtral_env\Scripts\activate.bat  # Windows

python app.py
```

### 3. Access the Application

Open your browser and navigate to:
```
http://localhost:5000
```

### 4. Transcribe Your First File

1. **Upload** - Drag and drop an audio/video file or click to browse
2. **Select Language** - Choose the language spoken in your file
3. **Start** - Click "Start Transcription"
4. **Monitor** - Watch real-time progress updates
5. **Export** - Copy to clipboard or download as text file

## Project Structure

```
transcribe-voxtral-main/VoxtralApp/
├── app.py                      # Flask web application
├── transcription_engine.py     # Core transcription logic
├── transcribe_voxtral.py       # CLI script for batch processing
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
├── static/
│   ├── css/                   # Stylesheets
│   ├── js/                    # JavaScript frontend
│   └── assets/                # Images and icons
├── templates/
│   └── index.html             # Main web interface
├── tests/                      # Comprehensive test suite
├── docs/
│   ├── API_DOCUMENTATION.md   # API reference
│   └── USER_GUIDE.md          # User manual
├── uploads/                    # Temporary uploads (auto-cleanup)
├── transcriptions_voxtral_final/  # Saved transcripts
└── voxtral_env/               # Python virtual environment
```

## Launcher Icons

Need a branded icon for the macOS `.command` launcher or the Windows `.bat` shortcut? New PNG, `.icns`, and `.ico` assets live under `assets/icons/`. Follow the short walkthrough in `docs/icon-guide.md` to apply them to your preferred launcher.

## Usage Modes

### Web Interface (Recommended)

Perfect for interactive use with real-time feedback:

```bash
cd transcribe-voxtral-main/VoxtralApp
./start_web.sh  # or start_web.bat on Windows
```

Access at `http://localhost:5000`

**Features:**
- Drag-and-drop file upload
- Live progress updates
- Visual feedback
- Copy/download transcripts
- Language selection

### Command Line (Batch Processing)

For automating multiple files or integration with scripts:

```bash
cd transcribe-voxtral-main/VoxtralApp
source voxtral_env/bin/activate
python transcribe_voxtral.py
```

**Features:**
- Batch processing of all audio files in a directory
- Headless operation
- Scriptable and automatable
- Lower memory overhead

## Supported Languages

The Voxtral model supports 30+ languages:

| Code | Language | Code | Language | Code | Language |
|------|----------|------|----------|------|----------|
| `en` | English | `fr` | French | `es` | Spanish |
| `de` | German | `it` | Italian | `pt` | Portuguese |
| `nl` | Dutch | `pl` | Polish | `ru` | Russian |
| `zh` | Chinese | `ja` | Japanese | `ko` | Korean |
| `ar` | Arabic | `hi` | Hindi | `tr` | Turkish |
| `sv` | Swedish | `da` | Danish | `no` | Norwegian |
| `fi` | Finnish | `cs` | Czech | `sk` | Slovak |
| `uk` | Ukrainian | `ro` | Romanian | `el` | Greek |
| `he` | Hebrew | `id` | Indonesian | `vi` | Vietnamese |
| `th` | Thai | `ms` | Malay | `ca` | Catalan |

## API Documentation

The Flask application provides a REST API and WebSocket interface for programmatic access.

### REST Endpoints

- `POST /api/upload` - Upload audio/video file
- `POST /api/transcribe` - Start transcription job
- `GET /api/status/<job_id>` - Get job status
- `GET /api/transcript/<job_id>` - Retrieve transcript
- `GET /api/transcript/<job_id>/download` - Download as file
- `GET /api/languages` - Get supported languages
- `GET /api/device-info` - Get device information

### WebSocket Events

- `transcription_progress` - Real-time progress updates
- `transcription_complete` - Completion notification
- `transcription_error` - Error notifications

For complete API reference, see [VoxtralApp/docs/API_DOCUMENTATION.md](VoxtralApp/docs/API_DOCUMENTATION.md)

## Configuration

### Web Application Settings

Edit `app.py` to configure:

```python
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB (line 26)
UPLOAD_FOLDER = BASE_DIR / "uploads"  # Upload directory (line 23)
OUTPUT_FOLDER = BASE_DIR / "transcriptions_voxtral_final"  # Output (line 24)
```

### Transcription Engine Settings

Edit `transcription_engine.py` to configure:

```python
chunk_duration_s: int = 2 * 60  # Chunk size in seconds (line 150)
sample_rate: int = 16000  # Audio sample rate (line 151)
```

### CLI Script Settings

Edit `transcribe_voxtral.py` to configure batch processing:

```python
INPUT_DIRECTORY = "."  # Where to find audio files
OUTPUT_SUBFOLDER_NAME = "transcriptions_voxtral_final"  # Output folder
```

## Device Detection & Performance

The application automatically detects and uses the best available hardware:

### Processing Speed (Approximate)

| Device | Speed | Example (10 min audio) |
|--------|-------|------------------------|
| Apple M1/M2/M3 (MPS) | ~1-2x realtime | 5-10 min processing |
| NVIDIA GPU (CUDA) | ~1-3x realtime | 3-10 min processing |
| CPU (Fallback) | ~0.1-0.5x realtime | 20-100 min processing |

**Note:** Actual speed varies based on audio quality, language, and specific hardware

### Device Details

**MPS (Apple Silicon)**
- M1, M2, M3, M4 chips
- Uses `bfloat16` precision
- Automatic cache clearing
- Fastest on Apple devices

**CUDA (NVIDIA GPUs)**
- Requires CUDA-compatible GPU
- Uses `bfloat16` precision
- Requires CUDA toolkit

**CPU (Universal)**
- Works on all systems
- Uses `float32` precision
- Slower but reliable

## Testing

The application includes a comprehensive test suite with pytest.

### Run Tests

```bash
cd transcribe-voxtral-main/VoxtralApp

# Activate test environment
source test_venv/bin/activate

# Run all tests (excluding model/GPU tests)
export TESTING=1
pytest tests/ -v -m "not requires_model and not requires_gpu and not slow"

# Run specific test categories
pytest tests/test_api.py -v          # API tests
pytest tests/test_integration.py -v  # Integration tests
pytest tests/ -v -m unit            # Unit tests only
```

### Test Categories

- `unit` - Unit tests for individual components
- `api` - API endpoint tests
- `integration` - Integration tests
- `slow` - Long-running tests
- `requires_model` - Tests needing the ML model (skipped in CI)
- `requires_gpu` - Tests requiring GPU (skipped in CI)
- `cross_platform` - Platform compatibility tests

For more details, see [VoxtralApp/tests/README.md](VoxtralApp/tests/README.md)

## Model Information

**Model:** Mistral AI Voxtral-Mini-3B-2507

- **Type:** Conditional Generation (Audio-to-Text)
- **Size:** ~20GB download
- **Architecture:** Transformer-based encoder-decoder
- **Sample Rate:** 16kHz (automatically resampled)
- **License:** [HuggingFace Model Page](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)

### First Run

The model is downloaded automatically on first use:

- **Download Size:** ~20GB
- **Download Time:** 10-60 minutes (depends on internet speed)
- **Cache Location:** `~/.cache/huggingface/hub/`
- **Redownload:** Not needed - model is cached locally

### Subsequent Runs

- Model loads from cache in 10-30 seconds
- No internet connection required
- Processing starts immediately

## Troubleshooting

### Application Won't Start

**Check Python version:**
```bash
python --version  # Should be 3.11+
```

**Reinstall dependencies:**
```bash
pip install -r requirements.txt --force-reinstall
```

### Can't Access Web Interface

**Solutions:**
1. Verify server is running (check terminal output)
2. Try `http://127.0.0.1:5000` instead
3. Check if port 5000 is in use
4. Check firewall settings

### Model Download Fails

**Solutions:**
1. Ensure 20GB+ free disk space
2. Check internet connection
3. Check firewall/proxy settings
4. Downloads resume automatically - try again

### Out of Memory Errors

**Solutions:**
1. Close other applications
2. Reduce chunk size in `transcription_engine.py`
3. Process shorter files
4. Restart your computer

### Poor Transcription Quality

**Solutions:**
1. **Verify correct language selected**
2. Use high-quality audio (minimal background noise)
3. Ensure adequate audio volume
4. Try with clear speech examples first

### Video Conversion Fails

**Solutions:**
1. Install FFmpeg (see requirements)
2. Install moviepy: `pip install moviepy`
3. Convert video manually using FFmpeg
4. Try different video format

For detailed troubleshooting, see [VoxtralApp/docs/USER_GUIDE.md](VoxtralApp/docs/USER_GUIDE.md)

## Development

### Code Quality

**Format code:**
```bash
cd transcribe-voxtral-main/VoxtralApp
source test_venv/bin/activate

# Auto-format with black
black app.py transcription_engine.py transcribe_voxtral.py tests/*.py

# Sort imports
isort app.py transcription_engine.py transcribe_voxtral.py tests/*.py --skip test_venv
```

**Lint code:**
```bash
flake8 . --config=.flake8
```

### Contributing

1. Run tests before committing
2. Follow code style (black, isort)
3. Add tests for new features
4. Update documentation

## Documentation

- **[User Guide](VoxtralApp/docs/USER_GUIDE.md)** - Complete user manual
- **[API Documentation](VoxtralApp/docs/API_DOCUMENTATION.md)** - API reference
- **[Test Documentation](VoxtralApp/tests/README.md)** - Testing guide
- **[Implementation Plan](VoxtralApp/IMPLEMENTATION_PLAN.md)** - Design document
- **[CLAUDE.md](../CLAUDE.md)** - Claude Code guidance

## Dependencies

### Core
- **torch** - Deep learning framework
- **transformers** - HuggingFace model interface
- **librosa** - Audio processing
- **soundfile** - Audio file I/O
- **Flask** - Web framework
- **Flask-SocketIO** - Real-time WebSocket support
- **mistral-common** - Mistral AI utilities

### Development
- **pytest** - Testing framework
- **black** - Code formatting
- **isort** - Import sorting
- **flake8** - Linting

See [requirements.txt](VoxtralApp/requirements.txt) and [requirements-dev.txt](VoxtralApp/requirements-dev.txt) for complete list.

## Privacy & Security

- ✅ **100% Local Processing** - No cloud uploads
- ✅ **No Data Collection** - No analytics or tracking
- ✅ **Open Source** - Fully auditable code
- ✅ **No Account Required** - Use immediately
- ✅ **Automatic Cleanup** - Temporary files deleted after processing

## Best Practices

1. **Test First** - Start with a short audio clip to verify setup
2. **Correct Language** - Always select the spoken language
3. **Quality Audio** - Use clear recordings for best results
4. **Adequate Storage** - Ensure 20GB+ free for model + files
5. **Monitor Progress** - Watch first transcription to verify quality
6. **Save Transcripts** - Download/copy before closing browser

## License

For educational and research purposes. Check Mistral AI's license terms for the Voxtral model at [HuggingFace Model Page](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)

## Support

For issues or questions:

1. **User Guide** - See [USER_GUIDE.md](VoxtralApp/docs/USER_GUIDE.md) for detailed help
2. **API Docs** - See [API_DOCUMENTATION.md](VoxtralApp/docs/API_DOCUMENTATION.md) for technical details
3. **Test Docs** - See [tests/README.md](VoxtralApp/tests/README.md) for testing help
4. **Troubleshooting** - Check error messages and logs in terminal

## Acknowledgments

Powered by:
- **Mistral AI** - Voxtral-Mini-3B model
- **HuggingFace** - Transformers library
- **Flask** - Web framework
- **LibROSA** - Audio processing
- **Socket.IO** - Real-time communications

---

Thank you for using Voxtral Transcription Application! 🎙️
