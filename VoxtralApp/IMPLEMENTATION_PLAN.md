# Voxtral Web Application - Implementation Plan

## Executive Summary

This document outlines the implementation plan for building a web-based front-end application for the Voxtral transcription system. The application will provide an intuitive user interface for uploading audio/video files, selecting transcription language, viewing transcripts, and copying results.

## Current Architecture Analysis

### Existing Python Script (`transcribe_voxtral.py`)

**Core Components:**
- **Model Loading** (lines 94-106): Loads Mistral AI Voxtral-Mini-3B model
- **Device Detection** (lines 83-92): Auto-detects MPS/CUDA/CPU
- **Chunk Processing** (lines 13-29): Transcribes 2-minute audio chunks
- **Audio Processing** (lines 31-76): Handles large files via chunking
- **Batch Processing** (lines 120-131): Processes multiple files

**Key Features:**
- Supports: WAV, MP3, FLAC, M4A formats
- Language configuration: Line 16 (currently hardcoded to "fr")
- 2-minute chunking for memory management
- UTF-8 encoded text output

## Proposed Solution: Flask Web Application

### Technology Stack

**Backend:**
- **Flask**: Lightweight Python web framework
- **Flask-SocketIO**: Real-time progress updates
- **Flask-CORS**: Cross-origin resource sharing
- **Existing libraries**: torch, librosa, transformers (already in requirements.txt)

**Frontend:**
- **HTML5**: Structure and file upload
- **CSS3**: Modern, responsive styling
- **JavaScript (Vanilla)**: Dynamic interactions
- **Socket.IO Client**: Real-time updates

**Why This Stack?**
- Pure Python backend integrates seamlessly with existing script
- No complex build tools required
- Easy to deploy and maintain
- Real-time progress updates enhance UX

### Cross-Platform Compatibility (Windows & macOS)

**Critical Design Decisions:**

- **Path Handling**: Use `pathlib.Path` and `os.path.join()` for all file paths (never hardcode `/` or `\`)
- **Shell Scripts**: Provide both `.sh` (macOS/Linux) and `.bat` (Windows) startup scripts
- **Virtual Environment**: Support both activation methods
  - macOS/Linux: `source voxtral_env/bin/activate`
  - Windows: `voxtral_env\Scripts\activate.bat`
- **FFmpeg**: Include platform-specific installation instructions
  - macOS: `brew install ffmpeg`
  - Windows: Download from ffmpeg.org or use `choco install ffmpeg`
- **Line Endings**: Use `\n` in Python, let Git handle CRLF conversion
- **Case Sensitivity**: Assume case-insensitive file systems (Windows default)

**Testing Strategy:**

- Test on both macOS and Windows environments
- Verify file path operations on both systems
- Ensure startup scripts work correctly
- Validate device detection (MPS on Mac, CUDA on Windows with NVIDIA GPU)

## Application Architecture

### Directory Structure

```
transcribe-voxtral-main/
├── app.py                          # Flask application entry point
├── transcribe_voxtral.py           # Existing script (modified for API)
├── requirements.txt                # Updated dependencies
├── static/
│   ├── css/
│   │   └── style.css              # Application styles
│   ├── js/
│   │   └── app.js                 # Frontend logic
│   └── assets/
│       └── logo.svg               # Branding assets
├── templates/
│   └── index.html                 # Main application page
├── uploads/                        # Temporary upload directory
├── transcriptions_voxtral_final/  # Output directory (existing)
└── docs/
    ├── IMPLEMENTATION_PLAN.md     # This document
    ├── API_DOCUMENTATION.md       # API endpoint docs
    └── USER_GUIDE.md              # End-user documentation
```

### System Flow

```
User → Upload File → Flask Backend → File Validation
                                    ↓
                        Convert to Audio (if video)
                                    ↓
                        Process with Voxtral Model
                                    ↓
                        Stream Progress to Frontend
                                    ↓
                        Return Transcript → Display & Copy
```

## Feature Specifications

### 1. File Upload Component

**Requirements:**
- Drag-and-drop interface
- File type validation (MP3, M4A, MP4, WAV, FLAC)
- File size display
- Multiple file queue support
- Visual upload progress

**Implementation:**
- HTML5 File API
- Client-side validation
- FormData for file transmission
- Progress event listeners

### 2. Language Selector

**Requirements:**
- Dropdown with all 30+ supported languages
- Search/filter capability
- Display language code and full name
- Persist selection across sessions (localStorage)

**Supported Languages:**
```
English (en), French (fr), Spanish (es), German (de), Italian (it),
Portuguese (pt), Dutch (nl), Polish (pl), Russian (ru), Chinese (zh),
Japanese (ja), Korean (ko), Arabic (ar), Hindi (hi), Turkish (tr),
Swedish (sv), Danish (da), Norwegian (no), Finnish (fi), Czech (cs),
Slovak (sk), Ukrainian (uk), Romanian (ro), Greek (el), Hebrew (he),
Indonesian (id), Vietnamese (vi), Thai (th), Malay (ms), Catalan (ca)
```

### 3. Transcript Display

**Requirements:**
- Full-width text display area
- Responsive design (mobile-friendly)
- Syntax highlighting for readability
- Character/word count statistics
- Auto-scroll during transcription

**Features:**
- Copy to clipboard button
- Download as TXT file
- Clear transcript option
- Timestamp display

### 4. Progress Tracking

**Requirements:**
- Real-time chunk processing updates
- Percentage completion
- Estimated time remaining
- Current chunk display (e.g., "Chunk 3/15")

**Implementation:**
- WebSocket (Socket.IO) for real-time updates
- Progress bar visualization
- Status messages (Loading model, Processing chunk X, Complete)

### 5. Status & Error Handling

**Requirements:**
- Clear error messages
- Device detection display (MPS/CUDA/CPU)
- Model loading status
- Network error handling

## API Design

### REST Endpoints

#### `GET /`
- **Description**: Serves main application page
- **Response**: HTML template

#### `POST /api/upload`
- **Description**: Upload audio/video file
- **Request**: Multipart form data with file
- **Response**: `{file_id, filename, size, status}`

#### `POST /api/transcribe`
- **Description**: Start transcription process
- **Request**: `{file_id, language}`
- **Response**: `{job_id, status}`

#### `GET /api/status/<job_id>`
- **Description**: Get transcription status
- **Response**: `{status, progress, current_chunk, total_chunks}`

#### `GET /api/transcript/<job_id>`
- **Description**: Retrieve completed transcript
- **Response**: `{transcript, filename, duration, language}`

#### `DELETE /api/file/<file_id>`
- **Description**: Delete uploaded file
- **Response**: `{status, message}`

### WebSocket Events

#### `connect`
- Client connects to server

#### `transcription_progress`
- **Emitted by**: Server
- **Data**: `{job_id, chunk, total_chunks, percentage, message}`

#### `transcription_complete`
- **Emitted by**: Server
- **Data**: `{job_id, transcript, duration}`

#### `transcription_error`
- **Emitted by**: Server
- **Data**: `{job_id, error_message}`

## Implementation Steps

### Phase 1: Backend Refactoring (3-4 hours)

1. **Modify `transcribe_voxtral.py`**
   - Extract transcription logic into reusable functions
   - Add callback support for progress updates
   - Create `TranscriptionEngine` class
   - Add video-to-audio conversion (using moviepy/ffmpeg)

2. **Create `app.py` Flask Application**
   - Initialize Flask and Flask-SocketIO
   - Configure upload directory and file size limits
   - Implement CORS for local development
   - Add session management

3. **Implement API Endpoints**
   - File upload handler with validation
   - Transcription job queue
   - Status checking endpoint
   - Transcript retrieval

4. **Add Video Conversion**
   - Install moviepy or use ffmpeg
   - Extract audio track from MP4 files
   - Convert to WAV/MP3 format

### Phase 2: Frontend Development (4-5 hours)

1. **Create `templates/index.html`**
   - Semantic HTML5 structure
   - File upload area with drag-drop
   - Language selector dropdown
   - Transcript display area
   - Progress indicators

2. **Build `static/css/style.css`**
   - Modern, clean design
   - Responsive layout (mobile-first)
   - Dark/light mode support
   - Smooth animations

3. **Develop `static/js/app.js`**
   - File upload logic
   - Socket.IO connection
   - Real-time progress updates
   - Clipboard copy functionality
   - Error handling

### Phase 3: Integration & Testing (2-3 hours)

1. **Integration Testing**
   - Upload various file formats
   - Test all language options
   - Verify progress updates
   - Test error scenarios

2. **Performance Testing**
   - Large file handling (>100MB)
   - Multiple concurrent uploads
   - Memory usage monitoring

3. **Cross-browser Testing**
   - Chrome, Firefox, Safari
   - Mobile browsers (iOS/Android)

### Phase 4: Documentation (1-2 hours)

1. **API Documentation** (`docs/API_DOCUMENTATION.md`)
   - Endpoint specifications
   - Request/response examples
   - Error codes and messages

2. **User Guide** (`docs/USER_GUIDE.md`)
   - Getting started
   - Feature walkthrough
   - Troubleshooting

3. **Code Documentation**
   - Inline comments
   - Function docstrings
   - Architecture diagrams

### Phase 5: Deployment Preparation (1 hour)

1. **Update `requirements.txt`**
   - Add Flask, Flask-SocketIO, Flask-CORS
   - Add moviepy or ffmpeg-python
   - Version pinning

2. **Create `start_web.sh`**
   - Activate virtual environment
   - Set environment variables
   - Start Flask server

3. **Add `.gitignore` Entries**
   - uploads/ directory
   - *.pyc files
   - Session files

## Modified Files

### `transcribe_voxtral.py` Changes

**Modifications:**
- Refactor into `TranscriptionEngine` class
- Add `progress_callback` parameter
- Extract device detection to separate function
- Add error handling and logging

**New Functions:**
```python
class TranscriptionEngine:
    def __init__(self, model_id, device=None, progress_callback=None)
    def transcribe_file(self, audio_path, language, output_path)
    def _transcribe_chunk(self, chunk_path)
    def _emit_progress(self, current, total, message)
```

### `requirements.txt` Additions

```
Flask==3.0.0
Flask-SocketIO==5.3.5
Flask-CORS==4.0.0
python-socketio==5.10.0
moviepy==1.0.3
python-dotenv==1.0.0
```

## User Interface Mockup

### Main Application Layout

```
┌─────────────────────────────────────────────────────────┐
│  🎙️ Voxtral Transcription                    [Settings]│
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────────────────────────────────────┐   │
│  │  📁 Drag & Drop Audio/Video File              │   │
│  │     or click to browse                         │   │
│  │                                                 │   │
│  │     Supported: MP3, M4A, MP4, WAV, FLAC       │   │
│  └────────────────────────────────────────────────┘   │
│                                                          │
│  Language: [English ▼]              [Start Transcribe] │
│                                                          │
│  ┌────────────────────────────────────────────────┐   │
│  │ Status: Ready                                   │   │
│  │ ████████░░░░░░░░░░░░░░░░░ 35%                │   │
│  │ Processing chunk 7/20...                       │   │
│  └────────────────────────────────────────────────┘   │
│                                                          │
│  ┌────────────────────────────────────────────────┐   │
│  │ Transcript                    [Copy] [Download]│   │
│  ├────────────────────────────────────────────────┤   │
│  │                                                 │   │
│  │ This is the transcribed text from your audio   │   │
│  │ file. It will appear here as the transcription │   │
│  │ progresses in real-time...                     │   │
│  │                                                 │   │
│  │                                                 │   │
│  │                                                 │   │
│  └────────────────────────────────────────────────┘   │
│                                                          │
│  Device: MPS (Apple Silicon) | Words: 1,247 | Chars: 6,891│
└─────────────────────────────────────────────────────────┘
```

## Security Considerations

1. **File Upload Security**
   - File type validation (both client and server)
   - File size limits (100MB default)
   - Virus scanning (optional, using clamav)
   - Secure file naming (UUID-based)

2. **Input Sanitization**
   - Validate language codes against whitelist
   - Sanitize file names
   - Prevent directory traversal

3. **Resource Management**
   - Automatic cleanup of old uploads (24h)
   - Temporary file deletion after processing
   - Memory limit enforcement

4. **Data Privacy**
   - No cloud uploads (local processing only)
   - Automatic transcript deletion option
   - No logging of file contents

## Performance Optimization

1. **Caching**
   - Model loaded once at startup (not per request)
   - Keep model in memory between requests
   - Cache language configurations

2. **Async Processing**
   - Background job queue for transcriptions
   - Non-blocking file uploads
   - Streaming responses

3. **Resource Limits**
   - Max concurrent transcriptions (default: 2)
   - Upload queue management
   - Graceful degradation on resource exhaustion

## Error Handling

### Client-Side Errors
- Invalid file type
- File too large
- Network disconnection
- Browser compatibility

### Server-Side Errors
- Model loading failure
- Out of memory
- Disk space exhaustion
- Audio processing errors

**Error Response Format:**
```json
{
  "status": "error",
  "code": "AUDIO_PROCESSING_ERROR",
  "message": "Failed to process audio file",
  "details": "Unsupported audio codec"
}
```

## Future Enhancements

1. **Advanced Features**
   - Speaker diarization (identify multiple speakers)
   - Timestamp markers every N seconds
   - Export to SRT/VTT subtitle format
   - Batch upload (multiple files at once)

2. **User Experience**
   - Transcription history
   - User accounts and saved settings
   - Dark mode toggle
   - Keyboard shortcuts

3. **Integration**
   - REST API for external applications
   - Browser extension
   - Desktop application (Electron wrapper)
   - Mobile app (React Native)

## Testing Strategy

### Unit Tests
- Transcription engine functions
- File validation logic
- Audio conversion utilities
- API endpoint handlers

### Integration Tests
- End-to-end transcription flow
- WebSocket communication
- File upload and download
- Error scenarios

### Manual Testing Checklist
- [ ] Upload MP3 file and transcribe
- [ ] Upload M4A file and transcribe
- [ ] Upload MP4 video and extract audio
- [ ] Test all 30 language options
- [ ] Copy transcript to clipboard
- [ ] Download transcript as TXT
- [ ] Test with large file (>50MB)
- [ ] Test with very short file (<10s)
- [ ] Test error handling (invalid file)
- [ ] Test progress updates
- [ ] Test on mobile device
- [ ] Test concurrent uploads

## Deployment Instructions

### Local Development

```bash
# Activate virtual environment
source voxtral_env/bin/activate

# Install new dependencies
pip install -r requirements.txt

# Set environment variables
export FLASK_ENV=development
export FLASK_APP=app.py

# Run the application
python app.py
```

Access at: `http://localhost:5000`

### Production Deployment

```bash
# Use Gunicorn with eventlet workers
gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:5000 app:app
```

**Note:** Use only 1 worker to avoid model loading multiple times (20GB per instance)

## Maintenance Plan

1. **Regular Updates**
   - Update dependencies monthly
   - Monitor Mistral AI model updates
   - Security patches

2. **Monitoring**
   - Log transcription errors
   - Track processing times
   - Monitor disk usage

3. **Backup**
   - Configuration files
   - User-generated transcripts (if saved)

## Success Metrics

- [ ] Web application successfully loads
- [ ] File upload works for all supported formats
- [ ] Language selector includes all 30+ languages
- [ ] Real-time progress updates display correctly
- [ ] Transcript displays in full with copy functionality
- [ ] Video files convert to audio automatically
- [ ] Error messages are clear and actionable
- [ ] Application works on mobile devices
- [ ] Documentation is complete and accurate
- [ ] All changes committed locally (not pushed)

## Timeline Estimate

| Phase | Description | Estimated Time |
|-------|-------------|----------------|
| 1 | Backend Refactoring | 3-4 hours |
| 2 | Frontend Development | 4-5 hours |
| 3 | Integration & Testing | 2-3 hours |
| 4 | Documentation | 1-2 hours |
| 5 | Deployment Prep | 1 hour |
| **Total** | **Complete Implementation** | **11-15 hours** |

## Conclusion

This implementation plan outlines a comprehensive web-based front-end for the Voxtral transcription system. The Flask-based architecture integrates seamlessly with the existing Python script while providing a modern, user-friendly interface. All components will be thoroughly documented and tested before local commit.

The modular design allows for future enhancements while maintaining the core privacy-focused, local-processing philosophy of the original application.
