# Voxtral Web Application - API Documentation

## Overview

This document describes the REST API and WebSocket events for the Voxtral Web Application. All API endpoints are prefixed with `/api/` unless otherwise noted.

## Base URL

```
http://localhost:5000
```

## Content Type

All POST requests should use `application/json` content type unless specified otherwise (e.g., file uploads use `multipart/form-data`).

---

## REST API Endpoints

### 1. Get Main Application Page

**Endpoint:** `GET /`

**Description:** Serves the main HTML application interface.

**Response:**
- **Content-Type:** `text/html`
- Returns the main application HTML page

**Example:**
```bash
curl http://localhost:5000/
```

---

### 2. Upload Audio/Video File

**Endpoint:** `POST /api/upload`

**Description:** Upload an audio or video file for transcription.

**Request:**
- **Content-Type:** `multipart/form-data`
- **Body:**
  - `file`: The audio/video file (required)

**Supported File Types:**
- Audio: `.wav`, `.mp3`, `.flac`, `.m4a`
- Video: `.mp4`, `.avi`, `.mov`

**Maximum File Size:** 500 MB

**Response (Success - 200):**
```json
{
  "status": "success",
  "file_id": "uuid-string",
  "filename": "audio.mp3",
  "size": 12345678,
  "size_mb": 11.77,
  "is_video": false
}
```

**Response (Error - 400/500):**
```json
{
  "error": "Error message"
}
```

**Example:**
```bash
curl -X POST -F "file=@audio.mp3" http://localhost:5000/api/upload
```

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('/api/upload', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

---

### 3. Start Transcription

**Endpoint:** `POST /api/transcribe`

**Description:** Start transcription job for an uploaded file.

**Request:**
- **Content-Type:** `application/json`
- **Body:**
```json
{
  "file_id": "uuid-string",
  "language": "en"
}
```

**Parameters:**
- `file_id` (string, required): File ID returned from upload endpoint
- `language` (string, optional): Language code (default: "en")

**Response (Success - 200):**
```json
{
  "status": "success",
  "job_id": "uuid-string",
  "message": "Transcription started"
}
```

**Response (Error - 400/500):**
```json
{
  "error": "Error message"
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/api/transcribe \
  -H "Content-Type: application/json" \
  -d '{"file_id": "abc123", "language": "en"}'
```

```javascript
fetch('/api/transcribe', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        file_id: 'abc123',
        language: 'en'
    })
})
.then(response => response.json())
.then(data => console.log(data));
```

---

### 4. Get Job Status

**Endpoint:** `GET /api/status/<job_id>`

**Description:** Get the current status of a transcription job.

**Parameters:**
- `job_id` (string): Job ID returned from transcribe endpoint

**Response (Success - 200):**
```json
{
  "job_id": "uuid-string",
  "file_id": "uuid-string",
  "filename": "audio.mp3",
  "language": "en",
  "status": "processing",
  "progress": 45,
  "current_chunk": 9,
  "total_chunks": 20,
  "started_at": "2024-11-18T10:30:00",
  "message": "Processing chunk 9/20"
}
```

**Status Values:**
- `queued`: Job is queued
- `loading_model`: Model is being loaded
- `loading_audio`: Audio file is being loaded
- `processing`: Transcription in progress
- `finalizing`: Combining transcription chunks
- `complete`: Transcription complete
- `error`: An error occurred

**Response (Error - 404):**
```json
{
  "error": "Job not found"
}
```

**Example:**
```bash
curl http://localhost:5000/api/status/abc123
```

---

### 5. Get Completed Transcript

**Endpoint:** `GET /api/transcript/<job_id>`

**Description:** Retrieve the completed transcript for a job.

**Parameters:**
- `job_id` (string): Job ID

**Response (Success - 200):**
```json
{
  "job_id": "uuid-string",
  "transcript": "This is the transcribed text...",
  "filename": "audio.mp3",
  "language": "en",
  "duration": 5.5,
  "word_count": 247,
  "char_count": 1523
}
```

**Response (Error - 400):**
```json
{
  "error": "Transcription not complete"
}
```

**Response (Error - 404):**
```json
{
  "error": "Job not found"
}
```

**Example:**
```bash
curl http://localhost:5000/api/transcript/abc123
```

---

### 6. Download Transcript File

**Endpoint:** `GET /api/transcript/<job_id>/download`

**Description:** Download the transcript as a text file.

**Parameters:**
- `job_id` (string): Job ID

**Response (Success - 200):**
- **Content-Type:** `text/plain`
- **Content-Disposition:** `attachment; filename="..."`
- Returns the transcript text file

**Response (Error - 400/404):**
```json
{
  "error": "Error message"
}
```

**Example:**
```bash
curl -O http://localhost:5000/api/transcript/abc123/download
```

---

### 7. Get Supported Languages

**Endpoint:** `GET /api/languages`

**Description:** Get list of all supported transcription languages.

**Response (Success - 200):**
```json
[
  { "code": "en", "name": "English" },
  { "code": "fr", "name": "French" },
  { "code": "es", "name": "Spanish" },
  ...
]
```

**Example:**
```bash
curl http://localhost:5000/api/languages
```

---

### 8. Get Device Information

**Endpoint:** `GET /api/device-info`

**Description:** Get information about the processing device (MPS/CUDA/CPU).

**Response (Success - 200):**
```json
{
  "device": "mps",
  "device_name": "MPS",
  "dtype": "torch.bfloat16",
  "mps_available": true,
  "cuda_available": false,
  "cuda_device_name": null
}
```

**Example:**
```bash
curl http://localhost:5000/api/device-info
```

---

## WebSocket Events

### Connection

**Event:** `connect`

**Direction:** Client → Server

**Description:** Establish WebSocket connection.

**Example:**
```javascript
const socket = io('http://localhost:5000');
socket.on('connect', () => {
    console.log('Connected to server');
});
```

---

### Connection Confirmation

**Event:** `connected`

**Direction:** Server → Client

**Description:** Server confirms connection.

**Data:**
```json
{
  "message": "Connected to Voxtral server"
}
```

**Example:**
```javascript
socket.on('connected', (data) => {
    console.log(data.message);
});
```

---

### Transcription Progress Update

**Event:** `transcription_progress`

**Direction:** Server → Client

**Description:** Real-time progress updates during transcription.

**Data:**
```json
{
  "job_id": "uuid-string",
  "status": "processing",
  "message": "Transcribing chunk 5/10",
  "progress": 50,
  "current_chunk": 5,
  "total_chunks": 10
}
```

**Example:**
```javascript
socket.on('transcription_progress', (data) => {
    console.log(`Progress: ${data.progress}%`);
    console.log(`${data.message}`);
});
```

---

### Transcription Complete

**Event:** `transcription_complete`

**Direction:** Server → Client

**Description:** Notification when transcription is complete.

**Data:**
```json
{
  "job_id": "uuid-string",
  "transcript": "Complete transcribed text...",
  "duration": 5.5,
  "word_count": 247
}
```

**Example:**
```javascript
socket.on('transcription_complete', (data) => {
    console.log('Transcription complete!');
    console.log(data.transcript);
});
```

---

### Transcription Error

**Event:** `transcription_error`

**Direction:** Server → Client

**Description:** Notification when an error occurs.

**Data:**
```json
{
  "job_id": "uuid-string",
  "error": "Error message"
}
```

**Example:**
```javascript
socket.on('transcription_error', (data) => {
    console.error('Transcription failed:', data.error);
});
```

---

## Error Codes

| HTTP Code | Description |
|-----------|-------------|
| 200 | Success |
| 400 | Bad Request (invalid parameters, missing fields) |
| 404 | Not Found (job/file not found) |
| 500 | Internal Server Error |
| 503 | Service Unavailable (engine not initialized) |

---

## Common Error Messages

| Error | Description | Solution |
|-------|-------------|----------|
| "No file provided" | File missing in upload request | Include file in form data |
| "File type not supported" | Unsupported file format | Use WAV, MP3, FLAC, M4A, MP4, AVI, or MOV |
| "File too large" | File exceeds 500MB limit | Compress or split the file |
| "Invalid file ID" | File ID not found or invalid | Upload file first and use returned file_id |
| "Job not found" | Job ID doesn't exist | Check job_id is correct |
| "Transcription not complete" | Attempted to get transcript before completion | Wait for completion or check status |
| "Engine not initialized" | Model not loaded | Wait for server startup to complete |

---

## Usage Example: Complete Workflow

```javascript
// 1. Connect to WebSocket
const socket = io('http://localhost:5000');

// 2. Listen for events
socket.on('transcription_progress', (data) => {
    console.log(`Progress: ${data.progress}%`);
});

socket.on('transcription_complete', (data) => {
    console.log('Transcript:', data.transcript);
});

socket.on('transcription_error', (data) => {
    console.error('Error:', data.error);
});

// 3. Upload file
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const uploadResponse = await fetch('/api/upload', {
    method: 'POST',
    body: formData
});
const uploadData = await uploadResponse.json();

// 4. Start transcription
const transcribeResponse = await fetch('/api/transcribe', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        file_id: uploadData.file_id,
        language: 'en'
    })
});
const transcribeData = await transcribeResponse.json();

// 5. Progress updates arrive via WebSocket automatically

// 6. Download transcript when complete
window.location.href = `/api/transcript/${transcribeData.job_id}/download`;
```

---

## Rate Limiting

Currently, there are no rate limits enforced. However, only one transcription can run at a time per file due to model memory constraints.

---

## CORS

CORS is enabled for all origins during development. For production deployment, configure appropriate CORS settings in `app.py`.

---

## Notes

- Video files (MP4, AVI, MOV) are automatically converted to audio before transcription
- The model is loaded once at server startup and reused for all transcriptions
- Temporary files are automatically cleaned up after processing
- WebSocket connection is required for real-time progress updates
- All timestamps use ISO 8601 format

---

## Support

For issues or questions, please refer to the main [README.md](../README.md) or [USER_GUIDE.md](USER_GUIDE.md).
