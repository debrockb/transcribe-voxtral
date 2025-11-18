"""
Voxtral Web Application
Flask-based web interface for audio transcription using Voxtral AI
Cross-platform compatible (Windows & macOS)
"""

import gc
import logging
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

import psutil
import requests
from flask import Flask, jsonify, render_template, request, send_file, session
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename

from config_manager import config
from transcription_engine import TranscriptionEngine
from update_checker import check_for_updates, get_current_version

# Configuration
BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
OUTPUT_FOLDER = BASE_DIR / "transcriptions_voxtral_final"
ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "m4a", "mp4", "avi", "mov"}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

# Ensure directories exist (cross-platform)
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
# Generate a random secret key on startup for session security
app.config["SECRET_KEY"] = secrets.token_hex(32)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"  # CSRF protection
app.config["SESSION_COOKIE_HTTPONLY"] = True  # XSS protection

# Enable CORS and SocketIO - restrict to localhost only to prevent CSRF attacks
# from malicious websites the user might visit while the app is running
ALLOWED_ORIGINS = [
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
CORS(app, origins=ALLOWED_ORIGINS)
# Let Flask-SocketIO auto-select async_mode to avoid dependency issues
socketio = SocketIO(app, cors_allowed_origins=ALLOWED_ORIGINS)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global storage for jobs and files
jobs = {}  # {job_id: {status, progress, transcript, etc}}
uploaded_files = {}  # {file_id: {filename, path, size, etc}}

# Transcription engine (loaded after user selects model)
transcription_engine = None
engine_loading = False  # Flag to track if model is currently loading


def validate_csrf_protection():
    """
    Validate that state-changing requests come from the actual application.

    Requires custom header that HTML forms cannot set. This prevents CSRF attacks
    via form submissions from malicious websites, since browsers block custom headers
    in forms, and CORS blocks fetch/XHR from cross-origin sites.
    """
    # Require custom header that forms cannot set
    custom_header = request.headers.get("X-Voxtral-Request")
    if custom_header != "voxtral-web-ui":
        logger.warning(
            f"CSRF validation failed: Missing or invalid X-Voxtral-Request header from {request.remote_addr}"
        )
        return False

    return True


def allowed_file(filename):
    """Check if file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_file_extension(filename):
    """Get file extension in lowercase."""
    return filename.rsplit(".", 1)[1].lower() if "." in filename else ""


def is_video_file(filename):
    """Check if file is a video format."""
    video_extensions = {"mp4", "avi", "mov", "mkv"}
    return get_file_extension(filename) in video_extensions


def convert_video_to_audio(video_path, output_audio_path):
    """
    Convert video file to audio (WAV format).
    Uses moviepy for cross-platform compatibility.
    """
    try:
        from moviepy.editor import VideoFileClip

        logger.info(f"Converting video to audio: {video_path}")
        video = VideoFileClip(str(video_path))
        video.audio.write_audiofile(str(output_audio_path), codec="pcm_s16le")
        video.close()
        logger.info(f"Video converted successfully: {output_audio_path}")
        return True
    except Exception as e:
        logger.error(f"Video conversion error: {e}")
        return False


def progress_callback(job_id, data):
    """Callback function for transcription progress updates."""
    if job_id in jobs:
        jobs[job_id].update(data)

        # Emit progress via WebSocket
        socketio.emit("transcription_progress", {"job_id": job_id, **data})


def transcribe_in_background(job_id, file_path, language, output_path):
    """Background task for transcription."""
    try:
        # Create progress callback with job_id
        def callback(data):
            progress_callback(job_id, data)

        # Update engine callback
        transcription_engine.progress_callback = callback

        # Start transcription
        result = transcription_engine.transcribe_file(
            input_audio_path=str(file_path), output_text_path=str(output_path), language=language
        )

        if result["status"] == "success":
            jobs[job_id].update(
                {
                    "status": "complete",
                    "transcript": result["transcript"],
                    "duration": result["duration_minutes"],
                    "word_count": result["word_count"],
                    "char_count": result["char_count"],
                    "completed_at": datetime.now().isoformat(),
                }
            )

            socketio.emit(
                "transcription_complete",
                {
                    "job_id": job_id,
                    "transcript": result["transcript"],
                    "duration": result["duration_minutes"],
                    "word_count": result["word_count"],
                },
            )
        else:
            jobs[job_id].update({"status": "error", "error": result.get("error", "Unknown error")})

            socketio.emit("transcription_error", {"job_id": job_id, "error": result.get("error", "Unknown error")})

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Transcription error for job {job_id}: {error_msg}")

        jobs[job_id].update({"status": "error", "error": error_msg})

        socketio.emit("transcription_error", {"job_id": job_id, "error": error_msg})


# Helper functions for consistent API responses
def error_response(message, status_code=400):
    """Return a standardized error response."""
    return jsonify({"status": "error", "message": message}), status_code


def success_response(data=None, message=None):
    """Return a standardized success response."""
    response = {"status": "success"}
    if message:
        response["message"] = message
    if data:
        response.update(data)
    return jsonify(response)


def validate_safe_path(base_dir, filename):
    """
    Validate that the requested filename stays within the base directory.

    Args:
        base_dir: Base directory (Path object)
        filename: User-provided filename

    Returns:
        Path object if safe, None if path traversal detected
    """
    try:
        # Resolve the full path
        full_path = (base_dir / filename).resolve()

        # Check if it's a file and within the base directory
        if not full_path.is_file():
            return None

        # Ensure the resolved path is within the base directory
        if base_dir.resolve() not in full_path.parents and full_path != base_dir.resolve():
            logger.warning(f"Path traversal attempt detected: {filename}")
            return None

        return full_path
    except Exception as e:
        logger.error(f"Error validating path for {filename}: {e}")
        return None


# Routes


@app.route("/")
def index():
    """Serve the main application page."""
    device_info = transcription_engine.get_device_info() if transcription_engine else {}
    return render_template("index.html", device_info=device_info)


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({"status": "ok", "message": "Voxtral transcription service is running"})


@app.route("/api/models", methods=["GET"])
def get_available_models():
    """Get list of available models."""
    try:
        models = config.get_all_models()
        current_version = config.get("model.version", "full")

        # Format response with additional UI-friendly information
        models_list = []
        for key, model_info in models.items():
            models_list.append(
                {
                    "id": key,
                    "name": model_info.get("name"),
                    "size_gb": model_info.get("size_gb"),
                    "format": model_info.get("format"),
                    "description": model_info.get("description"),
                    "memory_requirements": model_info.get("memory_requirements"),
                    "is_current": key == current_version,
                    "is_loaded": transcription_engine is not None and key == current_version,
                }
            )

        return jsonify({"models": models_list, "current_version": current_version})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/model/status", methods=["GET"])
def get_model_status():
    """Get current model loading status."""
    if transcription_engine:
        model_config = config.get_model_config()
        return jsonify(
            {
                "loaded": True,
                "loading": False,
                "model": model_config.get("name"),
                "model_id": model_config.get("id"),
                "version": config.get("model.version"),
                "device": transcription_engine.device if hasattr(transcription_engine, "device") else "unknown",
            }
        )
    elif engine_loading:
        return jsonify({"loaded": False, "loading": True, "message": "Model is currently loading..."})
    else:
        return jsonify({"loaded": False, "loading": False, "message": "No model loaded. Please select a model to begin."})


@app.route("/api/model/initialize", methods=["POST"])
def initialize_model():
    """Initialize model with selected version."""
    if transcription_engine:
        return (
            jsonify({"status": "error", "message": "Model already loaded. Please reload the application to change models."}),
            400,
        )

    if engine_loading:
        return jsonify({"status": "error", "message": "Model is currently loading. Please wait."}), 409

    # Validate JSON request body
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"status": "error", "message": "Invalid or missing JSON request body"}), 400

    model_version = data.get("version", "full")

    # Validate model version
    available_models = config.get_all_models()
    if model_version not in available_models:
        return jsonify({"status": "error", "message": f"Invalid model version: {model_version}"}), 400

    try:
        # Initialize engine in background thread to not block the API
        def load_model():
            try:
                initialize_engine(model_version)
            except Exception as e:
                logger.error(f"Failed to load model in background: {e}")

        thread = threading.Thread(target=load_model, daemon=False)
        thread.start()

        model_config = config.get_model_config(model_version)
        return jsonify(
            {
                "status": "loading",
                "message": f"Loading {model_config.get('name')}...",
                "model": model_config.get("name"),
                "version": model_version,
            }
        )

    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to start model loading: {str(e)}"}), 500


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Handle file upload."""
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"status": "error", "message": "No file selected"}), 400

    if not allowed_file(file.filename):
        return (
            jsonify({"status": "error", "message": f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}),
            400,
        )

    try:
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        original_filename = secure_filename(file.filename)

        # Save file with unique name (cross-platform path)
        file_extension = get_file_extension(original_filename)
        unique_filename = f"{file_id}.{file_extension}"
        file_path = UPLOAD_FOLDER / unique_filename

        file.save(str(file_path))
        file_size = file_path.stat().st_size

        # Store file metadata
        uploaded_files[file_id] = {
            "file_id": file_id,
            "original_filename": original_filename,
            "filename": unique_filename,
            "path": str(file_path),
            "size": file_size,
            "size_mb": round(file_size / (1024 * 1024), 2),
            "is_video": is_video_file(original_filename),
            "uploaded_at": datetime.now().isoformat(),
        }

        logger.info(f"File uploaded: {original_filename} ({file_size} bytes)")

        return jsonify(
            {
                "status": "success",
                "file_id": file_id,
                "filename": unique_filename,  # Return actual saved filename (UUID)
                "original_filename": original_filename,  # Also include original for reference
                "size": file_size,
                "size_mb": uploaded_files[file_id]["size_mb"],
                "is_video": uploaded_files[file_id]["is_video"],
            }
        )

    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/transcribe", methods=["POST"])
def start_transcription():  # noqa: C901
    """Start transcription job."""
    # Check if transcription engine is initialized
    if transcription_engine is None:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Transcription engine not initialized. Please select a model first.",
                }
            ),
            503,
        )

    # Validate JSON request body
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"status": "error", "message": "Invalid or missing JSON request body"}), 400

    file_id = data.get("file_id")
    filename = data.get("filename")
    language = data.get("language", "en")

    # Support both file_id and filename
    if filename and not file_id:
        # Look up file_id by filename
        for fid, finfo in uploaded_files.items():
            if finfo.get("filename") == filename or finfo.get("original_filename") == filename:
                file_id = fid
                break

        # If filename was provided but not found, return 404
        if not file_id:
            return jsonify({"status": "error", "message": "File not found"}), 404

    # If neither filename nor file_id provided
    if not file_id:
        return jsonify({"status": "error", "message": "Missing file ID or filename"}), 400

    # If file_id provided but doesn't exist
    if file_id not in uploaded_files:
        return jsonify({"status": "error", "message": "File not found"}), 404

    try:
        file_info = uploaded_files[file_id]
        file_path = Path(file_info["path"])

        # If video, convert to audio first
        if file_info["is_video"]:
            audio_path = UPLOAD_FOLDER / f"{file_id}_audio.wav"

            if not convert_video_to_audio(file_path, audio_path):
                return jsonify({"status": "error", "message": "Failed to convert video to audio"}), 500

            # Update file path to converted audio
            file_path = audio_path

        # Generate job ID and output path
        job_id = str(uuid.uuid4())
        # Include job_id in filename to prevent overwriting previous transcriptions
        output_filename = f"{Path(file_info['original_filename']).stem}_{job_id[:8]}_transcription.txt"
        output_path = OUTPUT_FOLDER / output_filename

        # Create job entry
        jobs[job_id] = {
            "job_id": job_id,
            "file_id": file_id,
            "filename": file_info["original_filename"],
            "language": language,
            "status": "queued",
            "progress": 0,
            "current_chunk": 0,
            "total_chunks": 0,
            "started_at": datetime.now().isoformat(),
            "output_path": str(output_path),
        }

        # Start transcription in background thread
        thread = threading.Thread(target=transcribe_in_background, args=(job_id, file_path, language, output_path))
        thread.daemon = True
        thread.start()

        logger.info(f"Transcription job started: {job_id} for file {file_info['original_filename']}")

        return jsonify({"status": "success", "job_id": job_id, "message": "Transcription started"})

    except Exception as e:
        logger.error(f"Transcription start error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/status/<job_id>", methods=["GET"])
def get_status(job_id):
    """Get transcription job status."""
    if job_id not in jobs:
        return jsonify({"status": "error", "message": "Job not found"}), 404

    return jsonify(jobs[job_id])


@app.route("/api/transcript/<job_id>", methods=["GET"])
def get_transcript(job_id):
    """Get completed transcript."""
    if job_id not in jobs:
        return jsonify({"status": "error", "message": "Job not found"}), 404

    job = jobs[job_id]

    if job["status"] != "complete":
        return jsonify({"status": "error", "message": "Transcription not complete"}), 400

    return jsonify(
        {
            "job_id": job_id,
            "transcript": job.get("transcript", ""),
            "filename": job.get("filename", ""),
            "language": job.get("language", ""),
            "duration": job.get("duration", 0),
            "word_count": job.get("word_count", 0),
            "char_count": job.get("char_count", 0),
        }
    )


@app.route("/api/transcript/<job_id>/download", methods=["GET"])
def download_transcript(job_id):
    """Download transcript as text file."""
    if job_id not in jobs:
        return jsonify({"status": "error", "message": "Job not found"}), 404

    job = jobs[job_id]

    if job["status"] != "complete":
        return jsonify({"status": "error", "message": "Transcription not complete"}), 400

    output_path = job.get("output_path")

    if not output_path or not Path(output_path).exists():
        return jsonify({"status": "error", "message": "Transcript file not found"}), 404

    return send_file(output_path, as_attachment=True)


@app.route("/api/languages", methods=["GET"])
def get_languages():
    """Get list of supported languages."""
    languages = [
        {"code": "en", "name": "English"},
        {"code": "fr", "name": "French"},
        {"code": "es", "name": "Spanish"},
        {"code": "de", "name": "German"},
        {"code": "it", "name": "Italian"},
        {"code": "pt", "name": "Portuguese"},
        {"code": "nl", "name": "Dutch"},
        {"code": "pl", "name": "Polish"},
        {"code": "ru", "name": "Russian"},
        {"code": "zh", "name": "Chinese"},
        {"code": "ja", "name": "Japanese"},
        {"code": "ko", "name": "Korean"},
        {"code": "ar", "name": "Arabic"},
        {"code": "hi", "name": "Hindi"},
        {"code": "tr", "name": "Turkish"},
        {"code": "sv", "name": "Swedish"},
        {"code": "da", "name": "Danish"},
        {"code": "no", "name": "Norwegian"},
        {"code": "fi", "name": "Finnish"},
        {"code": "cs", "name": "Czech"},
        {"code": "sk", "name": "Slovak"},
        {"code": "uk", "name": "Ukrainian"},
        {"code": "ro", "name": "Romanian"},
        {"code": "el", "name": "Greek"},
        {"code": "he", "name": "Hebrew"},
        {"code": "id", "name": "Indonesian"},
        {"code": "vi", "name": "Vietnamese"},
        {"code": "th", "name": "Thai"},
        {"code": "ms", "name": "Malay"},
        {"code": "ca", "name": "Catalan"},
    ]
    return jsonify(languages)


@app.route("/api/device-info", methods=["GET"])
def get_device_info():
    """Get device information."""
    if transcription_engine:
        return jsonify(transcription_engine.get_device_info())
    return jsonify({"status": "error", "message": "Engine not initialized"}), 503


@app.route("/api/history/transcriptions", methods=["GET"])
def list_transcriptions():
    """List all saved transcriptions."""
    try:
        transcriptions = []
        if OUTPUT_FOLDER.exists():
            for file_path in OUTPUT_FOLDER.glob("*.txt"):
                stat = file_path.stat()
                transcriptions.append(
                    {
                        "filename": file_path.name,
                        "size": stat.st_size,
                        "size_kb": round(stat.st_size / 1024, 2),
                        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    }
                )

        # Sort by modified date (newest first)
        transcriptions.sort(key=lambda x: x["modified"], reverse=True)
        return jsonify(transcriptions)

    except Exception as e:
        logger.error(f"Error listing transcriptions: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/history/transcriptions/<filename>", methods=["GET"])
def get_transcription_content(filename):
    """Get transcription content as JSON for viewing."""
    try:
        # Validate path to prevent directory traversal
        file_path = validate_safe_path(OUTPUT_FOLDER, filename)

        if not file_path:
            return jsonify({"status": "error", "message": "File not found"}), 404

        # Read file content
        content = file_path.read_text(encoding="utf-8")

        # Calculate stats
        word_count = len(content.split())
        char_count = len(content)

        return jsonify({"status": "success", "content": content, "word_count": word_count, "char_count": char_count})

    except Exception as e:
        logger.error(f"Error reading transcription {filename}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/history/transcriptions/<filename>/download", methods=["GET"])
def download_transcription_file(filename):
    """Download a specific transcription file."""
    try:
        # Validate path to prevent directory traversal
        file_path = validate_safe_path(OUTPUT_FOLDER, filename)

        if not file_path:
            return jsonify({"status": "error", "message": "File not found"}), 404

        return send_file(file_path, as_attachment=True, download_name=filename)

    except Exception as e:
        logger.error(f"Error downloading transcription {filename}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/history/transcriptions/<filename>", methods=["DELETE"])
def delete_transcription(filename):
    """
    Delete a specific transcription.

    SECURITY: Protected by custom header validation to prevent CSRF attacks.
    """
    # Validate CSRF protection
    if not validate_csrf_protection():
        return jsonify({"status": "error", "message": "Forbidden: Invalid request origin"}), 403

    try:
        # Validate path to prevent directory traversal
        file_path = validate_safe_path(OUTPUT_FOLDER, filename)

        if not file_path:
            return jsonify({"status": "error", "message": "File not found"}), 404

        file_path.unlink()
        logger.info(f"Deleted transcription: {filename}")

        return jsonify({"status": "success", "message": f"Deleted {filename}"})

    except Exception as e:
        logger.error(f"Error deleting transcription {filename}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/history/transcriptions/all", methods=["DELETE"])
def delete_all_transcriptions():
    """
    Delete all transcriptions.

    SECURITY: Protected by custom header validation to prevent CSRF attacks.
    """
    # Validate CSRF protection
    if not validate_csrf_protection():
        return jsonify({"status": "error", "message": "Forbidden: Invalid request origin"}), 403

    try:
        count = 0
        if OUTPUT_FOLDER.exists():
            for file_path in OUTPUT_FOLDER.glob("*.txt"):
                file_path.unlink()
                count += 1

        logger.info(f"Deleted {count} transcriptions")
        return jsonify({"status": "success", "count": count, "message": f"Deleted {count} transcriptions"})

    except Exception as e:
        logger.error(f"Error deleting all transcriptions: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/history/uploads", methods=["GET"])
def list_uploads():
    """List all uploaded files."""
    try:
        uploads = []
        if UPLOAD_FOLDER.exists():
            for file_path in UPLOAD_FOLDER.glob("*"):
                if file_path.is_file():
                    stat = file_path.stat()
                    uploads.append(
                        {
                            "filename": file_path.name,
                            "size": stat.st_size,
                            "size_mb": round(stat.st_size / (1024 * 1024), 2),
                            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        }
                    )

        # Sort by modified date (newest first)
        uploads.sort(key=lambda x: x["modified"], reverse=True)
        return jsonify(uploads)

    except Exception as e:
        logger.error(f"Error listing uploads: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/history/uploads/<filename>", methods=["GET"])
def download_upload(filename):
    """Download a specific uploaded file."""
    try:
        # Validate path to prevent directory traversal
        file_path = validate_safe_path(UPLOAD_FOLDER, filename)

        if not file_path:
            return jsonify({"status": "error", "message": "File not found"}), 404

        return send_file(file_path, as_attachment=True, download_name=filename)

    except Exception as e:
        logger.error(f"Error downloading upload {filename}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/history/uploads/<filename>", methods=["DELETE"])
def delete_upload(filename):
    """
    Delete a specific uploaded file.

    SECURITY: Protected by custom header validation to prevent CSRF attacks.
    """
    # Validate CSRF protection
    if not validate_csrf_protection():
        return jsonify({"status": "error", "message": "Forbidden: Invalid request origin"}), 403

    try:
        # Validate path to prevent directory traversal
        file_path = validate_safe_path(UPLOAD_FOLDER, filename)

        if not file_path:
            return jsonify({"status": "error", "message": "File not found"}), 404

        file_path.unlink()
        logger.info(f"Deleted upload: {filename}")

        return jsonify({"status": "success", "message": f"Deleted {filename}"})

    except Exception as e:
        logger.error(f"Error deleting upload {filename}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/history/uploads/all", methods=["DELETE"])
def delete_all_uploads():
    """
    Delete all uploaded files.

    SECURITY: Protected by custom header validation to prevent CSRF attacks.
    """
    # Validate CSRF protection
    if not validate_csrf_protection():
        return jsonify({"status": "error", "message": "Forbidden: Invalid request origin"}), 403

    try:
        count = 0
        if UPLOAD_FOLDER.exists():
            for file_path in UPLOAD_FOLDER.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
                    count += 1

        logger.info(f"Deleted {count} uploads")
        return jsonify({"status": "success", "count": count, "message": f"Deleted {count} uploads"})

    except Exception as e:
        logger.error(f"Error deleting all uploads: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# REMOVED: /api/shutdown endpoint
# This endpoint was removed for security reasons - it allowed any website to kill
# the server via CSRF attack, interrupting active transcriptions.
#
# To shutdown the server:
# 1. Press Ctrl+C in the terminal where the server is running
# 2. Or use: pkill -f "python app.py"
# 3. Or close the terminal window


@app.route("/api/system/memory", methods=["GET"])
def get_memory_status():
    """Get current system and process memory usage."""
    try:
        # System memory
        system_memory = psutil.virtual_memory()

        # Process memory (this Flask app)
        process = psutil.Process()
        process_memory = process.memory_info()

        # Determine status level
        percent = system_memory.percent
        if percent >= 90:
            status = "critical"
        elif percent >= 80:
            status = "warning"
        else:
            status = "normal"

        return jsonify(
            {
                "system": {
                    "total_gb": round(system_memory.total / (1024**3), 2),
                    "available_gb": round(system_memory.available / (1024**3), 2),
                    "used_gb": round(system_memory.used / (1024**3), 2),
                    "percent": round(percent, 1),
                    "status": status,
                },
                "process": {
                    "rss_mb": round(process_memory.rss / (1024**2), 2),
                    "vms_mb": round(process_memory.vms / (1024**2), 2),
                    "percent": round(process.memory_percent(), 2),
                },
            }
        )

    except Exception as e:
        logger.error(f"Error getting memory status: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/version", methods=["GET"])
def get_version():
    """Get current application version."""
    return jsonify({"version": get_current_version(), "app_name": "Voxtral Transcription"})


@app.route("/api/updates/check", methods=["GET"])
def check_updates():
    """Check for available updates from GitHub."""
    try:
        update_info = check_for_updates()
        return jsonify(update_info)
    except Exception as e:
        logger.error(f"Error checking for updates: {e}")
        return jsonify({"update_available": False, "current_version": get_current_version(), "error": str(e)}), 500


def perform_zip_update():
    """
    Perform ZIP-based update for non-git installations.

    Returns:
        tuple: (success: bool, message: str, data: dict)
    """
    try:
        # Emit progress update
        def emit_progress(stage, message, progress=0):
            socketio.emit("update_progress", {"stage": stage, "message": message, "progress": progress})
            logger.info(f"Update progress: {stage} - {message} ({progress}%)")

        emit_progress("checking", "Checking for updates...", 5)

        # Get update information
        update_info = check_for_updates()
        if not update_info.get("update_available"):
            return False, "No update available", {}

        download_url = update_info.get("download_url")
        if not download_url:
            return False, "Download URL not available", {}

        emit_progress("downloading", "Downloading update...", 10)

        # Create temporary working directory OUTSIDE install tree to avoid path issues when moving
        temp_dir = Path(tempfile.mkdtemp(prefix="voxtral_update_"))
        logger.info(f"Created temp directory: {temp_dir}")

        try:
            # Download ZIP file with progress
            zip_path = temp_dir / "update.zip"
            logger.info(f"Downloading from {download_url} to {zip_path}")

            response = requests.get(download_url, stream=True, timeout=300)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            last_progress = 0  # Track last emitted progress to throttle events

            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = 10 + int((downloaded / total_size) * 40)  # 10-50%
                            # Only emit if progress changed by at least 1% to avoid flooding
                            if progress - last_progress >= 1:
                                emit_progress("downloading", f"Downloading... {downloaded // 1024 // 1024}MB", progress)
                                last_progress = progress

            emit_progress("extracting", "Extracting files...", 55)

            # Extract ZIP
            extract_dir = temp_dir / "extracted"
            extract_dir.mkdir(exist_ok=True)

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            # Find the extracted folder (GitHub creates a folder like "debrockb-transcribe-voxtral-<hash>")
            extracted_folders = list(extract_dir.iterdir())
            if not extracted_folders:
                return False, "Extracted archive is empty", {}

            source_folder = extracted_folders[0]
            logger.info(f"Extracted to {source_folder}")

            emit_progress("backing_up", "Backing up user data...", 60)

            # Preserve user data
            backup_dir = temp_dir / "backup"
            backup_dir.mkdir(exist_ok=True)

            # Backup config, recordings, uploads, outputs
            user_data_paths = [
                ("VoxtralApp/config.json", "config.json"),
                ("VoxtralApp/uploads", "uploads"),
                ("VoxtralApp/output", "output"),
                ("VoxtralApp/recordings", "recordings"),
            ]

            for rel_path, backup_name in user_data_paths:
                src = BASE_DIR.parent / rel_path
                if src.exists():
                    dst = backup_dir / backup_name
                    if src.is_file():
                        shutil.copy2(src, dst)
                    else:
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    logger.info(f"Backed up {rel_path}")

            emit_progress("installing", "Installing update...", 70)

            # Atomically replace installation
            install_root = BASE_DIR.parent
            backup_root = install_root.parent / f"{install_root.name}-backup-{int(time.time())}"

            # Rename current installation to backup
            logger.info(f"Backing up current installation to {backup_root}")
            shutil.move(str(install_root), str(backup_root))

            try:
                # Move new version into place
                logger.info(f"Moving {source_folder} to {install_root}")
                shutil.move(str(source_folder), str(install_root))

                emit_progress("restoring", "Restoring user data...", 80)

                # Restore user data
                for rel_path, backup_name in user_data_paths:
                    src = backup_dir / backup_name
                    if src.exists():
                        dst = BASE_DIR.parent / rel_path
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        if src.is_file():
                            shutil.copy2(src, dst)
                        else:
                            shutil.copytree(src, dst, dirs_exist_ok=True)
                        logger.info(f"Restored {rel_path}")

                emit_progress("dependencies", "Updating dependencies...", 85)

                # Update dependencies
                requirements_file = BASE_DIR / "requirements.txt"
                if requirements_file.exists():
                    if sys.platform == "win32":
                        pip_exe = BASE_DIR / "voxtral_env" / "Scripts" / "pip.exe"
                    else:
                        pip_exe = BASE_DIR / "voxtral_env" / "bin" / "pip"

                    if pip_exe.exists():
                        pip_result = subprocess.run(
                            [str(pip_exe), "install", "-r", str(requirements_file), "--upgrade"],
                            capture_output=True,
                            text=True,
                            timeout=300,
                        )
                        if pip_result.returncode != 0:
                            logger.warning(f"Pip install warnings: {pip_result.stderr}")

                emit_progress("complete", "Update complete!", 100)

                # Clean up temp directory
                shutil.rmtree(temp_dir, ignore_errors=True)

                return True, f"Updated to version {update_info['latest_version']}", {"backup_path": str(backup_root)}

            except Exception as e:
                # Rollback: restore backup
                logger.error(f"Update failed, rolling back: {e}")
                if install_root.exists():
                    shutil.rmtree(install_root, ignore_errors=True)
                shutil.move(str(backup_root), str(install_root))
                raise

        finally:
            # Clean up temp directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    except requests.exceptions.RequestException as e:
        return False, f"Download failed: {str(e)}", {}
    except zipfile.BadZipFile:
        return False, "Downloaded file is not a valid ZIP archive", {}
    except Exception as e:
        logger.error(f"ZIP update failed: {e}", exc_info=True)
        return False, f"Update failed: {str(e)}", {}


@app.route("/api/updates/install", methods=["POST"])
def install_update():  # noqa: C901
    """
    Install available update from GitHub.

    Supports both git-based and ZIP-based installations.
    SECURITY: Protected by custom header validation to prevent CSRF attacks.
    CORS alone doesn't protect against HTML form submissions from malicious sites.
    """
    # Validate CSRF protection
    if not validate_csrf_protection():
        return jsonify({"status": "error", "message": "Forbidden: Invalid request origin"}), 403

    try:
        logger.info("Starting automatic update process...")

        # Check if running from git repository
        git_dir = BASE_DIR.parent / ".git"
        if not git_dir.exists():
            # ZIP-based installation - use ZIP update mechanism
            logger.info("Git repository not found, using ZIP-based update...")
            success, message, data = perform_zip_update()

            if not success:
                return jsonify({"status": "error", "message": message}), 400

            # Schedule restart after a brief delay
            def restart_app():
                time.sleep(2)  # Give time for response to be sent
                logger.info("Restarting application...")
                os.execv(sys.executable, [sys.executable] + sys.argv)

            restart_thread = threading.Thread(target=restart_app, daemon=True)
            restart_thread.start()

            return jsonify(
                {
                    "status": "success",
                    "message": f"{message} Application will restart in 2 seconds...",
                    "updated": True,
                    **data,
                }
            )

        # Determine pip executable based on platform
        if sys.platform == "win32":
            # Windows
            pip_exe = BASE_DIR / "voxtral_env" / "Scripts" / "pip.exe"
        else:
            # Mac/Linux
            pip_exe = BASE_DIR / "voxtral_env" / "bin" / "pip"

        # Step 1: Run git pull
        logger.info("Pulling latest changes from GitHub...")
        git_result = subprocess.run(["git", "pull"], cwd=str(BASE_DIR.parent), capture_output=True, text=True, timeout=60)

        if git_result.returncode != 0:
            error_msg = f"Git pull failed: {git_result.stderr}"
            logger.error(error_msg)
            return jsonify({"status": "error", "message": error_msg}), 500

        logger.info(f"Git pull output: {git_result.stdout}")

        # Check if anything was updated
        if "Already up to date" in git_result.stdout or "Already up-to-date" in git_result.stdout:
            return jsonify({"status": "success", "message": "Already up to date. No changes to install.", "updated": False})

        # Step 2: Update dependencies
        logger.info("Updating dependencies...")
        requirements_file = BASE_DIR / "requirements.txt"

        if requirements_file.exists():
            pip_result = subprocess.run(
                [str(pip_exe), "install", "-r", str(requirements_file), "--upgrade"],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if pip_result.returncode != 0:
                logger.warning(f"Pip install warnings: {pip_result.stderr}")
                # Don't fail on pip warnings, just log them

        logger.info("Update completed successfully!")

        # Schedule restart after a brief delay
        def restart_app():
            time.sleep(2)  # Give time for response to be sent
            logger.info("Restarting application...")
            os.execv(sys.executable, [sys.executable] + sys.argv)

        restart_thread = threading.Thread(target=restart_app, daemon=True)
        restart_thread.start()

        return jsonify(
            {
                "status": "success",
                "message": "Update installed successfully. Application will restart in 2 seconds...",
                "updated": True,
            }
        )

    except subprocess.TimeoutExpired:
        error_msg = "Update process timed out"
        logger.error(error_msg)
        return jsonify({"status": "error", "message": error_msg}), 500
    except Exception as e:
        error_msg = f"Error installing update: {str(e)}"
        logger.error(error_msg)
        return jsonify({"status": "error", "message": error_msg}), 500


# WebSocket events


@socketio.on("connect")
def handle_connect():
    """Handle client connection."""
    logger.info("Client connected")
    emit("connected", {"message": "Connected to Voxtral server"})


@socketio.on("disconnect")
def handle_disconnect():
    """Handle client disconnection."""
    logger.info("Client disconnected")


def start_memory_monitor():
    """Start background thread to monitor memory and emit warnings via WebSocket."""

    def monitor():
        """Background memory monitoring loop."""
        while True:
            try:
                memory = psutil.virtual_memory()
                percent = memory.percent

                if percent >= 80:
                    level = "critical" if percent >= 90 else "warning"
                    message = (
                        f"CRITICAL: Memory usage is very high ({percent}%). "
                        "Consider stopping transcription to prevent system slowdown."
                        if percent >= 90
                        else f"WARNING: Memory usage is high ({percent}%). Transcription may slow down."
                    )

                    socketio.emit(
                        "memory_warning",
                        {
                            "level": level,
                            "percent": round(percent, 1),
                            "available_gb": round(memory.available / (1024**3), 2),
                            "message": message,
                        },
                    )

                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Error in memory monitor: {e}")
                time.sleep(30)  # Wait longer if error occurs

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    logger.info("Memory monitoring started")


def initialize_engine(model_version=None):
    """
    Initialize the transcription engine with specified model.

    Args:
        model_version: Model version to load ('full' or 'quantized'). Uses config default if None.
    """
    global transcription_engine, engine_loading

    # Skip model loading in test mode (saves memory and time in CI/CD)
    if os.environ.get("TESTING") == "1":
        logger.info("Test mode detected - skipping model initialization")
        from unittest.mock import MagicMock

        transcription_engine = MagicMock()
        transcription_engine.device = "cpu"
        transcription_engine.transcribe_file = MagicMock(return_value=None)
        engine_loading = False
        return

    try:
        engine_loading = True

        # Get model configuration
        if model_version:
            model_config = config.get_model_config(model_version)
        else:
            model_config = config.get_model_config()

        model_id = model_config.get("id")
        model_name = model_config.get("name", "Unknown")
        quantization = model_config.get("quantization")

        logger.info(f"Initializing transcription engine with model: {model_name} ({model_id})")
        if quantization:
            logger.info(f"Using {quantization} quantization")

        # Emit loading progress via WebSocket
        socketio.emit("model_loading", {"status": "loading", "model": model_name, "message": f"Loading {model_name}..."})

        transcription_engine = TranscriptionEngine(model_id=model_id, quantization=quantization)

        # Save the selected model version to config
        if model_version:
            config.set_model_version(model_version)

        socketio.emit(
            "model_loading", {"status": "loaded", "model": model_name, "message": f"{model_name} loaded successfully!"}
        )

        logger.info(f"Engine initialized successfully with {model_name}")
        engine_loading = False

    except Exception as e:
        logger.error(f"Failed to initialize engine: {e}")
        engine_loading = False
        socketio.emit("model_loading", {"status": "error", "message": f"Failed to load model: {str(e)}"})
        raise


if __name__ == "__main__":
    # DON'T load the model on startup - let users select it first!
    # Model will be loaded after user selection in the web UI

    # Start memory monitoring
    start_memory_monitor()

    # Start Flask-SocketIO server
    # Using port 8000 as port 5000 is often used by macOS AirPlay Receiver
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting Voxtral Web Application on port {port}...")
    logger.info(f"Access the application at: http://localhost:{port}")
    logger.info("Model will be loaded after user selection in the web UI")

    socketio.run(app, host="0.0.0.0", port=port, debug=True, allow_unsafe_werkzeug=True)
