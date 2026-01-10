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
# Use threading mode explicitly - compatible with all environments without extra dependencies
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins=ALLOWED_ORIGINS)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global storage for jobs and files
jobs = {}  # {job_id: {status, progress, transcript, etc}}
uploaded_files = {}  # {file_id: {filename, path, size, etc}}

# Transcription engine (loaded after user selects model)
transcription_engine = None
engine_loading = False  # Flag to track if model is currently loading
engine_lock = threading.Lock()  # Prevent concurrent transcriptions (engine is not thread-safe)


def validate_csrf_protection():
    """
    Validate that state-changing requests come from the actual application.

    Requires custom header that HTML forms cannot set. This prevents CSRF attacks
    via form submissions from malicious websites, since browsers block custom headers
    in forms, and CORS blocks fetch/XHR from cross-origin sites.
    """
    # Skip CSRF validation in test mode
    if app.config.get("TESTING"):
        return True

    # Require custom header that forms cannot set
    custom_header = request.headers.get("X-Voxtral-Request")
    if custom_header != "voxtral-web-ui":
        logger.warning(f"CSRF validation failed: Missing or invalid X-Voxtral-Request header from {request.remote_addr}")
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


def transcribe_in_background(job_id, file_path, language, output_path, cleanup_path=None):
    """
    Background task for transcription.

    Args:
        job_id: Unique job identifier
        file_path: Path to audio file to transcribe
        language: Language code
        output_path: Path to write transcription
        cleanup_path: Optional path to delete after transcription (e.g., converted video WAV file)
    """
    # Acquire lock to ensure only one transcription runs at a time
    with engine_lock:
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
                # Only store metadata - transcript is already written to disk
                # This prevents memory leak from storing large transcripts in RAM
                jobs[job_id].update(
                    {
                        "status": "complete",
                        "duration": result["duration_minutes"],
                        "word_count": result["word_count"],
                        "char_count": result["char_count"],
                        "confidence": result.get("confidence", 0.0),
                        "completed_at": datetime.now().isoformat(),
                    }
                )

                socketio.emit(
                    "transcription_complete",
                    {
                        "job_id": job_id,
                        "transcript": result["transcript"],  # Send once via WebSocket then discard
                        "duration": result["duration_minutes"],
                        "word_count": result["word_count"],
                        "confidence": result.get("confidence", 0.0),
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

        finally:
            # Clean up converted audio file if it was from video
            if cleanup_path and Path(cleanup_path).exists():
                try:
                    Path(cleanup_path).unlink()
                    logger.info(f"Cleaned up converted audio file: {cleanup_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up converted audio file {cleanup_path}: {e}")


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
                    "backend": model_info.get("backend", "voxtral"),
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
    global engine_loading

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
        # Set flag BEFORE starting thread to prevent race condition
        engine_loading = True

        # Initialize engine in background thread to not block the API
        def load_model():
            try:
                initialize_engine(model_version)
            except Exception as e:
                global engine_loading
                engine_loading = False
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
        engine_loading = False
        return jsonify({"status": "error", "message": f"Failed to start model loading: {str(e)}"}), 500


@app.route("/api/model/switch", methods=["POST"])
def switch_model():
    """
    Switch to a different model by unloading current model and loading a new one.

    SECURITY: Protected by custom header validation to prevent CSRF attacks.
    """
    global transcription_engine, engine_loading

    # Validate CSRF protection
    if not validate_csrf_protection():
        return jsonify({"status": "error", "message": "Forbidden: Invalid request origin"}), 403

    # Check if engine is currently loading
    if engine_loading:
        return jsonify({"status": "error", "message": "Model is currently loading. Please wait."}), 409

    # Check if a transcription is in progress
    if engine_lock.locked():
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Cannot switch models while a transcription is in progress. Please wait for it to complete.",
                }
            ),
            409,
        )

    # Validate JSON request body
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"status": "error", "message": "Invalid or missing JSON request body"}), 400

    model_version = data.get("version")
    if not model_version:
        return jsonify({"status": "error", "message": "Model version is required"}), 400

    # Validate model version
    available_models = config.get_all_models()
    if model_version not in available_models:
        return jsonify({"status": "error", "message": f"Invalid model version: {model_version}"}), 400

    # Check if already on this model
    current_version = config.get("model.version", "full")
    if transcription_engine and model_version == current_version:
        return jsonify({"status": "error", "message": f"Model '{model_version}' is already loaded"}), 400

    try:
        # Set loading flag
        engine_loading = True

        # Unload current model if one is loaded
        if transcription_engine is not None:
            logger.info(f"Unloading current model: {current_version}")
            socketio.emit(
                "model_loading", {"status": "unloading", "model": current_version, "message": "Unloading current model..."}
            )

            # Clear the engine reference to allow garbage collection
            del transcription_engine
            transcription_engine = None

            # Force garbage collection to free GPU memory
            import gc

            gc.collect()

            # Clear CUDA cache if available
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    # MPS doesn't have an explicit cache clear, but gc.collect helps
                    pass
            except Exception as e:
                logger.warning(f"Could not clear GPU cache: {e}")

            logger.info("Current model unloaded successfully")

        # Initialize new model in background thread
        def load_new_model():
            try:
                initialize_engine(model_version)
            except Exception as e:
                global engine_loading
                engine_loading = False
                logger.error(f"Failed to load model in background: {e}")
                socketio.emit("model_loading", {"status": "error", "message": f"Failed to load model: {str(e)}"})

        thread = threading.Thread(target=load_new_model, daemon=False)
        thread.start()

        model_config = config.get_model_config(model_version)
        return jsonify(
            {
                "status": "switching",
                "message": f"Switching to {model_config.get('name')}...",
                "model": model_config.get("name"),
                "version": model_version,
            }
        )

    except Exception as e:
        engine_loading = False
        logger.error(f"Failed to switch model: {e}")
        return jsonify({"status": "error", "message": f"Failed to switch model: {str(e)}"}), 500


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """
    Handle file upload.

    SECURITY: Protected by custom header validation to prevent CSRF attacks and resource abuse.
    """
    # Validate CSRF protection
    if not validate_csrf_protection():
        return jsonify({"status": "error", "message": "Forbidden: Invalid request origin"}), 403

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
    """
    Start transcription job.

    SECURITY: Protected by custom header validation to prevent CSRF attacks and resource abuse.
    """
    # Validate CSRF protection
    if not validate_csrf_protection():
        return jsonify({"status": "error", "message": "Forbidden: Invalid request origin"}), 403

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

    # Check if another transcription is running (engine is not thread-safe)
    if not engine_lock.acquire(blocking=False):
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Another transcription is currently in progress. Please wait for it to complete.",
                }
            ),
            429,  # Too Many Requests
        )

    # Release lock immediately - will be acquired in background thread
    engine_lock.release()

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

        # Check if uploaded file still exists
        if not file_path.exists():
            # Clean up stale metadata
            uploaded_files.pop(file_id, None)
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Uploaded file no longer exists. Please re-upload the file.",
                    }
                ),
                404,
            )

        # Track cleanup path for video conversions
        cleanup_path = None

        # If video, convert to audio first
        if file_info["is_video"]:
            audio_path = UPLOAD_FOLDER / f"{file_id}_audio.wav"

            if not convert_video_to_audio(file_path, audio_path):
                return jsonify({"status": "error", "message": "Failed to convert video to audio"}), 500

            # Update file path to converted audio and mark for cleanup
            file_path = audio_path
            cleanup_path = audio_path

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
        thread = threading.Thread(
            target=transcribe_in_background, args=(job_id, file_path, language, output_path, cleanup_path)
        )
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
            "confidence": job.get("confidence", 0.0),
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

    Strategy: Two-stage update to avoid Windows file lock issues:
    1. Download and prepare new version (in-process)
    2. Launch external script to swap directories and restart (out-of-process)

    Returns:
        tuple: (success: bool, message: str, data: dict)
    """
    stage_start_time = time.time()

    try:
        # Emit progress update with timestamps
        def emit_progress(stage, message, progress=0):
            timestamp = datetime.now().strftime("%H:%M:%S")
            socketio.emit(
                "update_progress", {"stage": stage, "message": message, "progress": progress, "timestamp": timestamp}
            )
            logger.info(f"[{timestamp}] Update progress: {stage} - {message} ({progress}%)")

        emit_progress("checking", "Checking for updates...", 5)
        logger.info(f"=== Starting ZIP update process === [PID: {os.getpid()}, CWD: {os.getcwd()}]")

        # Get update information
        update_info = check_for_updates()
        if not update_info.get("update_available"):
            logger.info("No update available")
            return False, "No update available", {}

        download_url = update_info.get("download_url")
        latest_version = update_info.get("latest_version")
        if not download_url:
            logger.error("Download URL not available in update info")
            return False, "Download URL not available", {}

        logger.info(f"Update available: {latest_version} from {download_url}")
        emit_progress("downloading", "Downloading update...", 10)

        # Create temporary working directory OUTSIDE install tree
        temp_dir = Path(tempfile.mkdtemp(prefix="voxtral_update_"))
        logger.info(f"Created temp directory: {temp_dir} (exists: {temp_dir.exists()})")

        try:
            # Stage 1: Download
            download_start = time.time()
            zip_path = temp_dir / "update.zip"
            logger.info(f"[Stage 1: Download] Starting download from {download_url}")
            logger.info(f"  Target: {zip_path}")

            response = requests.get(download_url, stream=True, timeout=300)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            last_progress = 0

            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = 10 + int((downloaded / total_size) * 30)  # 10-40%
                            if progress - last_progress >= 1:
                                emit_progress("downloading", f"Downloading... {downloaded // 1024 // 1024}MB", progress)
                                last_progress = progress

            download_duration = time.time() - download_start
            logger.info(f"[Stage 1: Download] Complete: {downloaded} bytes in {download_duration:.1f}s")

            # Stage 2: Extract
            extract_start = time.time()
            emit_progress("extracting", "Extracting files...", 45)
            logger.info(f"[Stage 2: Extract] Starting extraction")

            extract_dir = temp_dir / "extracted"
            extract_dir.mkdir(exist_ok=True)

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            extracted_folders = list(extract_dir.iterdir())
            if not extracted_folders:
                logger.error("[Stage 2: Extract] Extracted archive is empty")
                return False, "Extracted archive is empty", {}

            new_install_dir = extracted_folders[0]
            extract_duration = time.time() - extract_start
            logger.info(f"[Stage 2: Extract] Complete in {extract_duration:.1f}s")
            logger.info(f"  New installation directory: {new_install_dir} (exists: {new_install_dir.exists()})")

            # Stage 3: Merge config
            config_start = time.time()
            emit_progress("configuring", "Merging configuration...", 55)
            logger.info(f"[Stage 3: Config] Merging configuration files")

            old_config_path = BASE_DIR / "config.json"
            new_config_path = new_install_dir / "VoxtralApp" / "config.json"

            def deep_merge_config(old_dict, new_dict):
                """
                Recursively merge old config into new config, preserving user settings.
                Strategy:
                - Start with new_dict (contains new schema and defaults)
                - For each key in old_dict:
                  - If value is a dict and exists in new_dict, recursively merge
                  - If value is primitive, preserve old value (user preference)
                  - Skip preserving app.version (always use new version)
                """
                merged = new_dict.copy()

                for key, old_value in old_dict.items():
                    # Special case: never preserve app.version (always use new)
                    if key == "app":
                        if isinstance(old_value, dict) and isinstance(merged.get(key), dict):
                            # Preserve other app settings but not version
                            app_merged = merged[key].copy()
                            for app_key, app_old_value in old_value.items():
                                if app_key != "version":
                                    app_merged[app_key] = app_old_value
                            merged[key] = app_merged
                        continue

                    # If key exists in new config
                    if key in merged:
                        new_value = merged[key]
                        # Both are dicts: recursively merge
                        if isinstance(old_value, dict) and isinstance(new_value, dict):
                            merged[key] = deep_merge_config(old_value, new_value)
                        # Old value is primitive: preserve user setting
                        else:
                            merged[key] = old_value
                    # Key doesn't exist in new config: preserve anyway (backward compat)
                    else:
                        merged[key] = old_value

                return merged

            if old_config_path.exists() and new_config_path.exists():
                try:
                    import json

                    # Load both configs
                    with open(old_config_path, "r", encoding="utf-8") as f:
                        old_config = json.load(f)
                    with open(new_config_path, "r", encoding="utf-8") as f:
                        new_config = json.load(f)

                    # Deep merge: preserve ALL user settings
                    merged_config = deep_merge_config(old_config, new_config)

                    # Log what was preserved
                    if "model" in old_config and "version" in old_config["model"]:
                        logger.info(f"  Preserved user model selection: {old_config['model']['version']}")

                    preserved_keys = []
                    for key in old_config.keys():
                        if key != "app":  # app.version is always from new
                            preserved_keys.append(key)
                    if preserved_keys:
                        logger.info(f"  Preserved user settings from: {', '.join(preserved_keys)}")

                    # Write merged config
                    with open(new_config_path, "w", encoding="utf-8") as f:
                        json.dump(merged_config, f, indent=2)

                    logger.info(
                        f"  Config merged successfully (new version: {merged_config.get('app', {}).get('version', 'unknown')})"
                    )
                except Exception as e:
                    logger.warning(f"  Config merge failed, using new config: {e}")

            config_duration = time.time() - config_start
            logger.info(f"[Stage 3: Config] Complete in {config_duration:.1f}s")

            # Stage 4: Copy user data
            data_start = time.time()
            emit_progress("migrating", "Migrating user data...", 65)
            logger.info(f"[Stage 4: Data Migration] Copying user data to new installation")
            logger.info(f"  Source: {BASE_DIR}")
            logger.info(f"  Destination: {new_install_dir / 'VoxtralApp'}")

            user_data_dirs = [
                "uploads",
                "output",
                "recordings",
                "transcriptions_voxtral_final",
            ]

            files_copied = 0
            for dir_name in user_data_dirs:
                src = BASE_DIR / dir_name
                dst = new_install_dir / "VoxtralApp" / dir_name

                logger.info(f"  Processing: {dir_name}")
                logger.info(f"    From: {src} (exists: {src.exists()})")
                logger.info(f"    To: {dst}")

                if src.exists():
                    try:
                        if src.is_file():
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src, dst)
                            logger.info(f"    ✓ Copied file: {dir_name}")
                            files_copied += 1
                        else:
                            shutil.copytree(src, dst, dirs_exist_ok=True)
                            file_count = len(list(src.rglob("*")))
                            logger.info(f"    ✓ Copied directory: {dir_name} ({file_count} items)")
                            files_copied += file_count
                    except Exception as e:
                        logger.error(f"    ✗ Failed to copy {dir_name}: {e}")
                        raise
                else:
                    logger.info(f"    - Skipped (does not exist)")

            data_duration = time.time() - data_start
            logger.info(f"[Stage 4: Data Migration] Complete in {data_duration:.1f}s ({files_copied} items copied)")

            # Stage 5: Create updater script
            script_start = time.time()
            emit_progress("preparing", "Preparing update script...", 75)
            logger.info(f"[Stage 5: Script] Creating platform-specific updater script")

            install_root = BASE_DIR.parent
            backup_dir = install_root.parent / f"{install_root.name}-backup-{int(time.time())}"

            logger.info(f"  Current installation: {install_root} (exists: {install_root.exists()})")
            logger.info(f"  New installation: {new_install_dir} (exists: {new_install_dir.exists()})")
            logger.info(f"  Backup location: {backup_dir}")

            if sys.platform == "win32":
                # Windows batch script with robocopy for reliability
                script_path = temp_dir / "voxtral_updater.bat"
                log_path = temp_dir / "voxtral_updater.log"
                failed_marker = install_root / ".UPDATE_FAILED"
                launcher_script = install_root / "Start Voxtral Web - Windows.bat"
                pid = os.getpid()

                script_content = f"""@echo off
setlocal enabledelayedexpansion
set LOG_FILE={log_path}
set FAILED_MARKER={failed_marker}
set BACKUP_DIR={backup_dir}
set INSTALL_ROOT={install_root}
set NEW_DIR={new_install_dir}
set LAUNCHER={launcher_script}

echo Voxtral Update Script > "%LOG_FILE%"
echo ===================== >> "%LOG_FILE%"
echo Started: %DATE% %TIME% >> "%LOG_FILE%"
echo PID: {pid} >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

echo Voxtral Update Script
echo =====================
echo Log: %LOG_FILE%

REM Wait for Flask process to exit (up to 120 seconds)
echo Waiting for application (PID {pid})...
echo Waiting for Flask {pid}... >> "%LOG_FILE%"

set /a WAIT_COUNT=0
:WAIT_LOOP
tasklist /FI "PID eq {pid}" 2>nul | find /I "{pid}" >nul
if errorlevel 1 goto PROCESS_EXITED

set /a WAIT_COUNT+=1
if %WAIT_COUNT% GTR 120 (
    echo WARNING: Still waiting after 120s >> "%LOG_FILE%"
    goto PROCESS_EXITED
)
timeout /t 1 /nobreak >nul
goto WAIT_LOOP

:PROCESS_EXITED
echo Process exited after %WAIT_COUNT%s >> "%LOG_FILE%"
echo Application closed

REM Wait additional 10 seconds for Windows to release file locks
echo Waiting for file locks to release... >> "%LOG_FILE%"
timeout /t 10 /nobreak >nul

REM Clean old backup if exists
if exist "%BACKUP_DIR%" (
    echo Cleaning old backup... >> "%LOG_FILE%"
    rmdir /S /Q "%BACKUP_DIR%" 2>>"%LOG_FILE%"
)

REM NEW APPROACH: Use MOVE (rename) instead of DELETE
REM Windows can rename locked directories even when it can't delete them
echo Moving current to backup...
set /a RETRY_COUNT=0
:MOVE_RETRY
move "%INSTALL_ROOT%" "%BACKUP_DIR%" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    set /a RETRY_COUNT+=1
    if %RETRY_COUNT% LEQ 30 (
        echo Retry %RETRY_COUNT%/30: Waiting for locks... >> "%LOG_FILE%"
        timeout /t 2 /nobreak >nul
        goto MOVE_RETRY
    )
    echo ERROR: Could not move after 30 retries >> "%LOG_FILE%"
    echo Update failed: Could not move current installation > "%FAILED_MARKER%"
    echo. >> "%FAILED_MARKER%"
    echo Windows file locks prevented moving directory. >> "%FAILED_MARKER%"
    echo. >> "%FAILED_MARKER%"
    echo TROUBLESHOOTING: >> "%FAILED_MARKER%"
    echo 1. Close all applications that might access files: >> "%FAILED_MARKER%"
    echo    - Windows Explorer windows showing this folder >> "%FAILED_MARKER%"
    echo    - Any text editors with files open from this folder >> "%FAILED_MARKER%"
    echo    - Antivirus or backup software >> "%FAILED_MARKER%"
    echo 2. Restart your computer to release all file locks >> "%FAILED_MARKER%"
    echo 3. Run the update again after restart >> "%FAILED_MARKER%"
    echo. >> "%FAILED_MARKER%"
    echo MANUAL RECOVERY (if restart doesn't help): >> "%FAILED_MARKER%"
    echo 1. Download the latest version from GitHub >> "%FAILED_MARKER%"
    echo 2. Extract to a NEW location >> "%FAILED_MARKER%"
    echo 3. Copy your config.json and data folders to the new location >> "%FAILED_MARKER%"
    echo. >> "%FAILED_MARKER%"
    echo Log: %LOG_FILE% >> "%FAILED_MARKER%"
    echo.
    echo ERROR: Could not move current installation
    echo.
    echo File locks prevented installation. Please see .UPDATE_FAILED file.
    pause
    exit /b 1
)
echo Move complete >> "%LOG_FILE%"

REM Move new version into place with retry
echo Installing new version...
set /a RETRY_COUNT=0
:INSTALL_RETRY
move "%NEW_DIR%" "%INSTALL_ROOT%" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    set /a RETRY_COUNT+=1
    if %RETRY_COUNT% LEQ 10 (
        echo Retry %RETRY_COUNT%/10: Waiting... >> "%LOG_FILE%"
        timeout /t 2 /nobreak >nul
        goto INSTALL_RETRY
    )
    echo ERROR: Install failed, restoring... >> "%LOG_FILE%"
    move "%BACKUP_DIR%" "%INSTALL_ROOT%" >> "%LOG_FILE%" 2>&1
    echo Update failed: Could not install new version > "%FAILED_MARKER%"
    echo Restored from backup. >> "%FAILED_MARKER%"
    echo Log: %LOG_FILE% >> "%FAILED_MARKER%"
    echo ERROR: Install failed, restored from backup
    pause
    exit /b 1
)
echo Install complete >> "%LOG_FILE%"

REM Restart via launcher script
echo Restarting...
echo Starting "%LAUNCHER%"... >> "%LOG_FILE%"

if exist "%LAUNCHER%" (
    start "" "%LAUNCHER%"
) else (
    echo WARNING: Launcher not found >> "%LOG_FILE%"
    cd /d "%INSTALL_ROOT%\\VoxtralApp"
    start "" voxtral_env\\Scripts\\python.exe app.py
)

echo Completed: %DATE% %TIME% >> "%LOG_FILE%"
echo Update successful!
echo.
echo Old version backed up to: %BACKUP_DIR%
echo You can safely delete the backup folder once you verify the update works.
timeout /t 3 /nobreak >nul
exit
"""
            else:
                # Mac/Linux shell script with rsync for reliability
                script_path = temp_dir / "voxtral_updater.sh"
                log_path = temp_dir / "voxtral_updater.log"
                failed_marker = install_root / ".UPDATE_FAILED"
                launcher_script = install_root / "VoxtralApp" / "start_web.sh"
                pid = os.getpid()

                script_content = f"""#!/bin/bash
LOG_FILE="{log_path}"
FAILED_MARKER="{failed_marker}"
BACKUP_DIR="{backup_dir}"
INSTALL_ROOT="{install_root}"
NEW_DIR="{new_install_dir}"
LAUNCHER="{launcher_script}"

echo "Voxtral Update Script" > "$LOG_FILE"
echo "=====================" >> "$LOG_FILE"
echo "Started: $(date)" >> "$LOG_FILE"
echo "PID: {pid}" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

echo "Voxtral Update Script"
echo "Log: $LOG_FILE"

# Wait for Flask process to exit (up to 120 seconds)
echo "Waiting for application (PID {pid})..."
echo "Waiting for Flask {pid}..." >> "$LOG_FILE"

WAIT_COUNT=0
while kill -0 {pid} 2>/dev/null; do
    WAIT_COUNT=$((WAIT_COUNT + 1))
    if [ $WAIT_COUNT -ge 120 ]; then
        echo "WARNING: Still waiting after 120s" >> "$LOG_FILE"
        break
    fi
    sleep 1
done

echo "Process exited after ${{WAIT_COUNT}}s" >> "$LOG_FILE"
echo "Application closed"

# Wait additional 3 seconds for file locks to release
echo "Waiting for file locks to release..." >> "$LOG_FILE"
sleep 3

# Clean old backup
[ -d "$BACKUP_DIR" ] && rm -rf "$BACKUP_DIR" 2>>"$LOG_FILE"

# Backup using rsync (better than mv for reliability)
echo "Backing up..."
mkdir -p "$BACKUP_DIR"
rsync -a --delete "$INSTALL_ROOT/" "$BACKUP_DIR/" >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: Backup failed" >> "$LOG_FILE"
    echo "Update failed: Backup failed" > "$FAILED_MARKER"
    echo "Log: $LOG_FILE" >> "$FAILED_MARKER"
    echo "ERROR: Backup failed"
    read -p "Press Enter to exit..."
    exit 1
fi
echo "Backup complete" >> "$LOG_FILE"

# Delete current installation with retry logic
echo "Removing current..."
RETRY_COUNT=0
while [ $RETRY_COUNT -lt 5 ]; do
    rm -rf "$INSTALL_ROOT" 2>>"$LOG_FILE"
    if [ ! -d "$INSTALL_ROOT" ]; then
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Retry $RETRY_COUNT/5: Waiting for locks..." >> "$LOG_FILE"
    sleep 2
done

if [ -d "$INSTALL_ROOT" ]; then
    echo "ERROR: Could not delete after 5 retries" >> "$LOG_FILE"
    {{
        echo "Update failed: Could not delete current installation"
        echo ""
        echo "File locks prevented deletion."
        echo "Your new version is ready in: $BACKUP_DIR"
        echo ""
        echo "TO RECOVER:"
        echo "1. Close all applications using Voxtral"
        echo "2. Delete this directory: $INSTALL_ROOT"
        echo "3. Rename to: $INSTALL_ROOT"
        echo "   From: $BACKUP_DIR"
        echo "4. Run: ./start_web.sh"
        echo ""
        echo "Log: $LOG_FILE"
    }} > "$FAILED_MARKER"
    echo ""
    echo "ERROR: Could not delete current installation"
    echo "Please see .UPDATE_FAILED file for recovery instructions."
    read -p "Press Enter to exit..."
    exit 1
fi

# Install new version using rsync
echo "Installing new..."
mkdir -p "$INSTALL_ROOT"
rsync -a --delete "$NEW_DIR/" "$INSTALL_ROOT/" >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: Install failed, restoring..." >> "$LOG_FILE"
    rm -rf "$INSTALL_ROOT" 2>>"$LOG_FILE"
    rsync -a --delete "$BACKUP_DIR/" "$INSTALL_ROOT/" >> "$LOG_FILE" 2>&1
    echo "Update failed: Installation failed" > "$FAILED_MARKER"
    echo "Log: $LOG_FILE" >> "$FAILED_MARKER"
    echo "ERROR: Install failed"
    read -p "Press Enter to exit..."
    exit 1
fi
echo "Install complete" >> "$LOG_FILE"

# Clean temp
rm -rf "$NEW_DIR" 2>>"$LOG_FILE"

# Restart via launcher script
echo "Restarting..."
echo "Starting $LAUNCHER..." >> "$LOG_FILE"

if [ -f "$LAUNCHER" ]; then
    cd "$INSTALL_ROOT/VoxtralApp"
    bash "$LAUNCHER" &
else
    echo "WARNING: Launcher not found" >> "$LOG_FILE"
    cd "$INSTALL_ROOT/VoxtralApp"
    ./voxtral_env/bin/python app.py &
fi

echo "Completed: $(date)" >> "$LOG_FILE"
echo "Update successful!"
sleep 2
exit 0
"""

            # Write script
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script_content)

            # Make executable on Unix
            if sys.platform != "win32":
                script_path.chmod(0o755)

            script_duration = time.time() - script_start
            logger.info(f"[Stage 5: Script] Complete in {script_duration:.1f}s")
            logger.info(f"  Script created: {script_path}")
            logger.info(f"  Log will be written to: {log_path}")

            # Log the full script content for debugging
            logger.info("=== BEGIN UPDATER SCRIPT ===")
            script_lines = script_content.split("\n")
            for i, line in enumerate(script_lines[:50], 1):  # Log first 50 lines
                logger.info(f"  {i:3d}: {line}")
            if len(script_lines) > 50:
                remaining = len(script_lines) - 50
                logger.info(f"  ... ({remaining} more lines)")
            logger.info("=== END UPDATER SCRIPT ===")

            emit_progress("launching", "Launching updater...", 90)
            logger.info(f"[Stage 6: Launch] Launching external updater script")
            logger.info(f"  Script: {script_path}")
            logger.info(f"  Process will exit, updater will take over")

            # Launch updater script
            if sys.platform == "win32":
                subprocess.Popen(["cmd", "/c", str(script_path)], creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                subprocess.Popen(["/bin/bash", str(script_path)])

            total_duration = time.time() - stage_start_time
            logger.info(f"=== Update preparation complete in {total_duration:.1f}s ===")
            logger.info(f"  Download: {download_duration:.1f}s")
            logger.info(f"  Extract: {extract_duration:.1f}s")
            logger.info(f"  Config merge: {config_duration:.1f}s")
            logger.info(f"  Data migration: {data_duration:.1f}s")
            logger.info(f"  Script creation: {script_duration:.1f}s")
            logger.info(f"External updater launched, this process will now exit in 2 seconds...")

            emit_progress("complete", "Update ready! Restarting...", 100)

            # Give the response time to send, then exit
            def exit_for_update():
                time.sleep(2)  # Increased from 1 to 2 seconds
                logger.info("Exiting for update (PID: {os.getpid()})...")
                os._exit(0)

            threading.Thread(target=exit_for_update, daemon=True).start()

            return (
                True,
                f"Update to {latest_version} prepared. Application restarting...",
                {"backup_path": str(backup_dir), "script_path": str(script_path), "log_path": str(log_path)},
            )

        except PermissionError as e:
            # Specific handling for permission errors
            logger.error(f"[Permission Error] {e}", exc_info=True)
            logger.error(f"  Failed path: {e.filename if hasattr(e, 'filename') else 'unknown'}")
            logger.error(f"  Process ID: {os.getpid()}, CWD: {os.getcwd()}")

            # Try to list open files
            try:
                process = psutil.Process()
                open_files = process.open_files()
                logger.error(f"  Open files ({len(open_files)}):")
                for f in open_files[:10]:  # Limit to first 10
                    logger.error(f"    - {f.path}")
            except Exception:
                pass

            return (
                False,
                f"Permission denied: Windows cannot move files while they are in use. Please close the application and run the update script manually.",
                {},
            )

        except Exception as e:
            # General error handling
            logger.exception(f"[Update Failed] {e}")
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed: {e}")
        return False, f"Download failed: {str(e)}", {}
    except zipfile.BadZipFile as e:
        logger.error(f"Invalid ZIP archive: {e}")
        return False, "Downloaded file is not a valid ZIP archive", {}
    except Exception as e:
        logger.exception(f"ZIP update failed: {e}")
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


def download_gguf_model(model_config, progress_callback=None):
    """
    Download a GGUF model from HuggingFace if not already cached.

    Args:
        model_config: Model configuration dict with gguf_filename
        progress_callback: Optional callback for progress updates

    Returns:
        Path to the downloaded model file
    """
    gguf_config = config.get("gguf", {})
    models_dir = Path(os.path.expanduser(gguf_config.get("models_dir", "~/.cache/whisper-gguf")))
    models_dir.mkdir(parents=True, exist_ok=True)

    filename = model_config.get("gguf_filename")
    if not filename:
        raise ValueError("GGUF model config missing 'gguf_filename'")

    model_path = models_dir / filename

    # Check if model already exists
    if model_path.exists():
        logger.info(f"GGUF model already cached: {model_path}")
        return str(model_path)

    # Download from HuggingFace
    download_base = gguf_config.get("download_url_base", "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/")
    url = f"{download_base}{filename}"

    logger.info(f"Downloading GGUF model from: {url}")
    if progress_callback:
        progress_callback({"status": "downloading", "message": f"Downloading {filename}...", "progress": 10})

    response = requests.get(url, stream=True, timeout=600)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(model_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0 and progress_callback:
                    progress = 10 + int((downloaded / total_size) * 40)  # 10-50%
                    progress_callback(
                        {
                            "status": "downloading",
                            "message": f"Downloading {filename}... {downloaded // (1024 * 1024)}MB",
                            "progress": progress,
                        }
                    )

    logger.info(f"GGUF model downloaded to: {model_path}")
    return str(model_path)


def initialize_engine(model_version=None):
    """
    Initialize the transcription engine with specified model.

    Args:
        model_version: Model version to load (e.g., 'full', 'quantized', 'whisper-base-gguf').
                      Uses config default if None.
    """
    global transcription_engine, engine_loading

    # Skip model loading in test mode (saves memory and time in CI/CD)
    if os.environ.get("TESTING") == "1":
        logger.info("Test mode detected - skipping model initialization")
        from unittest.mock import MagicMock

        transcription_engine = MagicMock()
        transcription_engine.device = "cpu"
        transcription_engine.backend = "mock"
        # Return proper dict structure instead of None to prevent TypeError
        transcription_engine.transcribe_file = MagicMock(
            return_value={"status": "success", "text": "Mock transcription", "language": "en"}
        )
        transcription_engine.get_device_info = MagicMock(
            return_value={"device": "cpu", "backend": "mock"}
        )
        engine_loading = False
        return

    try:
        # Note: engine_loading is already set to True before thread start in initialize_model()
        # This line is kept for direct calls to initialize_engine() outside of the API route
        engine_loading = True

        # Get model configuration
        if model_version:
            model_config = config.get_model_config(model_version)
        else:
            model_config = config.get_model_config()

        model_id = model_config.get("id")
        model_name = model_config.get("name", "Unknown")
        quantization = model_config.get("quantization")
        backend = model_config.get("backend", "voxtral")

        logger.info(f"Initializing transcription engine with model: {model_name} ({model_id})")
        logger.info(f"Backend: {backend}")
        if quantization:
            logger.info(f"Using {quantization} quantization")

        # Emit loading progress via WebSocket
        socketio.emit("model_loading", {"status": "loading", "model": model_name, "message": f"Loading {model_name}..."})

        # Handle GGUF backend
        if backend == "gguf":
            # Import GGUF backend constants
            from transcription_engine import BACKEND_GGUF

            # Define progress callback for download
            def download_progress(data):
                socketio.emit("model_loading", {"status": "loading", "model": model_name, **data})

            # Download model if needed
            gguf_model_path = download_gguf_model(model_config, progress_callback=download_progress)

            socketio.emit(
                "model_loading", {"status": "loading", "model": model_name, "message": "Initializing Whisper model..."}
            )

            transcription_engine = TranscriptionEngine(
                model_id=model_id,
                backend=BACKEND_GGUF,
                gguf_model_path=gguf_model_path,
            )
        else:
            # Voxtral backend (default)
            transcription_engine = TranscriptionEngine(model_id=model_id, quantization=quantization)

        # Save the selected model version to config
        if model_version:
            config.set_model_version(model_version)

        socketio.emit(
            "model_loading", {"status": "loaded", "model": model_name, "message": f"{model_name} loaded successfully!"}
        )

        logger.info(f"Engine initialized successfully with {model_name} (backend: {backend})")
        engine_loading = False

    except Exception as e:
        logger.error(f"Failed to initialize engine: {e}")
        engine_loading = False
        socketio.emit("model_loading", {"status": "error", "message": f"Failed to load model: {str(e)}"})
        raise


if __name__ == "__main__":
    # DON'T load the model on startup - let users select it first!
    # Model will be loaded after user selection in the web UI

    # Check for failed update marker
    install_root = BASE_DIR.parent
    failed_marker = install_root / ".UPDATE_FAILED"
    if failed_marker.exists():
        try:
            error_msg = failed_marker.read_text(encoding="utf-8")
            logger.error("=" * 80)
            logger.error("UPDATE FAILED - Previous update attempt encountered an error:")
            logger.error("-" * 80)
            for line in error_msg.strip().split("\n"):
                logger.error(f"  {line}")
            logger.error("-" * 80)
            logger.error("Please check the log file for details and try updating again.")
            logger.error("If the problem persists, please report it at:")
            logger.error("  https://github.com/debrockb/transcribe-voxtral/issues")
            logger.error("=" * 80)
            # Remove marker file after displaying (so we don't spam on every startup)
            failed_marker.unlink()
        except Exception as e:
            logger.warning(f"Failed to read update error marker: {e}")

    # Start memory monitoring
    start_memory_monitor()

    # Start Flask-SocketIO server
    # Using port 8000 as port 5000 is often used by macOS AirPlay Receiver
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting Voxtral Web Application on port {port}...")
    logger.info(f"Access the application at: http://localhost:{port}")
    logger.info("Model will be loaded after user selection in the web UI")

    # SECURITY: Bind to 127.0.0.1 (localhost) instead of 0.0.0.0 to prevent network access
    # This ensures only the local machine can access the server, preventing:
    # - Remote code execution via /api/updates/install
    # - Data exfiltration via /api/history endpoints
    # - Unauthorized access to transcripts and uploads
    socketio.run(app, host="127.0.0.1", port=port, debug=True, allow_unsafe_werkzeug=True)
