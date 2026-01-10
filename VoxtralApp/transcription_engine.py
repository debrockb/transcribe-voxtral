"""
Voxtral Transcription Engine
Refactored version of transcribe_voxtral.py for web application use
Supports progress callbacks and is designed for cross-platform compatibility (Windows & macOS)
Features: Automatic language detection, FFmpeg conversion, audio normalization
Supports multiple backends: Voxtral (HuggingFace) and GGUF (whisper.cpp)
"""

import gc
import logging
import os
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Callable, Dict, Literal, Optional, Union

# --- CRITICAL COMPATIBILITY FIX (MUST BE BEFORE TORCHAUDIO/SPEECHBRAIN IMPORT) ---
# We patch torchaudio FIRST, so when SpeechBrain loads, it sees the function exists.
import torchaudio

if not hasattr(torchaudio, "list_audio_backends"):

    def _list_audio_backends():
        return ["soundfile"]

    torchaudio.list_audio_backends = _list_audio_backends
# ----------------------------------------------------------------------

import librosa
import numpy as np
import psutil
import soundfile as sf
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, VoxtralForConditionalGeneration

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Enable debug for language detection diagnostics

# --- CONFIGURATION ---
SUPPORTED_LANGS = {"en", "fr", "de", "es", "it", "nl", "pt", "hi", "pl", "ru", "ja", "ko", "zh", "ar"}
DEFAULT_CHUNK_DURATION_S = 90  # 90s is optimal for multi-lingual switching
GGUF_CHUNK_DURATION_S = 30  # 30s is optimal for whisper.cpp

# Backend types
BACKEND_VOXTRAL = "voxtral"
BACKEND_GGUF = "gguf"


def convert_to_clean_wav(input_path: str, output_path: str) -> bool:
    """
    Uses FFmpeg to convert complex MP4/Teams audio to a clean,
    simple 16kHz Mono WAV file that Python can't fail on.

    Args:
        input_path: Path to input audio/video file
        output_path: Path for output WAV file

    Returns:
        True if conversion successful, False otherwise
    """
    try:
        command = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-i",
            input_path,  # Input
            "-ar",
            "16000",  # 16kHz Sample Rate
            "-ac",
            "1",  # Mono
            "-c:a",
            "pcm_s16le",  # Standard WAV PCM encoding
            "-vn",  # No Video
            "-loglevel",
            "error",  # Quiet mode
            output_path,
        ]
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e}")
        return False
    except FileNotFoundError:
        logger.error("FFmpeg not found. Please install FFmpeg (brew install ffmpeg / choco install ffmpeg)")
        return False


class TranscriptionEngine:
    """
    Engine for transcribing audio files using multiple backends:
    - Voxtral: Mistral AI Voxtral model via HuggingFace transformers
    - GGUF: Whisper models via whisper.cpp (pywhispercpp)

    Designed for cross-platform compatibility and real-time progress updates.
    Features: Automatic language detection, FFmpeg conversion, audio normalization.
    """

    def __init__(
        self,
        model_id: str = "mistralai/Voxtral-Mini-3B-2507",
        device: Optional[str] = None,
        progress_callback: Optional[Callable[[Dict], None]] = None,
        quantization: Optional[str] = None,
        auto_detect_language: bool = True,
        backend: str = BACKEND_VOXTRAL,
        gguf_model_path: Optional[str] = None,
    ):
        """
        Initialize the transcription engine.

        Args:
            model_id: HuggingFace model identifier (for Voxtral backend)
            device: Device to use (mps/cuda/cpu). Auto-detects if None.
            progress_callback: Function to call with progress updates
            quantization: Quantization mode ('4bit', '8bit', or None for full precision)
            auto_detect_language: Enable automatic language detection per chunk (requires speechbrain)
            backend: Backend to use ('voxtral' or 'gguf')
            gguf_model_path: Path to GGUF model file (required for GGUF backend)
        """
        self.model_id = model_id
        self.progress_callback = progress_callback
        self.quantization = quantization
        self.auto_detect_language = auto_detect_language
        self.backend = backend
        self.gguf_model_path = gguf_model_path

        # GGUF backend uses its own instance
        self.gguf_backend = None

        if backend == BACKEND_GGUF:
            logger.info(f"Initializing GGUF backend with model: {gguf_model_path}")
            self._load_gguf_backend()
            self.device = "cpu"  # GGUF uses CPU (with potential Metal/CUDA acceleration internally)
            self.dtype = None
            self.processor = None
            self.model = None
            self.lang_classifier = None
        else:
            # Voxtral backend
            self.device = device or self._detect_device()
            # IMPORTANT: MPS requires float32, only CUDA supports bfloat16
            if self.device == "mps":
                self.dtype = torch.float32
            elif self.device == "cuda":
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float32

            logger.info(f"Initializing TranscriptionEngine on device: {self.device}")
            if quantization:
                logger.info(f"Using {quantization} quantization")

            # Load model and processor
            self.processor = None
            self.model = None
            self.lang_classifier = None
            self._load_model()
            self._load_language_classifier()

    def _load_gguf_backend(self):
        """Load the GGUF Whisper backend."""
        if not self.gguf_model_path:
            raise ValueError("gguf_model_path is required for GGUF backend")

        try:
            from gguf_backend import GGUFWhisperBackend

            self.gguf_backend = GGUFWhisperBackend(
                model_path=self.gguf_model_path,
                progress_callback=self.progress_callback,
            )
        except ImportError as e:
            raise ImportError(
                f"GGUF backend dependencies not available: {e}. "
                "Install with: pip install pywhispercpp"
            )

    def _detect_device(self) -> str:
        """Auto-detect the best available device (MPS/CUDA/CPU)."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def _load_model(self):
        """Load the Voxtral model and processor."""
        try:
            self._emit_progress({"status": "loading_model", "message": f"Loading model '{self.model_id}'...", "progress": 0})

            logger.info(f"Loading model: {self.model_id}")
            self.processor = AutoProcessor.from_pretrained(self.model_id)

            # Prepare model loading arguments
            model_kwargs = {
                "torch_dtype": self.dtype,
            }

            # Add quantization config if requested
            # Note: bitsandbytes quantization only works with CUDA
            if self.quantization and self.device != "cuda":
                logger.warning(
                    f"Quantization requested but not supported on {self.device.upper()}. "
                    f"bitsandbytes quantization requires CUDA (NVIDIA GPU). "
                    f"Loading full precision model instead."
                )

                # Warn about slow CPU loading
                if self.device == "cpu":
                    logger.warning(
                        "Loading large model on CPU will be VERY slow (10-30 minutes). "
                        "For better performance, use a system with NVIDIA GPU."
                    )
                    self._emit_progress(
                        {
                            "status": "warning",
                            "message": "CPU detected - Model loading will take 10-30 minutes. Please be patient...",
                        }
                    )
                else:
                    self._emit_progress(
                        {
                            "status": "warning",
                            "message": f"Quantization not supported on {self.device.upper()}, using full precision",
                        }
                    )

                self.quantization = None  # Disable quantization
                model_kwargs["device_map"] = self.device
            elif self.quantization == "4bit":
                logger.info("Configuring 4-bit quantization")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"  # Required for quantization
            elif self.quantization == "8bit":
                logger.info("Configuring 8-bit quantization")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"  # Required for quantization
            else:
                # No quantization - use specified device
                model_kwargs["device_map"] = self.device

            self.model = VoxtralForConditionalGeneration.from_pretrained(self.model_id, **model_kwargs)

            self._emit_progress(
                {
                    "status": "model_loaded",
                    "message": "Model loaded successfully",
                    "progress": 0,
                    "device": self.device.upper(),
                    "quantization": self.quantization or "none",
                }
            )

            logger.info("Model and processor loaded successfully")

        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            logger.error(error_msg)
            self._emit_progress({"status": "error", "message": error_msg, "error": str(e)})
            raise

    def _load_language_classifier(self):
        """Load the SpeechBrain language classifier for automatic language detection."""
        if not self.auto_detect_language:
            logger.info("Automatic language detection disabled")
            return

        try:
            self._emit_progress({"status": "loading_classifier", "message": "Loading language detection model..."})

            # Import SpeechBrain here to avoid import errors if not installed
            from speechbrain.inference.classifiers import EncoderClassifier

            self.lang_classifier = EncoderClassifier.from_hparams(
                source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmpdir_lang_id"
            )

            logger.info("SpeechBrain language classifier loaded successfully")
            self._emit_progress({"status": "classifier_loaded", "message": "Language detection model loaded"})

        except ImportError:
            logger.warning(
                "SpeechBrain not installed. Automatic language detection disabled. " "Install with: pip install speechbrain"
            )
            self.auto_detect_language = False
            self.lang_classifier = None
        except Exception as e:
            logger.warning(f"Failed to load language classifier: {e}. Using manual language selection.")
            self.auto_detect_language = False
            self.lang_classifier = None

    def _detect_chunk_language(self, audio_path: str, last_detected_lang: str = "en") -> str:
        """
        Detect the language of an audio chunk using SpeechBrain classifier.

        Args:
            audio_path: Path to the audio chunk file
            last_detected_lang: Fallback language if detection fails

        Returns:
            Detected language code (e.g., 'en', 'fr', 'de')
        """
        if not self.lang_classifier:
            return last_detected_lang

        waveform = None
        resampler = None
        try:
            # Use soundfile directly to avoid torchaudio's torchcodec requirement
            audio_np, sr = sf.read(audio_path)
            waveform = torch.from_numpy(audio_np).float()
            # Ensure correct shape: (channels, samples)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.dim() == 2 and waveform.shape[1] < waveform.shape[0]:
                # soundfile returns (samples, channels), we need (channels, samples)
                waveform = waveform.T

            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
                del resampler
                resampler = None

            # Check if audio is too quiet (likely silence)
            if waveform.abs().mean() < 0.001:
                del waveform
                return last_detected_lang

            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Run classifier with no_grad to prevent gradient accumulation
            with torch.no_grad():
                _, _, _, text_lab = self.lang_classifier.classify_batch(waveform)

            # Immediately free the waveform
            del waveform
            waveform = None

            # Debug: log the raw output from classifier
            raw_label = text_lab[0] if text_lab else "empty"
            logger.debug(f"SpeechBrain raw output: '{raw_label}'")

            # Parse the language code - format is typically "fr: French" or just "fr"
            if ":" in raw_label:
                detected_code = raw_label.split(":")[0].strip().lower()
            else:
                detected_code = raw_label.strip().lower()

            logger.debug(f"Parsed language code: '{detected_code}'")

            # Only return if it's a supported language
            if detected_code not in SUPPORTED_LANGS:
                logger.debug(f"Language '{detected_code}' not in SUPPORTED_LANGS, using fallback '{last_detected_lang}'")
                return last_detected_lang

            return detected_code

        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return last_detected_lang

        finally:
            # Ensure cleanup even on exception
            if waveform is not None:
                del waveform
            if resampler is not None:
                del resampler
            # Force garbage collection and clear device caches
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _emit_progress(self, data: Dict):
        """Emit progress update via callback if available."""
        if self.progress_callback:
            try:
                self.progress_callback(data)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    def _get_optimal_chunk_duration(self, default_duration: int = 120) -> int:
        """
        Calculate optimal chunk duration based on available system memory.

        Args:
            default_duration: Default chunk duration in seconds (2 minutes)

        Returns:
            Adjusted chunk duration in seconds
        """
        try:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)

            # If memory is critically low (< 2GB available), use shorter chunks
            if available_gb < 2:
                logger.warning(f"Low memory detected ({available_gb:.2f}GB available), using 60s chunks")
                return 60  # 1 minute chunks
            # If memory is low (< 4GB available), use medium chunks
            elif available_gb < 4:
                logger.info(f"Limited memory detected ({available_gb:.2f}GB available), using 90s chunks")
                return 90  # 1.5 minute chunks
            else:
                # Normal memory available, use default
                return default_duration

        except Exception as e:
            logger.warning(f"Could not detect memory, using default chunk duration: {e}")
            return default_duration

    def _cleanup_memory(self):
        """Force garbage collection and clear device caches."""
        # Force Python garbage collection
        gc.collect()

        # Clear device-specific caches
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

    def _transcribe_chunk(self, temp_chunk_path: str, language: str) -> Dict:
        """
        Transcribe a single audio chunk and calculate confidence score.

        Args:
            temp_chunk_path: Path to temporary chunk file
            language: Language code (e.g., 'en', 'fr', 'es')

        Returns:
            Dictionary with 'text' and 'confidence' keys
        """
        inputs = None
        outputs = None
        scores = None
        try:
            inputs = self.processor.apply_transcription_request(
                language=language, model_id=self.model_id, audio=temp_chunk_path, return_tensors="pt"
            )

            inputs = inputs.to(self.device, dtype=self.dtype)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    num_beams=1,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            # Extract sequences and scores
            sequences = outputs.sequences
            scores = outputs.scores  # Tuple of logits for each generated token

            transcription = self.processor.batch_decode(sequences, skip_special_tokens=True)[0]

            # Calculate confidence score from token probabilities
            confidence = self._calculate_confidence(scores)

            # Clean up language tags that may appear in output
            for lang in SUPPORTED_LANGS:
                transcription = transcription.replace(f"lang:{lang}", "")

            return {"text": transcription.strip(), "confidence": confidence}

        finally:
            # Aggressive cleanup of tensors
            if inputs is not None:
                del inputs
            if outputs is not None:
                del outputs
            if scores is not None:
                del scores
            # Clear device caches immediately after model inference
            gc.collect()
            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()

    def _calculate_confidence(self, scores: tuple) -> float:
        """
        Calculate confidence score from model output scores.

        Uses the average probability of the selected tokens across all generation steps.

        Args:
            scores: Tuple of logits tensors, one per generated token

        Returns:
            Confidence score as percentage (0-100)
        """
        if not scores or len(scores) == 0:
            return 0.0

        try:
            confidences = []
            for step_scores in scores:
                # Get probabilities via softmax
                probs = torch.nn.functional.softmax(step_scores, dim=-1)
                # Get the max probability (the chosen token's probability)
                max_prob = probs.max(dim=-1).values
                confidences.append(max_prob.item())

            if not confidences:
                return 0.0

            # Average confidence across all tokens, scaled to percentage
            avg_confidence = sum(confidences) / len(confidences) * 100
            return round(avg_confidence, 1)

        except Exception as e:
            logger.warning(f"Error calculating confidence: {e}")
            return 0.0

    def transcribe_file(
        self,
        input_audio_path: str,
        output_text_path: str,
        language: str = "en",
        chunk_duration_s: int = DEFAULT_CHUNK_DURATION_S,
    ) -> Dict:
        """
        Transcribe a complete audio file with chunking.

        Features:
        - FFmpeg pre-conversion for complex audio/video formats
        - Audio normalization for better recognition
        - Automatic language detection per chunk (if enabled)
        - Supports multiple backends (Voxtral, GGUF)

        Args:
            input_audio_path: Path to input audio file
            output_text_path: Path to save transcription
            language: Language code for transcription (used if auto-detection disabled)
            chunk_duration_s: Duration of each chunk in seconds (default: 90s)

        Returns:
            Dictionary with transcription results and metadata
        """
        # Delegate to GGUF backend if using that backend
        if self.backend == BACKEND_GGUF and self.gguf_backend:
            return self.gguf_backend.transcribe_file(
                input_audio_path,
                output_text_path,
                language,
                chunk_duration_s=GGUF_CHUNK_DURATION_S,
            )

        temp_clean_wav = None
        try:
            # Convert paths to Path objects for cross-platform compatibility
            input_path = Path(input_audio_path)
            output_path = Path(output_text_path)

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            sample_rate = 16000

            self._emit_progress({"status": "converting", "message": f"Converting audio: {input_path.name}", "progress": 0})

            # Step 1: Convert to clean WAV using FFmpeg (handles complex formats better)
            logger.info(f"Converting audio file: {input_audio_path}")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_clean_wav = tmp.name

            if convert_to_clean_wav(str(input_path), temp_clean_wav):
                logger.info("FFmpeg conversion successful, loading clean WAV")
                audio_to_load = temp_clean_wav
            else:
                logger.warning("FFmpeg conversion failed, loading original file directly")
                audio_to_load = str(input_path)

            self._emit_progress({"status": "loading_audio", "message": f"Loading audio: {input_path.name}", "progress": 5})

            # Step 2: Load audio into RAM
            logger.info(f"Loading audio file: {audio_to_load}")
            waveform, _ = librosa.load(audio_to_load, sr=sample_rate, mono=True)
            total_duration_s = len(waveform) / sample_rate

            logger.info(f"Audio loaded. Duration: {total_duration_s / 60:.2f} minutes")

            # Adjust chunk duration based on available memory
            optimal_chunk_duration = self._get_optimal_chunk_duration(chunk_duration_s)
            if optimal_chunk_duration != chunk_duration_s:
                logger.info(
                    f"Chunk duration adjusted from {chunk_duration_s}s to {optimal_chunk_duration}s for memory optimization"
                )
            chunk_len = optimal_chunk_duration * sample_rate
            all_transcriptions = []
            chunk_confidences = []  # Track confidence scores per chunk
            num_chunks = int(np.ceil(len(waveform) / chunk_len))

            self._emit_progress(
                {
                    "status": "processing",
                    "message": f"Processing audio in {num_chunks} chunks",
                    "progress": 10,
                    "total_chunks": num_chunks,
                    "current_chunk": 0,
                }
            )

            logger.info(f"Processing audio in {num_chunks} chunks...")
            current_lang = language  # Track current language for multi-lingual files

            # Log initial memory state
            mem_info = psutil.virtual_memory()
            logger.info(
                f"Memory before processing: {mem_info.used / (1024**3):.2f}GB used / {mem_info.total / (1024**3):.2f}GB total"
            )

            for i, start in enumerate(range(0, len(waveform), chunk_len)):
                end = start + chunk_len
                chunk_waveform = waveform[start:end]

                # Skip very short chunks (less than 1 second)
                if len(chunk_waveform) < sample_rate:
                    continue

                # Step 3: Normalize audio for better recognition
                chunk_waveform = librosa.util.normalize(chunk_waveform)

                # Create temporary file (cross-platform)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    temp_chunk_path = tmp.name
                    sf.write(temp_chunk_path, chunk_waveform, sample_rate)

                # CRITICAL: Free chunk_waveform immediately after writing to disk
                del chunk_waveform
                chunk_waveform = None

                try:
                    chunk_num = i + 1
                    # Progress from 10% to 95% during chunk processing
                    progress_pct = 10 + int((chunk_num / num_chunks) * 85)

                    # Step 4: Detect language for this chunk (if auto-detection enabled)
                    if self.auto_detect_language and self.lang_classifier:
                        detected_lang = self._detect_chunk_language(temp_chunk_path, current_lang)
                        if detected_lang != current_lang:
                            logger.info(f"Language switch detected: {current_lang.upper()} -> {detected_lang.upper()}")
                            current_lang = detected_lang
                    else:
                        detected_lang = current_lang

                    self._emit_progress(
                        {
                            "status": "processing",
                            "message": f"Transcribing chunk {chunk_num}/{num_chunks} [{detected_lang.upper()}]",
                            "progress": progress_pct,
                            "total_chunks": num_chunks,
                            "current_chunk": chunk_num,
                            "detected_language": detected_lang,
                        }
                    )

                    logger.info(f"Transcribing chunk {chunk_num}/{num_chunks} [{detected_lang.upper()}]")

                    # Step 5: Transcribe with detected language
                    chunk_result = self._transcribe_chunk(temp_chunk_path, detected_lang)

                    chunk_text = chunk_result.get("text", "")
                    chunk_confidence = chunk_result.get("confidence", 0.0)

                    if chunk_text:
                        all_transcriptions.append(chunk_text)
                        chunk_confidences.append(chunk_confidence)
                        preview = chunk_text[:50].replace("\n", " ")
                        logger.info(f'Chunk {chunk_num} completed (confidence: {chunk_confidence:.1f}%): "{preview}..."')

                        # Emit chunk completion with confidence
                        self._emit_progress(
                            {
                                "status": "chunk_complete",
                                "message": f"Chunk {chunk_num}/{num_chunks} complete",
                                "progress": progress_pct,
                                "current_chunk": chunk_num,
                                "total_chunks": num_chunks,
                                "chunk_confidence": chunk_confidence,
                            }
                        )
                    else:
                        logger.info(f"Chunk {chunk_num}: [Silence/No text]")

                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_chunk_path):
                        os.remove(temp_chunk_path)

                    # Clean up memory after each chunk to prevent accumulation
                    self._cleanup_memory()

                    # Log memory usage after each chunk
                    mem_info = psutil.virtual_memory()
                    logger.info(f"Memory after chunk {i + 1}: {mem_info.used / (1024**3):.2f}GB used ({mem_info.percent}%)")

            # Clean up full waveform to free memory
            del waveform
            self._cleanup_memory()

            # Combine all transcriptions
            self._emit_progress({"status": "finalizing", "message": "Combining transcriptions...", "progress": 95})

            logger.info("Combining transcriptions...")
            final_transcription = " ".join(all_transcriptions).strip()

            # Calculate overall confidence score (weighted average based on chunk text length)
            overall_confidence = 0.0
            if chunk_confidences:
                # Weight confidence by the length of each chunk's transcription
                total_weight = sum(len(t) for t in all_transcriptions)
                if total_weight > 0:
                    weighted_sum = sum(
                        conf * len(text) for conf, text in zip(chunk_confidences, all_transcriptions)
                    )
                    overall_confidence = round(weighted_sum / total_weight, 1)
                else:
                    overall_confidence = round(sum(chunk_confidences) / len(chunk_confidences), 1)

            logger.info(f"Overall transcription confidence: {overall_confidence:.1f}%")

            # Save to file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_transcription)

            self._emit_progress(
                {
                    "status": "complete",
                    "message": "Transcription complete",
                    "progress": 100,
                    "transcript": final_transcription,
                    "output_path": str(output_path),
                    "confidence": overall_confidence,
                }
            )

            logger.info(f"Transcription saved to: {output_path}")

            return {
                "status": "success",
                "transcript": final_transcription,
                "output_path": str(output_path),
                "duration_seconds": total_duration_s,
                "duration_minutes": total_duration_s / 60,
                "chunks_processed": num_chunks,
                "language": language,
                "word_count": len(final_transcription.split()),
                "char_count": len(final_transcription),
                "confidence": overall_confidence,
            }

        except Exception as e:
            error_msg = f"Error transcribing {input_audio_path}: {str(e)}"
            logger.error(error_msg)

            self._emit_progress({"status": "error", "message": error_msg, "error": str(e)})

            return {"status": "error", "error": str(e), "message": error_msg}

        finally:
            # Clean up temporary WAV file created by FFmpeg
            if temp_clean_wav and os.path.exists(temp_clean_wav):
                try:
                    os.remove(temp_clean_wav)
                    logger.debug(f"Cleaned up temporary WAV: {temp_clean_wav}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file {temp_clean_wav}: {e}")

    def get_device_info(self) -> Dict:
        """Get information about the current device and capabilities."""
        # Delegate to GGUF backend if using that backend
        if self.backend == BACKEND_GGUF and self.gguf_backend:
            info = self.gguf_backend.get_device_info()
            info["backend"] = BACKEND_GGUF
            return info

        return {
            "device": self.device,
            "device_name": self.device.upper(),
            "dtype": str(self.dtype),
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "auto_detect_language": self.auto_detect_language,
            "lang_classifier_loaded": self.lang_classifier is not None,
            "backend": BACKEND_VOXTRAL,
        }


# Standalone function for backward compatibility with original script
def transcribe_audio_file(
    input_path: str, output_path: str, language: str = "en", progress_callback: Optional[Callable[[Dict], None]] = None
) -> Dict:
    """
    Convenience function to transcribe an audio file.

    Args:
        input_path: Path to input audio file
        output_path: Path to save transcription
        language: Language code (e.g., 'en', 'fr', 'es')
        progress_callback: Optional callback for progress updates

    Returns:
        Dictionary with transcription results
    """
    engine = TranscriptionEngine(progress_callback=progress_callback)
    return engine.transcribe_file(input_path, output_path, language)
