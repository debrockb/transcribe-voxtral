# Transcribe Voxtral Python Script

A Python tool that transcribes audio files using the Mistral AI Voxtral-Mini-3B model. This script leverages local AI processing for privacy-focused, cost-free audio transcription.

![Transcribe Voxtral hero banner](assets/voxtral_banner.svg)

## Features

- Transcribes audio files (WAV, MP3, FLAC, M4A) to text
- Processes large files in 2-minute chunks to manage memory efficiently
- Automatic device detection (MPS for Apple Silicon, CUDA for NVIDIA GPUs, or CPU)
- Batch processing of multiple audio files in one run
- No API costs - runs entirely on local hardware
- Multilingual support (30+ languages)

## Requirements

- Python 3.11 or later
- macOS, Linux, or Windows (optimized for Apple Silicon with MPS support)
- 20GB+ disk space for initial model download
- 8GB+ RAM recommended (16GB+ for optimal performance)

## Setup

### Quick Setup (macOS)

```bash
chmod +x setup.sh
./setup.sh
```

### Manual Setup (All Platforms)

```bash
python3.11 -m venv voxtral_env
source voxtral_env/bin/activate  # macOS/Linux
# OR
voxtral_env\Scripts\activate     # Windows

pip install -r requirements.txt
```

## Usage

### Running the Script

**Full command:**
```bash
cd "/Users/ddbco/Desktop/Transcribe Voxtral Python script"
"/Users/ddbco/Desktop/Transcribe Voxtral Python script/voxtral_env/bin/python" transcribe_voxtral.py
```

**Or use the start script:**
```bash
./start.sh
```

### Input Files

Place audio files in the script directory. Supported formats:

- `.wav` - Waveform Audio File
- `.mp3` - MPEG-1 Audio Layer III
- `.flac` - Free Lossless Audio Codec
- `.m4a` - MPEG-4 Audio

The script will process all audio files in the input directory automatically.

### Output

Transcriptions are saved in the `transcriptions_voxtral_final/` directory:
```
transcriptions_voxtral_final/
├── audio_file_1_transcription.txt
├── audio_file_2_transcription.txt
└── ...
```

Output files are UTF-8 encoded text files with the naming pattern:
```
[original_filename]_transcription.txt
```

## Configuration

### Language Settings

Edit line 16 in [transcribe_voxtral.py](transcribe_voxtral.py:16) to change the transcription language:

```python
language="fr",  # Change this to your desired language code
```

**Supported Language Codes:**

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

### Key Configuration Parameters

Edit [transcribe_voxtral.py](transcribe_voxtral.py) to customize:

| Parameter | Line | Default | Description |
|-----------|------|---------|-------------|
| **INPUT_DIRECTORY** | 79 | Current directory | Where audio files are located |
| **OUTPUT_SUBFOLDER_NAME** | 80 | `transcriptions_voxtral_final` | Output folder name |
| **MODEL_ID** | 81 | `mistralai/Voxtral-Mini-3B-2507` | AI model identifier |
| **language** | 16 | `"fr"` | Transcription language code |
| **chunk_duration_s** | 36 | `2 * 60` (120 seconds) | Chunk size in seconds |
| **sample_rate** | 37 | `16000` | Audio sample rate in Hz |
| **max_new_tokens** | 26 | `512` | Maximum tokens per chunk |

### Example Configuration Changes

**Change language to English:**

```python
# Line 16
language="en",
```

**Increase chunk size to 5 minutes:**

```python
# Line 36
chunk_duration_s = 5 * 60  # 5-minute chunks
```

**Change input directory:**

```python
# Line 79
INPUT_DIRECTORY = "/path/to/your/audio/files"
```

## Technical Details

### Model Information

- **Model:** Mistral AI Voxtral-Mini-3B-2507
- **Type:** Conditional Generation (Audio-to-Text)
- **Size:** ~20GB download
- **Architecture:** Transformer-based encoder-decoder
- **Sample Rate:** 16kHz (automatically resampled)

### Device Detection & Performance

The script automatically selects the best available device:

1. **MPS (Apple Silicon)** - M1/M2/M3 chips
   - Uses `bfloat16` precision
   - Fastest on Apple Silicon Macs
   - Includes automatic cache clearing

2. **CUDA (NVIDIA GPUs)**
   - Uses `bfloat16` precision
   - Requires CUDA-compatible GPU
   - Requires CUDA toolkit installed

3. **CPU (Fallback)**
   - Uses `float32` precision
   - Works on all systems
   - Slower than GPU options

### Processing Pipeline

```
Audio File (any format)
    ↓
Load & Resample to 16kHz mono
    ↓
Split into 2-minute chunks
    ↓
Save each chunk as temporary WAV
    ↓
Transcribe chunk using Voxtral model
    ↓
Clean up temporary files
    ↓
Combine all transcriptions
    ↓
Save as UTF-8 text file
```

### Memory Management

- **Chunking:** Files are split into 2-minute segments to prevent memory overflow
- **Sequential Processing:** Chunks are processed one at a time
- **Temporary Files:** Automatically deleted after each chunk
- **Cache Clearing:** MPS cache is cleared after each chunk on Apple Silicon

### File Structure

```
Transcribe Voxtral Python script/
├── transcribe_voxtral.py      # Main script
├── requirements.txt           # Python dependencies
├── setup.sh                   # macOS setup script
├── start.sh                   # Launch script
├── README.md                  # This file
├── voxtral_env/              # Python virtual environment
└── transcriptions_voxtral_final/  # Output directory (created on first run)
```

## Performance Notes

### Processing Speed (approximate)

| Device | Speed | Example |
|--------|-------|---------|
| Apple M1/M2/M3 (MPS) | ~1-2x realtime | 10 min audio = 5-10 min processing |
| NVIDIA GPU (CUDA) | ~1-3x realtime | 10 min audio = 3-10 min processing |
| CPU | ~0.1-0.5x realtime | 10 min audio = 20-100 min processing |

**Note:** Speeds vary based on audio quality, language, and specific hardware

### First Run

- Model download: 20GB (one-time, cached locally)
- Download time: 10-60 minutes depending on internet speed
- Model location: `~/.cache/huggingface/hub/`

### Subsequent Runs

- No download needed
- Processing starts immediately
- Model loads in 10-30 seconds

## Dependencies

See [requirements.txt](requirements.txt):

| Package | Purpose |
|---------|---------|
| `torch` | Deep learning framework |
| `librosa` | Audio loading and processing |
| `soundfile` | Audio file I/O |
| `transformers` | HuggingFace model interface |
| `accelerate` | Model optimization |
| `mistral-common` | Mistral AI utilities |
| `numpy<2.3` | Numerical operations |

## Troubleshooting

### Model Download Issues

**Problem:** Model download fails or times out

- Ensure 20GB+ free disk space
- Check internet connection stability
- Try again - downloads resume automatically
- Check firewall/proxy settings

**Problem:** "Failed to load the model" error

- Verify Python version is 3.11+
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

### Memory Issues

**Problem:** Out of memory errors during processing

- Reduce chunk duration (line 36): `chunk_duration_s = 1 * 60` (1 minute)
- Close other applications
- Restart your computer to free up RAM
- For large files, process them one at a time

**Problem:** System becomes unresponsive

- Lower chunk size to 30-60 seconds
- Ensure you have at least 8GB available RAM
- Monitor memory usage with Activity Monitor/Task Manager

### Audio Processing Errors

**Problem:** "Audio file not found"

- Check file path is correct
- Ensure file is in the INPUT_DIRECTORY
- Verify file permissions (readable)

**Problem:** "Audio format not supported"

- Convert to MP3, WAV, FLAC, or M4A
- Use tools like FFmpeg: `ffmpeg -i input.xxx output.mp3`

**Problem:** Poor transcription quality

- Check audio quality (clear speech, minimal background noise)
- Verify correct language code (line 16)
- Increase audio quality if possible (higher bitrate)
- Try larger chunk sizes for better context

### Device-Specific Issues

**MPS (Apple Silicon):**

- If MPS fails, script will automatically fall back to CPU
- Update macOS to latest version for best MPS support
- Check: `python -c "import torch; print(torch.backends.mps.is_available())"`

**CUDA (NVIDIA):**

- Install CUDA toolkit matching your PyTorch version
- Check: `python -c "import torch; print(torch.cuda.is_available())"`
- Update NVIDIA drivers

**CPU Mode:**

- Be patient - processing is significantly slower
- Consider using smaller chunk sizes (60 seconds)
- Process files overnight if large

### Permission Errors

**Problem:** Cannot write to output directory

- Check directory permissions
- Ensure you have write access to the script directory
- Try running from a different location

### Other Issues

**Problem:** Script processes same files multiple times

- Script only processes files that don't have existing transcriptions
- Delete old transcription files if you want to re-process
- Check for duplicate audio files with different extensions

**Problem:** Transcriptions are in wrong language

- Verify language parameter (line 16) matches your audio
- Model may auto-detect if parameter is incorrect

## Advanced Usage

### Processing Specific Files

To process only specific files, you can:

1. Move files to a separate directory
2. Update INPUT_DIRECTORY (line 79)
3. Run the script

### Batch Processing Large Collections

For processing many files:

1. Place all audio files in INPUT_DIRECTORY
2. Run the script once - it will process all files
3. Monitor progress in terminal output
4. Transcriptions appear in output folder as they complete

### Customizing Output Format

To modify output format, edit lines 67-71 in [transcribe_voxtral.py](transcribe_voxtral.py:67):

```python
# Current: space-separated chunks
final_transcription = " ".join(all_transcriptions).strip()

# Alternative: paragraph breaks between chunks
final_transcription = "\n\n".join(all_transcriptions).strip()
```

### Re-transcribing Files

The script skips files that already have transcriptions. To re-transcribe:

1. Delete the existing `_transcription.txt` file
2. Run the script again

## Best Practices

1. **Audio Quality:** Use high-quality recordings (minimal background noise)
2. **File Organization:** Keep audio files organized by project/date
3. **Storage:** Ensure adequate disk space before processing large batches
4. **Monitoring:** Watch the first few chunks to verify correct language/quality
5. **Backup:** Keep original audio files as backup
6. **Testing:** Test with a short audio clip first to verify setup

## License

For educational and research purposes. Check Mistral AI's license terms for the Voxtral model at [HuggingFace Model Page](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)

## Support

For issues or questions:

1. Check this README's troubleshooting section
2. Verify your configuration matches the examples
3. Check model documentation at HuggingFace
4. Review error messages carefully - they often indicate the solution
