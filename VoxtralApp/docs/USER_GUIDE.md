# Voxtral Web Application - User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Getting Started](#getting-started)
4. [Using the Application](#using-the-application)
5. [Features](#features)
6. [Troubleshooting](#troubleshooting)
7. [FAQ](#faq)

---

## Introduction

Welcome to the Voxtral Web Application! This guide will help you transcribe audio and video files using AI-powered speech recognition. The application runs locally on your computer, ensuring privacy and eliminating API costs.

### Key Features

- 🎵 **Audio & Video Support**: Works with MP3, M4A, MP4, WAV, FLAC, AVI, MOV
- 🌍 **30+ Languages**: Transcribe in English, French, Spanish, and more
- 🔒 **Privacy-Focused**: All processing happens locally on your computer
- ⚡ **Real-Time Progress**: See live updates as your file is transcribed
- 📋 **Easy Export**: Copy to clipboard or download as text file
- 💻 **Cross-Platform**: Works on both Windows and macOS

---

## Installation

### Prerequisites

- **Python 3.11 or later**
- **20GB+ free disk space** (for AI model download)
- **8GB+ RAM** (16GB recommended)
- **Internet connection** (for initial model download)

### Step 1: Download the Project

If you haven't already, download or clone the Voxtral transcription repository to your computer.

### Step 2: Install Dependencies

#### macOS/Linux

1. Open Terminal
2. Navigate to the project directory:
   ```bash
   cd path/to/transcribe-voxtral-main
   ```

3. Run the setup script:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

#### Windows

1. Open Command Prompt or PowerShell
2. Navigate to the project directory:
   ```cmd
   cd path\to\transcribe-voxtral-main
   ```

3. Create virtual environment and install dependencies:
   ```cmd
   python -m venv voxtral_env
   voxtral_env\Scripts\activate.bat
   pip install -r requirements.txt
   ```

### Step 3: First-Time Model Download

The first time you run the application, it will download the Voxtral AI model (~20GB). This may take 10-60 minutes depending on your internet speed. The model is cached locally and won't need to be downloaded again.

---

## Getting Started

### Starting the Web Application

#### macOS/Linux

```bash
./start_web.sh
```

#### Windows

Double-click `start_web.bat` or run in Command Prompt:
```cmd
start_web.bat
```

### Accessing the Application

Once started, open your web browser and go to:

```
http://localhost:5000
```

You should see the Voxtral Transcription interface.

### Stopping the Application

Press `Ctrl+C` in the terminal/command prompt where the application is running.

---

## Using the Application

### Step-by-Step Transcription

#### 1. Upload Your File

**Option A: Drag and Drop**
- Drag an audio or video file directly onto the upload area
- The file will be uploaded automatically

**Option B: Click to Browse**
- Click anywhere on the upload area
- Select your file from the file picker
- Click "Open" to upload

**Supported Formats:**
- Audio: `.wav`, `.mp3`, `.flac`, `.m4a`
- Video: `.mp4`, `.avi`, `.mov` (audio will be extracted automatically)

**File Size Limit:** 500MB

#### 2. Select Language

- Click the language dropdown
- Choose the language spoken in your audio/video
- Default is English

**Available Languages:**
- English, French, Spanish, German, Italian
- Portuguese, Dutch, Polish, Russian
- Chinese, Japanese, Korean, Arabic, Hindi
- Turkish, Swedish, Danish, Norwegian, Finnish
- Czech, Slovak, Ukrainian, Romanian, Greek
- Hebrew, Indonesian, Vietnamese, Thai, Malay, Catalan

#### 3. Start Transcription

- Click the **"Start Transcription"** button
- The application will begin processing your file

#### 4. Monitor Progress

- Watch the real-time progress bar
- See which chunk is being processed (e.g., "Chunk 5/20")
- View percentage completion

**Processing Speed:**
- Apple Silicon (M1/M2/M3): ~1-2x realtime
- NVIDIA GPU: ~1-3x realtime
- CPU: ~0.1-0.5x realtime

Example: A 10-minute audio file might take 5-10 minutes on Apple Silicon

#### 5. View Your Transcript

Once complete, the transcript will appear automatically with:
- Full transcribed text
- Word count
- Character count
- Audio duration

#### 6. Export Your Transcript

**Copy to Clipboard:**
- Click the **"Copy"** button
- Paste into any application (Word, Google Docs, email, etc.)

**Download as File:**
- Click the **"Download"** button
- Save the `.txt` file to your computer

**Clear Transcript:**
- Click the **"Clear"** button to remove the current transcript
- Upload a new file to start over

---

## Features

### Automatic Video to Audio Conversion

When you upload a video file (MP4, AVI, MOV), the application automatically:
1. Extracts the audio track
2. Converts it to a compatible format
3. Transcribes the audio
4. Provides the transcript

You don't need to manually convert videos!

### Chunked Processing

Large files are automatically split into 2-minute chunks to:
- Manage memory efficiently
- Prevent crashes on long recordings
- Provide granular progress updates

### Real-Time Updates

Using WebSocket technology, you see:
- Live progress percentages
- Current chunk being processed
- Estimated completion
- Instant transcript display

### Device Auto-Detection

The application automatically uses the best available hardware:
- **MPS** on Apple Silicon Macs (fastest)
- **CUDA** on NVIDIA GPUs
- **CPU** as fallback (slowest)

Check the badge at the top to see which device is active.

---

## Troubleshooting

### Application Won't Start

**Issue:** Error when running startup script

**Solutions:**
1. Ensure Python 3.11+ is installed:
   ```bash
   python --version
   ```
2. Check virtual environment exists:
   ```bash
   ls voxtral_env/
   ```
3. Reinstall dependencies:
   ```bash
   pip install -r requirements.txt --force-reinstall
   ```

---

### Can't Access http://localhost:5000

**Issue:** Browser shows "Can't connect" or "Connection refused"

**Solutions:**
1. Verify the server is running (check terminal for "Running on...")
2. Try `http://127.0.0.1:5000` instead
3. Check if port 5000 is already in use
4. Restart the application

---

### Model Download Fails

**Issue:** Error downloading Voxtral model

**Solutions:**
1. Check internet connection
2. Ensure 20GB+ free disk space
3. Check firewall/antivirus settings
4. Try again - downloads resume automatically
5. Manually download from [HuggingFace](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)

---

### Upload Fails

**Issue:** "Upload failed" error

**Solutions:**
1. Check file size is under 500MB
2. Verify file format is supported
3. Check disk space for uploads
4. Try renaming file (remove special characters)
5. Try a different file to isolate the issue

---

### Transcription Gets Stuck

**Issue:** Progress bar stops moving

**Solutions:**
1. Wait 5-10 minutes (large chunks take time)
2. Check terminal/console for error messages
3. Restart the application
4. Try with a shorter audio file first
5. Check available RAM (close other applications)

---

### Out of Memory Error

**Issue:** "Out of memory" or crash during transcription

**Solutions:**
1. Close other applications to free RAM
2. Restart your computer
3. Process shorter files (split large files)
4. Reduce chunk size in `transcription_engine.py` (line 150)
   ```python
   chunk_duration_s: int = 60  # 1 minute instead of 2
   ```

---

### Poor Transcription Quality

**Issue:** Incorrect or garbled transcription

**Solutions:**
1. **Verify correct language is selected**
2. Check audio quality (clear speech, minimal background noise)
3. Try a higher quality source file
4. For accented speech, try the language variant
5. Ensure audio volume is adequate

---

### Video Conversion Fails

**Issue:** "Failed to convert video to audio"

**Solutions:**
1. Install FFmpeg:
   - **macOS:** `brew install ffmpeg`
   - **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Install moviepy:
   ```bash
   pip install moviepy
   ```
3. Try converting the video manually first
4. Use a different video format

---

## FAQ

### Q: How much does it cost to use?

**A:** The application is completely free! All processing happens on your local computer with no API fees.

---

### Q: Is my audio data sent to the cloud?

**A:** No! All transcription happens locally on your computer. Your audio never leaves your machine.

---

### Q: How accurate are the transcriptions?

**A:** Accuracy depends on:
- Audio quality (clear speech = better results)
- Background noise (less noise = better accuracy)
- Speaker accent
- Correct language selection

Typical accuracy: 85-95% for clear audio

---

### Q: Can I transcribe multiple files at once?

**A:** Currently, the web app processes one file at a time. For batch processing, use the command-line script `transcribe_voxtral.py`.

---

### Q: What languages are supported?

**A:** 30+ languages including English, French, Spanish, German, Italian, Portuguese, Dutch, Polish, Russian, Chinese, Japanese, Korean, Arabic, Hindi, Turkish, Swedish, Danish, Norwegian, Finnish, Czech, Slovak, Ukrainian, Romanian, Greek, Hebrew, Indonesian, Vietnamese, Thai, Malay, and Catalan.

---

### Q: Can I use this for podcasts or YouTube videos?

**A:** Yes! Upload the audio file directly, or:
- For podcasts: Download the MP3
- For YouTube: Use a YouTube downloader to get MP4/M4A, then upload

---

### Q: How long does transcription take?

**A:** Depends on your hardware:
- **Apple Silicon**: ~1-2x realtime (10 min audio = 5-10 min processing)
- **NVIDIA GPU**: ~1-3x realtime
- **CPU**: ~0.1-0.5x realtime (10 min audio = 20-100 min)

---

### Q: Can I edit the transcript in the app?

**A:** The current version doesn't support editing. Copy the transcript to a text editor for modifications.

---

### Q: What if I have a very large file?

**A:** The app supports files up to 500MB. For larger files:
1. Split the file into smaller segments
2. Transcribe each segment separately
3. Combine transcripts manually

---

### Q: Can I use this offline?

**A:** Yes, after the initial model download! The model is cached locally, so subsequent transcriptions work without internet.

---

### Q: How do I update the application?

**A:** Pull the latest code from GitHub and reinstall dependencies:
```bash
git pull
pip install -r requirements.txt --upgrade
```

---

### Q: Where are my transcripts saved?

**A:** Transcripts are saved in the `transcriptions_voxtral_final/` folder as `.txt` files.

---

### Q: Can I change the chunk size?

**A:** Yes! Edit `transcription_engine.py` line 150. Smaller chunks use less memory but may reduce context accuracy.

---

## Best Practices

1. **Use high-quality audio** - Clear recordings produce better transcriptions
2. **Choose the correct language** - Wrong language = poor results
3. **Start with short files** - Test with 1-2 minute clips first
4. **Monitor resource usage** - Close other apps during large transcriptions
5. **Keep the app updated** - New versions may improve accuracy
6. **Save your transcripts** - Download or copy before clearing

---

## Getting Help

If you encounter issues not covered in this guide:

1. Check the [README.md](../README.md) for general information
2. Review [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for technical details
3. Check the [GitHub Issues](https://github.com/debrockb/transcribe-voxtral/issues)
4. Review error messages in the terminal/console

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Click upload area | Open file picker |
| Drag & Drop | Upload file |
| Ctrl+C (in terminal) | Stop server |

---

## Privacy & Security

- ✅ All processing is local
- ✅ No data sent to external servers
- ✅ No user tracking or analytics
- ✅ Open source code (auditable)
- ✅ No account required

---

## Performance Tips

1. **Close unnecessary apps** to free up RAM
2. **Use wired internet** for faster model download
3. **Process overnight** for very large files on CPU
4. **Update your graphics drivers** for GPU acceleration
5. **Use SSD storage** for faster file I/O

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Dual-core | Quad-core+ |
| RAM | 8GB | 16GB+ |
| Storage | 25GB free | 50GB+ free |
| OS | macOS 12+ / Windows 10+ | macOS 14+ / Windows 11+ |
| Python | 3.11 | 3.11+ |

---

## Acknowledgments

This application is powered by:
- **Mistral AI** - Voxtral-Mini-3B model
- **HuggingFace** - Transformers library
- **Flask** - Web framework
- **LibROSA** - Audio processing

---

Thank you for using Voxtral Transcription! 🎙️
