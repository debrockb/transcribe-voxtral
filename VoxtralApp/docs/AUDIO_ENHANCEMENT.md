# Distant Speaker Enhancement

## Overview

The **Distant Speaker Enhancement** feature improves transcription accuracy for recordings where speakers are far from the microphone (typically 5+ meters). It applies a sophisticated FFmpeg audio filter chain that enhances speech intelligibility before transcription.

## When to Use

Enable this feature when:

- **Conference room recordings** - Speakers seated around a table with a central microphone
- **Lecture recordings** - Captured from the back of a room or using room microphones
- **Interview recordings** - With inconsistent microphone distances between participants
- **Surveillance or security audio** - Fixed microphones in large spaces
- **Any recording where speech sounds quiet or muddy**

**Do NOT use when:**

- Audio was recorded with close microphones (podcasts, voice memos)
- Speech is already clear and at good volume levels
- You want to preserve the original audio characteristics

## How to Enable

### Web Interface

1. Upload your audio/video file
2. Select the transcription language
3. **Check the "Distant Speaker Enhancement" checkbox** (located below language selection)
4. Click "Start Transcription"

### API

Include `enable_audio_enhancement: true` in your `/api/transcribe` request:

```json
{
  "file_id": "your-file-id",
  "language": "en",
  "enable_audio_enhancement": true
}
```

## Technical Details

### Filter Chain

The enhancement applies the following FFmpeg filters in sequence:

```
highpass → lowpass → compand → equalizer (x4) → loudnorm
```

### 1. High-Pass Filter (80Hz)

```
highpass=f=80
```

**Purpose:** Removes low-frequency rumble below 80Hz.

**What it removes:**
- Air conditioning hum
- Traffic rumble
- Building vibrations
- Wind noise (low-frequency component)

**Why 80Hz:** Human speech fundamentals start around 85Hz (male) to 165Hz (female). The 80Hz cutoff preserves all speech while removing environmental noise.

### 2. Low-Pass Filter (8kHz)

```
lowpass=f=8000
```

**Purpose:** Removes high-frequency hiss above 8kHz.

**What it removes:**
- Electronic hiss
- High-frequency room noise
- Sibilance artifacts from cheap microphones

**Why 8kHz:** Human speech intelligibility peaks around 2-4kHz. Frequencies above 8kHz contribute little to understanding but add noise.

### 3. Dynamic Compression (Compand)

```
compand=attacks=0.01:decays=0.3:points=-80/-80|-45/-35|-27/-20|-15/-10|0/-5|20/-5:gain=8:volume=-90:delay=0.1
```

**Purpose:** Brings up quiet sounds (distant speech) while limiting loud sounds.

**Parameters explained:**
- `attacks=0.01` - Fast attack (10ms) to catch transients
- `decays=0.3` - Moderate decay (300ms) for natural sound
- `points` - Compression curve (input/output dB pairs)
- `gain=8` - 8dB makeup gain after compression
- `volume=-90` - Gate threshold (silence below -90dB)
- `delay=0.1` - 100ms lookahead for smoother compression

**Compression curve:**

| Input (dB) | Output (dB) | Effect |
|------------|-------------|--------|
| -80 | -80 | No change (silence) |
| -45 | -35 | +10dB boost (very quiet speech) |
| -27 | -20 | +7dB boost (quiet speech) |
| -15 | -10 | +5dB boost (normal speech) |
| 0 | -5 | -5dB reduction (loud sounds) |
| +20 | -5 | -25dB limiting (prevents clipping) |

### 4. Equalizer Boosts (Voice Frequencies)

Four EQ bands boost key speech frequencies:

```
equalizer=f=300:t=h:w=200:g=3    # Low voice fundamentals
equalizer=f=1000:t=h:w=500:g=5   # Voice body/presence
equalizer=f=2500:t=h:w=500:g=4   # Voice clarity
equalizer=f=3500:t=h:w=500:g=2   # Voice brightness
```

| Frequency | Bandwidth | Gain | What it boosts |
|-----------|-----------|------|----------------|
| 300 Hz | 200 Hz | +3 dB | Male voice fundamentals, warmth |
| 1000 Hz | 500 Hz | +5 dB | Voice body, presence, "fullness" |
| 2500 Hz | 500 Hz | +4 dB | Consonant clarity, intelligibility |
| 3500 Hz | 500 Hz | +2 dB | Brightness, "air", sibilants |

### 5. Loudness Normalization

```
loudnorm=I=-14:TP=-1:LRA=11
```

**Purpose:** Normalizes output to broadcast standard loudness.

**Parameters:**
- `I=-14` - Target integrated loudness: -14 LUFS (broadcast standard)
- `TP=-1` - True peak maximum: -1 dBTP (prevents clipping)
- `LRA=11` - Loudness range: 11 LU (natural dynamics)

**Why -14 LUFS:** This is the loudness standard for streaming platforms (Spotify, YouTube, podcasts). It ensures consistent, comfortable listening levels.

## Output Format

The enhanced audio is converted to:
- **Sample rate:** 16kHz (optimal for speech recognition)
- **Channels:** Mono
- **Format:** PCM 16-bit WAV

## Performance Impact

- **Processing time:** Adds 2-5 seconds per minute of audio
- **Disk space:** Creates temporary enhanced WAV file (deleted after transcription)
- **Memory:** Minimal additional memory usage

## Limitations

1. **Cannot recover severely degraded audio** - If speech is completely inaudible, enhancement won't help
2. **May affect music** - Not recommended for transcribing musical content
3. **Requires FFmpeg** - FFmpeg must be installed on the system

## Troubleshooting

### Enhancement not working

1. Verify FFmpeg is installed: `ffmpeg -version`
2. Check server logs for "Audio enhancement" messages
3. Ensure the checkbox is checked before clicking "Start Transcription"

### Audio sounds worse after enhancement

Some recordings don't benefit from enhancement:
- Already well-recorded audio
- Very noisy environments (enhancement can amplify noise)
- Audio with extreme clipping

In these cases, disable enhancement and transcribe the original audio.

### FFmpeg errors

If you see FFmpeg errors in the logs:
- Update FFmpeg to the latest version
- Check disk space for temporary files
- Verify input file isn't corrupted

## Example Use Cases

### Conference Call Recording

A 1-hour conference call recorded with a speakerphone in the center of a table:
- Multiple speakers at varying distances (2-5 meters)
- Some speakers are quiet, others are loud
- Background air conditioning hum

**Result with enhancement:** Quiet speakers are boosted to match loud speakers, AC hum is removed, all speech is normalized to consistent volume.

### Lecture Recording

A university lecture recorded from the back row:
- Professor is 15+ meters from recorder
- Room echo and reverberation
- Student questions barely audible

**Result with enhancement:** Professor's voice is boosted and clarified, some improvement to student questions (limited by severe distance), echo remains but speech is more intelligible.

### Security Camera Audio

Audio from a security camera in a retail store:
- Fixed position, varying customer distances
- Background music and ambient noise
- Low-quality built-in microphone

**Result with enhancement:** Conversations within 5 meters are improved, background music reduced relative to speech, overall intelligibility improved.

## API Reference

See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for the complete API reference, including the `enable_audio_enhancement` parameter.

## Related Documentation

- [User Guide](USER_GUIDE.md) - General usage instructions
- [API Documentation](API_DOCUMENTATION.md) - REST API reference
- [Main README](../../README.md) - Project overview
