# Sample Data

This directory contains sample audio/video files for testing DuoSynco functionality.

## Files

### TheAnnunaki.wav
- **Format**: WAV (RIFF), 16-bit, mono, 24kHz
- **Size**: ~8.8MB
- **Content**: Host and guest conversation sample
- **Speakers**: 2 (host + guest)
- **Usage**: Perfect for testing speaker diarization and separation

## Testing with Samples

```bash
# Quick test with the provided sample
./scripts/run.sh sample_data/TheAnnunaki.wav --speakers 2 --verbose

# Expected output in ./output/:
# - TheAnnunaki_speaker_1.mp4 (isolated host audio)
# - TheAnnunaki_speaker_2.mp4 (isolated guest audio)
```

## Adding Your Own Samples

Place your test files here following these guidelines:

- **Audio formats**: WAV, MP3, AAC, FLAC
- **Video formats**: MP4, AVI, MOV, MKV
- **Recommended**: Clear audio, minimal background noise
- **File size**: Keep under 50MB for repository efficiency

Note: Large files are gitignored by default. Add exceptions in `.gitignore` if needed.