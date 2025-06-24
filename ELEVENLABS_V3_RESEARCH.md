# ElevenLabs v3 Model Research Summary

## Model Availability Status

### ✅ **Available Models** (Public API)
- `eleven_flash_v2_5` - Ultra-fast, 75ms latency (recommended)
- `eleven_turbo_v2_5` - High quality, low latency  
- `eleven_multilingual_v2` - Current default (29 languages)
- `eleven_flash_v2` - English only
- `eleven_turbo_v2` - English only

### ⚠️ **ElevenLabs v3** (`eleven_v3`)
- **Status**: Alpha version - NOT publicly available
- **API Access**: Requires contacting ElevenLabs sales team
- **Limitations**: 
  - Not recommended for real-time applications
  - Requires multiple generations for best results
  - Limited availability

## Implementation Status

### ✅ **Completed**
1. **Model Parameter Support Added**:
   - Added `--model-id` / `-mid` parameter to main CLI
   - Added `-m` / `--model` parameter to ali CLI
   - Model ID properly passed through TTS settings pipeline

2. **API Integration Ready**:
   - Existing code already supports custom model IDs via `settings['model_id']`
   - Quality profiles updated to use appropriate models per level
   - Model parameter validation in place

3. **CLI Usage Examples**:
   ```bash
   # Use faster Flash v2.5 model
   ali tts output/transcript.json -m eleven_flash_v2_5
   
   # Use v3 model (requires alpha access)
   ali tts output/transcript.json -m eleven_v3
   
   # Via main DuoSynco CLI
   python -m src.main transcript.json --mode tts --model-id eleven_flash_v2_5
   ```

## Recommendations

### **For Current Use**
1. **Use `eleven_flash_v2_5`** for fast, high-quality TTS (75ms latency)
2. **Use `eleven_multilingual_v2`** for maximum language support
3. **Use `eleven_turbo_v2_5`** for balanced quality/speed

### **For v3 Access**
1. Contact ElevenLabs sales for alpha access
2. When available, expect to need multiple generations per text
3. Not suitable for real-time applications

## Quality Profile Defaults

The system automatically selects models based on quality settings:

- **Low**: `eleven_turbo_v2_5` (fastest)
- **Medium**: `eleven_multilingual_v2` (balanced)
- **High**: `eleven_multilingual_v2` (stable default)
- **Ultra**: `eleven_multilingual_v2` (maximum quality settings)

Users can override these with the `--model-id` parameter.

## Code Changes Made

1. **Main CLI** (`src/main.py`):
   - Added `--model-id` parameter
   - Updated `handle_tts_mode()` to accept and pass model_id
   - Model ID properly passed to TTS settings

2. **Ali CLI** (`ali/cli.py`, `ali/commands.py`, `ali/config.py`):
   - Added `-m` / `--model` parameter support
   - Updated configuration to include model_id
   - Commands pass model_id when specified

3. **Documentation** (`ali/cli.py`):
   - Added usage examples for different models
   - Included v3 alpha access note

## Testing

The implementation is ready for testing with available models:

```bash
# Test with Flash v2.5 (fastest)
ali tts output/annunaki-fa_edited_transcript_tts_format.json -m eleven_flash_v2_5 -v

# Test with Turbo v2.5 (balanced)
ali tts output/annunaki-fa_edited_transcript_tts_format.json -m eleven_turbo_v2_5 -v
```

Once ElevenLabs grants v3 alpha access, the same commands will work with `eleven_v3`.