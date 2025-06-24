"""
WhisperX-based Speaker Diarization Module
Provides advanced speaker diarization using WhisperX's transcription + diarization capabilities
"""

from pathlib import Path
from typing import List, Dict, Optional
import gc

try:
    import whisperx
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False

from .diarization import SpeakerSegment
from ..utils.config import Config


class WhisperXDiarizer:
    """
    WhisperX-based speaker diarization with word-level timestamps
    Follows the official WhisperX workflow for accurate speaker separation
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.alignment_model = None
        self.alignment_metadata = None
        self.diarize_model = None
        self.device = "cpu"  # Default to CPU for compatibility
        self.compute_type = "int8"  # Use int8 for better performance
        self.use_large_alignment = True  # Use wav2vec2-large for better alignment
        
        if WHISPERX_AVAILABLE:
            self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize WhisperX ASR model"""
        try:
            if self.config.verbose:
                print("🎯 Initializing WhisperX models...")
            
            # Load Whisper ASR model (step 1 of WhisperX workflow)
            self.model = whisperx.load_model(
                "base",  # Can be: tiny, base, small, medium, large
                self.device,
                compute_type=self.compute_type
            )
            
            if self.config.verbose:
                print("✅ WhisperX ASR model loaded")
            
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  WhisperX model initialization failed: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        """Check if WhisperX diarization is available"""
        return WHISPERX_AVAILABLE and self.model is not None
    
    def _load_large_alignment_model(self, language_code: str = "en"):
        """Load wav2vec2-large model for better alignment"""
        try:
            import torchaudio
            from torchaudio.pipelines import WAV2VEC2_LARGE
            from pathlib import Path
            
            if self.config.verbose:
                print("  🔧 Loading wav2vec2-large for enhanced alignment...")
            
            # Load the large model
            model = WAV2VEC2_LARGE.get_model()
            
            # Create metadata for English
            metadata = {
                'language': language_code,
                'dictionary': {
                    '-': 0, '|': 1, 'e': 2, 't': 3, 'a': 4, 'o': 5, 'n': 6, 'i': 7, 'h': 8, 's': 9, 
                    'r': 10, 'd': 11, 'l': 12, 'u': 13, 'm': 14, 'w': 15, 'c': 16, 'f': 17, 'g': 18, 
                    'y': 19, 'p': 20, 'b': 21, 'v': 22, 'k': 23, "'": 24, 'x': 25, 'j': 26, 'q': 27, 'z': 28
                },
                'type': 'torchaudio'
            }
            
            if self.config.verbose:
                print("  ✅ wav2vec2-large alignment model loaded")
            
            return model, metadata
            
        except Exception as e:
            if self.config.verbose:
                print(f"  ⚠️  Could not load wav2vec2-large: {e}")
                print("  🔄 Falling back to default alignment model")
            return None, None
    
    def diarize(self, audio_file: Path, num_speakers: int = 2) -> List[SpeakerSegment]:
        """
        Perform speaker diarization using WhisperX following the official workflow:
        1. Transcribe with original whisper (batched)
        2. Align whisper output  
        3. Assign speaker labels
        
        Args:
            audio_file: Path to audio file
            num_speakers: Expected number of speakers
            
        Returns:
            List of speaker segments with timestamps
        """
        if not self.is_available():
            if self.config.verbose:
                print("❌ WhisperX not available for diarization")
            return []
        
        try:
            if self.config.verbose:
                print(f"🎯 Running WhisperX diarization on {audio_file}")
            
            # Load audio
            audio = whisperx.load_audio(str(audio_file))
            
            # 1. Transcribe with original whisper (batched)
            if self.config.verbose:
                print("  📝 Transcribing audio...")
            
            result = self.model.transcribe(audio, batch_size=16)
            
            if self.config.verbose:
                print(f"  ✅ Transcribed {len(result['segments'])} segments")
                print(f"  🌍 Detected language: {result.get('language', 'unknown')}")
            
            # 2. Align whisper output
            if self.config.verbose:
                print("  🎯 Aligning transcription...")
            
            # Load alignment model for detected language (only once)
            if self.alignment_model is None or self.alignment_metadata is None:
                if self.use_large_alignment and result["language"] == "en":
                    # Try to use wav2vec2-large for English
                    self.alignment_model, self.alignment_metadata = self._load_large_alignment_model(result["language"])
                
                # Fallback to default WhisperX alignment model if large model failed or not English
                if self.alignment_model is None or self.alignment_metadata is None:
                    try:
                        self.alignment_model, self.alignment_metadata = whisperx.load_align_model(
                            language_code=result["language"], 
                            device=self.device
                        )
                    except Exception as e:
                        if self.config.verbose:
                            print(f"⚠️  Could not load alignment model: {e}")
                        return self._fallback_to_basic_segments(result)
            
            # Perform alignment for word-level timestamps
            try:
                result = whisperx.align(
                    result["segments"], 
                    self.alignment_model, 
                    self.alignment_metadata, 
                    audio, 
                    self.device, 
                    return_char_alignments=False
                )
                
                if self.config.verbose:
                    print("  ✅ Alignment completed")
            except Exception as e:
                if self.config.verbose:
                    print(f"⚠️  Alignment failed: {e}")
                return self._fallback_to_basic_segments(result)
            
            # 3. Assign speaker labels
            if self.config.verbose:
                print(f"  👥 Performing speaker diarization ({num_speakers} speakers)...")
            
            # Initialize diarization pipeline (only once)
            if self.diarize_model is None:
                try:
                    # Note: This may require HuggingFace token for some models
                    self.diarize_model = whisperx.diarize.DiarizationPipeline(
                        use_auth_token=None,  # Set to YOUR_HF_TOKEN if needed
                        device=self.device
                    )
                except Exception as e:
                    if self.config.verbose:
                        print(f"⚠️  Could not load diarization model: {e}")
                    return self._fallback_to_basic_segments(result)
            
            # Perform diarization with min/max speakers
            try:
                diarize_segments = self.diarize_model(
                    audio, 
                    min_speakers=num_speakers, 
                    max_speakers=num_speakers
                )
                
                # Assign word speakers
                result = whisperx.assign_word_speakers(diarize_segments, result)
                
                if self.config.verbose:
                    print("  ✅ Speaker assignment completed")
                    
            except Exception as e:
                if self.config.verbose:
                    print(f"⚠️  Speaker diarization failed: {e}")
                return self._fallback_to_basic_segments(result)
            
            # Convert to our SpeakerSegment format
            speaker_segments = self._convert_whisperx_to_segments(result)
            
            if self.config.verbose:
                total_time = sum(seg.end_time - seg.start_time for seg in speaker_segments)
                print(f"🎯 WhisperX found {len(speaker_segments)} speaker segments ({total_time:.1f}s total)")
            
            return speaker_segments
            
        except Exception as e:
            if self.config.verbose:
                print(f"❌ WhisperX diarization failed: {e}")
                import traceback
                traceback.print_exc()
            return []
    
    def _fallback_to_basic_segments(self, result: Dict) -> List[SpeakerSegment]:
        """
        Fallback to basic segments without word-level speaker assignment
        """
        if self.config.verbose:
            print("  🔄 Using basic segments without speaker diarization")
        
        segments = []
        for i, segment in enumerate(result.get("segments", [])):
            start_time = segment.get("start", 0.0)
            end_time = segment.get("end", 0.0)
            
            # Assign speakers alternately as a basic fallback
            speaker_id = f"SPEAKER_{i % 2}"
            
            segments.append(SpeakerSegment(
                speaker_id=speaker_id,
                start_time=float(start_time),
                end_time=float(end_time),
                confidence=0.5  # Lower confidence for fallback
            ))
        
        return segments
    
    def _convert_whisperx_to_segments(self, result: Dict) -> List[SpeakerSegment]:
        """
        Convert WhisperX result to our SpeakerSegment format using word-level speaker assignments
        This creates segments based on continuous speech from the same speaker
        """
        segments = []
        current_speaker = None
        segment_start = None
        segment_end = None
        
        for segment in result.get("segments", []):
            # Get words with speaker labels
            words = segment.get("words", [])
            
            for word in words:
                speaker = word.get("speaker")
                start_time = word.get("start")
                end_time = word.get("end")
                
                # Skip words without complete information
                if speaker is None or start_time is None or end_time is None:
                    continue
                
                # Convert timestamps to float if they aren't already
                try:
                    start_time = float(start_time)
                    end_time = float(end_time)
                except (ValueError, TypeError):
                    continue
                
                # If this is a new speaker or first word
                if speaker != current_speaker:
                    # Save previous segment if exists
                    if (current_speaker is not None and 
                        segment_start is not None and 
                        segment_end is not None):
                        segments.append(SpeakerSegment(
                            speaker_id=current_speaker,
                            start_time=segment_start,
                            end_time=segment_end,
                            confidence=0.95  # WhisperX typically has high confidence
                        ))
                    
                    # Start new segment
                    current_speaker = speaker
                    segment_start = start_time
                    segment_end = end_time
                else:
                    # Continue current segment - extend end time
                    segment_end = end_time
        
        # Don't forget the last segment
        if (current_speaker is not None and 
            segment_start is not None and 
            segment_end is not None):
            segments.append(SpeakerSegment(
                speaker_id=current_speaker,
                start_time=segment_start,
                end_time=segment_end,
                confidence=0.95
            ))
        
        # Merge very close segments from the same speaker
        merged_segments = self._merge_close_segments(segments)
        
        return merged_segments
    
    def _merge_close_segments(self, segments: List[SpeakerSegment], 
                            gap_threshold: float = 0.2) -> List[SpeakerSegment]:
        """
        Merge segments from the same speaker that are very close together
        Uses a smaller gap threshold for more precise merging
        
        Args:
            segments: List of speaker segments
            gap_threshold: Maximum gap in seconds to merge across (reduced to 0.2s)
            
        Returns:
            List of merged segments
        """
        if not segments:
            return []
        
        # Sort segments by start time
        segments.sort(key=lambda x: x.start_time)
        
        merged = []
        current_segment = segments[0]
        
        for next_segment in segments[1:]:
            # If same speaker and gap is small, merge
            gap = next_segment.start_time - current_segment.end_time
            if (next_segment.speaker_id == current_segment.speaker_id and 
                gap <= gap_threshold):
                # Extend current segment
                current_segment = SpeakerSegment(
                    speaker_id=current_segment.speaker_id,
                    start_time=current_segment.start_time,
                    end_time=next_segment.end_time,
                    confidence=max(current_segment.confidence, next_segment.confidence)
                )
            else:
                # Different speaker or gap too large, save current and start new
                merged.append(current_segment)
                current_segment = next_segment
        
        # Don't forget the last segment
        merged.append(current_segment)
        
        return merged
    
    def get_transcription(self, audio_file: Path) -> Optional[str]:
        """
        Get transcription text from WhisperX (bonus feature)
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Transcribed text or None if failed
        """
        if not self.is_available():
            return None
        
        try:
            # Load and transcribe audio
            audio = whisperx.load_audio(str(audio_file))
            result = self.model.transcribe(audio, batch_size=16)
            
            # Combine all segments into text
            text_segments = []
            for segment in result.get("segments", []):
                text = segment.get("text", "").strip()
                if text:
                    text_segments.append(text)
            
            return " ".join(text_segments)
            
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  WhisperX transcription failed: {e}")
            return None
    
    def cleanup(self) -> None:
        """Clean up WhisperX models and resources"""
        try:
            # Clear models to free memory
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
            
            if hasattr(self, 'alignment_model') and self.alignment_model is not None:
                del self.alignment_model
                self.alignment_model = None
                
            if hasattr(self, 'alignment_metadata') and self.alignment_metadata is not None:
                del self.alignment_metadata
                self.alignment_metadata = None
            
            if hasattr(self, 'diarize_model') and self.diarize_model is not None:
                del self.diarize_model
                self.diarize_model = None
            
            # Force garbage collection and free GPU memory if available
            gc.collect()
            
            # Clear CUDA cache if torch is available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            if self.config.verbose:
                print("🗑️  WhisperX models cleaned up")
                
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  WhisperX cleanup failed: {e}")