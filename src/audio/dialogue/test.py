"""
Dialogue Preview and Testing Functionality
"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import tempfile

from .base import DialogueBase, DialogueSegment
from .converter import TranscriptToDialogueConverter
from .generator import DialogueGenerator
from .profile import CharacterManager
from ..providers.elevenlabs.voice import VoiceManager

logger = logging.getLogger(__name__)


class DialogueTester:
    """
    Testing and preview functionality for dialogue generation
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize dialogue tester
        
        Args:
            api_key: ElevenLabs API key
        """
        self.api_key = api_key
        self.voice_manager = None
        self.generator = None
        
        if api_key:
            try:
                self.voice_manager = VoiceManager(api_key)
                self.generator = DialogueGenerator(api_key)
            except Exception as e:
                logger.warning("Failed to initialize dialogue components: %s", e)
    
    def test_transcript_parsing(self, transcript_file: Path) -> Dict[str, Any]:
        """
        Test transcript file parsing
        
        Args:
            transcript_file: Path to transcript file
            
        Returns:
            Parsing test results
        """
        results = {
            'file_exists': transcript_file.exists(),
            'parsing_success': False,
            'segments_found': 0,
            'speakers_found': [],
            'format_detected': 'unknown',
            'errors': []
        }
        
        if not results['file_exists']:
            results['errors'].append(f"File not found: {transcript_file}")
            return results
        
        try:
            converter = TranscriptToDialogueConverter(self.voice_manager)
            
            # Try parsing the file
            if transcript_file.suffix.lower() == '.json':
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                segments = converter.parse_stt_json(json_data)
                results['format_detected'] = 'json'
            else:
                segments = converter.parse_transcript_file(transcript_file)
                results['format_detected'] = 'text'
            
            results['parsing_success'] = True
            results['segments_found'] = len(segments)
            results['speakers_found'] = list(set(seg.speaker_id for seg in segments))
            
            # Additional analysis
            if segments:
                total_duration = sum(seg.duration for seg in segments if seg.duration)
                avg_segment_length = sum(len(seg.text.split()) for seg in segments) / len(segments)
                
                results['analysis'] = {
                    'total_duration': total_duration,
                    'average_segment_words': round(avg_segment_length, 1),
                    'longest_segment': max(len(seg.text) for seg in segments),
                    'shortest_segment': min(len(seg.text) for seg in segments)
                }
            
        except Exception as e:
            results['errors'].append(f"Parsing error: {e}")
        
        return results
    
    def test_voice_availability(self, language: str = "en") -> Dict[str, Any]:
        """
        Test voice availability for a language
        
        Args:
            language: Language code to test
            
        Returns:
            Voice availability test results
        """
        results = {
            'voice_manager_available': self.voice_manager is not None,
            'total_voices': 0,
            'language_voices': 0,
            'male_voices': 0,
            'female_voices': 0,
            'sample_voices': [],
            'errors': []
        }
        
        if not self.voice_manager:
            results['errors'].append("Voice manager not available - check API key")
            return results
        
        try:
            # Get all voices
            all_voices = self.voice_manager.get_all_voices()
            results['total_voices'] = len(all_voices)
            
            # Get voices for specific language
            language_voices = self.voice_manager.get_voices_by_language(language)
            results['language_voices'] = len(language_voices)
            
            # Count by gender
            for voice in language_voices:
                gender = voice.get('labels', {}).get('gender', '').lower()
                if gender == 'male':
                    results['male_voices'] += 1
                elif gender == 'female':
                    results['female_voices'] += 1
            
            # Get sample voices (first 5)
            results['sample_voices'] = [
                {
                    'name': voice.get('name', 'Unknown'),
                    'voice_id': voice.get('voice_id', ''),
                    'gender': voice.get('labels', {}).get('gender', 'unknown')
                }
                for voice in language_voices[:5]
            ]
            
        except Exception as e:
            results['errors'].append(f"Voice availability error: {e}")
        
        return results
    
    def test_dialogue_api_availability(self) -> Dict[str, Any]:
        """
        Test ElevenLabs Text to Dialogue API availability
        
        Returns:
            API availability test results
        """
        results = {
            'generator_available': self.generator is not None,
            'api_available': False,
            'test_successful': False,
            'response_time': None,
            'errors': []
        }
        
        if not self.generator:
            results['errors'].append("Dialogue generator not available - check API key")
            return results
        
        try:
            import time
            start_time = time.time()
            
            api_status = self.generator.test_dialogue_api_availability()
            results['response_time'] = round(time.time() - start_time, 2)
            
            results['api_available'] = api_status.get('available', False)
            results['api_status'] = api_status.get('status', 'Unknown')
            
            if api_status.get('available', False):
                results['test_successful'] = True
            else:
                results['errors'].append(api_status.get('status', 'API not available'))
            
        except Exception as e:
            results['errors'].append(f"API test error: {e}")
        
        return results
    
    def create_test_transcript(self, output_file: Path, language: str = "en") -> bool:
        """
        Create a test transcript file for testing dialogue functionality
        
        Args:
            output_file: Path to save test transcript
            language: Language for test content
            
        Returns:
            True if successful
        """
        try:
            if language == "fa":
                # Persian test content
                test_content = """speaker_0 [0.0s - 3.5s]: سلام، چطوری؟ امیدوارم حالت خوب باشه.
speaker_1 [3.5s - 6.2s]: سلام! ممنون، من خوبم. تو چطوری؟
speaker_0 [6.2s - 10.1s]: منم خوبم، ممنون. امروز کار زیادی داریم؟
speaker_1 [10.1s - 13.8s]: آره، چند تا پروژه مهم داریم که باید تموم کنیم.
speaker_0 [13.8s - 17.2s]: باشه، پس بهتره شروع کنیم."""
            else:
                # English test content
                test_content = """speaker_0 [0.0s - 3.5s]: Hello, how are you doing today?
speaker_1 [3.5s - 6.2s]: Hi there! I'm doing well, thank you. How about you?
speaker_0 [6.2s - 10.1s]: I'm great, thanks for asking. Do we have much work today?
speaker_1 [10.1s - 13.8s]: Yes, we have several important projects to complete.
speaker_0 [13.8s - 17.2s]: Alright, we should probably get started then."""
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(test_content)
            
            logger.info("Test transcript created: %s", output_file)
            return True
            
        except Exception as e:
            logger.error("Failed to create test transcript: %s", e)
            return False
    
    def run_comprehensive_test(
        self, 
        transcript_file: Optional[Path] = None,
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Run comprehensive dialogue functionality test
        
        Args:
            transcript_file: Optional transcript file to test with
            language: Language for testing
            
        Returns:
            Comprehensive test results
        """
        results = {
            'test_timestamp': str(Path().absolute()),
            'language': language,
            'transcript_parsing': {},
            'voice_availability': {},
            'api_availability': {},
            'dialogue_conversion': {},
            'overall_status': 'unknown',
            'recommendations': []
        }
        
        # Create test transcript if none provided
        if transcript_file is None:
            transcript_file = Path(tempfile.mktemp(suffix='.txt'))
            if self.create_test_transcript(transcript_file, language):
                results['test_transcript_created'] = str(transcript_file)
            else:
                results['overall_status'] = 'failed'
                results['recommendations'].append("Could not create test transcript")
                return results
        
        # Test 1: Transcript parsing
        print("🔍 Testing transcript parsing...")
        results['transcript_parsing'] = self.test_transcript_parsing(transcript_file)
        
        # Test 2: Voice availability
        print("🗣️  Testing voice availability...")
        results['voice_availability'] = self.test_voice_availability(language)
        
        # Test 3: API availability
        print("🔗 Testing dialogue API...")
        results['api_availability'] = self.test_dialogue_api_availability()
        
        # Test 4: Dialogue conversion (if possible)
        if (results['transcript_parsing']['parsing_success'] and 
            results['voice_availability']['voice_manager_available']):
            
            print("🎭 Testing dialogue conversion...")
            try:
                converter = TranscriptToDialogueConverter(self.voice_manager)
                dialogue = converter.convert_transcript_to_dialogue(
                    transcript_source=str(transcript_file),
                    language=language,
                    auto_assign=True
                )
                
                results['dialogue_conversion'] = {
                    'conversion_success': True,
                    'segments_converted': len(dialogue.segments),
                    'voices_assigned': len([seg for seg in dialogue.segments if seg.voice_id]),
                    'missing_voices': dialogue.validate_voice_ids()
                }
                
            except Exception as e:
                results['dialogue_conversion'] = {
                    'conversion_success': False,
                    'error': str(e)
                }
        
        # Determine overall status
        parsing_ok = results['transcript_parsing']['parsing_success']
        voices_ok = results['voice_availability']['voice_manager_available']
        api_ok = results['api_availability']['api_available']
        
        if parsing_ok and voices_ok and api_ok:
            results['overall_status'] = 'excellent'
            results['recommendations'].append("All systems operational - dialogue generation ready")
        elif parsing_ok and voices_ok:
            results['overall_status'] = 'good'
            results['recommendations'].append("Dialogue generation available with fallback method")
            if not api_ok:
                results['recommendations'].append("Text to Dialogue API not available - will use individual TTS calls")
        elif parsing_ok:
            results['overall_status'] = 'limited'
            results['recommendations'].append("Basic transcript parsing works - check API key for voice features")
        else:
            results['overall_status'] = 'failed'
            results['recommendations'].append("Multiple issues detected - check configuration and API access")
        
        # Cleanup test file if we created it
        if 'test_transcript_created' in results:
            try:
                Path(results['test_transcript_created']).unlink()
            except Exception:
                pass
        
        return results
    
    def generate_test_report(self, test_results: Dict[str, Any]) -> str:
        """
        Generate a formatted test report
        
        Args:
            test_results: Results from run_comprehensive_test
            
        Returns:
            Formatted test report string
        """
        report = []
        report.append("=" * 60)
        report.append("DuoSynco Dialogue Functionality Test Report")
        report.append("=" * 60)
        report.append(f"Test Language: {test_results['language']}")
        report.append(f"Overall Status: {test_results['overall_status'].upper()}")
        report.append("")
        
        # Transcript parsing
        parsing = test_results['transcript_parsing']
        report.append("📄 Transcript Parsing:")
        report.append(f"  ✅ Success: {parsing['parsing_success']}")
        if parsing['parsing_success']:
            report.append(f"  📊 Segments: {parsing['segments_found']}")
            report.append(f"  👥 Speakers: {len(parsing['speakers_found'])}")
            report.append(f"  📝 Format: {parsing['format_detected']}")
        else:
            for error in parsing.get('errors', []):
                report.append(f"  ❌ {error}")
        report.append("")
        
        # Voice availability
        voices = test_results['voice_availability']
        report.append("🗣️  Voice Availability:")
        report.append(f"  ✅ Manager Available: {voices['voice_manager_available']}")
        if voices['voice_manager_available']:
            report.append(f"  📊 Total Voices: {voices['total_voices']}")
            report.append(f"  🌍 Language Voices: {voices['language_voices']}")
            report.append(f"  👨 Male Voices: {voices['male_voices']}")
            report.append(f"  👩 Female Voices: {voices['female_voices']}")
        else:
            for error in voices.get('errors', []):
                report.append(f"  ❌ {error}")
        report.append("")
        
        # API availability
        api = test_results['api_availability']
        report.append("🔗 Dialogue API:")
        report.append(f"  ✅ Generator Available: {api['generator_available']}")
        report.append(f"  🔌 API Available: {api['api_available']}")
        if api.get('response_time'):
            report.append(f"  ⏱️  Response Time: {api['response_time']}s")
        if not api['api_available']:
            for error in api.get('errors', []):
                report.append(f"  ❌ {error}")
        report.append("")
        
        # Dialogue conversion
        if 'dialogue_conversion' in test_results:
            conv = test_results['dialogue_conversion']
            report.append("🎭 Dialogue Conversion:")
            report.append(f"  ✅ Success: {conv['conversion_success']}")
            if conv['conversion_success']:
                report.append(f"  📊 Segments: {conv['segments_converted']}")
                report.append(f"  🗣️  Voices Assigned: {conv['voices_assigned']}")
                if conv['missing_voices']:
                    report.append(f"  ⚠️  Missing Voices: {len(conv['missing_voices'])}")
            else:
                report.append(f"  ❌ Error: {conv.get('error', 'Unknown')}")
            report.append("")
        
        # Recommendations
        report.append("💡 Recommendations:")
        for rec in test_results['recommendations']:
            report.append(f"  • {rec}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)