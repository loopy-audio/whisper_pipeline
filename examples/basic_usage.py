#!/usr/bin/env python3
"""
Example script demonstrating the WhisperX pipeline usage.

This script shows how to use the pipeline programmatically
for transcribing, diarizing, and spatializing audio.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path to import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from whisper_pipeline import WhisperPipeline, PipelineConfig
from whisper_pipeline.config import SpeakerPosition


def main():
    """Demonstrate basic pipeline usage."""
    
    # Example audio file path (you would replace this with your actual audio file)
    audio_path = "example_audio.wav"
    
    if not os.path.exists(audio_path):
        print(f"Example audio file not found: {audio_path}")
        print("Please provide a valid audio file path to test the pipeline.")
        print("\nExample usage:")
        print("  python examples/basic_usage.py /path/to/your/audio.wav")
        if len(sys.argv) > 1:
            audio_path = sys.argv[1]
            if not os.path.exists(audio_path):
                print(f"Audio file not found: {audio_path}")
                return
        else:
            return
    
    print("WhisperX Pipeline Example")
    print("=" * 40)
    print(f"Processing audio file: {audio_path}")
    
    try:
        # Create a custom configuration
        config = PipelineConfig()
        
        # Configure transcription
        config.transcription.model_name = "base"  # Use smaller model for faster processing
        config.transcription.language = None  # Auto-detect language
        
        # Configure diarization
        config.diarization.num_speakers = None  # Auto-detect number of speakers
        config.diarization.min_speakers = 1
        config.diarization.max_speakers = 4
        
        # Configure output
        config.output.output_dir = "output"
        config.output.save_transcript = True
        config.output.save_diarization = True
        config.output.save_spatialized_audio = True
        config.output.transcript_format = "json"
        config.output.audio_format = "wav"
        
        # Set up speaker positions manually
        # Imagine a conversation between 2 people sitting across from each other
        config.speaker_positions = {
            "SPEAKER_00": SpeakerPosition(x=-1.0, y=2.0, z=0.0),  # Left speaker
            "SPEAKER_01": SpeakerPosition(x=1.0, y=2.0, z=0.0),   # Right speaker
        }
        
        # Enable verbose output
        config.verbose = True
        
        # Create and run pipeline
        pipeline = WhisperPipeline(config, verbose=True)
        
        print("\n🎙️  Starting pipeline processing...")
        result = pipeline.process_audio(audio_path)
        
        # Display results
        print("\n" + "="*50)
        print("✅ PROCESSING COMPLETE")
        print("="*50)
        
        print(f"\n📊 Results Summary:")
        print(f"   • Audio duration: {result['pipeline_config']['total_duration']:.2f} seconds")
        print(f"   • Processing time: {result['processing_time']:.2f} seconds")
        print(f"   • Speakers detected: {result['pipeline_config']['num_speakers_detected']}")
        
        if result.get('output_paths'):
            print(f"\n📁 Output files:")
            for file_type, path in result['output_paths'].items():
                print(f"   • {file_type}: {path}")
        
        # Show transcript preview
        print(f"\n📝 Transcript preview:")
        segments = result['merged_transcript'].get('segments', [])
        for i, segment in enumerate(segments[:5]):  # Show first 5 segments
            speaker = segment.get('speaker', 'Unknown')
            text = segment.get('text', '').strip()
            start = segment.get('start', 0)
            print(f"   [{start:6.1f}s] {speaker}: {text}")
        
        if len(segments) > 5:
            print(f"   ... and {len(segments) - 5} more segments")
        
        # Show speaker statistics
        if result['diarization'].get('speaker_stats'):
            print(f"\n👥 Speaker statistics:")
            for speaker, stats in result['diarization']['speaker_stats'].items():
                print(f"   • {speaker}: {stats['total_duration']:.1f}s total, "
                      f"{stats['num_segments']} segments")
        
        print(f"\n🎧 Binaural audio has been created with 3D spatialization!")
        print(f"   Listen with headphones for the full 3D effect.")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def demonstrate_individual_steps():
    """Demonstrate using individual pipeline steps."""
    
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "example_audio.wav"
    
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return
    
    print("\nDemonstrating individual pipeline steps...")
    print("=" * 50)
    
    try:
        config = PipelineConfig()
        config.transcription.model_name = "base"
        config.verbose = True
        
        pipeline = WhisperPipeline(config)
        
        # Step 1: Transcription only
        print("\n🎤 Step 1: Transcription only...")
        transcript = pipeline.transcribe_only(audio_path)
        print(f"   Transcribed {len(transcript.get('segments', []))} segments")
        
        # Step 2: Diarization only
        print("\n👥 Step 2: Diarization only...")
        diarization = pipeline.diarize_only(audio_path)
        print(f"   Detected {diarization['num_speakers']} speakers")
        
        print(f"\n✅ Individual steps completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error in individual steps: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--individual":
        demonstrate_individual_steps()
    else:
        sys.exit(main())