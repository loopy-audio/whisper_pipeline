#!/usr/bin/env python3
"""
Demo script showing the basic functionality of the WhisperX pipeline components.
This demonstrates the core concepts without requiring actual audio processing dependencies.
"""

import sys
from pathlib import Path

# Add the parent directory to the path to import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from whisper_pipeline.config import PipelineConfig, SpeakerPosition, get_default_config
from whisper_pipeline.utils import (
    _format_time, _format_srt_time, calculate_3d_distance, get_file_extension
)


def demo_configuration():
    """Demonstrate configuration management."""
    print("🔧 Configuration Demo")
    print("=" * 30)
    
    # Create default configuration
    config = get_default_config()
    
    print(f"Transcription model: {config.transcription.model_name}")
    print(f"Diarization model: {config.diarization.model_name}")
    print(f"Sample rate: {config.spatialization.sample_rate} Hz")
    print(f"Output directory: {config.output.output_dir}")
    
    # Show speaker positions
    print("\nSpeaker positions:")
    for speaker, pos in config.speaker_positions.items():
        print(f"  {speaker}: ({pos.x}, {pos.y}, {pos.z}) meters")
    
    # Demonstrate position modification
    config.speaker_positions["SPEAKER_02"] = SpeakerPosition(x=0.0, y=3.0, z=0.5)
    print(f"  Added SPEAKER_02: (0.0, 3.0, 0.5) meters")
    
    print()


def demo_utility_functions():
    """Demonstrate utility functions."""
    print("🛠️  Utility Functions Demo")
    print("=" * 30)
    
    # Time formatting
    times = [30.5, 65.2, 3661.75]
    for t in times:
        formatted = _format_time(t)
        srt_formatted = _format_srt_time(t)
        print(f"Time {t}s -> Regular: {formatted}, SRT: {srt_formatted}")
    
    # 3D distance calculation
    pos1 = (0, 0, 0)  # Listener position
    pos2 = (3, 4, 0)  # Speaker position
    distance = calculate_3d_distance(pos1, pos2)
    print(f"\nDistance from {pos1} to {pos2}: {distance:.2f} meters")
    
    # File extensions
    formats = ['wav', 'mp3', 'json', 'srt', 'txt']
    print(f"\nFile extensions:")
    for fmt in formats:
        ext = get_file_extension(fmt)
        print(f"  {fmt} -> {ext}")
    
    print()


def demo_pipeline_concept():
    """Demonstrate the pipeline processing concept."""
    print("🎙️  Pipeline Processing Concept Demo")
    print("=" * 40)
    
    # Simulate a conversation
    transcript_segments = [
        {"start": 0.0, "end": 2.5, "text": "Hello, how are you doing today?"},
        {"start": 3.0, "end": 5.2, "text": "I'm doing great, thanks for asking!"},
        {"start": 5.8, "end": 8.1, "text": "That's wonderful to hear."},
        {"start": 8.5, "end": 11.3, "text": "Yes, it's been a really good day so far."},
    ]
    
    # Simulate diarization results
    speaker_segments = [
        {"start": 0.0, "end": 2.5, "speaker": "SPEAKER_00"},
        {"start": 3.0, "end": 5.2, "speaker": "SPEAKER_01"},
        {"start": 5.8, "end": 8.1, "speaker": "SPEAKER_00"},
        {"start": 8.5, "end": 11.3, "speaker": "SPEAKER_01"},
    ]
    
    # Simulate merging transcript with speakers
    print("Step 1: Transcription")
    for i, seg in enumerate(transcript_segments):
        start_str = _format_time(seg["start"])
        print(f"  [{start_str}] {seg['text']}")
    
    print("\nStep 2: Speaker Diarization")
    for seg in speaker_segments:
        start_str = _format_time(seg["start"])
        end_str = _format_time(seg["end"])
        duration = seg["end"] - seg["start"]
        print(f"  [{start_str}-{end_str}] {seg['speaker']} ({duration:.1f}s)")
    
    print("\nStep 3: Merged Transcript with Speakers")
    for i, transcript_seg in enumerate(transcript_segments):
        speaker_seg = speaker_segments[i]  # Simplified matching
        start_str = _format_time(transcript_seg["start"])
        print(f"  [{start_str}] {speaker_seg['speaker']}: {transcript_seg['text']}")
    
    print("\nStep 4: 3D Spatialization")
    # Show speaker positions
    speaker_positions = {
        "SPEAKER_00": SpeakerPosition(x=-1.5, y=2.0, z=0.0),
        "SPEAKER_01": SpeakerPosition(x=1.5, y=2.0, z=0.0)
    }
    
    for speaker, pos in speaker_positions.items():
        distance = calculate_3d_distance((0, 0, 0), (pos.x, pos.y, pos.z))
        print(f"  {speaker} positioned at ({pos.x}, {pos.y}, {pos.z}) - {distance:.2f}m from listener")
    
    print("  -> Binaural audio created with left/right channel differences")
    print()


def demo_cli_usage():
    """Show CLI usage examples."""
    print("💻 CLI Usage Examples")
    print("=" * 25)
    
    examples = [
        "# Install dependencies",
        "pip install -r requirements.txt",
        "",
        "# Process complete pipeline",
        "whisper-pipeline process audio.wav",
        "",
        "# Transcription only",
        "whisper-pipeline transcribe audio.wav --output transcript.json",
        "",
        "# Diarization only", 
        "whisper-pipeline diarize audio.wav --num-speakers 2",
        "",
        "# Create configuration file",
        "whisper-pipeline init-config --output my_config.yaml",
        "",
        "# Set speaker positions",
        "whisper-pipeline set-speaker-position my_config.yaml --speaker SPEAKER_00 --x -1.5 --y 2.0 --z 0.0",
        "",
        "# Process with custom config",
        "whisper-pipeline process audio.wav --config my_config.yaml --output-dir results/",
    ]
    
    for example in examples:
        if example.startswith("#"):
            print(f"\033[92m{example}\033[0m")  # Green comments
        elif example == "":
            print()
        else:
            print(f"  {example}")
    
    print()


def main():
    """Run all demos."""
    print("🎧 WhisperX Pipeline - Functionality Demo")
    print("=" * 50)
    print("This demo shows the core concepts and functionality")
    print("without requiring the full dependency stack.\n")
    
    demo_configuration()
    demo_utility_functions()  
    demo_pipeline_concept()
    demo_cli_usage()
    
    print("✨ Demo completed!")
    print("\nTo use the full pipeline:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run: whisper-pipeline process your_audio.wav")
    print("3. Enjoy 3D binaural audio! 🎧")


if __name__ == "__main__":
    main()