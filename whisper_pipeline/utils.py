"""Utility functions for the WhisperX pipeline."""

import os
import json
from typing import Dict, List, Tuple, Union, Optional, Any
import logging

try:
    import numpy as np
    import soundfile as sf
    import librosa
    AUDIO_DEPS_AVAILABLE = True
except ImportError:
    AUDIO_DEPS_AVAILABLE = False
    np = None
    sf = None
    librosa = None


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging for the pipeline."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('whisper_pipeline')


def load_audio(audio_path: str, target_sr: int = 16000) -> Tuple[Any, int]:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    if not AUDIO_DEPS_AVAILABLE:
        raise ImportError("Audio dependencies not available. Please install: pip install librosa soundfile numpy")
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Load audio with librosa for better format support
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    
    return audio, sr


def save_audio(audio: Any, output_path: str, sample_rate: int = 44100, 
               bit_depth: int = 16) -> None:
    """
    Save audio to file.
    
    Args:
        audio: Audio data as numpy array
        output_path: Output file path
        sample_rate: Sample rate
        bit_depth: Bit depth (16 or 24)
    """
    if not AUDIO_DEPS_AVAILABLE:
        raise ImportError("Audio dependencies not available. Please install: pip install soundfile numpy")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Normalize audio to prevent clipping
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    # Normalize to [-1, 1] range
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / np.max(np.abs(audio))
    
    # Determine subtype based on bit depth
    subtype = f'PCM_{bit_depth}' if output_path.lower().endswith('.wav') else None
    
    sf.write(output_path, audio, sample_rate, subtype=subtype)


def save_transcript(transcript: Dict, output_path: str, format_type: str = "json") -> None:
    """
    Save transcript to file in specified format.
    
    Args:
        transcript: Transcript data
        output_path: Output file path
        format_type: Format type (json, srt, txt)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if format_type.lower() == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)
    
    elif format_type.lower() == "srt":
        _save_transcript_srt(transcript, output_path)
    
    elif format_type.lower() == "txt":
        _save_transcript_txt(transcript, output_path)
    
    else:
        raise ValueError(f"Unsupported transcript format: {format_type}")


def _save_transcript_srt(transcript: Dict, output_path: str) -> None:
    """Save transcript in SRT subtitle format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(transcript.get('segments', []), 1):
            start_time = _format_srt_time(segment['start'])
            end_time = _format_srt_time(segment['end'])
            text = segment['text'].strip()
            
            # Include speaker information if available
            if 'speaker' in segment:
                text = f"[{segment['speaker']}] {text}"
            
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n\n")


def _save_transcript_txt(transcript: Dict, output_path: str) -> None:
    """Save transcript in plain text format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for segment in transcript.get('segments', []):
            text = segment['text'].strip()
            
            # Include timestamp and speaker information if available
            start_time = _format_time(segment['start'])
            line = f"[{start_time}]"
            
            if 'speaker' in segment:
                line += f" {segment['speaker']}:"
            
            line += f" {text}\n"
            f.write(line)


def _format_srt_time(seconds: float) -> str:
    """Format time in SRT format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"


def _format_time(seconds: float) -> str:
    """Format time in HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def merge_transcript_with_diarization(transcript: Dict, diarization_result: Dict) -> Dict:
    """
    Merge transcript with diarization results.
    
    Args:
        transcript: WhisperX transcript result
        diarization_result: Speaker diarization result
        
    Returns:
        Merged transcript with speaker information
    """
    # Create a copy of the transcript
    merged = transcript.copy()
    
    # Map segments to speakers based on overlap
    for segment in merged.get('segments', []):
        segment_start = segment['start']
        segment_end = segment['end']
        segment_mid = (segment_start + segment_end) / 2
        
        # Find the speaker with the most overlap
        best_speaker = None
        best_overlap = 0
        
        for speaker_segment in diarization_result.get('segments', []):
            speaker = speaker_segment['speaker']
            speaker_start = speaker_segment['start']
            speaker_end = speaker_segment['end']
            
            # Calculate overlap
            overlap_start = max(segment_start, speaker_start)
            overlap_end = min(segment_end, speaker_end)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
        
        # Assign speaker if we found a good match
        if best_speaker and best_overlap > 0:
            segment['speaker'] = best_speaker
    
    return merged


def calculate_3d_distance(pos1: Tuple[float, float, float], 
                         pos2: Tuple[float, float, float]) -> float:
    """Calculate 3D Euclidean distance between two positions."""
    if AUDIO_DEPS_AVAILABLE:
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))
    else:
        # Fallback implementation without numpy
        return (sum((a - b) ** 2 for a, b in zip(pos1, pos2))) ** 0.5


def apply_distance_attenuation(audio: Any, distance: float, 
                              max_distance: float = 10.0, 
                              model: str = "linear") -> Any:
    """
    Apply distance-based attenuation to audio.
    
    Args:
        audio: Input audio
        distance: Distance in meters
        max_distance: Maximum distance for attenuation
        model: Attenuation model ('linear' or 'exponential')
        
    Returns:
        Attenuated audio
    """
    if not AUDIO_DEPS_AVAILABLE:
        raise ImportError("Audio dependencies not available. Please install: pip install numpy")
    
    if distance <= 0:
        return audio
    
    if model == "linear":
        attenuation = max(0, 1 - (distance / max_distance))
    elif model == "exponential":
        attenuation = np.exp(-distance / max_distance)
    else:
        raise ValueError(f"Unknown attenuation model: {model}")
    
    return audio * attenuation


def validate_audio_file(audio_path: str) -> bool:
    """
    Validate that an audio file exists and is readable.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(audio_path):
        return False
    
    try:
        # Try to read the first few samples
        info = sf.info(audio_path)
        return info.frames > 0
    except:
        return False


def ensure_output_directory(output_path: str) -> None:
    """Ensure output directory exists."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)


def get_file_extension(format_type: str) -> str:
    """Get file extension for given format type."""
    extensions = {
        'wav': '.wav',
        'mp3': '.mp3',
        'flac': '.flac',
        'json': '.json',
        'srt': '.srt',
        'txt': '.txt'
    }
    return extensions.get(format_type.lower(), '.wav')