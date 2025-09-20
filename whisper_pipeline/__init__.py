"""
WhisperX Pipeline: End-to-end audio processing pipeline.

This package provides tools for transcribing, diarizing, and spatializing audio
using WhisperX for transcription, pyannote.audio for speaker diarization,
and binaural rendering for spatialization.
"""

try:
    from .pipeline import WhisperPipeline
    from .config import PipelineConfig
except ImportError:
    # Handle case where dependencies are not yet installed
    WhisperPipeline = None
    PipelineConfig = None

__version__ = "0.1.0"
__all__ = ["WhisperPipeline", "PipelineConfig"]