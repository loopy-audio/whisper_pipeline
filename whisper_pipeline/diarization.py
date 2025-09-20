"""Speaker diarization module using pyannote.audio."""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Union
import logging

try:
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
    from pyannote.core import Annotation, Segment
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    Pipeline = None
    PretrainedSpeakerEmbedding = None
    Annotation = None
    Segment = None

from .config import DiarizationConfig
from .utils import load_audio


class DiarizationProcessor:
    """Handles speaker diarization using pyannote.audio."""
    
    def __init__(self, config: DiarizationConfig, verbose: bool = False):
        """
        Initialize the diarization processor.
        
        Args:
            config: Diarization configuration
            verbose: Enable verbose logging
        """
        if not PYANNOTE_AVAILABLE:
            raise ImportError(
                "pyannote.audio is not available. Please install it with: "
                "pip install pyannote.audio"
            )
        
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose
        
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if verbose:
            self.logger.info(f"Using device: {self.device}")
        
        # Initialize pipeline
        self.pipeline = None
        self.embedding_model = None
        self._load_pipeline()
    
    def _load_pipeline(self) -> None:
        """Load pyannote diarization pipeline."""
        try:
            if self.verbose:
                self.logger.info(f"Loading diarization pipeline: {self.config.model_name}")
            
            # Load the diarization pipeline
            self.pipeline = Pipeline.from_pretrained(
                self.config.model_name,
                use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
            )
            
            # Move to appropriate device
            if hasattr(self.pipeline, 'to'):
                self.pipeline.to(self.device)
            
            # Load embedding model for speaker verification
            try:
                self.embedding_model = PretrainedSpeakerEmbedding(
                    self.config.embeddings_model,
                    device=self.device
                )
                if self.verbose:
                    self.logger.info("Speaker embedding model loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load embedding model: {e}")
                self.embedding_model = None
            
            if self.verbose:
                self.logger.info("Diarization pipeline loaded successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to load diarization pipeline: {e}")
            raise
    
    def diarize(self, audio_path: str) -> Dict:
        """
        Perform speaker diarization on audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Diarization result dictionary
        """
        if self.pipeline is None:
            raise RuntimeError("Diarization pipeline not loaded")
        
        if self.verbose:
            self.logger.info(f"Starting diarization of: {audio_path}")
        
        try:
            # Configure pipeline parameters
            pipeline_params = {}
            
            if self.config.num_speakers is not None:
                pipeline_params["num_speakers"] = self.config.num_speakers
            else:
                pipeline_params["min_speakers"] = self.config.min_speakers
                pipeline_params["max_speakers"] = self.config.max_speakers
            
            # Apply configuration
            if hasattr(self.pipeline, 'instantiate'):
                self.pipeline.instantiate(pipeline_params)
            
            # Perform diarization
            diarization = self.pipeline(audio_path)
            
            if self.verbose:
                num_speakers = len(diarization.labels())
                total_speech = sum(segment.duration for segment in diarization.itersegments())
                self.logger.info(f"Diarization completed: {num_speakers} speakers, {total_speech:.2f}s total speech")
            
            # Convert to our format
            result = self._convert_diarization_result(diarization, audio_path)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Diarization failed: {e}")
            raise
    
    def _convert_diarization_result(self, diarization: Annotation, audio_path: str) -> Dict:
        """
        Convert pyannote diarization result to our format.
        
        Args:
            diarization: Pyannote diarization result
            audio_path: Path to audio file
            
        Returns:
            Formatted diarization result
        """
        segments = []
        speakers = set()
        
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            segment_dict = {
                "start": segment.start,
                "end": segment.end,
                "duration": segment.duration,
                "speaker": speaker
            }
            segments.append(segment_dict)
            speakers.add(speaker)
        
        # Sort segments by start time
        segments.sort(key=lambda x: x["start"])
        
        # Calculate speaker statistics
        speaker_stats = {}
        for speaker in speakers:
            speaker_segments = [s for s in segments if s["speaker"] == speaker]
            total_duration = sum(s["duration"] for s in speaker_segments)
            speaker_stats[speaker] = {
                "total_duration": total_duration,
                "num_segments": len(speaker_segments),
                "first_appearance": min(s["start"] for s in speaker_segments),
                "last_appearance": max(s["end"] for s in speaker_segments)
            }
        
        # Load audio to get duration
        try:
            audio, sr = load_audio(audio_path)
            audio_duration = len(audio) / sr
        except Exception:
            audio_duration = max(s["end"] for s in segments) if segments else 0.0
        
        result = {
            "audio_path": audio_path,
            "audio_duration": audio_duration,
            "segments": segments,
            "speakers": list(speakers),
            "num_speakers": len(speakers),
            "speaker_stats": speaker_stats,
            "model_name": self.config.model_name,
            "total_speech_duration": sum(s["duration"] for s in segments)
        }
        
        return result
    
    def merge_close_segments(self, diarization_result: Dict, 
                           gap_threshold: float = 0.5) -> Dict:
        """
        Merge segments from the same speaker that are close together.
        
        Args:
            diarization_result: Diarization result
            gap_threshold: Maximum gap in seconds to merge across
            
        Returns:
            Diarization result with merged segments
        """
        if not diarization_result.get("segments"):
            return diarization_result
        
        segments = diarization_result["segments"].copy()
        merged_segments = []
        
        if not segments:
            diarization_result["segments"] = merged_segments
            return diarization_result
        
        current_segment = segments[0].copy()
        
        for segment in segments[1:]:
            # Check if this segment can be merged with current
            if (segment["speaker"] == current_segment["speaker"] and
                segment["start"] - current_segment["end"] <= gap_threshold):
                
                # Merge segments
                current_segment["end"] = segment["end"]
                current_segment["duration"] = current_segment["end"] - current_segment["start"]
                
            else:
                # Add current segment and start new one
                merged_segments.append(current_segment)
                current_segment = segment.copy()
        
        # Add the last segment
        merged_segments.append(current_segment)
        
        # Update result
        result = diarization_result.copy()
        result["segments"] = merged_segments
        
        # Recalculate statistics
        speakers = set(s["speaker"] for s in merged_segments)
        speaker_stats = {}
        for speaker in speakers:
            speaker_segments = [s for s in merged_segments if s["speaker"] == speaker]
            total_duration = sum(s["duration"] for s in speaker_segments)
            speaker_stats[speaker] = {
                "total_duration": total_duration,
                "num_segments": len(speaker_segments),
                "first_appearance": min(s["start"] for s in speaker_segments),
                "last_appearance": max(s["end"] for s in speaker_segments)
            }
        
        result["speaker_stats"] = speaker_stats
        result["total_speech_duration"] = sum(s["duration"] for s in merged_segments)
        
        if self.verbose:
            original_count = len(diarization_result["segments"])
            merged_count = len(merged_segments)
            self.logger.info(f"Merged segments: {original_count} -> {merged_count}")
        
        return result
    
    def filter_short_segments(self, diarization_result: Dict, 
                            min_duration: float = 1.0) -> Dict:
        """
        Filter out segments shorter than minimum duration.
        
        Args:
            diarization_result: Diarization result
            min_duration: Minimum segment duration in seconds
            
        Returns:
            Filtered diarization result
        """
        if not diarization_result.get("segments"):
            return diarization_result
        
        segments = diarization_result["segments"]
        filtered_segments = [s for s in segments if s["duration"] >= min_duration]
        
        if self.verbose:
            original_count = len(segments)
            filtered_count = len(filtered_segments)
            self.logger.info(f"Filtered short segments: {original_count} -> {filtered_count}")
        
        # Update result
        result = diarization_result.copy()
        result["segments"] = filtered_segments
        
        # Recalculate statistics if segments were removed
        if len(filtered_segments) != len(segments):
            speakers = set(s["speaker"] for s in filtered_segments)
            speaker_stats = {}
            for speaker in speakers:
                speaker_segments = [s for s in filtered_segments if s["speaker"] == speaker]
                if speaker_segments:  # Only include speakers that still have segments
                    total_duration = sum(s["duration"] for s in speaker_segments)
                    speaker_stats[speaker] = {
                        "total_duration": total_duration,
                        "num_segments": len(speaker_segments),
                        "first_appearance": min(s["start"] for s in speaker_segments),
                        "last_appearance": max(s["end"] for s in speaker_segments)
                    }
            
            result["speakers"] = list(speakers)
            result["num_speakers"] = len(speakers)
            result["speaker_stats"] = speaker_stats
            result["total_speech_duration"] = sum(s["duration"] for s in filtered_segments)
        
        return result
    
    def get_pipeline_info(self) -> Dict:
        """Get information about the loaded pipeline."""
        return {
            "model_name": self.config.model_name,
            "embeddings_model": self.config.embeddings_model,
            "device": str(self.device),
            "min_speakers": self.config.min_speakers,
            "max_speakers": self.config.max_speakers,
            "num_speakers": self.config.num_speakers,
            "pipeline_loaded": self.pipeline is not None,
            "embedding_model_loaded": self.embedding_model is not None
        }