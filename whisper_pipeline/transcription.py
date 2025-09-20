"""Transcription module using WhisperX."""

import os
import torch
import whisperx
import numpy as np
from typing import Dict, Optional, Union
import logging

from .config import TranscriptionConfig
from .utils import load_audio


class TranscriptionProcessor:
    """Handles audio transcription using WhisperX."""
    
    def __init__(self, config: TranscriptionConfig, verbose: bool = False):
        """
        Initialize the transcription processor.
        
        Args:
            config: Transcription configuration
            verbose: Enable verbose logging
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose
        
        # Determine device
        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device
            
        if verbose:
            self.logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = None
        self.align_model = None
        self.align_metadata = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load WhisperX models."""
        try:
            if self.verbose:
                self.logger.info(f"Loading WhisperX model: {self.config.model_name}")
            
            # Load transcription model
            self.model = whisperx.load_model(
                self.config.model_name,
                device=self.device,
                compute_type=self.config.compute_type
            )
            
            if self.verbose:
                self.logger.info("WhisperX transcription model loaded successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to load WhisperX model: {e}")
            raise
    
    def _load_alignment_model(self, language: str) -> None:
        """Load alignment model for the detected language."""
        try:
            if self.verbose:
                self.logger.info(f"Loading alignment model for language: {language}")
            
            self.align_model, self.align_metadata = whisperx.load_align_model(
                language_code=language,
                device=self.device
            )
            
            if self.verbose:
                self.logger.info("Alignment model loaded successfully")
                
        except Exception as e:
            self.logger.warning(f"Failed to load alignment model for {language}: {e}")
            self.align_model = None
            self.align_metadata = None
    
    def transcribe(self, audio_path: str) -> Dict:
        """
        Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcription result dictionary
        """
        if self.model is None:
            raise RuntimeError("Transcription model not loaded")
        
        if self.verbose:
            self.logger.info(f"Starting transcription of: {audio_path}")
        
        try:
            # Load audio
            audio, sr = load_audio(audio_path, target_sr=16000)
            
            if self.verbose:
                self.logger.info(f"Audio loaded: {len(audio)/sr:.2f}s at {sr}Hz")
            
            # Transcribe
            result = self.model.transcribe(
                audio,
                batch_size=self.config.batch_size,
                language=self.config.language,
                condition_on_previous_text=self.config.condition_on_previous_text,
                temperature=self.config.temperature,
                compression_ratio_threshold=self.config.compression_ratio_threshold,
                logprob_threshold=self.config.logprob_threshold,
                no_speech_threshold=self.config.no_speech_threshold
            )
            
            if self.verbose:
                detected_language = result.get("language", "unknown")
                self.logger.info(f"Transcription completed. Detected language: {detected_language}")
                self.logger.info(f"Number of segments: {len(result.get('segments', []))}")
            
            # Perform word-level alignment if possible
            if result.get("language") and len(result.get("segments", [])) > 0:
                result = self._align_transcription(audio, result)
            
            # Add metadata
            result["audio_path"] = audio_path
            result["audio_duration"] = len(audio) / sr
            result["model_name"] = self.config.model_name
            result["device"] = self.device
            
            return result
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise
    
    def _align_transcription(self, audio: np.ndarray, result: Dict) -> Dict:
        """
        Perform word-level alignment on transcription result.
        
        Args:
            audio: Audio data
            result: Transcription result
            
        Returns:
            Aligned transcription result
        """
        try:
            language = result.get("language")
            if not language:
                if self.verbose:
                    self.logger.warning("No language detected, skipping alignment")
                return result
            
            # Load alignment model if not already loaded or language changed
            if (self.align_model is None or 
                self.align_metadata is None or 
                self.align_metadata.get("language") != language):
                self._load_alignment_model(language)
            
            # Skip alignment if model failed to load
            if self.align_model is None:
                if self.verbose:
                    self.logger.warning("Alignment model not available, skipping alignment")
                return result
            
            if self.verbose:
                self.logger.info("Performing word-level alignment")
            
            # Perform alignment
            aligned_result = whisperx.align(
                result["segments"],
                self.align_model,
                self.align_metadata,
                audio,
                device=self.device,
                return_char_alignments=False
            )
            
            # Update result with aligned segments
            result["segments"] = aligned_result["segments"]
            
            if self.verbose:
                self.logger.info("Word-level alignment completed")
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Alignment failed: {e}")
            return result
    
    def transcribe_with_timestamps(self, audio_path: str) -> Dict:
        """
        Transcribe audio with detailed timestamps.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcription result with word-level timestamps
        """
        result = self.transcribe(audio_path)
        
        # Ensure we have word-level timestamps
        for segment in result.get("segments", []):
            if "words" not in segment:
                # If no word-level alignment, estimate word boundaries
                segment["words"] = self._estimate_word_timestamps(segment)
        
        return result
    
    def _estimate_word_timestamps(self, segment: Dict) -> list:
        """
        Estimate word-level timestamps when alignment is not available.
        
        Args:
            segment: Transcript segment
            
        Returns:
            List of word dictionaries with timestamps
        """
        text = segment.get("text", "").strip()
        if not text:
            return []
        
        words = text.split()
        if not words:
            return []
        
        start_time = segment.get("start", 0.0)
        end_time = segment.get("end", start_time)
        duration = end_time - start_time
        
        # Estimate equal duration per word
        word_duration = duration / len(words) if len(words) > 0 else 0
        
        word_list = []
        for i, word in enumerate(words):
            word_start = start_time + (i * word_duration)
            word_end = start_time + ((i + 1) * word_duration)
            
            word_list.append({
                "word": word,
                "start": word_start,
                "end": word_end,
                "score": 0.5  # Estimated confidence
            })
        
        return word_list
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.config.model_name,
            "device": self.device,
            "compute_type": self.config.compute_type,
            "batch_size": self.config.batch_size,
            "model_loaded": self.model is not None,
            "alignment_model_loaded": self.align_model is not None
        }