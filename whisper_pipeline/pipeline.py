"""Main pipeline orchestrator for WhisperX audio processing."""

import os
import time
from typing import Dict, Optional, Tuple
import logging

from .config import PipelineConfig, SpeakerPosition
from .transcription import TranscriptionProcessor
from .diarization import DiarizationProcessor
from .spatialization import SpatializationProcessor
from .utils import (
    setup_logging, save_transcript, save_audio, 
    merge_transcript_with_diarization, validate_audio_file,
    ensure_output_directory, get_file_extension
)


class WhisperPipeline:
    """
    End-to-end pipeline for transcribing, diarizing, and spatializing audio.
    
    This class orchestrates the complete workflow:
    1. Audio transcription using WhisperX
    2. Speaker diarization using pyannote.audio
    3. Binaural spatialization for 3D audio rendering
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None, verbose: bool = False):
        """
        Initialize the WhisperX pipeline.
        
        Args:
            config: Pipeline configuration. If None, uses default config.
            verbose: Enable verbose logging
        """
        self.config = config or PipelineConfig()
        self.verbose = verbose or self.config.verbose
        
        # Set up logging
        self.logger = setup_logging(self.verbose)
        
        # Initialize processors
        self.transcription_processor = None
        self.diarization_processor = None
        self.spatialization_processor = None
        
        # Results storage
        self.transcription_result = None
        self.diarization_result = None
        self.merged_result = None
        self.binaural_audio = None
        
        if self.verbose:
            self.logger.info("WhisperX Pipeline initialized")
    
    def _initialize_processors(self) -> None:
        """Initialize all processing modules."""
        if self.verbose:
            self.logger.info("Initializing processors...")
        
        try:
            # Initialize transcription processor
            self.transcription_processor = TranscriptionProcessor(
                self.config.transcription, self.verbose
            )
            
            # Initialize diarization processor
            self.diarization_processor = DiarizationProcessor(
                self.config.diarization, self.verbose
            )
            
            # Initialize spatialization processor
            self.spatialization_processor = SpatializationProcessor(
                self.config.spatialization, self.verbose
            )
            
            if self.verbose:
                self.logger.info("All processors initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize processors: {e}")
            raise
    
    def process_audio(self, audio_path: str, output_dir: Optional[str] = None) -> Dict:
        """
        Process audio file through the complete pipeline.
        
        Args:
            audio_path: Path to input audio file
            output_dir: Output directory (overrides config if provided)
            
        Returns:
            Dictionary containing all results
        """
        if not validate_audio_file(audio_path):
            raise ValueError(f"Invalid audio file: {audio_path}")
        
        # Update output directory if provided
        if output_dir:
            self.config.output.output_dir = output_dir
        
        # Ensure output directory exists
        ensure_output_directory(os.path.join(self.config.output.output_dir, "placeholder"))
        
        start_time = time.time()
        
        if self.verbose:
            self.logger.info(f"Starting pipeline processing for: {audio_path}")
        
        # Initialize processors if not done already
        if self.transcription_processor is None:
            self._initialize_processors()
        
        try:
            # Step 1: Transcription
            self.logger.info("Step 1: Transcribing audio...")
            self.transcription_result = self.transcription_processor.transcribe(audio_path)
            
            # Step 2: Diarization
            self.logger.info("Step 2: Performing speaker diarization...")
            self.diarization_result = self.diarization_processor.diarize(audio_path)
            
            # Apply post-processing to diarization
            self.diarization_result = self.diarization_processor.merge_close_segments(
                self.diarization_result, gap_threshold=0.5
            )
            self.diarization_result = self.diarization_processor.filter_short_segments(
                self.diarization_result, min_duration=1.0
            )
            
            # Step 3: Merge transcription with diarization
            self.logger.info("Step 3: Merging transcription with diarization...")
            self.merged_result = merge_transcript_with_diarization(
                self.transcription_result, self.diarization_result
            )
            
            # Step 4: Spatialization
            self.logger.info("Step 4: Creating binaural spatialized audio...")
            
            # Ensure we have speaker positions
            self._ensure_speaker_positions()
            
            left_channel, right_channel = self.spatialization_processor.create_binaural_mix(
                self.merged_result, audio_path, self.config.speaker_positions
            )
            
            self.binaural_audio = (left_channel, right_channel)
            
            # Step 5: Save results
            self.logger.info("Step 5: Saving results...")
            output_paths = self._save_results(audio_path)
            
            # Create summary
            processing_time = time.time() - start_time
            
            result_summary = {
                "audio_path": audio_path,
                "processing_time": processing_time,
                "transcription": self.transcription_result,
                "diarization": self.diarization_result,
                "merged_transcript": self.merged_result,
                "output_paths": output_paths,
                "pipeline_config": {
                    "transcription_model": self.config.transcription.model_name,
                    "diarization_model": self.config.diarization.model_name,
                    "num_speakers_detected": self.diarization_result.get("num_speakers", 0),
                    "total_duration": self.transcription_result.get("audio_duration", 0),
                    "spatialization_enabled": True
                }
            }
            
            if self.verbose:
                self.logger.info(f"Pipeline completed in {processing_time:.2f} seconds")
                self.logger.info(f"Detected {result_summary['pipeline_config']['num_speakers_detected']} speakers")
                self.logger.info(f"Audio duration: {result_summary['pipeline_config']['total_duration']:.2f} seconds")
            
            return result_summary
            
        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {e}")
            raise
    
    def _ensure_speaker_positions(self) -> None:
        """Ensure all detected speakers have positions assigned."""
        if not self.diarization_result:
            return
        
        detected_speakers = self.diarization_result.get("speakers", [])
        
        # Check if any speakers need position assignment
        speakers_without_positions = [
            speaker for speaker in detected_speakers
            if speaker not in self.config.speaker_positions
        ]
        
        if speakers_without_positions:
            if self.verbose:
                self.logger.info(f"Assigning default positions to speakers: {speakers_without_positions}")
            
            # Assign default positions in a semicircle
            for i, speaker in enumerate(speakers_without_positions):
                # Arrange speakers in a semicircle in front of listener
                num_speakers = len(speakers_without_positions)
                if num_speakers == 1:
                    angle = 0  # Center
                else:
                    angle_step = 60 / (num_speakers - 1)  # Spread over 60 degrees
                    angle = -30 + (i * angle_step)  # Start from -30 degrees
                
                # Convert to cartesian coordinates (2 meters away)
                import math
                distance = 2.0
                x = distance * math.sin(math.radians(angle))
                y = distance * math.cos(math.radians(angle))
                z = 0.0
                
                self.config.speaker_positions[speaker] = SpeakerPosition(x=x, y=y, z=z)
                
                if self.verbose:
                    self.logger.debug(f"Speaker {speaker} positioned at ({x:.2f}, {y:.2f}, {z:.2f})")
    
    def _save_results(self, audio_path: str) -> Dict[str, str]:
        """Save all results to files."""
        output_paths = {}
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_dir = self.config.output.output_dir
        
        try:
            # Save transcript
            if self.config.output.save_transcript and self.merged_result:
                transcript_ext = get_file_extension(self.config.output.transcript_format)
                transcript_path = os.path.join(output_dir, f"{base_name}_transcript{transcript_ext}")
                
                save_transcript(
                    self.merged_result, 
                    transcript_path, 
                    self.config.output.transcript_format
                )
                output_paths["transcript"] = transcript_path
                
                if self.verbose:
                    self.logger.info(f"Transcript saved: {transcript_path}")
            
            # Save diarization results
            if self.config.output.save_diarization and self.diarization_result:
                diarization_path = os.path.join(output_dir, f"{base_name}_diarization.json")
                save_transcript(self.diarization_result, diarization_path, "json")
                output_paths["diarization"] = diarization_path
                
                if self.verbose:
                    self.logger.info(f"Diarization saved: {diarization_path}")
            
            # Save spatialized audio
            if self.config.output.save_spatialized_audio and self.binaural_audio:
                left_channel, right_channel = self.binaural_audio
                
                # Create stereo audio
                stereo_audio = np.column_stack([left_channel, right_channel])
                
                audio_ext = get_file_extension(self.config.output.audio_format)
                audio_path_out = os.path.join(output_dir, f"{base_name}_binaural{audio_ext}")
                
                save_audio(
                    stereo_audio,
                    audio_path_out,
                    self.config.output.sample_rate,
                    self.config.output.bit_depth
                )
                output_paths["binaural_audio"] = audio_path_out
                
                if self.verbose:
                    self.logger.info(f"Binaural audio saved: {audio_path_out}")
            
            return output_paths
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            raise
    
    def transcribe_only(self, audio_path: str) -> Dict:
        """
        Run only the transcription step.
        
        Args:
            audio_path: Path to input audio file
            
        Returns:
            Transcription result
        """
        if not validate_audio_file(audio_path):
            raise ValueError(f"Invalid audio file: {audio_path}")
        
        if self.transcription_processor is None:
            self.transcription_processor = TranscriptionProcessor(
                self.config.transcription, self.verbose
            )
        
        return self.transcription_processor.transcribe(audio_path)
    
    def diarize_only(self, audio_path: str) -> Dict:
        """
        Run only the diarization step.
        
        Args:
            audio_path: Path to input audio file
            
        Returns:
            Diarization result
        """
        if not validate_audio_file(audio_path):
            raise ValueError(f"Invalid audio file: {audio_path}")
        
        if self.diarization_processor is None:
            self.diarization_processor = DiarizationProcessor(
                self.config.diarization, self.verbose
            )
        
        result = self.diarization_processor.diarize(audio_path)
        
        # Apply post-processing
        result = self.diarization_processor.merge_close_segments(result)
        result = self.diarization_processor.filter_short_segments(result)
        
        return result
    
    def get_pipeline_info(self) -> Dict:
        """Get information about the pipeline and its components."""
        info = {
            "pipeline_version": "0.1.0",
            "config": {
                "transcription": {
                    "model_name": self.config.transcription.model_name,
                    "device": self.config.transcription.device,
                    "batch_size": self.config.transcription.batch_size,
                },
                "diarization": {
                    "model_name": self.config.diarization.model_name,
                    "min_speakers": self.config.diarization.min_speakers,
                    "max_speakers": self.config.diarization.max_speakers,
                },
                "spatialization": {
                    "sample_rate": self.config.spatialization.sample_rate,
                    "room_size": self.config.spatialization.room_size,
                    "reverb_amount": self.config.spatialization.reverb_amount,
                },
                "output": {
                    "output_dir": self.config.output.output_dir,
                    "transcript_format": self.config.output.transcript_format,
                    "audio_format": self.config.output.audio_format,
                }
            },
            "speaker_positions": {
                speaker: {"x": pos.x, "y": pos.y, "z": pos.z}
                for speaker, pos in self.config.speaker_positions.items()
            }
        }
        
        # Add processor-specific info if initialized
        if self.transcription_processor:
            info["transcription_processor"] = self.transcription_processor.get_model_info()
        
        if self.diarization_processor:
            info["diarization_processor"] = self.diarization_processor.get_pipeline_info()
        
        if self.spatialization_processor:
            info["spatialization_processor"] = self.spatialization_processor.get_processor_info()
        
        return info