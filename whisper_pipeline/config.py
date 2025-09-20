"""Configuration management for the WhisperX pipeline."""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import yaml


@dataclass
class TranscriptionConfig:
    """Configuration for WhisperX transcription."""
    model_name: str = "large-v2"
    device: str = "auto"  # auto, cpu, cuda
    batch_size: int = 16
    compute_type: str = "float16"  # float16, int8, float32
    language: Optional[str] = None  # auto-detect if None
    condition_on_previous_text: bool = False
    temperature: float = 0.0
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6


@dataclass
class DiarizationConfig:
    """Configuration for speaker diarization."""
    model_name: str = "pyannote/speaker-diarization-3.1"
    num_speakers: Optional[int] = None  # auto-detect if None
    min_speakers: int = 1
    max_speakers: int = 10
    clustering_threshold: float = 0.7
    embeddings_model: str = "pyannote/wespeaker-voxceleb-resnet34-LM"


@dataclass
class SpatializationConfig:
    """Configuration for binaural spatialization."""
    hrtf_dataset: str = "default"  # HRTF dataset to use
    sample_rate: int = 44100
    room_size: str = "medium"  # small, medium, large
    reverb_amount: float = 0.3  # 0.0 to 1.0
    distance_model: str = "linear"  # linear, exponential
    max_distance: float = 10.0  # meters
    use_doppler: bool = False


@dataclass
class SpeakerPosition:
    """3D position for a speaker."""
    x: float = 0.0  # left/right (meters)
    y: float = 0.0  # forward/back (meters)
    z: float = 0.0  # up/down (meters)
    
    
@dataclass
class OutputConfig:
    """Configuration for output formats and locations."""
    output_dir: str = "output"
    save_transcript: bool = True
    save_diarization: bool = True
    save_spatialized_audio: bool = True
    transcript_format: str = "json"  # json, srt, txt
    audio_format: str = "wav"  # wav, mp3, flac
    sample_rate: int = 44100
    bit_depth: int = 16


@dataclass
class PipelineConfig:
    """Main configuration class for the WhisperX pipeline."""
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    spatialization: SpatializationConfig = field(default_factory=SpatializationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Speaker positioning (speaker_id -> position)
    speaker_positions: Dict[str, SpeakerPosition] = field(default_factory=dict)
    
    # Global settings
    verbose: bool = False
    use_gpu: bool = True
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PipelineConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        config = cls()
        
        if 'transcription' in data:
            for key, value in data['transcription'].items():
                if hasattr(config.transcription, key):
                    setattr(config.transcription, key, value)
        
        if 'diarization' in data:
            for key, value in data['diarization'].items():
                if hasattr(config.diarization, key):
                    setattr(config.diarization, key, value)
        
        if 'spatialization' in data:
            for key, value in data['spatialization'].items():
                if hasattr(config.spatialization, key):
                    setattr(config.spatialization, key, value)
        
        if 'output' in data:
            for key, value in data['output'].items():
                if hasattr(config.output, key):
                    setattr(config.output, key, value)
        
        if 'speaker_positions' in data:
            config.speaker_positions = {
                speaker_id: SpeakerPosition(**pos)
                for speaker_id, pos in data['speaker_positions'].items()
            }
        
        # Global settings
        if 'verbose' in data:
            config.verbose = data['verbose']
        if 'use_gpu' in data:
            config.use_gpu = data['use_gpu']
            
        return config
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        data = {
            'transcription': {
                'model_name': self.transcription.model_name,
                'device': self.transcription.device,
                'batch_size': self.transcription.batch_size,
                'compute_type': self.transcription.compute_type,
                'language': self.transcription.language,
                'condition_on_previous_text': self.transcription.condition_on_previous_text,
                'temperature': self.transcription.temperature,
                'compression_ratio_threshold': self.transcription.compression_ratio_threshold,
                'logprob_threshold': self.transcription.logprob_threshold,
                'no_speech_threshold': self.transcription.no_speech_threshold,
            },
            'diarization': {
                'model_name': self.diarization.model_name,
                'num_speakers': self.diarization.num_speakers,
                'min_speakers': self.diarization.min_speakers,
                'max_speakers': self.diarization.max_speakers,
                'clustering_threshold': self.diarization.clustering_threshold,
                'embeddings_model': self.diarization.embeddings_model,
            },
            'spatialization': {
                'hrtf_dataset': self.spatialization.hrtf_dataset,
                'sample_rate': self.spatialization.sample_rate,
                'room_size': self.spatialization.room_size,
                'reverb_amount': self.spatialization.reverb_amount,
                'distance_model': self.spatialization.distance_model,
                'max_distance': self.spatialization.max_distance,
                'use_doppler': self.spatialization.use_doppler,
            },
            'output': {
                'output_dir': self.output.output_dir,
                'save_transcript': self.output.save_transcript,
                'save_diarization': self.output.save_diarization,
                'save_spatialized_audio': self.output.save_spatialized_audio,
                'transcript_format': self.output.transcript_format,
                'audio_format': self.output.audio_format,
                'sample_rate': self.output.sample_rate,
                'bit_depth': self.output.bit_depth,
            },
            'speaker_positions': {
                speaker_id: {'x': pos.x, 'y': pos.y, 'z': pos.z}
                for speaker_id, pos in self.speaker_positions.items()
            },
            'verbose': self.verbose,
            'use_gpu': self.use_gpu,
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)


def get_default_config() -> PipelineConfig:
    """Get a default configuration with reasonable settings."""
    config = PipelineConfig()
    
    # Set up default speaker positions (assuming 2 speakers)
    config.speaker_positions = {
        "SPEAKER_00": SpeakerPosition(x=-1.0, y=2.0, z=0.0),  # Left speaker
        "SPEAKER_01": SpeakerPosition(x=1.0, y=2.0, z=0.0),   # Right speaker
    }
    
    return config