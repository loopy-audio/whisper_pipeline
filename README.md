# WhisperX Pipeline

End-to-end pipeline for transcribing, diarizing, and spatializing audio with WhisperX and binaural rendering.

## 🎯 Features

- **🎤 Speech Transcription**: High-quality transcription using WhisperX with word-level timestamps
- **👥 Speaker Diarization**: Automatic speaker identification and segmentation using pyannote.audio
- **🎧 3D Spatialization**: Binaural audio rendering for immersive 3D listening experience
- **⚡ GPU Acceleration**: CUDA support for fast processing
- **🔧 Configurable**: Extensive configuration options via YAML files
- **📱 CLI Interface**: Easy-to-use command-line interface
- **🐍 Python API**: Programmatic access for integration into other projects

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/loopy-audio/whisper_pipeline.git
cd whisper_pipeline

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```bash
# Process an audio file through the complete pipeline
whisper-pipeline process audio.wav

# Transcribe only
whisper-pipeline transcribe audio.wav

# Speaker diarization only
whisper-pipeline diarize audio.wav

# Create a configuration file
whisper-pipeline init-config --output my_config.yaml

# Process with custom configuration
whisper-pipeline process audio.wav --config my_config.yaml
```

### Python API

```python
from whisper_pipeline import WhisperPipeline, PipelineConfig
from whisper_pipeline.config import SpeakerPosition

# Create configuration
config = PipelineConfig()

# Set speaker positions for 3D audio
config.speaker_positions = {
    "SPEAKER_00": SpeakerPosition(x=-1.5, y=2.0, z=0.0),  # Left
    "SPEAKER_01": SpeakerPosition(x=1.5, y=2.0, z=0.0),   # Right
}

# Create and run pipeline
pipeline = WhisperPipeline(config, verbose=True)
result = pipeline.process_audio("audio.wav")

print(f"Detected {result['pipeline_config']['num_speakers_detected']} speakers")
print(f"Processing took {result['processing_time']:.2f} seconds")
```

## 📋 Requirements

### System Dependencies
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Python Dependencies
- `whisperx>=3.1.0` - Speech transcription
- `pyannote.audio>=3.1.0` - Speaker diarization
- `torch>=2.0.0` - Deep learning framework
- `librosa>=0.10.0` - Audio processing
- `scipy>=1.10.0` - Scientific computing
- `click>=8.0.0` - CLI interface

See `requirements.txt` for the complete list.

## ⚙️ Configuration

The pipeline is highly configurable via YAML files. Generate a default configuration:

```bash
whisper-pipeline init-config --output config.yaml
```

### Configuration Sections

#### Transcription Settings
```yaml
transcription:
  model_name: "large-v2"          # WhisperX model size
  device: "auto"                  # auto, cpu, cuda
  language: null                  # Language code (auto-detect if null)
  batch_size: 16                  # Batch size for processing
```

#### Diarization Settings
```yaml
diarization:
  model_name: "pyannote/speaker-diarization-3.1"
  num_speakers: null              # Auto-detect if null
  min_speakers: 1
  max_speakers: 10
```

#### Spatialization Settings
```yaml
spatialization:
  sample_rate: 44100              # Output sample rate
  room_size: "medium"             # small, medium, large
  reverb_amount: 0.3              # Reverb amount (0.0-1.0)
  distance_model: "linear"        # Distance attenuation model
```

#### Speaker Positions
```yaml
speaker_positions:
  SPEAKER_00:                     # Speaker ID from diarization
    x: -1.5                       # Left/right position (meters)
    y: 2.0                        # Forward/back position (meters)  
    z: 0.0                        # Up/down position (meters)
```

## 🎧 3D Audio Spatialization

The pipeline creates binaural audio that provides a 3D listening experience when using headphones. 

### Coordinate System
- **X-axis**: Left (-) to Right (+)
- **Y-axis**: Back (-) to Front (+)
- **Z-axis**: Down (-) to Up (+)
- **Listener**: Positioned at origin (0, 0, 0) facing positive Y direction

### HRTF Processing
The spatialization uses Head-Related Transfer Functions (HRTFs) to simulate how sounds reach each ear:
- **ITD**: Interaural Time Differences for left/right positioning
- **ILD**: Interaural Level Differences for distance and angle
- **Frequency Response**: Elevation and distance-dependent filtering

## 📖 Examples

### Process Interview Recording
```bash
# Process a 2-person interview
whisper-pipeline process interview.wav \
    --num-speakers 2 \
    --transcript-format srt \
    --output-dir results/
```

### Configure Speaker Positions
```bash
# Set positions for a panel discussion
whisper-pipeline set-speaker-position config.yaml \
    --speaker SPEAKER_00 --x -2.0 --y 3.0 --z 0.0

whisper-pipeline set-speaker-position config.yaml \
    --speaker SPEAKER_01 --x 0.0 --y 3.0 --z 0.0

whisper-pipeline set-speaker-position config.yaml \
    --speaker SPEAKER_02 --x 2.0 --y 3.0 --z 0.0
```

### Python API Example
```python
import whisper_pipeline

# Quick transcription
pipeline = whisper_pipeline.WhisperPipeline()
transcript = pipeline.transcribe_only("audio.wav")

# Access segments with timestamps
for segment in transcript['segments']:
    print(f"[{segment['start']:.1f}s] {segment['text']}")
```

## 🔧 CLI Reference

### Main Commands

- `process` - Run complete pipeline (transcription + diarization + spatialization)
- `transcribe` - Transcription only
- `diarize` - Speaker diarization only
- `init-config` - Create default configuration file
- `set-speaker-position` - Set speaker position in config file
- `info` - Display system and pipeline information

### Common Options

- `--config, -c` - Configuration file path
- `--output-dir, -o` - Output directory
- `--verbose, -v` - Enable verbose output
- `--num-speakers` - Number of speakers (for diarization)
- `--language` - Audio language
- `--transcript-format` - Output format (json, srt, txt)

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   WhisperX      │    │  pyannote.audio │    │   Binaural      │
│  Transcription  │───▶│   Diarization   │───▶│ Spatialization  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Transcript with │    │ Speaker segments│    │ 3D Binaural    │
│ word timestamps │    │ and boundaries  │    │ Audio Output    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Components

1. **TranscriptionProcessor**: WhisperX integration with word-level alignment
2. **DiarizationProcessor**: Speaker identification and segmentation  
3. **SpatializationProcessor**: HRTF-based binaural rendering
4. **WhisperPipeline**: Main orchestrator coordinating all components

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [WhisperX](https://github.com/m-bain/whisperX) for high-quality speech transcription
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) for speaker diarization
- HRTF research community for 3D audio spatialization techniques

## 📞 Support

- 📧 Email: support@loopy-audio.com
- 🐛 Issues: [GitHub Issues](https://github.com/loopy-audio/whisper_pipeline/issues)
- 📖 Documentation: [Wiki](https://github.com/loopy-audio/whisper_pipeline/wiki)
