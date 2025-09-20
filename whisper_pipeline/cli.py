"""Command-line interface for the WhisperX pipeline."""

import os
import sys
import json
import click
from pathlib import Path
from typing import Optional

try:
    from .config import PipelineConfig, get_default_config, SpeakerPosition
    from .pipeline import WhisperPipeline
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = str(e)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, verbose):
    """WhisperX Pipeline: End-to-end audio processing with transcription, diarization, and spatialization."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    
    # Check dependencies
    if not DEPENDENCIES_AVAILABLE:
        if any(cmd in sys.argv for cmd in ['process', 'transcribe', 'diarize']):
            click.echo(f"Error: Required dependencies not installed: {IMPORT_ERROR}", err=True)
            click.echo("\nTo install dependencies, run:", err=True)
            click.echo("pip install -r requirements.txt", err=True)
            sys.exit(1)


@cli.command()
@click.argument('audio_path', type=click.Path(exists=True))
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Path to configuration YAML file')
@click.option('--output-dir', '-o', type=click.Path(), 
              help='Output directory for results')
@click.option('--transcription-model', type=str, 
              help='WhisperX model to use (e.g., large-v2)')
@click.option('--language', type=str, 
              help='Audio language (auto-detect if not specified)')
@click.option('--num-speakers', type=int, 
              help='Number of speakers (auto-detect if not specified)')
@click.option('--save-transcript/--no-save-transcript', default=True,
              help='Save transcript file')
@click.option('--save-diarization/--no-save-diarization', default=True,
              help='Save diarization results')
@click.option('--save-binaural/--no-save-binaural', default=True,
              help='Save binaural spatialized audio')
@click.option('--transcript-format', type=click.Choice(['json', 'srt', 'txt']), 
              default='json', help='Transcript output format')
@click.option('--audio-format', type=click.Choice(['wav', 'mp3', 'flac']), 
              default='wav', help='Audio output format')
@click.pass_context
def process(ctx, audio_path, config, output_dir, transcription_model, language, 
           num_speakers, save_transcript, save_diarization, save_binaural,
           transcript_format, audio_format):
    """Process audio file through the complete pipeline."""
    verbose = ctx.obj['verbose']
    
    try:
        # Load configuration
        if config:
            pipeline_config = PipelineConfig.from_yaml(config)
            if verbose:
                click.echo(f"Loaded configuration from: {config}")
        else:
            pipeline_config = get_default_config()
            if verbose:
                click.echo("Using default configuration")
        
        # Override config with command line options
        if transcription_model:
            pipeline_config.transcription.model_name = transcription_model
        if language:
            pipeline_config.transcription.language = language
        if num_speakers:
            pipeline_config.diarization.num_speakers = num_speakers
        if output_dir:
            pipeline_config.output.output_dir = output_dir
        
        # Output settings
        pipeline_config.output.save_transcript = save_transcript
        pipeline_config.output.save_diarization = save_diarization
        pipeline_config.output.save_spatialized_audio = save_binaural
        pipeline_config.output.transcript_format = transcript_format
        pipeline_config.output.audio_format = audio_format
        
        # Set verbose mode
        pipeline_config.verbose = verbose
        
        # Create pipeline
        pipeline = WhisperPipeline(pipeline_config, verbose=verbose)
        
        click.echo(f"Processing audio file: {audio_path}")
        
        # Process audio
        result = pipeline.process_audio(audio_path)
        
        # Display results
        click.echo("\n" + "="*50)
        click.echo("PROCESSING COMPLETE")
        click.echo("="*50)
        
        click.echo(f"Audio duration: {result['pipeline_config']['total_duration']:.2f} seconds")
        click.echo(f"Processing time: {result['processing_time']:.2f} seconds")
        click.echo(f"Speakers detected: {result['pipeline_config']['num_speakers_detected']}")
        
        if result.get('output_paths'):
            click.echo("\nOutput files:")
            for file_type, path in result['output_paths'].items():
                click.echo(f"  {file_type}: {path}")
        
        if verbose:
            click.echo(f"\nTranscript preview:")
            for i, segment in enumerate(result['merged_transcript'].get('segments', [])[:3]):
                speaker = segment.get('speaker', 'Unknown')
                text = segment.get('text', '').strip()
                start = segment.get('start', 0)
                click.echo(f"  [{start:.1f}s] {speaker}: {text}")
            
            if len(result['merged_transcript'].get('segments', [])) > 3:
                remaining = len(result['merged_transcript']['segments']) - 3
                click.echo(f"  ... and {remaining} more segments")
        
        click.echo("\nPipeline completed successfully!")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('audio_path', type=click.Path(exists=True))
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Path to configuration YAML file')
@click.option('--model', type=str, default='large-v2',
              help='WhisperX model to use')
@click.option('--language', type=str, 
              help='Audio language (auto-detect if not specified)')
@click.option('--output', '-o', type=click.Path(), 
              help='Output file path')
@click.option('--format', type=click.Choice(['json', 'srt', 'txt']), 
              default='json', help='Output format')
@click.pass_context
def transcribe(ctx, audio_path, config, model, language, output, format):
    """Transcribe audio file only (no diarization or spatialization)."""
    verbose = ctx.obj['verbose']
    
    try:
        # Load configuration
        if config:
            pipeline_config = PipelineConfig.from_yaml(config)
        else:
            pipeline_config = get_default_config()
        
        # Override config with command line options
        pipeline_config.transcription.model_name = model
        if language:
            pipeline_config.transcription.language = language
        
        pipeline_config.verbose = verbose
        
        # Create pipeline
        pipeline = WhisperPipeline(pipeline_config, verbose=verbose)
        
        click.echo(f"Transcribing audio file: {audio_path}")
        
        # Transcribe
        result = pipeline.transcribe_only(audio_path)
        
        # Save or display result
        if output:
            from .utils import save_transcript
            save_transcript(result, output, format)
            click.echo(f"Transcript saved to: {output}")
        else:
            # Display transcript
            click.echo("\nTranscript:")
            click.echo("-" * 40)
            for segment in result.get('segments', []):
                start = segment.get('start', 0)
                text = segment.get('text', '').strip()
                click.echo(f"[{start:.1f}s] {text}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('audio_path', type=click.Path(exists=True))
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Path to configuration YAML file')
@click.option('--num-speakers', type=int, 
              help='Number of speakers (auto-detect if not specified)')
@click.option('--output', '-o', type=click.Path(), 
              help='Output file path')
@click.pass_context
def diarize(ctx, audio_path, config, num_speakers, output):
    """Perform speaker diarization only."""
    verbose = ctx.obj['verbose']
    
    try:
        # Load configuration
        if config:
            pipeline_config = PipelineConfig.from_yaml(config)
        else:
            pipeline_config = get_default_config()
        
        # Override config with command line options
        if num_speakers:
            pipeline_config.diarization.num_speakers = num_speakers
        
        pipeline_config.verbose = verbose
        
        # Create pipeline
        pipeline = WhisperPipeline(pipeline_config, verbose=verbose)
        
        click.echo(f"Diarizing audio file: {audio_path}")
        
        # Diarize
        result = pipeline.diarize_only(audio_path)
        
        # Save or display result
        if output:
            from .utils import save_transcript
            save_transcript(result, output, 'json')
            click.echo(f"Diarization results saved to: {output}")
        else:
            # Display diarization
            click.echo(f"\nSpeaker diarization results:")
            click.echo(f"Detected speakers: {result['num_speakers']}")
            click.echo("-" * 40)
            
            for segment in result.get('segments', []):
                start = segment.get('start', 0)
                end = segment.get('end', 0)
                speaker = segment.get('speaker', 'Unknown')
                duration = segment.get('duration', 0)
                click.echo(f"[{start:.1f}s - {end:.1f}s] {speaker} ({duration:.1f}s)")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option('--output', '-o', type=click.Path(), default='config.yaml',
              help='Output configuration file path')
@click.pass_context
def init_config(ctx, output):
    """Create a default configuration file."""
    verbose = ctx.obj['verbose']
    
    try:
        config = get_default_config()
        config.to_yaml(output)
        
        click.echo(f"Default configuration saved to: {output}")
        
        if verbose:
            click.echo("\nConfiguration contents:")
            with open(output, 'r') as f:
                click.echo(f.read())
        
    except Exception as e:
        click.echo(f"Error creating config: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.option('--speaker', type=str, required=True, 
              help='Speaker ID (e.g., SPEAKER_00)')
@click.option('--x', type=float, required=True, 
              help='X position (left/right, meters)')
@click.option('--y', type=float, required=True, 
              help='Y position (forward/back, meters)')
@click.option('--z', type=float, default=0.0, 
              help='Z position (up/down, meters)')
@click.pass_context
def set_speaker_position(ctx, config_path, speaker, x, y, z):
    """Set speaker position in configuration file."""
    verbose = ctx.obj['verbose']
    
    try:
        # Load existing config
        config = PipelineConfig.from_yaml(config_path)
        
        # Set speaker position
        config.speaker_positions[speaker] = SpeakerPosition(x=x, y=y, z=z)
        
        # Save updated config
        config.to_yaml(config_path)
        
        click.echo(f"Speaker {speaker} position set to ({x}, {y}, {z})")
        
        if verbose:
            click.echo("\nAll speaker positions:")
            for spk_id, pos in config.speaker_positions.items():
                click.echo(f"  {spk_id}: ({pos.x}, {pos.y}, {pos.z})")
        
    except Exception as e:
        click.echo(f"Error setting speaker position: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def info(ctx):
    """Display information about the pipeline and system."""
    verbose = ctx.obj['verbose']
    
    click.echo("WhisperX Pipeline Information")
    click.echo("=" * 40)
    click.echo("Version: 0.1.0")
    
    if not DEPENDENCIES_AVAILABLE:
        click.echo(f"\n❌ Dependencies not installed: {IMPORT_ERROR}")
        click.echo("\nTo install dependencies, run:")
        click.echo("pip install -r requirements.txt")
        click.echo("\nRequired packages:")
        click.echo("  - whisperx>=3.1.0")
        click.echo("  - pyannote.audio>=3.1.0") 
        click.echo("  - torch>=2.0.0")
        click.echo("  - librosa>=0.10.0")
        return
    
    try:
        # Create a default pipeline to get info
        config = get_default_config()
        config.verbose = verbose
        pipeline = WhisperPipeline(config, verbose=False)  # Don't load models
        
        info_data = pipeline.get_pipeline_info()
        
        click.echo("\nConfiguration:")
        click.echo(f"  Transcription model: {info_data['config']['transcription']['model_name']}")
        click.echo(f"  Diarization model: {info_data['config']['diarization']['model_name']}")
        click.echo(f"  Sample rate: {info_data['config']['spatialization']['sample_rate']} Hz")
        click.echo(f"  Output directory: {info_data['config']['output']['output_dir']}")
        
        if info_data.get('speaker_positions'):
            click.echo("\nSpeaker positions:")
            for speaker, pos in info_data['speaker_positions'].items():
                click.echo(f"  {speaker}: ({pos['x']}, {pos['y']}, {pos['z']})")
        
        # System info
        import torch
        click.echo(f"\nSystem information:")
        click.echo(f"  Python version: {sys.version.split()[0]}")
        click.echo(f"  PyTorch version: {torch.__version__}")
        click.echo(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            click.echo(f"  CUDA version: {torch.version.cuda}")
            click.echo(f"  GPU count: {torch.cuda.device_count()}")
        
    except Exception as e:
        click.echo(f"Error getting pipeline info: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()