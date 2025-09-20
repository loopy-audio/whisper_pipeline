"""Spatialization module for binaural audio rendering."""

import numpy as np
import scipy.signal
from typing import Dict, List, Tuple, Optional
import logging

from .config import SpatializationConfig, SpeakerPosition
from .utils import load_audio, calculate_3d_distance, apply_distance_attenuation


class SpatializationProcessor:
    """Handles 3D spatialization and binaural rendering of audio."""
    
    def __init__(self, config: SpatializationConfig, verbose: bool = False):
        """
        Initialize the spatialization processor.
        
        Args:
            config: Spatialization configuration
            verbose: Enable verbose logging
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose
        
        # Initialize HRTF data (simplified version)
        self.hrtf_data = self._load_hrtf_data()
        
        # Listener position (assumed to be at origin)
        self.listener_position = (0.0, 0.0, 0.0)
        
        if verbose:
            self.logger.info("Spatialization processor initialized")
    
    def _load_hrtf_data(self) -> Dict:
        """
        Load HRTF (Head-Related Transfer Function) data.
        
        Note: This is a simplified implementation. In a real system,
        you would load actual HRTF measurements from a database.
        """
        # For this implementation, we'll use simplified HRTF approximations
        # In practice, you would load real HRTF data from sources like:
        # - CIPIC HRTF Database
        # - MIT KEMAR HRTF Database
        # - ARI HRTF Database
        
        sample_rate = self.config.sample_rate
        
        # Generate simplified HRTF impulse responses
        # These are very basic approximations for demonstration
        impulse_length = int(0.001 * sample_rate)  # 1ms impulse
        
        hrtf_data = {
            'sample_rate': sample_rate,
            'impulse_length': impulse_length,
            'azimuth_angles': np.arange(-180, 181, 15),  # Every 15 degrees
            'elevation_angles': np.arange(-40, 91, 10),  # Every 10 degrees
        }
        
        # Generate simplified HRTFs for different angles
        hrtf_data['left_ear'] = {}
        hrtf_data['right_ear'] = {}
        
        for azimuth in hrtf_data['azimuth_angles']:
            for elevation in hrtf_data['elevation_angles']:
                # Simplified HRTF generation (not acoustically accurate)
                left_hrtf, right_hrtf = self._generate_simplified_hrtf(
                    azimuth, elevation, impulse_length, sample_rate
                )
                
                key = (azimuth, elevation)
                hrtf_data['left_ear'][key] = left_hrtf
                hrtf_data['right_ear'][key] = right_hrtf
        
        return hrtf_data
    
    def _generate_simplified_hrtf(self, azimuth: float, elevation: float, 
                                 length: int, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate simplified HRTF impulse responses.
        
        Args:
            azimuth: Azimuth angle in degrees (-180 to 180)
            elevation: Elevation angle in degrees (-40 to 90)
            length: Impulse response length
            sample_rate: Sample rate
            
        Returns:
            Tuple of (left_ear_hrtf, right_ear_hrtf)
        """
        # Very simplified HRTF model based on basic acoustic principles
        # Real HRTFs are much more complex and should be measured
        
        # Convert to radians
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)
        
        # Simple delay model based on head size
        head_radius = 0.0875  # Average head radius in meters
        sound_speed = 343.0   # Speed of sound in m/s
        
        # Calculate ITD (Interaural Time Difference)
        itd = (head_radius / sound_speed) * (az_rad + np.sin(az_rad))
        itd_samples = int(itd * sample_rate)
        
        # Calculate ILD (Interaural Level Difference) - simplified
        # Sound from the right should be louder in right ear, etc.
        left_gain = 0.5 * (1 - np.sin(az_rad) * 0.5)
        right_gain = 0.5 * (1 + np.sin(az_rad) * 0.5)
        
        # Apply elevation effects (very simplified)
        elevation_factor = np.cos(el_rad)
        left_gain *= elevation_factor
        right_gain *= elevation_factor
        
        # Create impulse responses
        left_hrtf = np.zeros(length)
        right_hrtf = np.zeros(length)
        
        # Simple impulse with delay and gain
        if length > abs(itd_samples):
            if itd_samples >= 0:
                # Sound reaches left ear first
                left_hrtf[0] = left_gain
                if itd_samples < length:
                    right_hrtf[itd_samples] = right_gain
            else:
                # Sound reaches right ear first
                right_hrtf[0] = right_gain
                if abs(itd_samples) < length:
                    left_hrtf[abs(itd_samples)] = left_gain
        else:
            # Very short impulse, just use gains
            left_hrtf[0] = left_gain
            right_hrtf[0] = right_gain
        
        # Add some frequency shaping (very basic)
        if length > 10:
            # Simple high-frequency roll-off
            for i in range(1, min(length, 10)):
                factor = np.exp(-i * 0.1)
                left_hrtf[i] = left_hrtf[0] * factor * 0.1
                right_hrtf[i] = right_hrtf[0] * factor * 0.1
        
        return left_hrtf, right_hrtf
    
    def _cartesian_to_spherical(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Convert cartesian coordinates to spherical coordinates.
        
        Args:
            x, y, z: Cartesian coordinates
            
        Returns:
            Tuple of (distance, azimuth, elevation) in meters and degrees
        """
        distance = np.sqrt(x**2 + y**2 + z**2)
        
        if distance == 0:
            return 0.0, 0.0, 0.0
        
        # Azimuth: angle in horizontal plane from positive y-axis
        azimuth = np.degrees(np.arctan2(x, y))
        
        # Elevation: angle from horizontal plane
        elevation = np.degrees(np.arcsin(z / distance))
        
        return distance, azimuth, elevation
    
    def _find_closest_hrtf(self, azimuth: float, elevation: float) -> Tuple[float, float]:
        """
        Find the closest HRTF angles to the desired azimuth and elevation.
        
        Args:
            azimuth: Target azimuth in degrees
            elevation: Target elevation in degrees
            
        Returns:
            Tuple of (closest_azimuth, closest_elevation)
        """
        # Find closest azimuth
        azimuth_angles = self.hrtf_data['azimuth_angles']
        closest_az_idx = np.argmin(np.abs(azimuth_angles - azimuth))
        closest_azimuth = azimuth_angles[closest_az_idx]
        
        # Find closest elevation
        elevation_angles = self.hrtf_data['elevation_angles']
        closest_el_idx = np.argmin(np.abs(elevation_angles - elevation))
        closest_elevation = elevation_angles[closest_el_idx]
        
        return closest_azimuth, closest_elevation
    
    def spatialize_audio_segment(self, audio: np.ndarray, 
                                speaker_position: SpeakerPosition,
                                apply_reverb: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Spatialize an audio segment for a speaker at given position.
        
        Args:
            audio: Mono audio signal
            speaker_position: 3D position of the speaker
            apply_reverb: Whether to apply reverb
            
        Returns:
            Tuple of (left_channel, right_channel) binaural audio
        """
        if len(audio) == 0:
            return np.array([]), np.array([])
        
        # Convert speaker position to spherical coordinates relative to listener
        distance, azimuth, elevation = self._cartesian_to_spherical(
            speaker_position.x - self.listener_position[0],
            speaker_position.y - self.listener_position[1], 
            speaker_position.z - self.listener_position[2]
        )
        
        if self.verbose:
            self.logger.debug(f"Speaker position: distance={distance:.2f}m, "
                            f"azimuth={azimuth:.1f}°, elevation={elevation:.1f}°")
        
        # Apply distance attenuation
        attenuated_audio = apply_distance_attenuation(
            audio, distance, self.config.max_distance, self.config.distance_model
        )
        
        # Find closest HRTF
        closest_azimuth, closest_elevation = self._find_closest_hrtf(azimuth, elevation)
        
        # Get HRTF impulse responses
        hrtf_key = (closest_azimuth, closest_elevation)
        left_hrtf = self.hrtf_data['left_ear'][hrtf_key]
        right_hrtf = self.hrtf_data['right_ear'][hrtf_key]
        
        # Convolve audio with HRTFs
        left_channel = scipy.signal.convolve(attenuated_audio, left_hrtf, mode='full')
        right_channel = scipy.signal.convolve(attenuated_audio, right_hrtf, mode='full')
        
        # Truncate to original length plus HRTF length
        max_length = len(audio) + len(left_hrtf) - 1
        left_channel = left_channel[:max_length]
        right_channel = right_channel[:max_length]
        
        # Apply reverb if requested
        if apply_reverb and self.config.reverb_amount > 0:
            left_channel, right_channel = self._apply_reverb(
                left_channel, right_channel, distance
            )
        
        return left_channel, right_channel
    
    def _apply_reverb(self, left_channel: np.ndarray, right_channel: np.ndarray,
                     distance: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply simple reverb effect.
        
        Args:
            left_channel: Left audio channel
            right_channel: Right audio channel
            distance: Distance to source (affects reverb characteristics)
            
        Returns:
            Tuple of (left_reverb, right_reverb) audio with reverb applied
        """
        # Simple reverb using comb filters and all-pass filters
        sample_rate = self.config.sample_rate
        reverb_amount = self.config.reverb_amount
        
        # Adjust reverb based on room size
        room_multipliers = {"small": 0.5, "medium": 1.0, "large": 1.5}
        room_mult = room_multipliers.get(self.config.room_size, 1.0)
        
        # Comb filter delays (in samples)
        comb_delays = [
            int(0.029 * sample_rate * room_mult),
            int(0.036 * sample_rate * room_mult),
            int(0.041 * sample_rate * room_mult),
            int(0.044 * sample_rate * room_mult)
        ]
        
        # All-pass filter delays
        allpass_delays = [
            int(0.005 * sample_rate),
            int(0.017 * sample_rate)
        ]
        
        def apply_reverb_to_channel(channel):
            reverb_signal = np.zeros_like(channel)
            
            # Apply comb filters
            for delay in comb_delays:
                if delay < len(channel):
                    delayed = np.concatenate([np.zeros(delay), channel[:-delay]])
                    reverb_signal += delayed * 0.7 ** (delay / sample_rate)
            
            # Apply all-pass filters
            for delay in allpass_delays:
                if delay < len(reverb_signal):
                    delayed = np.concatenate([np.zeros(delay), reverb_signal[:-delay]])
                    reverb_signal = reverb_signal + delayed * 0.3
            
            # Mix dry and wet signals
            return channel + reverb_signal * reverb_amount
        
        left_reverb = apply_reverb_to_channel(left_channel)
        right_reverb = apply_reverb_to_channel(right_channel)
        
        return left_reverb, right_reverb
    
    def create_binaural_mix(self, transcript_with_diarization: Dict,
                           audio_path: str,
                           speaker_positions: Dict[str, SpeakerPosition]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a binaural mix from transcript with speaker diarization.
        
        Args:
            transcript_with_diarization: Transcript with speaker information
            audio_path: Path to original audio file
            speaker_positions: Dictionary mapping speaker IDs to positions
            
        Returns:
            Tuple of (left_channel, right_channel) binaural audio
        """
        if self.verbose:
            self.logger.info("Creating binaural mix")
        
        # Load original audio
        audio, sr = load_audio(audio_path, target_sr=self.config.sample_rate)
        
        # Initialize output channels
        left_mix = np.zeros_like(audio)
        right_mix = np.zeros_like(audio)
        
        # Process each segment
        segments = transcript_with_diarization.get('segments', [])
        
        for segment in segments:
            speaker = segment.get('speaker')
            if not speaker or speaker not in speaker_positions:
                if self.verbose:
                    self.logger.warning(f"No position found for speaker: {speaker}")
                continue
            
            start_time = segment['start']
            end_time = segment['end']
            
            # Convert time to samples
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            # Extract audio segment
            if start_sample < len(audio) and end_sample > start_sample:
                end_sample = min(end_sample, len(audio))
                audio_segment = audio[start_sample:end_sample]
                
                # Spatialize the segment
                left_spatial, right_spatial = self.spatialize_audio_segment(
                    audio_segment, speaker_positions[speaker]
                )
                
                # Add to mix (handle length differences)
                mix_end = min(start_sample + len(left_spatial), len(left_mix))
                spatial_end = mix_end - start_sample
                
                if spatial_end > 0:
                    left_mix[start_sample:mix_end] += left_spatial[:spatial_end]
                    right_mix[start_sample:mix_end] += right_spatial[:spatial_end]
        
        # Normalize to prevent clipping
        max_amplitude = max(np.max(np.abs(left_mix)), np.max(np.abs(right_mix)))
        if max_amplitude > 1.0:
            left_mix /= max_amplitude
            right_mix /= max_amplitude
        
        if self.verbose:
            self.logger.info(f"Binaural mix created: {len(left_mix)/sr:.2f}s")
        
        return left_mix, right_mix
    
    def get_processor_info(self) -> Dict:
        """Get information about the spatialization processor."""
        return {
            "sample_rate": self.config.sample_rate,
            "hrtf_dataset": self.config.hrtf_dataset,
            "room_size": self.config.room_size,
            "reverb_amount": self.config.reverb_amount,
            "distance_model": self.config.distance_model,
            "max_distance": self.config.max_distance,
            "use_doppler": self.config.use_doppler,
            "num_hrtf_positions": len(self.hrtf_data.get('azimuth_angles', [])) * 
                                 len(self.hrtf_data.get('elevation_angles', []))
        }