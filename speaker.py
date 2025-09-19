import spaudiopy as spa
import numpy as np
import soundfile as sf

class Speaker():
    def __init__(self, azi, elev, track, order=1):
        self.azi = azi
        self.elev = elev
        self.ambi_order = order
        self.track, self.fs = sf.read(track, always_2d=True) 
        self.mono_track = np.mean(self.track, axis=1).astype(np.float32)
        print(f"Loaded {len(self.mono_track)} samples at {self.fs} Hz")

    def spin_horizontal(self, revolutions=1, speed=1, clockwise=True, sofa_path=None, output_path=None):
        
        if output_path is None:
            output_path = f"spin_horizontal_{revolutions}rev.wav"
        
        rotation_period = len(self.mono_track) / self.fs 
        rotation_period = rotation_period / (revolutions * speed)
        
        # Paths for decoder
        t = np.arange(len(self.mono_track)) / self.fs
        self.azimuth = 2 * np.pi * (t / rotation_period)  
        self.elevation = np.full_like(self.azimuth, np.pi/2)

        if not clockwise:
            self.azimuth = -self.azimuth
        
        # Automatically decode to binaural
        return self.decode_binaural(sofa_path=sofa_path, output_path=output_path)
    
    def move(self, end_position, duration=None, ease="linear"):
        if duration is None:
            duration = len(self.track) / self.fs 
        
        start_azi, start_elev = self.azi, self.elev
        end_azi, end_elev = end_position
        # Todo: implement this
        return self
    
    def spin_vertical(self, revolutions=1, duration=None):
        # Todo: implement this
        return self
    
    def pan_lr(self, degrees=90, duration=None):
        # Todo: implement this
        end_azi = self.azi + np.radians(degrees)
        return self.move((end_azi, self.elev), duration)
        
    def decode_binaural(self, sofa_path=None, output_path="output.wav", save_ambisonic=True, ambisonic_path=None):
        order = self.ambi_order
        n_samples = len(self.mono_track) 
        n_channels = (order + 1) ** 2
        # Empty array to store future ambisonic signals
        ambi_signals = np.zeros((n_channels, n_samples), dtype=np.float32)

        block_size = 4096
        for start in range(0, n_samples, block_size):
            end = min(start + block_size, n_samples)
            Y = spa.sph.sh_matrix(order, self.azimuth[start:end], self.elevation[start:end]) 
            for ch in range(n_channels):
                ambi_signals[ch, start:end] = self.mono_track[start:end] * Y[:, ch]
        
        if save_ambisonic:
            if ambisonic_path is None:
                base_name = output_path.rsplit('.', 1)[0]
                ambisonic_path = f"{base_name}_ambisonic.wav"
            
            sf.write(ambisonic_path, ambi_signals.T, self.fs)
            print(f"Saved ambisonic as {ambisonic_path}")
        
        # Binaural decoding
        hrirs = spa.io.load_sofa_hrirs(sofa_path) if sofa_path else spa.io.load_hrirs(self.fs)
        hrirs_decoded = spa.decoder.magls_bin(hrirs, order)
        stereo = spa.decoder.sh2bin(ambi_signals, hrirs_decoded)
        sf.write(output_path, stereo.T, self.fs)
        print(f"Saved binaural as{output_path}")
        return output_path
"""
randomise

"""