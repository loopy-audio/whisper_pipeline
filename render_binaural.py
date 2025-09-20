#!/usr/bin/env python3
import json
import math
import os
from typing import List, Dict

import numpy as np
import soundfile as sf


def equal_power_gains(az_deg: float):
    """
    Equal-power pan gains for azimuth in degrees.
    """
    az = max(-90.0, min(90.0, az_deg))
    pan = az / 90.0  # [-1,1]
    left = math.sqrt(0.5 * (1.0 - pan))
    right = math.sqrt(0.5 * (1.0 + pan))
    return left, right


def distance_gain(dist: float, k: float = 1.0, min_gain: float = 0.2):
    """
    Simple distance attenuation: gain ~ k / max(dist,1).
    Clamped so it never gets too quiet.
    """
    if dist is None:
        dist = 1.0
    g = k / max(dist, 1e-6)
    return max(min_gain, min(1.0, g))


def fade_io(num_samples: int, fade_samps: int):
    """Create linear in/out fades of length fade_samps for a chunk."""
    fade_samps = min(fade_samps, max(0, num_samples // 3))
    win = np.ones(num_samples, dtype=np.float32)
    if fade_samps > 0:
        ramp = np.linspace(0.0, 1.0, fade_samps, dtype=np.float32)
        win[:fade_samps] *= ramp
        win[-fade_samps:] *= ramp[::-1]
    return win


def render_segments_mono_to_stereo(
    y_mono: np.ndarray,
    sr: int,
    segments: List[Dict],
    crossfade_ms: float = 20.0,
):
    """
    Render mono input into stereo using segment positions.
    Segments must be ordered and non-overlapping (t0<=t1).
    """
    n = len(y_mono)
    out = np.zeros((n, 2), dtype=np.float32)
    cf = int(sr * crossfade_ms / 1000.0)

    for seg in segments:
        a = max(0, int(seg["t0"] * sr))
        b = min(n, int(seg["t1"] * sr))
        if b <= a:
            continue

        chunk = y_mono[a:b].astype(np.float32, copy=False)

        pos = seg.get("position", {}) or {}
        az = float(pos.get("az", 0.0))
        # el currently unused (stub): float(pos.get("el", 0.0))
        dist = float(pos.get("dist", 1.0))

        gl, gr = equal_power_gains(az)
        gd = distance_gain(dist)

        # apply gains
        L = chunk * (gl * gd)
        R = chunk * (gr * gd)

        # crossfade envelope for this chunk
        env = fade_io(len(chunk), cf)
        L *= env
        R *= env

        out[a:b, 0] += L
        out[a:b, 1] += R

    # safety limiter (very light)
    peak = np.max(np.abs(out)) + 1e-12
    if peak > 0.999:
        out /= peak * 1.001
    return out


def ensure_mono(x: np.ndarray):
    """Convert to mono (average channels) if input is stereo/multichannel."""
    if x.ndim == 1:
        return x
    return np.mean(x, axis=1)


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Render simple binaural from timeline_spatial.json")
    ap.add_argument("--input", required=True, help="Path to source WAV/MP3 (mono or stereo)")
    ap.add_argument("--timeline", required=True, help="Path to timeline_spatial.json")
    ap.add_argument("--out", default="out/vocals_binaural.wav", help="Output WAV path")
    ap.add_argument("--mix-bed", default=None, help="Optional bed WAV to mix under (stereo)")
    ap.add_argument("--bed-gain", type=float, default=0.8, help="Gain for bed when mixing")
    ap.add_argument("--crossfade-ms", type=float, default=20.0, help="Segment fade in/out (ms)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Read audio
    y, sr = sf.read(args.input)
    y_mono = ensure_mono(y)

    # Load timeline
    T = json.load(open(args.timeline, "r"))
    segs = T["segments"]
    segs = sorted(segs, key=lambda s: (s["t0"], s["t1"]))

    # Render
    stereo = render_segments_mono_to_stereo(
        y_mono, sr, segs, crossfade_ms=args.crossfade_ms
    )

    # Optional mix with bed
        # Optional mix with bed
    if args.mix_bed:
        bed, sr2 = sf.read(args.mix_bed)

        if sr2 != sr:
            raise RuntimeError("Sample rate mismatch between input and bed.")
        if bed.ndim == 1:
            bed = np.stack([bed, bed], axis=1)
        n = min(len(stereo), len(bed))
        stereo = stereo[:n] + args.bed_gain * bed[:n]

    # Write
    sf.write(args.out, stereo, sr)
    print(f"SKRTTTT wrote {args.out} (sr={sr}, samples={len(stereo)})")


if __name__ == "__main__":
    main()
