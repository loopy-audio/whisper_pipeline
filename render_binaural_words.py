import argparse, json, math, os
import numpy as np, soundfile as sf

def ensure_mono(x): return x if x.ndim == 1 else np.mean(x, axis=1)

def equal_power_gains(az_deg):
    az = float(np.clip(az_deg, -90.0, 90.0)); pan = az/90.0
    L = math.sqrt(0.5*(1.0 - pan)); R = math.sqrt(0.5*(1.0 + pan))
    return L, R

def distance_gain(dist, k=1.0, min_gain=0.2):
    d = 1.0 if dist is None else float(dist)
    g = k / max(d, 1e-6)
    return max(min_gain, min(1.0, g))

def fade_io(n, f):
    f = min(f, max(0, n//4))  # cap fade to 1/4 of segment (less over-attenuation)
    win = np.ones(n, np.float32)
    if f > 0:
        r = np.linspace(0.0, 1.0, f, dtype=np.float32)
        win[:f] *= r; win[-f:] *= r[::-1]
    return win

def spatialize_chunk(mono, az=0.0, el=0.0, dist=1.0):
    gl, gr = equal_power_gains(az); gd = distance_gain(dist)
    L = mono * (gl*gd); R = mono * (gr*gd)
    return np.stack([L, R], axis=1)

def main():
    ap = argparse.ArgumentParser(description="Word-by-word binaural render with padding")
    ap.add_argument("--input", required=True)
    ap.add_argument("--word-timeline", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mix-bed", dest="mix_bed", default=None)
    ap.add_argument("--bed-gain", type=float, default=0.9)
    ap.add_argument("--crossfade-ms", type=float, default=6.0)
    ap.add_argument("--pad-ms", type=float, default=25.0, help="pre/post padding per word")
    ap.add_argument("--mode", choices=["overlay","concat"], default="overlay")
    ap.add_argument("--spk-left-id", default="SPEAKER_00")
    ap.add_argument("--spk-right-id", default="SPEAKER_01")
    ap.add_argument("--az-left", type=float, default=-25.0)
    ap.add_argument("--az-right", type=float, default=25.0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    y, sr = sf.read(args.input); y = ensure_mono(y)
    bed = None; sr_bed = sr
    if args.mix_bed:
        bed, sr_bed = sf.read(args.mix_bed)
        if bed.ndim == 1: bed = np.stack([bed, bed], axis=1)
        if sr_bed != sr: raise RuntimeError(f"SR mismatch: input {sr} vs bed {sr_bed}")

    TL = json.load(open(args.word_timeline))
    words = TL["words"]

    # Auto-positions from speaker if missing
    for w in words:
        if not w.get("position"):
            spk = w.get("speaker")
            if spk == args.spk_left_id:
                w["position"] = {"az": args.az_left, "el": 0.0, "dist": 1.0}
            elif spk == args.spk_right_id:
                w["position"] = {"az": args.az_right, "el": 0.0, "dist": 1.0}
            else:
                w["position"] = {"az": 0.0, "el": 0.0, "dist": 1.0}

    cf = int(sr * args.crossfade_ms / 1000.0)
    pad = int(sr * args.pad_ms / 1000.0)

    def render_overlay():
        total_len = max(len(y), len(bed) if bed is not None else 0)
        out = np.zeros((total_len, 2), np.float32)
        N = len(y)
        for w in words:
            a = max(0, int(w["t0"]*sr) - pad)
            b = min(N, int(w["t1"]*sr) + pad)
            if b <= a: continue
            seg = y[a:b].astype(np.float32, copy=False)
            pos = w["position"]; az = float(pos.get("az",0.0)); el=float(pos.get("el",0.0)); dist=float(pos.get("dist",1.0))
            seg2 = spatialize_chunk(seg, az=az, el=el, dist=dist)
            env = fade_io(len(seg), cf)
            seg2[:,0] *= env; seg2[:,1] *= env
            bb = min(total_len, a + len(seg2))
            out[a:bb] += seg2[:bb - a]
        # light limiter
        peak = float(np.max(np.abs(out))) + 1e-12
        if peak > 0.999: out /= (peak*1.001)
        return out

    def render_concat():
        pieces = []
        for w in words:
            a = max(0, int(w["t0"]*sr) - pad)
            b = min(len(y), int(w["t1"]*sr) + pad)
            if b <= a: continue
            seg = y[a:b].astype(np.float32, copy=False)
            pos = w["position"]; az = float(pos.get("az",0.0)); el=float(pos.get("el",0.0)); dist=float(pos.get("dist",1.0))
            seg2 = spatialize_chunk(seg, az=az, el=el, dist=dist)
            env = fade_io(len(seg2), cf)
            seg2[:,0] *= env; seg2[:,1] *= env
            if not pieces: pieces.append(seg2); continue
            fade_len = min(cf, len(seg2), len(pieces[-1]))
            if fade_len > 0:
                fo = np.linspace(1,0,fade_len, dtype=np.float32)[:,None]
                fi = np.linspace(0,1,fade_len, dtype=np.float32)[:,None]
                pieces[-1][-fade_len:] = pieces[-1][-fade_len:]*fo + seg2[:fade_len]*fi
                pieces[-1] = np.vstack([pieces[-1], seg2[fade_len:]])
            else:
                pieces.append(seg2)
        out = pieces[0] if pieces else np.zeros((0,2), np.float32)
        peak = float(np.max(np.abs(out))) + 1e-12
        if peak > 0.999: out /= (peak*1.001)
        return out

    stereo = render_overlay() if args.mode == "overlay" else render_concat()

    if bed is not None:
        n = min(len(stereo), len(bed))
        stereo = stereo[:n] + args.bed_gain * bed[:n]
        peak = float(np.max(np.abs(stereo))) + 1e-12
        if peak > 0.999: stereo /= (peak*1.001)

    sf.write(args.out, stereo, sr)
    print(f"SKRTTT wrote {args.out} (sr={sr}, samples={len(stereo)}, mode={args.mode}, pad_ms={args.pad_ms}, xfade_ms={args.crossfade_ms})")

if __name__ == "__main__":
    main()


