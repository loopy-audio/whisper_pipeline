#!/usr/bin/env python3
"""
wav_to_word_timeline.py
- Input:  --audio 
- Output: out/words.json, out/words.jsonl, out/word_timeline.json
- Features: WhisperX ASR + word-level alignment, optional diarization (pyannote via HF token),
           per-word timeline with neutral/default positions (speaker-based az mapping).

Usage:
  python wav_to_word_timeline.py \
    --audio stems/htdemucs/song/vocals.wav \
    --out-dir out \
    --asr-model small \
    --diarize           
"""

import os
import json
import time
import argparse
from typing import Dict, Any, List, Optional

import torch
import soundfile as sf
import whisperx  
# pip install git+https://github.com/m-bain/whisperx.git

# ỏ both old and new WhisperX locations for DiarizationPipeline
try:
    #Newer WhisperX (>= ~3.3.x)
    from whisperx.diarize import DiarizationPipeline as _WhisperxDiarization
except Exception:
    # Older WhisperX
    _WhisperxDiarization = getattr(whisperx, "DiarizationPipeline", None)

# Helpers

def load_audio_sr(path: str) -> int:
    """Return samplerate (header) for the input file; fallback to 48000 if unreadable."""
    try:
        _, sr = sf.read(path, always_2d=False)
        return int(sr)
    except Exception:
        return 48000


def ensure_asr_model(model_id: str, device: str, compute_type: str, lang_hint: Optional[str] = None):
    return whisperx.load_model(model_id, device, compute_type=compute_type, language=lang_hint)


def run_asr_and_align(audio_path: str, asr_model, device: str, lang_hint: Optional[str] = None):
    """
    1) WhisperX ASR -> segments
    2) Load alignment model for detected language
    3) Align to word-level timestamps
    """
    trans = asr_model.transcribe(audio_path, batch_size=16)
    language = trans.get("language", lang_hint or "en")
    align_model, align_meta = whisperx.load_align_model(language_code=language, device=device)
    aligned = whisperx.align(
        trans["segments"],
        align_model, align_meta,
        audio_path, device,
        return_char_alignments=False
    )
    return aligned, language


def maybe_diarize(audio_path: str, enable: bool, device: str):
    """
    Run diarization if requested. Requires HF_TOKEN in environment.
    Returns pyannote-style segments or None.
    """
    if not enable:
        return None
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Diarization requested but HF_TOKEN not set (export HF_TOKEN=hf_xxx).")
    if _WhisperxDiarization is None:
        raise RuntimeError("Your whisperx version does not expose DiarizationPipeline. "
                           "Try upgrading whisperx or importing from whisperx.diarize.")
    diar = _WhisperxDiarization(use_auth_token=token, device=device)
    return diar(audio_path)


def flatten_words(aligned: Dict[str, Any], conf_min: float = 0.0) -> List[Dict[str, Any]]:
    """
    Extract a flat list of words with t0/t1/confidence and any speaker carried.
    Filters out words with missing times or low confidence.
    """
    words: List[Dict[str, Any]] = []
    for seg in aligned.get("segments", []) or []:
        seg_spk = seg.get("speaker")
        for w in seg.get("words") or []:
            text = w.get("word")
            t0 = w.get("start")
            t1 = w.get("end")
            conf = w.get("confidence")
            if not text or t0 is None or t1 is None:
                continue
            if conf is not None and conf < conf_min:
                continue
            out = {
                "word": text,
                "t0": float(t0),
                "t1": float(t1),
                "confidence": float(conf) if conf is not None else None,
            }
            # prefer per-word speaker, else segment speaker, else None
            if w.get("speaker") is not None:
                out["speaker"] = w["speaker"]
            elif seg_spk is not None:
                out["speaker"] = seg_spk
            words.append(out)
    words.sort(key=lambda x: x["t0"])
    return words


def build_word_timeline(words: List[Dict[str, Any]], sr: int,
                        speaker_az: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Build per-word timeline entries.
    - Default neutral positions (az=0, el=0, dist=1.0)
    - If speaker_az provided, map known speakers to fixed azimuths (e.g., {'SPEAKER_00': -30, 'SPEAKER_01': 30})
    """
    speaker_az = speaker_az or {}
    out = {"sr": int(sr), "words": []}
    for i, w in enumerate(words):
        spk = w.get("speaker")
        az = float(speaker_az.get(spk, 0.0))
        out["words"].append({
            "id": f"w_{i:06d}",
            "t0": w["t0"],
            "t1": w["t1"],
            "word": w["word"],
            "speaker": spk if spk is not None else None,
            "confidence": w.get("confidence"),
            "position": {"az": az, "el": 0.0, "dist": 1.0}
        })
    return out

def main():
    ap = argparse.ArgumentParser(description="ASR+alignment (and optional diarization) → word-by-word timeline JSON")
    ap.add_argument("--audio", required=True, help="Path to input audio (wav/mp3/m4a/flac)")
    ap.add_argument("--out-dir", default="out", help="Output directory")
    ap.add_argument("--asr-model", default="small", help="WhisperX model id: tiny/base/small/medium/large-v2")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--compute-type", choices=["float16", "float32", "int8"], default=None,
                    help="Default: float16 on cuda, float32 otherwise")
    ap.add_argument("--language", default=None, help="Language hint (e.g., 'en'); usually auto-detected")
    ap.add_argument("--diarize", action="store_true", help="Enable speaker diarization (requires HF_TOKEN)")
    ap.add_argument("--conf-min", type=float, default=0.0, help="Drop words with confidence below this")
    ap.add_argument("--spk-left", default="SPEAKER_00", help="Map this speaker ID to -30° az")
    ap.add_argument("--spk-right", default="SPEAKER_01", help="Map this speaker ID to +30° az")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    compute_type = args.compute_type or ("float16" if device == "cuda" else "float32")

    os.makedirs(args.out_dir, exist_ok=True)
    sr_header = load_audio_sr(args.audio)

    t0 = time.time()
    asr_model = ensure_asr_model(args.asr_model, device, compute_type, args.language)
    aligned, lang = run_asr_and_align(args.audio, asr_model, device, args.language)

    diar_segments = maybe_diarize(args.audio, args.diarize, device)
    if diar_segments is not None:
        aligned = whisperx.assign_word_speakers(diar_segments, aligned)

    words = flatten_words(aligned, conf_min=args.conf_min)
    t1 = time.time()

    # Save raw word list
    words_json = os.path.join(args.out_dir, "words.json")
    words_jsonl = os.path.join(args.out_dir, "words.jsonl")
    with open(words_json, "w") as f:
        json.dump(words, f, indent=2)
    with open(words_jsonl, "w") as f:
        for i, w in enumerate(words):
            ww = dict(w)
            ww["id"] = f"w_{i:06d}"
            if "speaker" not in ww:
                ww["speaker"] = None
            f.write(json.dumps(ww) + "\n")

    # Build per-word timeline JSON with neutral/default positions (speaker-based az map)
    speaker_az = {args.spk_left: -30.0, args.spk_right: 30.0}
    word_timeline = build_word_timeline(words, sr_header, speaker_az=speaker_az)
    timeline_path = os.path.join(args.out_dir, "word_timeline.json")
    with open(timeline_path, "w") as f:
        json.dump(word_timeline, f, indent=2)

    print(f"SKRTTTT language={lang}, words={len(words)}, time={t1 - t0:.2f}s")
    print(f"* saved: {words_json}")
    print(f"* {words_jsonl}")
    print(f"* saved: {timeline_path}")


if __name__ == "__main__":
    main()
