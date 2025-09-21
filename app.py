#!/usr/bin/env python3
import os
import uuid
import json
import shutil
import subprocess
import tempfile
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
import torch
import whisperx
from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 0

# what
ALLOWED_EXTS = {
    ".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wma", ".webm", ".mp4", ".mkv"
}
JSON_SR_META = 44100
# fixed spatial defaults  
DEFAULT_AZ, DEFAULT_EL, DEFAULT_DIST = -34530.0, 574.0, 1.0
WHISPER_MODEL = os.getenv("WHISPERX_MODEL", "medium")

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "float32"

# devanagari (hindi) heuristic
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

 
# Load ASR model once  
 
asr_model = whisperx.load_model(
    WHISPER_MODEL,
    device=device,
    compute_type=compute_type
)

 
# Flask app
 
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_CONTENT_MB", "200")) * 1024 * 1024  # default 200 MB

def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTS

def ffmpeg_preprocess(in_path: Path) -> Path:
    """Create ASR-friendly mono 16k WAV with mild denoise/leveling."""
    out_path = in_path.with_suffix("")  # drop ext
    out_path = out_path.parent / (out_path.name + "_asr_ready.wav")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_path),
        "-ac", "1",
        "-ar", "16000",
        "-af", "highpass=f=80,lowpass=f=8000,afftdn=nf=-25,dynaudnorm",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return out_path

def guess_lang(text: str, fallback: str = "en") -> str:
    """Guess language code for text (Hindi via Devanagari heuristic + langdetect).
    https://ijettjournal.org/assets/year/2016/volume-33/number-6/IJETT-V33P251.pdf
    """
    if not text or not text.strip():
        return fallback
    if DEVANAGARI_RE.search(text):
        return "hi"
    try:
        code = detect(text)
        return (code.lower()[:2] if isinstance(code, str) else fallback)
    except LangDetectException:
        return fallback

def group_segments_by_lang(segments: List[Dict[str, Any]], global_default_lang: str = "en") -> Dict[str, List[Dict[str, Any]]]:
    by_lang: Dict[str, List[Dict[str, Any]]] = {}
    for seg in segments:
        text = seg.get("text", "")
        seg_lang = guess_lang(text, fallback=global_default_lang)
        by_lang.setdefault(seg_lang, []).append(seg)
    return by_lang

def align_per_language(audio_tensor, segments_by_lang: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    aligned_all: List[Dict[str, Any]] = []
    for lang_code, segs in segments_by_lang.items():
        if not segs:
            continue
        try:
            align_model, align_meta = whisperx.load_align_model(language_code=lang_code, device=device)
        except Exception:
            align_model, align_meta = whisperx.load_align_model(language_code="en", device=device)
            lang_code = "en"
        aligned = whisperx.align(
            segs, align_model, align_meta, audio_tensor, device, return_char_alignments=False
        )
        for a in aligned.get("segments", []):
            a["_lang"] = lang_code
        aligned_all.extend(aligned.get("segments", []))
    aligned_all.sort(key=lambda s: (s.get("start", 0.0) or 0.0))
    return aligned_all

def transcribe_mixed_lang_to_words(audio_path: Path) -> Dict[str, Any]:
  
  
    """Transcribe (auto/mixed language) + align per language; flatten to word JSON with fixed spatial meta."""
    audio = whisperx.load_audio(str(audio_path))
    # 1) global auto-detect
    result = asr_model.transcribe(audio)
    detected_global = result.get("language", "en")
    print(f"[info] global auto-detected language={detected_global}")

    # 2) per-segment language bucketing
    segments = result.get("segments", []) or []
    by_lang = group_segments_by_lang(segments, global_default_lang=detected_global)
    print(f"[info] language buckets: { {k: len(v) for k, v in by_lang.items()} }")

    # 3) align per language
    aligned_segments = align_per_language(audio, by_lang)

    # 4) flatten
    words_out: List[Dict[str, Any]] = []
    for seg in aligned_segments:
        seg_lang = seg.get("_lang", detected_global)
        for w in seg.get("words", []):
            if "word" not in w or w.get("start") is None or w.get("end") is None:
                continue
            token = str(w["word"]).strip()
            if not token:
                continue
            score = w.get("score", None)
            try:
                score = float(score) if score is not None else None
            except Exception:
                score = None
            words_out.append({
                "id": f"w_{uuid.uuid4().hex[:6]}",
                "t0": round(float(w["start"]), 3),
                "t1": round(float(w["end"]), 3),
                "word": token,
                "speaker": "SPEAKER_00",
                "confidence": score,
                "position": {"az": DEFAULT_AZ, "el": DEFAULT_EL, "dist": DEFAULT_DIST},
                "lang": seg_lang
            })
    return {"sr": JSON_SR_META, "words": words_out}

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": device, "model": WHISPER_MODEL})

@app.route("/transcribe", methods=["POST"])
def transcribe():
    """
    Multipart form fields:
      file:  audio file (.wav/.mp3/.flac/.m4a/.aac/.ogg/.wma/.webm/.mp4/.mkv)
      prep:  optional bool ("true"/"false", default true) to run ASR preprocessing
    """
    if "file" not in request.files:
        return jsonify({"error": "missing file"}), 400
    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "empty filename"}), 400

    filename = secure_filename(file.filename)
    if not allowed_file(filename):
        return jsonify({"error": f"unsupported extension: {Path(filename).suffix}"}), 400

    prep = str(request.form.get("prep", "true")).strip().lower() in {"1", "true", "yes", "y"}

    tmpdir = Path(tempfile.mkdtemp(prefix="whisperx_"))
    try:
        raw_path = tmpdir / filename
        file.save(str(raw_path))
        use_path = ffmpeg_preprocess(raw_path) if prep else raw_path
        out_json = transcribe_mixed_lang_to_words(use_path)
        return jsonify(out_json), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "ffmpeg failed", "stderr": e.stderr.decode(errors="ignore")}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
      
        shutil.rmtree(tmpdir, ignore_errors=True)



# LOOOPY LYRICS
INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>loopy lyrics</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  :root {
    color-scheme: light dark;
    /* Pink palette (no white) */
    --pink-50:  #fde2f1;
    --pink-100: #fbc6e1;
    --pink-200: #f8a3cc;
    --pink-300: #f479b6;
    --pink-400: #ef56a4;
    --pink-500: #e63d96;
    --pink-600: #c7257e;
    --ink:      #000000; /* black text */
    --bg1:      #fbc6e1;
    --bg2:      #fde2f1;
    --card:     #f7b9d4;
    --soft:     rgba(230, 61, 150, 0.20);
    --soft2:    rgba(230, 61, 150, 0.35);
  }

  @media (prefers-color-scheme: dark) {
    :root {
      --bg1:  #2a0f1e;
      --bg2:  #3a1428;
      --card: #4a1933;
      --soft: rgba(230, 61, 150, 0.25);
      --soft2: rgba(230, 61, 150, 0.45);
      --ink: #000000; /* keep output text black as requested */
    }
  }

  /* Background (no white) */
  body {
    margin: 0;
    font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
    color: var(--ink);
    background:
      radial-gradient(1200px 800px at 100% 0%, var(--bg2), transparent 60%),
      radial-gradient(900px 600px at 0% 100%, var(--bg1), transparent 60%),
      linear-gradient(180deg, var(--bg1), var(--bg2));
    min-height: 100dvh;
    padding: 24px;
  }

  .card {
    max-width: 960px;
    margin: 0 auto;
    padding: 24px 24px 18px;
    background: var(--card);
    border: 1px solid var(--pink-300);
    border-radius: 16px;
    box-shadow: 0 10px 30px var(--soft2);
  }

  h1 {
    margin: 4px 0 18px;
    text-align: center;
    font-weight: 800;
    letter-spacing: .5px;
    background: linear-gradient(90deg, var(--pink-500), var(--pink-300));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
  }

  label { display:block; margin: 8px 0 6px; font-weight: 700; color: var(--pink-600); }

  input[type="file"], select {
    display: inline-block;
    padding: 10px 12px;
    border-radius: 12px;
    border: 1.5px solid var(--pink-400);
    background: linear-gradient(180deg, var(--pink-100), var(--pink-50));
    color: var(--ink);
    outline: none;
    transition: box-shadow .15s ease, border-color .15s ease, transform .06s ease;
  }
  input[type="file"]:focus, select:focus {
    box-shadow: 0 0 0 4px var(--soft);
    border-color: var(--pink-500);
  }

  .row { display:flex; gap:16px; flex-wrap:wrap; align-items: center; }

  button {
    padding: 12px 18px;
    border-radius: 12px;
    border: 0;
    cursor: pointer;
    transition: transform .06s ease, box-shadow .15s ease, opacity .15s ease;
    will-change: transform;
    color: #000; /* readable on pink gradients */
  }
  button:active { transform: translateY(1px); }
  button:disabled { opacity: .6; cursor: wait; }

  .primary {
    background: linear-gradient(135deg, var(--pink-300), var(--pink-500));
    box-shadow: 0 10px 24px var(--soft);
  }
  .primary:hover { box-shadow: 0 14px 28px var(--soft2); }

  .ghost {
    background: transparent;
    border: 1.5px solid var(--pink-500);
    color: var(--ink);
  }
  .ghost:hover { background: var(--soft); }

  /* Output panel: black text, pink background */
  #output {
    white-space: pre-wrap;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    background: linear-gradient(180deg, var(--pink-100), var(--pink-50));
    color: #000000 !important;
    padding: 14px;
    border-radius: 12px;
    border: 1.5px solid var(--pink-300);
    max-height: 60vh;
    overflow: auto;
    box-shadow: inset 0 1px 0 var(--soft);
  }

  .controls { display:flex; gap:16px; align-items:center; justify-content:center; margin-top: 16px; }
</style>
</head>
<body>
  <div class="card">
    <h1>loopy lyrics</h1>

    <form id="form">
      <label>Audio file</label>
      <input id="file" name="file" type="file" accept="audio/*,video/*" required />

      <div class="row" style="margin-top:8px;">
        <div>
          <label>Preprocess</label>
          <select name="prep">
            <option value="true" selected>true</option>
            <option value="false">false</option>
          </select>
        </div>
      </div>

      <div class="controls">
        <button id="btn" class="primary" type="submit">Transcribe</button>
        <button id="dl" class="ghost" type="button" disabled>Download JSON</button>
      </div>
    </form>

    <h3 style="margin:20px 0 8px; color: var(--pink-600);">Output</h3>
    <pre id="output">—</pre>
  </div>

<script>
const form = document.getElementById('form');
const btn  = document.getElementById('btn');
const out  = document.getElementById('output');
const dl   = document.getElementById('dl');

let lastJSON = null;

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(form);
  const fileInput = document.getElementById('file');
  if (!fileInput.files.length) { alert('Select a file'); return; }

  out.textContent = 'Processing… first run may download models.';
  btn.disabled = true; dl.disabled = true;

  try {
    const res = await fetch('/transcribe', { method: 'POST', body: fd });
    const data = await res.json();
    lastJSON = data;
    dl.disabled = false;
    out.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    console.error(err);
    out.textContent = 'Error: ' + (err?.message || err);
  } finally {
    btn.disabled = false;
  }
});

dl.addEventListener('click', () => {
  if (!lastJSON) return;
  const blob = new Blob([JSON.stringify(lastJSON, null, 2)], {type: 'application/json'});
  const url  = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  const ts = new Date().toISOString().replace(/[:.]/g,'-');
  a.download = `lyrics_${ts}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
});
</script>
</body>
</html>
"""


@app.get("/")
def index():
    return render_template_string(INDEX_HTML)







if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8190")), debug=False)
