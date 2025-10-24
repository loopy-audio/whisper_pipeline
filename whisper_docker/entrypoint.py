import os, sys, subprocess
from pathlib import Path

ROOT = Path("/app")
WDR = ROOT / "whisper-diarization"

def main():
    # If args are passed, forward directly to diarize.py
    if len(sys.argv) > 1:
        cmd = ["python", str(WDR / "diarize.py")] + sys.argv[1:]
        raise SystemExit(subprocess.call(cmd))

    # Defaults via env vars
    audio = os.environ.get("AUDIO", "data/song_input.mp3")
    whisper_model = os.environ.get("WHISPER_MODEL", "small")
    language = os.environ.get("LANGUAGE", "en")
    device = os.environ.get("DEVICE", "cpu")
    batch_size = os.environ.get("BATCH_SIZE", "0")

    cmd = [
        "python", str(WDR / "diarize.py"),
        "-a", audio,
        "--whisper-model", whisper_model,
        "--language", language,
        "--device", device,
        "--batch-size", batch_size,
    ]
    print("Running:", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
