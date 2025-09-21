Install deps
# macOS: ffmpeg
brew install ffmpeg

# create env
conda create -n whisperx-server python=3.10 -y
conda activate whisperx-server

# PyTorch (CPU wheel shown; use CUDA wheel if you have NVIDIA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# App deps
pip install -r requirements.txt

Run the server
# PORT=8190 python app.py, last line
python app.py

OR

API:
curl -X POST "http://localhost:8190/transcribe" \
  -F "file=........song.wav" \
  -F "prep=true" | jq .