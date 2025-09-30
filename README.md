## Run diarization

### From inside whisper-diarization/
python diarize.py -a ../data/song_input.mp3 --whisper-model small --language en --device cpu

## expected output
'''
{
  "segments": [
    {
      "id": 0,
      "speaker": "SPEAKER_00",
      "start": 0.7,
      "end": 2.8,
      "text": "You're glowing",
      "words": [
        {"word": "You're", "start": 0.706, "end": 1.107, "speaker": "SPEAKER_00"},
        {"word": "glowing", "start": 1.127, "end": 2.850, "speaker": "SPEAKER_00"}
      ]
    }
  ]
}
'''