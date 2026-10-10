"""Standard timestamp datafile shared by the analyser and any visualiser.

Format (JSON, version 1):
{
  "format": "band_timestamps",
  "version": 1,
  "audio_file": "song.wav",
  "duration": 183.2,
  "analysis": {...free-form settings used to produce the data...},
  "bands": [
    {"index": 0, "name": "bass", "low_hz": 60, "high_hz": 250,
     "events": [{"t": 0.512, "strength": 0.83}, ...]},
    ...
  ]
}
"""
import json

FORMAT_NAME = "band_timestamps"
FORMAT_VERSION = 1


def build_datafile(audio_file, duration, analysis_settings, bands):
    return {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "audio_file": audio_file,
        "duration": float(duration),
        "analysis": analysis_settings,
        "bands": bands,
    }


def save_datafile(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)


def load_datafile(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("format") != FORMAT_NAME:
        raise ValueError("Not a band_timestamps datafile")
    if data.get("version", 0) > FORMAT_VERSION:
        raise ValueError("Datafile version is newer than supported")
    for band in data["bands"]:
        band["events"] = sorted(band["events"], key=lambda e: e["t"])
    return data
