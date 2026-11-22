from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import ingest
from main import app
from segments import clean, ts

S = Path(__file__).parent.parent / "samples"
client = TestClient(app)


def upload(url, name, as_name=None):
    return client.post(url, files={"file": (as_name or name, (S / name).read_bytes())})


# Q4: accepting audio ---------------------------------------------------------

@pytest.mark.parametrize("name", ["tones.wav", "tones.mp3"])
def test_accepts_and_normalizes(name):
    info, audio = ingest.load(S / name)
    assert info["channels"] == 2 and info["sample_rate"] == 44_100
    assert audio.dtype.name == "float32" and audio.ndim == 1
    assert abs(len(audio) / 16_000 - info["duration_s"]) < 0.1


def test_format_comes_from_content_not_extension():
    assert ingest.load(S / "tones_renamed.mp3")[0]["format"] == "wav"


@pytest.mark.parametrize("name,code", [
    ("tones.wav", 201), ("tones.mp3", 201), ("tones.ogg", 415),
    ("not_audio.mp3", 422), ("truncated.wav", 422),
])
def test_upload_status_codes(name, code):
    assert upload("/v1/audio", name).status_code == code


def test_empty_upload():
    r = client.post("/v1/audio", files={"file": ("x.wav", b"")})
    assert r.status_code == 400


def test_size_limit(monkeypatch):
    monkeypatch.setattr(ingest, "MAX_BYTES", 10_000)
    assert upload("/v1/audio", "tones.mp3").status_code == 413


def test_duration_limit(monkeypatch):
    monkeypatch.setattr(ingest, "MAX_SECONDS", 5)
    assert upload("/v1/audio", "tones.wav").status_code == 422


# Q5: transcription -------------------------------------------------------------

def test_transcribe():
    body = upload("/v1/transcribe", "tones.mp3").json()
    assert body["language"] == "en"
    assert body["text"].startswith("Hello there.")
    assert [s["needs_review"] for s in body["segments"]] == [False, False, True]


def test_transcribe_rejects_bad_input():
    assert upload("/v1/transcribe", "not_audio.mp3").status_code == 422


# Q6: timestamps per segment -------------------------------------------------------

def test_timestamp_format():
    assert ts(0) == "00:00:00.000"
    assert ts(3661.5) == "01:01:01.500"
    assert ts(3661.5, ",") == "01:01:01,500"


def test_clean_makes_times_monotonic():
    segs = clean([{"start": 0, "end": 2.1, "text": "a"},
                  {"start": 2.0, "end": 2.0, "text": "b"},
                  {"start": 5, "end": 6, "text": ""}])
    assert len(segs) == 2
    assert segs[1]["start"] >= segs[0]["end"] and segs[1]["end"] > segs[1]["start"]


def test_segments_json():
    segs = upload("/v1/transcribe/segments", "tones.wav").json()["segments"]
    assert [s["id"] for s in segs] == [0, 1, 2]
    assert segs[2]["start_ts"] == "01:01:01.500"


def test_segments_srt_and_vtt():
    srt = client.post("/v1/transcribe/segments?format=srt",
                      files={"file": ("a.mp3", (S / "tones.mp3").read_bytes())})
    assert srt.headers["content-type"].startswith("application/x-subrip")
    assert srt.text.startswith("1\n00:00:00,000 --> 00:00:02,100\nHello there.")
    vtt = client.post("/v1/transcribe/segments?format=vtt",
                      files={"file": ("a.mp3", (S / "tones.mp3").read_bytes())})
    assert vtt.text.startswith("WEBVTT")


def test_segments_rejects_unknown_format():
    r = client.post("/v1/transcribe/segments?format=xml",
                    files={"file": ("a.mp3", (S / "tones.mp3").read_bytes())})
    assert r.status_code == 422
