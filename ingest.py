# ingest.py: accepts WAV/MP3 (CLI or HTTP), validates by content, normalizes to 16 kHz mono.
# Decoding uses soundfile (libsndfile) + soxr for resampling: pip-installable, no ffmpeg.
import json, sys, tempfile, uuid
from pathlib import Path
import numpy as np
import soundfile as sf
import soxr
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

ALLOWED = {"WAV", "MP3"}             # format names as libsndfile detects them from file contents
MAX_BYTES = 200 * 1024 * 1024
MAX_SECONDS = 4 * 3600
SR = 16_000                          # what Whisper-family models expect

class AudioError(Exception):
    def __init__(self, msg, status=422):
        super().__init__(msg); self.status = status

def probe(path):
    """Read the real format from the file header (ignore extension/Content-Type). No decode."""
    try:
        info = sf.info(str(path))
    except RuntimeError:             # LibsndfileError: unknown, corrupt, empty or missing
        raise AudioError("file is not decodable audio")
    if info.format not in ALLOWED:
        raise AudioError(f"format '{info.format}' not accepted", 415)
    if info.duration > MAX_SECONDS:
        raise AudioError(f"audio is {info.duration:.0f}s; limit is {MAX_SECONDS}s")
    return {"format": info.format.lower(), "codec": info.subtype, "duration_s": round(info.duration, 3),
            "channels": info.channels, "sample_rate": info.samplerate}

def load(path):
    """Probe, then decode to mono float32 at 16 kHz."""
    info = probe(path)
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except RuntimeError:
        raise AudioError("decode failed")
    audio = audio.mean(axis=1)                       # downmix stereo/multichannel to mono
    if sr != SR:
        audio = soxr.resample(audio, sr, SR, quality="HQ")
    return info, np.ascontiguousarray(audio, dtype=np.float32)

async def save_upload(file, dest):
    """Stream to disk in 1 MB blocks, enforcing the size cap while reading."""
    size = 0
    with open(dest, "wb") as f:
        while block := await file.read(1 << 20):
            size += len(block)
            if size > MAX_BYTES:
                raise HTTPException(413, "file too large")
            f.write(block)
    if size == 0:
        raise HTTPException(400, "file is empty")

app = FastAPI()

@app.post("/v1/audio", status_code=201)
async def accept_audio(file: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as tmp:          # always cleaned up
        path = Path(tmp) / uuid.uuid4().hex             # never trust the client's filename
        await save_upload(file, path)
        try:
            info, audio = await run_in_threadpool(load, path)   # CPU-bound: off the event loop
        except AudioError as e:
            raise HTTPException(e.status, str(e))
    return {"status": "accepted", "source": info,
            "normalized": {"sample_rate": SR, "channels": 1, "samples": len(audio)}}

if __name__ == "__main__":                              # CLI: python ingest.py file.mp3
    try:
        print(json.dumps(load(sys.argv[1])[0], indent=2))
    except AudioError as e:
        sys.exit(f"error: {e}")
