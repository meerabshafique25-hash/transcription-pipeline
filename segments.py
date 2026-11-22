# segments.py: transcription with timestamps per segment, as JSON, SRT or WebVTT
import json, sys, tempfile, uuid
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse
from ingest import AudioError, load, save_upload   # Q4: validation + normalization
from transcribe import transcribe                   # Q5: faster-whisper

def ts(seconds, sep="."):
    """Seconds -> HH:MM:SS.mmm (SRT uses a comma before the milliseconds)."""
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000); m, ms = divmod(ms, 60_000); s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02}{sep}{ms:03}"

def clean(segments):
    """Models occasionally emit overlapping or zero-length segments; make times monotonic."""
    out, prev_end = [], 0.0
    for s in segments:
        if not s["text"]:
            continue
        start = max(s["start"], prev_end)
        end = max(s["end"], start + 0.01)
        out.append({**s, "id": len(out), "start": start, "end": end,
                    "start_ts": ts(start), "end_ts": ts(end)})
        prev_end = end
    return out

def to_srt(segs):
    return "\n".join(f"{i}\n{ts(s['start'], ',')} --> {ts(s['end'], ',')}\n{s['text']}\n"
                     for i, s in enumerate(segs, 1))

def to_vtt(segs):
    return "WEBVTT\n\n" + "\n".join(f"{ts(s['start'])} --> {ts(s['end'])}\n{s['text']}\n" for s in segs)

def render(result, fmt):
    segs = clean(result["segments"])
    if fmt == "srt":
        return to_srt(segs)
    if fmt == "vtt":
        return to_vtt(segs)
    return {**result, "segments": segs}

app = FastAPI()

@app.post("/v1/transcribe/segments")
async def transcribe_segments(file: UploadFile = File(...),
                              format: str = Query("json", pattern="^(json|srt|vtt)$")):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / uuid.uuid4().hex
        await save_upload(file, path)
        try:
            _, audio = await run_in_threadpool(load, path)
        except AudioError as e:
            raise HTTPException(e.status, str(e))
    out = render(await run_in_threadpool(transcribe, audio), format)
    media = {"srt": "application/x-subrip", "vtt": "text/vtt"}
    return PlainTextResponse(out, media_type=media[format]) if format in media else out

if __name__ == "__main__":       # CLI: python segments.py file.mp3 [json|srt|vtt]
    fmt = sys.argv[2] if len(sys.argv) > 2 else "json"
    try:
        _, audio = load(sys.argv[1])
    except AudioError as e:
        sys.exit(f"error: {e}")
    out = render(transcribe(audio), fmt)
    print(out if isinstance(out, str) else json.dumps(out, indent=2))
