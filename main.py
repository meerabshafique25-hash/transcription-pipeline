"""One server with every endpoint from Q4-Q6.

Run:  uvicorn main:app --port 8000
Then open http://localhost:8000/docs to upload files from the browser.
"""
from fastapi import FastAPI
from fastapi.routing import APIRoute

import ingest, segments, transcribe   # importing transcribe loads the Whisper model once

app = FastAPI(title="Transcription pipeline",
              description="Upload WAV or MP3 audio. Endpoints: validate (/v1/audio), "
                          "transcribe (/v1/transcribe), transcribe with timestamps as "
                          "JSON/SRT/VTT (/v1/transcribe/segments).")

for module in (ingest, transcribe, segments):
    for route in module.app.routes:
        if isinstance(route, APIRoute):
            app.router.routes.append(route)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
