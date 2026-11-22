# transcribe.py: speech-to-text with faster-whisper (open-source Whisper on CTranslate2)
import json, math, sys, tempfile, uuid
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from faster_whisper import WhisperModel
from ingest import AudioError, load, save_upload    # validation from Q4

REVIEW_BELOW = 0.6
model = WhisperModel("small", device="auto", compute_type="int8")   # load once, not per request

def transcribe(audio):
    segments, info = model.transcribe(
        audio, beam_size=5,
        vad_filter=True,                    # skip silence; avoids hallucinated text on quiet parts
        condition_on_previous_text=False)   # stops one error repeating through the file
    segs = []
    for s in segments:                      # generator: decoding happens here
        conf = round(math.exp(s.avg_logprob), 3)
        segs.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(),
                     "confidence": conf, "needs_review": conf < REVIEW_BELOW})
    return {"language": info.language, "language_probability": round(info.language_probability, 3),
            "duration_s": round(info.duration, 2), "text": " ".join(s["text"] for s in segs),
            "segments": segs}

app = FastAPI()

@app.post("/v1/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / uuid.uuid4().hex
        await save_upload(file, path)
        try:
            _, audio = await run_in_threadpool(load, path)
        except AudioError as e:
            raise HTTPException(e.status, str(e))
    return await run_in_threadpool(transcribe, audio)

if __name__ == "__main__":                  # CLI: python transcribe.py file.mp3
    try:
        _, audio = load(sys.argv[1])
    except AudioError as e:
        sys.exit(f"error: {e}")
    print(json.dumps(transcribe(audio), indent=2))
