# Transcription pipeline

Converts WAV or MP3 audio into text with per-segment timestamps, using the open-source
Whisper model (via faster-whisper). Works as a web service or from the command line.

> Draft for human review before submission or production use.

## Quick start

You need **Python 3.10–3.12** installed. Nothing else (no ffmpeg, no GPU).

**macOS / Linux**

```bash
./run.sh
```

**Windows**

```
run.bat
```

That creates a virtual environment, installs the dependencies, and starts the server.
The **first start downloads the Whisper "small" model (~500 MB)**, so give it a minute or two.
After that, starts are quick.

Then open **http://localhost:8000/docs** in your browser. Pick an endpoint, click
**Try it out**, choose an audio file, and click **Execute**.

## Getting a recording to test with

The files in `samples/` are synthetic tones for testing validation. They are not speech,
so Whisper returns little or no text for them. For a real test, use a recording of someone
talking:

- **macOS:** generate one with the built-in voice:
  ```bash
  say -o test.wav --data-format=LEI16@16000 "Hello, this is a test of the transcription pipeline. It should return this sentence with timestamps."
  ```
- **Any system:** download a public-domain audiobook chapter MP3 from LibriVox.
- **Note:** phone voice memos are usually **M4A**, which this service does not accept
  (only WAV and MP3). Convert them first, or use one of the options above.

## Using it

**Endpoints** (all take a multipart file upload named `file`):

| Endpoint | What it does |
|---|---|
| `POST /v1/audio` | Validates and normalizes the file, returns its details (Q4) |
| `POST /v1/transcribe` | Returns full text, language, and segments with confidence (Q5) |
| `POST /v1/transcribe/segments?format=json` | Segments with timestamps as JSON, `srt` or `vtt` (Q6) |

**With curl:**

```bash
curl -F "file=@test.wav" http://localhost:8000/v1/transcribe
curl -F "file=@test.wav" "http://localhost:8000/v1/transcribe/segments?format=srt"
```

**From the command line, without the server** (activate the environment first with
`source .venv/bin/activate`, or `.venv\Scripts\activate` on Windows):

```bash
python ingest.py test.wav            # validate only
python transcribe.py test.wav        # transcript as JSON
python segments.py test.wav srt      # subtitles with timestamps
```

## Running the tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The tests replace Whisper with a small fake (see `tests/conftest.py`), so they run offline
in about a second. Everything else (validation, decoding, timestamp handling, HTTP status
codes) runs for real.

## Project layout

```
ingest.py       Q4: accept WAV/MP3, detect format from contents, normalize to 16 kHz mono
transcribe.py   Q5: speech-to-text with faster-whisper
segments.py     Q6: per-segment timestamps as JSON, SRT or WebVTT
main.py         one server combining all endpoints
samples/        generator for test files (good and deliberately broken)
tests/          automated tests
```

## Key decisions

- **Format detection by content.** The file header decides what the file is, not its
  extension, so a WAV renamed to `.mp3` is handled correctly.
- **soundfile + soxr instead of ffmpeg.** Much smaller code surface processing untrusted
  files and pip-installable. Trade-off: no M4A/AAC or audio inside video.
- **One canonical format.** Everything is converted to 16 kHz mono float32, what Whisper expects.
- **Clear errors:** empty file 400, too large 413, unsupported format 415, not valid audio 422.
- **Safe uploads.** Streamed to disk with a size cap, stored under a random name, always deleted.
- **Model loaded once** at startup, not per request.
- **Silence skipped** with the built-in voice activity detection, which avoids Whisper
  inventing text on quiet audio.
- **Low-confidence segments flagged** for human review rather than silently trusted.
- **Timestamps cleaned** so they are always in order, with no overlaps or zero-length segments.

## Settings

- **Model size:** change `"small"` in `transcribe.py`. Options: `tiny` and `base` (faster,
  less accurate), `medium`, `large-v3`, `large-v3-turbo` (more accurate, slower).
- **Limits and allowed formats:** `ALLOWED`, `MAX_BYTES` and `MAX_SECONDS` in `ingest.py`.

## Known limitations

- Requests are synchronous. Long recordings keep the request open until done; production
  would use a background job queue (return a job ID, poll for the result).
- Decoding runs inside the server process. Production would decode in a separate,
  time-limited worker process for isolation.
- The confidence score comes from the model's average log-probability. It's useful for
  flagging segments to review, but it isn't a calibrated probability.
