"""Replace faster-whisper with a small fake so tests run offline in seconds,
without downloading a model. Everything else (validation, decoding, timestamps,
HTTP handling) runs for real."""
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace as N

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


class FakeWhisperModel:
    def __init__(self, *args, **kwargs):
        pass

    def transcribe(self, audio, **kwargs):
        segs = [N(start=0.0, end=2.1, text=" Hello there.", avg_logprob=-0.1),
                N(start=2.0, end=2.0, text=" Overlapping.", avg_logprob=-0.3),  # bad times on purpose
                N(start=3661.5, end=3664.25, text=" Much later.", avg_logprob=-0.9)]
        return iter(segs), N(language="en", language_probability=0.98, duration=len(audio) / 16000)


fake = ModuleType("faster_whisper")
fake.WhisperModel = FakeWhisperModel
sys.modules["faster_whisper"] = fake

if not (ROOT / "samples" / "tones.wav").exists():
    import runpy
    runpy.run_path(str(ROOT / "samples" / "make_samples.py"))
