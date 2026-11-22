"""Generate test files for the validation logic. The audio is synthetic tones, NOT speech,
so Whisper will return little or no text for it. For a real transcription test, use a
recording of someone talking (see README).

Run: python samples/make_samples.py
"""
from pathlib import Path
import numpy as np
import soundfile as sf

OUT = Path(__file__).parent
SR = 44_100
rng = np.random.default_rng(7)

def burst(dur):
    t = np.arange(int(dur * SR)) / SR
    phase = 2 * np.pi * np.cumsum(140 + 25 * np.sin(2 * np.pi * 0.7 * t)) / SR
    voice = sum(np.sin(k * phase) / k for k in range(1, 6))
    return 0.25 * voice * 0.5 * (1 + np.sin(2 * np.pi * 4 * t - np.pi / 2))

parts = []
for _ in range(6):
    parts += [burst(rng.uniform(2, 4)), np.zeros(int(rng.uniform(0.4, 1.2) * SR))]
mono = np.concatenate(parts)
stereo = np.stack([mono, mono * 0.9], axis=1).astype(np.float32)

sf.write(OUT / "tones.wav", stereo, SR, subtype="PCM_16")
sf.write(OUT / "tones.mp3", stereo, SR, format="MP3")
sf.write(OUT / "tones.ogg", stereo[: 5 * SR], SR, format="OGG")         # not allowed -> 415
(OUT / "tones_renamed.mp3").write_bytes((OUT / "tones.wav").read_bytes())  # WAV in disguise
(OUT / "not_audio.mp3").write_text("this is a text file pretending to be an mp3\n")
(OUT / "truncated.wav").write_bytes((OUT / "tones.wav").read_bytes()[:30])
print("wrote:", ", ".join(sorted(p.name for p in OUT.iterdir() if p.suffix != ".py")))
