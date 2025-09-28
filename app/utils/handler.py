import tempfile
import os
from pyannote.audio import Pipeline
import torch

# Load pipeline once (at import time)
HF_TOKEN = os.getenv("HF_TOKEN", None)

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)
# Send to GPU
pipeline.to(torch.device("cuda"))

def run_diarization(audio_bytes: bytes) -> dict:
    # Write to a temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        path = tmp.name

    diarization = pipeline(path)
    os.remove(path)

    # Convert to a serializable format (e.g. list of segments)
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": speaker
        })
    return {"segments": segments}
