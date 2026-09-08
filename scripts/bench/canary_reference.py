"""Transcribe the benchmark audio with NVIDIA Canary-1B-v2, as a fourth engine.

This is a measurement, not an integration. Canary claims 7.86% WER on Czech
FLEURS against Whisper large-v3's ~12%, but FLEURS is clean read speech and this
audio is a 16 kHz 59 kbps meeting with crosstalk. Whether the margin transfers
is exactly the question, and the only honest way to answer it is to run it.

Output is the same JSON shape `scripts/fetch_soniox.py` writes, so
`scripts/bench_wording.py` scores it with no new parsing code — as stream "D",
judged on the identical Teams-and-Soniox witness set that A0 is judged on.

Chunking: Canary was trained on utterances up to 40 s and the encoder's memory
grows with the square of the input, so the audio is cut into windows. The
windows OVERLAP, and each word is kept by whichever window's exclusive middle
region contains it. Cutting without overlap would slice a word at every
boundary — ~49 damaged words here, which is half the total error being measured
and would sink Canary for a reason that has nothing to do with Canary.

Run:
  docker run --rm --gpus all -v "$PWD:/work" -v "$PWD/cache:/cache" \
      transcribe:canary /work/scripts/canary_reference.py
"""

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Importable from anywhere: sibling helpers first, then the repo root so
# `app` resolves without relying on PYTHONPATH being set.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import corpus  # noqa: E402




AUDIO = corpus.audio()
OUT = corpus.results() / "canary" / "cs-plain.json"
MODEL = "nvidia/canary-1b-v2"

CHUNK = 120.0    # seconds of audio per window
OVERLAP = 6.0    # shared between neighbours; half is discarded on each side
SR = 16000


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def cut(path: Path, start: float, length: float, dest: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
         "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(path),
         "-ac", "1", "-ar", str(SR), str(dest)],
        check=True,
    )


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        print("no CUDA; refusing to produce a reference from a CPU run")
        return 1
    print(f"{torch.cuda.get_device_name(0)}  torch {torch.__version__}")

    total = duration(AUDIO)
    starts = []
    t = 0.0
    while t < total:
        starts.append(t)
        t += CHUNK - OVERLAP
    print(f"audio {total:.1f}s -> {len(starts)} windows "
          f"of {CHUNK:.0f}s overlapping {OVERLAP:.0f}s")

    from nemo.collections.asr.models import ASRModel

    model = ASRModel.from_pretrained(MODEL)
    model.eval()

    tmp = Path(tempfile.mkdtemp())
    words: list[dict] = []
    began = time.time()

    for i, start in enumerate(starts):
        length = min(CHUNK, total - start)
        if length < 0.5:
            continue
        clip = tmp / f"chunk{i:03d}.wav"
        cut(AUDIO, start, length, clip)

        out = model.transcribe([str(clip)], source_lang="cs", target_lang="cs",
                               timestamps=True, batch_size=1, verbose=False)
        stamps = getattr(out[0], "timestamp", None) or {}
        got = stamps.get("word") or []
        if not got:
            print(f"  window {i}: NO WORD TIMESTAMPS — aborting rather than "
                  f"emitting a reference with invented times")
            return 2

        # Exclusive region: drop the half-overlap this window shares with each
        # neighbour, so every word is claimed exactly once.
        lo = start + (OVERLAP / 2 if i > 0 else 0.0)
        hi = start + length - (OVERLAP / 2 if i < len(starts) - 1 else 0.0)
        kept = 0
        for w in got:
            mid = start + (float(w["start"]) + float(w["end"])) / 2
            if not (lo <= mid < hi):
                continue
            text = str(w["word"]).strip()
            if not text:
                continue
            words.append({
                "text": text,
                "start_ms": int(round((start + float(w["start"])) * 1000)),
                "end_ms": int(round((start + float(w["end"])) * 1000)),
                "confidence": 1.0,
                "speaker": None,
            })
            kept += 1
        clip.unlink(missing_ok=True)
        rate = (start + length) / max(time.time() - began, 1e-6)
        print(f"  window {i + 1}/{len(starts)}  {start:7.1f}s  "
              f"{len(got):4d} words -> kept {kept:4d}   ({rate:.1f}x realtime)")

    words.sort(key=lambda w: w["start_ms"])

    # `from_soniox` rejoins token text and asserts it equals `text`, so the
    # separator has to live inside the tokens.
    for w in words[:-1]:
        w["text"] += " "
    text = "".join(w["text"] for w in words)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "config": "canary",
        "model": MODEL,
        "request": {"source_lang": "cs", "target_lang": "cs",
                    "chunk_seconds": CHUNK, "overlap_seconds": OVERLAP},
        "audio": AUDIO.name,
        "transcript": {"id": "canary-local", "text": text, "tokens": words},
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    took = time.time() - began
    print(f"\n{len(words)} words in {took/60:.1f} min "
          f"({total/max(took,1e-6):.1f}x realtime) -> {OUT}")
    print(f"first 200 chars: {text[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
