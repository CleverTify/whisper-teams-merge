"""Run diarization on an already-prepared job and report cluster similarity.

Answers the question "are these really N different people?" by comparing the
speaker embeddings pyannote produces. Genuinely different voices sit far apart;
clusters that are one person split in two sit close together.

    docker compose run --rm app python scripts/probe_diarization.py output/<name> [min max]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from app.audio import load_wav
from app.config import settings, setup_logging
from app.diarize import diarize


def cosine(a, b) -> float:
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def main() -> int:
    setup_logging()
    root = Path(sys.argv[1])
    if not root.is_absolute():
        root = Path("/work") / root
    lo = int(sys.argv[2]) if len(sys.argv) > 2 else None
    hi = int(sys.argv[3]) if len(sys.argv) > 3 else None

    audio = load_wav(root / "work" / "audio.wav")
    print(f"audio: {len(audio)/16000/60:.1f} min   constraint: min={lo} max={hi}\n")

    df, emb = diarize(
        audio, hf_token=settings.hf_token, model_name=settings.diarize_model,
        device="cuda", min_speakers=lo, max_speakers=hi,
    )

    speakers = sorted(df["speaker"].unique())
    talk = df.groupby("speaker").apply(lambda g: (g["end"] - g["start"]).sum())
    total = float(talk.sum()) or 1.0
    print(f"\n=== {len(speakers)} cluster(s) ===")
    for s in speakers:
        print(f"  {s:<12}{talk[s]:8.1f}s  {talk[s]/total*100:5.1f}%")

    if emb is None:
        print("\n(no embeddings returned — cannot compare clusters)")
        return 0

    vecs = {}
    if isinstance(emb, dict):
        vecs = {str(k): np.asarray(v, float).ravel() for k, v in emb.items()}
    else:
        arr = np.asarray(emb, float)
        if arr.ndim == 2:
            vecs = {speakers[i]: arr[i] for i in range(min(len(speakers), arr.shape[0]))}

    if len(vecs) < 2:
        print("\n(not enough embeddings to compare)")
        return 0

    labels = sorted(vecs)
    print("\n=== cluster similarity (cosine) ===")
    print("        " + "".join(f"{l[-2:]:>8}" for l in labels))
    for a in labels:
        print(f"  {a[-6:]:<6}" + "".join(f"{cosine(vecs[a], vecs[b]):>8.3f}" for b in labels))

    pairs = sorted(
        ((cosine(vecs[a], vecs[b]), a, b)
         for i, a in enumerate(labels) for b in labels[i + 1:]),
        reverse=True,
    )
    print("\n=== most similar pairs ===")
    for sim, a, b in pairs[:5]:
        verdict = ("SAME PERSON — clusters should merge" if sim > 0.55
                   else "borderline" if sim > 0.35 else "distinct voices")
        print(f"  {a} vs {b}: {sim:.3f}   {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
