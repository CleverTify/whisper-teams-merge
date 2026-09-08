"""Inspect and sweep pyannote's clustering hyperparameters.

Post-hoc merging of cluster centroids can only go so far. The pipeline's own
clustering threshold decides how eagerly it splits a speaker in the first
place, so tuning that is the more direct fix.

    docker compose run --rm app python scripts/tune_clustering.py output/<name> [t1 t2 ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

from app.audio import load_wav
from app.config import settings, setup_logging


def main() -> int:
    setup_logging()
    root = Path(sys.argv[1])
    if not root.is_absolute():
        root = Path("/work") / root
    thresholds = [float(x) for x in sys.argv[2:]] or []

    from pyannote.audio import Pipeline
    import torch

    pipe = Pipeline.from_pretrained(settings.diarize_model, token=settings.hf_token)
    pipe.to(torch.device("cuda"))

    print("=== pipeline parameters ===")
    try:
        params = pipe.parameters(instantiated=True)
        for k, v in (params or {}).items():
            print(f"  {k}: {v}")
    except Exception as exc:
        print(f"  (could not read: {exc})")

    print("\n=== pipeline structure ===")
    for name in dir(pipe):
        if name.startswith("_"):
            continue
        obj = getattr(pipe, name, None)
        if obj is not None and name in ("clustering", "segmentation", "embedding",
                                        "segmentation_batch_size",
                                        "embedding_batch_size", "klustering"):
            print(f"  {name}: {obj}")

    if not thresholds:
        print("\n(pass thresholds to sweep, e.g. `... output/x 0.5 0.7 0.9`)")
        return 0

    audio = load_wav(root / "work" / "audio.wav")
    wav = {"waveform": torch.from_numpy(audio[None, :]), "sample_rate": 16000}
    print(f"\naudio: {len(audio)/16000/60:.1f} min")

    base = pipe.parameters(instantiated=True) or {}
    for t in thresholds:
        params = dict(base)
        if "clustering" in params and isinstance(params["clustering"], dict):
            params["clustering"] = {**params["clustering"], "threshold": t}
        else:
            print(f"  threshold {t}: pipeline exposes no clustering.threshold — skipping")
            continue
        try:
            pipe.instantiate(params)
            out = pipe(wav)
            # pyannote 4.x returns DiarizeOutput; .speaker_diarization is the
            # Annotation, and there is also an exclusive (overlap-free) variant.
            ann = getattr(out, "speaker_diarization", out)
            spk = sorted(ann.labels())
            durs = {s: sum(seg.duration for seg, _, l in ann.itertracks(yield_label=True)
                           if l == s) for s in spk}
            total = sum(durs.values()) or 1
            share = ", ".join(f"{s[-2:]}={durs[s]/total*100:.0f}%"
                              for s in sorted(spk, key=lambda x: -durs[x]))
            print(f"  threshold {t:<5} -> {len(spk)} speaker(s)   {share}")
        except Exception as exc:
            print(f"  threshold {t:<5} -> FAILED: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
