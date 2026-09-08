"""Does Canary-1B-v2 run on your GPU, and does it give word timestamps?

Two questions, cheaply, before writing the full benchmark run:

1. sm_120. The torch wheels are cu128 so this should work, but "should" has
   been wrong about sm_120 before.
2. `timestamps=True`. NeMo issue #14877 raises `'tuple' object has no
   attribute 'audio'` on some versions. The whole chunking strategy depends on
   the answer: with word times the reference can use 5-minute windows with an
   overlap, without them it has to interpolate and the benchmark's ±4 s pairing
   window stops meaning anything.

Run:
  docker run --rm --gpus all -v "$PWD:/work" -v "$PWD/cache:/cache" \
      transcribe:canary /work/scripts/canary_probe.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

# Importable from anywhere: sibling helpers first, then the repo root so
# `app` resolves without relying on PYTHONPATH being set.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import corpus  # noqa: E402




AUDIO = corpus.audio()
SECONDS = 30.0


def main() -> int:
    import torch

    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        print(f"device: {torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}")
        # sm_120 silently falls back to no kernel at all on mismatched wheels.
        free, total = torch.cuda.mem_get_info()
        print(f"vram: {free / 2**30:.2f} GiB free of {total / 2**30:.2f} GiB")
    else:
        print("no CUDA — stopping, a CPU run answers neither question")
        return 1

    tmp = Path(tempfile.mkdtemp())
    wav = tmp / "probe.wav"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
         "-ss", "600", "-t", str(SECONDS), "-i", str(AUDIO),
         "-ac", "1", "-ar", "16000", str(wav)],
        check=True,
    )
    print(f"probe clip: {wav} ({wav.stat().st_size} bytes)")

    from nemo.collections.asr.models import ASRModel

    model = ASRModel.from_pretrained("nvidia/canary-1b-v2")
    model.eval()
    print(f"model loaded: {type(model).__name__}")

    for ts in (True, False):
        try:
            out = model.transcribe(
                [str(wav)], source_lang="cs", target_lang="cs",
                timestamps=ts, batch_size=1,
            )
            first = out[0]
            text = getattr(first, "text", first)
            print(f"\ntimestamps={ts}: OK")
            print(f"  text: {str(text)[:200]}")
            stamps = getattr(first, "timestamp", None)
            if stamps:
                words = stamps.get("word") or []
                print(f"  word stamps: {len(words)}")
                for w in words[:5]:
                    print(f"    {w}")
            else:
                print("  no .timestamp payload")
        except Exception as exc:  # noqa: BLE001 - this is the question being asked
            print(f"\ntimestamps={ts}: FAILED  {type(exc).__name__}: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
