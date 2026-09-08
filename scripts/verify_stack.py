"""Build-time guard.

Fails the build in seconds rather than letting a broken stack surface halfway
through a 30-minute transcription. Which checks apply depends on the target:
the universal build has no CUDA and must not be held to CUDA requirements.
"""

import os
import sys


def version_tuple(text: str) -> tuple[int, ...]:
    parts = []
    for chunk in text.split(".")[:3]:
        digits = ""
        for ch in chunk:
            if not ch.isdigit():
                break
            digits += ch
        parts.append(int(digits or 0))
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def main() -> int:
    backend = os.environ.get("APP_BACKEND", "faster-whisper")
    problems: list[str] = []

    import torch

    print(f"backend={backend} torch={torch.__version__} torch.cuda={torch.version.cuda or 'cpu'}")

    if backend == "faster-whisper":
        import ctranslate2

        ct2 = ctranslate2.__version__
        cuda = torch.version.cuda or ""
        print(f"ctranslate2={ct2}")
        if not cuda.startswith("12.8"):
            problems.append(
                f"torch built for CUDA {cuda!r}; sm_120 (Blackwell) needs 12.8"
            )
        if version_tuple(ct2) < (4, 6, 3):
            problems.append(
                f"ctranslate2 {ct2} predates sm_120 support (need >=4.6.3; "
                "4.6.2 disabled INT8 for sm120, 4.6.3 added CUDA 12.8)"
            )
    else:
        # Universal target: the whisper.cpp binary must be present and runnable.
        import shutil
        import subprocess

        exe = shutil.which("whisper-cli")
        if not exe:
            problems.append("whisper-cli not found on PATH")
        else:
            proc = subprocess.run([exe, "--help"], capture_output=True, text=True)
            if proc.returncode not in (0, 1):
                problems.append(f"whisper-cli is not runnable (exit {proc.returncode})")
            else:
                print("whisper-cli: OK")

    # Both targets need these.
    for mod in ("whisperx", "pyannote.audio", "soundfile"):
        try:
            __import__(mod)
        except Exception as exc:
            problems.append(f"cannot import {mod}: {exc}")

    if problems:
        print("BUILD ABORTED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    print("OK: stack verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
