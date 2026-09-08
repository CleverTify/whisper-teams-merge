"""Where the benchmark corpus lives — on your machine, not on mine.

Every script in this directory was written against one recording, and each one
used to hardcode that recording's path as a module constant. That made the whole
directory dead weight to anyone else: the scripts ran, found nothing, and failed
somewhere deep. They now ask here instead.

Point them at your own material, either way round:

    export BENCH_OUT=output/my-meeting
    export BENCH_VTT="input/my meeting.vtt"
    python scripts/bench/bench_wording.py

    python scripts/bench/bench_wording.py output/my-meeting "input/my meeting.vtt"

`BENCH_MEDIA` (the source recording) and `BENCH_AUDIO` (a decoded copy for the
Canary comparison) are only needed by the scripts that re-run the pipeline.
"""

import os
import sys
from pathlib import Path

# Inside the container the repo is at /work; outside it, wherever you cloned it.
ROOT = Path(os.environ.get("REPO_ROOT", "/work"))
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[2]


def _resolve(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def _require(kind: str, value: str | None, env: str, example: str) -> Path:
    if not value:
        sys.exit(
            f"no {kind} given.\n"
            f"  pass it as an argument, or set {env}\n"
            f"  e.g. {env}={example}"
        )
    path = _resolve(value)
    if not path.exists():
        sys.exit(f"{kind} not found: {path}")
    return path


def out_dir(argv: list[str] | None = None, pos: int = 1) -> Path:
    """A finished job directory under output/."""
    argv = sys.argv if argv is None else argv
    given = argv[pos] if len(argv) > pos else os.environ.get("BENCH_OUT")
    return _require("output directory", given, "BENCH_OUT", "output/my-meeting")


def teams_vtt(argv: list[str] | None = None, pos: int = 2) -> Path:
    """The Teams transcript for that same recording."""
    argv = sys.argv if argv is None else argv
    given = argv[pos] if len(argv) > pos else os.environ.get("BENCH_VTT")
    return _require("Teams .vtt", given, "BENCH_VTT", '"input/my meeting.vtt"')


def media() -> Path:
    """The original recording — only for scripts that re-run the pipeline."""
    return _require("media file", os.environ.get("BENCH_MEDIA"), "BENCH_MEDIA",
                    '"input/my meeting.mp4"')


def audio() -> Path:
    """A decoded copy of the audio, for backends that want a plain file."""
    return _require("audio file", os.environ.get("BENCH_AUDIO"), "BENCH_AUDIO",
                    "output/my-meeting/work/audio.wav")


def results() -> Path:
    """Where scoring output is written. Untracked; see .gitignore."""
    d = _resolve(os.environ.get("BENCH_RESULTS", "bench"))
    d.mkdir(parents=True, exist_ok=True)
    return d
