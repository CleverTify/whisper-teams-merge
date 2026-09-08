"""Sweep the alignment constants rather than trusting the first guess.

Window and pad govern how much context the word matcher sees; the fill gap
governs how far an unmatched run may inherit a neighbour's owner before falling
back to the clock. All three were picked by hand, so all three get checked.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


from app import teams as teams_mod  # noqa: E402

# Importable from anywhere: sibling helpers first, then the repo root so
# `app` resolves without relying on PYTHONPATH being set.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import corpus  # noqa: E402

from app.diarize import build_turns  # noqa: E402
from bench_speakers import integrity, our_words, score, teams_words  # noqa: E402



OUT = corpus.out_dir()
VTT = corpus.teams_vtt()
TURN_ARGS = dict(merge_gap=1.0, smooth=3, min_turn_seconds=0.7, snap_window=3)


def run(base, cues, ref, **consts) -> tuple[float, float, int]:
    for k, v in consts.items():
        setattr(teams_mod, k, v)
    segs = teams_mod.assign_speakers(copy.deepcopy(base), cues)
    sc = score(our_words(segs), ref)
    integ = integrity(build_turns(segs, **TURN_ARGS))
    return sc["accuracy"], integ["cut_rate"], integ["short_turns"]


def main() -> None:
    raw = json.loads((OUT / "work" / "align.json").read_text(encoding="utf-8"))
    base = raw["segments"] if isinstance(raw, dict) else raw
    cues = teams_mod.load(VTT)
    ref = teams_words(cues)

    defaults = dict(ALIGN_WINDOW=60.0, ALIGN_PAD=10.0, FILL_MAX_GAP=8)

    for knob, values in (
        ("ALIGN_WINDOW", [20.0, 30.0, 45.0, 60.0, 90.0, 120.0, 300.0]),
        ("ALIGN_PAD", [0.0, 5.0, 10.0, 20.0, 30.0]),
        ("FILL_MAX_GAP", [1, 2, 4, 8, 16, 32, 10**6]),
    ):
        print(f"\n=== {knob} ===")
        print(f"  {'value':>10}  {'accuracy':>9}  {'cut rate':>9}  {'short turns':>11}")
        for v in values:
            acc, cut, short = run(base, cues, ref, **{**defaults, knob: v})
            mark = "  <- default" if v == defaults[knob] else ""
            label = "inf" if v == 10**6 else v
            print(f"  {label:>10}  {acc*100:8.2f}%  {cut*100:8.2f}%  {short:>11}{mark}")


if __name__ == "__main__":
    main()
