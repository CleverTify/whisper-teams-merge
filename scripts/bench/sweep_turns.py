"""Measure what turn-building does to speaker labels that arrived clean.

Two cautions about the numbers this prints.

The word-accuracy score is circular for matched words: the benchmark pairs our
words to Teams' words with the same windowed matcher the assignment uses, so a
word assigned that way scores correct by construction. What it can still measure
honestly is *damage* — take the assignment as the 100% baseline and see how much
of it the smoothing gives back, since both sides are scored identically.

The other two numbers are independent of that. `cut rate` asks whether a speaker
change lands mid-sentence, and `boundary error` compares our speaker-change
times against Teams' own, neither of which consults the alignment.
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


def teams_turns(cues) -> list[tuple[float, str]]:
    """Teams' own speaker changes: (time, new speaker)."""
    out = []
    for c in cues:
        if not out or out[-1][1] != c.speaker:
            out.append((c.start, c.speaker))
    return out


def boundary_error(turns, ref: list[tuple[float, str]]) -> tuple[float, int]:
    """Median seconds between our speaker changes and Teams' nearest one."""
    ours = []
    for a, b in zip(turns, turns[1:]):
        sa = a["speaker"] if isinstance(a, dict) else a.speaker
        sb = b["speaker"] if isinstance(b, dict) else b.speaker
        if sa != sb:
            ours.append(b["start"] if isinstance(b, dict) else b.start)
    if not ours or not ref:
        return 0.0, 0
    times = [t for t, _ in ref]
    deltas = []
    for t in ours:
        deltas.append(min(abs(t - r) for r in times))
    deltas.sort()
    return deltas[len(deltas) // 2], sum(1 for d in deltas if d > 2.0)


def main() -> None:
    raw = json.loads((OUT / "work" / "align.json").read_text(encoding="utf-8"))
    base = raw["segments"] if isinstance(raw, dict) else raw
    cues = teams_mod.load(VTT)
    ref = teams_words(cues)
    ref_turns = teams_turns(cues)

    assigned = teams_mod.assign_speakers(copy.deepcopy(base), cues)
    pre = score(our_words(assigned), ref)
    print(f"assignment before any turn-building: {pre['accuracy']*100:.2f}% "
          f"({pre['matched']} aligned words)\n")

    configs = [
        ("current defaults",  dict(merge_gap=1.0, smooth=3, min_turn_seconds=0.7, snap_window=3)),
        ("no word smoothing", dict(merge_gap=1.0, smooth=0, min_turn_seconds=0.7, snap_window=3)),
        ("no short-run absorb", dict(merge_gap=1.0, smooth=3, min_turn_seconds=0.0, snap_window=3)),
        ("no sentence snap",  dict(merge_gap=1.0, smooth=3, min_turn_seconds=0.7, snap_window=0)),
        ("nothing but merge", dict(merge_gap=1.0, smooth=0, min_turn_seconds=0.0, snap_window=0)),
        ("snap only",         dict(merge_gap=1.0, smooth=0, min_turn_seconds=0.0, snap_window=3)),
        ("absorb only",       dict(merge_gap=1.0, smooth=0, min_turn_seconds=0.7, snap_window=0)),
        ("gentle smooth",     dict(merge_gap=1.0, smooth=2, min_turn_seconds=0.3, snap_window=2)),
    ]

    print(f"  {'config':<22} {'kept':>7} {'cut rate':>9} {'turns':>7} {'short':>6} "
          f"{'bnd err':>8} {'>2s':>5}")
    for label, args in configs:
        segs = copy.deepcopy(assigned)
        turns = build_turns(segs, **args)
        sc = score(our_words(segs), ref)
        integ = integrity(turns)
        med, far = boundary_error(turns, ref_turns)
        print(f"  {label:<22} {sc['accuracy']*100:6.2f}% {integ['cut_rate']*100:8.2f}% "
              f"{integ['turns']:>7} {integ['short_turns']:>6} {med:>7.2f}s {far:>5}")


if __name__ == "__main__":
    main()
