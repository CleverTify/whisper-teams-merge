"""A/B speaker-assignment strategies against the cached ASR output.

Speaker assignment reads the aligned words and the Teams cues and nothing else,
so a candidate can be scored without re-running the pipeline: load work/align.json,
apply the strategy, build turns, score. Seconds per iteration rather than minutes,
which is the only way to try more than two or three ideas.
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
from bench_speakers import integrity, our_words, report, score, teams_words  # noqa: E402



OUT = corpus.out_dir()
VTT = corpus.teams_vtt()

# build_turns defaults, so a strategy is judged on assignment not on tuning
TURN_ARGS = dict(merge_gap=1.0, smooth=3, min_turn_seconds=0.7, snap_window=3)


def by_time(segments, cues):
    """The original: each word goes to whoever overlapped it longest."""
    owner = teams_mod._by_overlap(cues)
    for seg in segments:
        for w in seg.get("words") or []:
            if w.get("start") is not None and w.get("end") is not None:
                w["speaker"] = owner(float(w["start"]), float(w["end"]))
        seg["speaker"] = owner(float(seg.get("start", 0.0)), float(seg.get("end", 0.0)))
    return segments


def by_alignment(segments, cues):
    """The candidate: words are matched to Teams' words, clocks only anchor."""
    return teams_mod.assign_speakers(segments, cues)


STRATEGIES = {"time-overlap (old)": by_time, "word-alignment (new)": by_alignment}


def main(names: list[str], turn_args: dict | None = None) -> None:
    args = {**TURN_ARGS, **(turn_args or {})}
    raw = json.loads((OUT / "work" / "align.json").read_text(encoding="utf-8"))
    base = raw["segments"] if isinstance(raw, dict) else raw
    cues = teams_mod.load(VTT)
    ref = teams_words(cues)
    print(f"reference: {len(ref)} Teams words | turn args: {args}")

    results = {}
    for name, fn in STRATEGIES.items():
        if names and name not in names:
            continue
        segs = fn(copy.deepcopy(base), cues)
        turns = build_turns(segs, **args)
        sc = score(our_words(segs), ref)
        results[name] = (sc, integrity(turns))
        report(name, sc, integrity(turns))

    if len(results) == 2:
        (an, (a, ai)), (bn, (b, bi)) = list(results.items())
        print(f"\n=== {bn} vs {an} ===")
        print(f"  speaker accuracy : {a['accuracy']*100:.2f}% -> {b['accuracy']*100:.2f}%"
              f"  ({(b['accuracy']-a['accuracy'])*100:+.2f} pts)")
        print(f"  sentences cut    : {ai['cut_rate']*100:.2f}% -> {bi['cut_rate']*100:.2f}%"
              f"  ({(bi['cut_rate']-ai['cut_rate'])*100:+.2f} pts)")
        print(f"  short turns      : {ai['short_turns']} -> {bi['short_turns']}")
        print(f"  turns            : {ai['turns']} -> {bi['turns']}")


if __name__ == "__main__":
    main(sys.argv[1:])
