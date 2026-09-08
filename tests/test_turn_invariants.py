"""Guard the contract between a turn's text and its words.

`transcript.vtt` is written from `words`, and every word-level score reads
`words`, so the two have to tokenise identically. When they drifted apart
nothing failed — the VTT just repeated words at turn boundaries and every
metric gained a ~4% phantom denominator.

Covers the mechanism that broke it (`_realign_changed` handing one word to two
turns) and the invariant itself, and checks the shipped result when one exists.

    docker compose run --rm app python tests/test_turn_invariants.py
"""

from __future__ import annotations

import json
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import pipeline as P  # noqa: E402

# Optional: if you point this at a finished job, the shipped result.json is
# checked too. Without it the synthetic cases still run, so this is a real
# test on a clean clone rather than one that needs my machine.
_out = os.environ.get("BENCH_OUT")
OUT = Path(_out) if _out else None


@dataclass
class Turn:
    start: float
    end: float
    text: str
    language: str = "cs"
    speaker: str = "S"
    words: list = field(default_factory=list)

    @property
    def clock(self) -> str:
        m, s = divmod(int(self.start), 60)
        return f"{m:02d}:{s:02d}"


def test_no_word_claimed_twice() -> bool:
    """Two adjacent re-aligned turns must not both take a boundary word."""
    turns = [Turn(0.0, 2.0, "one two"), Turn(2.0, 4.0, "three four")]
    aligned = [{"words": [
        {"word": "one", "start": 0.10, "end": 0.50},
        {"word": "two", "start": 1.60, "end": 1.95},    # midpoint 1.78, in turn 0
        {"word": "three", "start": 2.05, "end": 2.40},  # midpoint 2.23, in turn 1
        {"word": "four", "start": 3.00, "end": 3.40},
    ]}]
    P.align_mod = types.SimpleNamespace(align_segments=lambda s, a, d: (aligned, None))
    P._realign_changed(turns, ["x", "y"], None, "cpu")

    total = sum(len(t.words) for t in turns)
    ok = total == 4
    print(f"{'PASS' if ok else 'FAIL'}  boundary word claimed once "
          f"(got {total} words from 4)")
    for t in turns:
        print(f"        [{t.clock}] {t.text!r} -> {[w['word'] for w in t.words]}")
    return ok


def test_invariant_detects_a_duplicate() -> bool:
    """The checker must actually catch a mismatch, not just return empty."""
    good = Turn(0.0, 2.0, "one two",
                words=[{"word": "one"}, {"word": "two"}])
    dup = Turn(2.0, 4.0, "three four",
               words=[{"word": "two"}, {"word": "three"}, {"word": "four"}])
    empty = Turn(4.0, 5.0, "five", words=[])  # deliberate: not a defect

    bad = P.check_word_text_invariant([good, dup, empty])
    ok = bad == [1]
    print(f"{'PASS' if ok else 'FAIL'}  checker flags the duplicated turn only "
          f"(flagged {bad}, expected [1])")
    return ok


def test_shipped_result() -> bool:
    """The invariant must hold on a real transcript, not only synthetic ones."""
    if OUT is None:
        print("SKIP  set BENCH_OUT=output/<job> to also check a real transcript")
        return True
    path = OUT / "result.json"
    if not path.exists():
        print(f"SKIP  no result.json in {OUT}")
        return True
    turns = [Turn(t["start"], t["end"], t["text"], words=t.get("words") or [])
             for t in json.loads(path.read_text(encoding="utf-8"))["turns"]]
    bad = P.check_word_text_invariant(turns)
    ok = not bad
    print(f"{'PASS' if ok else 'FAIL'}  shipped result.json: "
          f"{len(bad)} of {len(turns)} turns mismatched")
    return ok


def main() -> int:
    results = [
        test_no_word_claimed_twice(),
        test_invariant_detects_a_duplicate(),
        test_shipped_result(),
    ]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
