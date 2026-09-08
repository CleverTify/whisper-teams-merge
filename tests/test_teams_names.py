"""Check that name-variant merging keeps both real cases straight.

The merge exists for one person whose Teams display name changes mid-call, and
it must not fire for two people whose names merely look alike. Both are real,
and the second one cost a speaker on an actual call before this check existed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.teams import TeamsCue, normalise_names  # noqa: E402


def cue(start: float, dur: float, speaker: str) -> TeamsCue:
    return TeamsCue(start, start + dur, speaker, "x")


def case_two_people() -> list[TeamsCue]:
    """A dialogue: the two labels trade turns all the way through."""
    out = []
    t = 0.0
    for i in range(60):
        out.append(cue(t, 3.0, "Jan Novák" if i % 2 == 0 else "Janek Novák"))
        t += 3.5
    return out


def case_device_handover() -> list[TeamsCue]:
    """One person, name changes once when they switch device. No alternation."""
    out = []
    t = 0.0
    for i in range(60):
        out.append(cue(t, 3.0, "Tomas Marek" if i < 30 else "Tomáš Marek"))
        t += 3.5
    return out


def case_two_devices_at_once() -> list[TeamsCue]:
    """One person on two devices: labels interleave *and* overlap heavily."""
    out = []
    t = 0.0
    for _ in range(30):
        out.append(cue(t, 3.0, "Petr Svoboda"))
        out.append(cue(t + 0.2, 3.0, "Petr Svobodá"))  # same words, both mics
        t += 3.5
    return out


def merged(cues: list[TeamsCue]) -> int:
    canon = normalise_names(cues)
    return len({canon[c.speaker] for c in cues})


def main() -> int:
    checks = [
        ("two people, similar names", case_two_people, 2),
        ("one person, device handover", case_device_handover, 1),
        ("one person, two devices at once", case_two_devices_at_once, 1),
    ]
    failed = 0
    for label, build, expected in checks:
        got = merged(build())
        ok = got == expected
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {label}: expected {expected}, got {got}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
