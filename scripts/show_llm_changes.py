"""Show what the LLM actually changed, against the raw ASR and against Teams.

The guardrail only checks that a rewrite stayed close to the original. It cannot
tell an improvement from a confident invention, so the changes are worth reading
next to both other readings of the same seconds.
"""

from __future__ import annotations

import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path


from app.teams import load as load_teams, text_at  # noqa: E402

WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def norm(t: str) -> list[str]:
    return WORD.findall(t.lower())


def clock(t: float) -> str:
    m, s = divmod(int(t), 60)
    return f"{m:02d}:{s:02d}"


def main(out_dir: str, vtt: str, limit: int = 25) -> None:
    work = Path(out_dir, "work")
    raw = json.loads((work / "align.json").read_text(encoding="utf-8"))
    raw_segs = raw if isinstance(raw, list) else raw.get("segments", [])
    final = json.loads(Path(out_dir, "result.json").read_text(encoding="utf-8"))["turns"]
    cues = load_teams(Path(vtt))

    def raw_text(t0: float, t1: float) -> str:
        return " ".join(s.get("text", "") for s in raw_segs
                        if s.get("end", 0) > t0 and s.get("start", 0) < t1).strip()

    changed = []
    for turn in final:
        before = raw_text(turn["start"], turn["end"])
        after = turn["text"]
        if not before:
            continue
        ratio = SequenceMatcher(None, norm(before), norm(after)).ratio()
        if ratio < 0.995:
            changed.append((ratio, turn, before, after))

    changed.sort(key=lambda x: x[0])
    print(f"{len(changed)} of {len(final)} turns differ from the raw ASR\n")
    print("Lowest similarity first — the further from 1.00, the more the LLM rewrote.\n")

    for ratio, turn, before, after in changed[:limit]:
        teams = text_at(cues, turn["start"], turn["end"])
        print(f"--- [{clock(turn['start'])}] {turn['speaker']}  (similarity {ratio:.2f})")
        print(f"    ASR   : {before[:200]}")
        print(f"    final : {after[:200]}")
        print(f"    Teams : {teams[:200] or '(nothing)'}")

        # Does the rewrite move towards Teams, or away from everyone?
        b = SequenceMatcher(None, norm(before), norm(teams)).ratio() if teams else 0.0
        a = SequenceMatcher(None, norm(after), norm(teams)).ratio() if teams else 0.0
        if teams:
            verdict = ("closer to Teams" if a > b + 0.02 else
                       "further from Teams" if a < b - 0.02 else "no clearer")
            print(f"    -> vs Teams: {b:.2f} -> {a:.2f} ({verdict})")
        print()

    if changed:
        with_teams = [(x, SequenceMatcher(None, norm(x[2]), norm(text_at(cues, x[1]['start'], x[1]['end']))).ratio(),
                       SequenceMatcher(None, norm(x[3]), norm(text_at(cues, x[1]['start'], x[1]['end']))).ratio())
                      for x in changed if text_at(cues, x[1]["start"], x[1]["end"])]
        better = sum(1 for _, b, a in with_teams if a > b + 0.02)
        worse = sum(1 for _, b, a in with_teams if a < b - 0.02)
        print("=== overall direction of the LLM's edits ===")
        print(f"  {len(with_teams)} edited turns had Teams text to compare against")
        print(f"  moved towards Teams : {better}")
        print(f"  moved away          : {worse}")
        print(f"  no clear change     : {len(with_teams) - better - worse}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 25)
