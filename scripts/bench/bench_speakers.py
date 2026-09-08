"""Score speaker assignment against the Teams transcript.

Teams is ground truth here and only here: when a Teams transcript is supplied
the app takes diarization strictly from it, so any word we credit to someone
Teams credits to another person is our bug, not a difference of opinion.

Matching by time alone is what produced the bug this benchmark exists to catch,
so scoring cannot use time either. Both sides transcribed the same audio, so
their *word sequences* are alignable; a word of ours that lines up with a word
of theirs inherits an unambiguous correct answer. Words with no counterpart are
left unscored rather than guessed at.

Alignment is done inside time windows. A global diff over ~11k words drifts:
one repeated Czech filler can pair our minute 4 with their minute 40.

Run with no arguments to score the shipped result.json; pass a strategy name to
score a candidate implementation instead.
"""

from __future__ import annotations

import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path



# Importable from anywhere: sibling helpers first, then the repo root so
# `app` resolves without relying on PYTHONPATH being set.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import corpus  # noqa: E402

from app.teams import TeamsCue, load as load_teams  # noqa: E402



WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
WINDOW = 60.0        # seconds per alignment window
PAD = 10.0           # overlap so words near a boundary still find their pair


def toks(text: str) -> list[str]:
    return WORD.findall(text.lower())


def clock(t: float) -> str:
    m, s = divmod(int(t), 60)
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Reference and hypothesis word streams
# ---------------------------------------------------------------------------

def teams_words(cues: list[TeamsCue]) -> list[dict]:
    out = []
    for c in cues:
        ws = toks(c.text)
        if not ws:
            continue
        step = (c.end - c.start) / len(ws)
        for i, w in enumerate(ws):
            out.append({"w": w, "t": c.start + step * (i + 0.5), "speaker": c.speaker})
    return out


def our_words(segments: list[dict]) -> list[dict]:
    out = []
    for s in segments:
        for w in s.get("words") or []:
            token = toks(w.get("word", ""))
            if not token or w.get("start") is None:
                continue
            out.append({
                "w": token[0],
                "t": (float(w["start"]) + float(w["end"])) / 2,
                "speaker": w.get("speaker") or s.get("speaker"),
            })
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(ours: list[dict], theirs: list[dict]) -> dict:
    span = max((w["t"] for w in ours + theirs), default=0.0)
    matched = correct = 0
    errors: list[tuple[float, str, str, str]] = []

    t = 0.0
    while t < span + WINDOW:
        a = [w for w in ours if t - PAD <= w["t"] < t + WINDOW + PAD]
        b = [w for w in theirs if t - PAD <= w["t"] < t + WINDOW + PAD]
        if a and b:
            sm = SequenceMatcher(None, [w["w"] for w in a], [w["w"] for w in b],
                                 autojunk=False)
            for i, j, n in sm.get_matching_blocks():
                for k in range(n):
                    wa, wb = a[i + k], b[j + k]
                    # only score words actually inside this window, so the pad
                    # does not let a word be scored twice
                    if not (t <= wa["t"] < t + WINDOW):
                        continue
                    matched += 1
                    if wa["speaker"] == wb["speaker"]:
                        correct += 1
                    else:
                        errors.append((wa["t"], wa["w"], wa["speaker"], wb["speaker"]))
        t += WINDOW

    return {
        "matched": matched,
        "correct": correct,
        "accuracy": correct / matched if matched else 0.0,
        "errors": errors,
    }


def _field(turn, name: str):
    """Turns arrive as dicts from result.json and as Turn objects in-process."""
    return turn[name] if isinstance(turn, dict) else getattr(turn, name)


def integrity(turns: list) -> dict:
    """How often a speaker change lands in the middle of a sentence."""
    cuts = changes = 0
    for a, b in zip(turns, turns[1:]):
        if _field(a, "speaker") == _field(b, "speaker"):
            continue
        changes += 1
        prev, cur = _field(a, "text").rstrip(), _field(b, "text").lstrip()
        if prev and prev[-1] not in ".!?…" and cur[:1].islower():
            cuts += 1
    short = sum(1 for t in turns if len(toks(_field(t, "text"))) <= 5)
    return {"changes": changes, "cuts": cuts,
            "cut_rate": cuts / changes if changes else 0.0,
            "turns": len(turns), "short_turns": short}


def report(label: str, sc: dict, integ: dict | None, show_errors: int = 0) -> None:
    print(f"\n=== {label} ===")
    print(f"  speaker accuracy : {sc['accuracy']*100:6.2f}%  "
          f"({sc['correct']}/{sc['matched']} aligned words)")
    if integ:
        print(f"  sentences cut    : {integ['cut_rate']*100:6.2f}%  "
              f"({integ['cuts']}/{integ['changes']} speaker changes)")
        print(f"  turns            : {integ['turns']}  "
              f"({integ['short_turns']} of <=5 words)")
    for t, w, got, want in sc["errors"][:show_errors]:
        print(f"    [{clock(t)}] {w!r}: we said {got!r}, Teams says {want!r}")


def main() -> None:
    out_dir = corpus.out_dir()
    vtt = corpus.teams_vtt()
    cues = load_teams(vtt)
    ref = teams_words(cues)

    res = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    turns = res["turns"]
    hyp = our_words(turns)

    print(f"reference: {len(ref)} Teams words")
    print(f"hypothesis: {len(hyp)} of our words")
    report("shipped result.json", score(hyp, ref), integrity(turns), show_errors=12)


if __name__ == "__main__":
    main()
