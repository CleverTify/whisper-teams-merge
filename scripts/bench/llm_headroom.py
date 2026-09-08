"""How much is left for the LLM merge to win, and is it taking it?

Most words the two transcripts disagree about are not errors. Whisper writes
standard Czech where the speaker used the common spoken forms Teams preserves —
být/bejt, nějaký/nějakej, mají/maj — and adopting Teams there would make the
transcript worse, not better. Function-word noise (a/ale, to/tu) is likewise not
worth a model call.

What is worth winning is a *substantive* disagreement: a content word where the
two readings are genuinely different, not two spellings of one utterance. This
counts those, and how many of the turns containing one the LLM actually touched.
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

from app.teams import load as load_teams, text_at  # noqa: E402
from bench_speakers import toks  # noqa: E402



OUT = corpus.out_dir()
VTT = corpus.teams_vtt()

# Endings that mark a common-Czech variant of a standard form.
REGISTER = re.compile(r"(ej|ý|ej[íš]|ma|ma[jť]|ou)$")
FUNCTION = {
    "a", "ale", "i", "to", "tu", "ten", "ta", "te", "se", "si", "je", "no", "jo",
    "že", "jak", "tak", "jako", "by", "v", "u", "na", "o", "s", "z", "já", "ty",
    "on", "ono", "vono", "co", "když", "už", "ještě", "tam", "teď", "tady", "ok",
}


def substantive(a: str, b: str) -> bool:
    """A disagreement worth a model call, rather than dialect or filler."""
    if a in FUNCTION and b in FUNCTION:
        return False
    if len(a) < 4 and len(b) < 4:
        return False
    ratio = SequenceMatcher(None, a, b).ratio()
    if ratio >= 0.7:
        return False          # two spellings of one word: register or diacritics
    return True


def main() -> None:
    res = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    turns = res["turns"]
    cues = load_teams(VTT)

    raw = json.loads((OUT / "work" / "align.json").read_text(encoding="utf-8"))
    raw_segs = raw["segments"] if isinstance(raw, dict) else raw

    def asr_at(t0: float, t1: float) -> str:
        return " ".join(s.get("text", "") for s in raw_segs
                        if s.get("end", 0) > t0 and s.get("start", 0) < t1)

    total = subst = touched_any = touched_subst = 0
    examples = []

    for turn in turns:
        ours = toks(turn["text"])
        theirs = toks(text_at(cues, turn["start"], turn["end"]))
        if not ours or not theirs:
            continue
        total += 1

        sm = SequenceMatcher(None, ours, theirs, autojunk=False)
        hits = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag != "replace":
                continue
            for k in range(min(i2 - i1, j2 - j1)):
                a, b = ours[i1 + k], theirs[j1 + k]
                if substantive(a, b):
                    hits.append((a, b))

        changed = toks(asr_at(turn["start"], turn["end"])) != ours
        if changed:
            touched_any += 1
        if hits:
            subst += 1
            if changed:
                touched_subst += 1
            elif len(examples) < 12:
                examples.append((turn["start"], hits[:3], turn["text"][:90],
                                 text_at(cues, turn["start"], turn["end"])[:90]))

    print(f"turns with text on both sides : {total}")
    print(f"  containing a substantive disagreement : {subst} "
          f"({subst/total*100:.0f}%)")
    print(f"  turns the LLM changed at all          : {touched_any}")
    print(f"  of the substantive ones, changed      : {touched_subst} "
          f"({touched_subst/subst*100:.0f}% of {subst})")
    print(f"  -> left on the table                  : {subst - touched_subst}")

    print("\n=== substantive disagreements the LLM did not touch ===")
    for start, hits, mine, theirs in examples:
        m, s = divmod(int(start), 60)
        pairs = ", ".join(f"{a!r}/{b!r}" for a, b in hits)
        print(f"  [{m:02d}:{s:02d}] {pairs}")
        print(f"        ours : {mine}")
        print(f"        Teams: {theirs}")


if __name__ == "__main__":
    main()
