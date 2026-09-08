"""Unit-test the LLM guardrail without needing a running LLM.

The guardrail is the one thing standing between a helpful correction and a
silently rewritten transcript, so it is worth testing directly.

    docker compose run --rm app python tests/test_guardrail.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.llm import _parse_numbered, acceptable, windows  # noqa: E402

ORIG = "Tady to bily znamena co? To jsou jenom spatne vizualizovane tyhle."

CASES = [
    ("light correction", "Tady to bílé znamená co? To jsou jenom špatně vizualizované tyhle.", True),
    ("identical", ORIG, True),
    ("hallucinated essay", ORIG + " " + ("Navic bych dodal ze cely system je velmi zajimavy a " * 4), False),
    ("truncated away", "Tady.", False),
    ("translated to English", "This white one means what? Those are just badly visualised ones.", False),
    ("model commentary", "Here is the corrected transcript: Tady to bile znamena co?", False),
    ("empty", "   ", False),
]


class _T:
    def __init__(self, start, end):
        self.start, self.end = start, end


def main() -> int:
    failures = 0
    print(f"{'case':<24}{'expected':<11}{'got':<11}reason")
    print("-" * 68)
    for label, candidate, expect in CASES:
        ok, why = acceptable(ORIG, candidate, settings)
        mark = "" if ok == expect else "   <-- WRONG"
        if ok != expect:
            failures += 1
        print(f"{label:<24}{('accept' if expect else 'reject'):<11}"
              f"{('accept' if ok else 'reject'):<11}{why}{mark}")

    print("\n=== numbered-reply parsing ===")
    got = _parse_numbered("[1] first\n[2] second\nrandom noise\n[3] third", 3)
    print(" full reply      :", got)
    if len(got) != 3:
        failures += 1
        print("   <-- WRONG, expected 3 lines")
    partial = _parse_numbered("[2] only this one", 3)
    print(" partial reply   :", partial, "(missing lines keep their original)")
    over = _parse_numbered("[9] out of range", 3)
    print(" out-of-range    :", over, "(ignored)")
    if over:
        failures += 1
        print("   <-- WRONG, should be empty")

    print("\n=== windowing ===")
    turns = [_T(i * 20, i * 20 + 18) for i in range(10)]
    groups = windows(turns, 90)
    print(f" 10 turns of 20s into 90s windows -> {[len(g) for g in groups]}")
    if sum(len(g) for g in groups) != len(turns):
        failures += 1
        print("   <-- WRONG, turns lost")

    print("\n=== glossary ===")
    from app.llm import merge_system, screen_term_ok
    if "spelled correctly here" in merge_system([]):
        failures += 1
        print(" empty glossary  : <-- WRONG, rule 7 added with no terms")
    else:
        print(" empty glossary  : rule 7 omitted")

    if "appku" not in merge_system(["appku"]):
        failures += 1
        print(" one term        : <-- WRONG, term missing from the prompt")
    else:
        print(" one term        : listed in the prompt")

    # Why the glossary is safe to expose: a term may REPLACE a word, never
    # arrive as an extra one. Same guard the on-screen terms go through.
    cases = [
        ("mam tu apku otevrenou", "mam tu appku otevrenou", True,  "substitution"),
        ("mam tu otevrenou",      "mam tu appku otevrenou", True,  "one-word growth"),
        ("mam tu",                "mam tu appku otevrenou", False, "inserted"),
    ]
    for original, candidate, want, label in cases:
        ok, why = screen_term_ok(original, candidate, ["appku"])
        if ok != want:
            failures += 1
        mark = "" if ok == want else "   <-- WRONG"
        print(f" {label:<15} : ok={ok} ({why or 'clean'}){mark}")

    print("\n" + ("ALL PASS" if not failures else f"{failures} FAILURE(S)"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
