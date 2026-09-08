"""Score transcription *wording* against two independent readings.

Speaker attribution had a ground truth — Teams knows who spoke. Wording has
none: Teams writes `vrchní ve 2. Tří`, we write `WhatsApp` where Soniox hears
`odsaď`. Comparing two transcripts can only measure divergence, never who is
wrong, which is why every earlier accuracy figure here carried an asterisk.

Three readings break the tie. Where Teams and Soniox independently agree, that
agreement is a *witness*: if we differ there, we are very probably wrong. That
subset is the reference this benchmark scores against.

Read the header of the report before the numbers. Two things about this metric
matter more than its value:

* It is **not** a WER. The witness set covers exactly the audio that two
  systems found easy, so the hard parts — crosstalk, the 59 kbps stretches —
  are excluded from the denominator. True error over the whole file is higher.
* Our shipped transcript is **not independent of Teams**: the LLM merge reads
  Teams and sometimes copies it verbatim. A pipeline that parrots Teams scores
  well on a Teams-derived witness by construction. `A0` (raw ASR, pre-merge)
  and `WER_C` (scored against Soniox alone) exist to catch that, and the loop
  should only trust a change that improves both.

    python scripts/bench/bench_wording.py [bench/soniox/cs-plain.json]
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path



# Importable from anywhere: sibling helpers first, then the repo root so
# `app` resolves without relying on PYTHONPATH being set.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import corpus  # noqa: E402

from app.teams import load as load_teams  # noqa: E402
from bench_speakers import PAD, WINDOW, clock, teams_words, toks  # noqa: E402



OUT = corpus.out_dir()
VTT = corpus.teams_vtt()
RESULTS = corpus.results()
SONIOX = RESULTS / "soniox" / "cs-plain.json"
# Optional fourth engine, same JSON shape, scored on the SAME witness set as
# A0. Absent by default; produced by scripts/bench/canary_reference.py.
CANARY = RESULTS / "canary" / "cs-plain.json"

# A pairing further apart than this is difflib matching a repeated filler across
# minutes, not the same utterance. Teams' per-word times are interpolated inside
# each cue, so anything involving Teams needs a looser bound.
SKEW_AC = 4.0
SKEW_B = 8.0

FUNCTION = {
    "a", "ale", "i", "to", "tu", "ten", "ta", "te", "se", "si", "je", "no", "jo",
    "že", "jak", "tak", "jako", "by", "v", "u", "na", "o", "s", "z", "já", "ty",
    "on", "ono", "vono", "co", "když", "už", "ještě", "tam", "teď", "tady", "ok",
}

# Common-Czech vs standard endings. These are how people speak, not errors, and
# counting them would swamp everything that matters.
REGISTER_PAIRS = [
    ("ý", "ej"), ("í", "ej"), ("ého", "ýho"), ("ému", "ýmu"),
    ("ými", "ýma"), ("ími", "ejma"), ("ají", "aj"), ("ejí", "ej"),
    ("ít", "ejt"), ("é", "ý"),
]

LATIN_ISH = re.compile(r"^(?:[a-z]*(?:sh|ck|w|x|q|oo|ee|ea)[a-z]*)$")


# ---------------------------------------------------------------------------
# Streams
# ---------------------------------------------------------------------------

def norm(w: str) -> str:
    return unicodedata.normalize("NFC", w).lower()


def strip_marks(w: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", w)
                   if not unicodedata.combining(c))


def from_soniox(path: Path) -> list[dict]:
    """Soniox tokens are sub-word — `Beautiful` can arrive as Beau/ti/ful.

    Iterating tokens as words would inflate this stream by half again and break
    every alignment, so the token text is rejoined and then tokenised with the
    *same* regex used on the other two streams. Equal treatment is the point.
    """
    d = json.loads(path.read_text(encoding="utf-8"))["transcript"]
    tokens = d["tokens"]
    text = "".join(t["text"] for t in tokens)
    assert text == d.get("text", text), "rejoined tokens do not match the API text"

    # char offset -> token, so each word can recover a time from its characters
    owner, pos = [], 0
    for t in tokens:
        for _ in t["text"]:
            owner.append(t)
        pos += len(t["text"])

    out = []
    for m in re.finditer(r"[^\W\d_]+", text, re.UNICODE):
        covering = owner[m.start():m.end()]
        if not covering:
            continue
        out.append({
            "w": norm(m.group()), "raw": m.group(),
            "t": (min(c["start_ms"] for c in covering)
                  + max(c["end_ms"] for c in covering)) / 2000.0,
            "conf": min((c.get("confidence") or 0.0) for c in covering),
        })
    return out


def from_turns_text(turns: list[dict]) -> list[dict]:
    """Our shipped words, tokenised from `text` — what the user actually reads.

    `words` is for timing only. Timing per token comes from aligning the text
    tokens against `words`; anything unmatched is interpolated across the turn,
    the same trick teams_words() uses for cues.
    """
    out = []
    for turn in turns:
        words = [w for w in (turn.get("words") or [])
                 if w.get("start") is not None]
        text_toks = toks(turn["text"])
        if not text_toks:
            continue
        wt = [norm(w["word"]) for w in words]
        times = [None] * len(text_toks)
        for i, j, n in SequenceMatcher(None, [norm(t) for t in text_toks], wt,
                                       autojunk=False).get_matching_blocks():
            for k in range(n):
                w = words[j + k]
                times[i + k] = (float(w["start"]) + float(w["end"])) / 2
        lo, hi = turn["start"], max(turn["end"], turn["start"] + 0.01)
        for i, tok in enumerate(text_toks):
            t = times[i]
            if t is None:
                t = lo + (hi - lo) * (i + 0.5) / len(text_toks)
            out.append({"w": norm(tok), "raw": tok, "t": t, "conf": None})
    return out


def from_align(path: Path) -> list[dict]:
    """Raw ASR, before the LLM ever saw Teams. The independent version of us."""
    d = json.loads(path.read_text(encoding="utf-8"))
    segs = d["segments"] if isinstance(d, dict) else d
    out = []
    for s in segs:
        for w in s.get("words") or []:
            tok = toks(w.get("word", ""))
            if tok and w.get("start") is not None:
                out.append({"w": norm(tok[0]), "raw": tok[0],
                            "t": (float(w["start"]) + float(w["end"])) / 2,
                            "conf": w.get("score")})
    return out


def from_teams(path: Path) -> list[dict]:
    return [{"w": norm(x["w"]), "raw": x["w"], "t": x["t"], "conf": None}
            for x in teams_words(load_teams(path))]


# ---------------------------------------------------------------------------
# Equivalence
# ---------------------------------------------------------------------------

def classify(a: str, b: str) -> str:
    if a == b:
        return "identical"
    if strip_marks(a) == strip_marks(b):
        return "diacritic"
    for x, y in REGISTER_PAIRS:
        for p, q in ((x, y), (y, x)):
            if a.endswith(p) and b.endswith(q) and a[:-len(p)] == b[:-len(q)] \
                    and len(a) - len(p) >= 2:
                return "register"
    if a in FUNCTION and b in FUNCTION:
        return "function"
    stem = min(len(a), len(b))
    if stem >= 5 and a[:5] == b[:5] and SequenceMatcher(None, a, b).ratio() >= 0.8:
        return "inflection"
    return "different"


def same(a: str, b: str) -> bool:
    return classify(a, b) != "different"


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def align_pair(xs: list[dict], ys: list[dict], skew: float) -> dict[int, int]:
    """Map index in xs -> index in ys, rejecting pairings that are far apart."""
    out: dict[int, int] = {}
    sm = SequenceMatcher(None, [x["w"] for x in xs], [y["w"] for y in ys],
                         autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            pairs = [(i1 + k, j1 + k) for k in range(i2 - i1)]
        elif tag == "replace" and (i2 - i1) == (j2 - j1):
            pairs = [(i1 + k, j1 + k) for k in range(i2 - i1)]
        elif tag == "replace":
            # Unequal spans: pair greedily by best string similarity so a real
            # substitution is recorded as one, not as an insert plus a delete.
            pairs = []
            used = set()
            for i in range(i1, i2):
                best, best_r = None, 0.45
                for j in range(j1, j2):
                    if j in used:
                        continue
                    r = SequenceMatcher(None, xs[i]["w"], ys[j]["w"]).ratio()
                    if r > best_r:
                        best, best_r = j, r
                if best is not None:
                    used.add(best)
                    pairs.append((i, best))
        else:
            pairs = []
        for i, j in pairs:
            if abs(xs[i]["t"] - ys[j]["t"]) <= skew:
                out[i] = j
    return out


def columns(streams: dict[str, list[dict]]) -> tuple[list[dict], dict]:
    """Build one row per Soniox word, carrying whatever each stream matched.

    Soniox is the pivot because it is the only stream independent of both of the
    others; using ours would condition the measurement of ours on itself.
    """
    C = streams["C"]
    span = max((w["t"] for s in streams.values() for w in s), default=0.0)
    rows: list[dict] = []
    health = Counter()

    t = 0.0
    while t < span + WINDOW:
        lo, hi = t - PAD, t + WINDOW + PAD
        win = {k: [w for w in v if lo <= w["t"] < hi] for k, v in streams.items()}
        cw = win["C"]
        if cw:
            maps = {}
            for k in streams:
                if k == "C":
                    continue
                maps[k] = align_pair(cw, win[k], SKEW_B if k == "B" else SKEW_AC)
            for i, c in enumerate(cw):
                if not (t <= c["t"] < t + WINDOW):
                    continue  # the pad must not let a word be scored twice
                row = {"c": c}
                for k, m in maps.items():
                    row[k] = win[k][m[i]] if i in m else None
                rows.append(row)
                health["columns"] += 1
        t += WINDOW

    for k in streams:
        if k != "C":
            matched = sum(1 for r in rows if r.get(k))
            health[f"matched_{k}"] = matched
    health["C_words"] = len(C)
    return rows, health


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def witness(rows: list[dict]) -> list[dict]:
    """Columns where Teams and Soniox independently agree.

    Built from B and C only, so it is identical for every variant of ours that
    gets scored — which is what makes those scores comparable.
    """
    # The pivot word lives under "c" — every row has one by construction; what
    # varies is whether Teams matched into that column.
    return [r for r in rows if r.get("B") and same(r["B"]["w"], r["c"]["w"])]


def score_against(rows: list[dict], key: str, ref: str) -> dict:
    subs = miss = 0
    errors: list[tuple] = []
    for r in rows:
        target = r[ref]["w"] if ref != "C" else r["c"]["w"]
        mine = r.get(key)
        if mine is None:
            miss += 1
            continue
        if not same(mine["w"], target):
            subs += 1
            errors.append((r["c"]["t"], mine["w"], target,
                           r["B"]["w"] if r.get("B") else None))
    n = len(rows) or 1
    # Per-column 1/0, in row order. The witness set is built from B and C only,
    # so it is identical across arms and these vectors are directly pairable --
    # which is the only way to tell a real 0.2 pt move from resampling luck.
    flags = []
    for r in rows:
        target = r[ref]["w"] if ref != "C" else r["c"]["w"]
        mine = r.get(key)
        flags.append(0 if (mine is not None and same(mine["w"], target)) else 1)
    return {"n": n, "subs": subs, "missing": miss,
            "rate": (subs + miss) / n, "errors": errors, "flags": flags}


def main() -> int:
    soniox = Path(sys.argv[1]) if len(sys.argv) > 1 else SONIOX
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    streams = {
        "C": from_soniox(soniox),
        "A0": from_align(OUT / "work" / "align.json"),
        "A1": from_turns_text(result["turns"]),
        "B": from_teams(VTT),
    }
    if CANARY.exists():
        streams["D"] = from_soniox(CANARY)

    rows, health = columns(streams)
    W = witness(rows)

    print("=" * 72)
    print("WORDING BENCHMARK — relative to Soniox, NOT an absolute WER")
    print("=" * 72)
    print(f"reference: {soniox.name}")
    print("\nThis scores only the audio two independent systems agreed on, so it")
    print("excludes the hard parts and understates true error. Our shipped text")
    print("(A1) is not independent of Teams — read A0 and WER_C alongside it.\n")

    print("--- harness health ---")
    for k in [k for k in ("C", "A0", "A1", "B", "D") if k in streams]:
        print(f"  {k:<3} {len(streams[k]):>6} words")
    print(f"  columns (pivot=C)      {health['columns']:>6}")
    print(f"  witness |R|            {len(W):>6}"
          f"   ({len(W)/max(health['columns'],1)*100:.0f}% of columns)")
    for k in [k for k in ("A0", "A1", "B", "D") if k in streams]:
        print(f"  matched into columns {k:<3} {health[f'matched_{k}']:>6}")

    print("\n--- headline ---")
    res = {}
    judged = [k for k in ("A0", "A1", "B", "D") if k in streams]
    for k in judged:
        res[k] = score_against(W, k, "B")
    resC = {k: score_against(rows, k, "C")
            for k in judged if k in ("A0", "A1", "D")}
    print(f"  {'stream':<22} {'err vs witness':>14} {'subs':>6} {'missing':>8}")
    print(f"  {'A0 raw ASR':<22} {res['A0']['rate']*100:>13.2f}% "
          f"{res['A0']['subs']:>6} {res['A0']['missing']:>8}   <- independent of Teams")
    print(f"  {'A1 shipped':<22} {res['A1']['rate']*100:>13.2f}% "
          f"{res['A1']['subs']:>6} {res['A1']['missing']:>8}   <- NOT independent")
    if "D" in res:
        print(f"  {'D Canary-1b-v2':<22} {res['D']['rate']*100:>13.2f}% "
              f"{res['D']['subs']:>6} {res['D']['missing']:>8}   <- candidate ASR")
        print("  D and A0 are both raw ASR judged on the same witness columns, so")
        print("  D - A0 IS a valid comparison. That is the whole point of D.")
    print()
    print("  DO NOT read A1 - A0 as the merge's effect. The two streams are built")
    print("  differently: A0 is aligned words, A1 is re-tokenised turn text with")
    print("  interpolated timings, so they align to the witness differently even")
    print("  when the text is identical. Measured with the merge fully disabled,")
    print("  A1 still scored 0.74 pts worse than A0 - with nothing to blame it on.")
    print("  Compare A1 against A1 from another arm. That is the valid comparison,")
    print("  and by it the merge is slightly positive, not negative.")
    print("\n  anti-parrot control (vs Soniox alone, all columns):")
    print(f"  {'A0 raw ASR':<22} {resC['A0']['rate']*100:>13.2f}%")
    print(f"  {'A1 shipped':<22} {resC['A1']['rate']*100:>13.2f}%")
    if "D" in resC:
        print(f"  {'D Canary-1b-v2':<22} {resC['D']['rate']*100:>13.2f}%")
    print("  A change is real only if BOTH the witness error and this one improve.")
    print("  Witness better + this worse = the merge is copying Teams, not fixing it.")

    # ---- head to head, judged the same way for both sides ------------------
    # Scoring us against a witness that contains Soniox is rigged: Soniox agrees
    # with itself, so it would post a perfect score. The fair question is how
    # often each engine departs from the *other two*, and Teams is the neutral
    # third party available to both. Same rule, same judge, sides swapped.
    print("\n--- all three engines, judged identically ---")
    # Each engine is scored on the columns where the OTHER two agree, so no
    # engine ever sits on its own jury. Teams is the neutral party for the
    # ours-vs-Soniox question, and Soniox for the ours-vs-Teams one.
    def leg(judged: str, a_key: str, b_key: str) -> tuple[int, float]:
        """Columns where a_key and b_key agree, and how often `judged` differs."""
        def word(r, k):
            return r["c"] if k == "C" else r.get(k)
        wit = [r for r in rows
               if word(r, a_key) and word(r, b_key)
               and same(word(r, a_key)["w"], word(r, b_key)["w"])]
        miss = 0
        for r in wit:
            mine = word(r, judged)
            if mine is None or not same(mine["w"], word(r, a_key)["w"]):
                miss += 1
        return len(wit), miss / max(len(wit), 1)

    legs = [
        ("ours (Whisper + merge)", leg("A0", "B", "C")),
        ("Teams native", leg("B", "A0", "C")),
        ("Soniox stt-async-v5", leg("C", "A0", "B")),
    ]
    print(f"  {'engine':<26} {'columns judged':>15} {'departs from the other two':>28}")
    for label, (n, rate) in sorted(legs, key=lambda x: x[1][1]):
        print(f"  {label:<26} {n:>15} {rate*100:>27.2f}%")
    print("\n  Lower is better. This is 'disagrees with the consensus of the other")
    print("  two', not accuracy — none of the three is ground truth, and on any")
    print("  given word the odd one out may be the only one that is right.")

    # ---- does the reference actually catch what we know is wrong? ----------
    # If this block is empty the alignment is broken and nothing else here is
    # trustworthy: these are errors already confirmed by hand.
    print("\n--- sanity: errors we already know Teams makes ---")
    teams_wrong = [r for r in rows
                   if r.get("B") and r.get("A1")
                   and same(r["A1"]["w"], r["c"]["w"])
                   and not same(r["B"]["w"], r["c"]["w"])]
    tw = Counter((r["B"]["w"], r["c"]["w"]) for r in teams_wrong)
    for (bad, good), n in tw.most_common(8):
        print(f"    {n:>3}x  Teams {bad!r:<18} -> reference {good!r}")
    print(f"    {len(teams_wrong)} columns where we and the reference agree "
          f"and Teams differs")
    if len(teams_wrong) < 50:
        print("    FAIL: expected hundreds — the alignment is probably broken")

    # ---- contamination -----------------------------------------------------
    print("\n--- what the merge did (A0 -> A1) ---")
    buckets = Counter()
    for r in rows:
        a0, a1, b = r.get("A0"), r.get("A1"), r.get("B")
        if not a0 or not a1 or same(a0["w"], a1["w"]):
            continue
        c = r["c"]["w"]
        if b and same(a1["w"], b["w"]):
            if same(c, b["w"]):
                buckets["imported_good"] += 1
            elif same(c, a0["w"]):
                buckets["imported_bad"] += 1
            else:
                buckets["imported_unwitnessed"] += 1
        else:
            buckets["independent_change"] += 1
    for k in ("imported_good", "imported_bad", "imported_unwitnessed",
              "independent_change"):
        note = {"imported_bad": "  <- Teams' error copied into our transcript",
                "independent_change": "  <- the model invented this"}.get(k, "")
        print(f"  {k:<24} {buckets[k]:>5}{note}")

    # ---- ranked errors -----------------------------------------------------
    print("\n--- our most frequent errors (A1, on the witness set) ---")
    pairs = Counter((e[1], e[2]) for e in res["A1"]["errors"])
    times = defaultdict(list)
    for t, mine, ref, _ in res["A1"]["errors"]:
        times[(mine, ref)].append(t)
    tech, plain = [], []
    for (mine, ref), n in pairs.most_common():
        (tech if LATIN_ISH.match(mine) or LATIN_ISH.match(ref) else plain).append(
            (n, mine, ref))
    for label, group in (("technical / proper nouns", tech), ("content words", plain)):
        print(f"  {label}:")
        for n, mine, ref in group[:12]:
            at = ", ".join(clock(t) for t in times[(mine, ref)][:3])
            print(f"    {n:>3}x  {mine!r:<20} -> {ref!r:<20} [{at}]")
        if not group:
            print("    (none)")

    # ---- glossary candidates ----------------------------------------------
    ours_all = {w["w"] for w in streams["A1"]}
    cand = Counter()
    for r in W:
        c = r["c"]["w"]
        if c not in ours_all and (LATIN_ISH.match(c) or len(c) > 6):
            cand[c] += 1
    print("\n--- glossary candidates (witnessed words we never produce) ---")
    for w, n in cand.most_common(15):
        print(f"    {n:>3}x  {w}")

    (RESULTS / "wording.json").write_text(json.dumps({
        "reference": soniox.name,
        "witness": len(W), "columns": health["columns"],
        "stream_words": {k: len(v) for k, v in streams.items()},
        "witness_error": {k: res[k]["rate"] for k in res},
        "soniox_error": {k: resC[k]["rate"] for k in resC},
        "merge": dict(buckets),
        "top_errors": [[n, m, r] for (m, r), n in pairs.most_common(40)],
        # One character per column, not a JSON array of ints: same information,
        # a thirtieth of the file, and these get committed as evidence.
        "witness_t": [int(r["c"]["t"]) for r in W],
        "witness_flags": {k: "".join(str(f) for f in res[k]["flags"])
                          for k in res},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nmachine-readable -> bench/wording.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
