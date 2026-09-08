"""Validate every artifact an output directory produces.

Checks real format conformance and cross-format agreement, not just that files
exist — a WebVTT can be perfectly well-formed and still be missing half the
meeting or have mangled diacritics.

    docker compose run --rm app python scripts/validate_outputs.py
    docker compose run --rm app python scripts/validate_outputs.py output/<name>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

OUTPUT_DIR = Path("/work/output")
VTT_TS = re.compile(
    r"^(\d{2}):(\d{2}):(\d{2})\.(\d{3}) --> (\d{2}):(\d{2}):(\d{2})\.(\d{3})", re.M
)

problems: list[str] = []
notes: list[str] = []


def fail(art: str, msg: str) -> None:
    problems.append(f"{art}: {msg}")


def note(art: str, msg: str) -> None:
    notes.append(f"{art}: {msg}")


def secs(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def clock(s: float) -> str:
    s = int(max(0, s))
    return f"{s//3600:02d}:{(s//60)%60:02d}:{s%60:02d}"


# ---------------------------------------------------------------------------

def check_result(path: Path) -> tuple[float, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("metadata", "speakers", "turns", "name_map"):
        if key not in data:
            fail(path.name, f"missing top-level key {key!r}")
    meta = data.get("metadata", {})
    for key in ("asr", "speakers_from", "languages", "environment", "source", "llm"):
        if key not in meta:
            fail(path.name, f"metadata missing {key!r}")

    duration = float(meta.get("source", {}).get("duration") or 0)
    turns = data.get("turns", [])
    if not turns:
        fail(path.name, "no turns")

    bad_order = bad_word = 0
    prev = -1.0
    for t in turns:
        if not all(k in t for k in ("start", "end", "speaker", "text", "clock")):
            fail(path.name, "a turn is missing required keys")
            break
        if t["end"] < t["start"] or t["start"] < prev - 1e-6:
            bad_order += 1
        prev = t["start"]
        for w in t.get("words") or []:
            if w.get("start") is None or w.get("end") is None or w["end"] < w["start"]:
                bad_word += 1
    if bad_order:
        fail(path.name, f"{bad_order} turn(s) with inverted or out-of-order times")
    if bad_word:
        fail(path.name, f"{bad_word} word(s) with invalid timings")

    labels = {t["speaker"] for t in turns}
    stats = {s["speaker"] for s in data.get("speakers", [])}
    if labels - stats:
        fail(path.name, f"turns reference speakers absent from stats: {sorted(labels-stats)}")
    total = sum(s.get("share", 0) for s in data.get("speakers", []))
    if data.get("speakers") and abs(total - 1.0) > 0.02:
        fail(path.name, f"speaker shares sum to {total:.3f}, expected ~1.0")

    words = sum(len(t.get("words") or []) for t in turns)
    llm = meta.get("llm", {})
    note(path.name,
         f"{len(turns)} turns, {words} words, {len(stats)} speakers, "
         f"{duration:.0f}s, speakers from {meta.get('speakers_from', {}).get('source')}, "
         f"llm {llm.get('mode')} ({llm.get('turns_changed', 0)} changed, "
         f"{llm.get('turns_rejected', 0)} rejected)")
    return duration, data


def check_vtt(path: Path, duration: float) -> None:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("WEBVTT"):
        fail(path.name, "missing WEBVTT header")
    first = text.split("\n", 2)
    if len(first) > 1 and first[1].strip():
        fail(path.name, "no blank line after WEBVTT header")

    cues = VTT_TS.findall(text)
    if not cues:
        fail(path.name, "no cues found")
        return
    prev_end, zero, overlaps = -1.0, 0, 0
    for g in cues:
        st, en = secs(*g[:4]), secs(*g[4:])
        if en <= st:
            zero += 1
        if st < prev_end - 1e-6:
            overlaps += 1
        prev_end = max(prev_end, en)
        if duration and en > duration + 2:
            fail(path.name, f"cue ends at {en:.1f}s past duration {duration:.1f}s")
    if zero:
        fail(path.name, f"{zero} cue(s) with end <= start")
    if overlaps:
        fail(path.name, f"{overlaps} overlapping cue(s)")
    note(path.name, f"{len(cues)} cues parsed cleanly")


def check_md(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "�" in text:
        fail(path.name, "contains U+FFFD replacement characters (encoding damage)")
    if not text.startswith("# "):
        fail(path.name, "does not start with an H1 heading")
    if not re.search(r"^## \[\d{2}:\d{2}:\d{2}\] ", text, re.M):
        fail(path.name, "no speaker turn headings")
    note(path.name, f"{len(text)} chars")


def check_jsonl(path: Path) -> None:
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    for i, line in enumerate(lines, 1):
        try:
            json.loads(line)
        except Exception as exc:
            fail(path.name, f"line {i} is not valid JSON: {exc}")
            return
    note(path.name, f"{len(lines)} records" if lines else "empty (nothing filtered)")


def check_consistency(root: Path, data: dict) -> None:
    """Every format must describe the same transcript."""
    turns = data.get("turns", [])
    names = {t.get("name") or t["speaker"] for t in turns}
    md = (root / "transcript.md").read_text(encoding="utf-8")
    vtt = (root / "transcript.vtt").read_text(encoding="utf-8")

    md_turns = len(re.findall(r"^## \[\d{2}:\d{2}:\d{2}\] ", md, re.M))
    if md_turns != len(turns):
        fail("consistency", f"transcript.md has {md_turns} turns, result.json has {len(turns)}")

    for label, blob in (("transcript.md", md), ("transcript.vtt", vtt)):
        missing = [n for n in names if n not in blob]
        if missing:
            fail("consistency", f"{label} never mentions {sorted(missing)[:3]}")

    for t in [x for x in turns if len(x["text"]) > 40][:6]:
        head = " ".join(t["text"].split()[:5])
        if head and head not in md:
            fail("consistency", f"transcript.md is missing text {head!r}")

    accents = set("ěščřžýáíéúůňťďóĚŠČŘŽÝÁÍÉÚŮŇŤĎÓ")
    src = {c for t in turns for c in t["text"] if c in accents}
    if src:
        for label, blob in (("transcript.md", md), ("transcript.vtt", vtt)):
            if not (src & set(blob)):
                fail("consistency", f"{label} lost all diacritics")
        note("consistency", f"diacritics preserved ({len(src)} distinct)")

    meta = data.get("metadata", {})
    used = float(meta.get("prepared", {}).get("duration")
                 or meta.get("preprocess", {}).get("duration_limit")
                 or meta.get("source", {}).get("duration") or 0)
    if turns and used and turns[-1]["end"] > used + 2:
        fail("consistency",
             f"last turn ends {turns[-1]['end']:.1f}s beyond transcribed span {used:.1f}s")

    full = float(meta.get("source", {}).get("duration") or 0)
    if used and full and abs(used - full) > 1.0:
        if re.search(r"\*\*Duration:\*\* " + re.escape(clock(full)) + r"\s*$", md, re.M):
            fail("consistency",
                 f"transcript.md reports the full {clock(full)} but only {clock(used)} "
                 "was transcribed")

    # v1 files that are no longer regenerated would silently disagree.
    for stale in ("transcript.txt", "subtitles.srt", "subtitles.vtt",
                  "minutes.docx", "speakers.json"):
        if (root / stale).exists():
            fail("consistency", f"stale artifact from an older version: {stale}")

    note("consistency", f"{len(turns)} turns agree across md/vtt/json")


def validate(root: Path) -> None:
    print(f"\n{'='*66}\n{root.name}\n{'='*66}")
    before = len(problems)

    duration, data = 0.0, {}
    if (root / "result.json").exists():
        duration, data = check_result(root / "result.json")
    else:
        fail("result.json", "missing")

    for name, fn in (("transcript.md", check_md), ("removed.jsonl", check_jsonl)):
        p = root / name
        if p.exists():
            fn(p)
        elif name != "removed.jsonl":
            fail(name, "missing")

    p = root / "transcript.vtt"
    if p.exists():
        check_vtt(p, duration)
    else:
        fail("transcript.vtt", "missing")

    if data and (root / "transcript.md").exists() and (root / "transcript.vtt").exists():
        try:
            check_consistency(root, data)
        except Exception as exc:
            fail("consistency", f"check crashed: {exc}")

    for n in notes:
        print(f"  ok    {n}")
    notes.clear()
    new = problems[before:]
    for p2 in new:
        print(f"  FAIL  {p2}")
    print(f"  --> {'PASS' if not new else str(len(new)) + ' PROBLEM(S)'}")


def main() -> int:
    targets = [Path(a) for a in sys.argv[1:]] or [
        d for d in sorted(OUTPUT_DIR.iterdir())
        if d.is_dir() and (d / "result.json").exists()
    ]
    if not targets:
        print("no output directories found")
        return 1
    for t in targets:
        validate(t if t.is_absolute() else Path("/work") / t)
    print(f"\n{'='*66}")
    print(f"TOTAL: {len(problems)} problem(s)" if problems else "TOTAL: all artifacts valid")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
