"""Hallucination and repetition filtering.

Whisper was trained on a great deal of subtitle data, so during silence it
tends to emit subtitle boilerplate — "Titulky vytvořil…", "Subtitles by the
Amara.org community", "Vielen Dank fürs Zuschauen". On an 80-minute meeting
with long pauses this is not rare; it is expected. It also loops, repeating a
phrase dozens of times.

Everything removed here is written to `removed.jsonl` with a reason. Silent
deletion would make the transcript look cleaner than it is and would hide
genuine ASR failures.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path

log = logging.getLogger("transcribe.cleanup")

# Matched case-insensitively against the whole (stripped) segment text.
# "*" applies to every language.
HALLUCINATION_PATTERNS: dict[str, list[str]] = {
    "*": [
        r"^\W*$",                                   # punctuation / symbols only
        r"^[\s♪♫»«\-–—.·]*$",
        r"amara\.org",
        r"^\[?\s*(music|hudba|musik|musique|música|muzyka)\s*\]?[.!]?$",
        r"^\[?\s*(applause|potlesk|aplauz|applaus|apllausi)\s*\]?[.!]?$",
        r"^\(?\s*(inaudible|nesrozumiteln\w*)\s*\)?[.!]?$",
    ],
    "cs": [
        r"titulky\s+(vytvo[řr]il|z\s+odposlechu|pro)",
        r"p[řr]epis\s+a\s+korektura",
        r"^[čc]esk[éeě]\s+titulky",
        r"korektura[:\s]",
        r"d[íi]ky\s+za\s+sledov[áa]n[íi]",
    ],
    "sk": [
        r"titulky\s+(vytvoril|pre)",
        r"preklad[:\s]",
        r"[ďd]akujem\s+za\s+pozeranie",
    ],
    "en": [
        r"^thanks?\s+(you\s+)?for\s+watching",
        r"subtitles?\s+by",
        r"please\s+subscribe",
        r"^thank\s+you\.?$",
    ],
    "de": [
        r"untertitel\s+(von|im\s+auftrag)",
        r"vielen\s+dank\s+f[üu]rs?\s+zuschauen",
    ],
    "pl": [
        r"napisy\s+(stworzone|:)",
        r"dzi[eę]kuj[eę]\s+za\s+ogl[ąa]danie",
    ],
    "fr": [r"sous-titres?\s+(r[ée]alis|par)", r"merci\s+d'avoir\s+regard"],
    "es": [r"subt[íi]tulos?\s+(realizados|por)", r"gracias\s+por\s+ver"],
    "it": [r"sottotitoli\s+(e\s+revisione|a\s+cura)", r"grazie\s+per\s+aver"],
    "nl": [r"ondertiteling\s+(door|:)", r"bedankt\s+voor\s+het\s+kijken"],
    "pt": [r"legendas?\s+(pela|por)", r"obrigado\s+por\s+assistir"],
    "hu": [r"felirat(ot|:)\s"],
    "ro": [r"subtitrare[a:]?\s+(de|realizat)"],
}

_COMPILED: dict[str, list[re.Pattern]] = {
    lang: [re.compile(p, re.IGNORECASE | re.UNICODE) for p in pats]
    for lang, pats in HALLUCINATION_PATTERNS.items()
}

MAX_REPEATS = 3          # allow genuine emphasis ("ano, ano, ano")
LOOP_COVERAGE = 0.6      # fraction of the segment made of one repeated n-gram


def _matches(text: str, language: str) -> str | None:
    for pattern in _COMPILED.get("*", []):
        if pattern.search(text):
            return f"boilerplate:{pattern.pattern}"
    for pattern in _COMPILED.get((language or "").lower(), []):
        if pattern.search(text):
            return f"boilerplate:{language}:{pattern.pattern}"
    return None


def collapse_loop(text: str) -> tuple[str, bool]:
    """Collapse Whisper's repetition loops, keeping up to MAX_REPEATS copies."""
    words = text.split()
    if len(words) < 8:
        return text, False

    for n in range(1, 6):
        grams = [" ".join(words[i : i + n]) for i in range(0, len(words) - n + 1, n)]
        if len(grams) < 4:
            continue
        gram, count = Counter(grams).most_common(1)[0]
        if count >= 4 and (count * n) / len(words) >= LOOP_COVERAGE:
            kept = " ".join([gram] * min(count, MAX_REPEATS))
            remainder = [g for g in grams if g != gram]
            rebuilt = " ".join([kept] + remainder).strip()
            return rebuilt, True
    return text, False


def clean_segments(
    segments: list[dict], *, removed_path: Path | None = None
) -> tuple[list[dict], list[dict]]:
    """Drop hallucinated segments and collapse loops.

    Returns `(kept, removed)`; `removed` entries carry the original text and a
    machine-readable reason.
    """
    kept: list[dict] = []
    removed: list[dict] = []
    recent: list[str] = []

    for seg in segments:
        text = (seg.get("text") or "").strip()
        language = seg.get("language", "")

        if not text:
            removed.append({**_stamp(seg), "reason": "empty"})
            continue

        reason = _matches(text, language)
        if reason:
            removed.append({**_stamp(seg), "reason": reason})
            continue

        collapsed, looped = collapse_loop(text)
        if looped:
            seg = {**seg, "text": collapsed}
            removed.append({**_stamp(seg), "reason": "loop-collapsed", "kept_as": collapsed})
            text = collapsed

        # Same sentence emitted for several consecutive segments is a stall,
        # not speech. Compare against a short window so genuine repeated
        # phrases across a long meeting survive.
        norm = re.sub(r"\W+", " ", text.lower()).strip()
        if norm and recent.count(norm) >= 2:
            removed.append({**_stamp(seg), "reason": "consecutive-duplicate"})
            continue
        recent.append(norm)
        if len(recent) > 4:
            recent.pop(0)

        kept.append(seg)

    dropped = sum(1 for r in removed if r["reason"] != "loop-collapsed")
    if removed:
        log.info(
            "cleanup: dropped %d segment(s), collapsed %d loop(s)",
            dropped, len(removed) - dropped,
        )
    if removed_path is not None:
        removed_path.parent.mkdir(parents=True, exist_ok=True)
        with removed_path.open("w", encoding="utf-8") as fh:
            for row in removed:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return kept, removed


def drop_unvoiced(
    segments: list[dict], *, min_logprob: float = -1.0, min_no_speech: float = 0.8
) -> tuple[list[dict], list[dict]]:
    """Remove low-confidence segments that diarization saw no speaker in.

    All three conditions must hold — high no-speech probability, poor average
    log-probability, and no overlapping speaker — so that a genuinely quiet but
    real utterance is not discarded.
    """
    kept, removed = [], []
    for seg in segments:
        no_speech = seg.get("no_speech_prob")
        logprob = seg.get("avg_logprob")
        has_speaker = bool(seg.get("speaker")) or any(
            w.get("speaker") for w in (seg.get("words") or [])
        )
        if (
            no_speech is not None
            and logprob is not None
            and no_speech > min_no_speech
            and logprob < min_logprob
            and not has_speaker
        ):
            removed.append({**_stamp(seg), "reason": "unvoiced-low-confidence"})
        else:
            kept.append(seg)
    if removed:
        log.info("cleanup: dropped %d unvoiced low-confidence segment(s)", len(removed))
    return kept, removed


def append_removed(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _stamp(seg: dict) -> dict:
    return {
        "start": round(float(seg.get("start", 0.0)), 3),
        "end": round(float(seg.get("end", 0.0)), 3),
        "language": seg.get("language", ""),
        "text": (seg.get("text") or "").strip(),
    }
