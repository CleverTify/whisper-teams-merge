"""Ingest a Microsoft Teams transcript.

Teams gives you something our diarizer cannot: **real speaker names**. When a
transcript is supplied we take diarization strictly from it and never run
pyannote — which is both more accurate and much faster.

Real Teams VTT looks like this (CRLF, a cue-id line, HTML entities):

    WEBVTT

    3b818ebd-99cc-45a2-8419-a01ea15cb104/18-0
    00:00:14.722 --> 00:00:18.602
    <v Jan Novák>Tak jo, už by to mělo nahrávat.</v>

Teams also emits name variants for the same person across a call, so identities
are normalised before use or they become phantom speakers. That normalisation
cannot be done on the names alone — see `normalise_names`.
"""

from __future__ import annotations

import html
import logging
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

log = logging.getLogger("transcribe.teams")

CUE_TIME = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[.,](\d{3})"
)
VOICE = re.compile(r"<v\s+([^>]+?)\s*>(.*?)(?:</v>|$)", re.S)
# Teams .docx lines look like: "Jan Novák   0:14" then the text beneath.
DOCX_SPEAKER = re.compile(r"^(?P<name>[^\d]{2,60}?)\s+(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\s*$")

NAME_MATCH_THRESHOLD = 0.82
# Two labels that trade turns this often are holding a conversation, and so are
# two people however alike their names read.
MAX_ALTERNATION_RATE = 0.15
# ...unless they also talk *at the same time*, which is what one person on two
# devices looks like: both microphones catch the same words.
MIN_SIMULTANEOUS_FRACTION = 0.50


@dataclass
class TeamsCue:
    start: float
    end: float
    speaker: str
    text: str


def _secs(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _clean(text: str) -> str:
    return " ".join(html.unescape(text).replace("\r", " ").split()).strip()


# Structural characters, stripped from speaker NAMES only -- never from the
# words people actually said.
#
# A name is not free text. It goes into a `<v NAME>` cue in the exported VTT,
# a `## [00:00] NAME` heading in the Markdown, and an HTML attribute in the
# review UI. Only the first two are structural: an angle bracket ends a voice
# span early and corrupts every cue after it.
#
# Quotes are deliberately NOT stripped -- O'Brien is a name, and escaping for
# HTML is the renderer's job, done in index.html's esc(). Worth knowing why
# that escaping has to be right: the `<v ...>` pattern above excludes `>` but
# not `" `, and _clean() runs html.unescape(), so a cue written as
# `<v &quot; onfocus=... >` arrives holding a real quote.
_NAME_UNSAFE = str.maketrans({c: None for c in '<>`'})


def clean_name(name: str) -> str:
    """A speaker name that cannot break a VTT cue or a Markdown heading."""
    return _clean(name).translate(_NAME_UNSAFE).strip()


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------

def _key(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


def _evidence(cues: list[TeamsCue], a: str, b: str) -> tuple[float, float]:
    """How two labels behave towards each other: (alternation rate, overlap).

    Alternation counts turn swaps between just these two, ignoring anyone else
    in the call — two people who each reply to a third are not thereby replying
    to each other. Overlap is simultaneous speech as a share of the quieter
    label's airtime.
    """
    cues_a = [c for c in cues if c.speaker == a]
    cues_b = [c for c in cues if c.speaker == b]
    if not cues_a or not cues_b:
        return 0.0, 1.0

    pair = sorted(cues_a + cues_b, key=lambda c: c.start)
    swaps = sum(1 for x, y in zip(pair, pair[1:]) if x.speaker != y.speaker)
    alt_rate = swaps / min(len(cues_a), len(cues_b))

    shared, j = 0.0, 0
    for ca in cues_a:
        while j < len(cues_b) and cues_b[j].end <= ca.start:
            j += 1
        k = j
        while k < len(cues_b) and cues_b[k].start < ca.end:
            shared += min(ca.end, cues_b[k].end) - max(ca.start, cues_b[k].start)
            k += 1
    airtime = min(sum(c.end - c.start for c in cues_a),
                  sum(c.end - c.start for c in cues_b))
    return alt_rate, (shared / airtime if airtime else 0.0)


def normalise_names(cues: list[TeamsCue]) -> dict[str, str]:
    """Map Teams' name variants onto one canonical identity per person.

    Teams sometimes writes the same participant two ways in one transcript, and
    left alone that produces two "speakers" for one human. But a similar name is
    only a *candidate*: 'Jan Novák' and 'Janek Novák' score 0.889 and can
    easily be two different people in one meeting — Janek is a form of Jan
    and they share a surname, so the strings cannot separate them.

    The conversation can. Those two swapped turns 235 times and overlapped for
    only 4.3% of their speech: a dialogue. One person on two devices is the
    opposite — the labels almost never trade turns, and where they do the two
    streams overlap heavily because both microphones caught the same words.

    So a candidate is merged only on that evidence, and never on the name alone.
    """
    counts: dict[str, int] = {}
    for c in cues:
        if c.speaker:
            counts[c.speaker] = counts.get(c.speaker, 0) + 1

    groups: list[list[str]] = []
    for name in sorted(counts, key=lambda n: -counts[n]):
        for group in groups:
            if SequenceMatcher(None, _key(name), _key(group[0])).ratio() >= NAME_MATCH_THRESHOLD:
                group.append(name)
                break
        else:
            groups.append([name])

    canon = {name: name for name in counts}
    for group in groups:
        if len(group) == 1:
            continue
        winner = max(group, key=lambda n: counts[n])
        for member in group:
            if member == winner:
                continue
            alt, overlap = _evidence(cues, winner, member)
            same = alt < MAX_ALTERNATION_RATE or overlap > MIN_SIMULTANEOUS_FRACTION
            log.info(
                "Teams names %r / %r look alike: %d turn swaps (rate %.2f), "
                "%.0f%% simultaneous — %s",
                winner, member, round(alt * min(counts[winner], counts[member])),
                alt, overlap * 100,
                "one person, merging" if same else "two people, keeping both",
            )
            if same:
                canon[member] = winner
    return canon


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_vtt(path: Path) -> list[TeamsCue]:
    raw = path.read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n")
    blocks = [b for b in re.split(r"\n\s*\n", raw) if b.strip()]
    cues: list[TeamsCue] = []

    for block in blocks:
        m = CUE_TIME.search(block)
        if not m:
            continue
        start = _secs(*m.groups()[:4])
        end = _secs(*m.groups()[4:])
        body = block[m.end():].strip()

        found = VOICE.findall(body)
        if found:
            for name, text in found:
                text = _clean(text)
                if text:
                    cues.append(TeamsCue(start, end, clean_name(name), text))
        else:
            text = _clean(re.sub(r"<[^>]+>", "", body))
            if text:
                cues.append(TeamsCue(start, end, "", text))

    return cues


def parse_docx(path: Path) -> list[TeamsCue]:
    """Teams .docx export: a 'Name  0:14' line followed by the spoken text."""
    from docx import Document

    cues: list[TeamsCue] = []
    speaker, start = "", 0.0
    for para in Document(str(path)).paragraphs:
        line = para.text.strip()
        if not line:
            continue
        m = DOCX_SPEAKER.match(line)
        if m:
            speaker = clean_name(m.group("name"))
            parts = [int(p) for p in m.group("ts").split(":")]
            start = parts[0] * 60 + parts[1] if len(parts) == 2 else \
                parts[0] * 3600 + parts[1] * 60 + parts[2]
            continue
        text = _clean(line)
        if text and speaker:
            # end is refined below once the next cue's start is known
            cues.append(TeamsCue(start, start + 5.0, speaker, text))

    for a, b in zip(cues, cues[1:]):
        a.end = max(a.start + 0.5, min(b.start, a.start + 60.0))
    return cues


def load(path: Path) -> list[TeamsCue]:
    """Parse a Teams transcript and normalise speaker identities."""
    suffix = path.suffix.lower()
    if suffix == ".vtt":
        cues = parse_vtt(path)
    elif suffix == ".docx":
        cues = parse_docx(path)
    else:
        raise ValueError(f"unsupported transcript format {suffix!r} (expected .vtt or .docx)")

    if not cues:
        raise ValueError(f"no cues found in {path.name}")

    cues.sort(key=lambda c: c.start)
    canon = normalise_names(cues)
    for c in cues:
        c.speaker = canon.get(c.speaker, c.speaker) or "SPEAKER_00"

    speakers = sorted({c.speaker for c in cues})
    log.info(
        "Teams transcript: %d cues, %d speaker(s): %s",
        len(cues), len(speakers), ", ".join(speakers[:8]),
    )
    return cues


# ---------------------------------------------------------------------------
# Using it as the diarization source
# ---------------------------------------------------------------------------

TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)
ALIGN_WINDOW = 60.0   # seconds of context per alignment pass
ALIGN_PAD = 10.0      # overlap so a word near a boundary still finds its pair
FILL_MAX_GAP = 8      # words an unmatched run may inherit across before giving up


def _tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def _by_overlap(cues: list[TeamsCue]) -> "callable":
    """Fallback: whoever was speaking for most of this word's duration."""
    import bisect

    starts = [c.start for c in cues]

    def owner(t0: float, t1: float) -> str:
        i = max(0, bisect.bisect_right(starts, t0) - 1)
        best, best_ov = "", 0.0
        for c in cues[i : i + 8]:
            if c.start > t1:
                break
            ov = min(t1, c.end) - max(t0, c.start)
            if ov > best_ov:
                best, best_ov = c.speaker, ov
        if best:
            return best
        j = min(range(len(cues)), key=lambda k: abs(cues[k].start - t0))
        return cues[j].speaker

    return owner


def _teams_words(cues: list[TeamsCue]) -> list[dict]:
    """Teams' words with a time and an owner, spread evenly across their cue.

    `raw` keeps the original casing and punctuation so a span can be quoted back
    verbatim; `w` is the lowercased token used for matching. `i` is the word's
    position, which is how a turn later recovers its own slice of Teams.
    """
    out = []
    for c in cues:
        raws = c.text.split()
        ws = _tokens(c.text)
        if not ws:
            continue
        step = (c.end - c.start) / len(ws)
        for k, w in enumerate(ws):
            out.append({"w": w, "raw": raws[k] if k < len(raws) else w,
                        "t": c.start + step * (k + 0.5), "speaker": c.speaker,
                        "i": len(out)})
    return out


_WORDS_CACHE: tuple | None = None


def _teams_words_cached(cues: list[TeamsCue]) -> list[dict]:
    """One transcript is rebuilt for every turn otherwise."""
    global _WORDS_CACHE
    if _WORDS_CACHE is None or _WORDS_CACHE[0] is not cues:
        _WORDS_CACHE = (cues, _teams_words(cues))
    return _WORDS_CACHE[1]


def aligned_text(words: list[dict], cues: list[TeamsCue],
                 speaker: str | None = None) -> str:
    """Teams' own words for exactly these words of ours.

    Selecting Teams text by time overlap smears it: cue boundaries do not line
    up with our turns, so a turn picks up its neighbours' sentences. Feeding
    that to the LLM as "the same audio" is worse than useless, because the model
    is then asked to reconcile lines against text that belongs elsewhere.

    The word alignment already knows which Teams word each of our words matched,
    so a turn can quote precisely its own span.

    Pass `speaker` to keep the quote to that person. The span is widened at both
    ends to cover our unmatched words, and at a turn boundary that reaches into
    whoever spoke next: 7.8% of the text handed to the model belonged to someone
    else, and it would sometimes merge their words in — corrupting both the
    wording and the attribution. Teams records an owner per word, so this is
    simply filtered out.
    """
    stream = _teams_words_cached(cues)
    hits = [(k, w["tw"]) for k, w in enumerate(words) if w.get("tw") is not None]
    if not hits:
        return ""

    # Clipping to the outermost *matched* words would drop precisely the
    # disagreements worth showing: our 'lůna' does not match their 'runa', so
    # ending the span at the last match hides their reading of that very word.
    # Extend by however many of our words fell outside the matched range.
    (first_k, lo), (last_k, hi) = hits[0], hits[-1]
    lo = max(0, lo - first_k)
    hi = min(len(stream) - 1, hi + (len(words) - 1 - last_k))
    span = stream[lo : hi + 1]
    if speaker:
        span = [x for x in span if x["speaker"] == speaker]
    return " ".join(x["raw"] for x in span).strip()


def assign_speakers(segments: list[dict], cues: list[TeamsCue]) -> list[dict]:
    """Label our words with Teams' speakers by matching *words*, not clocks.

    This replaces pyannote entirely: the speaker timeline is taken strictly
    from Teams, which knows who actually held the microphone.

    Matching on time alone cannot survive crosstalk. Where two people overlap —
    14% of airtime on a real call — both have cues covering the same seconds, so
    picking the cue with the larger overlap flips from word to word and shreds
    one person's sentence across two speakers:

        Jan Novák        ...No jako já tam
        Petra Dvořáková  dám mobil, že
        Jan Novák        jo, protože když budeme preferovat
        Petra Dvořáková  mobil, tak bude asi lepší, aby tam byl mobil.

    But both sides transcribed the same audio, so their word *sequences* line
    up even where the clocks do not. A word of ours that matches a word of
    theirs simply inherits its owner, and crosstalk stops mattering. Alignment
    runs inside time windows because a global diff over ~11k words drifts — one
    repeated Czech filler is enough to pair our minute 4 with their minute 40.

    Words with no counterpart (our ASR heard something else) take the owner of
    the nearest matched word, and a stretch with no match at all falls back to
    time overlap.
    """
    if not cues:
        return segments

    ours: list[dict] = []
    for seg in segments:
        for w in seg.get("words") or []:
            tok = _tokens(w.get("word", ""))
            if tok and w.get("start") is not None and w.get("end") is not None:
                ours.append({"w": tok[0], "ref": w,
                             "t": (float(w["start"]) + float(w["end"])) / 2})

    theirs = _teams_words(cues)
    owner = _by_overlap(cues)

    if ours and theirs:
        span = max(ours[-1]["t"], theirs[-1]["t"])
        t = 0.0
        while t < span + ALIGN_WINDOW:
            lo, hi = t - ALIGN_PAD, t + ALIGN_WINDOW + ALIGN_PAD
            a = [x for x in ours if lo <= x["t"] < hi]
            b = [x for x in theirs if lo <= x["t"] < hi]
            if a and b:
                sm = SequenceMatcher(None, [x["w"] for x in a], [x["w"] for x in b],
                                     autojunk=False)
                for i, j, n in sm.get_matching_blocks():
                    for k in range(n):
                        wa = a[i + k]
                        if t <= wa["t"] < t + ALIGN_WINDOW:
                            wa["matched"] = b[j + k]["speaker"]
                            # Remember which Teams word this is, so the LLM can
                            # later be shown the *corresponding* Teams text.
                            wa["ref"]["tw"] = b[j + k]["i"]
            t += ALIGN_WINDOW

    # Unmatched words take the nearer confident neighbour; if a whole stretch is
    # unmatched there is nothing to lean on, so fall back to the clock.
    anchors = [i for i, x in enumerate(ours) if x.get("matched")]
    if anchors:
        import bisect

        for i, x in enumerate(ours):
            if x.get("matched"):
                x["speaker"] = x["matched"]
                continue
            p = bisect.bisect_left(anchors, i)
            left = anchors[p - 1] if p else None
            right = anchors[p] if p < len(anchors) else None
            if left is not None and right is not None:
                near = left if (i - left) <= (right - i) else right
            else:
                near = left if left is not None else right
            gap = abs(i - near)
            x["speaker"] = (ours[near]["matched"] if gap <= FILL_MAX_GAP
                            else owner(float(x["ref"]["start"]), float(x["ref"]["end"])))
    else:
        for x in ours:
            x["speaker"] = owner(float(x["ref"]["start"]), float(x["ref"]["end"]))

    for x in ours:
        x["ref"]["speaker"] = x["speaker"]

    # A segment's own label is whatever most of its words say; segments with no
    # usable words fall back to the clock.
    for seg in segments:
        labels = [w.get("speaker") for w in (seg.get("words") or []) if w.get("speaker")]
        seg["speaker"] = (Counter(labels).most_common(1)[0][0] if labels
                          else owner(float(seg.get("start", 0.0)),
                                     float(seg.get("end", 0.0))))
    return segments


def text_at(cues: list[TeamsCue], start: float, end: float) -> str:
    """Teams' own words for a time window — the second opinion the LLM merges."""
    parts = [c.text for c in cues if c.end > start and c.start < end]
    return " ".join(parts).strip()
