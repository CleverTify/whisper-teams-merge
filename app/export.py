"""Outputs: Markdown, WebVTT and JSON.

All three read the same `(turns, name_map, meta)`, so renaming a speaker
regenerates everything without re-transcribing.
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
from pathlib import Path

from app.audio import format_clock, format_timestamp
from app.config import JobPaths, dump_json
from app.diarize import Turn, speaker_stats

log = logging.getLogger("transcribe.export")

MAX_CHARS_PER_LINE = 42
MAX_LINES = 2
MAX_CUE_SECONDS = 6.0
SENTENCE_END = tuple(".!?…")


def display(name_map: dict[str, str], speaker: str) -> str:
    return name_map.get(speaker, speaker)


def duration_line(meta: dict) -> str:
    """Say how much audio the transcript actually covers.

    Reporting the source duration on a trimmed run is misleading — a two-minute
    smoke test would print "01:20:46" and read like a full meeting.
    """
    src, pre, prep = meta.get("source", {}), meta.get("preprocess", {}), meta.get("prepared", {})
    full = float(src.get("duration") or 0.0)
    used = float(prep.get("duration") or pre.get("duration_limit") or 0.0) or full
    start = float(pre.get("start") or 0.0)
    if full and used and abs(used - full) > 1.0:
        return (f"{format_clock(used)} transcribed "
                f"({format_clock(start)}–{format_clock(start + used)} of {format_clock(full)})")
    return format_clock(full)


def _language_summary(langs: dict) -> str:
    shares = langs.get("shares") or {}
    if not shares:
        return langs.get("dominant", "?")
    return ", ".join(f"{k} {v:.0%}" for k, v in list(shares.items())[:6])


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def write_markdown(path: Path, turns: list[Turn], name_map: dict, meta: dict,
                   screen: list[dict] | None = None) -> None:
    src = meta.get("source", {})
    spk = meta.get("speakers_from", {})
    llm = meta.get("llm", {})
    lines = [
        f"# {src.get('name', 'Transcript')}",
        "",
        f"- **Duration:** {duration_line(meta)}",
        f"- **Languages:** {_language_summary(meta.get('languages', {}))}",
        f"- **Speakers:** {len({t.speaker for t in turns})} "
        f"(from {spk.get('source', '?')})",
        f"- **ASR:** {meta.get('asr', {}).get('backend', '?')} "
        f"/ {meta.get('asr', {}).get('model', '?')}",
    ]
    if llm.get("mode") not in (None, "", "disabled", "skipped"):
        lines.append(
            f"- **LLM {llm['mode']}:** {llm.get('turns_changed', 0)} turns improved, "
            f"{llm.get('turns_rejected', 0)} rejected by guardrail"
        )
    lines += ["", "---", ""]

    last_lang = None
    pending = sorted(screen or [], key=lambda x: x["t"])
    seen_caption = None
    seen_at = None
    for turn in turns:
        # Screen annotations sit above the turn they precede, so a reader sees
        # what was on screen before reading what was said about it. Repeated
        # captions are dropped: a static screen would otherwise repeat itself
        # down the page and bury the conversation.
        while pending and pending[0]["t"] <= turn.start:
            entry = pending.pop(0)
            cap = entry.get("caption") or ""
            if not cap:
                continue
            # Near-duplicates, not just exact ones. A static screen produces a
            # caption per scene change and the model rewords it every time, so
            # matching on equality left 415 annotations in a 592-turn
            # transcript — one per turn, burying the conversation it was meant
            # to illuminate.
            if seen_caption and SequenceMatcher(
                    None, cap.lower(), seen_caption.lower()).ratio() > 0.5:
                continue
            # A reader wants the screen marked when it changes, not every time
            # a cursor moves. Scene detection fires on scrolling and typing, so
            # without a floor on the interval the annotations outnumber the
            # turns they are meant to annotate.
            if seen_at is not None and entry["t"] - seen_at < 45.0:
                continue
            lines += [f"> *screen [{format_clock(entry['t'])}]:* {cap}", ""]
            seen_caption, seen_at = cap, entry["t"]

        head = f"## [{format_clock(turn.start)}] {display(name_map, turn.speaker)}"
        if turn.language and turn.language != last_lang:
            head += f"  `{turn.language}`"
            last_lang = turn.language
        lines += [head, "", turn.text, ""]

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# WebVTT
# ---------------------------------------------------------------------------

def _split_evenly(turn: Turn, budget: int) -> list[tuple[float, float, str]]:
    words = turn.text.split()
    if not words:
        return []
    chunks, cur = [], []
    for w in words:
        cand = " ".join(cur + [w])
        if cur and len(cand) > budget:
            chunks.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        chunks.append(" ".join(cur))
    total = sum(len(c) for c in chunks) or 1
    out, cursor = [], turn.start
    for c in chunks:
        span = turn.duration * (len(c) / total)
        out.append((cursor, min(turn.end, cursor + span), c))
        cursor += span
    return out


def _cue(words: list[dict], speaker: str) -> dict:
    return {"start": float(words[0]["start"]), "end": float(words[-1]["end"]),
            "speaker": speaker,
            "text": " ".join(w["word"].strip() for w in words).strip()}


def cues_from_turns(turns: list[Turn], name_map: dict) -> list[dict]:
    budget = MAX_CHARS_PER_LINE * MAX_LINES
    cues: list[dict] = []
    for turn in turns:
        speaker = display(name_map, turn.speaker)
        timed = [w for w in turn.words
                 if w.get("start") is not None and w.get("end") is not None]
        if not timed:
            for s, e, t in _split_evenly(turn, budget):
                cues.append({"start": s, "end": e, "speaker": speaker, "text": t})
            continue
        buf: list[dict] = []
        for w in timed:
            tok = w["word"].strip()
            if not tok:
                continue
            cand = (" ".join(x["word"].strip() for x in buf) + " " + tok).strip()
            if buf and (len(cand) > budget or (w["end"] - buf[0]["start"]) > MAX_CUE_SECONDS):
                cues.append(_cue(buf, speaker))
                buf = []
            buf.append(w)
            if tok.endswith(SENTENCE_END) and len(
                " ".join(x["word"].strip() for x in buf)
            ) > budget * 0.5:
                cues.append(_cue(buf, speaker))
                buf = []
        if buf:
            cues.append(_cue(buf, speaker))

    for prev, nxt in zip(cues, cues[1:]):
        if nxt["start"] < prev["end"]:
            mid = (prev["end"] + nxt["start"]) / 2
            prev["end"] = nxt["start"] = mid
    return [c for c in cues if c["end"] > c["start"] and c["text"].strip()]


def _wrap(text: str) -> str:
    words, lines, cur = text.split(), [], ""
    for w in words:
        cand = f"{cur} {w}".strip()
        if cur and len(cand) > MAX_CHARS_PER_LINE and len(lines) < MAX_LINES - 1:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return "\n".join(lines[:MAX_LINES])


def write_vtt(path: Path, cues: list[dict]) -> None:
    out = ["WEBVTT", ""]
    for c in cues:
        out.append(f"{format_timestamp(c['start'], sep='.')} --> "
                   f"{format_timestamp(c['end'], sep='.')}")
        out.append(f"<v {c['speaker']}>{_wrap(c['text'])}")
        out.append("")
    path.write_text("\n".join(out), encoding="utf-8")


# ---------------------------------------------------------------------------

def write_json(path: Path, *, turns, name_map, meta, stats,
               screen=None) -> None:
    path.write_text(
        dump_json(
            {
                "metadata": meta,
                "speakers": [{**r, "name": display(name_map, r["speaker"])} for r in stats],
                "name_map": name_map,
                # Display-only: what was on screen, never part of the
                # transcript text and never fed back into it.
                "screen": screen or [],
                "turns": [{**t.to_dict(), "name": display(name_map, t.speaker)} for t in turns],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


# Files earlier versions produced. They are no longer regenerated, so leaving
# them behind means someone downloads a stale transcript that silently
# disagrees with the current one.
OBSOLETE = ("transcript.txt", "subtitles.srt", "subtitles.vtt",
            "minutes.docx", "speakers.json")


def _drop_stale(paths: JobPaths) -> None:
    for name in OBSOLETE:
        p = paths.root / name
        if p.exists():
            p.unlink()
            log.info("removed stale artifact %s", name)


def export_all(paths: JobPaths, *, turns: list[Turn], name_map: dict,
               meta: dict, screen: list[dict] | None = None) -> dict[str, str]:
    _drop_stale(paths)
    stats = speaker_stats(turns)
    write_markdown(paths.transcript_md, turns, name_map, meta, screen)
    write_vtt(paths.vtt, cues_from_turns(turns, name_map))
    write_json(paths.result_json, turns=turns, name_map=name_map,
               meta=meta, stats=stats, screen=screen)

    produced = {
        "transcript.md": str(paths.transcript_md),
        "transcript.vtt": str(paths.vtt),
        "result.json": str(paths.result_json),
    }
    log.info("exported %d artifacts to %s", len(produced), paths.root)
    return produced
