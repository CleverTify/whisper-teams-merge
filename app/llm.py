"""Local-LLM post-processing of the transcript.

Two modes:

* **merge** — a Teams transcript is available, so the LLM sees two independent
  ASR readings of the same audio and picks the better wording per turn.
* **cleanup** — only our transcript exists; fix obvious ASR slips, proper nouns
  and punctuation.

The whole design point is the guardrail. A model told to "improve a transcript"
will cheerfully rewrite it, merge turns, answer it, or translate it. Every
returned turn is therefore checked against the turn it replaces — length ratio
and token overlap — and anything that drifts is **discarded in favour of the
original**, with the rejection logged. A silent LLM rewrite would be far worse
than a few uncorrected ASR errors.

Talks to llama.cpp's OpenAI-compatible endpoint over stdlib urllib, so there is
no extra dependency.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

log = logging.getLogger("transcribe.llm")

SYSTEM_CLEANUP = (
    "You correct automatic speech-recognition output. You are given numbered "
    "lines of a transcript. Fix only clear recognition errors, proper nouns, "
    "casing and punctuation.\n"
    "STRICT RULES:\n"
    "1. Return exactly the same number of lines, with the same [n] numbering.\n"
    "2. Never merge, split, reorder, add or delete lines.\n"
    "3. Keep the original language. Never translate.\n"
    "4. Never add commentary, headings or explanation.\n"
    "5. If a line looks fine, repeat it unchanged.\n"
    "6. Do not invent content that is not already implied by the words.\n"
    "7. Some windows begin with a list of words that were visible ON SCREEN "
    "at that moment. Use one only to correct an obvious mishearing of that "
    "same word. Never insert one where the line has no corresponding word, "
    "and never repeat or mention the list itself."
)

SYSTEM_MERGE = (
    "You are given two automatic transcripts of the SAME audio. Each numbered "
    "line gives A (from Whisper) and B (from Microsoft Teams) for the same "
    "moment.\n"
    "**A is the transcript. Your default is to return A completely unchanged.** "
    "B is a second opinion, and it is often worse than A.\n"
    "Copy a word from B ONLY when BOTH of these hold:\n"
    "  (a) the word in A is not a real Czech word, or is an obvious mishearing "
    "of a name, product or technical term; AND\n"
    "  (b) B's version of that word is a real name, product or technical term.\n"
    "In every other case keep A's word, even if B looks plausible.\n"
    "NEVER take from B: different word choice, different word order, extra or "
    "missing words, punctuation, colloquial spellings (bejt, nějakej, maj, "
    "zas), or anything you merely prefer.\n"
    "STRICT RULES:\n"
    "1. Return exactly the same number of lines, with the same [n] numbering.\n"
    "2. Output the corrected line only — no 'A:' or 'B:' prefix, no commentary.\n"
    "3. Never merge, split, reorder, add or delete lines.\n"
    "4. Keep the original language. Never translate.\n"
    "5. Most lines need no change at all. Returning A verbatim is the correct "
    "answer far more often than not.\n"
    "6. Some windows begin with a list of words that were visible ON SCREEN "
    "at that moment. A word in A may be replaced by one of those ONLY when "
    "A's word is a mishearing of it. Never insert a screen word where A has "
    "no corresponding word, and never repeat or mention the list itself."
)

def merge_system(glossary: list[str]) -> str:
    """SYSTEM_MERGE, plus a spelling list when the operator supplied one.

    Phrased as spelling rather than vocabulary on purpose. "These words may
    appear" invites the model to find them; "this is how these words are
    spelled" only licenses fixing one that is already there, which is what
    `screen_term_ok()` then enforces mechanically.
    """
    if not glossary:
        return SYSTEM_MERGE
    return SYSTEM_MERGE + (
        "\n7. These names and technical terms are spelled correctly here: "
        + ", ".join(glossary)
        + ". If a word in A is an obvious mishearing of one of them, use this "
          "spelling. Do NOT insert any of these words where A has no "
          "corresponding word, and do not mention this list."
    )


SLOW_TOKENS_PER_SECOND = 25.0  # a healthy 8B on a modern GPU is 60-80 tok/s
MAX_WINDOW_SPLITS = 3          # 12 lines -> 6 -> 3 -> 1 before giving up on a window

LINE = re.compile(r"^\s*\[(\d+)\]\s*(.*)$")
# The A:/B: scaffolding from the merge prompt, echoed back by the model.
LABEL = re.compile(r"^\s*[AB]\s*:\s*")
_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass
class LLMStats:
    mode: str = ""
    model: str = ""
    windows: int = 0
    turns_sent: int = 0
    turns_changed: int = 0
    turns_rejected: int = 0
    rejections: list[dict] = field(default_factory=list)
    failed_windows: int = 0
    lines_missing: int = 0
    wake_seconds: float = 0.0
    labels_stripped: int = 0
    screen_terms_applied: int = 0
    screen_terms_rejected: int = 0
    token_rates: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "model": self.model,
            "windows": self.windows,
            "turns_sent": self.turns_sent,
            "turns_changed": self.turns_changed,
            "turns_rejected": self.turns_rejected,
            "failed_windows": self.failed_windows,
            "lines_missing": self.lines_missing,
            "wake_seconds": round(self.wake_seconds, 1),
            "labels_stripped": self.labels_stripped,
            "screen_terms_applied": self.screen_terms_applied,
            "screen_terms_rejected": self.screen_terms_rejected,
            "median_tokens_per_second": (
                sorted(self.token_rates)[len(self.token_rates) // 2]
                if self.token_rates else None
            ),
            "rejections": self.rejections[:50],
        }


class LLMUnavailable(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

def _post(base_url: str, payload: dict, timeout: int) -> dict:
    url = base_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def warm(base_url: str, model: str, timeout: int = 300) -> float:
    """Wake a sleeping server, and return how long it took.

    The llm service runs with `--sleep-idle-seconds`, so while ASR and pyannote
    are working the model is evicted and its ~5.9 GB goes back to them. The
    first request afterwards pays ~60s to load it again. Left inside the window
    loop that reads as a stall on window 1: it can exceed the per-window
    timeout, trigger the split-retry, and poison the throughput figures the
    degradation warning depends on. Paying it once, here, keeps all of that
    honest.

    Note `/v1/models` answers 200 while asleep, so a liveness poll cannot tell
    the difference — only a real completion wakes it.
    """
    t0 = time.monotonic()
    try:
        _post(base_url,
              {"model": model,
               "messages": [{"role": "user", "content": "ok"}],
               "max_tokens": 1, "temperature": 0.1,
               "chat_template_kwargs": {"enable_thinking": False}},
              timeout)
    except Exception as exc:
        log.warning("LLM warm-up failed (%s); the first window will pay the cost",
                    type(exc).__name__)
        return 0.0
    took = time.monotonic() - t0
    if took > 5:
        log.info("LLM woke from idle sleep in %.0fs", took)
    return took


def available(base_url: str, timeout: int = 5) -> bool:
    try:
        req = urllib.request.Request(base_url.rstrip("/") + "/models", method="GET")
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except Exception as exc:
        log.warning("local LLM not reachable at %s (%s)", base_url, type(exc).__name__)
        return False


# ---------------------------------------------------------------------------
# Guardrail
# ---------------------------------------------------------------------------

def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _WORD.findall(text)}


def acceptable(original: str, candidate: str, settings) -> tuple[bool, str]:
    """Decide whether an LLM line may replace the original.

    Cheap checks, deliberately strict: a rejected correction costs nothing,
    an accepted hallucination corrupts the transcript.
    """
    cand = candidate.strip()
    if not cand:
        return False, "empty"

    o_len, c_len = len(original.strip()), len(cand)
    if o_len == 0:
        return False, "empty-original"
    ratio = c_len / o_len
    if ratio > settings.llm_max_length_ratio:
        return False, f"too-long({ratio:.2f})"
    if ratio < settings.llm_min_length_ratio:
        return False, f"too-short({ratio:.2f})"

    o_tok = _tokens(original)
    if o_tok:
        overlap = len(o_tok & _tokens(cand)) / len(o_tok)
        if overlap < settings.llm_min_token_overlap:
            return False, f"low-overlap({overlap:.2f})"

    # Model breaking character and talking to us.
    if re.match(r"^(here (is|are)|sure[,!]|corrected|note:|i )", cand, re.I):
        return False, "commentary"
    return True, ""


def screen_term_ok(original: str, candidate: str, terms: list[str]) -> tuple[bool, str]:
    """Allow a screen term as a substitution, never as an addition.

    `acceptable()` judges by length ratio and token overlap, and neither can see
    this: a one-word term appended to a twelve-word turn scores a ratio of about
    1.08 and an overlap of 0.92, and sails straight through. That is the same
    blind spot that let a two-character "A: " prefix reach 292 shipped turns,
    one level up.

    A correction replaces a misheard word, so the token count should barely
    move. Growth means the model added a word that was on screen but never
    spoken, which is the specific way this feature would corrupt a transcript.
    """
    if not terms:
        return True, ""
    o_tok, c_tok = _WORD.findall(original), _WORD.findall(candidate)
    lowered = {t.lower() for t in terms}
    added = {t.lower() for t in c_tok} - {t.lower() for t in o_tok}
    if not (added & lowered):
        return True, ""
    if len(c_tok) > len(o_tok) + 1:
        return False, "screen-term-inserted"
    return True, "screen-term-applied"


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def windows(turns: list, seconds: float) -> list[list[int]]:
    """Group turn indices into ~`seconds` windows, never splitting a turn."""
    out: list[list[int]] = []
    cur: list[int] = []
    start = None
    for i, t in enumerate(turns):
        if start is None:
            start = t.start
        cur.append(i)
        if t.end - start >= seconds:
            out.append(cur)
            cur, start = [], None
    if cur:
        out.append(cur)
    return out


def _numbered(texts: list[str]) -> str:
    return "\n".join(f"[{i+1}] {t}" for i, t in enumerate(texts))


def _strip_label(text: str) -> str:
    """Drop a scaffold label the model echoed back.

    The merge prompt shows each line as `[n] A: ours` / `B: theirs`, and the
    model frequently replies `[n] A: corrected`. Stripping only the `[n]` left
    a literal `A: ` at the head of the turn, which reached transcript.md and
    transcript.vtt on 292 of 592 turns before anyone noticed — the guardrail
    could not catch it because two extra characters change neither the length
    ratio nor the token overlap enough to matter.
    """
    return LABEL.sub("", text).strip()


def _parse_numbered(reply: str, expected: int) -> dict[int, str]:
    out: dict[int, str] = {}
    for raw in reply.splitlines():
        m = LINE.match(raw)
        if not m:
            continue
        idx = int(m.group(1))
        if 1 <= idx <= expected:
            out[idx] = _strip_label(m.group(2))
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def process(turns: list, settings, *, teams_cues=None, screen=None,
            progress=None) -> LLMStats:
    """Correct `turns` in place. Returns what happened, including rejections."""
    mode = "merge" if teams_cues else "cleanup"
    stats = LLMStats(mode=mode, model=settings.llm_model)

    if not settings.llm_enabled:
        log.info("LLM post-processing disabled")
        return stats
    if not available(settings.llm_base_url):
        raise LLMUnavailable(
            f"local LLM not reachable at {settings.llm_base_url}. "
            "Is the `llm` service running?  docker compose up llm"
        )

    stats.wake_seconds = warm(settings.llm_base_url, settings.llm_model,
                              max(settings.llm_timeout, 300))

    groups = windows(turns, settings.llm_window_seconds)
    stats.windows = len(groups)
    glossary = settings.glossary_terms
    system = merge_system(glossary) if mode == "merge" else SYSTEM_CLEANUP
    log.info("LLM %s: %d turns in %d window(s) via %s",
             mode, len(turns), len(groups), settings.llm_model)

    slow_windows = 0

    def run_window(idxs: list[int], w: int, depth: int = 0) -> None:
        nonlocal slow_windows
        ours = [turns[i].text for i in idxs]
        if depth == 0:
            stats.turns_sent += len(idxs)

        # On-screen words for this window, as an unnumbered preamble — never a
        # per-line "C:" channel. The A:/B: scaffolding was echoed back into 292
        # of 592 shipped turns; a preamble offers no line-shaped template to
        # copy, and _parse_numbered drops anything unnumbered, so even a
        # verbatim echo cannot reach the transcript.
        #
        # OCR terms only. VLM captions are prose about the meeting, and prose in
        # a prompt whose whole discipline is "return A unchanged" is an
        # invitation to rewrite A into something that sounds like the caption.
        preamble = ""
        window_terms: list[str] = []
        if screen is not None:
            window_terms = screen.terms_for(
                " ".join(ours), turns[idxs[0]].start, turns[idxs[-1]].end)
            if window_terms:
                preamble = ("Words visible on screen during this passage: "
                            + ", ".join(window_terms) + "\n\n")

        if mode == "merge":
            from app import teams as teams_mod

            # Per line, not one block for the window: Teams' cue boundaries do
            # not line up with our turns, so selecting its text by time drags in
            # the neighbouring sentences and the model cannot tell which part of
            # B answers which line of A.
            pairs = []
            for n, i in enumerate(idxs, 1):
                theirs = teams_mod.aligned_text(turns[i].words or [], teams_cues,
                                                speaker=turns[i].speaker)
                pairs.append(f"[{n}] A: {turns[i].text}\n    B: {theirs or '(none)'}")
            user = (
                preamble
                + f"{len(ours)} lines, each with both readings:\n\n"
                + "\n".join(pairs)
                + f"\n\nReturn exactly {len(ours)} corrected lines, "
                  f"numbered [1]..[{len(ours)}]."
            )
        else:
            user = (
                preamble
                + f"{_numbered(ours)}\n\n"
                f"Return exactly {len(ours)} corrected lines, numbered [1]..[{len(ours)}]."
            )

        t0 = time.monotonic()
        try:
            resp = _post(
                settings.llm_base_url,
                {
                    "model": settings.llm_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 2048,
                    # Qwen3 thinks out loud unless told not to; the reasoning
                    # would be parsed as transcript lines.
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                settings.llm_timeout,
            )
            reply = resp["choices"][0]["message"]["content"]
        except Exception as exc:
            # Losing the window loses every correction in it — 16% of the
            # transcript went unmerged this way on a machine whose GPU was
            # running at a sixth of its clock. Half the lines is half the
            # prompt and half the generation, so a split usually fits inside
            # the same timeout where the whole window did not.
            if len(idxs) > 1 and depth < MAX_WINDOW_SPLITS:
                mid = len(idxs) // 2
                log.info("LLM window %d/%d timed out; retrying as %d + %d lines",
                         w, len(groups), mid, len(idxs) - mid)
                run_window(idxs[:mid], w, depth + 1)
                run_window(idxs[mid:], w, depth + 1)
                return
            stats.failed_windows += 1
            log.warning("LLM window %d/%d failed (%s); keeping original",
                        w, len(groups), type(exc).__name__)
            return

        # A llama.cpp server left running for many hours degrades badly — one
        # here fell from ~80 tok/s to 7 without erroring. Nothing failed
        # loudly: replies simply came back short and truncated, so most lines
        # were left uncorrected and the transcript quietly got worse. Slow is
        # therefore a quality signal, not just a speed one, and worth saying.
        elapsed = max(time.monotonic() - t0, 1e-6)
        produced = (resp.get("usage") or {}).get("completion_tokens") or 0
        if produced:
            rate = produced / elapsed
            stats.token_rates.append(rate)
            if rate < SLOW_TOKENS_PER_SECOND:
                slow_windows += 1

        parsed = _parse_numbered(reply, len(ours))
        missing = [n for n in range(1, len(ours) + 1) if n not in parsed]
        if missing:
            stats.lines_missing += len(missing)
        for n, turn_idx in enumerate(idxs, 1):
            candidate = parsed.get(n)
            if candidate is None:
                continue
            original = turns[turn_idx].text
            if candidate.strip() == original.strip():
                continue
            ok, why = acceptable(original, candidate, settings)
            # A glossary term and a screen term fail the same way -- appended
            # rather than substituted -- so they share one guard.
            guard_terms = window_terms + glossary
            if ok and guard_terms:
                ok, why = screen_term_ok(original, candidate, guard_terms)
                if ok and why == "screen-term-applied":
                    stats.screen_terms_applied += 1
                elif not ok:
                    stats.screen_terms_rejected += 1
            if ok:
                turns[turn_idx].text = candidate.strip()
                stats.turns_changed += 1
            else:
                stats.turns_rejected += 1
                stats.rejections.append(
                    {"clock": turns[turn_idx].clock, "reason": why,
                     "original": original[:120], "candidate": candidate[:120]}
                )

    for w, idxs in enumerate(groups, 1):
        run_window(idxs, w)
        if progress:
            progress("llm", w / len(groups))

    # Last line of defence against prompt scaffolding reaching the transcript.
    # The guardrail cannot see it: a two-character prefix barely moves the
    # length ratio or the token overlap, so it sails through as an accepted
    # correction and lands in the exported files.
    leaked = [t for t in turns if LABEL.match(t.text or "")]
    if leaked:
        for t in leaked:
            t.text = _strip_label(t.text)
        stats.labels_stripped = len(leaked)
        log.warning("stripped a leaked prompt label from %d turn(s) — the merge "
                    "prompt's scaffolding was echoed back", len(leaked))

    rate = stats.to_dict()["median_tokens_per_second"]
    log.info(
        "LLM %s done: %d changed, %d rejected by guardrail, %d window(s) failed"
        "%s",
        mode, stats.turns_changed, stats.turns_rejected, stats.failed_windows,
        f", {rate:.0f} tok/s" if rate else "",
    )
    if slow_windows or stats.lines_missing:
        log.warning(
            "LLM was degraded: %d/%d window(s) under %.0f tok/s, %d line(s) never "
            "came back. Output is worse than it should be — restart the llm "
            "service (`docker compose restart llm`) and re-run. A llama.cpp "
            "server left up for hours slows to a crawl and returns short, "
            "truncated replies instead of failing.",
            slow_windows, len(groups), SLOW_TOKENS_PER_SECOND, stats.lines_missing,
        )
    return stats
