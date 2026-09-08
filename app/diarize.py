"""Speaker diarization (pyannote community-1) and turn construction.

Model choice, for the record. NVIDIA Sortformer was evaluated and rejected on
three independent grounds, any one of which is disqualifying here:
  * it detects at most 4 speakers and degrades beyond that — a council meeting
    breaks this immediately;
  * it is CC-BY-NC-4.0, i.e. non-commercial, which rules it out for a product;
  * it is offline-only with a practical ceiling near 12 minutes of audio.
pyannote community-1 is CC-BY-4.0, has no speaker cap, and handles long form.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.audio import format_clock
from app.config import free_gpu

log = logging.getLogger("transcribe.diarize")


class DiarizationError(RuntimeError):
    pass


@dataclass
class Turn:
    start: float
    end: float
    speaker: str
    text: str
    language: str = ""
    words: list[dict] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def clock(self) -> str:
        """HH:MM:SS of the turn start — used in exports and LLM rejection logs."""
        return format_clock(self.start)

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "clock": self.clock,
            "speaker": self.speaker,
            "language": self.language,
            "text": self.text,
            "words": self.words,
        }


def diarize(
    audio,
    *,
    hf_token: str,
    model_name: str,
    device: str = "cuda",
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    cache_dir: str | None = None,
    cluster_threshold: float | None = None,
    progress=None,
):
    """Run pyannote and return `(diarize_df, embeddings)`.

    `embeddings` may be None if the installed pyannote wrapper does not return
    them; downstream code treats speaker naming as optional in that case.
    """
    if not hf_token:
        raise DiarizationError(
            "HF_TOKEN is empty. Diarization needs a HuggingFace token whose "
            "account has accepted the terms at\n"
            f"  https://huggingface.co/{model_name}\n"
            "  https://huggingface.co/pyannote/segmentation-3.0"
        )

    from whisperx.diarize import DiarizationPipeline

    log.info("loading diarizer %s on %s", model_name, device)
    try:
        pipeline = DiarizationPipeline(
            model_name=model_name, token=hf_token, device=device, cache_dir=cache_dir
        )
    except TypeError:
        # Older signature used use_auth_token=
        pipeline = DiarizationPipeline(
            model_name=model_name, use_auth_token=hf_token, device=device
        )
    except Exception as exc:
        raise DiarizationError(_access_hint(exc, model_name)) from exc

    _set_cluster_threshold(pipeline, cluster_threshold)

    kwargs: dict = {"return_embeddings": True}
    if min_speakers:
        kwargs["min_speakers"] = min_speakers
    if max_speakers:
        kwargs["max_speakers"] = max_speakers
    if progress:
        kwargs["progress_callback"] = lambda *a, **k: progress("diarize", _frac(a, k))

    log.info(
        "diarizing (min=%s max=%s) — this is usually the slowest stage",
        min_speakers or "auto", max_speakers or "auto",
    )
    try:
        result = pipeline(audio, **kwargs)
    except TypeError:
        kwargs.pop("progress_callback", None)
        kwargs.pop("return_embeddings", None)
        result = pipeline(audio, **kwargs)
    except Exception as exc:
        raise DiarizationError(_access_hint(exc, model_name)) from exc

    embeddings = None
    if isinstance(result, tuple):
        diarize_df, embeddings = result[0], result[1]
    else:
        diarize_df = result

    del pipeline
    free_gpu()

    n = diarize_df["speaker"].nunique() if hasattr(diarize_df, "speaker") else 0
    log.info("diarization found %d speaker(s) across %d turns", n, len(diarize_df))
    return diarize_df, embeddings


def _set_cluster_threshold(wrapper, threshold: float | None) -> None:
    """Raise pyannote's VBx clustering threshold so it splits speakers less.

    community-1 ships `clustering.threshold = 0.6`, which on real calls splits
    one person into several clusters whenever their acoustics shift. Measured
    on a 59-minute two-person Teams call:

        0.60 -> 5 speakers (over-segmented)
        0.75 -> 3 speakers, 44% / 44% / 12%
        0.85 -> 3 speakers, 44% / 44% / 12%   <- stable plateau
        0.95 -> 2 speakers, 83% / 17%         <- fuses the two real speakers

    0.80 sits in the middle of the stable band: it stops the spurious splitting
    without reaching the point where genuinely different voices collapse
    together. Fixing it here is better than merging afterwards, because the
    clustering also drives segmentation.
    """
    if not threshold:
        return
    inner = getattr(wrapper, "model", wrapper)
    try:
        params = inner.parameters(instantiated=True) or {}
        clustering = params.get("clustering")
        if not isinstance(clustering, dict) or "threshold" not in clustering:
            log.debug("diarizer exposes no clustering.threshold; leaving defaults")
            return
        current = clustering["threshold"]
        if abs(current - threshold) < 1e-6:
            return
        params = {**params, "clustering": {**clustering, "threshold": threshold}}
        inner.instantiate(params)
        log.info("clustering threshold %.2f -> %.2f", current, threshold)
    except Exception as exc:
        log.warning("could not set clustering threshold (%s); using defaults", exc)


def _frac(args, kwargs) -> float:
    for value in list(args) + list(kwargs.values()):
        if isinstance(value, (int, float)) and 0 <= value <= 1:
            return float(value)
    return 0.0


def _access_hint(exc: Exception, model_name: str) -> str:
    text = str(exc)
    if any(k in text.lower() for k in ("401", "403", "gated", "authoriz", "access")):
        return (
            f"Cannot access {model_name}. The token is valid only after the "
            "account accepts the model terms:\n"
            f"  https://huggingface.co/{model_name}\n"
            "  https://huggingface.co/pyannote/segmentation-3.0\n"
            f"Original error: {text}"
        )
    return f"Diarization failed: {text}"


def _cosine(a, b) -> float:
    import numpy as np

    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def merge_similar_speakers(diarize_df, embeddings, threshold: float = 0.60):
    """Fold together clusters whose voices are too alike to be different people.

    pyannote splits one speaker into several clusters when their acoustics shift
    mid-call — a headset swap, a device change, or VoIP bitrate adaptation is
    enough. Measured on a real two-person Teams call it reported **five**
    speakers, where three of them were mutually 0.75-0.80 cosine similar (one
    person) while the two genuinely different voices sat at 0.42.

    Single-link agglomeration at `threshold` separates those cases cleanly.
    Deliberately conservative: merging two real speakers is far worse than
    leaving one split, so the default sits well above the 0.42 observed between
    distinct voices.
    """
    import numpy as np

    if embeddings is None:
        return diarize_df, {}

    vecs: dict[str, np.ndarray] = {}
    if isinstance(embeddings, dict):
        vecs = {str(k): np.asarray(v, float).ravel() for k, v in embeddings.items()}
    else:
        arr = np.asarray(embeddings, float)
        if arr.ndim == 2:
            labels = sorted(diarize_df["speaker"].unique())
            vecs = {labels[i]: arr[i] for i in range(min(len(labels), arr.shape[0]))}
    if len(vecs) < 2:
        return diarize_df, {}

    labels = sorted(vecs)
    parent = {l: l for l in labels}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    merged_pairs = []
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            sim = _cosine(vecs[a], vecs[b])
            if sim >= threshold:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
                    merged_pairs.append((a, b, sim))

    mapping = {l: find(l) for l in labels}
    if not merged_pairs:
        return diarize_df, {}

    # Keep the label that talks most in each group, so the dominant speaker
    # keeps a stable name.
    talk = diarize_df.assign(_d=diarize_df["end"] - diarize_df["start"]) \
                     .groupby("speaker")["_d"].sum().to_dict()
    groups: dict[str, list[str]] = {}
    for src, dst in mapping.items():
        groups.setdefault(dst, []).append(src)
    final = {}
    for members in groups.values():
        keep = max(members, key=lambda m: talk.get(m, 0.0))
        for m in members:
            final[m] = keep

    for a, b, sim in merged_pairs:
        log.info("merging speakers %s + %s (cosine %.3f — same voice)", a, b, sim)
    changed = {k: v for k, v in final.items() if k != v}
    log.info(
        "speaker clusters: %d -> %d after merging",
        len(labels), len(set(final.values())),
    )
    diarize_df = diarize_df.copy()
    diarize_df["speaker"] = diarize_df["speaker"].map(lambda s: final.get(s, s))
    return diarize_df, changed


def collapse_turns_to_target(turns: list["Turn"], target: int) -> dict[str, str]:
    """Reduce speakers to exactly `target`, using conversational turn-taking.

    Must run on built *turns*, not raw diarization segments. pyannote emits
    ~1,169 short segments for a 59-minute call and they alternate constantly
    for reasons unrelated to who is talking; measured on that data the metric
    picked a pair with voice similarity -0.08. On the 255 merged turns the same
    metric is unambiguous:

        01 <-> 02   76 alternations   rate 0.95   (two people talking)
        02 <-> 00   66 alternations   rate 0.97   (two people talking)
        00 <-> 01   30 alternations   rate 0.44   -> same person, split in two

    Telling pyannote `max_speakers=N` instead is worse: the constraint also
    drives segmentation, and asking for 2 on this call fused the two *actual*
    speakers into one 83% cluster while keeping an outlier as the second.
    """
    labels = sorted({t.speaker for t in turns})
    if target < 1 or len(labels) <= target:
        return {}

    mapping = {l: l for l in labels}
    while len(set(mapping.values())) > target:
        groups = sorted(set(mapping.values()))
        counts = {g: sum(1 for t in turns if mapping[t.speaker] == g) for g in groups}
        alt: dict[tuple[str, str], int] = {}
        for a, b in zip(turns, turns[1:]):
            ga, gb = mapping[a.speaker], mapping[b.speaker]
            if ga != gb:
                key = tuple(sorted((ga, gb)))
                alt[key] = alt.get(key, 0) + 1

        best = None
        for i, ga in enumerate(groups):
            for gb in groups[i + 1:]:
                n = alt.get(tuple(sorted((ga, gb))), 0)
                rate = n / max(1, min(counts[ga], counts[gb]))
                if best is None or rate < best[0]:
                    best = (rate, ga, gb, n)

        rate, ga, gb, n = best
        keep, drop = (ga, gb) if counts[ga] >= counts[gb] else (gb, ga)
        for k, v in list(mapping.items()):
            if v == drop:
                mapping[k] = keep
        log.info(
            "merging %s into %s — only %d alternations (rate %.2f): same person",
            drop, keep, n, rate,
        )

    changed = {k: v for k, v in mapping.items() if k != v}
    if changed:
        log.info("collapsed %d speakers to %d", len(labels), target)
    return mapping


def collapse_to_target(diarize_df, embeddings, target: int):
    """Reduce clusters to exactly `target` speakers using turn-taking.

    Telling pyannote `max_speakers=N` is worse than doing this afterwards: the
    constraint also drives its segmentation, and on a real 59-minute two-person
    call asking for 2 fused the two *actual* speakers into one 83% cluster
    while keeping an acoustic outlier as the second.

    The signal that works is conversational structure. Two clusters that are
    the same person barely alternate with each other — they only swap when the
    diarizer flips mid-speech — whereas two people in conversation alternate
    constantly. On that same call:

        01 <-> 02   76 transitions   (two people talking)
        02 <-> 00   66 transitions   (two people talking)
        00 <-> 01   30 transitions   -> same person, split in two

    Embedding similarity alone picked the wrong pair here, so alternation rate
    leads and voice similarity only breaks ties.
    """
    import numpy as np

    labels = sorted(diarize_df["speaker"].unique())
    if target < 1 or len(labels) <= target:
        return diarize_df, {}

    rows = diarize_df.sort_values("start")[["speaker"]].to_numpy().ravel()
    counts = {l: int((rows == l).sum()) for l in labels}

    vecs: dict[str, np.ndarray] = {}
    if isinstance(embeddings, dict):
        vecs = {str(k): np.asarray(v, float).ravel() for k, v in embeddings.items()}
    elif embeddings is not None:
        arr = np.asarray(embeddings, float)
        if arr.ndim == 2:
            vecs = {labels[i]: arr[i] for i in range(min(len(labels), arr.shape[0]))}

    groups = {l: {l} for l in labels}
    mapping = {l: l for l in labels}
    merges = []

    while len(groups) > target:
        # alternation counts between current groups
        trans: dict[tuple[str, str], int] = {}
        for a, b in zip(rows, rows[1:]):
            ga, gb = mapping[a], mapping[b]
            if ga != gb:
                key = tuple(sorted((ga, gb)))
                trans[key] = trans.get(key, 0) + 1

        best = None
        for i, ga in enumerate(sorted(groups)):
            for gb in sorted(groups)[i + 1:]:
                n = trans.get(tuple(sorted((ga, gb))), 0)
                size = min(sum(counts[m] for m in groups[ga]),
                           sum(counts[m] for m in groups[gb])) or 1
                rate = n / size          # low  -> likely the same person
                sim = max((_cosine(vecs[x], vecs[y])
                           for x in groups[ga] for y in groups[gb]
                           if x in vecs and y in vecs), default=0.0)
                score = (rate, -sim)     # fewest alternations, then most alike
                if best is None or score < best[0]:
                    best = (score, ga, gb, n, rate, sim)

        _, ga, gb, n, rate, sim = best
        keep, drop = sorted((ga, gb))
        groups[keep] |= groups.pop(drop)
        for m in list(mapping):
            if mapping[m] == drop:
                mapping[m] = keep
        merges.append((drop, keep, n, rate, sim))
        log.info(
            "collapsing %s into %s (%d alternations, rate %.2f, voice %.2f)",
            drop, keep, n, rate, sim,
        )

    diarize_df = diarize_df.copy()
    diarize_df["speaker"] = diarize_df["speaker"].map(lambda s: mapping.get(s, s))
    log.info("collapsed %d clusters to %d speaker(s)", len(labels), target)
    return diarize_df, {d: k for d, k, *_ in merges}


def assign_speakers(segments: list[dict], diarize_df, embeddings=None) -> list[dict]:
    """Attach a speaker to every word and segment (longest temporal overlap)."""
    import whisperx

    payload = {"segments": segments}
    try:
        result = whisperx.assign_word_speakers(
            diarize_df, payload, speaker_embeddings=embeddings, fill_nearest=True
        )
    except TypeError:
        result = whisperx.assign_word_speakers(diarize_df, payload, fill_nearest=True)
    return result.get("segments", segments)


SENTENCE_END = (".", "!", "?", "…")


def _flat_words(segments: list[dict]) -> list[dict]:
    return [w for seg in segments for w in (seg.get("words") or []) if w.get("word")]


def snap_boundaries_to_sentences(segments: list[dict], window: int = 3) -> int:
    """Move each speaker change onto the nearest sentence end.

    Diarization puts a boundary where the *acoustics* change, which is often a
    word or two away from where the sentence actually ends. Measured on a real
    two-person Teams call, this cut questions in half:

        SPEAKER_00: "A tohle zelené znamená"
        SPEAKER_01: "co? To je jenom filtr."

    One question, two speakers. Snapping the change to just after "co?" keeps
    the sentence whole and attributes it to whoever asked it.

    Conservative by construction: it only *moves* an existing change, never
    adds or removes one, and only within `window` words. Returns how many
    boundaries were moved.
    """
    words = _flat_words(segments)
    n = len(words)
    if n < 3:
        return 0

    moved = 0
    i = 1
    while i < n:
        prev_spk, spk = words[i - 1].get("speaker"), words[i].get("speaker")
        if spk == prev_spk:
            i += 1
            continue

        # Candidate positions: index j means "change happens before words[j]",
        # which is valid when words[j-1] ends a sentence.
        best = None
        for j in range(max(1, i - window), min(n, i + window + 1)):
            if words[j - 1]["word"].strip().endswith(SENTENCE_END):
                dist = abs(j - i)
                if dist and (best is None or dist < best[0]):
                    best = (dist, j)

        if best:
            j = best[1]
            if j < i:  # pull the change earlier
                for k in range(j, i):
                    words[k]["speaker"] = spk
            else:      # push the change later
                for k in range(i, j):
                    words[k]["speaker"] = prev_spk
            moved += 1
            i = max(i, j)
        i += 1

    if moved:
        log.info("snapped %d speaker boundary/-ies onto sentence ends", moved)
    return moved


def absorb_short_runs(segments: list[dict], min_seconds: float = 0.7) -> int:
    """Fold away speaker runs too short to be a real turn.

    A 0.06-second run containing the single letter "S" is an artifact of
    word-level attribution, not somebody taking the floor. Runs below
    `min_seconds` are given to whichever neighbour is longer.
    """
    words = _flat_words(segments)
    n = len(words)
    if n < 3:
        return 0

    fixed = 0
    i = 0
    while i < n:
        j = i
        spk = words[i].get("speaker")
        while j < n and words[j].get("speaker") == spk:
            j += 1
        start = words[i].get("start")
        end = words[j - 1].get("end")
        if start is not None and end is not None and (end - start) < min_seconds:
            before = words[i - 1].get("speaker") if i > 0 else None
            after = words[j].get("speaker") if j < n else None
            target = before or after
            if before and after and before != after:
                # give it to the side that owns more surrounding speech
                target = before if (i - 1) >= (n - j) else after
            if target and target != spk:
                for k in range(i, j):
                    words[k]["speaker"] = target
                fixed += j - i
        i = j

    if fixed:
        log.info("absorbed %d word(s) in sub-%.1fs speaker runs", fixed, min_seconds)
    return fixed


def smooth_word_speakers(segments: list[dict], min_run: int = 3) -> int:
    """Absorb brief speaker flips that are flanked by the same speaker.

    Diarization assigns each word to whoever overlaps it most. During
    crosstalk that flips for a word or two and then flips back, which shreds
    one sentence across three speakers: a three-word clause comes back as
    SPEAKER_01 / SPEAKER_00 / SPEAKER_01, one label per fragment.

    Only runs shorter than `min_run` words *with the same speaker on both
    sides* are absorbed, so genuine handovers and interjections that actually
    change the floor are preserved. Mutates the word dicts in place and
    returns how many words were reassigned.
    """
    words = [w for seg in segments for w in (seg.get("words") or []) if w.get("word")]
    if len(words) < 3:
        return 0

    fixed = 0
    i, n = 0, len(words)
    while i < n:
        j = i
        speaker = words[i].get("speaker")
        while j < n and words[j].get("speaker") == speaker:
            j += 1
        if (j - i) < min_run and i > 0 and j < n:
            before, after = words[i - 1].get("speaker"), words[j].get("speaker")
            if before is not None and before == after and before != speaker:
                for k in range(i, j):
                    words[k]["speaker"] = before
                fixed += j - i
        i = j

    if fixed:
        log.info("smoothed %d word(s) out of spurious speaker flips", fixed)
    return fixed


def build_turns(
    segments: list[dict],
    merge_gap: float = 0.8,
    smooth: int = 3,
    min_turn_seconds: float = 0.7,
    snap_window: int = 3,
) -> list[Turn]:
    """Collapse segments into speaker turns, splitting where the speaker changes.

    Word-level speakers are preferred: a single ASR segment often spans a
    handover ("...souhlasím. — Dobře, děkuji."), and splitting on words keeps
    the two speakers apart instead of crediting both lines to one of them.
    """
    # Order matters: kill spurious flips first, then drop runs too short to be
    # a turn, then align whatever changes remain to sentence boundaries.
    if smooth > 1:
        smooth_word_speakers(segments, min_run=smooth)
    if min_turn_seconds > 0:
        absorb_short_runs(segments, min_seconds=min_turn_seconds)
    if snap_window > 0:
        snap_boundaries_to_sentences(segments, window=snap_window)

    turns: list[Turn] = []

    def push(start, end, speaker, text, language, words):
        text = " ".join(text.split()).strip()
        if not text:
            return
        if (
            turns
            and turns[-1].speaker == speaker
            and start - turns[-1].end <= merge_gap
        ):
            prev = turns[-1]
            prev.end = max(prev.end, end)
            prev.text = f"{prev.text} {text}".strip()
            prev.words.extend(words)
            return
        turns.append(
            Turn(start=start, end=end, speaker=speaker, text=text, language=language, words=list(words))
        )

    for seg in segments:
        seg_speaker = seg.get("speaker") or "UNKNOWN"
        language = seg.get("language", "")
        words = [w for w in (seg.get("words") or []) if w.get("word")]

        if not words:
            push(
                float(seg.get("start", 0.0)),
                float(seg.get("end", 0.0)),
                seg_speaker,
                seg.get("text", ""),
                language,
                [],
            )
            continue

        cur_speaker = words[0].get("speaker") or seg_speaker
        buf: list[dict] = []
        for word in words:
            spk = word.get("speaker") or seg_speaker
            if spk != cur_speaker and buf:
                push(
                    _wstart(buf, seg), _wend(buf, seg), cur_speaker,
                    "".join(w["word"] if w["word"].startswith(" ") else f" {w['word']}" for w in buf),
                    language, buf,
                )
                buf = []
                cur_speaker = spk
            buf.append(word)
        if buf:
            push(
                _wstart(buf, seg), _wend(buf, seg), cur_speaker,
                "".join(w["word"] if w["word"].startswith(" ") else f" {w['word']}" for w in buf),
                language, buf,
            )

    turns.sort(key=lambda t: t.start)
    log.info("built %d speaker turns", len(turns))
    return turns


def _wstart(words: list[dict], seg: dict) -> float:
    vals = [w["start"] for w in words if w.get("start") is not None]
    return float(min(vals)) if vals else float(seg.get("start", 0.0))


def _wend(words: list[dict], seg: dict) -> float:
    vals = [w["end"] for w in words if w.get("end") is not None]
    return float(max(vals)) if vals else float(seg.get("end", 0.0))


def speaker_stats(turns: list[Turn]) -> list[dict]:
    """Talk time, share and turn count per speaker — drives the DOCX table."""
    agg: dict[str, dict] = {}
    for turn in turns:
        row = agg.setdefault(turn.speaker, {"speaker": turn.speaker, "seconds": 0.0, "turns": 0, "words": 0})
        row["seconds"] += turn.duration
        row["turns"] += 1
        row["words"] += len(turn.text.split())

    total = sum(r["seconds"] for r in agg.values()) or 1.0
    rows = sorted(agg.values(), key=lambda r: -r["seconds"])
    for row in rows:
        row["share"] = row["seconds"] / total
        row["clock"] = format_clock(row["seconds"])
        row["seconds"] = round(row["seconds"], 2)
    return rows
