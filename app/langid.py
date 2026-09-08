"""Language identification across the whole recording, not just its first 30 s.

The single most common Whisper failure on long multilingual audio is that the
language is decided from the opening 30 seconds — which in a municipal meeting
is often a chairperson's greeting, or silence. We instead sample speech-dense
windows spread across the entire file, classify each, smooth the sequence over
time, and emit contiguous *language runs* that later stages transcribe and
align independently.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np

from app.audio import SAMPLE_RATE

log = logging.getLogger("transcribe.langid")

WINDOW_S = 30.0
HOP_S = 5.0
MIN_RUN_S = 20.0

# EU official languages. Irish (ga) is deliberately absent: Whisper has no
# Irish language token, so it cannot be detected or transcribed by this model.
EU_LANGUAGES = {
    "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu",
    "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "sk", "sl", "sv",
}

# Language pairs Whisper routinely confuses. When two members of the same
# group both score highly we keep them as separate runs rather than forcing a
# winner, because each has its own forced-alignment model and mislabelling one
# as the other costs timestamp accuracy.
CONFUSABLE_GROUPS: list[set[str]] = [
    {"cs", "sk"},
    {"hr", "sr", "bs", "sl"},
    {"da", "no", "nn", "sv"},
    {"ru", "uk", "be"},
    {"es", "gl", "pt"},
    {"id", "ms"},
]


@dataclass
class LanguageRun:
    start: float
    end: float
    language: str
    confidence: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return {**asdict(self), "duration": round(self.duration, 3)}


@dataclass
class LanguageReport:
    mode: str
    runs: list[LanguageRun]
    shares: dict[str, float] = field(default_factory=dict)
    samples: list[dict] = field(default_factory=list)
    dominant: str = ""
    code_switching: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "dominant": self.dominant,
            "code_switching": self.code_switching,
            "shares": {k: round(v, 4) for k, v in self.shares.items()},
            "runs": [r.to_dict() for r in self.runs],
            "samples": self.samples,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Window selection
# ---------------------------------------------------------------------------

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12)) if x.size else 0.0


def select_windows(audio: np.ndarray, n: int) -> list[float]:
    """Pick `n` speech-dense 30 s window start times spread across the file.

    Deterministic and model-free: the file is split into `n` equal buckets and
    within each bucket we take the loudest 30 s window. Meeting recordings are
    overwhelmingly speech, so energy is a good enough proxy for a VAD here and
    avoids coupling to a specific VAD implementation.
    """
    total_s = len(audio) / SAMPLE_RATE
    if total_s <= WINDOW_S:
        return [0.0]

    n = max(1, min(n, int(total_s // WINDOW_S)))
    bucket = total_s / n
    win = int(WINDOW_S * SAMPLE_RATE)
    hop = int(HOP_S * SAMPLE_RATE)

    # Global noise floor: windows quieter than this are silence, not speech.
    coarse = [
        _rms(audio[i : i + win])
        for i in range(0, max(1, len(audio) - win), max(hop, win // 2))
    ]
    floor = float(np.percentile(coarse, 20)) if coarse else 0.0

    starts: list[float] = []
    for b in range(n):
        lo = int(b * bucket * SAMPLE_RATE)
        hi = min(len(audio) - win, int(((b + 1) * bucket) * SAMPLE_RATE))
        if hi <= lo:
            continue
        best_pos, best_rms = lo, -1.0
        for pos in range(lo, hi + 1, hop):
            r = _rms(audio[pos : pos + win])
            if r > best_rms:
                best_pos, best_rms = pos, r
        if best_rms >= floor:
            starts.append(best_pos / SAMPLE_RATE)
    return starts or [0.0]


# ---------------------------------------------------------------------------
# Per-window classification
# ---------------------------------------------------------------------------

def _probs_for_chunk(pipeline, chunk: np.ndarray) -> list[tuple[str, float]]:
    """Language posteriors for one <=30 s chunk.

    Mirrors WhisperX's own detect_language path (mel -> encoder ->
    CTranslate2 detect_language) so it stays correct across whisperx versions,
    with a fallback to the public single-language API.
    """
    from whisperx.audio import N_SAMPLES, log_mel_spectrogram

    fw = pipeline.model  # faster_whisper.WhisperModel
    try:
        n_mels = fw.feat_kwargs.get("feature_size", 80)
        pad = max(0, N_SAMPLES - chunk.shape[0])
        mel = log_mel_spectrogram(chunk[:N_SAMPLES], n_mels=n_mels, padding=pad)
        encoder_output = fw.encode(mel)
        results = fw.model.detect_language(encoder_output)
        # e.g. [[('<|cs|>', 0.87), ('<|sk|>', 0.09), ...]]
        return [(tok.strip("<|>"), float(p)) for tok, p in results[0][:5]]
    except Exception as exc:  # pragma: no cover - depends on whisperx internals
        log.debug("low-level LID failed (%s); falling back", exc)
        try:
            lang = pipeline.detect_language(chunk)
            return [(str(lang), 1.0)]
        except Exception as exc2:
            log.warning("language detection failed for a window: %s", exc2)
            return []


# ---------------------------------------------------------------------------
# Smoothing and run construction
# ---------------------------------------------------------------------------

def _smooth(labels: list[str], k: int = 3) -> list[str]:
    """Majority filter over a sliding window; kills isolated misdetections."""
    if len(labels) < 3:
        return labels
    half = k // 2
    out: list[str] = []
    for i in range(len(labels)):
        lo, hi = max(0, i - half), min(len(labels), i + half + 1)
        out.append(Counter(labels[lo:hi]).most_common(1)[0][0])
    return out


def _runs_from_labels(
    centers: list[float], labels: list[str], confs: list[float], total_s: float
) -> list[LanguageRun]:
    """Turn per-window labels into contiguous timeline runs."""
    if not centers:
        return []

    runs: list[LanguageRun] = []
    start = 0.0
    for i, label in enumerate(labels):
        last = i == len(labels) - 1
        if last or labels[i + 1] != label:
            end = total_s if last else (centers[i] + centers[i + 1]) / 2.0
            same = [confs[j] for j in range(len(labels)) if labels[j] == label]
            if runs and runs[-1].language == label:
                runs[-1].end = end
            else:
                runs.append(
                    LanguageRun(start, end, label, float(np.mean(same)) if same else 0.0)
                )
            start = end

    # Absorb slivers into whichever neighbour is longer.
    merged: list[LanguageRun] = []
    for run in runs:
        if merged and run.duration < MIN_RUN_S:
            merged[-1].end = run.end
        else:
            merged.append(run)
    if len(merged) > 1 and merged[0].duration < MIN_RUN_S:
        merged[1].start = merged[0].start
        merged.pop(0)
    return merged


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def identify(
    pipeline,
    audio: np.ndarray,
    *,
    mode: str = "auto",
    forced_language: str = "",
    dominance: float = 0.85,
    n_samples: int = 30,
) -> LanguageReport:
    total_s = len(audio) / SAMPLE_RATE

    if mode == "single":
        if not forced_language:
            raise ValueError("lang_mode=single requires --language / LANGUAGE")
        log.info("language: forced to %r for the whole file", forced_language)
        return LanguageReport(
            mode="single",
            dominant=forced_language,
            shares={forced_language: 1.0},
            runs=[LanguageRun(0.0, total_s, forced_language, 1.0)],
            notes=["forced by configuration"],
        )

    starts = select_windows(audio, n_samples)
    log.info("language ID: probing %d windows across %.1f min", len(starts), total_s / 60)

    centers, labels, confs, samples = [], [], [], []
    weights: dict[str, float] = defaultdict(float)

    for st in starts:
        chunk = audio[int(st * SAMPLE_RATE) : int((st + WINDOW_S) * SAMPLE_RATE)]
        if chunk.size < SAMPLE_RATE:
            continue
        probs = _probs_for_chunk(pipeline, chunk)
        if not probs:
            continue
        lang, p = probs[0]
        centers.append(st + WINDOW_S / 2)
        labels.append(lang)
        confs.append(p)
        weights[lang] += p
        samples.append(
            {
                "start": round(st, 2),
                "language": lang,
                "confidence": round(p, 4),
                "runner_up": (
                    {"language": probs[1][0], "confidence": round(probs[1][1], 4)}
                    if len(probs) > 1
                    else None
                ),
            }
        )

    if not labels:
        log.warning("language ID produced no results; defaulting to English")
        return LanguageReport(
            mode=mode,
            dominant="en",
            shares={"en": 1.0},
            runs=[LanguageRun(0.0, total_s, "en", 0.0)],
            notes=["detection failed, defaulted to en"],
        )

    total_w = sum(weights.values()) or 1.0
    shares = {k: v / total_w for k, v in sorted(weights.items(), key=lambda kv: -kv[1])}
    dominant = next(iter(shares))
    notes: list[str] = []

    detected = {lang for lang, share in shares.items() if share >= 0.05}
    for group in CONFUSABLE_GROUPS:
        overlap = detected & group
        if len(overlap) > 1:
            notes.append(
                "Whisper commonly confuses "
                + "/".join(sorted(overlap))
                + "; each is transcribed and aligned with its own model"
            )

    log.info(
        "language shares: %s",
        ", ".join(f"{k} {v:.0%}" for k, v in list(shares.items())[:5]),
    )

    if mode == "auto" and shares[dominant] >= dominance:
        log.info("language: %r covers %.0f%% — single-language run", dominant, shares[dominant] * 100)
        return LanguageReport(
            mode="auto",
            dominant=dominant,
            shares=shares,
            samples=samples,
            runs=[LanguageRun(0.0, total_s, dominant, shares[dominant])],
            notes=notes,
        )

    runs = _runs_from_labels(centers, _smooth(labels), confs, total_s)
    if len(runs) > 1:
        notes.append(f"code-switching: {len(runs)} language runs")
        log.info(
            "code-switching detected — %d runs: %s",
            len(runs),
            ", ".join(f"{r.language} {r.duration/60:.1f}m" for r in runs),
        )
    return LanguageReport(
        mode=mode,
        dominant=dominant,
        shares=shares,
        samples=samples,
        runs=runs,
        code_switching=len(runs) > 1,
        notes=notes,
    )
