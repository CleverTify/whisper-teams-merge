"""Forced alignment — word-level timestamps per language.

Runs *after* the LLM pass, not before. LLM-corrected text no longer matches the
original word timings, so aligning last is what keeps the player's word
highlighting honest.

Resolution order per language: WhisperX's own torch/HF aligner, then a
gap-fill for the EU languages it omits, then the MMS universal aligner, then
nothing (segment-level timings only). Every failure falls through instead of
raising, so a withdrawn model degrades quality rather than killing a run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger("transcribe.align")

# EU official languages WhisperX ships no aligner for.
EU_GAP_FILL = {
    "bg": "anton-l/wav2vec2-large-xlsr-53-bulgarian",
    "et": "anton-l/wav2vec2-large-xlsr-53-estonian",
    "lt": "DeividasM/wav2vec2-large-xlsr-53-lithuanian",
    "mt": "carlosdanielhernandezmena/wav2vec2-large-xlsr-53-maltese-64h",
    "sv": "KBLab/wav2vec2-large-voxrex-swedish",
}
MMS_ALIGNER = "facebook/mms-300m-1130-forced-aligner"


@dataclass(frozen=True)
class Aligner:
    language: str
    model: str
    tier: str


def _upstream() -> tuple[dict, dict]:
    try:
        from whisperx.alignment import (
            DEFAULT_ALIGN_MODELS_HF,
            DEFAULT_ALIGN_MODELS_TORCH,
        )

        return DEFAULT_ALIGN_MODELS_TORCH, DEFAULT_ALIGN_MODELS_HF
    except Exception:
        return {}, {}


def candidates(language: str) -> list[tuple[str | None, str]]:
    """(model_name_or_None_for_upstream_default, tier) best first."""
    lang = (language or "").lower().split("-")[0]
    torch_map, hf_map = _upstream()
    out: list[tuple[str | None, str]] = []
    if lang in torch_map or lang in hf_map:
        out.append((None, "whisperx"))
    if lang in EU_GAP_FILL:
        out.append((EU_GAP_FILL[lang], "eu-gapfill"))
    out.append((MMS_ALIGNER, "mms"))
    return out


class AlignerCache:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._loaded: dict[str, tuple] = {}
        self._used: dict[str, str] = {}
        self._failed: set[str] = set()

    def get(self, language: str):
        lang = (language or "").lower().split("-")[0]
        if lang in self._loaded:
            return self._loaded[lang]

        import whisperx

        for model, tier in candidates(lang):
            key = f"{tier}:{model}"
            if key in self._failed:
                continue
            try:
                kwargs = {"language_code": lang, "device": self.device}
                if model:
                    kwargs["model_name"] = model
                log.info("loading aligner for %r (%s)", lang, tier)
                pair = whisperx.load_align_model(**kwargs)
                self._loaded[lang] = pair
                self._used[lang] = tier
                return pair
            except Exception as exc:
                self._failed.add(key)
                log.warning("aligner %s failed for %r (%s)", tier, lang, type(exc).__name__)

        log.warning("no aligner for %r — keeping segment-level timings", lang)
        self._used[lang] = "none"
        return None

    def used(self) -> dict[str, str]:
        return dict(self._used)

    def release(self) -> None:
        from app.config import free_gpu

        self._loaded.clear()
        free_gpu()


def align_segments(segments: list[dict], audio, device: str, progress=None) -> tuple[list[dict], dict]:
    """Align each language group; pass through whatever cannot be aligned."""
    import whisperx

    cache = AlignerCache(device=device)
    by_lang: dict[str, list[dict]] = {}
    for seg in segments:
        by_lang.setdefault(seg.get("language") or "unknown", []).append(seg)

    out: list[dict] = []
    done, total = 0, len(segments) or 1

    for lang, group in by_lang.items():
        pair = cache.get(lang)
        if pair is None:
            out.extend(group)
        else:
            model, metadata = pair
            try:
                res = whisperx.align(
                    group, model, metadata, audio, device, return_char_alignments=False
                )
                produced = res.get("segments", [])
                # Alignment silently drops segments it cannot fit; losing half
                # the speech is worse than losing word timings.
                if len(produced) < len(group) * 0.5:
                    log.warning(
                        "aligner kept only %d/%d segments for %r — using unaligned",
                        len(produced), len(group), lang,
                    )
                    out.extend(group)
                else:
                    for s in produced:
                        s.setdefault("language", lang)
                    out.extend(produced)
            except Exception as exc:
                log.warning("alignment failed for %r (%s); keeping unaligned", lang, exc)
                out.extend(group)
        done += len(group)
        if progress:
            progress("align", done / total)

    cache.release()
    out.sort(key=lambda s: s.get("start", 0.0))
    with_words = sum(1 for s in out if s.get("words"))
    log.info("alignment: %d/%d segments have word timings", with_words, len(out))
    return out, {"per_language": cache.used(), "segments_with_words": with_words}
