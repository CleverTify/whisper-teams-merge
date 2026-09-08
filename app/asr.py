"""Speech recognition with two interchangeable backends.

`faster-whisper` is the fastest option but CTranslate2 is CUDA-or-CPU only —
it has no ROCm backend, so it cannot drive an AMD GPU. `whisper.cpp` built with
Vulkan reaches AMD and Intel GPUs and falls back to CPU, which is what makes a
single image usable on NVIDIA, AMD, Apple and plain CPU.

Both return the same shape, so the rest of the pipeline never knows which ran:

    [{"start": float, "end": float, "text": str, "language": str}]
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from app.audio import SAMPLE_RATE, slice_audio
from app.config import free_gpu, hardware, resolve_backend, resolve_compute_type

log = logging.getLogger("transcribe.asr")

WHISPER_CPP_BIN = "/usr/local/bin/whisper-cli"

# Passed to faster-whisper.
#
# Most of these do nothing here, and it is worth knowing which. WhisperX drives
# CTranslate2 through its *batched* path, which calls `generate()` with only
# beam_size, patience, length_penalty, suppress_blank, suppress_tokens and
# max_length. There is no temperature-fallback loop, so `temperatures` and its
# three thresholds are never read, and batching already prevents context
# carry-over so `condition_on_previous_text` is moot. Verified against a cached
# run: the segments carry no `no_speech_prob`, `avg_logprob` or `temperature`
# field at all. They are kept only because switching to the sequential path
# would make them live again; nothing here depends on them working.
#
# `initial_prompt` IS honoured. It primes the decoder with vocabulary it would
# otherwise mangle — this is Czech speech about English software, and Whisper
# writes "do krem" for Dockerem and "trůhelníček" for trojúhelníček. `hotwords`
# is not usable: the batched path never forwards it.
ASR_OPTIONS = {
    "condition_on_previous_text": False,
    "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "repetition_penalty": 1.05,
}


class ASRError(RuntimeError):
    pass


def _offset(segments: list[dict], offset: float, language: str) -> list[dict]:
    out = []
    for s in segments:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        out.append(
            {
                "start": float(s.get("start", 0.0)) + offset,
                "end": float(s.get("end", 0.0)) + offset,
                "text": text,
                "language": s.get("language") or language,
            }
        )
    return out


# ---------------------------------------------------------------------------
# faster-whisper (CUDA)
# ---------------------------------------------------------------------------

class FasterWhisper:
    name = "faster-whisper"

    def __init__(self, settings):
        self.settings = settings
        self.hw = hardware()
        self.compute_type = resolve_compute_type(settings.compute_type, self.hw)
        self.batch_size = settings.batch_size
        self._pipe = None

    def pipeline(self):
        if self._pipe is not None:
            return self._pipe
        import whisperx

        opts = dict(ASR_OPTIONS, beam_size=self.settings.beam_size)
        if self.settings.initial_prompt:
            opts["initial_prompt"] = self.settings.initial_prompt
            log.info("priming the decoder with %d chars of domain vocabulary",
                     len(self.settings.initial_prompt))
        log.info(
            "loading Whisper %s (%s) on %s",
            self.settings.whisper_model, self.compute_type, self.hw.device,
        )
        self._pipe = whisperx.load_model(
            self.settings.whisper_model,
            device=self.hw.device,
            compute_type=self.compute_type,
            asr_options=opts,
            vad_options={
                "vad_onset": self.settings.vad_onset,
                "vad_offset": self.settings.vad_offset,
                "chunk_size": self.settings.vad_chunk_size,
            },
            threads=4,
        )
        return self._pipe

    def transcribe(self, audio: np.ndarray, language: str) -> list[dict]:
        pipe = self.pipeline()
        batch = self.batch_size
        while True:
            try:
                return pipe.transcribe(audio, batch_size=batch, language=language).get(
                    "segments", []
                )
            except Exception as exc:
                if not _is_oom(exc):
                    raise
                if batch <= 1:
                    raise ASRError(f"out of VRAM at batch_size=1: {exc}") from exc
                batch = max(1, batch // 2)
                self.batch_size = batch
                log.warning("CUDA OOM — retrying at batch_size=%d", batch)
                free_gpu()

    def release(self) -> None:
        if self._pipe is not None:
            try:
                del self._pipe.model
            except Exception:
                pass
            self._pipe = None
        free_gpu()

    def metadata(self) -> dict:
        return {
            "backend": self.name,
            "model": self.settings.whisper_model,
            "compute_type": self.compute_type,
            "device": self.hw.device,
        }


# ---------------------------------------------------------------------------
# whisper.cpp (Vulkan / CPU) — everything that is not NVIDIA
# ---------------------------------------------------------------------------

class WhisperCpp:
    name = "whisper.cpp"

    def __init__(self, settings):
        self.settings = settings
        self.model_path = Path("/cache/whispercpp") / settings.whisper_cpp_model

    def _ensure_model(self) -> Path:
        if self.model_path.exists():
            return self.model_path
        raise ASRError(
            f"whisper.cpp model missing: {self.model_path}\n"
            "Run `docker compose run --rm app warmup` to download it."
        )

    def transcribe(self, audio: np.ndarray, language: str) -> list[dict]:
        model = self._ensure_model()
        with tempfile.TemporaryDirectory(prefix="wcpp-") as tmp:
            wav = Path(tmp) / "chunk.wav"
            sf.write(str(wav), audio, SAMPLE_RATE, subtype="PCM_16")
            out = Path(tmp) / "out"
            cmd = [
                WHISPER_CPP_BIN,
                "-m", str(model),
                "-f", str(wav),
                "-l", language or "auto",
                "-oj", "-of", str(out),
                "-bs", str(self.settings.beam_size),
                "--no-prints",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-10:]
                raise ASRError("whisper.cpp failed:\n" + "\n".join(tail))
            data = json.loads((out.with_suffix(".json")).read_text(encoding="utf-8"))

        segments = []
        for s in data.get("transcription", []):
            offsets = s.get("offsets", {})
            segments.append(
                {
                    # whisper.cpp reports milliseconds
                    "start": float(offsets.get("from", 0)) / 1000.0,
                    "end": float(offsets.get("to", 0)) / 1000.0,
                    "text": (s.get("text") or "").strip(),
                }
            )
        return segments

    def release(self) -> None:
        pass

    def metadata(self) -> dict:
        return {
            "backend": self.name,
            "model": self.settings.whisper_cpp_model,
            "compute_type": "ggml",
            "device": "vulkan/cpu",
        }


# ---------------------------------------------------------------------------

def build(settings):
    which = resolve_backend(settings.backend)
    engine = FasterWhisper(settings) if which == "faster-whisper" else WhisperCpp(settings)
    log.info("ASR backend: %s", engine.name)
    return engine


def transcribe_runs(engine, audio: np.ndarray, runs, *, progress=None) -> list[dict]:
    """Transcribe each language run and return segments on the global timeline.

    Runs come from language identification, so a Czech/Slovak meeting is
    decoded with the right language for each stretch instead of one guess for
    the whole file.
    """
    total = sum(max(0.0, r.end - r.start) for r in runs) or 1.0
    done = 0.0
    segments: list[dict] = []

    for i, run in enumerate(runs, 1):
        chunk = slice_audio(audio, run.start, run.end)
        if chunk.size < SAMPLE_RATE // 2:
            continue
        log.info(
            "ASR run %d/%d: %s %.1f–%.1f min",
            i, len(runs), run.language, run.start / 60, run.end / 60,
        )
        segments.extend(_offset(engine.transcribe(chunk, run.language), run.start, run.language))
        done += run.end - run.start
        if progress:
            progress(f"asr:{run.language}", done / total)

    segments.sort(key=lambda s: s["start"])
    log.info("ASR produced %d segments", len(segments))
    return segments


def _is_oom(exc: BaseException) -> bool:
    """Narrow on purpose: catching every RuntimeError here would halve the
    batch size in response to unrelated bugs and hide them."""
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    text = str(exc).lower()
    return "out of memory" in text or "cuda_error_out_of_memory" in text
