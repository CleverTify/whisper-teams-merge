"""Configuration, paths, logging and hardware detection.

Two things here are load-bearing:

* `resolve_backend()` decides faster-whisper-on-CUDA vs whisper.cpp, which is
  what makes one image work on NVIDIA, AMD, Apple and plain CPU.
* `resolve_compute_type()` blocks INT8 on Blackwell, where it is a hard crash
  rather than a slowdown.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger("transcribe.config")

WORK_DIR = Path(os.environ.get("WORK_DIR", "/work"))
INPUT_DIR = WORK_DIR / "input"
OUTPUT_DIR = WORK_DIR / "output"
CACHE_DIR = Path(os.environ.get("CACHE_DIR", "/cache"))

LangMode = Literal["single", "auto", "multi"]
Backend = Literal["faster-whisper", "whisper.cpp"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    hf_token: str = ""

    # --- ASR ---
    whisper_model: str = "large-v3"
    compute_type: str = "float16"
    batch_size: int = 8
    beam_size: int = 5
    # "auto" picks faster-whisper when CUDA is present, whisper.cpp otherwise.
    backend: str = "auto"
    whisper_cpp_model: str = "ggml-large-v3.bin"
    # Primes the decoder with vocabulary it would otherwise mangle. **Empty on
    # purpose.** It does fix the named terms — a 196-char prompt turned "do krem"
    # into "dockerem" and "trůhelníček" into "trojúhelníček" — but it steers the
    # decoder into compressing everything else, and the cost dwarfs the gain.
    # Measured against an independent reference on a 92-minute Czech call:
    #
    #     prompt              error vs reference    words missing
    #     none                       19.51%              1048
    #     113 chars (terms)          21.73%              1319
    #     196 chars (sentence)       24.92%              1656
    #
    # Monotonic in prompt length, so shorter is less bad and zero is best. Set
    # it only if you have a benchmark showing it helps on *your* audio.
    initial_prompt: str = ""
    # Voice-activity gate in front of the decoder. Whisper never sees audio the
    # VAD rejects, so these decide what can be transcribed at all — on a real
    # call, missing words outnumbered wrong words three to one. `vad_onset` is
    # how confident it must be to *start* a speech region and `vad_offset` how
    # readily it ends one, so lowering either keeps more audio.
    vad_onset: float = 0.500
    vad_offset: float = 0.363
    vad_chunk_size: int = 30

    # --- screen context (video uploads only) ---
    # A screen share puts half the meaning on the screen. The words this
    # pipeline most reliably gets wrong are English product names inside Czech
    # speech, and those are usually printed in the frame while somebody says
    # them. Off for audio-only jobs by construction: the stage is gated on the
    # video track ffprobe already detects at ingest.
    screen_enabled: bool = True
    # Screen shares scroll and type rather than cut between shots. Measured on a
    # real 1080p recording, a threshold of 0.10 selected zero frames and 0.02
    # selected ~460 over 92 minutes, so the usual slide-detection defaults are
    # useless here.
    screen_scene_threshold: float = 0.02
    screen_max_frames: int = 600
    screen_frame_width: int = 1280
    screen_ocr_langs: str = "ces+eng"
    screen_terms_per_window: int = 20
    # Feed on-screen words into the merge prompt as correction candidates.
    # **Off by default, on measured evidence.** The idea was that the words
    # this pipeline mis-hears are printed on screen while they are spoken.
    # On a real 92-minute screen share that turned out to be false: across
    # 420 frames OCR never once read "docker", "trojuhelnicek", "kanban" or
    # "appku" — the words we get wrong. People were *describing* the screen
    # ("the little triangle") rather than reading text off it, so screen
    # text and spoken vocabulary barely intersect.
    #
    # Measured: zero terms were ever applied, and the preamble moved the
    # score in both directions across runs — perturbation, not correction.
    # Worth switching on for a slide deck or a documentation walkthrough,
    # where the words really are on screen. Verify with bench_wording.py
    # before trusting it on your material.
    screen_terms_in_prompt: bool = False
    # The VLM is the expensive half and only writes descriptions for the reader.
    # It can be switched off independently so a smaller card still gets the OCR
    # terms, which are the half carrying the accuracy claim.
    screen_vlm_enabled: bool = True
    screen_vlm_base_url: str = "http://llm-vision:8081/v1"
    screen_vlm_model: str = "qwen3-vl-8b"
    screen_vlm_timeout: int = 180

    # --- languages (kept: this is an EU-market tool) ---
    lang_mode: LangMode = "auto"
    language: str = ""
    lang_dominance: float = 0.85
    lid_samples: int = 30

    # --- diarization ---
    diarize: bool = True
    diarize_model: str = "pyannote/speaker-diarization-community-1"
    min_speakers: int | None = None
    max_speakers: int | None = None
    # Exact number of people, when you know it. Applied *after* diarization by
    # collapsing clusters on turn-taking — far more reliable than constraining
    # pyannote, which also degrades its segmentation. See app/diarize.py.
    speakers: int | None = None
    turn_merge_gap: float = 0.8
    speaker_smooth_words: int = 3
    min_turn_seconds: float = 0.7
    boundary_snap_words: int = 3
    # Clusters whose speaker embeddings are at least this similar are treated as
    # one person. Measured on a real two-person call: the same voice split into
    # three clusters scored 0.75-0.80 between them, while the two genuinely
    # different voices scored 0.42. 0.60 separates those without risking a merge
    # of real speakers — raise it to merge less, lower it to merge more.
    speaker_merge_threshold: float = 0.60
    # pyannote's own VBx clustering threshold. community-1 ships 0.6, which
    # over-splits real calls; 0.80 sits in the middle of the stable band
    # measured on a 59-minute two-person recording. See app/diarize.py.
    cluster_threshold: float = 0.80

    # --- local LLM (llama.cpp, OpenAI-compatible) ---
    llm_enabled: bool = True
    llm_base_url: str = "http://llm:8081/v1"
    llm_model: str = "qwen3-8b-instruct"
    llm_window_seconds: float = 90.0
    llm_timeout: int = 180
    # Guardrails. A model asked to "improve" a transcript will invent text, so
    # every chunk it returns must look like the chunk it was given.
    llm_max_length_ratio: float = 1.6
    llm_min_length_ratio: float = 0.5
    llm_min_token_overlap: float = 0.55
    # Domain terms the merge may spell correctly, comma- or newline-separated.
    # Whisper mishears names, products and jargon it has no prior for, and that
    # is the one error class where handing the model the right answer is honest
    # rather than leading. It is NOT a decoder prompt: `initial_prompt` was this
    # same idea one stage earlier and lost monotonically (19.51 -> 21.73 ->
    # 24.92 as the prompt grew). Here a term can only *replace* a word already
    # present -- `llm.screen_term_ok()` rejects it as an insertion -- so a wrong
    # glossary costs substitutions, never invented sentences.
    llm_glossary: str = ""

    # --- audio ---
    loudnorm: bool = True
    highpass_hz: int = 70
    denoise: bool = False

    log_level: str = "INFO"

    @property
    def glossary_terms(self) -> list[str]:
        """`llm_glossary` split on commas/newlines, de-duplicated, order kept."""
        seen: dict[str, None] = {}
        for raw in re.split(r"[,\n]", self.llm_glossary):
            term = raw.strip()
            if term:
                seen.setdefault(term, None)
        return list(seen)

    @field_validator("min_speakers", "max_speakers", "speakers", mode="before")
    @classmethod
    def _blank_to_none(cls, v):
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("language", mode="before")
    @classmethod
    def _language_to_str(cls, v):
        return "" if v is None else str(v).strip()


settings = Settings()


def with_overrides(base: "Settings", overrides: dict) -> "Settings":
    """Apply overrides **with validation**.

    `model_copy(update=...)` runs no validators, so values from the web UI
    (always strings) stayed strings and pyannote died on
    `'<' not supported between instances of 'int' and 'str'`. Re-validating the
    merged dict coerces types properly.
    """
    clean = {k: v for k, v in (overrides or {}).items() if v is not None and v != ""}
    if not clean:
        return base
    return type(base).model_validate({**base.model_dump(), **clean})


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Hardware:
    cuda: bool
    device: str
    name: str = ""
    capability: tuple[int, int] = (0, 0)
    vram_gb: float = 0.0
    torch_version: str = ""
    torch_cuda: str = ""

    @property
    def sm(self) -> int:
        return self.capability[0] * 10 + self.capability[1]

    @property
    def is_blackwell_or_newer(self) -> bool:
        return self.capability[0] >= 12


def hardware() -> Hardware:
    import torch

    if not torch.cuda.is_available():
        return Hardware(
            cuda=False, device="cpu", name="CPU", torch_version=torch.__version__
        )
    p = torch.cuda.get_device_properties(0)
    return Hardware(
        cuda=True,
        device="cuda",
        name=p.name,
        capability=(p.major, p.minor),
        vram_gb=round(p.total_memory / 1024**3, 2),
        torch_version=torch.__version__,
        torch_cuda=torch.version.cuda or "",
    )


def resolve_backend(requested: str, hw: Hardware | None = None) -> Backend:
    """faster-whisper needs CUDA (CTranslate2 has no ROCm); whisper.cpp is the
    universal fallback and reaches AMD/Intel GPUs through Vulkan."""
    hw = hw or hardware()
    if requested in ("faster-whisper", "whisper.cpp"):
        return requested  # type: ignore[return-value]
    return "faster-whisper" if hw.cuda else "whisper.cpp"


def resolve_compute_type(requested: str, hw: Hardware | None = None) -> str:
    """CTranslate2 4.6.2 disabled INT8 for sm_120. Asking for it on Blackwell
    raises CUBLAS_STATUS_NOT_SUPPORTED mid-batch rather than falling back."""
    hw = hw or hardware()
    if not hw.cuda:
        return "int8"  # CTranslate2 CPU is genuinely good at int8
    if hw.is_blackwell_or_newer and "int8" in requested:
        logging.getLogger("transcribe").warning(
            "compute_type=%s unsupported on %s (sm_%d); using float16",
            requested, hw.name, hw.sm,
        )
        return "float16"
    return requested


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def slugify(name: str) -> str:
    """ASCII-safe folder name. Teams names files
    "Team sync-20260804_173547-Meeting Recording"."""
    ascii_name = (
        unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    )
    slug = re.sub(r"[^\w\s-]", "", ascii_name).strip().lower()
    slug = re.sub(r"[\s_]+", "-", slug)
    return re.sub(r"-{2,}", "-", slug).strip("-") or "recording"


@dataclass(frozen=True)
class JobPaths:
    source: Path
    root: Path

    @property
    def work(self) -> Path:
        return self.root / "work"

    @property
    def audio(self) -> Path:
        return self.work / "audio.wav"

    @property
    def frames(self) -> Path:
        """Extracted video frames. Kept on disk so a wrong result can be looked
        at rather than guessed about."""
        return self.work / "frames"

    def stage(self, name: str) -> Path:
        return self.work / f"{name}.json"

    @property
    def transcript_md(self) -> Path:
        return self.root / "transcript.md"

    @property
    def vtt(self) -> Path:
        return self.root / "transcript.vtt"

    @property
    def result_json(self) -> Path:
        return self.root / "result.json"

    @property
    def removed_jsonl(self) -> Path:
        return self.root / "removed.jsonl"

    def ensure(self) -> "JobPaths":
        self.work.mkdir(parents=True, exist_ok=True)
        return self


def job_paths(source: Path, output_root: Path | None = None,
              trial: bool = False) -> JobPaths:
    """Where a job's artifacts live — one folder per source file.

    A trial run gets its own folder. `--duration 90 --force` otherwise lands in
    the same place as the real thing and replaces a finished 92-minute
    transcript with 90 seconds of one, cache included. The stage fingerprints
    exist to stop exactly that, and `--force` walks straight past them.
    """
    name = slugify(source.stem)
    if trial:
        name += "-trial"
    root = (output_root or OUTPUT_DIR) / name
    return JobPaths(source=source, root=root).ensure()


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def json_default(obj):
    """WhisperX returns numpy scalars and pyannote returns ndarrays; plain
    json.dumps raises TypeError on both."""
    import numpy as np

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def dump_json(obj, *, indent: int | None = None) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, indent=indent, default=json_default)


def setup_logging(level: str | None = None) -> logging.Logger:
    lvl = (level or settings.log_level).upper()
    root = logging.getLogger()
    if not root.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
        )
        root.addHandler(h)
    root.setLevel(lvl)
    for noisy in ("speechbrain", "pyannote", "urllib3", "filelock",
                  "huggingface_hub", "numba", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("transcribe")


def free_vram_gb() -> float:
    """Free VRAM on the whole device, not just this process.

    The models live in three different containers, so a process-local figure
    would be blind to exactly the thing that matters. mem_get_info reports the
    device, which is what lets one container wait for another to let go.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        free, _total = torch.cuda.mem_get_info()
        return free / 1024 ** 3
    except Exception:
        return 0.0


def wait_for_free(need_gb: float, *, timeout: float = 180.0,
                  reason: str = "") -> bool:
    """Block until need_gb is actually free, or give up.

    Two large models cannot share this card — 5.9 GB of text model plus 6.5 GB
    of vision model does not fit in 11.94 GB — and the services evict themselves
    on an idle *timer*. Trusting the timer fails silently and specifically: the
    next model loads while the previous one is still resident, CUDA runs out of
    memory inside a request, the caller's generic exception handler reads it as
    a timeout, retries, and the run finishes and reports success having quietly
    skipped work. So check, rather than assume.
    """
    import time

    if free_vram_gb() >= need_gb:
        return True
    log.info("waiting for %.1f GB of VRAM%s (%.1f free)",
             need_gb, f" — {reason}" if reason else "", free_vram_gb())
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(5)
        if free_vram_gb() >= need_gb:
            log.info("VRAM available after waiting")
            return True
    log.warning("still only %.1f GB free after %.0fs, needed %.1f",
                free_vram_gb(), timeout, need_gb)
    return False


def free_gpu() -> None:
    """Release CUDA blocks between stages — Whisper, the aligner, pyannote and
    the LLM cannot all be resident in 12 GB at once."""
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
