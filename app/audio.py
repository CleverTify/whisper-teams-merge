"""Audio ingest and preprocessing (ffmpeg, CPU).

Meeting recordings are far-field, reverberant and unevenly levelled. Whisper is
sensitive to level, so loudness normalisation is on by default. Spectral
denoise is *off* by default because on real meeting audio it removes consonant
energy and measurably hurts WER more often than it helps.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np

log = logging.getLogger("transcribe.audio")

SAMPLE_RATE = 16_000


class AudioError(RuntimeError):
    pass


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    log.debug("exec: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-15:]
        raise AudioError(f"{cmd[0]} failed ({proc.returncode}):\n" + "\n".join(tail))
    return proc


def probe(path: Path) -> dict:
    """Return duration/codec/channel metadata for any container ffmpeg reads."""
    if not path.exists():
        raise AudioError(f"input not found: {path}")
    proc = _run(
        [
            "ffprobe", "-v", "error",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(path),
        ]
    )
    data = json.loads(proc.stdout)
    streams = data.get("streams", [])
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if audio is None:
        raise AudioError(f"no audio stream in {path.name}")
    fmt = data.get("format", {})
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    return {
        "duration": float(fmt.get("duration") or audio.get("duration") or 0.0),
        "codec": audio.get("codec_name", "?"),
        "sample_rate": int(audio.get("sample_rate") or 0),
        "channels": int(audio.get("channels") or 0),
        "bit_rate": int(fmt.get("bit_rate") or 0),
        "size_bytes": int(fmt.get("size") or path.stat().st_size),
        "format": fmt.get("format_name", "?"),
        **_video_info(video),
    }


# Codecs that appear as a "video" stream but are really embedded artwork.
COVER_ART_CODECS = {"mjpeg", "png", "bmp", "gif", "webp"}


def _video_info(video: dict | None) -> dict:
    """Describe a genuine video track, ignoring embedded cover art.

    Plenty of .m4a and .mp3 files carry album art as an mjpeg/png "video"
    stream. Treating that as video would park a still image in the review
    player for every audio-only recording, so it is excluded explicitly.
    """
    if not video:
        return {"has_video": False}
    if video.get("disposition", {}).get("attached_pic"):
        return {"has_video": False}
    if video.get("codec_name") in COVER_ART_CODECS:
        return {"has_video": False}
    return {
        "has_video": True,
        "video_codec": video.get("codec_name", "?"),
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
    }


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def build_filter_chain(
    *, loudnorm: bool = True, highpass_hz: int = 70, denoise: bool = False
) -> str:
    """Order matters: denoise, then rumble removal, then level."""
    chain: list[str] = []
    if denoise:
        chain.append("afftdn=nf=-25")
    if highpass_hz > 0:
        chain.append(f"highpass=f={highpass_hz}")
    if loudnorm:
        # Meeting-friendly target: quieter than broadcast (-18 LUFS) with
        # generous range so soft speakers are lifted without pumping.
        chain.append("loudnorm=I=-18:TP=-2:LRA=11")
    return ",".join(chain)


def preprocess(
    src: Path,
    dst: Path,
    *,
    loudnorm: bool = True,
    highpass_hz: int = 70,
    denoise: bool = False,
    start: float | None = None,
    duration: float | None = None,
    force: bool = False,
) -> dict:
    """Decode any input to 16 kHz mono s16 WAV, normalised for ASR.

    Returns metadata describing both the source and the produced WAV. Skips
    work when the target already exists unless `force`.
    """
    if shutil.which("ffmpeg") is None:
        raise AudioError("ffmpeg not found — this must run inside the container")

    meta = probe(src)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and not force:
        log.info("reusing preprocessed audio: %s", dst.name)
    else:
        cmd = ["ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error"]
        if start:
            cmd += ["-ss", f"{start:.3f}"]
        cmd += ["-i", str(src)]
        if duration:
            cmd += ["-t", f"{duration:.3f}"]
        chain = build_filter_chain(
            loudnorm=loudnorm, highpass_hz=highpass_hz, denoise=denoise
        )
        cmd += ["-vn", "-map", "0:a:0", "-ac", "1", "-ar", str(SAMPLE_RATE)]
        if chain:
            cmd += ["-af", chain]
        cmd += ["-c:a", "pcm_s16le", str(dst)]

        log.info(
            "decoding %s (%.1f min, %s %d ch @ %d Hz) -> 16 kHz mono%s",
            src.name,
            meta["duration"] / 60,
            meta["codec"],
            meta["channels"],
            meta["sample_rate"],
            f" [{chain}]" if chain else "",
        )
        _run(cmd)

    out_meta = probe(dst)
    return {
        "source": {
            "path": str(src),
            "name": src.name,
            "sha256": sha256(src),
            **meta,
        },
        "prepared": {
            "path": str(dst),
            "duration": out_meta["duration"],
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
        },
        "preprocess": {
            "loudnorm": loudnorm,
            "highpass_hz": highpass_hz,
            "denoise": denoise,
            "start": start,
            "duration_limit": duration,
            "filters": build_filter_chain(
                loudnorm=loudnorm, highpass_hz=highpass_hz, denoise=denoise
            ),
        },
    }


def load_wav(path: Path) -> np.ndarray:
    """Load the preprocessed WAV as float32 mono in [-1, 1] (WhisperX's format)."""
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != SAMPLE_RATE:
        raise AudioError(f"expected {SAMPLE_RATE} Hz, got {sr}")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32)


def slice_audio(audio: np.ndarray, start: float, end: float) -> np.ndarray:
    a = max(0, int(round(start * SAMPLE_RATE)))
    b = min(len(audio), int(round(end * SAMPLE_RATE)))
    return audio[a:b] if b > a else np.zeros(0, dtype=np.float32)


def format_timestamp(seconds: float, *, sep: str = ",", always_hours: bool = True) -> str:
    """SRT (`,`) / VTT (`.`) timestamp."""
    seconds = max(0.0, float(seconds))
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    if always_hours or h:
        return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"
    return f"{m:02d}:{s:02d}{sep}{ms:03d}"


def format_clock(seconds: float) -> str:
    """Human-readable `HH:MM:SS` for transcript headings."""
    total = int(round(max(0.0, float(seconds))))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
