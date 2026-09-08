"""On-screen context from a screen recording.

When a meeting is a screen share, the hardest words for the audio model are the
easiest ones for anything that can read the screen. This pipeline writes
`do krem` for `dockerem` and `teamcech` for `teamsech` — English product names
inside Czech speech — and those words are usually printed right there in the
frame while somebody says them.

Two independent halves, deliberately separable:

* **OCR** extracts exact strings. These are what can safely reach the merge
  prompt: a one-word term swapped for a one-word mishearing is a small, checkable
  edit.
* **VLM** writes a sentence about what is on screen. This is for the reader, and
  it never enters the merge prompt — prose in a prompt whose entire discipline is
  "return the transcript unchanged" is an invitation to rewrite the transcript
  into something that sounds like the caption.

The halves are separable because the VLM is the expensive, fragile one. If it is
unavailable the OCR terms still land, and OCR is the half carrying the accuracy
claim.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger("transcribe.screen")

# showinfo prints one line per frame that survived the select filter.
PTS = re.compile(r"pts_time:([0-9.]+)")


class ScreenError(RuntimeError):
    """Frame extraction failed — a local, deterministic problem."""


class ScreenUnavailable(RuntimeError):
    """A service this stage needs is down. Transient: never cache the result."""


def extract_frames(
    source: Path,
    dest: Path,
    *,
    threshold: float = 0.02,
    max_frames: int = 600,
    width: int = 1280,
    start: float | None = None,
    duration: float | None = None,
) -> list[dict]:
    """Pull one frame per scene change, returning [{index, path, t}].

    Screen shares do not cut between shots; they scroll, type and move a cursor.
    Measured on a real 92-minute 1080p recording, a scene threshold of 0.10
    selected *zero* frames and 0.02 selected about 460 — so the usual
    slide-detection defaults are useless here and the threshold has to be low.

    Frames go to disk rather than down a pipe: `app.audio._run` is text-mode and
    cannot carry JPEG bytes, and having the frames on disk means a wrong result
    can be looked at rather than guessed about.
    """
    if not shutil.which("ffmpeg"):
        raise ScreenError("ffmpeg not found; this must run inside the container")
    # The upload lives in input/ as a bind-mounted host file, so it can be gone
    # by the time a cached job is re-run.
    if not source.is_file():
        raise ScreenError(f"source video is gone: {source}")

    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("*.jpg"):
        old.unlink()

    cmd = ["ffmpeg", "-nostdin", "-y", "-hide_banner"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(source)]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += [
        "-an",
        "-vf", f"select='gt(scene,{threshold})',showinfo,scale={width}:-2",
        "-vsync", "vfr", "-q:v", "3",
        str(dest / "%05d.jpg"),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise ScreenError(f"ffmpeg failed extracting frames:\n{tail}")

    times = [float(m) for m in PTS.findall(proc.stderr)]
    files = sorted(dest.glob("*.jpg"))
    if not files:
        log.warning("no scene changes above %.3f — the screen may be static; "
                    "lower SCREEN_SCENE_THRESHOLD to sample more", threshold)
        return []

    # showinfo and the muxer can disagree by one if the last frame is truncated.
    n = min(len(files), len(times))
    frames = [{"index": i, "path": str(files[i]), "t": times[i] + (start or 0.0)}
              for i in range(n)]

    if len(frames) > max_frames:
        # Keep an even spread rather than the first N: truncating the tail would
        # silently drop the entire second half of the meeting.
        step = len(frames) / max_frames
        frames = [frames[int(i * step)] for i in range(max_frames)]
        keep = {f["path"] for f in frames}
        for f in files:
            if str(f) not in keep:
                f.unlink()
        log.info("capped %d scene changes to %d evenly spread frames",
                 n, max_frames)

    log.info("extracted %d frame(s) at scene threshold %.3f", len(frames), threshold)
    return frames


# ---------------------------------------------------------------------------
# OCR — exact strings, the half that may reach the merge prompt
# ---------------------------------------------------------------------------

# Anything that looks like a credential. The user asked for security filtering
# only — names and contact details on screen are theirs and stay — but a token
# or password caught in a frame should never be written into a transcript or
# posted to a model, even a local one.
SECRET = re.compile(
    r"(?i)\b(?:password|passwd|heslo|token|api[_-]?key|secret|bearer|authorization"
    r"|client[_-]?secret|private[_-]?key)\b"
)
# Long high-entropy runs: session ids, access keys, hashes. Deliberately not
# requiring lowercase — an AWS access key is 20 uppercase-and-digit characters
# and slipped straight through a stricter pattern.
ENTROPIC = re.compile(r"^(?=.*\d)(?=.*[A-Za-z])[A-Za-z0-9+/_-]{20,}$")

WORDLIKE = re.compile(r"[^\W\d_][\w./-]{2,}", re.UNICODE)


def _is_secret(token: str, line: str) -> bool:
    return bool(SECRET.search(line) or ENTROPIC.match(token))


def ocr_frames(frames: list[dict], *, langs: str = "ces+eng") -> list[dict]:
    """Read each frame's visible text, returning per-frame token sets.

    Exact strings are the point. A vision model describing a screen gives prose
    that cannot safely be fed to a merge whose whole discipline is "change
    nothing"; `Dockerem` read literally off a button can.
    """
    if not shutil.which("tesseract"):
        raise ScreenError("tesseract not found; rebuild the image")

    out = []
    for f in frames:
        proc = subprocess.run(
            ["tesseract", f["path"], "stdout", "-l", langs, "--psm", "6"],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            log.warning("OCR failed on %s: %s", f["path"],
                        proc.stderr.strip().splitlines()[-1:] or "?")
            out.append({**f, "terms": []})
            continue

        terms: set[str] = set()
        for line in proc.stdout.splitlines():
            for m in WORDLIKE.finditer(line):
                tok = m.group()
                if len(tok) >= 3 and not _is_secret(tok, line):
                    terms.add(tok)
        out.append({**f, "terms": sorted(terms)})
    log.info("OCR read %d frame(s), %d distinct term(s)",
             len(out), len({t for f in out for t in f["terms"]}))
    return out


def check_ocr_language(langs: str = "ces") -> None:
    """Fail loudly if a traineddata is missing.

    tesseract does not error on an unknown language in every build — it can fall
    back and return fluent English-model output for Czech text. That failure is
    invisible downstream: the terms look plausible and are simply wrong.
    """
    proc = subprocess.run(["tesseract", "--list-langs"],
                          capture_output=True, text=True)
    have = {l.strip() for l in proc.stdout.splitlines()[1:] if l.strip()}
    missing = [l for l in langs.split("+") if l not in have]
    if missing:
        raise ScreenError(
            f"tesseract is missing traineddata for {missing}; it would silently "
            f"fall back and return confident nonsense. Have: {sorted(have)}"
        )


# ---------------------------------------------------------------------------
# VLM — description, for the reader only
# ---------------------------------------------------------------------------

CAPTION_PROMPT = (
    "This is a frame from a screen recording of a work meeting. In ONE short "
    "sentence, say what is on screen: the application, the view, and what the "
    "user appears to be doing. Do not guess at anything you cannot see. If the "
    "frame is blank or shows no application, say exactly: blank screen."
)


def _b64_jpeg(path: str) -> str:
    import base64

    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _ask_vision(base_url: str, model: str, image_b64: str, prompt: str,
                timeout: int) -> str:
    from app.llm import _post

    resp = _post(
        base_url,
        {"model": model, "max_tokens": 120, "temperature": 0.1,
         "messages": [{"role": "user", "content": [
             {"type": "text", "text": prompt},
             {"type": "image_url",
              "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
         ]}]},
        timeout,
    )
    return (resp["choices"][0]["message"]["content"] or "").strip()


def probe_vision(base_url: str, model: str, timeout: int = 300) -> str:
    """Prove the model can actually see, by showing it a word only it could read.

    This is the one check that matters, and no ordinary health check can do it.
    If the multimodal projector fails to load, llama.cpp still serves a perfectly
    healthy *text-only* model: `/v1/models` returns 200, a request carrying an
    image is accepted, the image is silently ignored, and the reply is a fluent,
    entirely invented description of a screen the model never saw. Every stage
    downstream succeeds, captions appear in the transcript, and the feature
    ships doing nothing at all.

    So: render a random nonce into an image, ask the model to read it back, and
    refuse to run the stage unless it does.
    """
    import secrets

    from PIL import Image, ImageDraw

    nonce = secrets.token_hex(3).upper()
    img = Image.new("RGB", (640, 200), "white")
    # Default bitmap font is tiny; scaling up a small render keeps it legible
    # without depending on a TTF being installed in the image.
    small = Image.new("RGB", (160, 50), "white")
    ImageDraw.Draw(small).text((10, 20), nonce, fill="black")
    img.paste(small.resize((640, 200), Image.LANCZOS), (0, 0))

    tmp = Path("/tmp/vision_probe.jpg")
    img.save(tmp, "JPEG", quality=95)

    try:
        reply = _ask_vision(
            base_url, model, _b64_jpeg(str(tmp)),
            "Read the text in this image. Reply with only that text.", timeout,
        )
    except Exception as exc:
        raise ScreenUnavailable(
            f"vision service did not answer the capability probe: {exc!r}"
        ) from exc

    if nonce.lower() not in reply.lower().replace(" ", ""):
        raise ScreenUnavailable(
            f"vision service is blind: it was shown {nonce!r} and replied "
            f"{reply[:120]!r}. The mmproj projector is almost certainly not "
            f"loaded — the server answers happily but never sees the image."
        )
    log.info("vision capability probe passed (read %s back correctly)", nonce)
    return nonce


def caption_frames(frames: list[dict], *, base_url: str, model: str,
                   timeout: int = 180, progress=None) -> list[dict]:
    """One sentence per frame. For the reader — never for the merge prompt."""
    out = []
    for i, f in enumerate(frames, 1):
        try:
            caption = _ask_vision(base_url, model, _b64_jpeg(f["path"]),
                                  CAPTION_PROMPT, timeout)
        except Exception as exc:
            log.warning("caption failed on frame %d (%s)", f["index"],
                        type(exc).__name__)
            caption = ""
        out.append({**f, "caption": caption})
        if progress:
            progress("screen", i / max(len(frames), 1))
    described = sum(1 for f in out if f["caption"])
    log.info("captioned %d/%d frame(s)", described, len(out))
    return out


# ---------------------------------------------------------------------------
# Choosing which terms are worth showing the merge
# ---------------------------------------------------------------------------

# Similar enough that the ASR plausibly misheard this word; different enough
# that it is not already the same word. Outside this band a term is either
# irrelevant or already correct, and in both cases it is noise in the prompt.
NEAR_LO, NEAR_HI = 0.55, 0.97


def select_terms(window_text: str, window_frames: list[dict],
                 chrome: set[str], *, limit: int = 20,
                 vocabulary: set[str] | None = None,
                 min_frames: int = 2) -> list[str]:
    """Pick the few on-screen terms that might fix this window's transcript.

    Two filters, and the second one exists because the first was not enough.

    Persistent UI chrome goes first: a label in almost every frame is furniture,
    not what anyone is discussing. What survives must then be a *near-miss* of a
    word already in the transcript, which is what turns "everything on screen"
    into "spellings the transcript probably got wrong".

    On its own that selects the wrong half of the pair. OCR truncates, and a
    truncation resembles the word it came from more than anything else does, so
    the near-miss test happily surfaced `lefon` for a spoken "telefon" and
    `komunika` for "komunikace" — offering the model a fragment as the
    correction for a word it had already got right. Nothing was ever applied,
    which is exactly what the applied/rejected counters are there to reveal.

    So a candidate must also look like a real label: seen in more than one
    frame, long enough not to be debris, and not merely a piece of some longer
    term that OCR also read. `Telefon` survives; `lefon` does not.
    """
    from difflib import SequenceMatcher

    spoken = {w.lower() for w in WORDLIKE.findall(window_text)}
    if not spoken:
        return []

    seen: dict[str, int] = {}
    for f in window_frames:
        for t in f.get("terms", []):
            if t.lower() not in chrome:
                seen[t] = seen.get(t, 0) + 1

    vocab = {v.lower() for v in (vocabulary or set())}

    scored: list[tuple[float, int, str]] = []
    for term, frames_seen in seen.items():
        low = term.lower()
        if low in spoken or len(low) < 4 or frames_seen < min_frames:
            continue
        # A fragment of something else OCR read on the same screen.
        if any(low != other and low in other for other in vocab):
            continue
        best = max((SequenceMatcher(None, low, s).ratio() for s in spoken),
                   default=0.0)
        if NEAR_LO <= best <= NEAR_HI:
            scored.append((best, frames_seen, term))

    scored.sort(reverse=True)
    return [t for _, _, t in scored[:limit]]


def chrome_terms(frames: list[dict], *, threshold: float = 0.8,
                 min_frames: int = 10) -> set[str]:
    """Labels present in most frames: menus, toolbars, the window title.

    Needs enough frames to mean anything. Over three frames, "present in 80% of
    them" describes almost every word on screen, so the filter would discard the
    very terms it exists to surface — on a two-frame sample it removed the one
    correct term and returned nothing at all.
    """
    if len(frames) < min_frames:
        return set()
    counts: dict[str, int] = {}
    for f in frames:
        for t in set(x.lower() for x in f.get("terms", [])):
            counts[t] = counts.get(t, 0) + 1
    cut = threshold * len(frames)
    return {t for t, n in counts.items() if n >= cut}


class ScreenContext:
    """What was on screen, queryable by time window.

    Holds the OCR terms and the captions apart on purpose. `terms_for()` feeds
    the merge prompt; `captions_for()` feeds the export and nothing else. Mixing
    them would make any accuracy result unattributable — and would put prose
    into a prompt that exists to stop the model writing prose.
    """

    def __init__(self, frames: list[dict], *, limit: int = 20):
        self.frames = frames or []
        self.limit = limit
        self.chrome = chrome_terms(self.frames)
        # Every string OCR read anywhere, so a candidate that is
        # merely a piece of a longer one can be discarded.
        self.vocabulary = {t.lower() for f in self.frames
                           for t in f.get("terms", [])}

    def __bool__(self) -> bool:
        return bool(self.frames)

    def _window(self, t0: float, t1: float, pad: float = 30.0) -> list[dict]:
        # Padded: people keep discussing a screen after they stop changing it,
        # and at one scene change per ~12s a bare 90s window can hold very few.
        return [f for f in self.frames if t0 - pad <= f["t"] <= t1 + pad]

    def terms_for(self, window_text: str, t0: float, t1: float) -> list[str]:
        return select_terms(window_text, self._window(t0, t1), self.chrome,
                            limit=self.limit,
                            vocabulary=self.vocabulary)

    def captions_for(self, t0: float, t1: float) -> list[dict]:
        return [{"t": f["t"], "caption": f["caption"]}
                for f in self._window(t0, t1, pad=0.0) if f.get("caption")]

    def to_track(self) -> list[dict]:
        """The display-only annotation track written into result.json."""
        out = []
        for f in self.frames:
            if f.get("caption"):
                out.append({"t": round(f["t"], 2), "caption": f["caption"]})
        return out
