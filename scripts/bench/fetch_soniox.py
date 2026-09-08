"""Fetch one Soniox reading of an audio file, then delete it from their servers.

This is the **only** part of the project that leaves the machine, and it exists
solely to give the wording benchmark a third opinion. Our transcript and the
Teams transcript both have errors, so where they disagree neither can settle it;
a reading that is independent of both can. Nothing in the shipped pipeline calls
this — the app stays local.

Run it once per configuration and cache the result. The scorer reads the cached
JSON and never touches the network, so a benchmark stays reproducible after the
vendor's `stt-async-v5` has moved on underneath it.

    SONIOX_API_KEY=... python scripts/fetch_soniox.py audio.m4a bench/soniox/cs-plain.json [config]

The key is read from the environment and never written to disk or logged. The
uploaded file and the transcription job are both deleted before this exits: the
audio is somebody's real meeting and should not be left sitting in a third
party's bucket.
"""

from __future__ import annotations

import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

# Importable from anywhere: sibling helpers first, then the repo root so
# `app` resolves without relying on PYTHONPATH being set.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


BASE = "https://api.soniox.com/v1"
POLL_SECONDS = 10
POLL_TIMEOUT = 3600

# Terms whose correct spelling is externally verifiable: product names, and
# the participant names from your own Teams metadata if you add them.
# Deliberately NOT the Teams transcript text — feeding that in would improve
# Soniox's output while destroying the independence that is the whole point
# of asking a second engine. Note these strings are uploaded, so put nothing
# here you would not send to a third party.
TERMS = [
    "Docker", "Fluent UI", "WhatsApp", "Outlook", "Azure", "Kanban", "React",
    "Microsoft Teams", "Visual Studio Code", "CRM", "iPhone", "macOS",
    "FaceTime", "Markdown", "API", "SDK",
]

CONFIGS: dict[str, dict] = {
    # The clean independent reference: no priming at all.
    "cs-plain": {
        "language_hints": ["cs"],
    },
    # Same, plus domain vocabulary. Czech ASR mangles English product names;
    # naming them should not bias ordinary wording.
    "cs-context": {
        "language_hints": ["cs"],
        "context": {
            "general": [
                {"key": "domain", "value": "Software development"},
                {"key": "topic", "value": "CRM product review meeting"},
                {"key": "language", "value": "Czech with English technical terms"},
            ],
            "terms": TERMS,
        },
    },
    # Only worth running if the first two mangle embedded English.
    "cs-en-context": {
        "language_hints": ["cs", "en"],
        "context": {
            "general": [
                {"key": "domain", "value": "Software development"},
                {"key": "topic", "value": "CRM product review meeting"},
            ],
            "terms": TERMS,
        },
    },
}


def _key() -> str:
    key = os.environ.get("SONIOX_API_KEY", "").strip()
    if not key:
        sys.exit("SONIOX_API_KEY is not set. Pass it as an environment variable; "
                 "do not put it in .env or any file.")
    return key


def _request(method: str, path: str, key: str, *, body: bytes | None = None,
             headers: dict | None = None, timeout: int = 120,
             attempts: int = 3) -> dict:
    """One Soniox call, retried on truncated or dropped responses.

    The transcript for 92 minutes of audio is several megabytes and arrived
    short once (`IncompleteRead`), which is a transport hiccup rather than an
    API error — retrying the GET is safe because it is idempotent. HTTP errors
    are not retried: those are real answers and repeating them just burns money.
    """
    last = None
    for attempt in range(1, attempts + 1):
        req = urllib.request.Request(
            BASE + path, data=body, method=method,
            headers={"Authorization": f"Bearer {key}", **(headers or {})},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            # Never echo the request headers here — they carry the key.
            raise SystemExit(f"Soniox {method} {path} failed: HTTP {exc.code} {detail}")
        except Exception as exc:  # IncompleteRead, URLError, socket timeout
            last = exc
            if attempt < attempts:
                print(f"  {type(exc).__name__} on {method} {path}, "
                      f"retry {attempt}/{attempts - 1}", flush=True)
                time.sleep(5 * attempt)
    raise SystemExit(f"Soniox {method} {path} failed after {attempts} attempts: {last!r}")


def upload(path: Path, key: str) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n".encode()
        + f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'.encode()
        + f"Content-Type: {mime}\r\n\r\n".encode()
        + path.read_bytes()
        + f"\r\n--{boundary}--\r\n".encode()
    )
    print(f"uploading {path.name} ({len(body) / 1024**2:.0f} MB)...", flush=True)
    out = _request("POST", "/files", key, body=body,
                   headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                            "Content-Length": str(len(body))},
                   timeout=1800)
    file_id = out.get("id")
    if not file_id:
        raise SystemExit(f"no file id in upload response: {out}")
    print(f"  file_id={file_id}")
    return file_id


def transcribe(file_id: str, config: str, key: str) -> str:
    payload = {"model": "stt-async-v5", "file_id": file_id, **CONFIGS[config]}
    out = _request("POST", "/transcriptions", key,
                   body=json.dumps(payload).encode("utf-8"),
                   headers={"Content-Type": "application/json"})
    tid = out.get("id")
    if not tid:
        raise SystemExit(f"no transcription id in response: {out}")
    print(f"  transcription_id={tid}  config={config}")
    return tid


def wait(tid: str, key: str) -> None:
    waited = 0
    while waited < POLL_TIMEOUT:
        info = _request("GET", f"/transcriptions/{tid}", key)
        status = info.get("status")
        if status == "completed":
            print(f"  completed after {waited}s")
            return
        if status == "error":
            raise SystemExit(f"transcription failed: {info.get('error_message') or info}")
        print(f"  status={status} ({waited}s)", flush=True)
        time.sleep(POLL_SECONDS)
        waited += POLL_SECONDS
    raise SystemExit(f"timed out after {POLL_TIMEOUT}s waiting for {tid}")


def cleanup(tid: str | None, file_id: str | None, key: str) -> None:
    """Best effort, and always attempted — this is somebody's real meeting."""
    for label, path in (("transcription", f"/transcriptions/{tid}" if tid else None),
                        ("file", f"/files/{file_id}" if file_id else None)):
        if not path:
            continue
        try:
            _request("DELETE", path, key)
            print(f"  deleted {label} from Soniox")
        except SystemExit as exc:
            print(f"  WARNING: could not delete {label}: {exc}")


def main() -> int:
    audio = Path(sys.argv[1])
    if not audio.is_absolute():
        audio = Path("/work") / audio
    out_path = Path(sys.argv[2])
    if not out_path.is_absolute():
        out_path = Path("/work") / out_path
    config = sys.argv[3] if len(sys.argv) > 3 else "cs-plain"
    if config not in CONFIGS:
        raise SystemExit(f"unknown config {config!r}; have {sorted(CONFIGS)}")

    key = _key()
    tid = file_id = None
    try:
        file_id = upload(audio, key)
        tid = transcribe(file_id, config, key)
        wait(tid, key)
        transcript = _request("GET", f"/transcriptions/{tid}/transcript", key, timeout=300)
    finally:
        if tid or file_id:
            cleanup(tid, file_id, key)

    tokens = transcript.get("tokens") or []
    text = transcript.get("text") or ""
    span = max((t.get("end_ms", 0) for t in tokens), default=0) / 1000.0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"config": config, "model": "stt-async-v5", "request": CONFIGS[config],
         "audio": audio.name, "transcript": transcript},
        ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n{len(tokens)} tokens, text {len(text)} chars, span {span/60:.1f} min")
    print("NOTE: Soniox tokens are sub-word — 'Beautiful' can arrive as Beau/ti/ful.")
    print("      The scorer must rejoin token text before tokenising into words.")
    print(f"saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
