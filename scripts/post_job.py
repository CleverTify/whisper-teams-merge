"""Submit a job to the running web app exactly as the browser does.

Useful for testing the web path (which is a different code path from the CLI —
the UI goes through the long-running server process, the CLI spawns a fresh
one).

    docker compose exec app python scripts/post_job.py input/file.mp4 [input/file.vtt]
"""

from __future__ import annotations

import json
import mimetypes
import sys
import urllib.request
import uuid
from pathlib import Path

URL = "http://localhost:8080/api/jobs"


def part(name: str, path: Path) -> bytes:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    head = (
        f'Content-Disposition: form-data; name="{name}"; filename="{path.name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode()
    return head + path.read_bytes() + b"\r\n"


def main() -> int:
    media = Path(sys.argv[1])
    if not media.is_absolute():
        media = Path("/work") / media
    transcript = None
    if len(sys.argv) > 2:
        transcript = Path(sys.argv[2])
        if not transcript.is_absolute():
            transcript = Path("/work") / transcript

    boundary = uuid.uuid4().hex
    sep = f"--{boundary}\r\n".encode()
    body = sep + part("media", media)
    if transcript:
        body += sep + part("transcript", transcript)
    body += f"--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        URL, data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "Content-Length": str(len(body))},
    )
    print(f"posting {media.name}"
          + (f" + {transcript.name}" if transcript else "")
          + f" ({len(body)/1024**2:.1f} MB)")
    with urllib.request.urlopen(req, timeout=900) as resp:
        print("->", json.loads(resp.read().decode()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
