"""Web UI and API.

One upload flow: a media file plus an optional Teams transcript. A single
background worker runs jobs — the GPU is serial, so a queue of one is the
honest model. Results are keyed on the output directory, so transcripts made
from the CLI are reviewable too.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from app import __version__
from app.config import (
    INPUT_DIR,
    OUTPUT_DIR,
    dump_json,
    hardware,
    resolve_backend,
    settings,
    setup_logging,
    with_overrides,
)

log = setup_logging()
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

DB_PATH = OUTPUT_DIR / "jobs.db"
STATIC = Path(__file__).parent / "static"

MEDIA_SUFFIXES = {
    ".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus", ".aac", ".wma",
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v",
}
TRANSCRIPT_SUFFIXES = {".vtt", ".docx"}
ARTIFACTS = {
    "transcript.md": "text/markdown",
    "transcript.vtt": "text/vtt",
    "result.json": "application/json",
    "removed.jsonl": "application/json",
}
# Containers a browser <video> can decode; anything else falls back to the
# preprocessed 16 kHz WAV, which always plays.
PLAYABLE = {
    ".m4a": "audio/mp4", ".mp4": "video/mp4", ".m4v": "video/mp4",
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg",
    ".opus": "audio/ogg", ".webm": "audio/webm", ".aac": "audio/aac",
    ".flac": "audio/flac",
}

app = FastAPI(title="Transcribe", version=__version__)

# The server has no authentication, by design: it is a local tool bound to
# 127.0.0.1 by docker-compose. That makes two browser-level attacks worth
# closing, because neither needs the attacker to reach the port directly.
MAX_UPLOAD_BYTES = 16 * 1024 ** 3
ALLOWED_ORIGINS = frozenset({
    "http://127.0.0.1:8080", "http://localhost:8080",
})
ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost", ""})


@app.middleware("http")
async def _local_only(request, call_next):
    """Refuse cross-origin writes and unfamiliar Host headers.

    POST /api/jobs takes multipart/form-data, which the browser treats as a
    CORS "simple request" -- no preflight. So any page the user happens to
    visit could submit a form to http://127.0.0.1:8080/api/jobs, write a file
    into ./input and queue a GPU job. The attacker cannot read the response,
    but the write and the machine time are real.

    The Host check is for DNS rebinding: an attacker-controlled name that
    resolves to 127.0.0.1 would otherwise give a page same-origin access to
    every transcript on the machine.
    """
    host = (request.headers.get("host") or "").rsplit(":", 1)[0]
    if host not in ALLOWED_HOSTS:
        # Say why, because the alternative is a bare 400 to someone who has
        # just published the port and cannot see what they did wrong.
        return JSONResponse(
            {"detail": f"this server answers on localhost only, and the request "
                       f"arrived for host {host!r}. It is unauthenticated by "
                       f"design -- see SECURITY.md before exposing it. To reach "
                       f"it from another machine, put an authenticating reverse "
                       f"proxy in front and have that proxy send "
                       f"Host: 127.0.0.1."},
            status_code=400)
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        if origin and origin not in ALLOWED_ORIGINS:
            return JSONResponse(
                {"detail": "cross-origin request refused"}, status_code=403)
    response = await call_next(request)
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; img-src 'self' data:; media-src 'self' blob:; "
        "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    return response
_queue: "queue.Queue[int]" = queue.Queue()
_cache: dict[str, tuple[float, dict]] = {}


# ---------------------------------------------------------------------------
# Job store
# ---------------------------------------------------------------------------

def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c


# Columns the current schema expects, with their definitions. Kept as data so
# an older database can be brought up to date rather than failing at insert.
JOB_COLUMNS = {
    "filename": "TEXT NOT NULL DEFAULT ''",
    "source": "TEXT NOT NULL DEFAULT ''",
    "teams": "TEXT DEFAULT ''",
    "status": "TEXT NOT NULL DEFAULT 'queued'",
    "stage": "TEXT DEFAULT ''",
    "progress": "REAL DEFAULT 0",
    "options": "TEXT DEFAULT '{}'",
    "output_dir": "TEXT DEFAULT ''",
    "error": "TEXT DEFAULT ''",
    "created_at": "TEXT NOT NULL DEFAULT ''",
}


def init_db() -> None:
    with closing(_conn()) as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL, source TEXT NOT NULL,
                teams TEXT DEFAULT '', status TEXT NOT NULL,
                stage TEXT DEFAULT '', progress REAL DEFAULT 0,
                options TEXT DEFAULT '{}', output_dir TEXT DEFAULT '',
                error TEXT DEFAULT '', created_at TEXT NOT NULL)"""
        )

        # CREATE TABLE IF NOT EXISTS does nothing when the table already exists,
        # so a database written by an older version keeps its old columns and
        # every insert fails with "table jobs has no column named ...".
        # Add whatever is missing instead.
        existing = {r["name"] for r in c.execute("PRAGMA table_info(jobs)")}
        for name, decl in JOB_COLUMNS.items():
            if name not in existing:
                log.info("migrating jobs table: adding column %r", name)
                c.execute(f"ALTER TABLE jobs ADD COLUMN {name} {decl}")

        # A job left "running" by a restart can never finish.
        c.execute("UPDATE jobs SET status='error', error='interrupted by restart' "
                  "WHERE status IN ('running','queued')")
        c.commit()


def _row(job_id: int) -> dict | None:
    with closing(_conn()) as c:
        r = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return dict(r) if r else None


def _update(job_id: int, **fields) -> None:
    if not fields:
        return
    sets = ", ".join(f"{k}=?" for k in fields)
    with closing(_conn()) as c:
        c.execute(f"UPDATE jobs SET {sets} WHERE id=?", (*fields.values(), job_id))
        c.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker() -> None:
    from app.pipeline import Options, run

    while True:
        job_id = _queue.get()
        job = _row(job_id)
        if not job or job["status"] != "queued":
            _queue.task_done()
            continue
        _update(job_id, status="running", stage="preprocess", progress=0.0)
        try:
            opts = json.loads(job["options"] or "{}")
            known = type(settings).model_fields
            active = with_overrides(settings, {k: v for k, v in opts.items() if k in known})

            def progress(stage: str, fraction: float, _id=job_id) -> None:
                _update(_id, stage=stage, progress=round(float(fraction), 4))

            result = run(
                Path(job["source"]), active,
                Options(
                    teams_transcript=Path(job["teams"]) if job["teams"] else None,
                    duration=opts.get("duration"),
                    skip_llm=bool(opts.get("no_llm")),
                ),
                progress,
            )
            _update(job_id, status="done", progress=1.0, stage="done",
                    output_dir=result["output_dir"])
            log.info("job %d finished: %s", job_id, result["output_dir"])
        except Exception as exc:
            log.exception("job %d failed", job_id)
            _update(job_id, status="error", error=str(exc)[:2000])
        finally:
            _queue.task_done()


@app.on_event("startup")
def _startup() -> None:
    init_db()
    threading.Thread(target=_worker, name="worker", daemon=True).start()
    hw = hardware()
    log.info("ready on :8080 — %s, ASR %s",
             hw.name, resolve_backend(settings.backend, hw))


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def _output_dir(name: str) -> Path:
    if not name or name.startswith(".") or "/" in name or "\\" in name:
        raise HTTPException(400, "invalid name")
    root = (OUTPUT_DIR / name).resolve()
    try:
        root.relative_to(OUTPUT_DIR.resolve())
    except ValueError:
        raise HTTPException(400, "invalid name") from None
    if not (root / "result.json").is_file():
        raise HTTPException(404, f"no result.json in output/{name}")
    return root


def _read(root: Path) -> dict:
    p = root / "result.json"
    stamp = p.stat().st_mtime
    hit = _cache.get(str(p))
    if hit and hit[0] == stamp:
        return hit[1]
    data = json.loads(p.read_text(encoding="utf-8"))
    _cache[str(p)] = (stamp, data)
    return data


def _media_for(root: Path, meta: dict):
    """Original file when a browser can decode it, else the 16 kHz WAV."""
    # The path comes out of result.json, which lives in a bind mount and is
    # therefore not ours to trust. Without this check, anything that can write
    # a result.json can have the server hand back any file on the box whose
    # extension happens to be playable.
    src = Path(meta.get("source", {}).get("path") or "")
    suffix = src.suffix.lower()
    if suffix in PLAYABLE and src.is_file():
        try:
            src.resolve().relative_to(INPUT_DIR.resolve())
        except ValueError:
            log.warning("source outside input/, serving the WAV instead: %s", src)
        else:
            return src, PLAYABLE[suffix], "source"
    wav = root / "work" / "audio.wav"
    if wav.is_file():
        return wav, "audio/wav", "preprocessed"
    return None


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))


@app.get("/api/config")
def api_config() -> dict:
    hw = hardware()
    from app import llm as llm_mod

    return {
        "version": __version__,
        "device": hw.name,
        "backend": resolve_backend(settings.backend, hw),
        "vram_gb": hw.vram_gb,
        "hf_token": bool(settings.hf_token),
        "llm_enabled": settings.llm_enabled,
        "llm_ready": settings.llm_enabled and llm_mod.available(settings.llm_base_url, 2),
        "llm_model": settings.llm_model,
    }


@app.get("/api/outputs")
def api_outputs() -> list[dict]:
    if not OUTPUT_DIR.exists():
        return []
    rows = []
    for d in sorted(OUTPUT_DIR.iterdir()):
        if not d.is_dir() or not (d / "result.json").is_file():
            continue
        try:
            data = _read(d)
        except Exception:
            continue
        meta = data.get("metadata", {})
        src = meta.get("source", {})
        media = _media_for(d, meta)
        rows.append({
            "name": d.name,
            "title": src.get("name") or d.name,
            "duration": src.get("duration", 0),
            "turns": len(data.get("turns", [])),
            "speakers": len(data.get("speakers", [])),
            "languages": meta.get("languages", {}).get("shares", {}),
            "speakers_from": meta.get("speakers_from", {}).get("source", "?"),
            "llm": meta.get("llm", {}).get("mode", ""),
            "generated_at": meta.get("generated_at", ""),
            "has_video": bool(media and media[2] == "source" and src.get("has_video")),
        })
    rows.sort(key=lambda r: r["generated_at"], reverse=True)
    return rows


@app.get("/api/outputs/{name}/result")
def api_result(name: str, words: int = 0) -> JSONResponse:
    root = _output_dir(name)
    data = dict(_read(root))
    turns = []
    for t in data.get("turns", []):
        t = dict(t)
        if not words:
            t.pop("words", None)
        turns.append(t)
    data["turns"] = turns

    meta = data.get("metadata", {})
    src = meta.get("source", {})
    media = _media_for(root, meta)
    data["name"] = name
    data["title"] = src.get("name") or name
    data["media"] = {
        "available": media is not None,
        "url": f"/api/outputs/{name}/media" if media else None,
        "kind": media[2] if media else None,
        "has_video": bool(media and media[2] == "source" and src.get("has_video")),
        "offset": float(meta.get("preprocess", {}).get("start") or 0.0),
    }
    data["artifacts"] = [n for n in ARTIFACTS if (root / n).exists()]
    return JSONResponse(data)


@app.get("/api/outputs/{name}/media")
def api_media(name: str) -> FileResponse:
    root = _output_dir(name)
    media = _media_for(root, _read(root).get("metadata", {}))
    if media is None:
        raise HTTPException(404, "no playable media")
    path, mime, _ = media
    # Starlette's FileResponse implements HTTP Range (206 + Content-Range),
    # so seeking in a large recording streams instead of loading it.
    return FileResponse(path, media_type=mime)


@app.get("/api/outputs/{name}/download/{artifact}")
def api_download(name: str, artifact: str) -> FileResponse:
    if artifact not in ARTIFACTS:
        raise HTTPException(404, "unknown artifact")
    path = _output_dir(name) / artifact
    if not path.exists():
        raise HTTPException(404, f"{artifact} not produced")
    return FileResponse(path, media_type=ARTIFACTS[artifact],
                        filename=f"{name}-{artifact}")


@app.post("/api/outputs/{name}/rename")
def api_rename(name: str, payload: dict) -> dict:
    root = _output_dir(name)
    from app.teams import clean_name

    # Same treatment as a name arriving from a Teams transcript: this one is
    # typed by hand, but it reaches the same VTT cues and Markdown headings.
    names = {k: clean_name(v) for k, v in (payload.get("names") or {}).items()
             if isinstance(v, str) and clean_name(v)}
    if not names:
        raise HTTPException(400, "no names supplied")
    from app.pipeline import reexport

    try:
        result = reexport(root, names)
    except Exception as exc:
        log.exception("re-export failed for %s", name)
        raise HTTPException(500, str(exc)) from exc
    _cache.pop(str(root / "result.json"), None)
    return {"name_map": result["name_map"]}


# ---------------------------------------------------------------------------
# Upload + jobs
# ---------------------------------------------------------------------------

def _save_upload(f: UploadFile, allowed: set[str]) -> Path:
    """Store an upload in input/, replacing any same-named file atomically.

    Writing straight to the target breaks when something else already has that
    path open — re-uploading a recording that is already in input/ is the
    obvious case — and a failure mid-copy would leave a truncated file that
    looks valid. Write to a temp file beside it and rename instead.
    """
    suffix = Path(f.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"unsupported file type {suffix!r}")

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    target = INPUT_DIR / Path(f.filename).name
    tmp = target.with_name(target.name + ".part")
    try:
        # Counted rather than copied blind: ./input is a bind mount, so an
        # unbounded upload fills the host disk, and the request that starts it
        # need not even come from this page (see _local_only above).
        written = 0
        with tmp.open("wb") as fh:
            while chunk := f.file.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    tmp.unlink(missing_ok=True)
                    raise HTTPException(
                        413, f"file larger than {MAX_UPLOAD_BYTES // 1024 ** 3} GB")
                fh.write(chunk)
        os.replace(tmp, target)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise HTTPException(
            500,
            f"could not save {target.name}: {exc.strerror or exc}. "
            "If that file is open in another program, close it and retry.",
        ) from exc
    return target


@app.post("/api/jobs")
async def api_create(
    media: UploadFile = File(...),
    transcript: UploadFile | None = File(None),
    speakers: str = Form(""),
    lang_mode: str = Form(""),
    no_llm: str = Form(""),
) -> dict:
    # Fail here, not forty minutes in. Without a Teams transcript, speakers come
    # from pyannote, which is gated behind a HuggingFace token -- and diarization
    # is stage four, so the old behaviour was to run ingest, language ID, ASR and
    # alignment on the whole recording before saying "no token".
    has_teams = bool(transcript and transcript.filename)
    if not has_teams and settings.diarize and not settings.hf_token:
        raise HTTPException(
            400,
            "no HF_TOKEN, so speakers cannot be detected. Either upload the "
            "Teams transcript alongside the recording (better anyway -- it "
            "carries the real names), set HF_TOKEN in .env, or set "
            "DIARIZE=false to transcribe without speaker labels.")

    src = _save_upload(media, MEDIA_SUFFIXES)
    teams = _save_upload(transcript, TRANSCRIPT_SUFFIXES) if has_teams else None

    opts = {k: v for k, v in {
        "lang_mode": lang_mode, "speakers": speakers,
    }.items() if v}
    if no_llm:
        opts["no_llm"] = True

    with closing(_conn()) as c:
        cur = c.execute(
            "INSERT INTO jobs (filename, source, teams, status, options, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (src.name, str(src), str(teams) if teams else "", "queued",
             dump_json(opts), _now()),
        )
        c.commit()
        job_id = cur.lastrowid

    _queue.put(job_id)
    log.info("queued job %d: %s%s", job_id, src.name,
             f" + {teams.name}" if teams else "")
    return {"id": job_id}


@app.get("/api/jobs")
def api_jobs() -> list[dict]:
    with closing(_conn()) as c:
        # Named columns, not SELECT *: the table also holds each recording's
        # original path and the raw text of any exception, and this endpoint is
        # unauthenticated.
        rows = c.execute(
            "SELECT id, filename, status, stage, progress, error, created_at "
            "FROM jobs ORDER BY id DESC LIMIT 50").fetchall()
    return [dict(r) for r in rows]


@app.get("/api/jobs/{job_id}/events")
async def api_events(job_id: int) -> StreamingResponse:
    async def stream():
        last = None
        while True:
            job = _row(job_id)
            if not job:
                yield 'data: {"error":"gone"}\n\n'
                return
            snap = (job["status"], job["stage"], round(job["progress"], 3))
            if snap != last:
                last = snap
                yield "data: " + json.dumps({
                    "status": job["status"], "stage": job["stage"],
                    "progress": job["progress"], "error": job["error"],
                    "output": Path(job["output_dir"]).name if job["output_dir"] else None,
                }) + "\n\n"
            if job["status"] in ("done", "error"):
                return
            await asyncio.sleep(0.7)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


def serve() -> None:
    import uvicorn

    # 0.0.0.0 inside the container; compose publishes on 127.0.0.1 only.
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")


if __name__ == "__main__":
    serve()
