# Security

## Reporting a vulnerability

Open a [private security advisory](https://github.com/CleverTify/whisper-teams-merge/security/advisories/new)
rather than a public issue. We will confirm receipt within a week.

## What this software assumes

**The web UI has no authentication, and that is deliberate.** It is a local tool:
`docker-compose.yml` publishes it on `127.0.0.1:8080` only, and the llama.cpp
services declare no host ports at all. The security model is "nothing outside
this machine can reach it".

If you change that binding to `0.0.0.0:8080`, you are publishing an
unauthenticated service that can read every transcript on the box and queue GPU
work. Put it behind a reverse proxy that does authentication, or do not do it.

Two browser-level defences exist because a hostile *web page* can reach
`127.0.0.1` even when the network cannot:

- Requests that change state are refused when they carry a foreign `Origin`,
  and unexpected `Host` headers are rejected outright (DNS rebinding).
- A `Content-Security-Policy` is set on every response.

## Untrusted input

Two inputs come from outside and are treated as hostile:

- **Speaker names in an uploaded Teams transcript.** A name reaches an HTML
  attribute, a VTT `<v NAME>` cue and a Markdown heading. Structural characters
  are stripped when the transcript is parsed, and the UI escapes quotes at
  render. Both layers are covered by `tests/test_speaker_name_safety.py`.
- **Media files.** ffmpeg and tesseract parse them, and a memory-safety bug in a
  demuxer is the most plausible code-execution path in this project. So the work
  runs unprivileged: `docker-entrypoint.sh` is root only long enough to hand the
  bind mounts to uid 1000, then `setpriv` drops to it before anything touches a
  file. The container also mounts its own source read-only — nothing writes to
  `app/` or `scripts/`, and a writable source tree under a long-running service
  is a foothold for no benefit.

**Transcript text is not trusted either** — a recording of someone reading
instructions aloud becomes text in an LLM prompt. The merge is bounded by
length-ratio and token-overlap guardrails (`app/llm.py`), so the worst case is a
corrupted transcript, and the model is local, so nothing can be exfiltrated by
it.

## What we do not consider a vulnerability

- Reaching the API from the same machine. That is the intended interface.
- Anything requiring write access to `./output` or `./cache` — an attacker with
  that already has more than this app can give them.
- Unpinned model downloads. They are HTTPS from Hugging Face; supply-chain
  pinning is a known gap, tracked openly in `THIRD_PARTY.md`.
