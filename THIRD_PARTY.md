# Third-party code and models

This project is Apache-2.0. It **redistributes no model weights** — every model
is downloaded from its own source, by you, on first run. So the licences below
constrain what *you* may do with a transcript this produces, not what you may do
with this repository.

Read the first section before deploying commercially. It is the one that bites.

---

## Non-commercial models, and when you hit them

Forced alignment is what turns Whisper's segment timings into per-word timings.
WhisperX picks the aligner by language, and **four of them are CC-BY-NC-4.0**:

| Language | Aligner | Licence |
|---|---|---|
| German | `torchaudio` `VOXPOPULI_ASR_BASE_10K_DE` | **CC-BY-NC-4.0** |
| Spanish | `VOXPOPULI_ASR_BASE_10K_ES` | **CC-BY-NC-4.0** |
| French | `VOXPOPULI_ASR_BASE_10K_FR` | **CC-BY-NC-4.0** |
| Italian | `VOXPOPULI_ASR_BASE_10K_IT` | **CC-BY-NC-4.0** |
| anything with no dedicated aligner | `facebook/mms-300m-1130-forced-aligner` ([`app/align.py`](app/align.py)) | **CC-BY-NC-4.0** |
| English | `WAV2VEC2_ASR_BASE_960H` | MIT-family (LibriSpeech) |
| 36 others incl. Czech, Polish, Dutch, Portuguese | per-language wav2vec2 repos | individually licensed — check yours |

torchaudio states this itself: the VoxPopuli bundles were "originally published
by the authors of VoxPopuli under CC BY-NC 4.0 and redistributed with the same
license" ([VOXPOPULI_ASR_BASE_10K_DE](https://docs.pytorch.org/audio/main/generated/torchaudio.pipelines.VOXPOPULI_ASR_BASE_10K_DE.html)).
MMS is CC-BY-NC-4.0 from Pratap et al., *Scaling Speech Technology to 1,000+
Languages*.

Worth stating plainly, because it is the opposite of what you would guess from a
tool aimed at EU meetings: **four of the largest EU languages after English run
on non-commercial weights by default**, and so does every long-tail language via
the MMS fallback. If that matters to you, supply your own aligner for those
languages, or restrict `LANGUAGE` / `LANG_MODE` to ones that do not use them.
Nothing here stops you — it just is not obvious, and nobody tells you at runtime.

---

## Gated models

Two repositories require the account behind your `HF_TOKEN` to accept their
terms individually, in a browser, before any download works:

- [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)
  — CC-BY-4.0, gated (verified September 2026).
- [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0)
  — gated.

Accepting one does not accept the other. Both are only needed for automatic
speaker detection; supplying a Teams transcript skips diarization entirely and
needs no token at all.

---

## Models downloaded at runtime

| Model | Used for | Licence |
|---|---|---|
| `Systran/faster-whisper-large-v3` | ASR (NVIDIA path) | MIT |
| `ggerganov/whisper.cpp` `ggml-large-v3` | ASR (universal path) | MIT |
| `Qwen/Qwen3-8B-GGUF` | transcript merge / cleanup | Apache-2.0 |
| `Qwen/Qwen3-VL-8B-Instruct-GGUF` + mmproj | screen-recording captions | Apache-2.0 |
| pyannote community-1, segmentation-3.0 | diarization | CC-BY-4.0, gated |
| forced aligners | word timings | see the first section |
| `nvidia/canary-1b-v2` | benchmark comparison only, never in the pipeline | see the model card |

Model files are fetched from Hugging Face over HTTPS and are **not
checksum-pinned**. TLS is the only integrity guarantee.

## Libraries

| Component | Licence |
|---|---|
| WhisperX | BSD-2-Clause |
| faster-whisper | MIT |
| CTranslate2 | MIT |
| pyannote.audio (the code) | MIT |
| PyTorch, torchaudio | BSD-3-Clause (bundles NVIDIA CUDA runtime under NVIDIA's terms) |
| FastAPI, uvicorn, pydantic-settings, python-multipart | MIT / BSD |
| python-docx, soundfile, uroman | MIT / BSD-family |
| llama.cpp, whisper.cpp | MIT |
| tesseract | Apache-2.0 |
| NVIDIA NeMo (benchmark image only) | Apache-2.0 |

## Container images

- `nvidia/cuda:12.8.1-*` — the **NVIDIA Deep Learning Container Licence**, not an
  OSS licence. Fine for building and running locally; read it before pushing an
  image built on it to a public registry.
- `ghcr.io/ggml-org/llama.cpp:server-cuda12`, `ghcr.io/ggml-org/whisper.cpp` — MIT.
- **ffmpeg** comes from Ubuntu's archive, which is the GPL build. Invoking it as
  a separate process does not affect this project's licence, but if you publish
  a built image, that image carries ffmpeg's terms.

## The one thing that leaves your machine

[`scripts/bench/fetch_soniox.py`](scripts/bench/fetch_soniox.py) uploads audio to
Soniox to produce a benchmark reference. It is a manual, standalone script; the
pipeline never calls it, and it needs a `SONIOX_API_KEY` you supply in the
environment. It deletes the upload and the job afterwards. Everything else in
this project stays local — see the README.

---

Corrections welcome, and please send them: these are statements about other
people's work, and any of them can go stale when an upstream model card changes.
