# Multilingual transcription + diarization.
#
# No `# syntax=` directive on purpose: it makes BuildKit fetch an external
# frontend image on every build, which on a throttled link to Docker Hub stalls
# for 20+ minutes before the build even starts.
#
# Two targets from one file:
#   BACKEND=nvidia    faster-whisper on CUDA 12.8   (fastest; needs an NVIDIA GPU)
#   BACKEND=universal whisper.cpp on Vulkan -> CPU  (AMD, Intel, Apple, CPU)
#
# CUDA 12.8 is required rather than preferred. Blackwell cards (sm_120 —
# the RTX 50 series) have no kernels in any earlier CUDA, so an older base
# image does not run slowly on them, it does not run at all. Older GPUs are
# perfectly happy on 12.8, so there is nothing to trade off.

# Declared before any FROM so it is a *global* ARG. An ARG declared after a
# FROM belongs to that stage only and resolves to empty in a later FROM.
ARG BACKEND=nvidia

# Prebuilt Vulkan whisper.cpp — copying the binary avoids compiling it here.
FROM ghcr.io/ggml-org/whisper.cpp:main-vulkan AS whispercpp

# --------------------------------------------------------------------------
FROM nvidia/cuda:12.8.1-base-ubuntu24.04 AS base-nvidia
ENV TORCH_INDEX=https://download.pytorch.org/whl/cu128 \
    APP_BACKEND=faster-whisper

FROM ubuntu:24.04 AS base-universal
ENV TORCH_INDEX=https://download.pytorch.org/whl/cpu \
    APP_BACKEND=whisper.cpp

# --------------------------------------------------------------------------
FROM base-${BACKEND} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# tesseract reads on-screen text from screen recordings. The Czech
# traineddata is not optional: without it tesseract silently falls back to
# English and returns confident garbage for Czech UI text, which would then
# be fed to the merge as if it were evidence.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev build-essential \
        ffmpeg curl ca-certificates libsndfile1 \
        libvulkan1 libgomp1 \
        tesseract-ocr tesseract-ocr-ces tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# whisper.cpp CLI is present in both targets so `--backend whisper.cpp` works
# even on an NVIDIA build (useful for comparison and as a fallback).
#
# The whole bin/ directory, not just the binary: upstream now builds whisper.cpp
# against libwhisper.so and libggml*.so, which sit beside the executable. Copying
# `whisper-cli` alone produced a binary that exits 127 the moment it runs — and
# 127 is what a missing *file* looks like too, so it reads as "not installed".
# That went unnoticed because the NVIDIA path never invokes it.
COPY --from=whispercpp /app/build/bin/ /opt/whispercpp/bin/
RUN printf '/opt/whispercpp/bin\n' > /etc/ld.so.conf.d/whispercpp.conf \
 && ldconfig \
 && ln -sf /opt/whispercpp/bin/whisper-cli /usr/local/bin/whisper-cli

# Ubuntu marks the system Python externally managed (PEP 668); build in a venv.
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --upgrade pip wheel setuptools

# Torch first, from the index matching the target. On the nvidia target this
# must be cu128 — PyPI's default wheel has no sm_120 kernels and every CUDA op
# would fail with "no kernel image is available for execution on the device".
RUN pip install --index-url "$TORCH_INDEX" torch torchaudio

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# pip may silently swap torch while resolving whisperx's dependency tree, and it
# swaps in whatever PyPI serves by default — which on Linux is the CUDA build.
# Both targets therefore have to check they still have the torch they asked for.
#
# Both reinstalls pin the version already resolved. Asking for a bare `torch`
# fetches the newest one, and a torch that no longer matches the torchvision
# beside it fails as "partially initialized module 'torchvision' has no
# attribute 'extension'" — a circular-import message that says nothing about
# the actual cause. Same flavour, same version, different index.
COPY scripts/pin_torch.sh /tmp/pin_torch.sh
RUN if [ "$APP_BACKEND" = "faster-whisper" ] && \
       ! python -c "import torch,sys; sys.exit(0 if (torch.version.cuda or '').startswith('12.8') else 1)"; then \
      echo ">>> restoring cu128 torch"; \
      sh /tmp/pin_torch.sh https://download.pytorch.org/whl/cu128; \
    fi

# The mirror image, and it was missing: on the universal target pip pulled the
# default PyPI wheel back over the CPU one, so a build meant for AMD, Intel,
# Apple and CPU-only machines shipped gigabytes of CUDA that could never be
# used. Silent, because CPU inference works perfectly well with a CUDA torch.
RUN if [ "$APP_BACKEND" = "whisper.cpp" ] && \
       python -c "import torch,sys; sys.exit(0 if torch.version.cuda else 1)"; then \
      echo ">>> replacing CUDA torch with the CPU build of the same version"; \
      sh /tmp/pin_torch.sh https://download.pytorch.org/whl/cpu; \
    fi

# The cu128 wheels ship cuDNN/cuBLAS under site-packages/nvidia/*/lib, where the
# dynamic loader does not look. PyTorch preloads them for itself; CTranslate2
# dlopen()s libcudnn.so.9 by soname and would fail without this.
COPY scripts/link_cuda_libs.sh /tmp/link_cuda_libs.sh
RUN if [ "$APP_BACKEND" = "faster-whisper" ]; then sh /tmp/link_cuda_libs.sh; fi

COPY scripts/verify_stack.py /tmp/verify_stack.py
RUN python /tmp/verify_stack.py

# HOME must be set explicitly. setpriv changes the uid but not the
# environment, so without this the app user inherits root's HOME and
# pyannote's first write to ~/.pyannote/database.yml dies with EACCES --
# during model prefetch, where it reads as a model problem rather than a
# permissions one.
ENV HOME=/home/app \
    HF_HOME=/cache/hf \
    TORCH_HOME=/cache/torch \
    XDG_CACHE_HOME=/cache/xdg \
    MPLCONFIGDIR=/cache/mpl \
    OMP_NUM_THREADS=8 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=/work

# Links the GHCR package back to this repository, so the package page shows the
# README and the repo sidebar shows the package. It does NOT make the package
# public — container package visibility is UI-only, there is no REST endpoint
# for it, and GHCR defaults to private even under a public repo.
LABEL org.opencontainers.image.source="https://github.com/CleverTify/whisper-teams-merge" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.description="Local meeting transcription with speaker diarization, merged with a Microsoft Teams transcript by a local LLM."

WORKDIR /work
COPY app /work/app
COPY scripts /work/scripts
COPY tests /work/tests
COPY docker-entrypoint.sh /usr/local/bin/entrypoint
RUN chmod +x /usr/local/bin/entrypoint && mkdir -p /cache /work/input /work/output

# The account the work actually runs as. The entrypoint starts as root only long
# enough to hand it the bind mounts, then drops to this uid with setpriv — see
# docker-entrypoint.sh. ffmpeg and tesseract parse fully untrusted media, which
# is the most plausible code-execution path here, and none of it needs uid 0.
#
# uid 1000 specifically, so transcripts written into ./output belong to the
# first real account on a typical Linux host rather than arriving root-owned.
# Ubuntu 24.04 images already ship a placeholder user on that uid and useradd
# exits 4 rather than reusing it, so retire it first.
# Note what is NOT chowned: /opt/venv. On a copy-on-write filesystem `chown -R`
# rewrites every file it touches into a new layer, and doing that to the venv
# added 8.3 GB to this image for no benefit — the app user only ever reads it,
# and it is already world-readable. The three directories below are empty at
# build time, so chowning them costs nothing.
RUN userdel --remove ubuntu 2>/dev/null || true; \
    useradd --create-home --uid 1000 app \
 && chown -R app:app /cache /work/input /work/output

EXPOSE 8080
# Dispatches: no args -> the web app, a CLI verb -> app.cli, anything else
# verbatim. See docker-entrypoint.sh for why this cannot just be the server.
ENTRYPOINT ["/usr/local/bin/entrypoint"]
