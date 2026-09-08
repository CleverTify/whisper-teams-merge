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
COPY --from=whispercpp /app/build/bin/whisper-cli /usr/local/bin/whisper-cli

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

# pip may silently swap torch while resolving whisperx's dependency tree.
RUN if [ "$APP_BACKEND" = "faster-whisper" ] && \
       ! python -c "import torch,sys; sys.exit(0 if (torch.version.cuda or '').startswith('12.8') else 1)"; then \
      echo ">>> restoring cu128 torch"; \
      pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu128 \
          torch torchaudio; \
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
RUN userdel --remove ubuntu 2>/dev/null || true; \
    useradd --create-home --uid 1000 app \
 && chown -R app:app /cache /work/input /work/output /opt/venv

EXPOSE 8080
# Dispatches: no args -> the web app, a CLI verb -> app.cli, anything else
# verbatim. See docker-entrypoint.sh for why this cannot just be the server.
ENTRYPOINT ["/usr/local/bin/entrypoint"]
