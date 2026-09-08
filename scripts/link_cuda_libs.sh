#!/bin/sh
# Make the CUDA libraries that ship inside the PyTorch cu128 wheels visible to
# every process in the image, not just to Python once torch has been imported.
#
# Background: we use a ~400 MB `nvidia/cuda:*-base` image instead of the 7.3 GB
# `*-cudnn-runtime` one. The base image has no cuDNN and no cuBLAS. PyTorch's
# cu128 wheels bundle both (as nvidia-cudnn-cu12 / nvidia-cublas-cu12), but
# they land under site-packages/nvidia/*/lib where the dynamic loader does not
# look. PyTorch preloads them for itself; CTranslate2 dlopen()s libcudnn.so.9
# by soname and would fail with "cannot open shared object file".
#
# Registering those directories with ldconfig fixes it for everyone.
set -eu

SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
CONF=/etc/ld.so.conf.d/zz-nvidia-pip.conf
: > "$CONF"

found=0
for dir in "$SITE"/nvidia/*/lib; do
    if [ -d "$dir" ]; then
        echo "$dir" >> "$CONF"
        found=$((found + 1))
    fi
done

if [ "$found" -eq 0 ]; then
    echo "BUILD ABORTED: no nvidia/*/lib directories under $SITE" >&2
    echo "The cu128 wheels should have provided cuDNN and cuBLAS." >&2
    exit 1
fi

echo "registered $found CUDA library directories:"
cat "$CONF"
ldconfig

# CTranslate2 needs cuDNN 9 and cuBLAS 12 specifically. Verify the loader can
# actually resolve them now, rather than finding out mid-transcription.
missing=""
for lib in libcudnn.so.9 libcublas.so.12; do
    if ! ldconfig -p | grep -q "$lib"; then
        missing="$missing $lib"
    fi
done

if [ -n "$missing" ]; then
    echo "BUILD ABORTED: dynamic loader cannot resolve:$missing" >&2
    ldconfig -p | grep -E 'libcudnn|libcublas' >&2 || true
    exit 1
fi

echo "OK: libcudnn.so.9 and libcublas.so.12 resolvable"
