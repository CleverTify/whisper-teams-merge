#!/bin/sh
# Reinstall the torch stack from a specific index, keeping the versions already
# resolved.
#
# Build-time only, called twice from the Dockerfile: once to restore cu128 torch
# on the NVIDIA target, once to restore CPU torch on the universal one. Both
# exist because `pip install -r requirements.txt` resolves whisperx's dependency
# tree and can quietly pull the default PyPI wheel over the one just installed
# from a chosen index.
#
# Why the versions are pinned rather than left bare: `pip install torch` fetches
# the newest release, and torchvision is compiled against a specific torch. Get
# them out of step and the failure is
#
#   partially initialized module 'torchvision' has no attribute 'extension'
#   (most likely due to a circular import)
#
# which names neither torch nor the version mismatch that caused it.
#
# Usage: pin_torch.sh <index-url>
set -e

INDEX="$1"
[ -n "$INDEX" ] || { echo "usage: pin_torch.sh <index-url>" >&2; exit 2; }

version() {
    python - "$1" <<'PY' 2>/dev/null || true
import importlib, sys
try:
    mod = importlib.import_module(sys.argv[1])
except Exception:
    sys.exit(0)
# "2.8.0+cu128" -> "2.8.0": the local version tag names the build, and the
# index we are installing from decides that, not us.
print(getattr(mod, "__version__", "").split("+")[0])
PY
}

set -- --force-reinstall --index-url "$INDEX"

for pkg in torch torchaudio torchvision; do
    v=$(version "$pkg")
    if [ -n "$v" ]; then
        echo "  pinning $pkg==$v"
        set -- "$@" "$pkg==$v"
    fi
done

pip install "$@"
