"""Measure run-to-run variation before trusting any A/B delta.

The merge runs at temperature 0.1, not 0, so two runs of an identical
configuration produce different transcripts. Every tuning result reported so far
was a single run, which means a delta smaller than this spread was never
evidence of anything. Run this once; treat its range as the floor a change must
clear to count.

Only the llm stage is recomputed each time — ingest/asr/align/speakers are
reused — so this measures exactly the noise the merge injects, which is the only
stage an A/B of prompt or guardrail settings can touch.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

# Importable from anywhere: sibling helpers first, then the repo root so
# `app` resolves without relying on PYTHONPATH being set.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import corpus  # noqa: E402




OUT = corpus.out_dir()
FP = OUT / "work" / "fingerprints.json"


def drop_llm_fingerprint() -> None:
    fp = json.loads(FP.read_text(encoding="utf-8"))
    fp.pop("llm", None)
    FP.write_text(json.dumps(fp), encoding="utf-8")


def run_once(n: int) -> dict:
    drop_llm_fingerprint()
    subprocess.run(
        [sys.executable, "-m", "app.cli", "transcribe",
         str(corpus.media()),
         "--teams", str(corpus.teams_vtt())],
        cwd="/work", check=True, capture_output=True, text=True,
    )
    scored = subprocess.run(
        [sys.executable, "scripts/bench_wording.py"],
        cwd="/work", check=True, capture_output=True, text=True,
    ).stdout
    data = json.loads(corpus.results() / "wording.json".read_text(encoding="utf-8"))
    llm = json.loads((OUT / "result.json").read_text(encoding="utf-8"))["metadata"]["llm"]
    row = {
        "run": n,
        "witness_A1": data["witness_error"]["A1"] * 100,
        "soniox_A1": data["soniox_error"]["A1"] * 100,
        "witness_A0": data["witness_error"]["A0"] * 100,
        "turns_changed": llm.get("turns_changed"),
    }
    print(f"  run {n}: witness {row['witness_A1']:.2f}%  "
          f"soniox {row['soniox_A1']:.2f}%  changed {row['turns_changed']}",
          flush=True)
    return row


def main() -> int:
    runs = [run_once(i) for i in range(1, 4)]
    print()
    for key in ("witness_A1", "soniox_A1", "witness_A0"):
        vals = [r[key] for r in runs]
        spread = max(vals) - min(vals)
        sd = statistics.stdev(vals) if len(vals) > 2 else 0.0
        print(f"  {key:<12} min {min(vals):.2f}%  max {max(vals):.2f}%  "
              f"spread {spread:.2f} pts  sd {sd:.2f}")
    corpus.results() / "noise-floor.json".write_text(
        json.dumps(runs, indent=1), encoding="utf-8")
    print("\n  A0 must be identical across runs (ASR is cached and untouched).")
    print("  Any future A/B delta smaller than the A1 spread is not evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
