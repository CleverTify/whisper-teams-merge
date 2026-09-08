"""Diagnose an over-segmented diarization.

When the detected speaker count does not match reality, this shows whether the
extra "speakers" are real voices or artifacts — how much each one actually
says, how long their turns are, and whether they are scattered through the
recording or clustered in one stretch.

    docker compose run --rm app python scripts/diag_speakers.py output/<name>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def clock(s: float) -> str:
    s = int(max(0, s))
    return f"{s//3600:02d}:{(s//60)%60:02d}:{s%60:02d}"


def main() -> int:
    root = Path(sys.argv[1])
    if not root.is_absolute():
        root = Path("/work") / root
    data = json.loads((root / "result.json").read_text(encoding="utf-8"))
    turns = data["turns"]
    total = sum(t["end"] - t["start"] for t in turns) or 1.0

    print(f"{root.name}: {len(turns)} turns, {len(data['speakers'])} speakers\n")

    print("=== per speaker ===")
    print(f"  {'spk':<12}{'turns':>6}{'talk':>10}{'share':>8}"
          f"{'med turn':>10}{'<1.5s':>8}{'words/turn':>12}")
    rows = {}
    for s in data["speakers"]:
        spk = s["speaker"]
        ts = [t for t in turns if t["speaker"] == spk]
        d = np.array([t["end"] - t["start"] for t in ts])
        w = np.array([len(t["text"].split()) for t in ts])
        rows[spk] = (ts, d)
        print(f"  {spk:<12}{len(ts):>6}{clock(d.sum()):>10}{d.sum()/total*100:>7.1f}%"
              f"{np.median(d):>9.1f}s{100*(d < 1.5).mean():>7.0f}%{w.mean():>12.1f}")

    print("\n=== where each speaker appears (recording split into 10 blocks) ===")
    span = max(t["end"] for t in turns)
    print(f"  {'spk':<12}" + "".join(f"{int(i*span/10/60):>5}m" for i in range(10)))
    for spk, (ts, _) in rows.items():
        cells = []
        for i in range(10):
            lo, hi = i * span / 10, (i + 1) * span / 10
            sec = sum(min(t["end"], hi) - max(t["start"], lo)
                      for t in ts if t["end"] > lo and t["start"] < hi)
            pct = sec / (span / 10) * 100
            cells.append("     " if pct < 1 else f"{pct:>4.0f}%")
        print(f"  {spk:<12}" + "".join(cells))

    print("\n=== adjacency: who speaks right after whom ===")
    pairs: dict[tuple[str, str], int] = {}
    for a, b in zip(turns, turns[1:]):
        if a["speaker"] != b["speaker"]:
            pairs[(a["speaker"], b["speaker"])] = pairs.get((a["speaker"], b["speaker"]), 0) + 1
    for (a, b), n in sorted(pairs.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {a} -> {b}: {n}")

    print("\n=== shortest-turn speakers (likely fragments of a real speaker) ===")
    for spk, (ts, d) in sorted(rows.items(), key=lambda kv: np.median(kv[1][1])):
        if np.median(d) < 2.0 or len(ts) < 15:
            sample = [t["text"][:44] for t in ts[:3]]
            print(f"  {spk}: median {np.median(d):.1f}s, {len(ts)} turns")
            for s in sample:
                print(f"      {s!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
