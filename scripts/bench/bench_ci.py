"""Is the difference between two arms bigger than resampling luck?

Every A/B in this repo so far has been two headline percentages and a judgement
call. That was enough for the large effects (Canary is 8 points worse; nobody
needs statistics for that) and useless for the small ones — the screen arms
landed 0.02 pts apart, which is exactly the spread three *identical* runs
produced, and I only knew that because I had happened to run the same config
twice.

This does the comparison properly:

* **Paired.** The witness set is built from Teams and Soniox alone, so it is the
  same columns in the same order for every arm. Comparing per-column error
  vectors removes all the variance that comes from which columns are hard,
  which is most of it.
* **Blocked by minute.** Neighbouring words are not independent — one botched
  passage fails twenty columns together. Resampling individual columns would
  understate the interval several-fold, so whole 60-second blocks are resampled.

Usage:
  bench_ci.py A/wording.json B/wording.json [stream]      # default stream: A1
"""

import json
import random
import sys
from pathlib import Path

# Importable from anywhere: sibling helpers first, then the repo root so
# `app` resolves without relying on PYTHONPATH being set.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


BLOCK = 60.0        # seconds per resampling block
DRAWS = 5000
SEED = 20260908     # fixed, so a re-run of the same inputs gives the same CI


def load(path: Path, stream: str) -> tuple[list[float], list[int]]:
    d = json.loads(path.read_text(encoding="utf-8"))
    if "witness_flags" not in d:
        raise SystemExit(
            f"{path} predates per-column flags — re-run bench_wording.py for "
            f"that arm before comparing it")
    flags = d["witness_flags"].get(stream)
    if flags is None:
        raise SystemExit(f"{path} has no stream {stream!r}")
    # Written as a "0101..." string; older files used a list of ints.
    return d["witness_t"], [int(c) for c in flags]


def blocks(times: list[float]) -> list[list[int]]:
    out: dict[int, list[int]] = {}
    for i, t in enumerate(times):
        out.setdefault(int(t // BLOCK), []).append(i)
    return [out[k] for k in sorted(out)]


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    pa, pb = Path(sys.argv[1]), Path(sys.argv[2])
    stream = sys.argv[3] if len(sys.argv) > 3 else "A1"

    ta, fa = load(pa, stream)
    tb, fb = load(pb, stream)
    if ta != tb:
        # Different witness columns means the two arms were scored against
        # different audio or a different reference; the delta would be
        # meaningless and quietly plausible, which is the worst kind of wrong.
        raise SystemExit(
            f"witness columns differ ({len(ta)} vs {len(tb)}) — these two arms "
            f"are not comparable")

    groups = blocks(ta)
    n = len(fa)
    ra, rb = sum(fa) / n, sum(fb) / n

    rng = random.Random(SEED)
    deltas = []
    for _ in range(DRAWS):
        ea = eb = tot = 0
        for _ in range(len(groups)):
            g = groups[rng.randrange(len(groups))]
            for i in g:
                ea += fa[i]
                eb += fb[i]
            tot += len(g)
        deltas.append((eb - ea) / tot)
    deltas.sort()
    lo = deltas[int(0.025 * DRAWS)]
    hi = deltas[int(0.975 * DRAWS)]
    point = rb - ra

    print(f"stream {stream}, {n} witness columns in {len(groups)} "
          f"{BLOCK:.0f}s blocks, {DRAWS} paired resamples\n")
    print(f"  A  {pa.parent.name:<28} {ra*100:6.2f}%")
    print(f"  B  {pb.parent.name:<28} {rb*100:6.2f}%")
    print(f"\n  B - A  {point*100:+6.2f} pts   95% CI "
          f"[{lo*100:+.2f}, {hi*100:+.2f}]")

    # Disagreements only. Columns both arms get right or both get wrong carry no
    # information about which is better, and there are thousands of them.
    b_only = sum(1 for x, y in zip(fa, fb) if y and not x)
    a_only = sum(1 for x, y in zip(fa, fb) if x and not y)
    print(f"  columns B alone gets wrong: {b_only}   "
          f"A alone gets wrong: {a_only}")

    if lo <= 0 <= hi:
        print("\n  VERDICT: not resolvable. The interval spans zero — this "
              "change is\n           indistinguishable from running the same "
              "config twice.")
    elif hi < 0:
        print(f"\n  VERDICT: B is better by {-point*100:.2f} pts, and the "
              f"interval clears zero.")
    else:
        print(f"\n  VERDICT: B is worse by {point*100:.2f} pts, and the "
              f"interval clears zero.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
