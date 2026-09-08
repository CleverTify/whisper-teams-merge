"""Generate a Teams-format VTT from an existing result.json.

Only for exercising the Teams ingestion path when you do not have a matching
recording+transcript pair to hand. It reproduces Teams' actual quirks — CRLF,
cue ids, `<v Name>` tags, HTML entities and a deliberate name variant — so the
parser is tested against the shape of real input, not an idealised one.

    python scripts/make_test_vtt.py <output-dir> "Name A" "Name B" > out.vtt
"""

import json
import sys
from pathlib import Path


def ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def main() -> int:
    root = Path(sys.argv[1])
    if not root.is_absolute():
        root = Path("/work") / root
    names = sys.argv[2:] or ["Jan Novák", "Petra Dvořáková"]

    data = json.loads((root / "result.json").read_text(encoding="utf-8"))
    labels = sorted({t["speaker"] for t in data["turns"]})
    mapping = {lab: names[i % len(names)] for i, lab in enumerate(labels)}

    out = ["WEBVTT", ""]
    for i, t in enumerate(data["turns"]):
        name = mapping[t["speaker"]]
        # Teams really does emit variants of one person's name in a single
        # file; reproduce that so normalisation is exercised.
        if i % 17 == 16 and name == "Jan Novák":
            name = "Janek Novák"
        # ...and HTML-escapes non-ASCII in some exports.
        name_out = name.replace("š", "&#353;")
        out.append(f"c9f1a2b3-0000-4000-8000-00000000{i:04x}/{i}-0")
        out.append(f"{ts(t['start'])} --> {ts(t['end'])}")
        out.append(f"<v {name_out}>{t['text']}</v>")
        out.append("")

    sys.stdout.write("\r\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
