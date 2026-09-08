"""A speaker name is attacker-controlled. Check it cannot break its sinks.

The name in a Teams transcript comes from whoever made the recording, which
often is not the person running this. It then reaches three places with three
different escaping rules: an HTML attribute in the review UI, a `<v NAME>` cue
in the exported VTT, and a `## [00:00] NAME` heading in the Markdown.

The bug this guards against was live: `<v ...>` excludes `>` but not `"`, and
`_clean()` runs `html.unescape()`, so `<v &quot; onfocus=... >` arrived holding
a real quote and escaped the `value="..."` attribute it was rendered into.
Two layers were added — structural characters stripped at the source, quotes
escaped at render — and this checks both, plus the parser in between.

    docker compose run --rm app python tests/test_speaker_name_safety.py
"""

import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.teams import clean_name, parse_vtt  # noqa: E402

Q = chr(34)
INDEX = Path(__file__).resolve().parents[1] / "app" / "web" / "static" / "index.html"

# Names that must survive intact. Stripping these would be its own bug: people
# are called this.
KEEP = [
    "Jan Novák",
    "O'Brien",
    "Anne " + Q + "Annie" + Q + " Smith",
    "Smith & Co",
    "Šárka Dvořáková",
]

# Names that must not come out able to break a cue, a heading or an attribute.
NEUTRALISE = [
    Q + " autofocus onfocus=alert(1) x=" + Q,
    "<script>alert(1)</script>",
    "A<v Someone Else>B",
    "Name\nOn Two Lines",
]


def vtt_file(name: str) -> Path:
    """A one-cue Teams transcript naming `name` as the speaker."""
    path = Path(tempfile.mkdtemp()) / "cue.vtt"
    path.write_text(
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:03.000\n"
        f"<v {name}>Ahoj.</v>\n",
        encoding="utf-8",
    )
    return path


def main() -> int:
    failures = 0

    print("=== names that must survive ===")
    for name in KEEP:
        got = clean_name(name)
        ok = got == name
        failures += not ok
        print(f" {name!r:<34} -> {got!r}{'' if ok else '   <-- WRONG, was altered'}")

    print("\n=== names that must be neutralised ===")
    for name in NEUTRALISE:
        got = clean_name(name)
        ok = "<" not in got and ">" not in got and "\n" not in got
        failures += not ok
        print(f" {name!r:<34} -> {got!r}{'' if ok else '   <-- WRONG'}")

    # The parser is the layer that actually feeds the UI, so exercise it rather
    # than trusting clean_name() is wired in.
    print("\n=== through parse_vtt, the path a real upload takes ===")
    payload = Q + " autofocus onfocus=alert(1) x=" + Q
    cues = parse_vtt(vtt_file(payload))
    speaker = cues[0].speaker if cues else "<no cue parsed>"
    ok = "<" not in speaker and ">" not in speaker
    failures += not ok
    print(f" parsed speaker: {speaker!r}{'' if ok else '   <-- WRONG'}")

    # Second layer: the renderer must escape what the source deliberately keeps.
    # A quote is legitimate in a name and lands inside value="...", so esc() is
    # what stands between it and an attribute break.
    print("\n=== the UI escaper covers quotes ===")
    src = INDEX.read_text(encoding="utf-8")
    m = re.search(r"const esc = s =>.*?\);", src, re.S)
    body = m.group(0) if m else ""
    for ch, label in ((Q, "double quote"), ("'", "single quote"),
                      ("<", "less-than"), ("&", "ampersand")):
        ok = f"'{ch}':" in body or f'"{ch}":' in body
        failures += not ok
        print(f" esc() handles {label:<14}{'yes' if ok else 'NO   <-- WRONG'}")

    print("\n" + ("ALL PASS" if not failures else f"{failures} FAILURE(S)"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
