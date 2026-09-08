"""Show what the LLM pass did to a transcript, including what was rejected.

    docker compose run --rm app python scripts/show_llm.py output/<name>
"""

import json
import sys
from pathlib import Path


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    if not root.is_absolute():
        root = Path("/work") / root
    data = json.loads((root / "result.json").read_text(encoding="utf-8"))
    meta = data["metadata"]
    llm = meta.get("llm", {})

    print(f"speakers from : {meta.get('speakers_from')}")
    print(f"asr           : {meta.get('asr', {}).get('backend')} / "
          f"{meta.get('asr', {}).get('model')}")
    print(f"llm mode      : {llm.get('mode')}  model={llm.get('model')}")
    print(f"  windows     : {llm.get('windows')}  ({llm.get('failed_windows', 0)} failed)")
    print(f"  turns sent  : {llm.get('turns_sent')}")
    print(f"  improved    : {llm.get('turns_changed')}")
    print(f"  rejected    : {llm.get('turns_rejected')} (guardrail)")

    rejections = llm.get("rejections") or []
    if rejections:
        print("\n=== rejected by guardrail (original kept) ===")
        for r in rejections:
            print(f"  [{r['clock']}]  reason: {r['reason']}")
            print(f"    kept     : {r['original'][:110]}")
            print(f"    discarded: {r['candidate'][:110]}")

    print("\n=== first turns ===")
    for t in data["turns"][:6]:
        print(f"  {t['clock']}  {t.get('name', t['speaker']):<15} {t['text'][:64]}")

    words = sum(len(t.get("words") or []) for t in data["turns"])
    print(f"\nturns={len(data['turns'])}  words with timings={words}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
