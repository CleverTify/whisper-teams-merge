"""Time one merge window and show what the model actually returns.

A window went from ~8s to ~130s when the prompt started carrying both readings
per line. The prompt only doubled, so the cost is more likely in what is being
generated than in what is being sent — which only the raw reply can settle.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path


from app import llm as llm_mod, teams as teams_mod  # noqa: E402

# Importable from anywhere: sibling helpers first, then the repo root so
# `app` resolves without relying on PYTHONPATH being set.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import corpus  # noqa: E402

from app.config import settings  # noqa: E402
from app.diarize import build_turns  # noqa: E402



OUT = corpus.out_dir()
VTT = corpus.teams_vtt()


def main(which: int = 3) -> None:
    st = settings
    raw = json.loads((OUT / "work" / "align.json").read_text(encoding="utf-8"))
    base = raw["segments"] if isinstance(raw, dict) else raw
    cues = teams_mod.load(VTT)
    segs = teams_mod.assign_speakers(copy.deepcopy(base), cues)
    turns = build_turns(segs, merge_gap=1.0, smooth=3,
                        min_turn_seconds=0.7, snap_window=3)

    groups = llm_mod.windows(turns, st.llm_window_seconds)
    idxs = groups[which]
    print(f"window {which}: {len(idxs)} turns")

    pairs = []
    for n, i in enumerate(idxs, 1):
        theirs = teams_mod.aligned_text(turns[i].words or [], cues)
        pairs.append(f"[{n}] A: {turns[i].text}\n    B: {theirs or '(none)'}")
    user = (f"{len(idxs)} lines, each with both readings:\n\n" + "\n".join(pairs)
            + f"\n\nReturn exactly {len(idxs)} corrected lines, "
              f"numbered [1]..[{len(idxs)}].")

    print(f"prompt chars: {len(user)} (~{len(user)//3} tokens)\n")

    for label, max_tokens in (("as shipped", 2048), ("tight cap", 700)):
        t0 = time.time()
        try:
            resp = llm_mod._post(
                st.llm_base_url,
                {
                    "model": st.llm_model,
                    "messages": [
                        {"role": "system", "content": llm_mod.SYSTEM_MERGE},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.1,
                    "max_tokens": max_tokens,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                st.llm_timeout,
            )
        except Exception as exc:
            print(f"  {label}: FAILED after {time.time()-t0:.1f}s ({exc!r})")
            continue
        dt = time.time() - t0
        reply = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {})
        print(f"  {label}: {dt:.1f}s | prompt {usage.get('prompt_tokens')} "
              f"-> completion {usage.get('completion_tokens')} tokens "
              f"| finish={resp['choices'][0].get('finish_reason')}")
        print(f"    reply chars: {len(reply)}")
        print("    first 400 chars:")
        print("      " + reply[:400].replace("\n", "\n      "))
        print()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
