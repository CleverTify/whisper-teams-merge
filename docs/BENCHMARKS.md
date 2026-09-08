# How accuracy was measured, and what it cost to learn

Everything here is from one 92-minute Czech call with two speakers, a
16 kHz 59 kbps recording with a lot of crosstalk. It is one recording, so
read the numbers as "what happened on this audio", not as benchmarks. The
recording itself is not in this repository and neither is any output derived
from it: it is someone's conversation.

The method transfers even though the numbers do not, and the harness is in
`scripts/bench/`. Point it at your own material — see `scripts/bench/corpus.py`.

The short version, if you read nothing else: **seven ideas were measured
properly here. Six lost.** The one that won, won by 0.07 points.

### Measuring wording, when nothing is ground truth

Speaker attribution can be scored because Teams *knows* who spoke. Wording
cannot: Teams writes `vrchní ve 2. Tří`, we write `WhatsApp` where a third
engine hears `odsaď`. Comparing two transcripts measures divergence, never who
is right — so "73% agreement" said nothing about accuracy.

`scripts/bench/bench_wording.py` scores against **two** independent readings: the
Teams transcript and a cached Soniox one. Where those two agree, that agreement
is a witness; if we differ there, we are probably wrong. Fetch a reference once
with `scripts/bench/fetch_soniox.py` (the only thing here that ever leaves the
machine — it deletes the upload afterwards); the scorer only reads the cache, so
benchmarks stay reproducible after the vendor's model moves on.

Two numbers, and you need both:

| number | what it catches |
|---|---|
| **error vs witness** | how often we depart from two-engine consensus |
| **error vs Soniox alone** | whether we are improving, or just copying Teams |

The second exists because our LLM merge reads Teams, so a pipeline that parrots
Teams scores well on a Teams-derived witness by construction. A change is real
only when both improve. Witness better while the control gets worse means the
merge is absorbing Teams, not fixing anything.

It is **not** a WER. The witness covers exactly the audio two engines found
easy, so crosstalk and the low-bitrate stretches are excluded and true error is
higher. `scripts/bench/bench_wording.py` prints that caveat above its own results.

### How this compares to a commercial engine, and what tuning can reach

Judged the same way — each engine scored where the *other two* agree, Teams as
neutral third party — on a 92-minute Czech call:

| | departs from the other two |
|---|---|
| this pipeline | **13.94%** |
| Soniox `stt-async-v5` | **1.59%** |

Six tuning experiments were run against that benchmark and **all six lost**:

| change | error vs reference |
|---|---|
| shipped config | **19.51%** |
| `initial_prompt`, 196 chars | 24.92% |
| `initial_prompt`, 113 chars | 21.73% |
| permissive VAD (0.35 / 0.25) | 20.70% |
| `beam_size` 10 | 19.83% |
| guardrail overlap 0.75 | 19.93% |

`large-v3` at `beam_size 5` with WhisperX's default VAD is already a local
optimum for this audio, and `initial_prompt` is actively harmful — it fixes the
terms you name and compresses everything else. The gap to a commercial engine is
the model and the 16 kHz 59 kbps source, not the configuration, so do not expect
local tuning to close it. Record at a higher bitrate before reaching for
settings.

#### Deltas this small need a paired test, not two percentages

Three *identical* runs of the shipped config land 0.02 pts apart, so any single
comparison inside a few tenths of a point is unreadable by itself. The witness
columns are built from Teams and Soniox alone, which makes them identical across
arms — so `scripts/bench/bench_wording.py` records a per-column right/wrong vector and
`scripts/bench/bench_ci.py` compares two arms **column by column**, resampling whole
60-second blocks because neighbouring words fail together:

```bash
docker compose run --rm app python scripts/bench/bench_ci.py \
    bench/A/wording.json bench/B/wording.json A1
```

It prints the paired delta with a 95% interval, plus the count of columns each
arm alone gets wrong — usually a dozen or two, against ten thousand columns that
carry no information about the change at all. Compare A1 to A1, never A1 to A0:
those two streams are built differently (aligned words versus re-tokenised turn
text) and differ by 0.74 pts even with the merge switched off entirely.

#### Canary-1B-v2 was tested as a replacement, and lost

NVIDIA's Canary-1B-v2 reports 7.86% WER on Czech FLEURS against `large-v3`'s
~12%, which would be a large win if it transferred. It does not. Scored as a
fourth engine on the identical witness columns (`Dockerfile.canary` plus
`scripts/bench/canary_reference.py`, which chunks the audio into overlapping windows
so no word is cut at a boundary):

| raw ASR, same witness columns | vs witness | vs Soniox alone |
|---|---|---|
| Whisper `large-v3` | **13.94%** | **19.51%** |
| Canary-1B-v2 | 21.97% | 26.97% |

Both metrics agree and both are hundreds of times the noise floor, so this is
not a tuning artifact — and Canary's substitutions alone (494 against 344) are
worse on the words it *did* align, which no alignment quirk can explain. FLEURS
is clean read speech; this is a 16 kHz meeting with crosstalk. Canary is not
wired into the pipeline. It is 28x realtime and free to re-check on your own
audio if you have a cleaner recording — the image and the script are still here.

#### A glossary helps, by about as much as you would expect

`LLM_GLOSSARY` is a comma-separated list of names, products and jargon Whisper
has no prior for. The merge may use those spellings to fix a word that is
already in the transcript, and `llm.screen_term_ok()` rejects them as
insertions, so a wrong entry costs a substitution and cannot invent a sentence.

Measured with 12 terms hand-picked from the same call's *own* known errors —
the most favourable case there is, and firmly fitted to the test set:

| | vs witness | vs Soniox alone |
|---|---|---|
| no glossary | 14.48% | 20.03% |
| 12 terms | **14.41%** | **19.98%** |

Both moved the right way, and the paired interval clears zero (−0.07 pts, 95%
CI [−0.14, −0.01]): 10 columns fixed against 3 broken. It is the first change in
this repo to survive a proper paired test — and it is worth 0.07 points at its
ceiling. That is the honest size of the lever. It ships empty, because a
glossary is your vocabulary and not the app's.



#### Feeding on-screen words to the merge is **off by default**

The obvious next step is to hand those OCR'd words to the LLM as correction
candidates — this pipeline writes `do krem` for `dockerem`, and surely the word
is printed right there. Measured on a real screen share, it is not:

```
across 420 frames, OCR never once read:  docker  dockerem  trojúhelníček
                                         teamsech  appku  kanban
```

People *describe* interfaces rather than read them aloud. `trojúhelníček` is
somebody's word for a small triangle icon; `Dockerem` came up while a CRM was on
screen. Screen text and spoken vocabulary barely intersect, zero terms were ever
applied, and the preamble moved the score in both directions across runs —
perturbation, not correction.

`SCREEN_TERMS_IN_PROMPT=true` switches it on. It is plausibly useful for a slide
deck or a documentation walkthrough, where the words really are on screen.
Measure with `scripts/bench/bench_wording.py` before believing it on your material,
and check `screen_terms_applied` in `result.json`: zero means it did nothing.

