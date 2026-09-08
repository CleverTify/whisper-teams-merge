# Diarization: who said what, and where it goes wrong

Setting `HF_TOKEN` up is covered in the README. This page is the behaviour
underneath — what the speaker-detection settings actually do, why the
defaults are what they are, and the two failure modes worth recognising.

As in `docs/BENCHMARKS.md`, the measurements come from one two-person Czech
call. The reasoning generalises; the exact thresholds were tuned on it.

### Teams name variants are a guess, and get checked

Teams sometimes writes one participant under two display names in a single
transcript — a second device, a guest account — and that shows up as a phantom
speaker. So near-identical names are treated as a *candidate* merge and then
tested against the conversation, because the strings alone cannot decide it:

```
'Jan Novák' vs 'Janek Novák'   similarity 0.889
```

Those two can easily be different people in one meeting. Janek is a form
of Jan and they share a surname, so the strings cannot separate them.
how they behave: **308 turn swaps, only 11% simultaneous speech** — a dialogue.
One person on two devices is the mirror image, barely trading turns with
themselves, and overlapping heavily where they do because both microphones
caught the same words. So a merge needs that evidence, and every decision is
logged either way:

```
Teams names 'Jan Novák' / 'Janek Novák' look alike: 308 turn swaps
(rate 0.98), 11% simultaneous — two people, keeping both
```

`tests/test_teams_names.py` covers both directions.

### Over-segmentation, and why forcing a speaker count can backfire

pyannote splits one person into several clusters when their acoustics shift
mid-call — a device change or VoIP bitrate adaptation is enough. So after
diarization the speaker embeddings are compared and clusters that are too alike
to be different people are merged (`SPEAKER_MERGE_THRESHOLD`, default 0.60).

Measured on a real two-person Teams call, pyannote reported **five** speakers:

```
        00      01      02      03      04
00   1.000   0.747  -0.108  -0.012   0.779
01   0.747   1.000   0.186   0.163   0.797
02  -0.108   0.186   1.000   0.417   0.037
03  -0.012   0.163   0.417   1.000   0.379
04   0.779   0.797   0.037   0.379   1.000
```

`00/01/04` are mutually 0.75–0.80 — one voice in three clusters. Merging them
gives 3. The threshold sits at 0.60 because the two *genuinely* different
speakers scored 0.417, and merging real speakers is far worse than leaving one
split.

### Telling it how many people were in the room

If you know the count, put it in the **How many people?** box (or `--speakers N`).
That is applied *after* diarization by collapsing clusters on **turn-taking**,
which is far more reliable than constraining pyannote.

Two clusters that are really one person barely alternate with each other — they
only swap when the diarizer flips mid-speech. Two people in conversation
alternate constantly. On the same call:

```
01 <-> 02   76 alternations   rate 0.95   two people talking
02 <-> 00   66 alternations   rate 0.97   two people talking
00 <-> 01   30 alternations   rate 0.44   -> same person, split in two
```

Collapsing on that gives a clean **55.8% / 44.2%** two-speaker split. Note that
embedding similarity alone picks the *wrong* pair here, and
`--min-speakers/--max-speakers 2` is worse still: it fused the two real
speakers into one 83.8% cluster and kept the acoustic outlier as the second,
because the constraint also drives segmentation.

The metric must be computed on merged turns, not raw diarization segments —
pyannote emits ~1,169 short segments for a 59-minute call and those alternate
for reasons unrelated to who is speaking.

Speaker boundaries follow the acoustics, which lands them a word or two off
where sentences end. Changes are snapped to the nearest sentence boundary
within a few words, and speaker runs shorter than `MIN_TURN_SECONDS` are
treated as attribution artifacts rather than someone taking the floor.

