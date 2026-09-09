# Contributing

## Before you file a bug

Almost every failure in this project is hardware-specific, so a report without
the environment is a report we cannot act on. Run:

```bash
docker compose run --rm app warmup
```

and paste the output. It prints the GPU, the compute capability, whether CUDA is
visible, whether `HF_TOKEN` works and whether the gated pyannote repos have been
accepted. The issue template asks for the same thing.

## Running it

```bash
docker compose up      # add -f docker-compose.cpu.yml if you have no NVIDIA GPU
```

Code in `app/` and `scripts/` is bind-mounted, so an edit takes effect on the
next run without rebuilding. Rebuild only when `Dockerfile` or `requirements.txt`
changes.

## Tests

They need no GPU and no models:

```bash
docker compose run --rm app python tests/test_guardrail.py
docker compose run --rm app python tests/test_teams_names.py
docker compose run --rm app python tests/test_speaker_name_safety.py
docker compose run --rm app python tests/test_turn_invariants.py
```

`test_turn_invariants.py` also checks a real transcript when you point it at
one: `BENCH_OUT=output/<job>`.

## The bar for a change that claims to improve accuracy

This is the part worth reading before opening a PR.

Transcription changes are unusually easy to fool yourself about. Three
*identical* runs of the shipped configuration land 0.02 points apart, and most
plausible-sounding improvements land inside that. Over the life of this project,
seven ideas were measured properly: **six lost, and one won by 0.07 points.**
That is the base rate.

So a PR that says "this improves accuracy" needs a number, produced the same way
for both arms:

1. Score both arms with `scripts/bench/bench_wording.py` on the same recording.
2. Compare them with `scripts/bench/bench_ci.py`, which pairs the two runs
   column by column and gives a confidence interval. A delta whose interval
   spans zero is not a result.
3. Report **both** headline numbers. A change that improves agreement with the
   witness while the Soniox-only control gets worse is a change that learned to
   copy Teams, which is the specific failure that metric exists to catch.
4. Compare A1 to A1, never A1 to A0. Those two streams are built differently and
   differ by 0.74 points even with the merge switched off.

`docs/BENCHMARKS.md` explains the whole method, including the several ways it
has already been got wrong.

A change with no accuracy claim — a bug fix, a language, better errors, docs —
needs none of this. Just say what broke and how you know it is fixed.

## Style

Match the file you are editing. One habit is worth adopting explicitly: comments
here explain *why*, usually with the measurement or the failure that forced the
decision. `# increment counter` is noise; `# 0.60 over-split a two-person call
into five speakers` is why the constant is what it is.

## Privacy

Do not commit audio, transcripts, or benchmark output. `.gitignore` covers
`input/`, `output/`, `cache/` and `bench/` for exactly this reason: those
directories hold real conversations, and a benchmark artifact is still a
recording of people talking. If you want to share a result, share the numbers.
