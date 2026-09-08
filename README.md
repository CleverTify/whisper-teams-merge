# Transcribe

Multilingual meeting transcription with speaker diarization. Upload a recording;
optionally add the Teams transcript and it will use Teams' real speaker names
and have a local LLM merge both transcripts into a better one.

Everything runs locally. Nothing leaves the machine.

![The web UI on first run: a drop zone for a recording, and an empty transcript
list. The header reports the detected GPU, the ASR backend and the merge model.](docs/img/ui.png)

## Run it

```bash
cp .env.example .env && docker compose up
```

Then open **http://127.0.0.1:8080**.

You need Docker, and for the NVIDIA path the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
— without it `docker compose up` fails with `could not select device driver ""
with capabilities: [[gpu]]`, which is not a hint anyone enjoys receiving.

Budget **~25 GB of disk**: about 10 GB of image, and ~15 GB of models fetched on
first start into `./cache` — Whisper large-v3, the pyannote diarizer, the merge
LLM, and the vision model that describes screen recordings. After that it never
needs the network again.

Set `HF_TOKEN` in `.env` if you want automatic speaker detection — see
[Diarization](#diarization). You do not need it when you upload a Teams
transcript, which is the better path anyway: Teams knows the real names.

Check the machine before committing an hour to a long recording:

```bash
docker compose run --rm app warmup
```

That reports the GPU and its compute capability, whether CUDA is visible in the
container, whether `HF_TOKEN` works, and whether the two gated pyannote
repositories have been accepted. It is also the output to paste into a bug
report.

## Hardware

One line in `.env` selects the build:

```bash
COMPOSE_PROFILES=nvidia,screen     # NVIDIA GPU — faster-whisper on CUDA 12.8
COMPOSE_PROFILES=universal        # AMD / Intel / Apple Silicon / CPU
```

| | ASR | Diarization |
|---|---|---|
| **NVIDIA** | faster-whisper (CTranslate2, fp16) | pyannote on GPU |
| **AMD / Intel** | whisper.cpp via Vulkan | pyannote on CPU |
| **Apple Silicon** | whisper.cpp on **CPU** | pyannote on CPU |
| **CPU only** | whisper.cpp | pyannote on CPU |

Two things worth knowing rather than discovering later:

- **Apple Silicon gets no GPU in Docker.** Apple's Hypervisor.framework exposes
  no virtual GPU, and Apple's own Container runtime lists passthrough only on
  its roadmap. On a Mac this runs on CPU — correct, just slow.
- **AMD needs whisper.cpp** because CTranslate2 has no ROCm backend at all, so
  faster-whisper cannot drive an AMD GPU. Vulkan covers AMD and Intel.

## How it works

```
preprocess → language ID → ASR → speakers → LLM → alignment → export
                                    │         │
                    Teams transcript ┘         ├─ merge (with Teams)
                    or pyannote                └─ cleanup (without)
```

**With a Teams transcript** — speakers come **strictly** from it, so you get
real names instead of `SPEAKER_00`, pyannote never runs, and the LLM sees two
independent readings of the same audio and picks the better wording per turn.

Both of those work by matching **words**, not clocks. Teams' cue boundaries do
not line up with our turns, and where two people talk over each other — 14% of
airtime on a real call — both have cues covering the same seconds. Picking by
time overlap then flips speaker from word to word and shreds one person's
sentence across two:

```
Jan Novák        ...No jako já tam
Petra Dvořáková  dám mobil, že
Jan Novák        jo, protože když budeme preferovat
Petra Dvořáková  mobil, tak bude asi lepší, aby tam byl mobil.
```

Since both sides transcribed the same audio, their word sequences align even
where the clocks disagree. Each of our words inherits its match's speaker, and
each turn is shown exactly its own Teams text. Measured on that call:

| | by time | by words |
|---|---|---|
| speaker accuracy | 86.8% | **95.9%** |
| speaker changes cutting a sentence | 26.5% | **0.21%** |
| disagreements with Teams the LLM resolved | 19% | **63%** |

Teams' own sentence-cut rate is 1%, so ours is now better than the source it
learns speakers from.

The same alignment feeds the LLM. Each line is shown *its own* Teams reading
rather than one smeared block per window, filtered to that turn's speaker —
without the filter 7.8% of the quoted text belonged to whoever spoke next, and
the model would occasionally merge their words in.

`scripts/bench/bench_speakers.py` scores a run; `scripts/bench/ab_speakers.py` A/Bs a change
against cached ASR in seconds instead of a full pipeline run.

One thing this does **not** chase: Whisper writes standard Czech (`být`,
`nějaký`, `mají`) where speakers used the colloquial forms Teams preserves
(`bejt`, `nějakej`, `maj`). Those dominate the raw disagreement count and are a
matter of register, not accuracy, so the merge prompt is told to keep our
spelling and use Teams mainly for names and technical terms.

**Without one** — pyannote detects speakers and the LLM cleans up our
transcript on its own.

Either way the output is a transcript with diarization.

### How good is it, and how do you know

Wording cannot be scored against a ground truth, because there isn't one:
comparing two transcripts measures divergence, not correctness. So this is
scored against **two** independent readings — the Teams transcript and a
cached Soniox one — on the columns where those two agree, with a second
control that catches a pipeline "improving" by simply copying Teams.

On that 92-minute Czech call, each engine judged where the *other two*
agree:

| | departs from the other two |
|---|---|
| this pipeline | **13.94%** |
| Microsoft Teams | **10.03%** |
| Soniox `stt-async-v5` | **1.59%** |

That gap is the model and a 16 kHz 59 kbps source, not the configuration.
Six tuning experiments were run against it and all six lost; a seventh, a
glossary, won by 0.07 points. Whisper `large-v3` at `beam_size 5` with
WhisperX's default VAD is already a local optimum for this audio, and
`initial_prompt` is actively harmful.

**[docs/BENCHMARKS.md](docs/BENCHMARKS.md)** has the method, every arm, the
0.02-point noise floor, why Canary-1B-v2 lost by 8 points despite its FLEURS
figure, and the two occasions the measurement itself turned out to be the
thing that was wrong.

### Screen recordings: what was on screen

Upload a video with a screen share and the transcript gains an on-screen track,
so a reader can follow a demo instead of guessing what "klikni sem" referred to:

```
> *screen [00:35:23]:* The user is viewing the "Lidé" (People) list in the
  invoicing app, with "brn" typed into the customer search field.

## [00:35:26] Jan Novák
Tady vidíš, že to filtruje hnedka.
```

Frames are sampled at scene changes, read by tesseract, and described by a local
vision model (`llm-vision`, Qwen3-VL-8B). The `screen` profile is in the default
`COMPOSE_PROFILES` because without it a video job silently produces no
annotations — handled, but the app quietly cannot do what it claims. Audio-only
work pays nothing for it: the model evicts itself when idle and loads only when
a video arrives. On a 92-minute 1080p recording that is 420 frames and about 15
minutes on top of the transcription.

Gated entirely on the video track ffprobe already detects, so **audio-only input
is untouched** — the stage records `no-video` and does nothing.

**The vision service is checked, not trusted.** If the multimodal projector
fails to load, llama.cpp still serves a perfectly healthy *text-only* model: it
answers `/v1/models` with 200, accepts a request carrying an image, silently
ignores the image, and returns a fluent invented description. Every stage
downstream then succeeds and the captions are fiction. So before captioning, the
pipeline renders a random nonce into an image and refuses to continue unless the
model reads it back.

### The LLM will not be trusted blindly

A model told to "improve a transcript" will happily rewrite, merge, answer or
translate it. So every line it returns is checked against the line it replaces —
length ratio and token overlap — and anything that drifts is **discarded in
favour of the original**, with the rejection recorded in `result.json`. A silent
rewrite would be far worse than a few uncorrected ASR errors.

Turn it off per job with the **skip LLM** checkbox, or globally with
`LLM_ENABLED=false`.

### If the transcript looks under-corrected, check the LLM's speed

A slow LLM does not fail — it returns **short, truncated replies**, and every
line that never comes back is silently kept as-is. So throughput is a quality
signal, not just a speed one. The merge records `median_tokens_per_second` and
`lines_missing` in `result.json` and warns when either goes wrong:

```
LLM was degraded: 9/55 window(s) under 25 tok/s, 214 line(s) never came back.
```

Two causes seen in practice:

- **A llama.cpp server left up for hours** drifts from ~80 tok/s to ~7.
  `docker compose restart llm` takes ten seconds and fixes it.
- **The GPU sitting in a low power state.** Check it under load:

  ```bash
  docker compose exec llm nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,pstate --format=csv
  ```

  `pstate P8` with the memory clock at a fraction of its maximum while
  utilisation is 99% means the card is being held down by the host's power
  settings, not by the workload — and everything, ASR included, runs several
  times slower. It is a Windows/driver setting, not something the container can
  change.

A window that times out is now retried as halves rather than abandoned, so a
slow machine loses wording improvements gradually instead of dropping whole
minutes of the transcript.

### The LLM gets out of the way while Whisper works

ASR and the merge never run at the same time, but the model would otherwise sit
on ~5.9 GB of 12 for the whole transcription. The `llm` service runs with
`--sleep-idle-seconds` (60 by default, `LLM_SLEEP_IDLE_SECONDS` to change), so it
evicts itself once ASR starts and hands the memory back:

```
t=30s   5888 MiB   loaded
t=60s    172 MiB   asleep — ASR has the card to itself
t=180s  7027 MiB   ASR at full stretch
t=420s  6111 MiB   awake again for the merge
```

Waking costs ~60s. The pipeline pays it once, deliberately, before the window
loop — inside the loop it would look like a stalled first window, trip the
split-retry, and skew the throughput figures the degradation warning relies on.
`/v1/models` answers 200 while asleep, so only a real completion can tell the
difference, and `wake_seconds` is recorded in `result.json`.

Accuracy is unchanged by this, as it must be: sentence-cut rate identical at
0.21%, resolved disagreements 63% → 65%, zero failed windows either way.

### Alignment runs last

Word timings are computed *after* the LLM, not before. Corrected text no longer
matches the original timings, and stale ones would desynchronise the player's
word highlighting from what is on screen.

## Multi-language

Built for the EU market, so a single recording may contain several languages.
Language ID samples ~30 speech-dense windows across the **whole file** rather
than trusting the first 30 seconds — the standard Whisper failure on long
meetings. Contiguous stretches become *language runs*, each transcribed and
aligned with its own models.

On a real Czech/Slovak council recording this mattered: the file opens with
6 minutes of Slovak, so first-30-seconds detection would have transcribed all
81 minutes as Slovak when **76% was actually Czech**.

```
LANG_MODE=auto    # one language if it covers ≥85% of speech
LANG_MODE=multi   # full code-switching
LANG_MODE=single  # force LANGUAGE
```

## Diarization

Uses `pyannote/speaker-diarization-community-1` (CC-BY-4.0 but gated). Put a
token in `.env` and accept the terms on both:

- https://huggingface.co/pyannote/speaker-diarization-community-1
- https://huggingface.co/pyannote/segmentation-3.0

Two behaviours are worth knowing before you touch the settings: Teams
sometimes writes one person's name two ways in a single transcript, and
pyannote over-splits by default so forcing a speaker count often makes
things worse rather than better. Both are covered, with the measurements
behind the defaults, in **[docs/DIARIZATION.md](docs/DIARIZATION.md)**.

## Outputs

Written to `output/<name>/`:

| File | Contents |
|---|---|
| `transcript.md` | Speaker-labelled turns with timestamps |
| `transcript.vtt` | Word-timed subtitles with speaker tags |
| `result.json` | Every word with time/speaker/language, plus full provenance — models, LLM changes and rejections, hardware |
| `removed.jsonl` | Audit log of every filtered segment and why |

[docs/example-output.md](docs/example-output.md) shows what all four look
like, with invented dialogue — no real transcript ships in this repo.

Folder names are slugged, so
`Team sync-20260804_173547-Meeting Recording.mp4` becomes
`output/team-sync-20260804-173547-meeting-recording/`. The original
name is kept in `result.json` and shown in the UI.

## Command line

```bash
docker compose run --rm app transcribe input/meeting.mp4 --teams input/meeting.vtt
docker compose run --rm app warmup
docker compose run --rm app reexport output/meeting --name SPEAKER_00="Jan Novák"
```

Useful flags: `--lang-mode multi`, `--speakers 2`, `--no-llm`,
`--duration 120` (quick test), `--force`, `--backend whisper.cpp`.

`--duration` and `--start` write to `<name>-trial/`, so a quick check can never
replace a finished transcript — `--force` ignores the stage fingerprints that
would otherwise protect it.

## A note on recording quality

The model cannot recover information the recording never captured. A call
recorded by pointing a laptop mic at a phone on speaker measured **79% of its
energy below 3.4 kHz** — the band that carries consonants was gone, and no ASR
system can fix that.

Record digitally at source. A Teams recording of the same participants scored
noticeably better on every speaker. If you have the choice, that single change
beats every model-side improvement combined.

## Privacy

Nothing leaves the machine at run time. The models are downloaded once, from
Hugging Face, and everything after that — ASR, diarization, the merge LLM, the
vision model — runs in containers on your hardware with no outbound calls.

The one exception is opt-in, manual, and not part of the pipeline:
`scripts/bench/fetch_soniox.py` uploads audio to a cloud API to produce a
benchmark reference. It needs an API key you supply, the pipeline never calls
it, and it deletes the upload afterwards.

The web UI is **unauthenticated by design** and published on `127.0.0.1` only.
If you change that binding, read [SECURITY.md](SECURITY.md) first.

`input/`, `output/`, `cache/` and `bench/` are gitignored, because all four hold
recordings of people talking.

## Contributing

Bug reports want the `warmup` output — see [CONTRIBUTING.md](CONTRIBUTING.md).

If you plan to propose an accuracy improvement, read
[docs/BENCHMARKS.md](docs/BENCHMARKS.md) first. Of the seven ideas measured
properly here, six lost, so the bar is a paired comparison with a confidence
interval rather than a plausible argument. `scripts/bench/bench_ci.py` does that
part for you.

## Licence

[Apache-2.0](LICENSE). Copyright 2026 CleverTify s.r.o.

No model weights are redistributed — every model is fetched from its own source
on first run — so the licences that bind *you* are theirs, not ours.
[THIRD_PARTY.md](THIRD_PARTY.md) lists them, and leads with the one that
surprises people: the forced aligners for German, Spanish, French and Italian,
and the fallback used for every language without a dedicated one, are
**CC-BY-NC-4.0 — non-commercial**.
