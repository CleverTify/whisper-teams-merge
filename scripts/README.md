# scripts/

Three kinds of thing live here, and only the first is meant for everyday use.

Every command below works from a clone with no arguments beyond the ones shown:

```bash
docker compose run --rm app python scripts/<name>.py <args>
```

## Useful with your own transcripts

| Script | What it does |
|---|---|
| `show_llm.py` | What the merge changed, and what it rejected, in one output directory. The first place to look when a transcript seems under-corrected. |
| `show_llm_changes.py` | The same, side by side with the Teams reading of each line. |
| `validate_outputs.py` | Checks a finished job's artifacts agree with each other — words against text, VTT against JSON. |
| `post_job.py` | Queue a job against the running web app from a shell. |

## Diagnostics, for when speakers come out wrong

| Script | What it does |
|---|---|
| `diag_speakers.py` | Where speaker labels change, and how confident each change was. |
| `probe_diarization.py` | Re-runs diarization at different speaker counts on cached audio. |
| `tune_clustering.py` | Sweeps the clustering threshold. `docs/DIARIZATION.md` explains why 0.80 and why forcing a count often backfires. |
| `make_test_vtt.py` | Synthesises a Teams-shaped `.vtt` from a finished job, for exercising the Teams path without a real transcript. |

## Build-time only

`verify_stack.py` and `link_cuda_libs.sh` run inside `Dockerfile`. Nothing calls
them afterwards.

## bench/

The measurement harness. It scores a transcript against two independent
readings and compares two runs with a paired confidence interval —
`docs/BENCHMARKS.md` is the method, and the reason the bar for an accuracy claim
is what it is.

These used to hardcode one recording's path, which made them useless to anyone
else. They now take the corpus from the environment or the command line; see
`scripts/bench/corpus.py`:

```bash
export BENCH_OUT=output/my-meeting
export BENCH_VTT="input/my meeting.vtt"
docker compose run --rm app python scripts/bench/bench_wording.py
```

`fetch_soniox.py` is the only thing in this repository that sends audio
anywhere. It is manual, needs a `SONIOX_API_KEY` you supply, and deletes the
upload when it is done.
