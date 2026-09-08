"""Stage orchestration.

    preprocess -> language ID -> ASR -> speakers (Teams | pyannote)
        -> LLM (merge | cleanup) -> forced alignment -> export

Alignment comes **after** the LLM on purpose: corrected text no longer matches
the original word timings, and stale timings would desynchronise the player's
word highlighting from what is on screen.

Every stage caches to `work/` and is skipped when nothing it depends on has
changed, so a failure late in a 30-minute job does not throw away the
transcription.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app import (
    __version__,
    align as align_mod,
    asr as asr_mod,
    screen as screen_mod,
    teams as teams_mod,
)
from app.audio import load_wav, preprocess
from app.cleanup import append_removed, clean_segments, drop_unvoiced
from app.config import (
    JobPaths,
    dump_json,
    free_gpu,
    hardware,
    job_paths,
    wait_for_free,
    resolve_backend,
)
from app.diarize import (
    assign_speakers,
    build_turns,
    collapse_turns_to_target,
    diarize,
    merge_similar_speakers,
    speaker_stats,
)
from app.export import export_all
from app.langid import LanguageReport, LanguageRun, identify

log = logging.getLogger("transcribe.pipeline")

# Alignment sits *before* speaker assignment: both Teams and pyannote label
# individual words, and a single ASR segment routinely spans a handover. With
# no word timings yet, a 27-second segment covering two people collapses to
# whoever overlapped most — which silently loses a speaker.
# Turns whose text the LLM changes are re-aligned afterwards so the word
# timings still match what is displayed.
# "screen" sits between speakers and llm on purpose. Its output only feeds
# the merge, and the fingerprints chain forward, so putting it any earlier
# would make a scene-threshold tweak invalidate ASR — 36% of the pipeline —
# for a change that cannot possibly affect it.
STAGE_ORDER = ["ingest", "language", "asr", "align", "speakers", "screen", "llm"]
STAGE_WEIGHTS = {
    "preprocess": 0.04, "langid": 0.05, "asr": 0.32,
    "speakers": 0.19, "screen": 0.12, "llm": 0.16, "align": 0.09,
    "export": 0.03,
}


@dataclass
class Options:
    force: bool = False
    duration: float | None = None
    start: float | None = None
    teams_transcript: Path | None = None
    skip_diarize: bool = False
    skip_llm: bool = False
    output_root: Path | None = None


class Progress:
    def __init__(self, cb=None):
        self.cb, self.done, self.stage = cb, 0.0, ""

    def begin(self, stage: str) -> None:
        self.stage = stage
        self._emit(0.0)

    def update(self, _label: str, fraction: float) -> None:
        self._emit(max(0.0, min(1.0, fraction)))

    def finish(self, stage: str) -> None:
        self.done += STAGE_WEIGHTS.get(stage, 0.0)
        self._emit(0.0)

    def _emit(self, within: float) -> None:
        if not self.cb:
            return
        try:
            self.cb(self.stage, min(1.0, self.done + STAGE_WEIGHTS.get(self.stage, 0.0) * within))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Stage cache
# ---------------------------------------------------------------------------

def _h(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def _fingerprints(source: Path, settings, options: Options) -> dict[str, str]:
    """Chained per-stage fingerprints.

    Without this, `--duration 120` would leave a two-minute transcript cached
    and the next full run would silently reuse it.

    Everything that changes a stage's *output* has to be in here. A setting left
    out does not merely miss a refresh — the run reports success while serving
    the previous answer, which is worse than an error and invisible in the logs.
    `turn_merge_gap` and `min_turn_seconds` were the sharp case: they renumber
    turns, and the cached LLM text is restored onto turns by position, so a
    changed value silently pasted corrected text onto the wrong turns.
    """
    from app.asr import ASR_OPTIONS

    st = source.stat()
    tt = options.teams_transcript
    parts = {
        "ingest": {"src": source.name, "size": st.st_size, "mtime": st.st_mtime_ns,
                   "start": options.start, "duration": options.duration,
                   "loudnorm": settings.loudnorm, "highpass": settings.highpass_hz,
                   "denoise": settings.denoise},
        "language": {"mode": settings.lang_mode, "language": settings.language,
                     "dominance": settings.lang_dominance, "samples": settings.lid_samples},
        # ASR_OPTIONS is a module constant rather than a setting, so editing it
        # would otherwise change decoding without invalidating anything.
        "asr": {"backend": resolve_backend(settings.backend),
                "model": settings.whisper_model, "beam": settings.beam_size,
                "compute_type": settings.compute_type,
                "batch": settings.batch_size,
                "initial_prompt": settings.initial_prompt,
                "vad": (settings.vad_onset, settings.vad_offset,
                        settings.vad_chunk_size),
                "decode": sorted((k, str(v)) for k, v in ASR_OPTIONS.items())},
        "align": {"device": hardware().device},
        "speakers": {"teams": tt.name if tt else None,
                     "teams_size": tt.stat().st_size if tt and tt.exists() else None,
                     "on": bool(settings.diarize and not options.skip_diarize),
                     "model": settings.diarize_model,
                     "min": settings.min_speakers, "max": settings.max_speakers,
                     "target": settings.speakers,
                     "cluster_threshold": settings.cluster_threshold,
                     "snap": settings.boundary_snap_words,
                     "smooth": settings.speaker_smooth_words,
                     "merge": settings.speaker_merge_threshold,
                     # Both feed build_turns() and so decide how many turns
                     # exist; see the docstring above.
                     "turn_merge_gap": settings.turn_merge_gap,
                     "min_turn_seconds": settings.min_turn_seconds},
        # start/duration belong here or a `--duration 120` trial caches two
        # minutes of frames and the next full run reuses them.
        "screen": {"on": bool(settings.screen_enabled),
                   "start": options.start, "duration": options.duration,
                   "threshold": settings.screen_scene_threshold,
                   "max_frames": settings.screen_max_frames,
                   "width": settings.screen_frame_width,
                   "langs": settings.screen_ocr_langs,
                   "vlm": bool(settings.screen_vlm_enabled),
                   "vlm_model": settings.screen_vlm_model,
                   "terms_in_prompt": bool(settings.screen_terms_in_prompt)},
        "llm": {"on": bool(settings.llm_enabled and not options.skip_llm),
                "model": settings.llm_model, "window": settings.llm_window_seconds,
                # The guardrail decides which corrections survive, so it changes
                # the transcript as surely as the model does.
                "max_ratio": settings.llm_max_length_ratio,
                "min_ratio": settings.llm_min_length_ratio,
                "min_overlap": settings.llm_min_token_overlap,
                "glossary": settings.glossary_terms},
    }
    chain, running = {}, ""
    for name in STAGE_ORDER:
        running = _h({"prev": running, **parts[name]})
        chain[name] = running
    return chain


class StageCache:
    def __init__(self, paths: JobPaths, current: dict[str, str], force: bool):
        self.paths, self.current, self.force = paths, current, force
        self.path = paths.work / "fingerprints.json"
        try:
            self.stored = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self.stored = {}
        self.invalid_from = None
        for i, name in enumerate(STAGE_ORDER):
            if self.stored.get(name) != current[name]:
                self.invalid_from = i
                break
        if self.invalid_from is not None and self.stored:
            log.info("configuration changed — recomputing from %r onwards",
                     STAGE_ORDER[self.invalid_from])

    def stale(self, name: str) -> bool:
        if self.force:
            return True
        if self.invalid_from is None:
            return False
        return STAGE_ORDER.index(name) >= self.invalid_from

    def commit(self, name: str) -> None:
        self.stored[name] = self.current[name]
        self.path.write_text(dump_json(self.stored), encoding="utf-8")


def _load(paths: JobPaths, name: str, stale: bool):
    p = paths.stage(name)
    if stale or not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        log.info("resuming: reusing stage %r", name)
        return data
    except Exception:
        return None


def _save(paths: JobPaths, name: str, data) -> None:
    paths.stage(name).write_text(dump_json(data), encoding="utf-8")


# ---------------------------------------------------------------------------

def run(source: Path, settings, options: Options | None = None, progress_cb=None) -> dict:
    options = options or Options()
    # A partial run is a trial, and must not overwrite the real transcript.
    paths = job_paths(source, options.output_root,
                      trial=bool(options.duration or options.start))
    progress = Progress(progress_cb)
    started = time.time()
    hw = hardware()

    log.info("=" * 68)
    log.info("input : %s", source.name)
    log.info("output: %s", paths.root)
    log.info("device: %s", f"{hw.name} sm_{hw.sm} {hw.vram_gb} GB" if hw.cuda else "CPU")
    log.info("asr   : %s", resolve_backend(settings.backend, hw))
    if options.teams_transcript:
        log.info("teams : %s (diarization taken strictly from it)",
                 options.teams_transcript.name)
    log.info("=" * 68)

    cache = StageCache(paths, _fingerprints(source, settings, options), options.force)

    # -- 0. preprocess -----------------------------------------------------
    progress.begin("preprocess")
    ingest = _load(paths, "ingest", cache.stale("ingest"))
    if ingest is None:
        ingest = preprocess(
            source, paths.audio,
            loudnorm=settings.loudnorm, highpass_hz=settings.highpass_hz,
            denoise=settings.denoise, start=options.start, duration=options.duration,
            force=cache.stale("ingest"),
        )
        _save(paths, "ingest", ingest)
    cache.commit("ingest")
    progress.finish("preprocess")

    audio = load_wav(paths.audio)
    engine = asr_mod.build(settings)

    # -- 1. language identification ---------------------------------------
    progress.begin("langid")
    cached = _load(paths, "language", cache.stale("language"))
    if cached:
        report = LanguageReport(
            mode=cached["mode"], dominant=cached.get("dominant", ""),
            shares=cached.get("shares", {}), samples=cached.get("samples", []),
            code_switching=cached.get("code_switching", False),
            notes=cached.get("notes", []),
            runs=[LanguageRun(r["start"], r["end"], r["language"], r.get("confidence", 0.0))
                  for r in cached["runs"]],
        )
    else:
        report = _identify(engine, settings, audio)
        _save(paths, "language", report.to_dict())
    cache.commit("language")
    progress.finish("langid")
    for note in report.notes:
        log.info("language note: %s", note)

    # -- 2. ASR ------------------------------------------------------------
    progress.begin("asr")
    cached = _load(paths, "asr", cache.stale("asr"))
    if cached:
        segments, asr_meta = cached["segments"], cached["meta"]
    else:
        segments = asr_mod.transcribe_runs(engine, audio, report.runs, progress=progress.update)
        asr_meta = engine.metadata()
        segments, removed = clean_segments(segments, removed_path=paths.removed_jsonl)
        _save(paths, "asr", {"segments": segments, "meta": asr_meta, "removed": len(removed)})
    cache.commit("asr")
    engine.release()
    free_gpu()
    progress.finish("asr")

    if not segments:
        raise RuntimeError(f"ASR produced nothing usable — check {paths.audio}")

    # -- 3. alignment: word timings, needed for word-level speaker labels --
    progress.begin("align")
    cached = _load(paths, "align", cache.stale("align"))
    if cached:
        segments, align_meta = cached["segments"], cached["meta"]
    else:
        segments, align_meta = align_mod.align_segments(
            segments, audio, hw.device, progress=progress.update
        )
        _save(paths, "align", {"segments": segments, "meta": align_meta})
    cache.commit("align")
    free_gpu()
    progress.finish("align")

    # -- 4. speakers: Teams transcript (strict) or pyannote ----------------
    progress.begin("speakers")
    teams_cues = None
    if options.teams_transcript:
        teams_cues = teams_mod.load(options.teams_transcript)

    cached = _load(paths, "speakers", cache.stale("speakers"))
    if cached:
        segments, spk_meta = cached["segments"], cached["meta"]
    else:
        if teams_cues:
            segments = teams_mod.assign_speakers(segments, teams_cues)
            spk_meta = {"source": "teams", "file": options.teams_transcript.name,
                        "speakers": sorted({c.speaker for c in teams_cues})}
            log.info("speakers taken from Teams transcript: %s",
                     ", ".join(spk_meta["speakers"]))
        elif settings.diarize and not options.skip_diarize:
            df, emb = diarize(
                audio, hf_token=settings.hf_token, model_name=settings.diarize_model,
                device=hw.device, min_speakers=settings.min_speakers,
                max_speakers=settings.max_speakers,
                cluster_threshold=settings.cluster_threshold,
                progress=progress.update,
            )
            before = int(df["speaker"].nunique())
            # pyannote splits one person into several clusters when their
            # acoustics change mid-call; fold those back together.
            df, merged = merge_similar_speakers(df, emb, settings.speaker_merge_threshold)
            segments = assign_speakers(segments, df)
            segments, unvoiced = drop_unvoiced(segments)
            append_removed(paths.removed_jsonl, unvoiced)
            spk_meta = {"source": "pyannote", "model": settings.diarize_model,
                        "min": settings.min_speakers, "max": settings.max_speakers,
                        "clusters_found": before,
                        "clusters_after_merge": int(df["speaker"].nunique()),
                        "target": settings.speakers, "merged": merged}
        else:
            for s in segments:
                s.setdefault("speaker", "SPEAKER_00")
            spk_meta = {"source": "none"}
        _save(paths, "speakers", {"segments": segments, "meta": spk_meta})
    cache.commit("speakers")
    free_gpu()
    progress.finish("speakers")

    # -- 5. screen: what was visible while they were talking ----------------
    # Only for video uploads. The words this pipeline most reliably gets wrong
    # are product names spoken in Czech, and on a screen share those are usually
    # printed in the frame at the moment somebody says them.
    progress.begin("screen")
    screen_frames: list[dict] = []
    screen_meta: dict = {"mode": "off"}
    commit_screen = True
    cached = _load(paths, "screen", cache.stale("screen"))
    if cached:
        screen_frames, screen_meta = cached["frames"], cached["meta"]
    elif not settings.screen_enabled:
        screen_meta = {"mode": "disabled"}
        _save(paths, "screen", {"frames": [], "meta": screen_meta})
    elif not ingest["source"].get("has_video"):
        # Audio-only input. Deterministic, so caching it is correct.
        screen_meta = {"mode": "no-video"}
        _save(paths, "screen", {"frames": [], "meta": screen_meta})
    else:
        try:
            screen_frames = screen_mod.extract_frames(
                Path(ingest["source"]["path"]), paths.frames,
                threshold=settings.screen_scene_threshold,
                max_frames=settings.screen_max_frames,
                width=settings.screen_frame_width,
                start=options.start, duration=options.duration,
            )
            screen_mod.check_ocr_language(settings.screen_ocr_langs)
            screen_frames = screen_mod.ocr_frames(
                screen_frames, langs=settings.screen_ocr_langs)
            screen_meta = {"mode": "ocr", "frames": len(screen_frames)}

            if settings.screen_vlm_enabled and screen_frames:
                free_gpu()
                # The text model must be out of the way first: 5.9 GB of it plus
                # 6.5 GB of vision model does not fit on this card, and the
                # resulting OOM surfaces as a timeout three stages later.
                # Advisory, not fatal. The point is to give the text model
                # time to evict itself before the vision model loads, but
                # the vision service is long-lived and may already hold
                # its memory — in which case there is nothing to wait for
                # and demanding headroom would block the feature it is
                # meant to protect. A genuine shortage surfaces as a
                # failed capability probe on the next line, which is a
                # real test rather than a guess.
                wait_for_free(7.0, timeout=120.0, reason="vision model")
                screen_mod.probe_vision(
                    settings.screen_vlm_base_url, settings.screen_vlm_model,
                    settings.screen_vlm_timeout)
                screen_frames = screen_mod.caption_frames(
                    screen_frames, base_url=settings.screen_vlm_base_url,
                    model=settings.screen_vlm_model,
                    timeout=settings.screen_vlm_timeout,
                    progress=progress.update)
                screen_meta["mode"] = "ocr+vlm"
            _save(paths, "screen",
                  {"frames": screen_frames, "meta": screen_meta})
        except screen_mod.ScreenUnavailable as exc:
            # Transient: a service was down or the card was busy. Caching this
            # would turn a temporary outage into a permanently context-free
            # transcript, exactly as it would for the LLM stage below.
            log.warning("screen context unavailable (%s) — not caching, "
                        "a later run should retry it", exc)
            screen_meta = {"mode": "skipped", "reason": str(exc)}
            commit_screen = False
        except screen_mod.ScreenError as exc:
            # Local and deterministic — a missing video, no ffmpeg, no
            # traineddata. Retrying changes nothing, so record it and move on.
            log.warning("screen context failed: %s", exc)
            screen_meta = {"mode": "error", "reason": str(exc)}
            _save(paths, "screen", {"frames": [], "meta": screen_meta})
    if commit_screen:
        cache.commit("screen")
    free_gpu()
    progress.finish("screen")

    screen_ctx = screen_mod.ScreenContext(
        screen_frames, limit=settings.screen_terms_per_window)

    # -- 6. turns, then LLM ------------------------------------------------
    progress.begin("llm")
    def _turns():
        return build_turns(
            segments, merge_gap=settings.turn_merge_gap,
            smooth=settings.speaker_smooth_words,
            min_turn_seconds=settings.min_turn_seconds,
            snap_window=settings.boundary_snap_words,
        )

    turns = _turns()

    # Collapsing to a known speaker count needs conversational turns, so it
    # happens here rather than on the raw diarization segments; the turns are
    # then rebuilt so adjacent same-speaker turns merge properly.
    if settings.speakers and teams_cues is None:
        mapping = collapse_turns_to_target(turns, settings.speakers)
        if any(k != v for k, v in mapping.items()):
            for seg in segments:
                if seg.get("speaker"):
                    seg["speaker"] = mapping.get(seg["speaker"], seg["speaker"])
                for w in seg.get("words") or []:
                    if w.get("speaker"):
                        w["speaker"] = mapping.get(w["speaker"], w["speaker"])
            turns = _turns()

    cached = _load(paths, "llm", cache.stale("llm"))
    if cached:
        for turn, saved in zip(turns, cached["turns"]):
            turn.text = saved["text"]
            if saved.get("words"):
                turn.words = saved["words"]
        llm_meta = cached["meta"]
    elif settings.llm_enabled and not options.skip_llm:
        from app import llm as llm_mod

        before = [t.text for t in turns]
        unavailable = False
        try:
            stats = llm_mod.process(turns, settings, teams_cues=teams_cues,
                                    screen=(screen_ctx or None)
                                    if settings.screen_terms_in_prompt else None,
                                    progress=progress.update)
            llm_meta = stats.to_dict()
        except llm_mod.LLMUnavailable as exc:
            log.warning("%s — continuing without LLM post-processing", exc)
            llm_meta = {"mode": "skipped", "reason": str(exc)}
            unavailable = True
        _realign_changed(turns, before, audio, hw.device)
        if unavailable:
            # Caching this would be worse than the missed pass: the next run
            # would see a satisfied fingerprint and reuse an unmerged
            # transcript, so a service that was briefly down turns into a
            # permanently worse result that nothing reports.
            log.warning("not caching the LLM stage — a later run should retry it")
        else:
            _save(paths, "llm",
                  {"turns": [{"text": t.text, "words": t.words} for t in turns],
                   "meta": llm_meta})
            cache.commit("llm")
    else:
        # Skipped deliberately (--no-llm, or LLM_ENABLED=false). Commit the
        # fingerprint *and* write the stage, because committing alone left
        # the previous run's merged turns in work/llm.json: the next
        # --no-llm run then found a satisfied fingerprint, restored that
        # file, and produced a merged transcript from an invocation that
        # asked for none. Nothing reported it.
        llm_meta = {"mode": "disabled"}
        _save(paths, "llm",
              {"turns": [{"text": t.text, "words": t.words} for t in turns],
               "meta": llm_meta})
        cache.commit("llm")
    if cached:
        cache.commit("llm")
    free_gpu()
    progress.finish("llm")

    mismatched = check_word_text_invariant(turns)
    llm_meta = {**llm_meta, "words_text_mismatch": len(mismatched)}

    # -- 6. export ---------------------------------------------------------
    progress.begin("export")
    labels = sorted({t.speaker for t in turns})
    meta = {
        "pipeline_version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.time() - started, 1),
        "source": ingest["source"],
        "prepared": ingest["prepared"],
        "preprocess": ingest["preprocess"],
        "asr": asr_meta,
        "speakers_from": spk_meta,
        "llm": llm_meta,
        "alignment": align_meta,
        "languages": {
            "mode": report.mode, "dominant": report.dominant, "shares": report.shares,
            "code_switching": report.code_switching,
            "runs": [r.to_dict() for r in report.runs], "notes": report.notes,
        },
        "environment": {
            "device": hw.device, "gpu": hw.name,
            "compute_capability": f"{hw.capability[0]}.{hw.capability[1]}" if hw.cuda else "",
            "torch": hw.torch_version, "torch_cuda": hw.torch_cuda,
        },
    }
    name_map = {label: label for label in labels}
    artifacts = export_all(paths, turns=turns, name_map=name_map, meta=meta, screen=screen_ctx.to_track())
    progress.finish("export")

    elapsed = time.time() - started
    duration = ingest["prepared"]["duration"] or 1.0
    log.info("done in %.1f min (%.1fx realtime) — %d turns, %d speaker(s)",
             elapsed / 60, duration / elapsed, len(turns), len(labels))

    return {
        "paths": artifacts, "output_dir": str(paths.root), "turns": len(turns),
        "speakers": labels, "name_map": name_map,
        "stats": speaker_stats(turns), "metadata": meta,
    }


_INVARIANT_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def check_word_text_invariant(turns) -> list[int]:
    """Every turn's `words` must tokenise to exactly its `text`.

    This is the contract the exports and every scoring script rely on: the VTT
    is written from `words`, so a duplicate there ships repeated words at turn
    boundaries, and anything measuring `words` is then measuring a stream that
    is not the transcript.

    It broke silently once. `_realign_changed` selected each turn's words with a
    ±0.25s tolerance, which made neighbouring windows overlap, and both turns
    claimed the words in between — 432 duplicated tokens across 229 turns,
    inflating the denominator of every word-level metric by ~4%. Nothing failed;
    the numbers just quietly stopped meaning what they said. Hence this check.

    Returns the indices that disagree, and says so loudly rather than raising:
    aborting after a 20-minute run would destroy a transcript that is still
    largely usable. The count is recorded in result.json so it cannot hide.
    """
    bad: list[int] = []
    for i, turn in enumerate(turns):
        text_toks = [w.lower() for w in _INVARIANT_WORD.findall(turn.text or "")]
        word_toks = [
            w.lower()
            for entry in (turn.words or [])
            for w in _INVARIANT_WORD.findall(entry.get("word", ""))
        ]
        # A turn with no words at all is a known, deliberate outcome of failed
        # re-alignment; only a *populated* mismatch is a defect.
        if word_toks and text_toks != word_toks:
            bad.append(i)

    if bad:
        log.error(
            "word/text invariant broken on %d of %d turn(s) — the VTT will "
            "repeat or drop words and every word-level metric is off. First: %s",
            len(bad), len(turns),
            ", ".join(turns[i].clock for i in bad[:5]),
        )
    return bad


def _realign_changed(turns, before: list[str], audio, device: str) -> int:
    """Refresh word timings for the turns the LLM rewrote.

    Only those turns are re-aligned: the rest still match their original audio
    exactly, and a full second pass would double the alignment cost for no
    benefit. Without this the player would highlight words that no longer
    correspond to the text on screen.
    """
    changed = [i for i, t in enumerate(turns) if t.text != before[i]]
    if not changed:
        return 0

    log.info("re-aligning %d LLM-corrected turn(s)", len(changed))
    segments = [
        {"start": turns[i].start, "end": turns[i].end,
         "text": turns[i].text, "language": turns[i].language}
        for i in changed
    ]
    try:
        aligned, _ = align_mod.align_segments(segments, audio, device)
    except Exception as exc:
        log.warning("re-alignment failed (%s); keeping previous word timings", exc)
        return 0

    # Do NOT match results back by index or start time: the aligner splits one
    # input segment into several and moves `start` onto the first word, so a
    # one-to-one mapping silently finds nothing. Collect every word it produced
    # and hand each turn the words that fall inside its own time range.
    words = [
        w
        for s in aligned
        for w in (s.get("words") or [])
        if w.get("start") is not None and w.get("end") is not None
    ]
    words.sort(key=lambda w: w["start"])

    restored = 0
    # Each word belongs to exactly one turn. Selecting per turn with a ±0.25s
    # tolerance made neighbouring windows overlap, so a word near a boundary was
    # handed to *both* turns — 432 duplicated tokens across 229 turns, which is
    # why turn["words"] stopped matching turn["text"]. Assign in the other
    # direction instead: give every word to its single closest turn.
    def distance(turn, mid: float) -> float:
        if turn.start <= mid <= turn.end:
            return 0.0
        return min(abs(mid - turn.start), abs(mid - turn.end))

    claimed: dict[int, list[dict]] = {i: [] for i in changed}
    for w in words:
        mid = (float(w["start"]) + float(w["end"])) / 2
        best = min(changed, key=lambda i: (distance(turns[i], mid), abs(mid - turns[i].start)))
        if distance(turns[best], mid) <= 0.25:
            claimed[best].append(w)

    for i in changed:
        turn = turns[i]
        mine = claimed[i]
        if mine:
            for w in mine:
                w.setdefault("speaker", turn.speaker)
            turn.words = mine
            restored += 1
        else:
            # Nothing usable: drop the stale timings rather than highlight the
            # wrong words. The turn still renders, just without word sync.
            turn.words = []
    log.info("re-aligned %d/%d corrected turn(s)", restored, len(changed))
    return restored


def _identify(engine, settings, audio) -> LanguageReport:
    if settings.lang_mode == "single":
        return identify(None, audio, mode="single",
                        forced_language=settings.language or "en")
    pipeline = getattr(engine, "pipeline", None)
    if pipeline is None:
        # whisper.cpp does its own per-run detection; treat the file as one run.
        log.info("language ID: backend detects per run")
        return identify(None, audio, mode="single",
                        forced_language=settings.language or "auto")
    return identify(
        pipeline(), audio, mode=settings.lang_mode,
        forced_language=settings.language, dominance=settings.lang_dominance,
        n_samples=settings.lid_samples,
    )


# ---------------------------------------------------------------------------

def reexport(root: Path, name_map: dict[str, str]) -> dict:
    """Regenerate outputs with new speaker names — no models, no re-transcription."""
    root = Path(root)
    payload = json.loads((root / "result.json").read_text(encoding="utf-8"))
    meta = payload["metadata"]
    paths = JobPaths(source=Path(meta["source"]["path"]), root=root).ensure()

    from app.diarize import Turn

    turns = [
        Turn(start=t["start"], end=t["end"], speaker=t["speaker"],
             text=t["text"], language=t.get("language", ""), words=t.get("words", []))
        for t in payload["turns"]
    ]
    merged = dict(payload.get("name_map", {}))
    merged.update({k: v for k, v in name_map.items() if v and v.strip()})
    # Carry the screen annotations across. Renaming a speaker must not
    # quietly strip them from the exports it regenerates.
    artifacts = export_all(paths, turns=turns, name_map=merged, meta=meta,
                           screen=payload.get("screen") or [])
    return {"paths": artifacts, "name_map": merged, "output_dir": str(root)}
