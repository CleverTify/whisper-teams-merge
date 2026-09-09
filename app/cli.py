"""Command line.

    transcribe warmup
    transcribe <media> [--teams TRANSCRIPT] [options]
    transcribe reexport <output-dir> --name SPEAKER_00="Jan Novák"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.config import (
    CACHE_DIR,
    INPUT_DIR,
    OUTPUT_DIR,
    hardware,
    resolve_backend,
    resolve_compute_type,
    settings,
    setup_logging,
    with_overrides,
)

COMMANDS = {"warmup", "transcribe", "reexport"}


def cmd_warmup(args) -> int:
    setup_logging()
    ok = True
    hw = hardware()
    backend = resolve_backend(settings.backend, hw)

    print("\n=== hardware ===")
    print(f"  torch        : {hw.torch_version} (CUDA {hw.torch_cuda or 'n/a'})")
    print(f"  device       : {hw.name}")
    if hw.cuda:
        print(f"  capability   : {hw.capability[0]}.{hw.capability[1]} (sm_{hw.sm})")
        print(f"  VRAM         : {hw.vram_gb} GB")
        import torch

        try:
            x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
            torch.cuda.synchronize()
            _ = (x @ x).sum().item()
            print("  fp16 GEMM    : OK")
        except Exception as exc:
            print(f"  fp16 GEMM    : FAILED — {exc}")
            ok = False
    print(f"  ASR backend  : {backend}")
    if hw.cuda:
        print(f"  compute_type : {resolve_compute_type(settings.compute_type, hw)}")

    print("\n=== HuggingFace ===")
    if not settings.hf_token:
        print("  HF_TOKEN     : missing — diarization will not work")
        print("                 (not needed when you supply a Teams transcript)")
    else:
        print(f"  HF_TOKEN     : {settings.hf_token[:6]}…{settings.hf_token[-4:]}")
        for repo in (settings.diarize_model, "pyannote/segmentation-3.0"):
            ok &= _check_gated(repo)

    print("\n=== local LLM ===")
    from app import llm as llm_mod

    if not settings.llm_enabled:
        print("  disabled")
    elif llm_mod.available(settings.llm_base_url):
        print(f"  reachable    : {settings.llm_base_url} ({settings.llm_model})")
    else:
        print(f"  NOT reachable: {settings.llm_base_url}")
        print("                 start it with `docker compose up llm`")

    if args.skip_download:
        return 0 if ok else 1

    print(f"\n=== prefetching models into {CACHE_DIR} ===")
    ok &= _prefetch(backend, [l.strip() for l in (args.languages or "cs,en").split(",") if l.strip()])
    print("\n" + ("All checks passed." if ok else "FAILED — see above."))
    return 0 if ok else 1


def _check_gated(repo: str) -> bool:
    from huggingface_hub import HfApi

    try:
        HfApi(token=settings.hf_token).model_info(repo)
        print(f"  {repo:<45} OK")
        return True
    except Exception as exc:
        print(f"  {repo:<45} DENIED ({type(exc).__name__})")
        print(f"    accept the terms at https://huggingface.co/{repo}")
        return False


def _prefetch(backend: str, languages: list[str]) -> bool:
    ok = True
    if backend == "faster-whisper":
        try:
            print(f"  Whisper {settings.whisper_model} …", flush=True)
            from app.asr import FasterWhisper

            e = FasterWhisper(settings)
            e.pipeline()
            e.release()
            print("    OK")
        except Exception as exc:
            print(f"    FAILED: {exc}")
            ok = False
    else:
        path = CACHE_DIR / "whispercpp" / settings.whisper_cpp_model
        if path.exists():
            print(f"  whisper.cpp model present: {path.name}")
        else:
            print(f"  downloading {settings.whisper_cpp_model} …", flush=True)
            ok &= _download_ggml(path)

    try:
        from app.align import AlignerCache

        cache = AlignerCache(device=hardware().device)
        for lang in languages:
            print(f"  aligner {lang:<4} {'OK' if cache.get(lang) else 'none'}")
        cache.release()
    except Exception as exc:
        print(f"  aligner prefetch FAILED: {exc}")
        ok = False

    if settings.diarize and settings.hf_token:
        try:
            print("  pyannote diarizer …", flush=True)
            from whisperx.diarize import DiarizationPipeline

            DiarizationPipeline(model_name=settings.diarize_model,
                                token=settings.hf_token, device=hardware().device)
            print("    OK")
        except Exception as exc:
            print(f"    FAILED: {exc}")
            ok = False
    return ok


def _download_ggml(path: Path) -> bool:
    """Prefetch for `warmup`. The download itself lives in asr.download_ggml,
    which is also what a job calls when it finds the model missing."""
    from app.asr import download_ggml

    try:
        got = download_ggml(path)
        print(f"    OK ({got.stat().st_size / 1024**2:.0f} MB)")
        return True
    except Exception as exc:
        print(f"    FAILED: {exc}")
        return False


# ---------------------------------------------------------------------------

def cmd_transcribe(args) -> int:
    log = setup_logging(args.log_level)

    source = Path(args.media)
    if not source.exists():
        candidate = INPUT_DIR / source.name
        if candidate.exists():
            source = candidate
    if not source.exists():
        log.error("not found: %s (place files in ./input)", args.media)
        return 2

    teams = None
    if args.teams:
        teams = Path(args.teams)
        if not teams.exists():
            teams = INPUT_DIR / Path(args.teams).name
        if not teams.exists():
            log.error("Teams transcript not found: %s", args.teams)
            return 2

    overrides = {
        k: getattr(args, k)
        for k in ("whisper_model", "lang_mode", "language", "speakers",
                  "min_speakers", "max_speakers", "batch_size", "backend")
        if getattr(args, k, None) is not None
    }
    active = with_overrides(settings, overrides)

    # Fail here rather than at stage four. Without a Teams transcript the
    # speakers come from pyannote, which is gated behind a HuggingFace token,
    # and diarization runs after ingest, language ID, ASR and alignment -- so
    # the old behaviour was to chew through a whole recording before saying
    # 'no token'. The web UI already refuses at submit; this matches it.
    if not teams and active.diarize and not active.hf_token and not args.no_diarize:
        log.error(
            "no HF_TOKEN, so speakers cannot be detected. Either pass --teams "
            "with the Teams transcript (better anyway, it carries the real "
            "names), put HF_TOKEN in .env, or pass --no-diarize to transcribe "
            "without speaker labels.")
        return 2

    from app.pipeline import Options, run

    try:
        result = run(source, active, Options(
            force=args.force, duration=args.duration, start=args.start,
            teams_transcript=teams, skip_diarize=args.no_diarize, skip_llm=args.no_llm,
        ))
    except Exception as exc:
        log.error("%s", exc)
        if args.log_level == "DEBUG":
            raise
        return 1

    print("\n=== output ===")
    for label, path in result["paths"].items():
        print(f"  {label:<16} {path}")
    print("\n=== speakers ===")
    for row in result["stats"]:
        print(f"  {result['name_map'].get(row['speaker'], row['speaker']):<24}"
              f"{row['clock']}  {row['share']:.1%}  {row['turns']} turns")
    return 0


def cmd_reexport(args) -> int:
    setup_logging(args.log_level)
    directory = Path(args.directory)
    if not directory.exists():
        directory = OUTPUT_DIR / args.directory
    if not directory.exists():
        print(f"not found: {args.directory}", file=sys.stderr)
        return 2

    names = {}
    for item in args.name or []:
        if "=" not in item:
            print(f"--name expects LABEL=Name, got {item!r}", file=sys.stderr)
            return 2
        k, v = item.split("=", 1)
        names[k.strip()] = v.strip()

    from app.pipeline import reexport

    result = reexport(directory, names)
    print(f"re-exported {len(result['paths'])} artifacts to {result['output_dir']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="transcribe",
        description="Multilingual transcription with diarization, "
                    "optionally merged with a Teams transcript by a local LLM.",
    )
    p.add_argument("--log-level", default=settings.log_level,
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = p.add_subparsers(dest="command", required=True)

    w = sub.add_parser("warmup", help="check hardware and prefetch models")
    w.add_argument("--languages", default="cs,en")
    w.add_argument("--skip-download", action="store_true")
    w.set_defaults(func=cmd_warmup)

    t = sub.add_parser("transcribe", help="transcribe audio or video")
    t.add_argument("media")
    t.add_argument("--teams", metavar="FILE",
                   help="Teams transcript (.vtt/.docx): speakers taken strictly from it")
    t.add_argument("--backend", choices=["auto", "faster-whisper", "whisper.cpp"])
    t.add_argument("--whisper-model", dest="whisper_model")
    t.add_argument("--lang-mode", dest="lang_mode", choices=["single", "auto", "multi"])
    t.add_argument("--language")
    t.add_argument("--batch-size", dest="batch_size", type=int)
    t.add_argument("--speakers", type=int,
                   help="exact number of people, when you know it")
    t.add_argument("--min-speakers", dest="min_speakers", type=int)
    t.add_argument("--max-speakers", dest="max_speakers", type=int)
    t.add_argument("--start", type=float)
    t.add_argument("--duration", type=float)
    t.add_argument("--no-diarize", action="store_true")
    t.add_argument("--no-llm", action="store_true")
    t.add_argument("--force", action="store_true")
    t.set_defaults(func=cmd_transcribe)

    r = sub.add_parser("reexport", help="regenerate outputs with new speaker names")
    r.add_argument("directory")
    r.add_argument("--name", action="append", metavar="LABEL=Name")
    r.set_defaults(func=cmd_reexport)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in COMMANDS and not argv[0].startswith("-"):
        argv.insert(0, "transcribe")
    args = build_parser().parse_args(argv)
    if getattr(args, "language", None):
        args.lang_mode = args.lang_mode or "single"
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
