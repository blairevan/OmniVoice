#!/usr/bin/env python3
"""Prepare and validate the model caches required by offline OmniVoice."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable, Sequence

from omnivoice.utils.offline_cache import (
    CacheCheckResult,
    CacheComponent,
    build_manifest,
    validate_component,
)


OMNIVOICE_ID = "k2-fsa/OmniVoice"
AUDIO_TOKENIZER_ID = "eustlb/higgs-audio-v2-tokenizer"
FUNASR_ID = "iic/speech_timestamp_prediction-v1-16k-offline"
FUNASR_REVISION = "v2.0.4"
DEFAULT_FUNASR_PATH = (
    "~/.cache/modelscope/models/"
    "iic--speech_timestamp_prediction-v1-16k-offline/snapshots/v2.0.4"
)
DEFAULT_WHISPERX_PATH = "/opt/app/aining/digital_human/whisperx/models/zh"

HuggingFaceDownloader = Callable[..., str]
ModelScopeDownloader = Callable[..., str]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse cache preparation and validation options."""
    parser = argparse.ArgumentParser(
        description="Download or validate OmniVoice offline model caches"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only validate local caches; never download or update models",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=None,
        help="Write the successful cache manifest to this JSON path",
    )
    parser.add_argument("--omnivoice-id", default=OMNIVOICE_ID)
    parser.add_argument("--omnivoice-path", type=Path, default=None)
    parser.add_argument("--audio-tokenizer-id", default=AUDIO_TOKENIZER_ID)
    parser.add_argument("--audio-tokenizer-path", type=Path, default=None)
    parser.add_argument("--funasr-model-id", default=FUNASR_ID)
    parser.add_argument("--funasr-revision", default=FUNASR_REVISION)
    parser.add_argument("--funasr-model-path", type=Path, default=None)
    parser.add_argument(
        "--whisperx-model-path",
        type=Path,
        default=Path(DEFAULT_WHISPERX_PATH),
    )
    return parser.parse_args(argv)


def _download_huggingface(
    model_id: str,
    check_only: bool,
    downloader: HuggingFaceDownloader | None = None,
) -> Path:
    """Resolve a HuggingFace model from cache or download it once."""
    if downloader is None:
        from huggingface_hub import snapshot_download

        downloader = snapshot_download
    return Path(
        downloader(
            repo_id=model_id,
            local_files_only=check_only,
        )
    )


def _download_funasr(
    model_id: str,
    revision: str,
    check_only: bool,
    downloader: ModelScopeDownloader | None = None,
) -> Path:
    """Resolve the FunASR ModelScope snapshot without loading a model."""
    if check_only:
        return Path(DEFAULT_FUNASR_PATH).expanduser()
    if downloader is None:
        from modelscope import snapshot_download

        downloader = snapshot_download
    return Path(downloader(model_id=model_id, revision=revision))


def _component_specs(args: argparse.Namespace) -> tuple[CacheComponent, ...]:
    """Build validation contracts from resolved component paths."""
    omnivoice_path = (
        args.omnivoice_path
        if args.omnivoice_path is not None
        else _download_huggingface(args.omnivoice_id, args.check_only)
    )
    audio_tokenizer_path = (
        args.audio_tokenizer_path
        if args.audio_tokenizer_path is not None
        else _download_huggingface(args.audio_tokenizer_id, args.check_only)
    )
    funasr_path = (
        args.funasr_model_path
        if args.funasr_model_path is not None
        else _download_funasr(
            args.funasr_model_id,
            args.funasr_revision,
            args.check_only,
        )
    )
    return (
        CacheComponent(
            "omnivoice",
            args.omnivoice_id,
            Path(omnivoice_path),
            ("config.json", "tokenizer_config.json"),
            ("*.safetensors", "*.bin"),
        ),
        CacheComponent(
            "audio_tokenizer",
            args.audio_tokenizer_id,
            Path(audio_tokenizer_path),
            ("config.json", "preprocessor_config.json"),
            ("*.safetensors", "*.bin"),
        ),
        CacheComponent(
            "funasr_fa_zh",
            f"{args.funasr_model_id}@{args.funasr_revision}",
            Path(funasr_path),
            ("model.pt",),
            ("*.pt",),
        ),
        CacheComponent(
            "whisperx_zh",
            str(args.whisperx_model_path),
            args.whisperx_model_path,
            ("config.json", "preprocessor_config.json", "tokenizer_config.json"),
            ("*.safetensors", "*.bin"),
        ),
    )


def _error_result(name: str, source: str, error: Exception) -> CacheCheckResult:
    """Convert a download or path-resolution exception to a manifest result."""
    return CacheCheckResult(name, source, "", False, 0, (), (), str(error))


def prepare_cache(args: argparse.Namespace) -> int:
    """Resolve, validate, print, and optionally manifest all cache components."""
    if args.check_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    try:
        components = _component_specs(args)
        results = tuple(validate_component(component) for component in components)
    except Exception as error:
        result = _error_result("cache_resolution", "local cache", error)
        results = (result,)

    manifest = build_manifest(results)
    for result in results:
        status = "ok" if result.ok else "failed"
        detail = result.error or f"{result.file_count} file(s)"
        print(f"[{status}] {result.name}: {result.path} ({detail})")

    if not manifest["ok"]:
        return 1
    if args.output_manifest is not None:
        args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.output_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Manifest written to {args.output_manifest}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the offline cache preparation command."""
    return prepare_cache(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
