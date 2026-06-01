#!/usr/bin/env python3
"""Download SegFace pretrained weights from HuggingFace (kartiknarayan/SegFace)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.segface_weights import hf_checkpoint_filename, local_weights_dir_name  # noqa: E402
from app.config import settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=settings.segface_model,
        help="Backbone name, e.g. mobilenet, swin_base, convnext_base",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=settings.segface_input_size,
        help="Input resolution used in checkpoint folder name",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub", file=sys.stderr)
        return 1

    hf_filename = hf_checkpoint_filename(args.model, args.size)
    dest_root = settings.models_dir / "segface"
    dest_root.mkdir(parents=True, exist_ok=True)

    path = hf_hub_download(
        repo_id="kartiknarayan/SegFace",
        filename=hf_filename,
        local_dir=str(dest_root),
    )
    downloaded = Path(path)
    local_dir = dest_root / local_weights_dir_name(args.model, args.size)
    local_dir.mkdir(parents=True, exist_ok=True)
    target = local_dir / "model_299.pt"
    if downloaded.resolve() != target.resolve():
        target.write_bytes(downloaded.read_bytes())
    print(f"Downloaded: {downloaded}")
    print(f"Installed:  {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
