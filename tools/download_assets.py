#!/usr/bin/env python3
"""Download all real assets for StreamPulse-Edge onto the HDD (/mnt/data).

Downloads:
  * two CC-BY-4.0 industrial sample videos from Intel IoT DevKit
  * pretrained yolo11n_416.onnx (COCO 80) from Hugging Face vitorsalomao/yolo11n-onnx

Reusable and idempotent: existing files are verified and skipped. Raw video
bytes and ONNX weights never touch the SSD -- set --videos-dir / --models-dir
to relocate if the HDD is mounted elsewhere.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

VIDEOS = {
    "bottle-detection.mp4": (
        "https://github.com/intel-iot-devkit/sample-videos/raw/master/bottle-detection.mp4"
    ),
    "worker-zone-detection.mp4": (
        "https://github.com/intel-iot-devkit/sample-videos/raw/master/worker-zone-detection.mp4"
    ),
}
HF_REPO = "vitorsalomao/yolo11n-onnx"
HF_FILE = "yolo11n_416.onnx"
DEFAULT_VIDEOS = Path("/mnt/data/datasets/streampulse_videos")
DEFAULT_MODELS = Path("/mnt/data/models/streampulse")


def download_url(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 100_000:
        print(f"[assets] skip existing {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[assets] GET {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "streampulse-edge/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as fh:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r[assets] {done / 1e6:.1f}/{total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)
    return dest


def download_model(models_dir: Path) -> Path:
    expected = models_dir / HF_FILE
    if expected.exists() and expected.stat().st_size > 100_000:
        print(f"[assets] skip existing {expected} ({expected.stat().st_size / 1e6:.1f} MB)")
        return expected
    from huggingface_hub import hf_hub_download
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"[assets] HF {HF_REPO}/{HF_FILE}")
    path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE,
                           local_dir=str(models_dir))
    p = Path(path)
    if p != expected and p.exists():
        p.replace(expected)
    print(f"[assets] model -> {expected} ({expected.stat().st_size / 1e6:.1f} MB)")
    return expected


def verify_model(path: Path) -> None:
    import onnx
    m = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(m)
    inp = m.graph.input[0]
    out = m.graph.output[0]
    print(f"[assets] ONNX input={inp.name} {[d.dim_value for d in inp.type.tensor_type.shape.dim]} "
          f"output={out.name} {[d.dim_value for d in out.type.tensor_type.shape.dim]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS)
    ap.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    args = ap.parse_args()

    args.videos_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    for name, url in VIDEOS.items():
        download_url(url, args.videos_dir / name)
    model = download_model(args.models_dir)
    verify_model(model)
    print("[assets] done. Next: tools/quantize_yolo_int8.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
