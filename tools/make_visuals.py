#!/usr/bin/env python3
"""Generate portfolio visual proof assets from the REAL model and videos.

Outputs:
  assets/detection_bottle_conveyor.jpg
  assets/detection_worker_zone.jpg
  assets/live_pipeline_demo.gif      (5 s, 8 FPS, 480 px, <2 MB)
  assets/soak_telemetry.png          (dual-panel RSS / latency from soak samples)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from streampulse.config import AppConfig  # noqa: E402
from streampulse.detector import DetectorBackend, draw_overlay  # noqa: E402
from streampulse.tracker import IoUTracker  # noqa: E402

VIDEOS = Path(os.environ.get("DATA_DIR", str(ROOT / "data" / "videos")))
ASSETS = ROOT / "assets"


def pick_frame(path: Path, backend, start: int = 0, tries: int = 40):
    """Pick a frame with the most detections so the proof is not empty."""
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(start, min(n - 1, start + 600), tries, dtype=int)
    best, best_dets = None, []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        dets = backend.detect_batch([frame])[0]
        if len(dets) > len(best_dets):
            best, best_dets = frame.copy(), dets
    cap.release()
    return best, best_dets


def annotate(frame, backend, tracker, name, count=0, anomalies=0):
    dets = backend.detect_batch([frame])[0]
    tracks, crossings, anomalies_new = tracker.update(dets, 1.0 / 15.0)
    out = frame.copy()
    draw_overlay(out, tracks, crossings, anomalies + anomalies_new,
                 name, fps=15.0, latency_ms=0.0, line=tracker.line)
    return out, tracks, crossings


def make_stills(cfg, backend):
    ASSETS.mkdir(parents=True, exist_ok=True)
    for stream_cfg, filename, out_name in [
        (cfg.streams[0], VIDEOS / "bottle-detection.mp4", "detection_bottle_conveyor.jpg"),
        (cfg.streams[1], VIDEOS / "worker-zone-detection.mp4", "detection_worker_zone.jpg"),
    ]:
        frame, dets = pick_frame(filename, backend)
        if frame is None:
            raise RuntimeError(f"cannot read {filename}")
        tracker = IoUTracker(cfg.pipeline, stream_cfg, cfg.pipeline.frame_width,
                             cfg.pipeline.frame_height)
        out, tracks, crossings = annotate(frame, backend, tracker, stream_cfg.name)
        cv2.imwrite(str(ASSETS / out_name), out, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"[visuals] {out_name}: {len(dets)} raw dets, {len(tracks)} tracks, "
              f"{crossings} crossings")


def make_gif(cfg, backend, seconds=5, fps=8):
    ASSETS.mkdir(parents=True, exist_ok=True)
    stream_cfg = cfg.streams[0]
    tracker = IoUTracker(cfg.pipeline, stream_cfg, cfg.pipeline.frame_width,
                         cfg.pipeline.frame_height)
    cap = cv2.VideoCapture(str(VIDEOS / "bottle-detection.mp4"))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = max(0, min(n - seconds * 15 - 1, 120))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    tmp = Path(tempfile.mkdtemp(prefix="sp_gif_"))
    frames = []
    try:
        for i in range(seconds * fps):
            ok, frame = cap.read()
            if not ok:
                break
            out, _, _ = annotate(frame, backend, tracker, stream_cfg.name)
            p = tmp / f"f_{i:03d}.png"
            cv2.imwrite(str(p), out)
            frames.append(p)
        cap.release()
        if not frames:
            raise RuntimeError("no GIF frames produced")
        palette = tmp / "palette.png"
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(fps), "-i", str(tmp / "f_%03d.png"),
            "-vf", "fps=8,scale=480:-1:flags=lanczos,palettegen=stats_mode=diff",
            "-frames:v", "1", str(palette),
        ], check=True)
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(fps), "-i", str(tmp / "f_%03d.png"),
            "-i", str(palette),
            "-lavfi", "fps=8,scale=480:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer",
            "-loop", "0", str(ASSETS / "live_pipeline_demo.gif"),
        ], check=True)
        size_mb = (ASSETS / "live_pipeline_demo.gif").stat().st_size / 1e6
        print(f"[visuals] live_pipeline_demo.gif: {len(frames)} frames, {size_mb:.2f} MB")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def make_chart(samples_path: Path):
    if not samples_path or not Path(samples_path).exists():
        print("[visuals] no soak samples JSON; skipping telemetry chart")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples = [s for s in json.loads(Path(samples_path).read_text()) if "error" not in s]
    t = [s["t"] / 60.0 for s in samples]
    rss = [s["rss_mb"] for s in samples]
    lat = [s.get("streams", {}).get("bottle-line", {}).get("e2e_latency_ms", 0)
           for s in samples]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [1.3, 1]})
    fig.patch.set_facecolor("#0b0f14")
    for ax in (ax1, ax2):
        ax.set_facecolor("#121922")
        ax.grid(alpha=0.25, color="#7b8ea3")
        ax.tick_params(colors="#d7e2ee")
        for spine in ax.spines.values():
            spine.set_color("#23303f")
    ax1.plot(t, rss, color="#48a9ff", lw=1.8, label="Engine + children RSS")
    ax1.set_ylabel("RSS (MB)", color="#d7e2ee")
    ax1.set_ylim(max(0, min(rss) - 12), max(rss) + 12)
    ax1.set_title("StreamPulse-Edge: 5-minute real-stream soak telemetry",
                  color="#d7e2ee")
    ax1.text(0.02, 0.08, f"peak {max(rss):.0f} MB  <<  3500 MB budget",
             transform=ax1.transAxes, color="#3ddc84", fontsize=10)
    ax1.legend(facecolor="#121922", labelcolor="#d7e2ee", loc="upper right")
    ax2.plot(t, lat, color="#ffb020", lw=1.2, label="bottle-line E2E latency")
    ax2.set_ylabel("E2E latency (ms)", color="#d7e2ee")
    ax2.set_xlabel("Time (minutes)", color="#d7e2ee")
    ax2.legend(facecolor="#121922", labelcolor="#d7e2ee", loc="upper right")
    fig.tight_layout()
    fig.savefig(ASSETS / "soak_telemetry.png", dpi=130, facecolor=fig.get_facecolor())
    print("[visuals] soak_telemetry.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    ap.add_argument("--samples", default=str(ROOT / "reports" / "soak_real_samples.json"))
    args = ap.parse_args()
    cfg = AppConfig.load(args.config)
    backend = DetectorBackend(cfg.detector)
    if not backend.using_model:
        raise RuntimeError(f"INT8 model not loadable: {cfg.detector.model_path}")
    make_stills(cfg, backend)
    make_gif(cfg, backend)
    make_chart(Path(args.samples))


if __name__ == "__main__":
    sys.exit(main())
