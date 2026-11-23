#!/usr/bin/env python3
"""Quantize the downloaded YOLO11n FP32 ONNX model to INT8 and verify it.

Strategy (automatic, in order):
  1. static QDQ INT8 calibrated on 60 real frames from both Intel videos;
  2. static QDQ with the final Detect head excluded (keeps box scores in FP32);
  3. quantize_dynamic (weights INT8, activations FP32) as a last resort.

Every candidate is evaluated with the SAME decoder used in production
(``streampulse.detector.DetectorBackend``): detection count, mean confidence,
coordinate sanity, and single-frame CPU latency are compared against FP32.
A candidate is rejected if it collapses confidence or drops detections.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                      QuantType, quantize_dynamic, quantize_static)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from streampulse.config import DetectorConfig  # noqa: E402
from streampulse.detector import DetectorBackend, letterbox  # noqa: E402

COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]
VIDEO_NAMES = ["bottle-detection.mp4", "worker-zone-detection.mp4"]


class FrameCalibrator(CalibrationDataReader):
    def __init__(self, frames, input_name: str):
        self.frames = list(frames)
        self.input_name = input_name
        self.i = 0

    def get_next(self):
        if self.i >= len(self.frames):
            return None
        img = self.frames[self.i]
        self.i += 1
        return {self.input_name: img}


def extract_frames(videos_dir: Path, count: int, size: int = 416):
    frames, sources = [], []
    for name in VIDEO_NAMES:
        path = videos_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n = max(1, n)
        per = max(1, count // 2)
        idxs = np.linspace(0, n - 1, per, dtype=int)
        for k in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
            ok, frame = cap.read()
            if not ok:
                continue
            lb, _, _, _ = letterbox(frame, size, size)
            rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
            # Calibration reader must feed the exact 1x3x416x416 model input.
            frames.append(np.expand_dims(rgb.transpose(2, 0, 1), 0).astype(np.float32) / 255.0)
            sources.append((name, frame.copy()))
        cap.release()
    if not frames:
        raise RuntimeError("no calibration frames extracted")
    return frames, sources


def detect_head_nodes(model_path: Path):
    """Nodes producing / feeding the final output -- excluded in fallback #2."""
    graph = onnx.load(str(model_path)).graph
    out_names = {o.name for o in graph.output}
    excluded = set()
    frontier = [n for n in graph.node if out_names.intersection(n.output)]
    for _ in range(6):
        nxt = []
        for n in frontier:
            excluded.add(n.name)
            for inp in n.input:
                for producer in graph.node:
                    if inp in producer.output and producer.name not in excluded:
                        nxt.append(producer)
        frontier = nxt
    return sorted(excluded)


def cfg_for(model: Path) -> DetectorConfig:
    return DetectorConfig(model_path=str(model), backend="onnxruntime",
                          input_size=416, conf_threshold=0.25,
                          iou_threshold=0.45, num_threads=2,
                          class_names=COCO80)


def evaluate(model: Path, frames, conf_threshold: float = 0.25):
    backend = DetectorBackend(cfg_for(model))
    if not backend.using_model:
        raise RuntimeError(f"could not load {model}")
    dets_per_frame, latencies = [], []
    for f in frames:
        img = (np.clip(f[0], 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        t0 = time.perf_counter()
        dets = backend.detect_batch([img])[0]
        latencies.append((time.perf_counter() - t0) * 1000)
        dets_per_frame.append(dets)
    counts = [len(d) for d in dets_per_frame]
    confs = [d.score for dets in dets_per_frame for d in dets]
    finite = all(np.isfinite(d.bbox).all() for dets in dets_per_frame for d in dets)
    in_range = all(0 <= x <= 416 and 0 <= y <= 416
                   for dets in dets_per_frame for d in dets
                   for x, y in ((d.bbox[0], d.bbox[1]), (d.bbox[2], d.bbox[3])))
    return {
        "detections": sum(counts),
        "frames_with_detections": sum(c > 0 for c in counts),
        "mean_conf": float(np.mean(confs)) if confs else 0.0,
        "max_conf": float(np.max(confs)) if confs else 0.0,
        "latency_ms": float(np.median(latencies)),
        "finite": bool(finite),
        "in_range": bool(in_range),
        "classes": sorted({d.class_name for dets in dets_per_frame for d in dets}),
    }


def acceptable(cand, ref) -> bool:
    if not cand["finite"] or not cand["in_range"]:
        return False
    # Class scores are sigmoid outputs in [0, 1]. Values > 1 mean the QDQ
    # graph corrupted the Detect head; reject rather than ship garbage boxes.
    if cand["max_conf"] > 1.001 or cand["mean_conf"] > 1.001:
        return False
    if ref["detections"] == 0:
        return cand["detections"] > 0
    return (cand["detections"] >= 0.6 * ref["detections"]
            and cand["frames_with_detections"] >= 0.6 * ref["frames_with_detections"]
            and cand["mean_conf"] >= 0.6 * ref["mean_conf"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos-dir", default="/mnt/data/datasets/streampulse_videos")
    ap.add_argument("--models-dir", default="/mnt/data/models/streampulse")
    ap.add_argument("--fp32", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--calib-frames", type=int, default=60)
    args = ap.parse_args()

    videos_dir, models_dir = Path(args.videos_dir), Path(args.models_dir)
    fp32 = Path(args.fp32) if args.fp32 else models_dir / "yolo11n_416.onnx"
    output = Path(args.output) if args.output else models_dir / "yolo11n_416_int8.onnx"
    if not fp32.exists():
        raise FileNotFoundError(f"run tools/download_assets.py first: {fp32}")

    frames, sources = extract_frames(videos_dir, args.calib_frames)
    print(f"[quant] {len(frames)} calibration frames from {len(VIDEO_NAMES)} videos")

    graph = onnx.load(str(fp32)).graph
    inp_name = graph.input[0].name
    print(f"[quant] FP32 graph input={inp_name} output={[o.name for o in graph.output]}")

    print("[quant] evaluating FP32 reference ...")
    ref = evaluate(fp32, frames)
    print(f"[quant] FP32 {ref}")

    created, chosen, report = [], None, None
    head_nodes = detect_head_nodes(fp32)

    def try_static(tag, nodes_to_exclude, dst, fmt, act):
        label = "QDQ" if fmt == QuantFormat.QDQ else "QOperator"
        print(f"[quant] static {label} INT8 ({tag}) ...")
        quantize_static(str(fp32), str(dst),
                        calibration_data_reader=FrameCalibrator(frames, inp_name),
                        quant_format=fmt,
                        activation_type=act,
                        weight_type=QuantType.QInt8,
                        nodes_to_exclude=nodes_to_exclude,
                        per_channel=False, reduce_range=True,
                        extra_options={"MinimumCalibrationPoints": 32})
        r = evaluate(dst, frames)
        print(f"[quant] {tag}: {r}")
        created.append((tag, dst, r))
        return r

    # 1. QOperator (QLinearOps) static INT8 -- fastest on Ice Lake CPUs.
    #    Full-model quantization often corrupts the sigmoid Detect head, so the
    #    verifier rejects it if class scores leave [0, 1].
    for tag, excl, dst, name in [
        ("qoperator-full", [], models_dir / "yolo11n_416_int8_full.onnx", "static-qoperator-int8"),
        ("qoperator-head-fp32", head_nodes,
         models_dir / "yolo11n_416_int8_headfp32.onnx", "static-qoperator-int8-head-fp32"),
    ]:
        if chosen is not None:
            break
        try:
            r = try_static(tag, excl, dst, QuantFormat.QOperator, QuantType.QUInt8)
            if acceptable(r, ref):
                chosen, report = name, r
        except Exception as exc:
            print(f"[quant] {tag} failed: {type(exc).__name__}: {exc}")

    # 2. QDQ static INT8 with the Detect head excluded (slower but accurate).
    if chosen is None:
        try:
            dst = models_dir / "yolo11n_416_int8_qdq.onnx"
            r = try_static("qdq-head-fp32", head_nodes, dst, QuantFormat.QDQ, QuantType.QInt8)
            if acceptable(r, ref):
                chosen, report = "static-qdq-int8-head-fp32", r
        except Exception as exc:
            print(f"[quant] QDQ failed: {type(exc).__name__}: {exc}")

    # 3. dynamic INT8 (weights only) as a last resort.
    if chosen is None:
        try:
            print("[quant] dynamic INT8 fallback ...")
            dst = models_dir / "yolo11n_416_int8_dynamic.onnx"
            quantize_dynamic(str(fp32), str(dst), weight_type=QuantType.QInt8,
                             extra_options={"MinimumCalibrationPoints": 32})
            r = evaluate(dst, frames)
            print(f"[quant] dynamic: {r}")
            created.append(("dynamic", dst, r))
            if acceptable(r, ref):
                chosen, report = "dynamic-int8", r
        except Exception as exc:
            print(f"[quant] dynamic failed: {type(exc).__name__}: {exc}")

    if chosen is None:
        # Nothing passed verification: keep FP32 active rather than ship garbage.
        print("[quant] WARNING: no INT8 candidate passed verification; using FP32")
        chosen, report = "fp32-fallback", ref
        output = fp32
    else:
        chosen_path = next(p for tag, p, r in created if r is report)
        if chosen_path != output:
            chosen_path.replace(output)

    summary = {
        "strategy": chosen,
        "fp32": ref,
        "int8": report,
        "speedup": round(ref["latency_ms"] / max(0.01, report["latency_ms"]), 2),
        "output": str(output),
        "candidates": [{"tag": t, "path": str(p), **r} for t, p, r in created],
    }
    with open(models_dir / "quantization_report.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("[quant] summary:", json.dumps({k: summary[k] for k in
                                          ("strategy", "speedup", "output")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
