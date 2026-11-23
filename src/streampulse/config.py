"""Configuration loading with sane, CPU-friendly defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import yaml


@dataclass
class StreamConfig:
    name: str
    source: str                 # rtsp://... , /dev/video0 , or "synthetic"
    enabled: bool = True
    roi_line: List[float] = field(default_factory=lambda: [0.5, 0.0, 0.5, 1.0])  # x1,y1,x2,y2 normalised, vertical conveyor line
    dwell_max_s: float = 6.0    # dwell longer than this inside ROI => anomaly
    anomaly_classes: List[str] = field(default_factory=list)


COCO_CLASSES = [
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


@dataclass
class DetectorConfig:
    # INT8 ONNX path; an empty or invalid path selects the synthetic detector.
    model_path: str = ""
    backend: str = "auto"       # auto | onnxruntime | synthetic
    input_size: int = 320       # 320x320 keeps INT8 inference cheap on i3 iGPU-less CPU
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    max_detections: int = 32
    num_threads: int = 2        # i3-1005G1 = 2c/4t; leave threads for decode + web
    class_names: List[str] = field(default_factory=lambda: list(COCO_CLASSES))


@dataclass
class PipelineConfig:
    ring_capacity: int = 3             # bounded latest-frame ring buffer depth per stream
    ingest_fps: float = 15.0           # target capture rate / simulator rate
    inference_interval_s: float = 0.0  # 0 = run as fast as backpressure allows
    frame_width: int = 640
    frame_height: int = 480
    tracker_max_age: int = 12          # frames before a missing track is deleted
    tracker_iou_threshold: float = 0.25
    line_hysteresis_px: float = 6.0
    latency_window: int = 300          # samples kept for p95 (bounded deque)


@dataclass
class BackpressureConfig:
    cpu_high_pct: float = 85.0
    cpu_low_pct: float = 70.0
    rss_high_mb: float = 3584.0        # 3.5 GB soft limit
    rss_critical_mb: float = 4096.0    # hard limit => emergency GC + max skip
    ram_high_pct: float = 85.0
    max_skip_ratio: int = 7            # process at most 1 of every 8 frames
    control_interval_s: float = 0.5
    recovery_interval_s: float = 2.0


@dataclass
class WebConfig:
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    mjpeg_fps: int = 12


@dataclass
class AppConfig:
    streams: List[StreamConfig]
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    backpressure: BackpressureConfig = field(default_factory=BackpressureConfig)
    web: WebConfig = field(default_factory=WebConfig)
    log_level: str = "INFO"

    @classmethod
    def load(cls, path: str) -> "AppConfig":
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        os.environ.setdefault("DATA_DIR", os.path.join(root, "data", "videos"))
        os.environ.setdefault("MODEL_DIR", os.path.join(root, "models"))
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        streams = [StreamConfig(**s) for s in raw.get("streams", [])]
        if not streams:
            streams = [
                StreamConfig(name="cam1", source="synthetic"),
                StreamConfig(name="cam2", source="synthetic",
                             roi_line=[0.0, 0.62, 1.0, 0.62]),
            ]
        detector = DetectorConfig(**raw.get("detector", {}))
        detector.model_path = os.path.expandvars(detector.model_path)
        return cls(
            streams=streams,
            detector=detector,
            pipeline=PipelineConfig(**raw.get("pipeline", {})),
            backpressure=BackpressureConfig(**raw.get("backpressure", {})),
            web=WebConfig(**raw.get("web", {})),
            log_level=raw.get("log_level", "INFO"),
        )

    def resolved_sources(self) -> List[StreamConfig]:
        """Expand ${ENV_VAR} references in source strings."""
        out = []
        for s in self.streams:
            if s.enabled:
                s.source = os.path.expandvars(s.source)
                out.append(s)
        return out
