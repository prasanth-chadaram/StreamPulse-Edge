"""Small, allocation-light data structures shared across threads/processes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2 in PIXEL coords


@dataclass(slots=True)
class Detection:
    bbox: BBox
    score: float
    class_id: int
    class_name: str = "object"


@dataclass(slots=True)
class Track:
    track_id: int
    bbox: BBox
    score: float
    class_id: int
    class_name: str
    age: int = 0
    misses: int = 0
    centroid: Tuple[float, float] = (0.0, 0.0)
    side: int = 0                 # which side of the virtual counting line
    dwell_s: float = 0.0
    counted: bool = False
    anomaly: bool = False
    anomaly_reported: bool = False


@dataclass(slots=True)
class StreamSnapshot:
    """Plain snapshot used by the web layer / soak test (no shared memory)."""

    name: str
    connected: bool = False
    source: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0
    inference_fps: float = 0.0
    e2e_latency_ms: float = 0.0
    latency_p95_ms: float = 0.0
    dropped_pct: float = 0.0
    decoded_total: int = 0
    processed_total: int = 0
    dropped_total: int = 0
    objects_counted: int = 0
    anomalies: int = 0
    active_tracks: int = 0
    last_error: str = ""


@dataclass(slots=True)
class SystemSnapshot:
    cpu_percent: float = 0.0
    rss_bytes: int = 0
    rss_mb: float = 0.0
    ram_percent: float = 0.0
    mem_available_mb: float = 0.0
    uptime_s: float = 0.0
    backpressure_state: str = "NOMINAL"
    skip_ratio: int = 0
    throttle_reason: str = ""
    inference_ms: float = 0.0
    batch_size: int = 0
    stream_count: int = 0
    total_counted: int = 0
    total_anomalies: int = 0
    gc_collections: int = 0
    streams: dict = field(default_factory=dict)
