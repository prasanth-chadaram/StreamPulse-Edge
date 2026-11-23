"""Fixed numeric telemetry arrays shared between processes.

Constant allocation, no frame pickling, O(1) snapshots. Inference writes; the
web layer and soak test read.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from typing import Dict

from .types import StreamSnapshot, SystemSnapshot

STREAM_FIELDS = (
    "fps", "inf_fps", "latency_ms", "latency_p95", "dropped_pct",
    "decoded_total", "processed_total", "dropped_total", "counted",
    "anomalies", "active_tracks", "connected", "width", "height",
    "last_error_code",
)
GLOBAL_FIELDS = (
    "cpu_percent", "rss_bytes", "ram_percent", "mem_available_mb",
    "uptime_s", "inference_ms", "batch_size", "skip_ratio",
    "state_code", "throttle_code", "gc_collections", "total_counted",
    "total_anomalies",
)

STATE_NAMES = {0: "NOMINAL", 1: "THROTTLED", 2: "CRITICAL"}
THROTTLE_NAMES = {0: "", 1: "CPU", 2: "RSS", 3: "RAM", 4: "CPU+RSS",
                  5: "CPU+RAM", 6: "RSS+RAM", 7: "CPU+RSS+RAM"}


class Telemetry:
    def __init__(self, stream_names):
        self.names = list(stream_names)
        self.sources = {n: "" for n in self.names}
        self._n = len(self.names)
        self._sf = {f: i for i, f in enumerate(STREAM_FIELDS)}
        self._gf = {f: i for i, f in enumerate(GLOBAL_FIELDS)}
        self.stream_arr = mp.Array("d", self._n * len(STREAM_FIELDS), lock=True)
        self.global_arr = mp.Array("d", len(GLOBAL_FIELDS), lock=True)
        self._t0 = time.monotonic()

    # ---- writes -----------------------------------------------------------
    def set_stream(self, idx: int, field: str, value: float) -> None:
        i = idx * len(STREAM_FIELDS) + self._sf[field]
        with self.stream_arr.get_lock():
            self.stream_arr[i] = float(value)

    def add_stream(self, idx: int, field: str, delta: float) -> None:
        i = idx * len(STREAM_FIELDS) + self._sf[field]
        with self.stream_arr.get_lock():
            self.stream_arr[i] += float(delta)

    def set_global(self, field: str, value: float) -> None:
        with self.global_arr.get_lock():
            self.global_arr[self._gf[field]] = float(value)

    # ---- reads ------------------------------------------------------------
    def get_stream(self, idx: int, field: str) -> float:
        i = idx * len(STREAM_FIELDS) + self._sf[field]
        with self.stream_arr.get_lock():
            return self.stream_arr[i]

    def get_global(self, field: str) -> float:
        with self.global_arr.get_lock():
            return self.global_arr[self._gf[field]]

    def snapshot(self) -> SystemSnapshot:
        streams: Dict[str, StreamSnapshot] = {}
        with self.stream_arr.get_lock():
            raw = list(self.stream_arr)
        width = len(STREAM_FIELDS)
        for idx, name in enumerate(self.names):
            s = raw[idx * width:(idx + 1) * width]
            streams[name] = StreamSnapshot(
                name=name,
                connected=bool(s[self._sf["connected"]]),
                source=self.sources.get(name, ""),
                width=int(s[self._sf["width"]]),
                height=int(s[self._sf["height"]]),
                fps=s[self._sf["fps"]],
                inference_fps=s[self._sf["inf_fps"]],
                e2e_latency_ms=s[self._sf["latency_ms"]],
                latency_p95_ms=s[self._sf["latency_p95"]],
                dropped_pct=s[self._sf["dropped_pct"]],
                decoded_total=int(s[self._sf["decoded_total"]]),
                processed_total=int(s[self._sf["processed_total"]]),
                dropped_total=int(s[self._sf["dropped_total"]]),
                objects_counted=int(s[self._sf["counted"]]),
                anomalies=int(s[self._sf["anomalies"]]),
                active_tracks=int(s[self._sf["active_tracks"]]),
            )
        with self.global_arr.get_lock():
            g = list(self.global_arr)
        gv = {f: g[i] for f, i in self._gf.items()}
        return SystemSnapshot(
            cpu_percent=gv["cpu_percent"],
            rss_bytes=int(gv["rss_bytes"]),
            rss_mb=round(gv["rss_bytes"] / (1024 * 1024), 1),
            ram_percent=gv["ram_percent"],
            mem_available_mb=gv["mem_available_mb"],
            uptime_s=gv["uptime_s"] or (time.monotonic() - self._t0),
            backpressure_state=STATE_NAMES.get(int(gv["state_code"]), "NOMINAL"),
            skip_ratio=int(gv["skip_ratio"]),
            throttle_reason=THROTTLE_NAMES.get(int(gv["throttle_code"]), ""),
            inference_ms=gv["inference_ms"],
            batch_size=int(gv["batch_size"]),
            stream_count=self._n,
            total_counted=int(gv["total_counted"]),
            total_anomalies=int(gv["total_anomalies"]),
            gc_collections=int(gv["gc_collections"]),
            streams=streams,
        )
