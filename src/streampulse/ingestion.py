"""Per-camera ingestion threads.

OpenCV decode blocks in native code with the GIL released, so threads are the
right primitive here. Each frame only costs a memcpy into the shared ring.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import List

import cv2
import numpy as np

from .config import StreamConfig, PipelineConfig
from .ring_buffer import SharedFrameRing
from .telemetry import Telemetry

log = logging.getLogger("streampulse.ingest")


class _SyntheticCamera:
    """Deterministic VisA-PCB-like frame generator. Zero external media needed."""

    def __init__(self, seed: int, w: int, h: int):
        self.seed = seed
        self.w = w
        self.h = h
        self.t = 0.0
        rng = np.random.default_rng(seed)
        # Static PCB texture generated once, reused every frame.
        base = np.full((h, w, 3), (26, 46, 30), dtype=np.uint8)  # dark green
        for _ in range(24):
            x = int(rng.integers(0, w))
            y = int(rng.integers(0, h))
            cv2.line(base, (x, y), (int(rng.integers(0, w)), int(rng.integers(0, h))),
                     (38, 92, 52), int(rng.integers(1, 3)), cv2.LINE_AA)
        for _ in range(40):
            x, y = int(rng.integers(10, w - 10)), int(rng.integers(10, h - 10))
            cv2.circle(base, (x, y), int(rng.integers(2, 5)), (150, 130, 70), -1)
        cv2.rectangle(base, (4, 4), (w - 5, h - 5), (60, 110, 70), 2)
        self.base = base

    def frame(self, dt: float) -> np.ndarray:
        self.t += dt
        img = self.base.copy()
        # Two moving "components" crossing the virtual counting line.
        for k in range(2):
            speed = 42 + 9 * k + 3 * (self.seed % 3)
            x = int((self.t * speed + k * 310 + self.seed * 57) % (self.w + 120)) - 60
            y = int(self.h * (0.34 + 0.24 * ((self.seed + k) % 3)))
            color = ((60, 60, 220), (60, 200, 200))[k]
            cv2.rectangle(img, (x, y), (x + 46, y + 32), color, -1)
            cv2.rectangle(img, (x + 38, y + 8), (x + 54, y + 24), (200, 200, 90), -1)
        return img


class StreamIngestor:
    def __init__(self, idx: int, cfg: StreamConfig, pcfg: PipelineConfig,
                 ring: SharedFrameRing, telemetry: Telemetry):
        self.idx = idx
        self.cfg = cfg
        self.pcfg = pcfg
        self.ring = ring
        self.telemetry = telemetry
        self.source = cfg.source
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self._fps_window: List[float] = []
        self._synthetic = _SyntheticCamera(seed=idx + 1, w=pcfg.frame_width,
                                            h=pcfg.frame_height) if cfg.source == "synthetic" else None
        self._cap = None

    def start(self) -> None:
        self.telemetry.set_stream(self.idx, "connected", 0)
        self.telemetry.set_stream(self.idx, "width", self.pcfg.frame_width)
        self.telemetry.set_stream(self.idx, "height", self.pcfg.frame_height)
        self.thread = threading.Thread(target=self._run, name=f"ingest-{self.cfg.name}",
                                       daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.telemetry.set_stream(self.idx, "connected", 0)

    def _open_cap(self):
        try:
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.pcfg.frame_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.pcfg.frame_height)
                cap.set(cv2.CAP_PROP_FPS, self.pcfg.ingest_fps)
                return cap
            cap.release()
        except Exception as exc:  # pragma: no cover - depends on ffmpeg build
            log.warning("[%s] open failed: %s", self.cfg.name, exc)
        return None

    def _run(self) -> None:
        period = 1.0 / max(1.0, self.pcfg.ingest_fps)
        last = time.monotonic()
        last_fps_tick = last
        frame_count = 0
        cap_opened_at = last
        cap_frames = 0

        while not self.stop_event.is_set():
            now = time.monotonic()
            wait = period - (now - last)
            if wait > 0:
                # Event.wait releases the GIL and is interruptible by stop().
                if self.stop_event.wait(wait):
                    break
            last = time.monotonic()

            try:
                if self._synthetic is not None:
                    frame = self._synthetic.frame(period)
                    ok = True
                else:
                    if self._cap is None:
                        self._cap = self._open_cap()
                        if self._cap is None:
                            self.telemetry.set_stream(self.idx, "connected", 0)
                            if self.stop_event.wait(2.0):
                                break
                            continue
                        cap_opened_at, cap_frames = time.monotonic(), 0
                    ok, frame = self._cap.read()
                    if not ok:
                        self._cap.release()
                        self._cap = None
                        self.telemetry.set_stream(self.idx, "connected", 0)
                        continue
                    cap_frames += 1
                    # FFmpeg's UDP/MPEG-TS decoder keeps native packet buffers
                    # that grow slowly when a stream runs for hours. Recycle
                    # the capture handle periodically so native memory stays
                    # bounded; the ring buffer hides the sub-second reopen.
                    if cap_frames > 300 or time.monotonic() - cap_opened_at > 10.0:
                        self._cap.release()
                        self._cap = None

                if frame is None:
                    continue
                if frame.shape[1] != self.pcfg.frame_width or frame.shape[0] != self.pcfg.frame_height:
                    frame = cv2.resize(frame, (self.pcfg.frame_width, self.pcfg.frame_height),
                                       interpolation=cv2.INTER_AREA)

                self.ring.write(np.ascontiguousarray(frame))
                frame_count += 1
                self.telemetry.add_stream(self.idx, "decoded_total", 1)
                self.telemetry.set_stream(self.idx, "connected", 1)

                # Sliding 2-second FPS estimate (bounded list -> no leak).
                self._fps_window.append(time.monotonic())
                if now - last_fps_tick >= 1.0:
                    cutoff = now - 2.0
                    self._fps_window = [t for t in self._fps_window if t >= cutoff]
                    self.telemetry.set_stream(self.idx, "fps", len(self._fps_window) / 2.0)
                    last_fps_tick = now
            except Exception as exc:  # keep the thread alive no matter what
                log.exception("[%s] ingest error: %s", self.cfg.name, exc)
                self.telemetry.set_stream(self.idx, "connected", 0)
                if self.stop_event.wait(1.0):
                    break


class IngestorPool:
    def __init__(self, configs: List[StreamConfig], pcfg: PipelineConfig,
                 rings: List[SharedFrameRing], telemetry: Telemetry):
        self.ingestors = [
            StreamIngestor(i, c, pcfg, rings[i], telemetry)
            for i, c in enumerate(configs)
        ]

    def start_all(self) -> None:
        for ing in self.ingestors:
            ing.start()

    def stop_all(self) -> None:
        for ing in self.ingestors:
            ing.stop()
