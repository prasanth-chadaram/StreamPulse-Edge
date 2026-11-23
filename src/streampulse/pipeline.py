"""Top-level orchestration: rings, ingest threads, inference process, web."""

from __future__ import annotations

import ctypes
import gc
import logging
import multiprocessing as mp
import threading
import time
from typing import List

import psutil

from .backpressure import BackpressureController, PressureSample
from .config import AppConfig
from .detector import InferenceWorker
from .ingestion import IngestorPool
from .output_frames import OutputFrames
from .ring_buffer import SharedFrameRing
from .telemetry import Telemetry
from .types import SystemSnapshot

log = logging.getLogger("streampulse.pipeline")


class Pipeline:
    def __init__(self, cfg: AppConfig, start_web: bool = True):
        self.cfg = cfg
        self.streams = cfg.resolved_sources()
        self.n = len(self.streams)
        if self.n == 0:
            raise ValueError("No enabled streams configured")
        names = [s.name for s in self.streams]
        self.telemetry = Telemetry(names)
        for s in self.streams:
            self.telemetry.sources[s.name] = s.source

        w, h = cfg.pipeline.frame_width, cfg.pipeline.frame_height
        self.rings: List[SharedFrameRing] = [
            SharedFrameRing(f"sp_ring_{s.name}_{id(self)}", w, h, cfg.pipeline.ring_capacity)
            for s in self.streams
        ]
        self.outputs = OutputFrames(self.n, w, h)

        self.stop_event = mp.Event()
        self.skip_value = mp.Value("i", 0)
        self.interval_value = mp.Value("d", cfg.pipeline.inference_interval_s)
        self.controller = BackpressureController(cfg.backpressure)

        self.ingestors = IngestorPool(self.streams, cfg.pipeline, self.rings, self.telemetry)
        self.worker = InferenceWorker(
            cfg.detector, cfg.pipeline, self.streams, self.rings, self.outputs,
            self.telemetry, self.stop_event, self.skip_value, self.interval_value,
        )
        self._monitor_stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._proc = psutil.Process()
        psutil.cpu_percent(interval=None)  # prime the percentage accumulator
        self._start_time = time.monotonic()
        self._last_trim = self._start_time
        self._web_thread = None
        self._web_server = None
        self.start_web = start_web and cfg.web.enabled

    # ---- lifecycle --------------------------------------------------------
    def start(self) -> None:
        log.info("Starting %d streams (%s)", self.n,
                 ", ".join(s.name for s in self.streams))
        self.ingestors.start_all()
        self.worker.start()
        self._monitor_thread = threading.Thread(target=self._monitor, name="monitor", daemon=True)
        self._monitor_thread.start()
        if self.start_web:
            self._start_web()
        self.telemetry.set_global("uptime_s", 0)

    def stop(self) -> None:
        log.info("Stopping pipeline")
        self._monitor_stop.set()
        self.stop_event.set()
        self.ingestors.stop_all()
        if self.worker.is_alive():
            self.worker.join(timeout=5.0)
        if self.worker.is_alive():
            log.warning("Inference worker did not exit cleanly; terminating")
            self.worker.terminate()
            self.worker.join(timeout=2.0)
        if self._web_server is not None:
            self._web_server.should_exit = True
        if self._web_thread is not None:
            self._web_thread.join(timeout=5.0)
        if self._monitor_thread:
            self._monitor_thread.join(timeout=3.0)
        for r in self.rings:
            r.close(); r.unlink()
        self.outputs.close(); self.outputs.unlink()
        log.info("Pipeline stopped cleanly")

    def snapshot(self) -> SystemSnapshot:
        return self.telemetry.snapshot()

    def _rss_mb(self) -> float:
        total = self._proc.memory_info().rss
        try:
            for child in self._proc.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except psutil.Error:
                    pass
        except psutil.Error:
            pass
        return total / (1024 * 1024)

    def _monitor(self) -> None:
        interval = self.cfg.backpressure.control_interval_s
        while not self._monitor_stop.wait(interval):
            try:
                cpu = psutil.cpu_percent(interval=None)
                rss_mb = self._rss_mb()
                vm = psutil.virtual_memory()
                sample = PressureSample(cpu, rss_mb, vm.percent, vm.available / (1024 * 1024))
                skip = self.controller.update(sample)
                self.skip_value.value = skip

                self.telemetry.set_global("cpu_percent", cpu)
                self.telemetry.set_global("rss_bytes", rss_mb * 1024 * 1024)
                self.telemetry.set_global("ram_percent", vm.percent)
                self.telemetry.set_global("mem_available_mb", vm.available / (1024 * 1024))
                self.telemetry.set_global("uptime_s", time.monotonic() - self._start_time)
                self.telemetry.set_global("skip_ratio", skip)
                self.telemetry.set_global("state_code", self.controller.state_code)
                self.telemetry.set_global("throttle_code", self.controller.throttle_code)
                self.telemetry.set_global("gc_collections",
                                          sum(s.get("collections", s.get("count", 0))
                                              for s in gc.get_stats()))
                total_c = total_a = 0
                for i in range(self.n):
                    total_c += int(self.telemetry.get_stream(i, "counted"))
                    total_a += int(self.telemetry.get_stream(i, "anomalies"))
                self.telemetry.set_global("total_counted", total_c)
                self.telemetry.set_global("total_anomalies", total_a)

                # OpenCV / ONNX allocate through glibc in native code. Free'd
                # pages can stay in per-thread arenas and show up as apparent
                # RSS growth during a soak. A periodic trim returns them to the
                # OS so the monitored curve stays flat. Bounded rings already
                # guarantee there is no unbounded Python object growth.
                now = time.monotonic()
                if now - self._last_trim >= 60.0:
                    gc.collect()
                    try:
                        ctypes.CDLL("libc.so.6").malloc_trim(0)
                    except Exception:
                        pass
                    self._last_trim = now
            except Exception as exc:
                log.exception("Monitor tick failed: %s", exc)

    def _start_web(self) -> None:
        import uvicorn
        from .web import create_app, set_pipeline
        set_pipeline(self)
        app = create_app(self)
        config = uvicorn.Config(app, host=self.cfg.web.host, port=self.cfg.web.port,
                                log_level="warning", access_log=False,
                                loop="asyncio", workers=1)
        self._web_server = uvicorn.Server(config)
        self._web_thread = threading.Thread(target=self._web_server.run,
                                            name="web", daemon=True)
        self._web_thread.start()
        log.info("Web dashboard on http://%s:%d", self.cfg.web.host, self.cfg.web.port)
