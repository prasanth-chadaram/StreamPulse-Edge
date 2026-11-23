"""FastAPI observability layer: dashboard, Prometheus metrics, MJPEG, WebSocket."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict
from typing import AsyncGenerator

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from . import __version__

_PIPELINE = None

METRIC_PREFIX = "streampulse_"


def set_pipeline(pipeline) -> None:
    global _PIPELINE
    _PIPELINE = pipeline


def _snap():
    return _PIPELINE.snapshot()


def create_app(pipeline=None) -> FastAPI:
    if pipeline is not None:
        set_pipeline(pipeline)
    app = FastAPI(title="StreamPulse-Edge", version=__version__)

    # ---- static dashboard -------------------------------------------------
    html_path = os.path.join(os.path.dirname(__file__), "web", "index.html")

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        with open(html_path, "r", encoding="utf-8") as fh:
            return fh.read()

    @app.get("/api/health")
    async def health():
        snap = _snap()
        return {"status": "ok", "uptime_s": round(snap.uptime_s, 1),
                "streams": snap.stream_count, "version": __version__}

    @app.get("/api/stats")
    async def stats():
        s = _snap()
        return JSONResponse({
            "cpu_percent": round(s.cpu_percent, 1),
            "rss_mb": s.rss_mb,
            "ram_percent": round(s.ram_percent, 1),
            "uptime_s": round(s.uptime_s, 1),
            "backpressure_state": s.backpressure_state,
            "skip_ratio": s.skip_ratio,
            "throttle_reason": s.throttle_reason,
            "inference_ms": round(s.inference_ms, 2),
            "batch_size": s.batch_size,
            "total_counted": s.total_counted,
            "total_anomalies": s.total_anomalies,
            "gc_collections": s.gc_collections,
            "streams": {k: asdict(v) for k, v in s.streams.items()},
        })

    @app.get("/metrics")
    async def metrics():
        return Response(content=_prometheus(_snap()), media_type="text/plain; version=0.0.4")

    @app.get("/api/stream/{idx}/mjpeg")
    async def mjpeg(idx: int):
        if idx < 0 or idx >= _PIPELINE.n:
            return Response(status_code=404)
        fps = max(1, min(30, _PIPELINE.cfg.web.mjpeg_fps))

        async def gen() -> AsyncGenerator[bytes, None]:
            while True:
                frame = await asyncio.to_thread(_PIPELINE.outputs.read, idx)
                if frame is not None:
                    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
                    if ok:
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                               b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
                               + jpg.tobytes() + b"\r\n")
                await asyncio.sleep(1.0 / fps)

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/ws/stats")
    async def ws_stats(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                s = _snap()
                payload = {
                    "t": round(s.uptime_s, 1),
                    "cpu": round(s.cpu_percent, 1),
                    "rss_mb": s.rss_mb,
                    "state": s.backpressure_state,
                    "skip": s.skip_ratio,
                    "inf_ms": round(s.inference_ms, 2),
                    "total_counted": s.total_counted,
                    "total_anomalies": s.total_anomalies,
                    "streams": {k: {
                        "fps": round(v.fps, 1),
                        "inf_fps": round(v.inference_fps, 1),
                        "latency_ms": round(v.e2e_latency_ms, 1),
                        "latency_p95_ms": round(v.latency_p95_ms, 1),
                        "dropped_pct": round(v.dropped_pct, 1),
                        "counted": v.objects_counted,
                        "anomalies": v.anomalies,
                        "connected": v.connected,
                        "active_tracks": v.active_tracks,
                    } for k, v in s.streams.items()},
                }
                await ws.send_text(json.dumps(payload))
                await asyncio.sleep(0.5)
        except WebSocketDisconnect:
            return

    return app


def _prometheus(s) -> str:
    lines = [
        "# HELP streampulse_up Engine is running.",
        "# TYPE streampulse_up gauge",
        "streampulse_up 1",
        "# HELP streampulse_build_info Version information.",
        "# TYPE streampulse_build_info gauge",
        f'streampulse_build_info{{version="{__version__}"}} 1',
        "# HELP streampulse_cpu_percent CPU utilisation percent.",
        "# TYPE streampulse_cpu_percent gauge",
        f"streampulse_cpu_percent {s.cpu_percent:.2f}",
        "# HELP streampulse_rss_bytes Resident memory of engine and children.",
        "# TYPE streampulse_rss_bytes gauge",
        f"streampulse_rss_bytes {int(s.rss_bytes)}",
        "# HELP streampulse_ram_percent System RAM utilisation percent.",
        "# TYPE streampulse_ram_percent gauge",
        f"streampulse_ram_percent {s.ram_percent:.2f}",
        "# HELP streampulse_uptime_seconds Seconds since start.",
        "# TYPE streampulse_uptime_seconds counter",
        f"streampulse_uptime_seconds {s.uptime_s:.1f}",
        "# HELP streampulse_inference_ms Last batched inference duration.",
        "# TYPE streampulse_inference_ms gauge",
        f"streampulse_inference_ms {s.inference_ms:.3f}",
        "# HELP streampulse_batch_size Frames in last inference batch.",
        "# TYPE streampulse_batch_size gauge",
        f"streampulse_batch_size {s.batch_size}",
        "# HELP streampulse_backpressure_state 0 nominal, 1 throttled, 2 critical.",
        "# TYPE streampulse_backpressure_state gauge",
        f"streampulse_backpressure_state {0 if s.backpressure_state == 'NOMINAL' else (1 if s.backpressure_state == 'THROTTLED' else 2)}",
        "# HELP streampulse_skip_ratio Backpressure frame-skip ratio.",
        "# TYPE streampulse_skip_ratio gauge",
        f"streampulse_skip_ratio {s.skip_ratio}",
        "# HELP streampulse_objects_counted_total Objects crossing the ROI line.",
        "# TYPE streampulse_objects_counted_total counter",
    ]
    for name, v in s.streams.items():
        lbl = f'stream="{name}"'
        lines.append(f"streampulse_objects_counted_total{{{lbl}}} {v.objects_counted}")
    lines += [
        "# HELP streampulse_anomalies_total Anomalies / dwell violations.",
        "# TYPE streampulse_anomalies_total counter",
    ]
    for name, v in s.streams.items():
        lines.append(f"streampulse_anomalies_total{{stream=\"{name}\"}} {v.anomalies}")
    for help_text, mname, field, typ in [
        ("Capture FPS", "capture_fps", "fps", "gauge"),
        ("Inference FPS", "inference_fps", "inference_fps", "gauge"),
        ("End-to-end latency ms", "e2e_latency_ms", "e2e_latency_ms", "gauge"),
        ("p95 latency ms", "e2e_latency_p95_ms", "latency_p95_ms", "gauge"),
        ("Dropped frame percentage", "dropped_pct", "dropped_pct", "gauge"),
        ("Decoded frames", "decoded_frames_total", "decoded_total", "counter"),
        ("Processed frames", "processed_frames_total", "processed_total", "counter"),
    ]:
        lines.append(f"# HELP streampulse_{mname} {help_text}.")
        lines.append(f"# TYPE streampulse_{mname} {typ}")
        for name, v in s.streams.items():
            lines.append(f'streampulse_{mname}{{stream="{name}"}} {getattr(v, field)}')
    return "\n".join(lines) + "\n"
