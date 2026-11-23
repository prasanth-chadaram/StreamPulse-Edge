# StreamPulse-Edge

I built this to answer one question: how many real video streams can a CPU-only
i3-1005G1 handle without falling over? It reads two to four industrial camera
feeds, runs YOLO11n INT8 over zero-copy POSIX shared-memory ring buffers, tracks
objects across a virtual counting line, and sheds inference work when the CPU
or RSS climbs. The box has two cores, four threads, 8 GB RAM, and no GPU.

## Visual proof

Five seconds of live tracking and counting on the bottle conveyor (8 FPS, 480 px):

![live pipeline demo](assets/live_pipeline_demo.gif)

| Bottle conveyor | Factory worker zone |
|---|---|
| ![bottle conveyor](assets/detection_bottle_conveyor.jpg) | ![worker zone](assets/detection_worker_zone.jpg) |

RSS and end-to-end latency from the five-minute soak:

![soak telemetry](assets/soak_telemetry.png)

## What I measured

Single-frame inference, 416×416 input, two ORT threads:

| Model | Median latency | Detections (60 frames) | Mean confidence |
|---|---|---|---|
| FP32 `yolo11n_416.onnx` | 26.6 ms | 131 | 0.760 |
| INT8 `yolo11n_416_int8.onnx` | 16.8 ms | 102 | 0.837 |
| Speedup | 1.58× | | |

Production soak, two real streams, five minutes:

| Metric | Value |
|---|---|
| RSS min / max | 245.3 MB / 262.2 MB |
| RSS growth over the run | +7.6 MB |
| CPU avg / max | 82.8% / 100% |
| E2E latency p50 / p95 / p99 | 73.2 ms / 267.4 ms / 337.9 ms |
| Objects counted | 19 |
| OOM / crashes | 0 / 0 |

Backpressure stress, four real streams, two minutes:

| Metric | Value |
|---|---|
| CPU avg / max | 88.2% / 100% |
| Peak RSS | 318.0 MB |
| E2E latency p99 | 291.0 ms |
| Backpressure skip ratio | 7 / 7 for the whole run |
| OOM / crashes | 0 / 0 |

Both reports, including per-stream tables and pass criteria, are in `reports/`.

## What Broke Under Real Load (Engineering Notes)

**Full-model INT8 broke the Detect head.** Static quantization of every node
pushed sigmoid class scores above 1.0 and dropped most detections. I kept the
last few nodes of the YOLO11n Detect head in FP32 and quantized the backbone.
Scores went back into `[0, 1]`, detections came back, and the model got 1.58×
faster than FP32. `tools/quantize_yolo_int8.py` verifies this on 60 real frames
and falls back to QDQ or dynamic INT8 if a candidate fails.

**RSS crept during the first soak.** Python allocations were flat under
`tracemalloc`; the growth was native. FFmpeg and OpenCV hold packet buffers that
accumulate when a UDP source runs for hours. I recycle the capture handle every
10 seconds (about 300 frames) and call `malloc_trim` periodically from the
monitor thread. The ring buffer hides the short reopen. Steady-state RSS then
stayed inside a 17 MB band for five minutes.

**Four streams had tighter p99 than two.** With two streams, latency reached
337.9 ms p99. Under four streams the controller pinned the skip ratio at 7 and
p99 fell to 291.0 ms. Aggressive shedding keeps inference latency bounded when
the i3 saturates. The controller raises the skip ratio under CPU/RSS pressure
and lowers it slowly with hysteresis, so it does not oscillate.

## How it is wired

Each camera owns a fixed three-slot ring in POSIX shared memory. The ingest
thread overwrites the oldest slot, so a slow detector drops frames instead of
growing a queue. OpenCV decode releases the GIL in native code; ONNX Runtime
runs in a separate process, pinned to two intra-op threads. Frames cross the
boundary in shared memory and never get pickled. An IoU tracker does the line
counting and dwell checks. FastAPI serves the dashboard, MJPEG, WebSocket stats,
and Prometheus `/metrics`.

## Quickstart

```bash
git clone <your-fork-url> StreamPulse-Edge
cd StreamPulse-Edge
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

export DATA_DIR="$PWD/data/videos"
export MODEL_DIR="$PWD/models"
python tools/download_assets.py
python tools/quantize_yolo_int8.py

# Feed two real cameras (RTSP via MediaMTX, or UDP when it is not installed).
tools/start_rtsp_sim.sh

# Dashboard at http://localhost:8000, then the five-minute soak:
python -m streampulse --config config/default.yaml
python tests/soak_test.py
```

For UDP without MediaMTX, use `NO_MEDIAMTX=1`, the `config/*_udp.yaml` files,
and `NUM_STREAMS=4` with `config/stress_4stream_udp.yaml` for the stress run.
`DATA_DIR` and `MODEL_DIR` are read by the scripts and by the YAML configs, so
point them at any disk with enough space.

Tests: `.venv/bin/python -m pytest -q`
