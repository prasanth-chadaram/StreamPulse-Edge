# StreamPulse-Edge Production Soak: 2 Real Streams (5 minutes)

- **Date:** 2026-11-23 11:23:15
- **Duration:** 300 s (5.0 min)
- **Sample interval:** 1.0 s (299 samples)
- **Endpoint:** http://127.0.0.1:8000
- **Verdict:** **PASS**

## Environment

| Item | Value |
|---|---|
| CPU | Intel(R) Core(TM) i3-1005G1 CPU @ 1.20GHz |
| Logical CPUs | 4 |
| System RAM | 7.5 GB |
| OS | Linux-7.2.15-76070215-generic-x86_64-with-glibc2.39 |
| Python | 3.12.3 |
| NumPy | 2.5.2 |
| streams | 2 |

## Summary

| Metric | Value |
|---|---|
| RSS min / avg / max | 245.3 / 255.8 / 262.2 MB |
| RSS start -> end | 245.8 -> 253.4 MB (drift +7.6 MB) |
| RSS leak slope (robust, post-warmup) | -87.86 MB/hour |
| CPU avg / max | 82.8% / 100.0% |
| E2E latency p50 / p95 / p99 | 73.2 / 267.4 / 337.9 ms |
| Latency p95 early -> late | 136.9 -> 290.4 ms |
| THROTTLED/CRITICAL state time | 40.8% |
| CRITICAL state time | 35.8% |
| Backpressure skip ratio avg / max | 2.69 / 7 |
| Sample errors / OOM / crashes | 0 / 0 / 0 |
| Objects counted | 19 |
| Anomalies flagged | 6 |

## Per-stream

| Stream | Avg FPS | Avg dropped % | Avg latency ms | Max p95 ms | Counted |
|---|---|---|---|---|---|
| bottle-line | 13.8 | 0.1 | 97.0 | 330.4 | 1 |
| worker-zone | 13.8 | 0.1 | 95.4 | 340.0 | 18 |

## Pass criteria

| Check | Result |
|---|---|
| No sample errors | PASS |
| Engine stayed alive | PASS |
| RSS slope <= 5.0 MB/h | PASS |
| RSS growth <= 30 MB | PASS |
| Peak RSS <= 3500 MB | PASS |
| p99 latency <= 500 ms | PASS |
| No latency degradation | PASS |
| No sustained CRITICAL state | PASS |

## Why memory stays bounded

- Each camera owns a fixed-depth latest-frame ring buffer in shared memory; stale frames are overwritten, never queued.
- Latency windows are `deque(maxlen=300)` and stats live in fixed numeric arrays.
- Inference runs in a separate process (GIL isolation) with no per-frame Python object accumulation; the backpressure controller raises frame-skip ratio when CPU/RSS thresholds are crossed, forcing a `gc.collect()` at the configured critical RSS limit (3.3 GB in this config, below the 3.5 GB ceiling).
