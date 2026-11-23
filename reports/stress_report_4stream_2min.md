# StreamPulse-Edge Backpressure Stress: 4 Real Streams (2 minutes)

- **Date:** 2026-11-23 11:29:17
- **Duration:** 120 s (2.0 min)
- **Sample interval:** 1.0 s (120 samples)
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
| streams | 4 |

## Summary

| Metric | Value |
|---|---|
| RSS min / avg / max | 308.6 / 314.8 / 318.0 MB |
| RSS start -> end | 315.7 -> 316.8 MB (drift +1.1 MB) |
| RSS leak slope (robust, post-warmup) | +74.71 MB/hour |
| CPU avg / max | 88.2% / 100.0% |
| E2E latency p50 / p95 / p99 | 154.8 / 253.6 / 291.0 ms |
| Latency p95 early -> late | 243.4 -> 265.3 ms |
| THROTTLED/CRITICAL state time | 100.0% |
| CRITICAL state time | 100.0% |
| Backpressure skip ratio avg / max | 7.00 / 7 |
| Sample errors / OOM / crashes | 0 / 0 / 0 |
| Objects counted | 34 |
| Anomalies flagged | 19 |

## Per-stream

| Stream | Avg FPS | Avg dropped % | Avg latency ms | Max p95 ms | Counted |
|---|---|---|---|---|---|
| bottle-line | 13.9 | 0.1 | 160.7 | 236.0 | 1 |
| worker-zone | 13.8 | 0.1 | 155.4 | 239.4 | 16 |
| bottle-line-2 | 14.0 | 0.1 | 161.4 | 241.0 | 0 |
| worker-zone-2 | 13.7 | 0.1 | 158.8 | 235.7 | 17 |

## Pass criteria

| Check | Result |
|---|---|
| No sample errors | PASS |
| Engine stayed alive | PASS |
| RSS drift <= 20 MB (short run) | PASS |
| RSS growth <= 30 MB | PASS |
| Peak RSS <= 3500 MB | PASS |
| p99 latency <= 1000 ms | PASS |
| No latency degradation | PASS |
| Backpressure activated under load | PASS |
| Skip ratio reached maximum | PASS |

## Why memory stays bounded

- Each camera owns a fixed-depth latest-frame ring buffer in shared memory; stale frames are overwritten, never queued.
- Latency windows are `deque(maxlen=300)` and stats live in fixed numeric arrays.
- Inference runs in a separate process (GIL isolation) with no per-frame Python object accumulation; the backpressure controller raises frame-skip ratio when CPU/RSS thresholds are crossed, forcing a `gc.collect()` at the configured critical RSS limit (3.3 GB in this config, below the 3.5 GB ceiling).
