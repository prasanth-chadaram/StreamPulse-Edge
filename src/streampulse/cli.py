"""Command-line entry point."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time

from . import __version__
from .config import AppConfig
from .pipeline import Pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="streampulse",
                                description="CPU-only multi-stream edge video analytics")
    here = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    p.add_argument("--config", default=os.path.join(here, "config", "default.yaml"),
                   help="YAML configuration file")
    p.add_argument("--no-web", action="store_true", help="disable REST/WebSocket dashboard")
    p.add_argument("--port", type=int, default=None, help="override web port")
    p.add_argument("--duration", type=float, default=0.0,
                   help="auto-stop after N seconds (0 = run forever)")
    p.add_argument("--status-interval", type=float, default=10.0,
                   help="seconds between console status lines")
    p.add_argument("--log-level", default=None)
    p.add_argument("--version", action="version", version=f"StreamPulse-Edge {__version__}")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = AppConfig.load(args.config)
    if args.port:
        cfg.web.port = args.port
    if args.log_level:
        cfg.log_level = args.log_level
    logging.basicConfig(level=getattr(logging, cfg.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s")
    log = logging.getLogger("streampulse")
    pipeline = Pipeline(cfg, start_web=not args.no_web)

    stopping = {"v": False}

    def _stop(signum, frame):
        stopping["v"] = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    pipeline.start()
    start = time.monotonic()
    last_report = 0.0
    try:
        while not stopping["v"]:
            time.sleep(0.25)
            now = time.monotonic() - start
            if now - last_report >= args.status_interval:
                s = pipeline.snapshot()
                log.info("fps=[%s] lat_p95=[%s]ms drop=[%s]%% cpu=%.0f%% rss=%.0fMB "
                         "state=%s skip=%d count=%d anom=%d",
                         ",".join(f"{v.fps:.0f}" for v in s.streams.values()),
                         ",".join(f"{v.latency_p95_ms:.0f}" for v in s.streams.values()),
                         ",".join(f"{v.dropped_pct:.0f}" for v in s.streams.values()),
                         s.cpu_percent, s.rss_mb, s.backpressure_state,
                         s.skip_ratio, s.total_counted, s.total_anomalies)
                last_report = now
            if args.duration and now >= args.duration:
                break
    finally:
        pipeline.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
