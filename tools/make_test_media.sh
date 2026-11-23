#!/usr/bin/env bash
# Generate synthetic PCB image sequences for the RTSP simulator (no downloads).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"
"$PY" "$HERE/make_test_media.py" --out "${DATA_DIR:-./data/media}" "$@"
