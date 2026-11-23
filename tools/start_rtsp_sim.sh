#!/usr/bin/env bash
# StreamPulse-Edge camera simulator.
# Loops the real Intel industrial videos into 2-4 RTSP cameras at 15 FPS /
# 640x480 via MediaMTX, with a UDP MPEG-TS fallback when MediaMTX is absent.
#
# Usage:
#   tools/start_rtsp_sim.sh                              # 2 real streams
#   NUM_STREAMS=4 tools/start_rtsp_sim.sh                # stress test (repeats 2 videos)
#   NO_MEDIAMTX=1 tools/start_rtsp_sim.sh                # UDP fallback
#
# Env:
#   DATA_DIR        directory with the two Intel videos (default ./data/videos)
#   NUM_STREAMS     2..4
#   FPS / WIDTH / HEIGHT
#   RTSP_HOST / RTSP_PORT
#   MEDIAMTX_BIN    explicit path to mediamtx
#   NO_MEDIAMTX=1   force UDP fallback
set -euo pipefail

VIDEOS_DIR="${DATA_DIR:-./data/videos}"
NUM_STREAMS="${NUM_STREAMS:-2}"
FPS="${FPS:-15}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"
RTSP_HOST="${RTSP_HOST:-127.0.0.1}"
RTSP_PORT="${RTSP_PORT:-8554}"
NO_MEDIAMTX="${NO_MEDIAMTX:-0}"

BOTTLE="$VIDEOS_DIR/bottle-detection.mp4"
WORKER="$VIDEOS_DIR/worker-zone-detection.mp4"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg not found. Install: sudo apt install ffmpeg" >&2
  exit 1
fi
if (( NUM_STREAMS < 1 || NUM_STREAMS > 4 )); then
  echo "ERROR: NUM_STREAMS must be 1..4 (got $NUM_STREAMS)" >&2
  exit 1
fi
for f in "$BOTTLE" "$WORKER"; do
  [[ -f "$f" ]] || { echo "ERROR: missing video $f - run tools/download_assets.py" >&2; exit 1; }
done

# --- locate MediaMTX -------------------------------------------------------
MTX_BIN="${MEDIAMTX_BIN:-}"
if [[ "$NO_MEDIAMTX" != "1" && -z "$MTX_BIN" ]]; then
  for c in mediamtx mediaMTX rtsp-simple-server; do
    if command -v "$c" >/dev/null 2>&1; then MTX_BIN="$(command -v "$c")"; break; fi
  done
  [[ -x ./mediamtx ]] && MTX_BIN="./mediamtx"
fi
USE_MTX=0
[[ -n "$MTX_BIN" && "$NO_MEDIAMTX" != "1" ]] && USE_MTX=1

PIDS=()
CLEANED=0
cleanup() {
  (( CLEANED )) && return
  CLEANED=1
  echo "[sim] stopping ${#PIDS[@]} publisher(s)..."
  for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if (( USE_MTX )); then
  echo "[sim] MediaMTX: $MTX_BIN -> rtsp://$RTSP_HOST:$RTSP_PORT/{bottle,worker}"
  "$MTX_BIN" >/tmp/streampulse-mediamtx.log 2>&1 &
  PIDS+=("$!")
  sleep 1
else
  echo "[sim] MediaMTX not found - UDP MPEG-TS fallback."
  echo "[sim] sources: udp://127.0.0.1:5001?overrun_nonfatal=1&fifo_size=1048576 (etc.)"
fi

for (( i=0; i<NUM_STREAMS; i++ )); do
  cam=$((i+1))
  if (( i % 2 == 0 )); then
    base="bottle"; file="$BOTTLE"
  else
    base="worker"; file="$WORKER"
  fi
  port=$((5000+cam))
  name="$base"
  (( cam > 2 )) && name="${base}${cam}"   # unique RTSP path for stress test cameras
  VF="scale=${WIDTH}:${HEIGHT}:force_original_aspect_ratio=decrease,pad=${WIDTH}:${HEIGHT}:(ow-iw)/2:(oh-ih)/2,fps=${FPS},format=yuv420p"
  if (( USE_MTX )); then
    OUT_ARGS=(-f rtsp -rtsp_transport tcp "rtsp://$RTSP_HOST:$RTSP_PORT/$name")
    echo "[sim] stream $cam ($name) -> rtsp://$RTSP_HOST:$RTSP_PORT/$name"
  else
    OUT_ARGS=(-f mpegts "udp://127.0.0.1:$port?localaddr=127.0.0.1")
    echo "[sim] stream $cam ($name) -> udp://127.0.0.1:$port"
  fi
  ffmpeg -hide_banner -loglevel error -nostdin \
    -stream_loop -1 -re -i "$file" -vf "$VF" \
    -c:v libx264 -preset ultrafast -tune zerolatency \
    -g 15 -pix_fmt yuv420p -an -threads 1 "${OUT_ARGS[@]}" \
    >/tmp/streampulse-ffmpeg-$cam.log 2>&1 &
  PIDS+=("$!")
done

echo "[sim] $NUM_STREAMS real streams running. Press Ctrl-C to stop."
wait
