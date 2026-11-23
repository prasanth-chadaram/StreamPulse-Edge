"""ONNX Runtime detector and the isolated inference worker process.

A separate process isolates allocator churn and crashes. Frames arrive through
shared memory, results leave through shared telemetry; neither is pickled.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from collections import deque
from typing import List, Tuple

import cv2
import numpy as np

from .config import DetectorConfig, PipelineConfig, StreamConfig
from .output_frames import OutputFrames
from .ring_buffer import SharedFrameRing
from .telemetry import Telemetry
from .tracker import IoUTracker
from .types import Detection

log = logging.getLogger("streampulse.detector")


def letterbox(img: np.ndarray, w: int, h: int) -> Tuple[np.ndarray, float, float, float]:
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = int(round(iw * scale)), int(round(ih * scale))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((h, w, 3), 114, dtype=np.uint8)
    px, py = (w - nw) // 2, (h - nh) // 2
    out[py:py + nh, px:px + nw] = small
    return out, scale, px, py


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    if boxes.size == 0:
        return np.empty(0, dtype=int)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        iw = np.maximum(0.0, xx2 - xx1)
        ih = np.maximum(0.0, yy2 - yy1)
        inter = iw * ih
        union = areas[i] + areas[order[1:]] - inter + 1e-6
        iou = inter / union
        order = order[1:][iou <= iou_thr]
    return np.array(keep, dtype=int)


class DetectorBackend:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.session = None
        self.input_name = None
        self.input_hw: Tuple[int, int] = (cfg.input_size, cfg.input_size)
        self.batch_dim = None       # fixed ONNX batch axis (1 for exported YOLO)
        self._synthetic_t = 0.0
        self.using_model = False
        if cfg.backend != "synthetic" and cfg.model_path and os.path.isfile(cfg.model_path):
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = cfg.num_threads
                opts.inter_op_num_threads = 1
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.session = ort.InferenceSession(
                    cfg.model_path, opts, providers=["CPUExecutionProvider"])
                inp = self.session.get_inputs()[0]
                self.input_name = inp.name
                shp = inp.shape
                self.batch_dim = int(shp[0]) if isinstance(shp[0], int) else None
                self.input_hw = (int(shp[-2]) if isinstance(shp[-2], int) else cfg.input_size,
                                 int(shp[-1]) if isinstance(shp[-1], int) else cfg.input_size)
                self.using_model = True
                log.info("Loaded INT8/ONNX model %s input=%s", cfg.model_path, self.input_hw)
            except Exception as exc:
                log.warning("Model load failed (%s); using synthetic detector", exc)
                self.session = None

    def _synthetic(self, frame: np.ndarray, idx: int) -> List[Detection]:
        h, w = frame.shape[:2]
        self._synthetic_t += 0.066
        out: List[Detection] = []
        n = 2 + (idx % 2)
        for k in range(n):
            speed = 42 + 9 * k + 3 * (idx % 3)
            x = (self._synthetic_t * speed + k * 310 + idx * 57) % (w + 120) - 60
            y = int(h * (0.34 + 0.24 * ((idx + k) % 3)))
            out.append(Detection((float(x), float(y), float(x + 46), float(y + 32)),
                                 0.90 - 0.05 * k, 0 if k % 2 == 0 else 71,
                                 "pcb" if k % 2 == 0 else "component"))
        return out

    def _postprocess(self, outputs, scale: float, px: int, py: int,
                     orig_w: int, orig_h: int) -> List[Detection]:
        """Decode YOLO11 raw output: [1, 84, 3549] = 4 box coords + 80 COCO
        class scores for each anchor. Coordinates are in letterboxed 416px
        space and mapped back to original pixels below."""
        raw = np.asarray(outputs[0])[0]
        if raw.ndim == 2 and raw.shape[0] < raw.shape[1]:
            raw = raw.T  # YOLO [4+nc, N] -> [N, 4+nc]
        if raw.ndim != 2 or raw.shape[1] < 5:
            return []
        boxes_xywh = raw[:, :4].astype(np.float32)
        scores_all = raw[:, 4:].astype(np.float32)
        cls = scores_all.argmax(axis=1)
        conf = scores_all.max(axis=1)
        # Top-k before NMS: keeps the full 3549-anchor grid available instead
        # of truncating to the first N rows (which would drop real objects).
        k = min(len(conf), max(self.cfg.max_detections * 8, 256))
        if k < len(conf):
            top = np.argpartition(conf, -k)[-k:]
            boxes_xywh, conf, cls = boxes_xywh[top], conf[top], cls[top]
        mask = conf >= self.cfg.conf_threshold
        if not np.any(mask):
            return []
        boxes_xywh, conf, cls = boxes_xywh[mask], conf[mask], cls[mask]
        # Convert letterbox coords -> original pixels.
        cx = (boxes_xywh[:, 0] - px) / scale
        cy = (boxes_xywh[:, 1] - py) / scale
        bw = boxes_xywh[:, 2] / scale
        bh = boxes_xywh[:, 3] / scale
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, orig_w)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, orig_h)
        keep = nms(boxes, conf, self.cfg.iou_threshold)
        dets = []
        for i in keep:
            cid = int(cls[i])
            name = self.cfg.class_names[cid] if cid < len(self.cfg.class_names) else str(cid)
            dets.append(Detection(tuple(float(v) for v in boxes[i]), float(conf[i]), cid, name))
        return dets

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        if self.session is None:
            return [self._synthetic(f, i) for i, f in enumerate(frames)]
        iw, ih = self.input_hw
        batch = np.empty((len(frames), 3, ih, iw), dtype=np.float32)
        geoms = []
        for i, f in enumerate(frames):
            lb, scale, px, py = letterbox(f, iw, ih)
            rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
            batch[i] = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
            geoms.append((scale, px, py, f.shape[1], f.shape[0]))
        results = []
        if self.batch_dim == 1:
            # Exported YOLO11 ONNX has a fixed batch axis of 1. "Batched"
            # inference then means a tight sequential loop over the latest
            # frame of each stream inside a single worker process -- no extra
            # Session overhead and still one shared model in RAM.
            for i, g in enumerate(geoms):
                outs = self.session.run(None, {self.input_name: batch[i:i + 1]})
                results.append(self._postprocess([o[0:1] for o in outs], *g))
        else:
            outs = self.session.run(None, {self.input_name: batch})
            for i, g in enumerate(geoms):
                results.append(self._postprocess([o[i:i + 1] for o in outs], *g))
        return results


def draw_overlay(frame: np.ndarray, tracks, crossings: int, anomalies: int,
                 name: str, fps: float, latency_ms: float,
                 line=None) -> np.ndarray:
    """Draw the ROI line, boxes and a compact HUD. In-place on a copied frame."""
    out = frame
    h, w = out.shape[:2]
    if line is not None:
        cv2.line(out, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])),
                 (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(out, "ROI", (int(line[0]) + 4, max(12, int(line[1]) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1, cv2.LINE_AA)
    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t.bbox]
        color = (0, 0, 255) if t.anomaly else (0, 220, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"#{t.track_id} {t.class_name} {t.score:.2f}"
        cv2.putText(out, label, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, color, 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (w, 26), (0, 0, 0), -1)
    cv2.putText(out, f"{name} | fps {fps:.0f} | lat {latency_ms:.0f}ms | "
                     f"count {crossings} | anom {anomalies}", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 230, 220), 1, cv2.LINE_AA)
    return out


class InferenceWorker(mp.Process):
    """Runs in its own process. Reads latest frames, batches detection, tracks."""

    def __init__(self, dcfg: DetectorConfig, pcfg: PipelineConfig,
                 stream_cfgs: List[StreamConfig], rings: List[SharedFrameRing],
                 outputs: OutputFrames, telemetry: Telemetry,
                 stop_event: mp.Event, skip_value: mp.Value, interval_value: mp.Value):
        super().__init__(name="streampulse-inference", daemon=True)
        self.dcfg, self.pcfg = dcfg, pcfg
        self.stream_cfgs = stream_cfgs
        self.rings = rings
        self.outputs = outputs
        self.telemetry = telemetry
        self.stop_event = stop_event
        self.skip_value = skip_value
        self.interval_value = interval_value

    def run(self) -> None:  # pragma: no cover - exercised in integration
        logging.basicConfig(level=logging.INFO)
        det = DetectorBackend(self.dcfg)
        trackers = [IoUTracker(self.pcfg, c, self.pcfg.frame_width, self.pcfg.frame_height)
                    for c in self.stream_cfgs]
        lat_windows = [deque(maxlen=self.pcfg.latency_window) for _ in self.stream_cfgs]
        counters = [0] * len(self.stream_cfgs)
        inf_frames = [0] * len(self.stream_cfgs)
        last_inf_tick = time.monotonic()
        log.info("Inference worker online (backend=%s)", "ONNX" if det.using_model else "synthetic")

        try:
            while not self.stop_event.is_set():
                t0 = time.monotonic()
                skip = self.skip_value.value
                batch_idx, frames, metas = [], [], []
                for i, ring in enumerate(self.rings):
                    counters[i] += 1
                    frame, seq, ts = ring.latest(copy=True)
                    if frame is None:
                        continue
                    # Always refresh the display frame so MJPEG stays live even
                    # when detection is being skipped for backpressure.
                    self.outputs.write(i, frame.copy())
                    if ring.read_seq.value >= seq:
                        continue
                    if not (counters[i] % (skip + 1) == 0):
                        continue
                    batch_idx.append(i)
                    frames.append(frame)
                    metas.append((seq, ts))

                if frames:
                    inf_start = time.monotonic()
                    dets_per_frame = det.detect_batch(frames)
                    inf_ms = (time.monotonic() - inf_start) * 1000.0
                    self.telemetry.set_global("inference_ms", inf_ms)
                    self.telemetry.set_global("batch_size", len(frames))
                    now = time.monotonic()
                    for i in batch_idx:
                        inf_frames[i] += 1

                    dt = max(0.001, min(0.5, now - getattr(self, "_last_track_t", now)))
                    self._last_track_t = now
                    for pos, i in enumerate(batch_idx):
                        tracks, crossings, anomalies = trackers[i].update(dets_per_frame[pos], dt)
                        seq, ts = metas[pos]
                        latency_ms = (now - ts) * 1000.0
                        lat_windows[i].append(latency_ms)
                        self.rings[i].acknowledge(seq)
                        self.telemetry.add_stream(i, "processed_total", 1)
                        self.telemetry.add_stream(i, "counted", crossings)
                        self.telemetry.add_stream(i, "anomalies", anomalies)
                        self.telemetry.set_stream(i, "active_tracks", len(tracks))
                        self.telemetry.set_stream(i, "latency_ms", latency_ms)
                        self.telemetry.set_stream(i, "latency_p95", _p95(lat_windows[i]))

                    # Real inference rate: frames processed per wall-clock second.
                    if now - last_inf_tick >= 1.0:
                        for i in range(len(self.stream_cfgs)):
                            self.telemetry.set_stream(
                                i, "inf_fps", inf_frames[i] / (now - last_inf_tick))
                            inf_frames[i] = 0
                        last_inf_tick = now
                        disp = frames[pos].copy()
                        draw_overlay(disp, tracks, crossings, anomalies,
                                     self.stream_cfgs[i].name,
                                     self.telemetry.get_stream(i, "fps"), latency_ms,
                                     trackers[i].line)
                        self.outputs.write(i, disp)

                # Drop/processed accounting for the backpressure report.
                for i, ring in enumerate(self.rings):
                    dropped = ring.dropped_since()
                    self.telemetry.set_stream(i, "dropped_total", dropped)
                    decoded = self.telemetry.get_stream(i, "decoded_total")
                    self.telemetry.set_stream(i, "dropped_pct",
                                              100.0 * dropped / max(1.0, decoded))

                interval = self.interval_value.value
                elapsed = time.monotonic() - t0
                if interval > elapsed:
                    self.stop_event.wait(interval - elapsed)
                else:
                    self.stop_event.wait(0.002)
        except Exception as exc:
            log.exception("Inference worker crashed: %s", exc)
        finally:
            log.info("Inference worker stopped")
            for r in self.rings:
                r.close()
            self.outputs.close()


def _p95(window) -> float:
    if not window:
        return 0.0
    arr = np.fromiter(window, dtype=np.float64, count=len(window))
    return float(np.percentile(arr, 95))
