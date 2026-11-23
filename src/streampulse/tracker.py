"""Greedy IoU tracker with virtual-line counting and dwell alarms.

Chosen over ByteTrack to keep the CPU budget on an i3; deterministic and
dependency-free at 15 FPS.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .config import PipelineConfig, StreamConfig
from .types import Detection, Track


def _iou(a, b) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


class IoUTracker:
    def __init__(self, pcfg: PipelineConfig, scfg: StreamConfig, width: int, height: int):
        self.cfg = pcfg
        self.scfg = scfg
        self.width = width
        self.height = height
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}
        x1, y1, x2, y2 = scfg.roi_line
        self.line = (x1 * width, y1 * height, x2 * width, y2 * height)
        self._line_len2 = max(1.0, (x2 - x1) ** 2 * width * width + (y2 - y1) ** 2 * height * height)

    def _side_and_dist(self, cx: float, cy: float) -> Tuple[int, float]:
        lx1, ly1, lx2, ly2 = self.line
        dx, dy = lx2 - lx1, ly2 - ly1
        # Signed perpendicular distance from the point to the infinite line.
        val = (cx - lx1) * dy - (cy - ly1) * dx
        dist = abs(val) / (self._line_len2 ** 0.5)
        return (1 if val >= 0 else -1), dist

    def update(self, dets: List[Detection], dt_s: float) -> Tuple[List[Track], int, int]:
        """Returns (tracks, crossings, new_anomalies) for this tick."""
        # --- greedy IoU association -----------------------------------------
        unmatched_tracks = set(self.tracks)
        used_dets = set()
        for ti, det in enumerate(dets):
            best_id, best_iou = None, self.cfg.tracker_iou_threshold
            for tid in unmatched_tracks:
                score = _iou(det.bbox, self.tracks[tid].bbox)
                if score > best_iou:
                    best_iou, best_id = score, tid
            if best_id is None:
                continue
            t = self.tracks[best_id]
            t.bbox = det.bbox
            t.score, t.class_id, t.class_name = det.score, det.class_id, det.class_name
            t.age += 1
            t.misses = 0
            unmatched_tracks.discard(best_id)
            used_dets.add(ti)

        for ti, det in enumerate(dets):
            if ti in used_dets:
                continue
            cx = (det.bbox[0] + det.bbox[2]) / 2.0
            cy = (det.bbox[1] + det.bbox[3]) / 2.0
            side, dist = self._side_and_dist(cx, cy)
            t = Track(track_id=self.next_id, bbox=det.bbox, score=det.score,
                      class_id=det.class_id, class_name=det.class_name,
                      age=1, centroid=(cx, cy), side=side)
            self.tracks[self.next_id] = t
            self.next_id += 1

        for tid in unmatched_tracks:
            self.tracks[tid].misses += 1
            if self.tracks[tid].misses > self.cfg.tracker_max_age:
                del self.tracks[tid]

        # --- line crossing / dwell / anomaly rules --------------------------
        crossings = 0
        new_anomalies = 0
        hyst = self.cfg.line_hysteresis_px
        dwell_band = max(40.0, self.height * 0.18)
        anomaly_classes = set(self.scfg.anomaly_classes)
        for t in list(self.tracks.values()):
            cx = (t.bbox[0] + t.bbox[2]) / 2.0
            cy = (t.bbox[1] + t.bbox[3]) / 2.0
            t.centroid = (cx, cy)
            side, dist = self._side_and_dist(cx, cy)
            if dist > hyst and side != t.side:
                if side == 1 and t.side == -1:  # count only one direction
                    crossings += 1
                    t.counted = True
                t.side = side
            if dist <= dwell_band:
                t.dwell_s += dt_s
            else:
                t.dwell_s = max(0.0, t.dwell_s - dt_s * 0.5)

            is_anom = (t.dwell_s > self.scfg.dwell_max_s
                       or (t.class_name in anomaly_classes))
            t.anomaly = is_anom
            if is_anom and not t.anomaly_reported:
                new_anomalies += 1
                t.anomaly_reported = True
            if not is_anom:
                t.anomaly_reported = False
        return list(self.tracks.values()), crossings, new_anomalies
