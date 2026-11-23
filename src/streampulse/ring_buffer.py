"""Fixed-size POSIX shared-memory frame rings.

Writers overwrite the oldest slot, so a slow detector drops frames instead of
growing memory. Pixels stay outside Python objects and cross to the inference
process without pickling.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from multiprocessing import shared_memory
from typing import Optional, Tuple

import numpy as np


class SharedFrameRing:
    """Fixed-size ring of uint8 frames shared between ingest thread(s) and the
    inference process. One writer (the stream's ingest thread), one reader
    (the inference worker)."""

    def __init__(self, name: str, width: int, height: int, capacity: int,
                 create: bool = True):
        self.name = name
        self.width = int(width)
        self.height = int(height)
        self.capacity = max(2, int(capacity))
        self.frame_nbytes = self.width * self.height * 3
        self.shape = (self.height, self.width, 3)

        self.shm = shared_memory.SharedMemory(
            name=name if not create else None,
            create=create,
            size=self.capacity * self.frame_nbytes,
        )
        # Monotonic number of frames written. Slot index = seq % capacity.
        self.head = mp.Value("q", 0)
        self.read_seq = mp.Value("q", 0)   # last seq consumed by inference
        self.lock = mp.Lock()
        self.timestamps = mp.Array("d", [0.0] * self.capacity)

    def write(self, frame: np.ndarray, ts: Optional[float] = None) -> int:
        """Store a decoded frame. Overwrites the oldest slot when full."""
        if frame.shape != self.shape:
            frame = cv2_resize(frame, self.width, self.height)
        ts = time.perf_counter() if ts is None else ts
        with self.lock:
            seq = self.head.value
            slot = seq % self.capacity
            buf = np.ndarray(self.shape, dtype=np.uint8, buffer=self.shm.buf,
                             offset=slot * self.frame_nbytes)
            np.copyto(buf, frame, casting="unsafe")
            self.timestamps[slot] = ts
            self.head.value = seq + 1
        return seq

    def latest(self, copy: bool = True) -> Tuple[Optional[np.ndarray], int, float]:
        """Return the newest complete (frame, seq, capture_ts).

        Copying is the default: it makes the snapshot immune to the writer
        overwriting the slot while we run inference on it. The copy is ~0.9 MB
        for 640x480 and costs < 0.3 ms -- negligible next to model execution.
        """
        head = self.head.value
        if head <= 0:
            return None, -1, 0.0
        seq = head - 1          # sequence number of the newest stored frame
        slot = seq % self.capacity
        with self.lock:
            view = np.ndarray(self.shape, dtype=np.uint8, buffer=self.shm.buf,
                              offset=slot * self.frame_nbytes)
            out = view.copy() if copy else view
            ts = self.timestamps[slot]
        return out, seq, ts

    def dropped_since(self) -> int:
        """Frames produced but never consumed since last sample."""
        last_seq = self.head.value - 1
        if last_seq < 0:
            return 0
        consumed = self.read_seq.value
        if consumed < 0:
            return last_seq + 1
        return max(0, last_seq - consumed)

    def acknowledge(self, seq: int) -> None:
        """Mark frames up to ``seq`` as consumed (monotonic)."""
        with self.lock:
            if seq > self.read_seq.value:
                self.read_seq.value = seq

    def close(self) -> None:
        try:
            self.shm.close()
        except Exception:
            pass

    def unlink(self) -> None:
        try:
            self.shm.unlink()
        except Exception:
            pass


def cv2_resize(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    """Lazy cv2 import kept local so ring_buffer can be used in tests alone."""
    import cv2
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
