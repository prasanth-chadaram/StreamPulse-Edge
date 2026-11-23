"""Shared annotated-frame output used by the MJPEG endpoints."""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np


class OutputFrames:
    """One fixed slot per stream for the most recent annotated JPEG-source frame."""

    def __init__(self, stream_count: int, width: int, height: int,
                 create: bool = True, name: str | None = None):
        self.n = stream_count
        self.width = width
        self.height = height
        self.shape = (height, width, 3)
        self.frame_nbytes = width * height * 3
        self.shm = shared_memory.SharedMemory(
            name=name, create=create,
            size=max(1, stream_count) * self.frame_nbytes,
        )
        self.seq = mp.Array("q", [0] * max(1, stream_count), lock=True)

    def write(self, idx: int, frame: np.ndarray) -> None:
        if frame.shape != self.shape:
            return
        buf = np.ndarray(self.shape, dtype=np.uint8, buffer=self.shm.buf,
                         offset=idx * self.frame_nbytes)
        np.copyto(buf, frame, casting="unsafe")
        with self.seq.get_lock():
            self.seq[idx] += 1

    def read(self, idx: int) -> np.ndarray | None:
        buf = np.ndarray(self.shape, dtype=np.uint8, buffer=self.shm.buf,
                         offset=idx * self.frame_nbytes)
        with self.seq.get_lock():
            if self.seq[idx] <= 0:
                return None
        return buf.copy()

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
