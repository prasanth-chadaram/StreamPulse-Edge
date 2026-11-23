"""Unit checks for the bounded ring + backpressure logic (no media needed)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from streampulse.backpressure import BackpressureController, PressureSample
from streampulse.config import BackpressureConfig
from streampulse.ring_buffer import SharedFrameRing


def test_ring_bounded_and_latest():
    r = SharedFrameRing("sp_test_ring", 8, 8, capacity=2)
    try:
        assert r.head.value == 0
        r.write(np.zeros((8, 8, 3), np.uint8))
        r.write(np.ones((8, 8, 3), np.uint8) * 10)
        r.write(np.ones((8, 8, 3), np.uint8) * 20)
        assert r.head.value == 3
        frame, seq, _ = r.latest()
        assert seq == 2 and int(frame[0, 0, 0]) == 20
        r.acknowledge(2)
        assert r.dropped_since() == 0
    finally:
        r.close(); r.unlink()


def test_backpressure_ramps_and_recovers():
    cfg = BackpressureConfig()
    cfg.control_interval_s = 0.0   # deterministic: no wall-clock sleeps in unit test
    cfg.recovery_interval_s = 0.0
    c = BackpressureController(cfg)
    # Simulate sustained high CPU.
    for _ in range(10):
        s = c.update(PressureSample(95, 500, 60, 4000))
    assert s > 0 and c.state_code in (1, 2)
    # Recover.
    for _ in range(20):
        s = c.update(PressureSample(40, 400, 50, 5000))
    assert s == 0 and c.state_code == 0


if __name__ == "__main__":
    test_ring_bounded_and_latest()
    test_backpressure_ramps_and_recovers()
    print("ring/backpressure tests OK")
