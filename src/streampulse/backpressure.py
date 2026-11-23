"""CPU/RSS controller that sheds inference frames under load.

Skip ratio grows under pressure and recovers slowly with hysteresis. At the
critical RSS limit it forces a collection and maximum skipping.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass

from .config import BackpressureConfig


@dataclass
class PressureSample:
    cpu_percent: float
    rss_mb: float
    ram_percent: float
    mem_available_mb: float


class BackpressureController:
    def __init__(self, cfg: BackpressureConfig):
        self.cfg = cfg
        self.skip_ratio = 0
        self.state_code = 0          # 0 nominal, 1 throttled, 2 critical
        self.throttle_code = 0       # bitmask: 1 CPU, 2 RSS, 4 RAM
        self._ema_cpu = 20.0
        self._ema_rss = 256.0
        self._last_change = time.monotonic()
        self._last_reason = 0
        self.emergency_gc_count = 0

    def update(self, sample: PressureSample) -> int:
        now = time.monotonic()
        a = 0.3  # EMA smoothing: ignores 1-second CPU spikes from other processes
        self._ema_cpu = a * sample.cpu_percent + (1 - a) * self._ema_cpu
        self._ema_rss = a * sample.rss_mb + (1 - a) * self._ema_rss

        cpu_hot = self._ema_cpu >= self.cfg.cpu_high_pct
        rss_hot = self._ema_rss >= self.cfg.rss_high_mb
        ram_hot = sample.ram_percent >= self.cfg.ram_high_pct
        rss_critical = sample.rss_mb >= self.cfg.rss_critical_mb
        reason = (1 if cpu_hot else 0) | (2 if rss_hot else 0) | (4 if ram_hot else 0)
        self.throttle_code = reason

        if rss_critical:
            # Last line of defence: reclaim cyclic garbage and stop processing
            # almost every frame until memory drops.
            gc.collect()
            self.emergency_gc_count += 1
            self.skip_ratio = self.cfg.max_skip_ratio
            self.state_code = 2
            self._last_change = now
            return self.skip_ratio

        hot = cpu_hot or rss_hot or ram_hot
        cool = (self._ema_cpu <= self.cfg.cpu_low_pct
                and self._ema_rss <= self.cfg.rss_high_mb * 0.78
                and sample.ram_percent <= self.cfg.ram_high_pct - 10)

        if hot and reason != self._last_reason:
            # Re-evaluate immediately when the *cause* changes, otherwise
            # throttle one step per control interval to avoid over-reacting.
            self.skip_ratio = min(self.cfg.max_skip_ratio, self.skip_ratio + 1)
            self._last_change = now
        elif hot and now - self._last_change >= self.cfg.control_interval_s:
            self.skip_ratio = min(self.cfg.max_skip_ratio, self.skip_ratio + 1)
            self._last_change = now
        elif cool and self.skip_ratio > 0 and now - self._last_change >= self.cfg.recovery_interval_s:
            self.skip_ratio -= 1
            self._last_change = now

        self._last_reason = reason
        self.state_code = 2 if self.skip_ratio >= self.cfg.max_skip_ratio else (1 if self.skip_ratio > 0 else 0)
        return self.skip_ratio

    def should_process(self, frame_counter: int) -> bool:
        """True when this frame should be handed to the detector."""
        if self.skip_ratio <= 0:
            return True
        return frame_counter % (self.skip_ratio + 1) == 0
