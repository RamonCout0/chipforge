"""Envelopes de amplitude. ADSR normal + o modo 'degrau' dos chips antigos."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ADSR:
    attack: float = 0.005    # segundos
    decay: float = 0.08
    sustain: float = 0.7     # 0..1
    release: float = 0.12
    # Modo chip: em vez de rampa continua, o volume anda em N degraus (4 bits = 16)
    steps: int = 0

    def render(self, n_samples: int, sr: int, release_tail: int = 0) -> np.ndarray:
        a = max(1, int(self.attack * sr))
        d = max(1, int(self.decay * sr))
        r = max(1, int(self.release * sr)) if release_tail == 0 else release_tail
        total = n_samples + r
        env = np.empty(total, dtype=np.float32)
        # attack
        na = min(a, total)
        env[:na] = np.linspace(0.0, 1.0, na, endpoint=False)
        # decay
        nd = min(d, max(0, total - na))
        if nd:
            env[na:na + nd] = np.linspace(1.0, self.sustain, nd, endpoint=False)
        # sustain
        ns = max(0, n_samples - na - nd)
        if ns:
            env[na + nd:na + nd + ns] = self.sustain
        # release
        start_lvl = env[min(n_samples, total) - 1] if total else 0.0
        tail = total - max(n_samples, na + nd)
        if tail > 0:
            env[total - tail:] = np.linspace(start_lvl, 0.0, tail)
        if self.steps:
            env = np.round(env * self.steps) / self.steps
        return env[:total]


PRESET_ENVELOPES = {
    "pluck":   ADSR(0.001, 0.06, 0.0, 0.03, steps=16),
    "lead":    ADSR(0.004, 0.05, 0.75, 0.10, steps=16),
    "pad":     ADSR(0.15, 0.20, 0.85, 0.45),
    "bass":    ADSR(0.001, 0.10, 0.60, 0.06, steps=16),
    "stab":    ADSR(0.001, 0.12, 0.0, 0.05, steps=16),
    "perc":    ADSR(0.0005, 0.05, 0.0, 0.02),
    "organ":   ADSR(0.01, 0.0, 1.0, 0.05, steps=16),
}
