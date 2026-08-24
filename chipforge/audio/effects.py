"""Efeitos. Os tres primeiros sao o que define o 'som de console'."""
from __future__ import annotations

import numpy as np

try:                                    # scipy e opcional: acelera os filtros
    from scipy.signal import lfilter as _lfilter
except Exception:                       # pragma: no cover
    _lfilter = None


def bitcrush(x: np.ndarray, bits: int = 8) -> np.ndarray:
    """Reduz a resolucao de amplitude. 8 = NES, 16 = SNES, 4 = destruicao."""
    if bits >= 24:
        return x
    levels = 2 ** (bits - 1)
    # quantizador mid-riser: da exatamente 2**bits degraus, como o DAC real
    q = np.floor(np.clip(x, -1.0, 0.9999999) * levels) + 0.5
    return (q / levels).astype(np.float32)


def downsample(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Sample-and-hold: joga a taxa la pra baixo e devolve no sr original.
    E o que da o 'granulado' de sampler antigo (SNES rodava a ~32 kHz)."""
    if target_sr >= sr:
        return x
    factor = max(1, int(round(sr / target_sr)))
    n = len(x)
    idx = (np.arange(n) // factor) * factor
    return x[np.minimum(idx, n - 1)].astype(np.float32)


def delay(x: np.ndarray, sr: int, time_s: float = 0.18,
          feedback: float = 0.35, mix: float = 0.3) -> np.ndarray:
    d = max(1, int(time_s * sr))
    n = len(x)
    buf = np.zeros(n + d * 6, dtype=np.float32)
    buf[:n] = x
    if _lfilter is not None:
        # y[i] = x[i] + fb*y[i-d]  ->  IIR com um unico polo atrasado
        a = np.zeros(d + 1); a[0] = 1.0; a[d] = -feedback
        buf = _lfilter([1.0], a, buf).astype(np.float32)
    else:
        for i in range(d, len(buf)):
            buf[i] += buf[i - d] * feedback
    wet = buf[:n]
    return ((1 - mix) * x + mix * wet).astype(np.float32)


def lowpass(x: np.ndarray, sr: int, cutoff: float = 4000.0) -> np.ndarray:
    """Filtro de 1 polo — barato e suficiente para tirar aspereza."""
    if cutoff >= sr / 2:
        return x
    a = float(np.exp(-2.0 * np.pi * cutoff / sr))
    if _lfilter is not None:
        return _lfilter([1 - a], [1.0, -a], x).astype(np.float32)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc = (1 - a) * x[i] + a * acc
        y[i] = acc
    return y.astype(np.float32)


def soft_clip(x: np.ndarray, drive: float = 1.0) -> np.ndarray:
    return np.tanh(x * drive).astype(np.float32)


def stereo_pan(mono: np.ndarray, pan: float = 0.0) -> np.ndarray:
    """pan -1 (esq) .. +1 (dir), lei de potencia constante."""
    ang = (pan + 1.0) * np.pi / 4.0
    return np.stack([mono * np.cos(ang), mono * np.sin(ang)], axis=1).astype(np.float32)
