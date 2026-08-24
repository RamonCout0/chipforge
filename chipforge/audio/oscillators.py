"""
chipforge.audio.oscillators
===========================
Os geradores de forma de onda. Tudo por acumulacao de fase, entao pitch bend,
vibrato e portamento saem de graca (basta variar o array de frequencia).

Sabores disponiveis correspondem aos canais reais dos chips classicos:
  pulse    -> canal quadrado do NES/GameBoy (duty 12.5 / 25 / 50 / 75%)
  triangle -> canal triangular do NES (quantizado em 16 degraus, como no chip)
  saw      -> "dente de serra", som mais agressivo (estilo C64 / SID)
  noise    -> LFSR de 15 bits, o mesmo algoritmo do NES: bateria e explosao
  wave     -> wavetable de 32 amostras em 4 bits, o canal WAV do GameBoy
  sine     -> util para sub-graves e FM simples
"""
from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


def _phase(freq: np.ndarray, sr: int, phase0: float = 0.0) -> np.ndarray:
    """Fase normalizada 0..1 acumulada amostra a amostra."""
    inc = freq / sr
    return (phase0 + np.cumsum(inc)) % 1.0


def _polyblep(t: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Correcao de descontinuidade — mata o aliasing feio no registro agudo."""
    out = np.zeros_like(t)
    m1 = t < dt
    x = np.where(m1, t / np.maximum(dt, 1e-12), 0.0)
    out = np.where(m1, x + x - x * x - 1.0, out)
    m2 = t > 1.0 - dt
    y = np.where(m2, (t - 1.0) / np.maximum(dt, 1e-12), 0.0)
    out = np.where(m2, y * y + y + y + 1.0, out)
    return out


def pulse(freq: np.ndarray, sr: int, duty: float = 0.5,
          antialias: bool = True, phase0: float = 0.0) -> np.ndarray:
    ph = _phase(freq, sr, phase0)
    sig = np.where(ph < duty, 1.0, -1.0)
    if antialias:
        dt = freq / sr
        sig += _polyblep(ph, dt)
        sig -= _polyblep((ph - duty) % 1.0, dt)
    return sig.astype(np.float32)


def triangle(freq: np.ndarray, sr: int, steps: int = 16,
             phase0: float = 0.0) -> np.ndarray:
    """Triangular do NES: 16 degraus, o que da aquele brilho caracteristico."""
    ph = _phase(freq, sr, phase0)
    tri = 2.0 * np.abs(2.0 * ph - 1.0) - 1.0
    if steps:
        tri = np.round(tri * (steps / 2)) / (steps / 2)
    return tri.astype(np.float32)


def saw(freq: np.ndarray, sr: int, antialias: bool = True,
        phase0: float = 0.0) -> np.ndarray:
    ph = _phase(freq, sr, phase0)
    sig = 2.0 * ph - 1.0
    if antialias:
        sig -= _polyblep(ph, freq / sr)
    return sig.astype(np.float32)


def sine(freq: np.ndarray, sr: int, phase0: float = 0.0) -> np.ndarray:
    return np.sin(TWO_PI * _phase(freq, sr, phase0)).astype(np.float32)


_LFSR_PERIOD = 32767            # periodo do registrador de 15 bits
_LFSR_CACHE: dict[tuple, np.ndarray] = {}


def _lfsr_sequence(short_period: bool, seed: int = 1) -> np.ndarray:
    """Gera UM periodo inteiro do LFSR e guarda em cache.

    O registrador e sequencial, entao nao da para vetorizar o passo a passo —
    mas ele e periodico. Gerando o periodo completo uma vez (32767 amostras,
    128 KB) o ruido vira uma simples indexacao, e uma faixa de tres minutos
    com bateria em semicolcheias deixa de custar milhoes de iteracoes em
    Python puro.
    """
    key = (short_period, seed)
    cached = _LFSR_CACHE.get(key)
    if cached is not None:
        return cached
    reg = (seed & 0x7FFF) or 1
    tap = 6 if short_period else 1
    out = np.empty(_LFSR_PERIOD, dtype=np.float32)
    for i in range(_LFSR_PERIOD):
        bit = (reg ^ (reg >> tap)) & 1
        reg = (reg >> 1) | (bit << 14)
        out[i] = 1.0 if (reg & 1) else -1.0
    _LFSR_CACHE[key] = out
    return out


def noise(freq: np.ndarray, sr: int, short_period: bool = False,
          seed: int = 1) -> np.ndarray:
    """LFSR de 15 bits identico ao do NES.

    short_period=True usa o tap 6 em vez do tap 1 e produz o timbre metalico
    de 93 passos (o 'zumbido' de robo dos jogos da epoca).
    """
    seq = _lfsr_sequence(short_period, seed)
    steps = np.floor(np.cumsum(freq / sr)).astype(np.int64)
    return seq[steps % len(seq)].astype(np.float32)


def wavetable(freq: np.ndarray, sr: int, table: np.ndarray,
              phase0: float = 0.0) -> np.ndarray:
    """Canal WAV do GameBoy: 32 amostras, 4 bits (16 niveis), sem interpolacao."""
    ph = _phase(freq, sr, phase0)
    idx = (ph * len(table)).astype(np.int32) % len(table)
    return table[idx].astype(np.float32)


# ---- wavetables prontas (valores -1..1, quantizados em 4 bits) --------------
def _quant4(a: np.ndarray) -> np.ndarray:
    return np.round(np.clip(a, -1, 1) * 7.5) / 7.5


WAVETABLES: dict[str, np.ndarray] = {
    "gb_sine":    _quant4(np.sin(np.linspace(0, TWO_PI, 32, endpoint=False))),
    "gb_ramp":    _quant4(np.linspace(-1, 1, 32, endpoint=False)),
    "gb_organ":   _quant4(np.sin(np.linspace(0, TWO_PI, 32, endpoint=False))
                          + 0.5 * np.sin(np.linspace(0, 2 * TWO_PI, 32, endpoint=False))),
    "gb_bell":    _quant4(np.sin(np.linspace(0, TWO_PI, 32, endpoint=False))
                          + 0.35 * np.sin(np.linspace(0, 5 * TWO_PI, 32, endpoint=False))),
    "gb_hollow":  _quant4(np.concatenate([
        np.sin(np.linspace(0, np.pi, 16)), -0.3 * np.ones(16)])),
    "gb_vocal":   _quant4(np.sin(np.linspace(0, TWO_PI, 32, endpoint=False)) ** 3
                          + 0.4 * np.sin(np.linspace(0, 3 * TWO_PI, 32, endpoint=False))),
}

OSC_LABELS_PT = {
    "pulse12": "Pulse 12.5% (lead fininho)",
    "pulse25": "Pulse 25% (lead classico)",
    "pulse50": "Pulse 50% (quadrada cheia)",
    "pulse75": "Pulse 75% (nasal)",
    "triangle": "Triangular (baixo do NES)",
    "saw": "Serra (agressivo, C64)",
    "sine": "Senoide (sub-grave)",
    "noise": "Ruido (bateria/SFX)",
    "noise_metal": "Ruido metalico (robo)",
    "wave": "Wavetable (canal WAV)",
}
