"""
chipforge.audio.sfz
===================
Leitor de instrumentos SFZ (texto + amostras WAV), com o mesmo contrato do
SamplerInstrument: render_note(pitch, dur_s, sr, velocity, ...).

Por que existe: os melhores bancos livres de orquestra e de bateria (VSCO 2 CE,
Salamander Drumkit) não são SoundFont — são uma pasta de WAVs e um arquivo de
texto dizendo qual amostra toca em qual tecla e em qual força. Converter para
.sf2 perderia as camadas de velocity e os round robins, que é justamente o que
faz uma caixa de bateria não soar como metralhadora.

O que é lido:
  <control> default_path · <global>/<group>/<region> com herança de opcodes
  sample, key, lokey, hikey, pitch_keycenter, lovel, hivel, seq_length,
  seq_position, lorand, hirand, volume, tune, transpose, pan, ampeg_attack, ampeg_release,
  loop_mode (one_shot para bateria), trigger (regiões de release são ignoradas)

    from chipforge.audio import sfz
    caixa = sfz.instrument("~/.chipforge/library/salamander_drumkit/.../All.sfz", is_drum=True)
"""
from __future__ import annotations

import random
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import effects as FX
from .envelope import ADSR

_NOTAS = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}


def _nota(valor: str) -> int:
    """'60', 'c4', 'c#4', 'db3' -> MIDI (convenção SFZ: c4 = 60)."""
    v = valor.strip().lower()
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    m = re.fullmatch(r"([a-g])([#b]?)(-?\d+)", v)
    if not m:
        return 60
    n = _NOTAS[m.group(1)] + (1 if m.group(2) == "#" else -1 if m.group(2) == "b" else 0)
    return n + (int(m.group(3)) + 1) * 12


def _num(valor, padrao: float = 0.0) -> float:
    """Número tolerante: bancos reais trazem coisas como '0.66s' ou '-3 '."""
    m = re.match(r"\s*(-?\d+(?:\.\d+)?)", str(valor))
    return float(m.group(1)) if m else padrao


# --------------------------------------------------------------------------
# WAV: PCM 8/16/24/32 bits e float 32, mono ou estéreo (vira mono pela média).
def _ler_wav(caminho: Path) -> tuple[np.ndarray, int]:
    dados = caminho.read_bytes()
    if dados[:4] != b"RIFF" or dados[8:12] != b"WAVE":
        raise ValueError(f"não é WAV: {caminho}")
    pos, fmt, pcm = 12, None, None
    while pos + 8 <= len(dados):
        cid, tam = dados[pos:pos + 4], struct.unpack("<I", dados[pos + 4:pos + 8])[0]
        corpo = dados[pos + 8:pos + 8 + tam]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", corpo[:16])
        elif cid == b"data":
            pcm = corpo
        pos += 8 + tam + (tam & 1)
    if fmt is None or pcm is None:
        raise ValueError(f"WAV sem fmt/data: {caminho}")
    tag, canais, sr, _, _, bits = fmt
    if tag == 0xFFFE:   # WAVE_FORMAT_EXTENSIBLE: o subformato vem depois, assume PCM/float pelo tamanho
        tag = 3 if bits == 32 and len(pcm) % 4 == 0 and _parece_float(pcm) else 1
    if tag == 3:
        x = np.frombuffer(pcm, dtype="<f4").astype(np.float32)
    elif bits == 16:
        x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    elif bits == 24:
        b = np.frombuffer(pcm[:len(pcm) - len(pcm) % 3], dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v & 0x800000, v - 0x1000000, v)
        x = v.astype(np.float32) / 8388608.0
    elif bits == 32:
        x = np.frombuffer(pcm, dtype="<i4").astype(np.float32) / 2147483648.0
    elif bits == 8:
        x = (np.frombuffer(pcm, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"WAV de {bits} bits não suportado: {caminho}")
    if canais > 1:
        x = x[:len(x) - len(x) % canais].reshape(-1, canais).mean(axis=1)
    return x, sr


def _parece_float(pcm: bytes) -> bool:
    amostra = np.frombuffer(pcm[:4096 - 4096 % 4], dtype="<f4")
    return bool(np.all(np.isfinite(amostra)) and np.max(np.abs(amostra)) <= 4.0)


# --------------------------------------------------------------------------
@dataclass
class Regiao:
    sample: Path
    lokey: int = 0
    hikey: int = 127
    root: int = 60
    lovel: int = 1
    hivel: int = 127
    seq_length: int = 1
    seq_position: int = 1
    lorand: float = 0.0
    hirand: float = 1.0
    volume_db: float = 0.0
    tune: float = 0.0
    transpose: int = 0
    pan: float = 0.0
    attack: float = 0.0
    release: float = 0.0
    one_shot: bool = False


def _tokens(texto: str):
    """Gera (cabeçalho, {opcodes}) na ordem do arquivo. Valores de 'sample' podem ter
    espaços: o valor vai até o próximo 'nome=' ou fim de linha."""
    texto = re.sub(r"//[^\n]*", "", texto)
    for m in re.finditer(r"<(\w+)>([^<]*)", texto):
        cab, corpo = m.group(1).lower(), m.group(2)
        ops = {}
        for linha in corpo.splitlines():
            for o in re.finditer(r"(\w+)=(.*?)(?=\s+\w+=|$)", linha.strip()):
                ops[o.group(1).lower()] = o.group(2).strip()
        yield cab, ops


def ler(caminho: str | Path) -> tuple[list[Regiao], str]:
    caminho = Path(caminho).expanduser()
    base = caminho.parent
    default_path = ""
    glob_ops: dict = {}
    grupo_ops: dict = {}
    master_ops: dict = {}
    regioes: list[Regiao] = []
    for cab, ops in _tokens(caminho.read_text(encoding="utf-8", errors="replace")):
        if cab == "control":
            default_path = ops.get("default_path", default_path)
        elif cab == "global":
            glob_ops, master_ops, grupo_ops = ops, {}, {}
        elif cab == "master":
            master_ops, grupo_ops = ops, {}
        elif cab == "group":
            grupo_ops = ops
        elif cab == "region":
            o = {**glob_ops, **master_ops, **grupo_ops, **ops}
            if o.get("trigger", "attack") != "attack" or "sample" not in o:
                continue
            amostra = (base / default_path.replace("\\", "/") / o["sample"].replace("\\", "/")).resolve()
            r = Regiao(sample=amostra)
            if "key" in o:
                r.lokey = r.hikey = r.root = _nota(o["key"])
            r.lokey = _nota(o["lokey"]) if "lokey" in o else r.lokey
            r.hikey = _nota(o["hikey"]) if "hikey" in o else r.hikey
            r.root = _nota(o["pitch_keycenter"]) if "pitch_keycenter" in o else (
                r.root if "key" in o else r.lokey if "lokey" in o and r.lokey == r.hikey else r.root)
            r.lovel = int(_num(o.get("lovel", 1), 1))
            r.hivel = int(_num(o.get("hivel", 127), 127))
            r.seq_length = int(_num(o.get("seq_length", 1), 1))
            r.seq_position = int(_num(o.get("seq_position", 1), 1))
            r.lorand = float(_num(o.get("lorand", 0.0), 0.0))
            r.hirand = float(_num(o.get("hirand", 1.0), 1.0))
            r.volume_db = float(_num(o.get("volume", 0), 0))
            r.tune = float(_num(o.get("tune", 0), 0))
            r.transpose = int(_num(o.get("transpose", 0), 0))
            r.pan = float(_num(o.get("pan", 0), 0)) / 100.0
            r.attack = float(_num(o.get("ampeg_attack", 0), 0))
            r.release = float(_num(o.get("ampeg_release", 0.0), 0.0))
            r.one_shot = o.get("loop_mode", "") == "one_shot"
            regioes.append(r)
    return regioes, caminho.stem


@dataclass
class SfzInstrument:
    """Instrumento SFZ com o mesmo contrato do SamplerInstrument."""
    regioes: list
    name: str = ""
    gain: float = 0.9
    pan: float = 0.0
    is_drum: bool = False
    env: ADSR = field(default_factory=lambda: ADSR(0.002, 0.05, 1.0, 0.25))
    detune_cents: float = 0.0
    lowpass_hz: float = 0.0
    drive: float = 1.0
    cauda_max_s: float = 4.0
    # campos da interface comum (a UI lê sem saber que é SFZ)
    osc: str = "sample"
    bits: int = 0
    sample_rate_crush: int = 0
    _cache: dict = field(default_factory=dict, repr=False)
    _rr: dict = field(default_factory=dict, repr=False)
    _sorte: random.Random = field(default_factory=lambda: random.Random(7), repr=False)

    def _audio(self, r: Regiao) -> tuple[np.ndarray, int]:
        if r.sample not in self._cache:
            self._cache[r.sample] = _ler_wav(r.sample)
        return self._cache[r.sample]

    def _escolhe(self, pitch: int, velocity: int) -> Regiao | None:
        cands = [r for r in self.regioes if r.lokey <= pitch <= r.hikey and r.lovel <= velocity <= r.hivel]
        if not cands:
            cands = [r for r in self.regioes if r.lokey <= pitch <= r.hikey]
        if not cands:
            if not self.regioes or self.is_drum:
                return None
            cands = [min(self.regioes, key=lambda r: abs(r.root - pitch))]
        # Round robin: alterna as gravações da mesma tecla e da mesma força.
        n = max(r.seq_length for r in cands)
        if n > 1:
            chave = (pitch, min(r.lovel for r in cands))
            vez = self._rr.get(chave, 0)
            self._rr[chave] = vez + 1
            daqui = [r for r in cands if r.seq_position == vez % n + 1]
            cands = daqui or cands
        # Sorteio (lorand/hirand): várias gravações da mesma batida, uma escolhida ao acaso.
        if any(r.lorand > 0.0 or r.hirand < 1.0 for r in cands):
            sorte = self._sorte.random()
            daqui = [r for r in cands if r.lorand <= sorte < r.hirand or (sorte >= r.hirand == 1.0)]
            cands = daqui or cands
        return min(cands, key=lambda r: abs(r.root - pitch))

    def render_note(self, pitch: float, dur_s: float, sr: int, velocity: int = 100,
                    prev_pitch=None, release: bool = True) -> np.ndarray:
        r = self._escolhe(int(round(pitch)), int(velocity))
        n = max(1, int(dur_s * sr))
        solta = (r.release if r and r.release > 0 else self.env.release) if release else 0.0
        rel_n = max(1, int(min(solta, self.cauda_max_s) * sr)) if release else 0
        if r is None:
            return np.zeros(n + rel_n, dtype=np.float32)
        data, sr_amostra = self._audio(r)
        if self.is_drum or r.one_shot:
            # Bateria: a amostra toca inteira, não importa a duração da nota.
            total = min(len(data) * sr // max(sr_amostra, 1) + 1, int(self.cauda_max_s * sr))
            n, rel_n = total, 0
        total = n + rel_n
        cents = (pitch - r.root + r.transpose) * 100.0 + r.tune + self.detune_cents
        ratio = (2.0 ** (cents / 1200.0)) * (sr_amostra / sr)
        idx = np.arange(total, dtype=np.float64) * ratio
        idx = np.clip(idx, 0, len(data) - 1.001)
        i0 = idx.astype(np.int64)
        frac = (idx - i0).astype(np.float32)
        i1 = np.minimum(i0 + 1, len(data) - 1)
        sig = data[i0] * (1 - frac) + data[i1] * frac
        sig[idx >= len(data) - 1.01] = 0.0
        env = np.ones(total, dtype=np.float32)
        if r.attack > 0:
            a = min(total, int(r.attack * sr))
            env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
        if rel_n:
            env[n:] *= np.exp(-5.0 * np.linspace(0.0, 1.0, rel_n, dtype=np.float32))
        sig = sig * env * (10.0 ** (r.volume_db / 20.0)) * (velocity / 127.0) * self.gain
        if self.lowpass_hz:
            sig = FX.lowpass(sig, sr, self.lowpass_hz)
        if self.drive != 1.0:
            sig = FX.soft_clip(sig, self.drive)
        return sig.astype(np.float32)


def instrument(caminho: str | Path, **kw) -> SfzInstrument:
    """Atalho: sfz.instrument('.../TubularBells.sfz', gain=0.8)"""
    regioes, nome = ler(caminho)
    if not regioes:
        raise ValueError(f"Nenhuma região tocável em {caminho}")
    kw.setdefault("name", nome)
    return SfzInstrument(regioes, **kw)
