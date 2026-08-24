"""Gera um .sf2 minimo mas valido — usado para testar o leitor sem baixar 200 MB."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def _chunk(cid: bytes, data: bytes) -> bytes:
    out = cid + struct.pack("<I", len(data)) + data
    return out + (b"\0" if len(data) & 1 else b"")


def _list(ltype: bytes, body: bytes) -> bytes:
    return _chunk(b"LIST", ltype + body)


def _name(s: str, n: int = 20) -> bytes:
    return s.encode("latin-1")[:n - 1].ljust(n, b"\0")


def build(path: str | Path, sr: int = 22050) -> Path:
    # --- dois samples: senoide (A4) e ruido curto -------------------------
    t = np.arange(sr) / sr
    sine = (np.sin(2 * np.pi * 440 * t) * 0.8 * 32767).astype("<i2")
    noise = (np.random.default_rng(7).uniform(-0.6, 0.6, sr // 4) * 32767).astype("<i2")
    pad = np.zeros(46, dtype="<i2")          # 46 zeros obrigatorios entre samples
    pcm = np.concatenate([sine, pad, noise, pad])
    smpl = _chunk(b"smpl", pcm.tobytes())
    sdta = _list(b"sdta", smpl)

    s1_start, s1_end = 0, len(sine)
    s2_start = len(sine) + 46
    s2_end = s2_start + len(noise)

    shdr = b""
    shdr += _name("Sine A4") + struct.pack(
        "<5I", s1_start, s1_end, s1_start + 100, s1_end - 100, sr) + \
        struct.pack("<BbHH", 69, 0, 0, 1)
    shdr += _name("Noise Hit") + struct.pack(
        "<5I", s2_start, s2_end, s2_start, s2_end, sr) + \
        struct.pack("<BbHH", 60, 0, 0, 1)
    shdr += _name("EOS") + struct.pack("<5I", 0, 0, 0, 0, sr) + \
        struct.pack("<BbHH", 0, 0, 0, 0)

    # --- instrumentos -----------------------------------------------------
    inst = _name("Sine Inst") + struct.pack("<H", 0)
    inst += _name("Noise Inst") + struct.pack("<H", 1)
    inst += _name("EOI") + struct.pack("<H", 2)

    igen = b""
    igen += struct.pack("<HH", 43, (127 << 8) | 0)     # keyRange 0-127
    igen += struct.pack("<HH", 54, 1)                  # loop on
    igen += struct.pack("<HH", 53, 0)                  # sampleID 0
    igen += struct.pack("<HH", 43, (127 << 8) | 0)
    igen += struct.pack("<HH", 53, 1)                  # sampleID 1
    ibag = struct.pack("<HH", 0, 0) + struct.pack("<HH", 3, 0) + \
        struct.pack("<HH", 5, 0)

    pgen = struct.pack("<HH", 41, 0) + struct.pack("<HH", 41, 1) + \
        struct.pack("<HH", 0, 0)
    pbag = struct.pack("<HH", 0, 0) + struct.pack("<HH", 1, 0) + \
        struct.pack("<HH", 2, 0)

    phdr = b""
    phdr += _name("Test Sine") + struct.pack("<HHHIII", 0, 0, 0, 0, 0, 0)
    phdr += _name("Test Drum") + struct.pack("<HHHIII", 0, 128, 1, 0, 0, 0)
    phdr += _name("EOP") + struct.pack("<HHHIII", 0, 0, 2, 0, 0, 0)

    pdta = _list(b"pdta",
                 _chunk(b"phdr", phdr) + _chunk(b"pbag", pbag) +
                 _chunk(b"pmod", b"\0" * 10) + _chunk(b"pgen", pgen) +
                 _chunk(b"inst", inst) + _chunk(b"ibag", ibag) +
                 _chunk(b"imod", b"\0" * 10) + _chunk(b"igen", igen) +
                 _chunk(b"shdr", shdr))

    info = _list(b"INFO",
                 _chunk(b"ifil", struct.pack("<HH", 2, 1)) +
                 _chunk(b"isng", b"EMU8000\0") +
                 _chunk(b"INAM", b"Chipforge Test Bank\0"))

    body = b"sfbk" + info + sdta + pdta
    data = b"RIFF" + struct.pack("<I", len(body)) + body
    p = Path(path)
    p.write_bytes(data)
    return p


if __name__ == "__main__":
    print(build("/tmp/test_bank.sf2"))
