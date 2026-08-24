"""
chipforge.core.generators
=========================
Geradores de material musical. Nao substituem voce compondo — servem para
sair do papel em branco e para preencher as camadas chatas (arpejo, baixo,
bateria) enquanto voce cuida da melodia.

Tudo devolve listas de Note e respeita a tonalidade do projeto.
"""
from __future__ import annotations

import random
from typing import Sequence

from . import theory as T
from .project import Note, Pattern

ARP_MODES = ("up", "down", "updown", "random", "chip")


def voice_in_octave(chord: Sequence[int], low: int) -> list[int]:
    """Reposiciona o acorde dentro da oitava que comeca em `low`.

    Serve para manter o acompanhamento SEMPRE abaixo da melodia. A causa mais
    comum de choque numa mistura e o arpejo subir sem querer ate o registro
    onde o tema esta cantando — resolver isso na voicing e melhor do que pedir
    desculpa depois no medidor.
    """
    return sorted({low + ((p - low) % 12) for p in chord})


def arpeggiate(chords: Sequence[Sequence[int]], ppq: int, bars_per_chord: int = 1,
               beats_per_bar: int = 4, rate: int = 4, mode: str = "up",
               velocity: int = 88, octaves: int = 1, seed: int | None = None) -> list[Note]:
    """Transforma acordes em arpejo escrito nota a nota.

    rate = subdivisoes por semininma (4 = semicolcheia).
    mode 'chip' usa o vai-e-vem rapido de tracker: o acorde inteiro em um
    piscar, repetido — e o som de NES por excelencia.
    """
    rng = random.Random(seed)
    step = max(1, ppq // rate)
    bar = ppq * beats_per_bar
    out: list[Note] = []
    t = 0
    for ch in chords:
        pool = [p + 12 * o for o in range(octaves) for p in ch]
        if mode == "down":
            pool = pool[::-1]
        elif mode == "updown":
            pool = pool + pool[-2:0:-1]
        total = bar * bars_per_chord
        i = 0
        while i * step < total:
            if mode == "random":
                pitch = rng.choice(pool)
            else:
                pitch = pool[i % len(pool)]
            out.append(Note(t + i * step, max(1, step - 8), pitch,
                            velocity if i % rate else min(127, velocity + 14)))
            i += 1
        t += total
    return out


def bassline(chords: Sequence[Sequence[int]], ppq: int, bars_per_chord: int = 1,
             beats_per_bar: int = 4, style: str = "root_fifth",
             octave_shift: int = -12, velocity: int = 112) -> list[Note]:
    """style: 'root' | 'root_fifth' | 'octave' | 'walking' | 'pulse8'."""
    bar = ppq * beats_per_bar
    out: list[Note] = []
    t = 0
    for ch in chords:
        root = min(ch) + octave_shift
        fifth = root + 7
        total = bar * bars_per_chord
        if style == "root":
            out.append(Note(t, total - 30, root, velocity))
        elif style == "octave":
            for j in range(beats_per_bar * bars_per_chord):
                out.append(Note(t + j * ppq, ppq - 30,
                                root + (12 if j % 2 else 0), velocity))
        elif style == "pulse8":
            for j in range(beats_per_bar * bars_per_chord * 2):
                out.append(Note(t + j * (ppq // 2), ppq // 2 - 20, root, velocity))
        elif style == "walking":
            steps = [root, root + 3, fifth, root + 10]
            for j in range(beats_per_bar * bars_per_chord):
                out.append(Note(t + j * ppq, ppq - 30, steps[j % 4], velocity))
        else:  # root_fifth
            out.append(Note(t, ppq * 2 - 30, root, velocity))
            out.append(Note(t + ppq * 2, ppq - 30, root, velocity - 8))
            out.append(Note(t + ppq * 3, ppq - 30, fifth, velocity - 8))
        t += total
    return out


DRUM_STYLES = {
    "chip_rock":  {"kick": [0, 2], "snare": [1, 3], "hat": "8"},
    "marcha":     {"kick": [0, 1, 2, 3], "snare": [2], "hat": "4"},
    "corrida":    {"kick": [0, 1.5, 2], "snare": [1, 3], "hat": "16"},
    "batalha":    {"kick": [0, 0.75, 2, 2.5], "snare": [1, 3], "hat": "8"},
    "menu_calmo": {"kick": [0], "snare": [2], "hat": "4"},
}

KICK, SNARE, HAT = 36, 38, 42


def drums(bars: int, ppq: int, beats_per_bar: int = 4,
          style: str = "chip_rock") -> list[Note]:
    spec = DRUM_STYLES.get(style, DRUM_STYLES["chip_rock"])
    bar = ppq * beats_per_bar
    out: list[Note] = []
    div = {"4": 1, "8": 2, "16": 4}[spec["hat"]]
    for b in range(bars):
        base = b * bar
        for beat in spec["kick"]:
            out.append(Note(base + int(beat * ppq), 140, KICK, 120))
        for beat in spec["snare"]:
            out.append(Note(base + int(beat * ppq), 140, SNARE, 102))
        for j in range(beats_per_bar * div):
            out.append(Note(base + j * (ppq // div), 90, HAT,
                            70 if j % div == 0 else 52))
    return out


def melody(chords: Sequence[Sequence[int]], key_root: int, scale: str, ppq: int,
           bars_per_chord: int = 1, beats_per_bar: int = 4,
           octave: int = 5, seed: int | None = None,
           rhythm: Sequence[float] = (1, 0.5, 0.5, 1, 1)) -> list[Note]:
    """Melodia guiada: cai em nota do acorde nos tempos fortes, nota da escala
    nos fracos, e evita saltos maiores que uma sexta."""
    rng = random.Random(seed)
    allowed = sorted(T.scale_pitch_classes(key_root, scale))
    bar = ppq * beats_per_bar
    out: list[Note] = []
    t = 0
    base = (octave + 1) * 12
    last = base + allowed[0]
    for ch in chords:
        total = bar * bars_per_chord
        pos = 0
        ri = 0
        chord_pcs = sorted({p % 12 for p in ch})
        while pos < total:
            dur_beats = rhythm[ri % len(rhythm)]
            ri += 1
            dur = int(dur_beats * ppq)
            strong = (pos % ppq == 0) and (pos // ppq) % 2 == 0
            pool_pcs = chord_pcs if strong else allowed
            cands = [base + o * 12 + pc for pc in pool_pcs for o in (-1, 0, 1)]
            cands = [c for c in cands if abs(c - last) <= 9 and c != last]
            pitch = rng.choice(cands) if cands else last
            out.append(Note(t + pos, max(60, dur - 25), pitch,
                            104 if strong else 92))
            last = pitch
            pos += dur
        t += total
    return out


def fill_pattern(pattern: Pattern, notes: list[Note], replace: bool = True) -> Pattern:
    if replace:
        pattern.notes.clear()
    pattern.notes.extend(notes)
    end = max((n.start + n.dur for n in pattern.notes), default=0)
    pattern.length_ticks = max(pattern.length_ticks, end)
    return pattern
