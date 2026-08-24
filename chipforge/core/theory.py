"""
chipforge.core.theory
=====================
Nucleo de teoria musical: notas, escalas, acordes, intervalos, tonalidade.

Tudo aqui e puro (sem dependencia de audio ou UI) e testavel isoladamente.
Pitch class = 0..11 (C=0). Nota MIDI = 0..127 (C4 = 60).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

# --------------------------------------------------------------------------
# Nomes de notas
# --------------------------------------------------------------------------
SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
PT_NAMES = ["Do", "Do#", "Re", "Re#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]

_NAME_TO_PC = {}
for _i, _n in enumerate(SHARP_NAMES):
    _NAME_TO_PC[_n.upper()] = _i
for _i, _n in enumerate(FLAT_NAMES):
    _NAME_TO_PC[_n.upper()] = _i

A4_MIDI = 69
A4_HZ = 440.0


def midi_to_hz(note: float, a4: float = A4_HZ) -> float:
    """Converte nota MIDI (aceita fracionaria, p/ pitch bend) em Hz."""
    return a4 * (2.0 ** ((note - A4_MIDI) / 12.0))


def hz_to_midi(hz: float, a4: float = A4_HZ) -> float:
    import math
    return 12.0 * math.log2(hz / a4) + A4_MIDI


def midi_to_name(note: int, flats: bool = False, octave: bool = True) -> str:
    names = FLAT_NAMES if flats else SHARP_NAMES
    n = names[note % 12]
    return f"{n}{note // 12 - 1}" if octave else n


def name_to_midi(name: str) -> int:
    """'C4' -> 60, 'F#3' -> 54, 'Bb5' -> 82."""
    s = name.strip()
    i = 1
    if len(s) > 1 and s[1] in "#b":
        i = 2
    pc_part, oct_part = s[:i], s[i:]
    pc = _NAME_TO_PC[pc_part.upper()]
    octv = int(oct_part) if oct_part else 4
    return pc + (octv + 1) * 12


# --------------------------------------------------------------------------
# Escalas
# --------------------------------------------------------------------------
SCALES: dict[str, tuple[int, ...]] = {
    "major":            (0, 2, 4, 5, 7, 9, 11),
    "natural_minor":    (0, 2, 3, 5, 7, 8, 10),
    "harmonic_minor":   (0, 2, 3, 5, 7, 8, 11),
    "melodic_minor":    (0, 2, 3, 5, 7, 9, 11),
    "dorian":           (0, 2, 3, 5, 7, 9, 10),
    "phrygian":         (0, 1, 3, 5, 7, 8, 10),
    "lydian":           (0, 2, 4, 6, 7, 9, 11),
    "mixolydian":       (0, 2, 4, 5, 7, 9, 10),
    "locrian":          (0, 1, 3, 5, 6, 8, 10),
    "major_pentatonic": (0, 2, 4, 7, 9),
    "minor_pentatonic": (0, 3, 5, 7, 10),
    "blues":            (0, 3, 5, 6, 7, 10),
    "whole_tone":       (0, 2, 4, 6, 8, 10),
    "octatonic":        (0, 2, 3, 5, 6, 8, 9, 11),
    "chromatic":        tuple(range(12)),
    # sabores uteis para trilha de jogo
    "phrygian_dominant": (0, 1, 4, 5, 7, 8, 10),   # boss / deserto / vilao
    "hirajoshi":         (0, 2, 3, 7, 8),          # japones, menu zen
    "in_sen":            (0, 1, 5, 7, 10),         # misterio
    "prometheus":        (0, 2, 4, 6, 9, 10),      # sci-fi / espaco
}

SCALE_LABELS_PT = {
    "major": "Maior (jonio)",
    "natural_minor": "Menor natural (eolio)",
    "harmonic_minor": "Menor harmonica",
    "melodic_minor": "Menor melodica",
    "dorian": "Dorico",
    "phrygian": "Frigio",
    "lydian": "Lidio",
    "mixolydian": "Mixolidio",
    "locrian": "Locrio",
    "major_pentatonic": "Pentatonica maior",
    "minor_pentatonic": "Pentatonica menor",
    "blues": "Blues",
    "whole_tone": "Tons inteiros",
    "octatonic": "Octatonica (dim)",
    "chromatic": "Cromatica",
    "phrygian_dominant": "Frigio dominante",
    "hirajoshi": "Hirajoshi",
    "in_sen": "In Sen",
    "prometheus": "Prometheus",
}


def scale_pitch_classes(root: int, scale: str) -> set[int]:
    return {(root + s) % 12 for s in SCALES[scale]}


def scale_notes(root_midi: int, scale: str, octaves: int = 1) -> list[int]:
    steps = SCALES[scale]
    out = []
    for o in range(octaves):
        for s in steps:
            out.append(root_midi + s + 12 * o)
    out.append(root_midi + 12 * octaves)
    return out


def quantize_to_scale(note: int, root: int, scale: str) -> int:
    """Puxa a nota para o grau mais proximo da escala (usado no 'scale lock')."""
    allowed = scale_pitch_classes(root, scale)
    for delta in range(0, 7):
        for d in ((-delta, delta) if delta else (0,)):
            if (note + d) % 12 in allowed:
                return note + d
    return note


# --------------------------------------------------------------------------
# Acordes
# --------------------------------------------------------------------------
CHORDS: dict[str, tuple[int, ...]] = {
    "5":      (0, 7),              # power chord — pao com manteiga do chiptune
    "maj":    (0, 4, 7),
    "min":    (0, 3, 7),
    "dim":    (0, 3, 6),
    "aug":    (0, 4, 8),
    "sus2":   (0, 2, 7),
    "sus4":   (0, 5, 7),
    "6":      (0, 4, 7, 9),
    "min6":   (0, 3, 7, 9),
    "7":      (0, 4, 7, 10),
    "maj7":   (0, 4, 7, 11),
    "min7":   (0, 3, 7, 10),
    "minMaj7": (0, 3, 7, 11),
    "dim7":   (0, 3, 6, 9),
    "m7b5":   (0, 3, 6, 10),
    "add9":   (0, 2, 4, 7),
    "9":      (0, 2, 4, 7, 10),
    "maj9":   (0, 2, 4, 7, 11),
    "min9":   (0, 2, 3, 7, 10),
    "7sus4":  (0, 5, 7, 10),
}

# quanto "peso" cada grau do acorde tem ao identificar (a terca importa mais)
_CHORD_PRIORITY = ["maj", "min", "5", "7", "min7", "maj7", "dim", "aug", "sus4",
                   "sus2", "m7b5", "dim7", "6", "min6", "add9", "9", "maj9",
                   "min9", "minMaj7", "7sus4"]


@dataclass(frozen=True)
class ChordMatch:
    root: int          # pitch class
    quality: str
    score: float       # 0..1
    bass: int | None = None

    @property
    def name(self) -> str:
        base = f"{SHARP_NAMES[self.root]}{'' if self.quality == 'maj' else self.quality}"
        if self.bass is not None and self.bass % 12 != self.root:
            base += f"/{SHARP_NAMES[self.bass % 12]}"
        return base


def detect_chord(notes: Iterable[int]) -> ChordMatch | None:
    """Identifica o acorde a partir de notas MIDI soando juntas."""
    notes = sorted(set(notes))
    if not notes:
        return None
    pcs = {n % 12 for n in notes}
    bass = notes[0]
    best: ChordMatch | None = None
    for root in range(12):
        rel = {(p - root) % 12 for p in pcs}
        for qi, quality in enumerate(_CHORD_PRIORITY):
            tpl = set(CHORDS[quality])
            hit = len(rel & tpl)
            extra = len(rel - tpl)
            missing = len(tpl - rel)
            if hit == 0:
                continue
            score = hit / (hit + extra * 1.5 + missing * 1.2)
            # bonus se a fundamental esta no baixo, e desempate por prioridade
            if bass % 12 == root:
                score += 0.08
            score -= qi * 0.001
            if best is None or score > best.score:
                best = ChordMatch(root, quality, min(score, 1.0), bass)
    return best


# --------------------------------------------------------------------------
# Tonalidade (Krumhansl-Schmuckler)
# --------------------------------------------------------------------------
_KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def _corr(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = sum((a[i] - ma) ** 2 for i in range(n)) ** 0.5
    db = sum((b[i] - mb) ** 2 for i in range(n)) ** 0.5
    return num / (da * db) if da and db else 0.0


@dataclass(frozen=True)
class KeyGuess:
    root: int
    mode: str          # "major" | "natural_minor"
    confidence: float  # 0..1

    @property
    def name(self) -> str:
        suffix = "maior" if self.mode == "major" else "menor"
        return f"{SHARP_NAMES[self.root]} {suffix}"


def detect_key(weighted_pcs: Sequence[float],
               tonic_bonus: dict[int, float] | None = None) -> KeyGuess:
    """weighted_pcs: 12 pesos (normalmente duracao total de cada pitch class).

    O histograma sozinho nao distingue relativa maior de relativa menor: Dm e
    Fa maior usam exatamente as mesmas sete notas, e o perfil de Krumhansl
    costuma votar na maior. Quem desempata e a HARMONIA — qual acorde comeca e
    termina o trecho. Passe esse peso em `tonic_bonus` (pitch class -> bonus)
    e a deteccao para de trocar Re menor por Fa maior.
    """
    tonic_bonus = tonic_bonus or {}
    best = KeyGuess(0, "major", 0.0)
    best_score = -9.9
    for root in range(12):
        rot = [weighted_pcs[(root + i) % 12] for i in range(12)]
        bonus = tonic_bonus.get(root, 0.0)
        for corr, mode in ((_corr(rot, _KS_MAJOR), "major"),
                           (_corr(rot, _KS_MINOR), "natural_minor")):
            score = corr + bonus
            if score > best_score:
                best_score = score
                best = KeyGuess(root, mode, corr)
    return KeyGuess(best.root, best.mode, max(0.0, min(1.0, best.confidence)))


# --------------------------------------------------------------------------
# Graus / numerais romanos
# --------------------------------------------------------------------------
_ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII"]


def roman_numeral(chord: ChordMatch, key_root: int, mode: str) -> str:
    steps = SCALES[mode]
    rel = (chord.root - key_root) % 12
    if rel in steps:
        deg = steps.index(rel)
        r = _ROMAN[deg]
    else:
        below = max((s for s in steps if s < rel), default=steps[0])
        deg = steps.index(below)
        r = "#" + _ROMAN[deg]
    minorish = chord.quality.startswith("min") or chord.quality.startswith("dim") \
        or chord.quality == "m7b5"
    if minorish:
        r = r.lower()
    if chord.quality in ("dim", "dim7"):
        r += "o"
    elif chord.quality == "m7b5":
        r += "0"
    elif chord.quality == "aug":
        r += "+"
    elif chord.quality == "7":
        r += "7"
    elif chord.quality == "maj7":
        r += "maj7"
    elif chord.quality in ("min7", "min9"):
        r += "7"
    return r


# --------------------------------------------------------------------------
# Progressoes prontas (gerador de ideias)
# --------------------------------------------------------------------------
PROGRESSIONS: dict[str, list[int]] = {
    "Pop eterno (I-V-vi-IV)":        [0, 4, 5, 3],
    "Aventura (I-IV-V-I)":           [0, 3, 4, 0],
    "Melancolia (vi-IV-I-V)":        [5, 3, 0, 4],
    "Andamento 50s (I-vi-IV-V)":     [0, 5, 3, 4],
    "Boss fight (i-VI-III-VII)":     [0, 5, 2, 6],
    "Dungeon (i-iv-v-i)":            [0, 3, 4, 0],
    "Misterio (i-VII-VI-VII)":       [0, 6, 5, 6],
    "Vitoria (IV-V-I)":              [3, 4, 0],
    "Jazz de menu (ii-V-I)":         [1, 4, 0],
    "Loop de vilarejo (I-vi-ii-V)":  [0, 5, 1, 4],
}


def diatonic_triad(key_root: int, mode: str, degree: int) -> list[int]:
    """Triade diatonica no grau (0-indexado), como pitch classes."""
    steps = SCALES[mode]
    n = len(steps)
    return [(key_root + steps[(degree + i) % n] + 12 * ((degree + i) // n)) % 12
            for i in (0, 2, 4)]


def build_progression(key_root: int, mode: str, degrees: Sequence[int],
                      octave: int = 4) -> list[list[int]]:
    """Acordes em notas MIDI com conducao de voz de verdade.

    Para cada acorde, testa todas as inversoes/oitavas e escolhe a que exige
    o menor movimento total das vozes em relacao ao acorde anterior. E isso que
    evita quintas paralelas e faz a progressao soar 'colada'.
    """
    base = (octave + 1) * 12
    out: list[list[int]] = []
    prev: list[int] | None = None
    for d in degrees:
        pcs = diatonic_triad(key_root, mode, d)
        candidates: list[list[int]] = []
        for shift in (-12, 0, 12):
            for inv in range(len(pcs)):
                rot = pcs[inv:] + pcs[:inv]
                voiced = [base + shift + rot[0]]
                for pc in rot[1:]:
                    nxt = base + shift + pc
                    while nxt <= voiced[-1]:
                        nxt += 12
                    voiced.append(nxt)
                if all(36 <= v <= 84 for v in voiced):
                    candidates.append(voiced)
        if prev is None:
            out.append(sorted(candidates[0]))
        else:
            def cost(c: list[int]) -> float:
                pairs = sorted(zip(sorted(prev), sorted(c)))
                move = sum(abs(b - a) for a, b in pairs)
                # penaliza acordes que abrem demais ou fogem do centro
                spread = abs((max(c) - min(c)) - 7) * 0.3
                centre = abs(sum(c) / len(c) - base - 4) * 0.2
                return move + spread + centre
            out.append(sorted(min(candidates, key=cost)))
        prev = out[-1]
    return out
