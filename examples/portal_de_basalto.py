"""
"Portal de Basalto" — tema de batalha orquestral/rock, ~2:45.

Musica ORIGINAL. A faixa de referencia que o Ramon mandou foi medida com
`chipforge.analysis.reference` e serviu para definir os PARAMETROS da peca —
andamento, tamanho das secoes, arco de dinamica e alvo de balanco espectral.
Nada do material melodico ou harmonico vem de la; tema, progressoes e riffs
sao escritos aqui.

O que a analise da referencia entregou, e o que fiz com cada numero:

    166 BPM                  -> mesmo andamento: e o pulso que da a sensacao
                                de perseguicao sem virar correria
    ~115 compassos de 4/4    -> mesma metragem, sete secoes
    secoes de ~16 compassos  -> cada bloco dura ~23 s, tempo de estabelecer
                                uma ideia e sair antes de cansar
    faixa dinamica ~17 dB    -> o comeco precisa ser MUITO menor que o fim
    grave dominante, pouco   -> percussao grave carrega; nada de brilho
    brilho acima de 7 kHz       exagerado no topo

    python3 examples/portal_de_basalto.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from chipforge.audio import engine
from chipforge.audio import fx
from chipforge.audio import sampler as SF
from chipforge.audio.instrument import factory_by_name
from chipforge.core import generators as G
from chipforge.core import harmony as H
from chipforge.core import intensity as INT
from chipforge.core import theory as T
from chipforge.core.project import Note, Project
from chipforge.midiio import library as L
from chipforge.midiio import smf

BPM = 166
Q = 480
BAR = Q * 4
N = T.name_to_midi
ACOMP = N("A3")
GRAVE = N("A1")


def ac(raiz, qual="min", oit=4):
    r = N(f"{raiz}{oit}")
    return [r + i for i in T.CHORDS[qual]]


def prog(seq, vezes=1):
    return [ac(*c) for c in seq] * vezes


def dobrar(seq):
    """Cada acorde ocupa dois compassos."""
    return [c for c in seq for _ in (0, 1)]


SECOES = [
    ("INVOCACAO", 16, dobrar(prog([("A", "min"), ("F", "maj"),
                                   ("A", "min"), ("E", "5")], 2))),
    ("AVANCO",    16, prog([("A", "min"), ("F", "maj"),
                            ("C", "maj"), ("G", "maj")], 4)),
    ("TEMA",      16, prog([("A", "min"), ("F", "maj"),
                            ("C", "maj"), ("E", "5")], 4)),
    ("RUPTURA",   16, prog([("D", "min"), ("A", "min"),
                            ("A#", "maj"), ("E", "5")], 4)),
    ("PERSEGUICAO", 16, prog([("A", "min"), ("G", "maj"),
                              ("F", "maj"), ("E", "5")], 4)),
    ("COROAMENTO", 20, prog([("C", "min"), ("G#", "maj"),
                             ("D#", "maj"), ("A#", "maj")], 4)
                       + prog([("F", "min"), ("G#", "maj"),
                               ("A#", "maj"), ("E", "5")])),
    ("PO",        15, dobrar(prog([("A", "min"), ("F", "maj"),
                                   ("D", "min"), ("A", "min"),
                                   ("A", "min"), ("F", "maj"),
                                   ("A", "min")]))[:15]),
]
INICIO, _b = {}, 0
for _n, _k, _ in SECOES:
    INICIO[_n] = _b
    _b += _k
TOTAL = _b

# --------------------------------------------------------------------------
# Material tematico — tudo escrito nota a nota
# --------------------------------------------------------------------------
TEMA = [
    ("A4", 0, 0.0, 0.5), ("E5", 0, 0.5, 0.5), ("A5", 0, 1.0, 1.0),
    ("G5", 0, 2.0, 0.75), ("E5", 0, 2.75, 0.25), ("D5", 0, 3.0, 1.0),
    ("C5", 1, 0.0, 1.0), ("F5", 1, 1.0, 1.0), ("A5", 1, 2.0, 1.5),
    ("F5", 1, 3.5, 0.5),
    ("E5", 2, 0.0, 0.5), ("G5", 2, 0.5, 0.5), ("C6", 2, 1.0, 1.0),
    ("B5", 2, 2.0, 1.0), ("G5", 2, 3.0, 1.0),
    ("B5", 3, 0.0, 1.0), ("A5", 3, 1.0, 1.0), ("G#5", 3, 2.0, 1.5),
    ("E5", 3, 3.5, 0.5),
    ("A5", 4, 0.0, 0.5), ("B5", 4, 0.5, 0.5), ("C6", 4, 1.0, 1.0),
    ("B5", 4, 2.0, 0.75), ("A5", 4, 2.75, 0.25), ("E5", 4, 3.0, 1.0),
    ("F5", 5, 0.0, 1.0), ("A5", 5, 1.0, 1.0), ("G5", 5, 2.0, 2.0),
    ("E5", 6, 0.0, 1.0), ("G5", 6, 1.0, 1.0), ("C6", 6, 2.0, 1.0),
    ("D6", 6, 3.0, 1.0),
    ("B5", 7, 0.0, 1.5), ("G#5", 7, 1.5, 0.5), ("A5", 7, 2.0, 2.0),
]

# figura de perseguicao: semicolcheias subindo e descendo pelo acorde
PERSEGUICAO = [
    ("A4", 0, 0.0, 0.25), ("C5", 0, 0.25, 0.25), ("E5", 0, 0.5, 0.25),
    ("A5", 0, 0.75, 0.25), ("G5", 0, 1.0, 0.5), ("E5", 0, 1.5, 0.5),
    ("C5", 0, 2.0, 0.5), ("E5", 0, 2.5, 0.5), ("A5", 0, 3.0, 1.0),
    ("G4", 1, 0.0, 0.25), ("B4", 1, 0.25, 0.25), ("D5", 1, 0.5, 0.25),
    ("G5", 1, 0.75, 0.25), ("F5", 1, 1.0, 0.5), ("D5", 1, 1.5, 0.5),
    ("B4", 1, 2.0, 0.5), ("D5", 1, 2.5, 0.5), ("G5", 1, 3.0, 1.0),
    ("F4", 2, 0.0, 0.25), ("A4", 2, 0.25, 0.25), ("C5", 2, 0.5, 0.25),
    ("F5", 2, 0.75, 0.25), ("E5", 2, 1.0, 0.5), ("C5", 2, 1.5, 0.5),
    ("A4", 2, 2.0, 0.5), ("C5", 2, 2.5, 0.5), ("F5", 2, 3.0, 1.0),
    ("E5", 3, 0.0, 0.25), ("G#5", 3, 0.25, 0.25), ("B5", 3, 0.5, 0.25),
    ("E6", 3, 0.75, 0.25), ("D6", 3, 1.0, 0.5), ("B5", 3, 1.5, 0.5),
    ("G#5", 3, 2.0, 0.5), ("B5", 3, 2.5, 0.5), ("E5", 3, 3.0, 1.0),
]

# ruptura: metade do tempo, muito espaco, so nota longa
RUPTURA = [
    ("D5", 0, 0.0, 2.0), ("F5", 0, 2.0, 2.0),
    ("E5", 1, 0.0, 2.0), ("A4", 1, 2.0, 2.0),
    ("F5", 2, 0.0, 2.0), ("D5", 2, 2.0, 2.0),
    ("B4", 3, 0.0, 2.0), ("E5", 3, 2.0, 2.0),
]

# chamado da invocacao: dois gestos, um por frase de oito compassos
INVOCACAO = [
    ("A4", 1, 2.0, 2.0),
    ("E5", 3, 0.0, 4.0),
    ("F5", 5, 2.0, 2.0),
    ("E5", 7, 0.0, 4.0),
]

PO = [
    ("E5", 0, 0.0, 2.0), ("C5", 0, 2.0, 2.0),
    ("A4", 2, 0.0, 4.0),
    ("F5", 4, 0.0, 2.0), ("E5", 4, 2.0, 2.0),
    ("D5", 6, 0.0, 4.0),
    ("C5", 8, 0.0, 4.0),
    ("A4", 10, 0.0, 8.0),
]


def escrever(spec, transpor=0, oitava=0, vel=104, corte=25, vezes=1, comp=8):
    out = []
    for r in range(vezes):
        base = r * comp * BAR
        for nome, c, b, d in spec:
            out.append(Note(base + c * BAR + int(b * Q),
                            max(50, int(d * Q) - corte),
                            N(nome) + transpor + 12 * oitava, vel))
    return out


def triade(ch):
    return sorted(ch)[:3]


# --------------------------------------------------------------------------
# Camadas
# --------------------------------------------------------------------------
def pad(acordes, low=ACOMP, vel=70, sem_terca=False):
    out = []
    for i, ch in enumerate(acordes):
        if sem_terca:
            raiz = G.voice_in_octave([min(ch)], low)[0]
            notas = [raiz, raiz - 5]
        else:
            notas = G.voice_in_octave(triade(ch), low)
        for n in notas:
            out.append(Note(i * BAR, BAR - 40, n, vel))
    return out


def ostinato(acordes, rate=4, vel=82, low=ACOMP):
    passo = Q // rate
    out = []
    for i, ch in enumerate(acordes):
        v = G.voice_in_octave(triade(ch), low)
        for j in range(4 * rate):
            out.append(Note(i * BAR + j * passo, passo - 10,
                            v[0] if j % 2 == 0 else v[min(2, len(v) - 1)],
                            vel + (14 if j % rate == 0 else 0)))
    return out


GALOPE = (0.0, 0.75, 1.0, 1.5, 2.0, 2.75, 3.0, 3.5)
COLCHEIA = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
SEMI = tuple(i * 0.25 for i in range(16))


def riff(acordes, celula=GALOPE, low=GRAVE, vel=112, quinta=True, dur=0.42):
    out = []
    for i, ch in enumerate(acordes):
        raiz = G.voice_in_octave([min(ch)], low)[0]
        for b in celula:
            out.append(Note(i * BAR + int(b * Q), int(Q * dur), raiz, vel))
            if quinta:
                out.append(Note(i * BAR + int(b * Q), int(Q * dur), raiz + 7,
                                vel - 12))
    return out


def baixo(acordes, celula=COLCHEIA, low=None, vel=114):
    low = low if low is not None else GRAVE
    out = []
    for i, ch in enumerate(acordes):
        raiz = G.voice_in_octave([min(ch)], low)[0]
        for j, b in enumerate(celula):
            nota = raiz + (12 if j == len(celula) - 1 else 0)
            out.append(Note(i * BAR + int(b * Q), int(Q * 0.45), nota, vel))
    return out


def bateria(compassos, estilo="rock", prato_inicial=True):
    """36 bumbo · 38 caixa · 42 chimbal · 46 aberto · 49 prato · 43/45/47 toms"""
    out = []
    for c in range(compassos):
        b = c * BAR
        if estilo == "rock":
            for x in (0.0, 0.75, 2.0, 2.5):
                out.append(Note(b + int(x * Q), 100, 36, 118))
            for x in (1.0, 3.0):
                out.append(Note(b + int(x * Q), 100, 38, 114))
            for j in range(8):
                out.append(Note(b + j * (Q // 2), 90, 42, 72 if j % 2 else 90))
        elif estilo == "duplo":
            for j in range(8):
                out.append(Note(b + j * (Q // 2), 90, 36, 112))
            for x in (1.0, 3.0):
                out.append(Note(b + int(x * Q), 100, 38, 118))
            for j in range(16):
                out.append(Note(b + j * (Q // 4), 70, 42, 64 if j % 2 else 84))
        elif estilo == "meio":
            out.append(Note(b, 110, 36, 116))
            out.append(Note(b + Q * 2, 120, 38, 112))
            for j in range(4):
                out.append(Note(b + j * Q, 110, 46, 66))
        elif estilo == "ritual":
            for x in (0.0, 1.5, 2.0, 3.5):
                out.append(Note(b + int(x * Q), 110, 36, 104))
        if c == 0 and prato_inicial:
            out.append(Note(b, 220, 49, 116))
        if c % 8 == 7:
            for j, p in enumerate((47, 45, 43, 38)):
                out.append(Note(b + int((3.0 + j * 0.25) * Q), 90, p, 114))
    return out


def taiko(acordes, celula=(0.0, 1.5, 2.0, 3.5), nota="A1", vel=112):
    return [Note(i * BAR + int(b * Q), Q // 2, N(nota), vel)
            for i in range(len(acordes)) for b in celula]


def timpano(acordes, vel=112, rufo_no_fim=True):
    out = []
    for i, ch in enumerate(acordes):
        raiz = G.voice_in_octave([min(ch)], N("A1"))[0]
        out.append(Note(i * BAR, Q, raiz, vel))
        if rufo_no_fim and i % 4 == 3:
            for j in range(4):
                out.append(Note(i * BAR + Q * 3 + j * (Q // 4), Q // 4,
                                raiz, 78 + j * 13))
    return out


def sino(acordes, vel=80, dur=Q * 2, cada=2):
    return [Note(i * BAR, dur, G.voice_in_octave([min(ch)], ACOMP)[0], vel)
            for i, ch in enumerate(acordes) if i % cada == 0]


# --------------------------------------------------------------------------
def build(sf2: Path) -> Project:
    p = Project("Portal de Basalto", bpm=BPM, bit_depth=24)
    p.key_root, p.scale = 9, "natural_minor"        # La menor
    banco = SF.open_soundfont(sf2)

    def gm(nome, **kw):
        return SF.instrument(banco, nome, **kw)

    p.master_gain = 0.95
    p.master_reverb = {"sala": 0.78, "damp": 0.45, "largura": 1.0,
                       "predelay_ms": 26.0}
    # A referencia e escura e pesada no grave. Em vez de adivinhar, o alvo veio
    # medido: sub como banda dominante e o topo caindo forte depois de 4 kHz.
    p.master_inserts = [
        {"tipo": "eq", "modo": "lowshelf", "freq": 130, "ganho_db": 2.5},
        {"tipo": "eq", "freq": 480, "ganho_db": -1.5, "q": 0.8},
        {"tipo": "eq", "modo": "highshelf", "freq": 3800, "ganho_db": -4.0},
        {"tipo": "eq", "modo": "lowpass", "freq": 13000, "q": 0.7},
        {"tipo": "comp", "limiar_db": -15.0, "razao": 2.4,
         "attack_ms": 12.0, "release_ms": 200.0, "makeup_db": 2.0},
    ]

    t = {}
    # ---- orquestra
    t["ost"] = p.new_track("Cordas ostinato", gm("Stereo Strings Fast", gain=0.5))
    t["ost"].section = {"vozes": 4, "detune": 9, "spread_ms": 20, "pan": 0.7}
    t["ost"].reverb_send, t["ost"].volume = 0.42, 0.5

    t["pad"] = p.new_track("Cordas pad", gm("Stereo Strings Slow", gain=0.5))
    t["pad"].section = {"vozes": 4, "detune": 12, "spread_ms": 32, "pan": 0.8}
    t["pad"].reverb_send, t["pad"].volume = 0.6, 0.44

    t["cello"] = p.new_track("Cellos", gm("Cello", gain=0.85))
    t["cello"].section = {"vozes": 3, "detune": 8, "spread_ms": 18, "pan": 0.4}
    t["cello"].reverb_send, t["cello"].volume = 0.35, 0.6

    t["trompas"] = p.new_track("Trompas", gm("French Horns", gain=0.75))
    t["trompas"].section = {"vozes": 3, "detune": 8, "spread_ms": 16, "pan": 0.35}
    t["trompas"].reverb_send, t["trompas"].volume = 0.42, 0.58

    t["metais"] = p.new_track("Metais", gm("Brass Section", gain=0.6))
    t["metais"].section = {"vozes": 3, "detune": 7, "spread_ms": 13, "pan": 0.45}
    t["metais"].reverb_send, t["metais"].volume = 0.36, 0.5

    t["coro"] = p.new_track("Coro", gm("Concert Choir", gain=0.6))
    t["coro"].section = {"vozes": 4, "detune": 14, "spread_ms": 36, "pan": 0.85}
    t["coro"].reverb_send, t["coro"].volume = 0.72, 0.46

    t["timpani"] = p.new_track("Timpani", gm("Timpani", gain=0.9))
    t["timpani"].reverb_send, t["timpani"].volume = 0.35, 0.72

    t["taiko"] = p.new_track("Taiko", gm("Taiko Drum", gain=0.95))
    t["taiko"].instrument.is_drum = True
    t["taiko"].reverb_send, t["taiko"].volume = 0.2, 0.78

    t["sinos"] = p.new_track("Sinos", gm("Tubular Bells", gain=0.5))
    t["sinos"].reverb_send, t["sinos"].volume = 0.55, 0.34

    # ---- banda
    for lado, pan in (("esq", -0.85), ("dir", 0.85)):
        tr = p.new_track(f"Guitarra {lado}", gm("Distortion Guitar", gain=0.9))
        tr.inserts = [dict(fx.PRESETS_AMP["rock"], ganho=10.0, grave=-1.0,
                           cab=4200, pan=pan)]
        tr.reverb_send, tr.volume = 0.1, 0.6
        t[f"gtr_{lado}"] = tr

    t["gtr_lead"] = p.new_track("Guitarra lead", gm("Overdrive Guitar", gain=0.9))
    t["gtr_lead"].inserts = [dict(fx.PRESETS_AMP["crunch"], ganho=8.0, pan=0.1),
                             {"tipo": "eq", "freq": 1200, "ganho_db": 2.0, "q": 0.8}]
    t["gtr_lead"].reverb_send, t["gtr_lead"].volume = 0.28, 0.54

    t["baixo"] = p.new_track("Baixo", gm("Finger Bass", gain=0.95))
    t["baixo"].inserts = [{"tipo": "eq", "modo": "lowshelf", "freq": 100,
                           "ganho_db": 5.0},
                          {"tipo": "eq", "freq": 850, "ganho_db": 2.5, "q": 1.1},
                          {"tipo": "comp", "limiar_db": -20.0, "razao": 4.5,
                           "attack_ms": 5.0, "release_ms": 100.0, "makeup_db": 3.5}]
    t["baixo"].volume = 0.74

    t["bateria"] = p.new_track("Bateria", gm("Power", gain=0.9))
    t["bateria"].instrument.is_drum = True
    t["bateria"].inserts = [{"tipo": "comp", "limiar_db": -16.0, "razao": 3.8,
                             "attack_ms": 4.0, "release_ms": 120.0,
                             "makeup_db": 2.5}]
    t["bateria"].reverb_send, t["bateria"].volume = 0.15, 0.66

    t["chip"] = p.new_track("Chip Lead", factory_by_name("NES Lead 25%"))
    t["chip"].instrument.vibrato_depth_cents = 16
    t["chip"].reverb_send, t["chip"].volume = 0.25, 0.28

    acordes = {n: c for n, _, c in SECOES}
    ncomp = {n: k for n, k, _ in SECOES}

    def por(track, nome, secao, notas):
        if not notas:
            return
        pat = p.new_pattern(f"{nome} · {secao}", bars=ncomp[secao])
        pat.notes.extend(notas)
        p.place(pat, t[track], INICIO[secao])

    # ======================================== INVOCACAO (1-16)
    s, ch = "INVOCACAO", acordes["INVOCACAO"]
    por("pad", "Pad", s, pad(ch, vel=54))
    por("coro", "Coro", s, pad(ch, low=N("F3"), vel=58, sem_terca=True))
    por("cello", "Cello", s, baixo(ch, (0.0,), low=N("A1"), vel=88))
    por("taiko", "Taiko", s, taiko(ch, (0.0, 2.0), vel=96))
    por("timpani", "Timpani", s, timpano(ch, 86, rufo_no_fim=False))
    por("sinos", "Sino", s, sino(ch, 74, Q * 3, cada=4))
    por("trompas", "Chamado", s, escrever(INVOCACAO, vel=82, vezes=2))
    por("bateria", "Bateria", s, bateria(16, "ritual", prato_inicial=False))

    # ======================================== AVANCO (17-32)
    s, ch = "AVANCO", acordes["AVANCO"]
    for lado in ("esq", "dir"):
        por(f"gtr_{lado}", "Riff", s, riff(ch, GALOPE))
    por("baixo", "Baixo", s, baixo(ch))
    por("bateria", "Bateria", s, bateria(16, "rock"))
    por("ost", "Ostinato", s, ostinato(ch, rate=4, vel=74))
    por("cello", "Cello", s, baixo(ch, COLCHEIA, low=N("A1"), vel=100))
    por("taiko", "Taiko", s, taiko(ch, vel=104))
    por("timpani", "Timpani", s, timpano(ch, 102))
    por("metais", "Golpes", s, pad(ch, low=N("F3"), vel=84, sem_terca=True))

    # ======================================== TEMA (33-48)
    s, ch = "TEMA", acordes["TEMA"]
    por("trompas", "Tema", s, escrever(TEMA, vel=106, vezes=2))
    por("chip", "Tema", s, escrever(TEMA, vel=80, vezes=2))
    for lado in ("esq", "dir"):
        por(f"gtr_{lado}", "Riff", s, riff(ch, COLCHEIA))
    por("baixo", "Baixo", s, baixo(ch))
    por("bateria", "Bateria", s, bateria(16, "rock"))
    por("ost", "Ostinato", s, ostinato(ch, rate=4, vel=80))
    por("pad", "Pad", s, pad(ch, vel=58))
    por("cello", "Cello", s, baixo(ch, COLCHEIA, low=N("A1"), vel=102))
    por("taiko", "Taiko", s, taiko(ch, vel=106))
    por("timpani", "Timpani", s, timpano(ch, 104))

    # ======================================== RUPTURA (49-64)
    s, ch = "RUPTURA", acordes["RUPTURA"]
    por("coro", "Coro", s, pad(ch, low=N("F3"), vel=76, sem_terca=True))
    por("pad", "Pad", s, pad(ch, vel=56))
    por("trompas", "Ruptura", s, escrever(RUPTURA, vel=94, vezes=4, comp=4))
    por("cello", "Cello", s, baixo(ch, (0.0, 2.0), low=N("A1"), vel=96))
    por("baixo", "Baixo", s, baixo(ch, (0.0, 2.0), vel=98))
    por("bateria", "Bateria", s, bateria(16, "meio"))
    por("taiko", "Taiko", s, taiko(ch, (0.0, 2.0), vel=100))
    por("sinos", "Sino", s, sino(ch, 72, Q * 2, cada=2))

    # ======================================== PERSEGUICAO (65-80)
    s, ch = "PERSEGUICAO", acordes["PERSEGUICAO"]
    por("ost", "Perseguicao", s, escrever(PERSEGUICAO, vel=92, vezes=4, comp=4))
    por("gtr_lead", "Perseguicao", s, escrever(PERSEGUICAO, vel=106,
                                               vezes=4, comp=4))
    por("chip", "Perseguicao", s, escrever(PERSEGUICAO, vel=74, vezes=4, comp=4))
    for lado in ("esq", "dir"):
        por(f"gtr_{lado}", "Riff", s, riff(ch, SEMI, vel=104, dur=0.22))
    por("baixo", "Baixo", s, baixo(ch))
    por("bateria", "Bateria", s, bateria(16, "duplo"))
    por("cello", "Cello", s, baixo(ch, COLCHEIA, low=N("A1"), vel=104))
    # sem pad sustentado aqui: numa corrida de semicolcheias ele so embola
    por("coro", "Coro", s, pad(ch, low=N("F3"), vel=66, sem_terca=True))
    por("taiko", "Taiko", s, taiko(ch, (0.0, 1.0, 2.0, 3.0), vel=108))
    por("timpani", "Timpani", s, timpano(ch, 108))

    # ======================================== COROAMENTO (81-100)
    s, ch = "COROAMENTO", acordes["COROAMENTO"]
    sub = 3                        # sobe uma terca menor: La menor -> Do menor
    por("trompas", "Tema+", s, escrever(TEMA, transpor=sub, vel=110, vezes=2))
    por("metais", "Tema+", s, escrever(TEMA, transpor=sub, vel=104, vezes=2))
    por("chip", "Tema+", s, escrever(TEMA, transpor=sub, vel=88, vezes=2))
    # em unissono com trompas e metais, NAO uma oitava abaixo: la embaixo ela
    # cai em cima do pad e vira briga de segunda e tritono
    por("gtr_lead", "Tema+", s, escrever(TEMA, transpor=sub, vel=102, vezes=2))
    for lado in ("esq", "dir"):
        por(f"gtr_{lado}", "Riff", s, riff(ch, GALOPE))
    por("baixo", "Baixo", s, baixo(ch))
    por("bateria", "Bateria", s, bateria(20, "duplo"))
    por("ost", "Ostinato", s, ostinato(ch, rate=4, vel=86))
    por("pad", "Pad", s, pad(ch, vel=62))
    por("coro", "Coro", s, pad(ch, low=N("F3"), vel=80, sem_terca=True))
    por("cello", "Cello", s, baixo(ch, COLCHEIA, low=N("A1"), vel=106))
    por("taiko", "Taiko", s, taiko(ch, vel=112))
    por("timpani", "Timpani", s, timpano(ch, 112))
    por("sinos", "Sino", s, sino(ch, 80, Q * 2, cada=2))

    # ======================================== PO (101-115)
    s, ch = "PO", acordes["PO"]
    por("pad", "Pad", s, pad(ch, vel=54))
    por("coro", "Coro", s, pad(ch, low=N("F3"), vel=58, sem_terca=True))
    por("trompas", "Po", s, escrever(PO, vel=82))
    por("cello", "Cello", s, baixo(ch, (0.0,), low=N("A1"), vel=84))
    por("timpani", "Timpani", s, timpano(ch, 80, rufo_no_fim=False))
    por("sinos", "Sino", s, sino(ch, 70, Q * 3, cada=4))
    return p


# --------------------------------------------------------------------------
def main():
    out = Path(__file__).parent
    sf2 = L.ensure("generaluser_gs")
    print(f"Banco: {SF.open_soundfont(sf2).name}\n")

    ref = None
    mp3 = glob.glob("/root/.claude/uploads/*/*.mp3")
    if mp3:
        from chipforge.analysis.reference import analisar
        ref = analisar(mp3[0])
        print("=== REFERENCIA MEDIDA " + "=" * 43)
        print(ref.relatorio_pt())
        print("=" * 64 + "\n")

    p = build(sf2)
    dur = p.ticks_to_seconds(p.length_ticks)
    print(f"Portal de Basalto — {TOTAL} compassos, {p.bpm} BPM, "
          f"{int(dur//60)}:{int(dur%60):02d}")
    print(f"{len(p.tracks)} tracks · {len(p.patterns)} patterns · "
          f"{sum(len(x.notes) for x in p.patterns.values())} notas\n")

    secoes = [(n, INICIO[n] * 4.0, (INICIO[n] + k) * 4.0) for n, k, _ in SECOES]
    reps = H.analyze_sections(p.harmony_events(), secoes, beats_per_chord=4.0)
    print(H.sections_table_pt(reps))

    c = INT.calcular(p)
    print("\nIntensidade (o que o Palco recebe):")
    print("  " + c.resumo_pt(86))
    for nome, k, _ in SECOES:
        t0 = INICIO[nome] * 4 * 60 / BPM
        t1 = (INICIO[nome] + k) * 4 * 60 / BPM
        sel = (c.tempos >= t0) & (c.tempos < t1)
        print(f"  {nome:12s} nivel {c.niveis[sel].mean():.1f} "
              f"(pico {c.niveis[sel].max()})")

    print("\nRenderizando...")
    audio = engine.render_project(p)

    if ref is not None:
        print("\nBalanco espectral contra a referencia:")
        for k, v in ref.comparar(audio, p.sample_rate).items():
            marca = "falta" if v < -3 else "sobra" if v > 3 else "ok"
            print(f"  {k:12s} {v:+6.1f} dB   {marca}")

    wav = engine.export_wav(audio, out / "portal_de_basalto", p.sample_rate, 16)
    chip = p.save(out / "portal_de_basalto")
    mid = smf.write_midi(smf.project_to_midi(p), out / "portal_de_basalto")
    js = c.exportar_json(out / "portal_de_basalto_intensidade", p.name)
    pico = float(np.abs(audio).max())
    print(f"\nWAV     : {wav}  ({len(audio)/p.sample_rate:.1f}s, pico {pico:.3f})")
    print(f"CHIP    : {chip}")
    print(f"MIDI    : {mid}")
    print(f"SPRITES : {js}")


if __name__ == "__main__":
    main()
