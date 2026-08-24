"""
"Vigilia do Colosso" — orquestra + banda de rock na mesma faixa, ~2:50.

O que esta demo existe para provar: o que faltava nao era banco de samples, e
o ar em volta. Mesmo banco de antes, agora com

  * naipe de cordas por DOBRA (4 vozes desafinadas e espalhadas no palco)
  * guitarras com simulador de amplificador e gravacao dupla (uma em cada lado)
  * sala de concerto por envio de reverb, uma so para todos os instrumentos
  * compressor no barramento de bateria e no master

Secoes e o nivel de intensidade que cada uma manda para o Palco:

    ABERTURA  c. 1-8    so orquestra, sem bateria          nivel 1-2
    RIFF      c. 9-24   entra a banda: guitarra e bateria  nivel 4
    TEMA      c.25-40   tema orquestral sobre o rock       nivel 5
    PONTE     c.41-52   quebra: coro e guitarra limpa      nivel 2-3
    SOLO      c.53-68   lead por cima de tudo              nivel 5
    FINAL     c.69-84   sobe um tom (Fa# menor), tudo      nivel 5
    ECO       c.85-92   orquestra sozinha, resolve         nivel 1-2

    python3 examples/vigilia_do_colosso.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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

BPM = 132
Q = 480
BAR = Q * 4
N = T.name_to_midi
ACOMP = N("A3")        # oitava do acompanhamento — sempre abaixo do tema
GRAVE = N("E2")


def ac(raiz, qual="min", oit=4):
    r = N(f"{raiz}{oit}")
    return [r + i for i in T.CHORDS[qual]]


def prog(seq, vezes=1):
    return [ac(*c) for c in seq] * vezes


SECOES = [
    ("ABERTURA",  8, prog([("E", "min"), ("E", "min"), ("C", "maj"), ("C", "maj"),
                           ("G", "maj"), ("G", "maj"), ("D", "maj"), ("D", "maj")])),
    ("RIFF",     16, prog([("E", "5"), ("E", "5"), ("C", "5"), ("D", "5")], 4)),
    ("TEMA",     16, prog([("E", "min"), ("C", "maj"), ("G", "maj"), ("D", "maj")], 4)),
    ("PONTE",    12, prog([("A", "min"), ("E", "min"), ("C", "maj"), ("D", "maj")], 3)),
    ("SOLO",     16, prog([("E", "min"), ("C", "maj"), ("G", "maj"), ("D", "maj")], 4)),
    ("FINAL",    16, prog([("F#", "min"), ("D", "maj"), ("A", "maj"), ("E", "maj")], 4)),
    ("ECO",       8, prog([("E", "min"), ("E", "min"), ("C", "maj"), ("C", "maj"),
                           ("A", "min"), ("A", "min"), ("E", "min"), ("E", "min")])),
]
INICIO, _b = {}, 0
for _n, _k, _ in SECOES:
    INICIO[_n] = _b
    _b += _k
TOTAL = _b

# --------------------------------------------------------------------------
# Temas
# --------------------------------------------------------------------------
TEMA = [
    ("E5", 0, 0.0, 1.0), ("G5", 0, 1.0, 0.5), ("F#5", 0, 1.5, 0.5),
    ("E5", 0, 2.0, 1.0), ("B4", 0, 3.0, 1.0),
    ("C5", 1, 0.0, 1.5), ("E5", 1, 1.5, 0.5), ("G5", 1, 2.0, 2.0),
    ("D5", 2, 0.0, 1.0), ("B4", 2, 1.0, 1.0), ("G5", 2, 2.0, 2.0),
    ("F#5", 3, 0.0, 1.0), ("A5", 3, 1.0, 1.0), ("D5", 3, 2.0, 2.0),
    ("B4", 4, 0.0, 0.5), ("E5", 4, 0.5, 0.5), ("G5", 4, 1.0, 1.0),
    ("B5", 4, 2.0, 1.5), ("A5", 4, 3.5, 0.5),
    ("G5", 5, 0.0, 1.0), ("E5", 5, 1.0, 1.0), ("C5", 5, 2.0, 2.0),
    ("D5", 6, 0.0, 1.0), ("G5", 6, 1.0, 1.0), ("B5", 6, 2.0, 1.0), ("A5", 6, 3.0, 1.0),
    ("F#5", 7, 0.0, 1.5), ("E5", 7, 1.5, 0.5), ("D5", 7, 2.0, 2.0),
]

SOLO = [
    ("B4", 0, 0.0, 0.25), ("D5", 0, 0.25, 0.25), ("E5", 0, 0.5, 0.25),
    ("G5", 0, 0.75, 0.25), ("B5", 0, 1.0, 1.0), ("A5", 0, 2.0, 0.5),
    ("G5", 0, 2.5, 0.5), ("E5", 0, 3.0, 1.0),
    ("G5", 1, 0.0, 0.5), ("E5", 1, 0.5, 0.5), ("C5", 1, 1.0, 1.0),
    ("E5", 1, 2.0, 0.25), ("G5", 1, 2.25, 0.25), ("C6", 1, 2.5, 1.5),
    ("B5", 2, 0.0, 1.0), ("G5", 2, 1.0, 0.5), ("D5", 2, 1.5, 0.5),
    ("G5", 2, 2.0, 0.25), ("B5", 2, 2.25, 0.25), ("D6", 2, 2.5, 1.5),
    ("C6", 3, 0.0, 0.5), ("B5", 3, 0.5, 0.5), ("A5", 3, 1.0, 1.0),
    ("F#5", 3, 2.0, 0.5), ("A5", 3, 2.5, 0.5), ("D5", 3, 3.0, 1.0),
    ("E5", 4, 0.0, 0.25), ("F#5", 4, 0.25, 0.25), ("G5", 4, 0.5, 0.25),
    ("A5", 4, 0.75, 0.25), ("B5", 4, 1.0, 1.5), ("A5", 4, 2.5, 0.5),
    ("G5", 4, 3.0, 1.0),
    ("E6", 5, 0.0, 2.0), ("C6", 5, 2.0, 2.0),
    ("B5", 6, 0.0, 1.0), ("D6", 6, 1.0, 1.0), ("B5", 6, 2.0, 0.5),
    ("A5", 6, 2.5, 0.5), ("G5", 6, 3.0, 1.0),
    ("F#5", 7, 0.0, 1.0), ("A5", 7, 1.0, 1.0), ("E5", 7, 2.0, 2.0),
]

PONTE_MEL = [
    ("A4", 0, 0.0, 2.0), ("C5", 0, 2.0, 2.0),
    ("B4", 1, 0.0, 2.0), ("E5", 1, 2.0, 2.0),
    ("G4", 2, 0.0, 2.0), ("E5", 2, 2.0, 2.0),
    ("F#5", 3, 0.0, 4.0),
]

ECO_MEL = [
    ("B4", 0, 0.0, 2.0), ("E5", 0, 2.0, 2.0),
    ("G5", 1, 0.0, 4.0),
    ("E5", 2, 0.0, 2.0), ("C5", 2, 2.0, 2.0),
    ("B4", 3, 0.0, 4.0),
    ("A4", 4, 0.0, 4.0),
    ("E5", 6, 0.0, 8.0),
]


def escrever(spec, transpor=0, oitava=0, vel=104, corte=25, vezes=1, comp=8):
    out = []
    for r in range(vezes):
        base = r * comp * BAR
        for nome, c, b, d in spec:
            out.append(Note(base + c * BAR + int(b * Q),
                            max(60, int(d * Q) - corte),
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


def ostinato(acordes, rate=2, vel=82, low=ACOMP):
    passo = Q // rate
    out = []
    for i, ch in enumerate(acordes):
        v = G.voice_in_octave(triade(ch), low)
        for j in range(4 * rate):
            out.append(Note(i * BAR + j * passo, passo - 12,
                            v[0] if j % 2 == 0 else v[min(2, len(v) - 1)],
                            vel + (12 if j % rate == 0 else 0)))
    return out


# gallop: a celula ritmica que faz um riff soar rock e nao marcha
GALOPE = (0.0, 0.75, 1.0, 1.5, 2.0, 2.75, 3.0, 3.5)
CHUGA = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)


def riff(acordes, celula=GALOPE, oitava_grave=GRAVE, vel=112, quinta=True):
    """Guitarra base: fundamental e quinta, bem grave, no ritmo do galope."""
    out = []
    for i, ch in enumerate(acordes):
        raiz = G.voice_in_octave([min(ch)], oitava_grave)[0]
        for b in celula:
            out.append(Note(i * BAR + int(b * Q), int(Q * 0.42), raiz, vel))
            if quinta:
                out.append(Note(i * BAR + int(b * Q), int(Q * 0.42), raiz + 7,
                                vel - 10))
    return out


def baixo(acordes, celula=CHUGA, oitava_grave=None, vel=114):
    low = oitava_grave or (GRAVE - 12)
    out = []
    for i, ch in enumerate(acordes):
        raiz = G.voice_in_octave([min(ch)], low)[0]
        for j, b in enumerate(celula):
            nota = raiz + (12 if j == len(celula) - 1 else 0)
            out.append(Note(i * BAR + int(b * Q), int(Q * 0.45), nota, vel))
    return out


def bateria(compassos, estilo="rock", inicio_fill=True):
    """Kit GM: 36 bumbo, 38 caixa, 42 chimbal fechado, 46 aberto, 49 prato,
    45/47/50 toms."""
    out = []
    for c in range(compassos):
        b = c * BAR
        if estilo == "rock":
            for x in (0.0, 0.5, 2.0, 2.75):
                out.append(Note(b + int(x * Q), 100, 36, 118))
            for x in (1.0, 3.0):
                out.append(Note(b + int(x * Q), 100, 38, 112))
            for j in range(8):
                out.append(Note(b + j * (Q // 2), 90, 42, 74 if j % 2 else 88))
        elif estilo == "duplo":
            for j in range(8):
                out.append(Note(b + j * (Q // 2), 90, 36, 112))
            for x in (1.0, 3.0):
                out.append(Note(b + int(x * Q), 100, 38, 116))
            for j in range(16):
                out.append(Note(b + j * (Q // 4), 70, 42, 66 if j % 2 else 84))
        elif estilo == "meio":
            out.append(Note(b, 100, 36, 116))
            out.append(Note(b + Q * 2, 110, 38, 110))
            for j in range(4):
                out.append(Note(b + j * Q, 110, 46, 70))
        if c == 0 and inicio_fill:
            out.append(Note(b, 200, 49, 118))
        if c % 8 == 7:                     # virada no fim de cada frase
            for j, p in enumerate((47, 45, 43, 38)):
                out.append(Note(b + int((3.0 + j * 0.25) * Q), 90, p, 112))
    return out


def timpano(acordes, vel=112):
    out = []
    for i, ch in enumerate(acordes):
        raiz = G.voice_in_octave([min(ch)], N("E2"))[0]
        out.append(Note(i * BAR, Q, raiz, vel))
        if i % 4 == 3:
            for j in range(4):
                out.append(Note(i * BAR + Q * 3 + j * (Q // 4), Q // 4,
                                raiz, 80 + j * 12))
    return out


# --------------------------------------------------------------------------
def build(sf2: Path) -> Project:
    p = Project("Vigilia do Colosso", bpm=BPM, bit_depth=24)
    p.key_root, p.scale = 4, "natural_minor"          # Mi menor
    banco = SF.open_soundfont(sf2)

    def gm(nome, **kw):
        return SF.instrument(banco, nome, **kw)

    # a sala onde tudo acontece
    p.master_reverb = {"sala": 0.74, "damp": 0.38, "largura": 1.0,
                       "predelay_ms": 22.0}
    p.master_inserts = [{"tipo": "comp", "limiar_db": -14.0, "razao": 2.2,
                         "attack_ms": 14.0, "release_ms": 220.0,
                         "makeup_db": 1.5}]

    t = {}
    # ---- orquestra -----------------------------------------------------
    t["cordas_ost"] = p.new_track("Cordas ostinato", gm("Stereo Strings Fast", gain=0.5))
    t["cordas_ost"].section = {"vozes": 4, "detune": 9, "spread_ms": 22, "pan": 0.7}
    t["cordas_ost"].reverb_send = 0.45
    t["cordas_ost"].volume = 0.5

    t["cordas_pad"] = p.new_track("Cordas pad", gm("Stereo Strings Slow", gain=0.5))
    t["cordas_pad"].section = {"vozes": 4, "detune": 11, "spread_ms": 30, "pan": 0.8}
    t["cordas_pad"].reverb_send = 0.6
    t["cordas_pad"].volume = 0.45

    t["metais"] = p.new_track("Metais", gm("Brass Section", gain=0.6))
    t["metais"].section = {"vozes": 3, "detune": 7, "spread_ms": 14, "pan": 0.45}
    t["metais"].reverb_send = 0.38
    t["metais"].volume = 0.5

    t["trompas"] = p.new_track("Trompas", gm("French Horns", gain=0.7))
    t["trompas"].section = {"vozes": 3, "detune": 8, "spread_ms": 18, "pan": 0.35}
    t["trompas"].reverb_send = 0.42
    t["trompas"].volume = 0.55

    t["coro"] = p.new_track("Coro", gm("Concert Choir", gain=0.55))
    t["coro"].section = {"vozes": 4, "detune": 13, "spread_ms": 34, "pan": 0.85}
    t["coro"].reverb_send = 0.7
    t["coro"].volume = 0.42

    t["timpani"] = p.new_track("Timpani", gm("Timpani", gain=0.85))
    t["timpani"].reverb_send = 0.4

    # ---- banda ---------------------------------------------------------
    # gravacao dupla: duas guitarras iguais, uma em cada lado. E o truque mais
    # antigo do rock e continua sendo o que da largura de verdade.
    for lado, pan in (("esq", -0.85), ("dir", 0.85)):
        tr = p.new_track(f"Guitarra {lado}", gm("Distortion Guitar", gain=0.9))
        tr.inserts = [dict(fx.PRESETS_AMP["rock"], pan=pan)]
        tr.reverb_send = 0.12
        tr.volume = 0.62
        t[f"gtr_{lado}"] = tr

    t["gtr_lead"] = p.new_track("Guitarra lead", gm("Overdrive Guitar", gain=0.9))
    t["gtr_lead"].inserts = [dict(fx.PRESETS_AMP["crunch"], ganho=7.0, pan=0.12),
                             {"tipo": "eq", "freq": 1400, "ganho_db": 2.5, "q": 0.8}]
    t["gtr_lead"].reverb_send = 0.3
    t["gtr_lead"].volume = 0.55

    t["gtr_limpa"] = p.new_track("Guitarra limpa", gm("Clean Guitar", gain=0.85))
    t["gtr_limpa"].inserts = [dict(fx.PRESETS_AMP["limpo"], pan=-0.3),
                              {"tipo": "chorus", "profundidade_ms": 7.0,
                               "taxa": 0.4, "mix": 0.45}]
    t["gtr_limpa"].reverb_send = 0.35
    t["gtr_limpa"].volume = 0.5

    t["baixo"] = p.new_track("Baixo", gm("Finger Bass", gain=0.95))
    t["baixo"].inserts = [{"tipo": "eq", "modo": "lowshelf", "freq": 110,
                           "ganho_db": 4.0},
                          {"tipo": "eq", "freq": 900, "ganho_db": 3.0, "q": 1.1},
                          {"tipo": "comp", "limiar_db": -20.0, "razao": 4.0,
                           "attack_ms": 6.0, "release_ms": 110.0, "makeup_db": 3.0}]
    t["baixo"].volume = 0.72

    t["bateria"] = p.new_track("Bateria", gm("Power", gain=0.9))
    t["bateria"].instrument.is_drum = True
    t["bateria"].inserts = [{"tipo": "comp", "limiar_db": -16.0, "razao": 3.5,
                             "attack_ms": 4.0, "release_ms": 130.0,
                             "makeup_db": 2.5}]
    t["bateria"].reverb_send = 0.18
    t["bateria"].volume = 0.68

    # ---- o fio de chiptune que atravessa tudo --------------------------
    t["chip"] = p.new_track("Chip Lead", factory_by_name("NES Lead 25%"))
    t["chip"].instrument.vibrato_depth_cents = 18
    t["chip"].reverb_send = 0.25
    t["chip"].volume = 0.32

    acordes = {n: c for n, _, c in SECOES}
    ncomp = {n: k for n, k, _ in SECOES}

    def por(track, nome, secao, notas):
        if not notas:
            return
        pat = p.new_pattern(f"{nome} · {secao}", bars=ncomp[secao])
        pat.notes.extend(notas)
        p.place(pat, t[track], INICIO[secao])

    # =============================================== ABERTURA (1-8)
    s, ch = "ABERTURA", acordes["ABERTURA"]
    por("cordas_pad", "Pad", s, pad(ch, vel=62))
    por("trompas", "Coral", s, pad(ch, low=N("G3"), vel=64, sem_terca=True))
    por("timpani", "Timpani", s, timpano(ch, 88))
    por("cordas_ost", "Ostinato", s, ostinato(ch[4:], rate=2, vel=62))

    # =============================================== RIFF (9-24)
    s, ch = "RIFF", acordes["RIFF"]
    for lado in ("esq", "dir"):
        por(f"gtr_{lado}", "Riff", s, riff(ch))
    por("baixo", "Baixo", s, baixo(ch))
    por("bateria", "Bateria", s, bateria(16, "rock"))
    por("cordas_ost", "Ostinato", s, ostinato(ch, rate=4, vel=74))
    por("timpani", "Timpani", s, timpano(ch, 104))
    por("metais", "Golpes", s, pad(ch, low=N("G3"), vel=88, sem_terca=True))

    # =============================================== TEMA (25-40)
    s, ch = "TEMA", acordes["TEMA"]
    por("trompas", "Tema", s, escrever(TEMA, vel=104, vezes=2))
    por("chip", "Tema", s, escrever(TEMA, vel=88, vezes=2))
    for lado in ("esq", "dir"):
        por(f"gtr_{lado}", "Riff", s, riff(ch, CHUGA))
    por("baixo", "Baixo", s, baixo(ch))
    por("bateria", "Bateria", s, bateria(16, "rock"))
    por("cordas_ost", "Ostinato", s, ostinato(ch, rate=4, vel=80))
    por("cordas_pad", "Pad", s, pad(ch, vel=62))
    por("timpani", "Timpani", s, timpano(ch, 106))

    # =============================================== PONTE (41-52)
    s, ch = "PONTE", acordes["PONTE"]
    por("coro", "Coro", s, pad(ch, low=N("G3"), vel=74, sem_terca=True))
    por("gtr_limpa", "Arpejo", s,
        G.arpeggiate([G.voice_in_octave(triade(c), ACOMP) for c in ch], Q,
                     rate=2, mode="updown", octaves=1, velocity=86))
    por("cordas_pad", "Pad", s, pad(ch, vel=58))
    por("trompas", "Melodia", s, escrever(PONTE_MEL, vel=92, vezes=3, comp=4))
    por("baixo", "Baixo", s, baixo(ch, (0.0, 2.0), vel=100))
    por("bateria", "Bateria", s, bateria(12, "meio"))

    # =============================================== SOLO (53-68)
    s, ch = "SOLO", acordes["SOLO"]
    por("gtr_lead", "Solo", s, escrever(SOLO, vel=110, vezes=2))
    por("chip", "Solo", s, escrever(SOLO, oitava=1, vel=72, vezes=2))
    for lado in ("esq", "dir"):
        por(f"gtr_{lado}", "Riff", s, riff(ch, GALOPE))
    por("baixo", "Baixo", s, baixo(ch))
    por("bateria", "Bateria", s, bateria(16, "duplo"))
    por("cordas_ost", "Ostinato", s, ostinato(ch, rate=4, vel=82))
    por("cordas_pad", "Pad", s, pad(ch, vel=60))
    por("timpani", "Timpani", s, timpano(ch, 108))

    # =============================================== FINAL (69-84)
    s, ch = "FINAL", acordes["FINAL"]
    sub = 2                     # sobe um tom: Mi menor -> Fa# menor
    por("trompas", "Tema+", s, escrever(TEMA, transpor=sub, vel=108, vezes=2))
    por("metais", "Tema+", s, escrever(TEMA, transpor=sub, vel=100, vezes=2))
    por("chip", "Tema+", s, escrever(TEMA, transpor=sub, vel=92, vezes=2))
    por("gtr_lead", "Tema+", s, escrever(TEMA, transpor=sub, oitava=-1,
                                         vel=104, vezes=2))
    for lado in ("esq", "dir"):
        por(f"gtr_{lado}", "Riff", s, riff(ch, GALOPE))
    por("baixo", "Baixo", s, baixo(ch))
    por("bateria", "Bateria", s, bateria(16, "duplo"))
    por("cordas_ost", "Ostinato", s, ostinato(ch, rate=4, vel=86))
    por("cordas_pad", "Pad", s, pad(ch, vel=64))
    por("coro", "Coro", s, pad(ch, low=N("G3"), vel=78, sem_terca=True))
    por("timpani", "Timpani", s, timpano(ch, 112))

    # =============================================== ECO (85-92)
    s, ch = "ECO", acordes["ECO"]
    por("cordas_pad", "Pad", s, pad(ch, vel=58))
    por("trompas", "Eco", s, escrever(ECO_MEL, vel=84))
    por("coro", "Coro", s, pad(ch, low=N("G3"), vel=62, sem_terca=True))
    por("timpani", "Timpani", s, timpano(ch, 84))
    return p


def main():
    out = Path(__file__).parent
    print("Banco de instrumentos...")
    sf2 = L.ensure("generaluser_gs",
                   lambda g, tt: print(f"  {g/1e6:5.1f} MB", end="\r", flush=True))
    print(f"  {SF.open_soundfont(sf2).name}\n")

    p = build(sf2)
    dur = p.ticks_to_seconds(p.length_ticks)
    print(f"Vigilia do Colosso — {TOTAL} compassos, {p.bpm:.0f} BPM, "
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
        t0, t1 = INICIO[nome] * 4 * 60 / BPM, (INICIO[nome] + k) * 4 * 60 / BPM
        sel = (c.tempos >= t0) & (c.tempos < t1)
        print(f"  {nome:9s} nivel {c.niveis[sel].mean():.1f} "
              f"(pico {c.niveis[sel].max()})")

    print("\nRenderizando (amp, dobra de secao e sala custam caro)...")
    audio = engine.render_project(p)
    wav = engine.export_wav(audio, out / "vigilia_do_colosso", p.sample_rate, 16)
    chip = p.save(out / "vigilia_do_colosso")
    mid = smf.write_midi(smf.project_to_midi(p), out / "vigilia_do_colosso")
    js = c.exportar_json(out / "vigilia_do_colosso_intensidade", p.name)
    print(f"WAV       : {wav}  ({len(audio)/p.sample_rate:.1f}s)")
    print(f"CHIP      : {chip}")
    print(f"MIDI      : {mid}")
    print(f"SPRITES   : {js}  (tempo -> nivel, para a engine)")


if __name__ == "__main__":
    main()
