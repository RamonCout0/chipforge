"""
"Cantiga da Torre Afogada" — trilha cinematográfica com voz solista e coro.

Esta é a peça de referência do modo cinematográfico. Ela existe para provar,
com medida e não com adjetivo, que o Chipforge alcança o patamar de uma OST
comercial. Tudo que ela usa foi construído para isto:

  * VOZ SOLISTA por síntese de formantes (`audio/voz.py`) — não é sample de
    "Choir Aahs": é fonte glotal com jitter, vibrato de entrada atrasada e o
    formante do cantor em 3 kHz, que é o que faz uma voz atravessar a
    orquestra sem gritar
  * CORO SATB, quatro naipes, três gargantas cada, com condução de vozes
  * SALA POR CONVOLUÇÃO (`audio/espaco.py`) — resposta ao impulso sintetizada
    com decaimento por banda, sem as ressonâncias de filtro pente
  * DESENHO DE SOM (`audio/cinema.py`) — impacto, riser, braam, taiko, sub drop
  * MAPA DE TEMPO (`core/dinamica.py`) — accelerando na fúria, ritardando no
    fim; a peça não anda no metrônomo
  * AUTOMAÇÃO DE VOLUME por seção — o arco que uma velocity de nota não faz
  * MASTER com alvo de LUFS, excitador de harmônicos e limitador com
    antecipação (`audio/master.py`)

E, se você passar o caminho de uma faixa de referência na linha de comando, o
master persegue o BALANÇO ESPECTRAL medido dela — sem copiar uma única nota:

    python examples/cantiga_da_torre_afogada.py
    python examples/cantiga_da_torre_afogada.py "caminho/da/referencia.mp3"

Forma (84 compassos, 100 BPM, Ré menor, ~3:20):

    PROLOGO   c. 1-8    voz sozinha e harpa                  nível 1
    INVOCACAO c. 9-20   coro de boca fechada, violoncelos    nível 2-3
    TEMA      c.21-36   orquestra e ostinato; o tema         nível 4
    QUEBRA    c.37-44   tudo cai: voz, harpa e um pedal      nível 2
    FURIA     c.45-60   metais, taiko, braam; accelerando    nível 5
    LAMENTO   c.61-72   sobe uma terça; voz sobre cordas     nível 3
    CODA      c.73-84   tutti, e ritardando até o silêncio   nível 5->1
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from chipforge.audio import cinema as CIN
from chipforge.audio import engine
from chipforge.audio import espaco
from chipforge.audio import fx
from chipforge.audio import master as MST
from chipforge.audio import sampler as SF
from chipforge.audio import voz as VOZ
from chipforge.core import dinamica as DIN
from chipforge.core import expressao as EXP
from chipforge.core import harmony as H
from chipforge.core import intensity as INT
from chipforge.core import orquestra as ORQ
from chipforge.core import theory as T
from chipforge.core.project import Note, Project
from chipforge.midiio import library as L
from chipforge.midiio import smf

BPM = 100
Q = 480
COMPASSO = Q * 4
N = T.name_to_midi


def ac(raiz: str, qual: str = "min", oit: int = 3) -> list[int]:
    r = N(f"{raiz}{oit}")
    return [r + i for i in T.CHORDS[qual]]


def prog(seq, vezes: int = 1) -> list[list[int]]:
    return [ac(*c) for c in seq] * vezes


# --------------------------------------------------------------------------
# Harmonia — Ré menor; o LAMENTO sobe uma terça menor (Fá menor)
# --------------------------------------------------------------------------
# O sexto grau de Re menor e SI BEMOL, escrito aqui como "A#" porque
# `name_to_midi` fala em sustenidos. Escrever "B" seria si natural — um acorde
# que nao existe na tonalidade e que arrasta a analise para outro tom.
SIb = "A#"
MIb, LAb, REb = "D#", "G#", "C#"        # o LAMENTO mora em Fa menor

SECOES = [
    ("PROLOGO", 8, prog([("D", "min"), ("D", "min"), (SIb, "maj"), (SIb, "maj"),
                         ("F", "maj"), ("F", "maj"), ("A", "min"), ("A", "min")])),
    ("INVOCACAO", 12, prog([("D", "min"), (SIb, "maj"), ("F", "maj"), ("C", "maj")], 3)),
    ("TEMA", 16, prog([("D", "min"), (SIb, "maj"), ("F", "maj"), ("C", "maj")], 4)),
    ("QUEBRA", 8, prog([("G", "min"), ("G", "min"), ("D", "min"), ("D", "min"),
                        (SIb, "maj"), (SIb, "maj"), ("A", "maj"), ("A", "maj")])),
    # A FURIA reafirma o TEMA, entao ela precisa da harmonia CONTRA A QUAL o
    # tema foi escrito. A primeira versao usava a tetracorde descendente
    # (Dm-C-Sib-La) por soar mais tensa, e o medidor apontou o preco: o La
    # maior traz do# contra o do natural da melodia, e o tema batia de segunda
    # menor com o coro em treze pontos. Tensao de climax se faz com
    # orquestracao e andamento, nao contra a propria melodia.
    ("FURIA", 16, prog([("D", "min"), (SIb, "maj"), ("F", "maj"), ("C", "maj")], 3)
     + prog([("D", "min"), (SIb, "maj"), ("C", "maj"), ("A", "maj")])),
    ("LAMENTO", 12, prog([("F", "min"), (REb, "maj"), (LAb, "maj"), (MIb, "maj")], 3)),
    ("CODA", 12, prog([("D", "min"), (SIb, "maj"), ("G", "min"), ("A", "maj"),
                       ("D", "min"), (SIb, "maj"), ("F", "maj"), ("C", "maj"),
                       ("G", "min"), ("A", "maj"), ("D", "min"), ("D", "min")])),
]

INICIO: dict[str, int] = {}
_b = 0
for _n, _k, _ in SECOES:
    INICIO[_n] = _b
    _b += _k
TOTAL = _b

# Volume relativo de cada seção — é este dicionário que vira o arco da peça.
# Sem ele, a faixa mede LRA perto de zero e soa toda no mesmo volume.
NIVEIS = {"PROLOGO": 0.30, "INVOCACAO": 0.52, "TEMA": 0.82, "QUEBRA": 0.38,
          "FURIA": 1.00, "LAMENTO": 0.55, "CODA": 0.95}

# --------------------------------------------------------------------------
# Temas
# --------------------------------------------------------------------------
# (nota, compasso, tempo, duração em semínimas)
TEMA = [
    ("A4", 0, 0.0, 1.5), ("A#4", 0, 1.5, 0.5), ("A4", 0, 2.0, 1.0), ("F4", 0, 3.0, 1.0),
    ("G4", 1, 0.0, 2.0), ("E4", 1, 2.0, 2.0),
    ("F4", 2, 0.0, 1.0), ("A4", 2, 1.0, 1.0), ("D5", 2, 2.0, 2.0),
    ("C5", 3, 0.0, 1.5), ("A4", 3, 1.5, 0.5), ("A4", 3, 2.0, 1.5),
    ("D5", 4, 0.0, 1.0), ("E5", 4, 1.0, 1.0), ("F5", 4, 2.0, 2.0),
    ("E5", 5, 0.0, 1.5), ("D5", 5, 1.5, 0.5), ("C5", 5, 2.0, 2.0),
    ("A#4", 6, 0.0, 1.0), ("D5", 6, 1.0, 1.0), ("F5", 6, 2.0, 1.0), ("E5", 6, 3.0, 1.0),
    ("D5", 7, 0.0, 2.5),
]

PROLOGO_MEL = [
    ("D4", 0, 1.0, 2.0), ("F4", 0, 3.0, 1.0),
    ("E4", 1, 0.0, 2.0), ("D4", 1, 2.0, 1.5),
    ("A#3", 2, 0.5, 1.5), ("D4", 2, 2.0, 2.0),
    ("C4", 3, 0.0, 3.0),
    ("F4", 4, 0.0, 1.5), ("E4", 4, 1.5, 0.5), ("D4", 4, 2.0, 2.0),
    ("C4", 5, 0.0, 2.0), ("A3", 5, 2.0, 2.0),
    ("A3", 6, 0.0, 1.0), ("C4", 6, 1.0, 1.0), ("E4", 6, 2.0, 2.0),
    ("D4", 7, 0.0, 3.0),
]

QUEBRA_MEL = [
    ("G4", 0, 0.0, 2.0), ("A#4", 0, 2.0, 2.0),
    ("A4", 1, 0.0, 3.0),
    ("F4", 2, 0.0, 1.5), ("A4", 2, 1.5, 0.5), ("D5", 2, 2.0, 2.0),
    ("C5", 3, 0.0, 2.5),
    ("A#4", 4, 0.0, 2.0), ("D5", 4, 2.0, 2.0),
    ("A4", 5, 0.0, 3.0),
    ("C#5", 6, 0.0, 2.0), ("E5", 6, 2.0, 2.0),
    ("A4", 7, 0.0, 3.0),
]

LAMENTO_MEL = [                      # Fa menor: Fa Sol Lab Sib Do Reb Mib
    ("F5", 0, 0.0, 2.0), ("D#5", 0, 2.0, 2.0),
    ("C#5", 1, 0.0, 2.0), ("C5", 1, 2.0, 2.0),
    ("A#4", 2, 0.0, 1.5), ("G#4", 2, 1.5, 0.5), ("G4", 2, 2.0, 2.0),
    ("F4", 3, 0.0, 3.0),
]

CODA_MEL = [
    ("D5", 0, 0.0, 1.5), ("F5", 0, 1.5, 0.5), ("E5", 0, 2.0, 2.0),
    ("D5", 1, 0.0, 2.0), ("A#4", 1, 2.0, 2.0),
    ("G4", 2, 0.0, 1.0), ("A#4", 2, 1.0, 1.0), ("D5", 2, 2.0, 2.0),
    ("C#5", 3, 0.0, 2.0), ("A4", 3, 2.0, 2.0),
    ("D5", 4, 0.0, 1.0), ("E5", 4, 1.0, 1.0), ("F5", 4, 2.0, 2.0),
    ("E5", 5, 0.0, 1.5), ("D5", 5, 1.5, 0.5), ("A4", 5, 2.0, 2.0),
    ("A#4", 6, 0.0, 2.0), ("A4", 6, 2.0, 2.0),
    ("D5", 7, 0.0, 3.5),
]


def mel(linha, **kw) -> list[Note]:
    return ORQ.escrever(linha, Q, 4, **kw)


# --------------------------------------------------------------------------
def build(sf2: Path) -> Project:
    sf = SF.open_soundfont(sf2)

    def gm(nome: str, **kw):
        return SF.instrument(sf, nome, **kw)

    p = Project(name="Cantiga da Torre Afogada", bpm=BPM, ppq=Q,
                bit_depth=24, master_gain=0.9,
                key_root=T.name_to_midi("D4") % 12, scale="minor")

    # A sala. Catedral no envio geral porque a peça é carregada por voz e coro,
    # e é lá que voz e coro moram. O corte de grave em 150 Hz é o que impede
    # que o contrabaixo e os taikos virem nuvem dentro da sala.
    p.master_reverb = {"sala": "catedral", "tamanho": 0.82, "wet": 0.9,
                       "corte_grave": 150.0, "corte_agudo": 8500.0}
    p.master_perfil = "cinematico"

    # ---- mapa de tempo -------------------------------------------------
    # A peça respira: acelera entrando na FURIA, segura na CODA. Andar no
    # metrônomo do início ao fim é a assinatura sonora de MIDI.
    mapa = DIN.MapaDeTempo(ppq=Q, bpm_base=BPM)
    mapa.marcar(0, BPM * 0.94)
    mapa.rampa(INICIO["INVOCACAO"] * COMPASSO, INICIO["TEMA"] * COMPASSO,
               BPM * 0.94, BPM)
    mapa.rampa((INICIO["FURIA"] - 2) * COMPASSO, (INICIO["FURIA"] + 8) * COMPASSO,
               BPM, BPM * 1.07)
    mapa.rampa(INICIO["LAMENTO"] * COMPASSO, (INICIO["LAMENTO"] + 4) * COMPASSO,
               BPM * 1.07, BPM * 0.90)
    mapa.rampa(INICIO["CODA"] * COMPASSO, (INICIO["CODA"] + 4) * COMPASSO,
               BPM * 0.90, BPM * 1.02)
    mapa.ritardando(TOTAL * COMPASSO, 4, COMPASSO, fator=0.68)
    p.tempo_map = mapa.to_list()

    t: dict[str, object] = {}

    # =================================================== VOZ E CORO
    t["voz"] = p.new_track("Voz solista", VOZ.voz("Solista feminina", pan=-0.05))
    t["voz"].volume = 0.80
    t["voz"].reverb_send = 0.42
    t["voz"].inserts = [
        # o corte de grave tira o ronco do banco de formantes; o realce de
        # 2.5 kHz é a presença que faz a letra ser entendida por cima do tutti
        {"tipo": "eq", "modo": "highpass", "freq": 130.0, "q": 0.7},
        {"tipo": "eq", "freq": 2500.0, "ganho_db": 2.5, "q": 0.9},
        {"tipo": "comp", "limiar_db": -20.0, "razao": 3.0,
         "attack_ms": 12.0, "release_ms": 160.0, "makeup_db": 3.0},
    ]

    t["voz_eco"] = p.new_track("Voz (eco)", VOZ.voz("Lamento (u)", pan=0.35,
                                                    gain=0.28))
    t["voz_eco"].volume = 0.34
    t["voz_eco"].reverb_send = 0.85

    for naipe, preset, pan in (
            ("soprano", "Coro soprano (aah)", -0.45),
            ("contralto", "Coro contralto (ooh)", 0.40),
            ("tenor", "Coro tenor (aah)", -0.30),
            ("baixo", "Coro baixo (ooh)", 0.28)):
        tr = p.new_track(f"Coro {naipe}", VOZ.voz(preset, pan=pan, seed=hash(naipe) % 999))
        tr.section = dict(VOZ.SECAO_CORO_INTIMO)
        tr.reverb_send = 0.75
        tr.volume = 0.46
        t[f"coro_{naipe}"] = tr

    t["hum"] = p.new_track("Coro (boca fechada)",
                           VOZ.voz("Coro de boca fechada", pan=0.0))
    t["hum"].section = dict(VOZ.SECAO_CORO)
    t["hum"].reverb_send = 0.8
    t["hum"].volume = 0.42

    # =================================================== ORQUESTRA
    t["harpa"] = p.new_track("Harpa", gm("Orchestral Harp", gain=0.9))
    t["harpa"].reverb_send = 0.5
    t["harpa"].volume = 0.5
    t["harpa"].pan = 0.25

    t["cordas_pad"] = p.new_track("Cordas (pad)", gm("Stereo Strings Slow", gain=0.85))
    t["cordas_pad"].section = {"vozes": 4, "detune": 10, "spread_ms": 26, "pan": 0.7}
    t["cordas_pad"].reverb_send = 0.55
    t["cordas_pad"].volume = 0.50

    t["cordas_ost"] = p.new_track("Cordas (ostinato)", gm("Stereo Strings Fast", gain=0.85))
    t["cordas_ost"].section = {"vozes": 3, "detune": 8, "spread_ms": 16, "pan": 0.55}
    t["cordas_ost"].reverb_send = 0.35
    t["cordas_ost"].volume = 0.44
    t["cordas_ost"].inserts = [{"tipo": "eq", "freq": 3200.0, "ganho_db": 2.0, "q": 0.8}]

    t["celos"] = p.new_track("Violoncelos", gm("Cello", gain=0.9))
    t["celos"].section = {"vozes": 3, "detune": 7, "spread_ms": 22, "pan": 0.4}
    t["celos"].reverb_send = 0.42
    t["celos"].volume = 0.58

    t["baixos"] = p.new_track("Contrabaixos", gm("Double Bass", gain=0.95))
    t["baixos"].reverb_send = 0.16
    t["baixos"].volume = 0.62
    t["baixos"].inserts = [{"tipo": "eq", "modo": "lowshelf", "freq": 90.0,
                            "ganho_db": 2.5}]

    t["trompas"] = p.new_track("Trompas", gm("French Horns", gain=0.9))
    t["trompas"].section = {"vozes": 3, "detune": 8, "spread_ms": 20, "pan": 0.35}
    t["trompas"].reverb_send = 0.48
    t["trompas"].volume = 0.52

    t["metais"] = p.new_track("Metais", gm("Brass Section", gain=0.9))
    t["metais"].section = {"vozes": 2, "detune": 6, "spread_ms": 14, "pan": 0.45}
    t["metais"].reverb_send = 0.38
    t["metais"].volume = 0.46

    t["timpani"] = p.new_track("Timpani", gm("Timpani", gain=0.9))
    t["timpani"].reverb_send = 0.4
    t["timpani"].volume = 0.6

    # =================================================== DESENHO DE SOM
    t["taiko"] = p.new_track("Taiko", CIN.instrumento("Taiko"))
    t["taiko"].reverb_send = 0.22
    t["taiko"].volume = 0.66

    t["taiko_alto"] = p.new_track("Taiko pequeno",
                                  CIN.instrumento("Taiko pequeno", pan=-0.4))
    t["taiko_alto"].reverb_send = 0.3
    t["taiko_alto"].volume = 0.4

    t["impacto"] = p.new_track("Impacto", CIN.instrumento("Impacto grave"))
    t["impacto"].reverb_send = 0.45
    t["impacto"].volume = 0.72

    t["riser"] = p.new_track("Riser", CIN.instrumento("Riser longo"))
    t["riser"].reverb_send = 0.3
    t["riser"].volume = 0.34

    t["swell"] = p.new_track("Swell", CIN.instrumento("Swell invertido"))
    t["swell"].reverb_send = 0.5
    t["swell"].volume = 0.30

    t["braam"] = p.new_track("Braam", CIN.instrumento("Braam"))
    t["braam"].reverb_send = 0.35
    t["braam"].volume = 0.42

    t["subdrop"] = p.new_track("Sub drop", CIN.instrumento("Sub drop"))
    t["subdrop"].reverb_send = 0.1
    t["subdrop"].volume = 0.55

    # ---- arco de dinâmica ---------------------------------------------
    arco = DIN.arco_de_secoes([(n, k) for n, k, _ in SECOES], COMPASSO, NIVEIS)
    for chave, tr in t.items():
        # percussão e desenho de som ficam de fora: o impacto da CODA precisa
        # bater com o mesmo tamanho do impacto da FURIA, senão o fim encolhe
        if chave in ("impacto", "subdrop", "braam", "riser", "swell"):
            continue
        tr.automacao = {"volume": arco.to_list()}

    # ================================================================
    acordes = {n: c for n, _, c in SECOES}
    ncomp = {n: k for n, k, _ in SECOES}

    def por(track: str, nome: str, secao: str, notas):
        if not notas:
            return
        pat = p.new_pattern(f"{nome} · {secao}", bars=ncomp[secao])
        pat.notes.extend(notas)
        p.place(pat, t[track], INICIO[secao])

    def humano(notas, espalhar=20.0, vel=8):
        return ORQ.humanizar_naipe(notas, Q, espalhar, BPM, vel)

    # =============================================== PROLOGO
    s, ch = "PROLOGO", acordes["PROLOGO"]
    por("voz", "Melodia", s, EXP.arco(mel(PROLOGO_MEL, velocity=76), "sino", 0.22))
    por("harpa", "Arpejo", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=2, low=52, high=76,
                     velocity=58, contorno="sobe"), 26, 10))
    por("cordas_pad", "Pad", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=57, high=79, velocity=46), 30))
    por("celos", "Pedal", s, humano(ORQ.pedal(ch, Q, 1, 4, "violoncelo", 54), 34))

    # =============================================== INVOCACAO
    s, ch = "INVOCACAO", acordes["INVOCACAO"]
    por("hum", "Hum", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=4, low=50, high=74, velocity=62), 40, 10))
    por("celos", "Pedal", s, humano(ORQ.pedal(ch, Q, 1, 4, "violoncelo", 66), 30))
    por("baixos", "Pedal", s, humano(ORQ.pedal(ch, Q, 1, 4, "contrabaixo", 72,
                                               oitava_dobrada=False), 36))
    por("cordas_pad", "Pad", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=60, high=81, velocity=52), 28))
    por("harpa", "Arpejo", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=2, low=52, high=79,
                     velocity=60, contorno="sobe_desce"), 24, 10))
    por("taiko", "Distante", s, ORQ.taiko(12, Q, "meio", 4, 36, 74))
    por("voz", "Contracanto", s, EXP.arco(
        mel(PROLOGO_MEL, velocity=70, transpor=0, oitava=1, compassos=8)[:10],
        "sino", 0.2))

    # =============================================== TEMA
    s, ch = "TEMA", acordes["TEMA"]
    por("voz", "Tema", s, EXP.arco(mel(TEMA, velocity=96, vezes=2), "crescendo", 0.18))
    por("voz_eco", "Tema", s, mel(TEMA, velocity=62, vezes=2))
    por("cordas_ost", "Ostinato", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=4, low=64, high=86, velocity=70), 14, 7))
    por("cordas_pad", "Pad", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=4, low=57, high=81, velocity=62), 26))
    por("celos", "Linha", s, humano(ORQ.pedal(ch, Q, 1, 4, "violoncelo", 78), 28))
    por("baixos", "Pedal", s, humano(ORQ.pedal(ch, Q, 1, 4, "contrabaixo", 82), 32))
    por("trompas", "Coral", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=48, high=69, velocity=72), 24))
    por("coro_soprano", "Aah", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=1, low=67, high=79, velocity=72), 38, 9))
    por("coro_contralto", "Ooh", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=1, low=58, high=72, velocity=70), 38, 9))
    por("timpani", "Marcação", s, ORQ.taiko(16, Q, "meio", 4,
                                            ORQ.encaixar(N("D2"), "timpani"), 84))
    por("taiko", "Marcha", s, ORQ.taiko(16, Q, "marcha", 4, 36, 88))
    por("harpa", "Arpejo", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=2, low=55, high=84, velocity=62), 22, 10))

    # =============================================== QUEBRA
    s, ch = "QUEBRA", acordes["QUEBRA"]
    por("voz", "Quebra", s, EXP.arco(mel(QUEBRA_MEL, velocity=82), "sino", 0.24))
    por("harpa", "Arpejo", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=2, low=50, high=74,
                     velocity=54, contorno="pendulo"), 26, 10))
    por("baixos", "Pedal", s, humano(ORQ.pedal(ch, Q, 1, 4, "contrabaixo", 62,
                                               oitava_dobrada=False), 34))
    por("cordas_pad", "Tremolo", s, humano(
        ORQ.tremolo(ch[4:], Q, 1, 4, low=62, high=79, velocity=40), 12, 6))

    # =============================================== FURIA
    s, ch = "FURIA", acordes["FURIA"]
    por("voz", "Tema forte", s, EXP.arco(
        mel(TEMA, velocity=112, vezes=2, oitava=0), "crescendo", 0.12))
    por("cordas_ost", "Ostinato", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=4, low=64, high=88, velocity=88), 12, 8))
    por("cordas_pad", "Pad", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=4, low=57, high=81, velocity=76), 24))
    por("metais", "Bloco", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=52, high=76, velocity=94), 20))
    por("trompas", "Bloco", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=45, high=67, velocity=92), 22))
    por("celos", "Linha", s, humano(ORQ.pedal(ch, Q, 1, 4, "violoncelo", 92), 26))
    por("baixos", "Pedal", s, humano(ORQ.pedal(ch, Q, 1, 4, "contrabaixo", 96), 30))
    for naipe, low, high, vel in (("soprano", 67, 81, 92), ("contralto", 58, 72, 88),
                                  ("tenor", 50, 67, 88), ("baixo", 43, 60, 90)):
        por(f"coro_{naipe}", "Tutti", s, humano(
            ORQ.bloco(ch, Q, 1, 4, n_vozes=1, low=low, high=high, velocity=vel), 34, 9))
    por("taiko", "Guerra", s, ORQ.taiko(16, Q, "guerra", 4, 36, 112))
    por("taiko_alto", "Galope", s, ORQ.taiko(16, Q, "galope", 4, 36, 88))
    por("timpani", "Rufo", s, ORQ.taiko(16, Q, "marcha", 4,
                                        ORQ.encaixar(N("D2"), "timpani"), 100))
    por("braam", "Golpes", s, [Note(c * COMPASSO, COMPASSO * 2, N("D2"), 112)
                               for c in (0, 8)])

    # =============================================== LAMENTO
    s, ch = "LAMENTO", acordes["LAMENTO"]
    por("voz", "Lamento", s, EXP.arco(
        mel(LAMENTO_MEL, velocity=84, vezes=3), "sino", 0.26))
    por("cordas_pad", "Pad", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=4, low=55, high=79, velocity=54), 30))
    por("celos", "Pedal", s, humano(ORQ.pedal(ch, Q, 1, 4, "violoncelo", 64), 32))
    por("hum", "Hum", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=50, high=72, velocity=58), 42, 10))
    por("harpa", "Arpejo", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=2, low=52, high=79,
                     velocity=52, contorno="desce"), 26, 10))

    # =============================================== CODA
    s, ch = "CODA", acordes["CODA"]
    por("voz", "Coda", s, EXP.arco(mel(CODA_MEL, velocity=108), "crescendo", 0.14))
    por("voz_eco", "Coda", s, mel(CODA_MEL, velocity=66))
    por("cordas_ost", "Ostinato", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=4, low=64, high=88, velocity=82), 13, 8))
    por("cordas_pad", "Pad", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=4, low=57, high=84, velocity=74), 24))
    por("metais", "Bloco", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=52, high=76, velocity=88), 20))
    por("trompas", "Bloco", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=45, high=67, velocity=86), 22))
    por("celos", "Linha", s, humano(ORQ.pedal(ch, Q, 1, 4, "violoncelo", 88), 26))
    por("baixos", "Pedal", s, humano(ORQ.pedal(ch, Q, 1, 4, "contrabaixo", 92), 30))
    for naipe, low, high, vel in (("soprano", 67, 81, 90), ("contralto", 58, 72, 86),
                                  ("tenor", 50, 67, 86), ("baixo", 43, 60, 88)):
        por(f"coro_{naipe}", "Tutti", s, humano(
            ORQ.bloco(ch, Q, 1, 4, n_vozes=1, low=low, high=high, velocity=vel), 34, 9))
    por("taiko", "Final", s, ORQ.taiko(9, Q, "marcha", 4, 36, 104))
    por("timpani", "Final", s, ORQ.taiko(9, Q, "meio", 4,
                                         ORQ.encaixar(N("D2"), "timpani"), 96))
    por("harpa", "Arpejo", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=2, low=55, high=86, velocity=64), 22, 10))

    # ---- preparação das entradas ---------------------------------------
    # Cada uma destas coincidências (riser terminando no downbeat, impacto
    # começando nele) custa três notas e é a diferença entre "a seção mudou"
    # e "a seção CHEGOU".
    for alvo, forca in (("TEMA", 104), ("FURIA", 120), ("CODA", 118)):
        ev = ORQ.preparacao(INICIO[alvo], Q, 4, 2.0, True, True,
                            pitch_riser=N("D3"), pitch_impacto=N("D1"))
        for chave, notas in ev.items():
            if not notas:
                continue
            # `preparacao` devolve posições absolutas na peça; o pattern precisa
            # delas relativas ao próprio início, e o clip carrega o resto
            base = int(max(0, min(n.start for n in notas) // COMPASSO))
            pat = p.new_pattern(f"{chave} -> {alvo}", bars=3)
            pat.notes.extend(Note(n.start - base * COMPASSO, n.dur, n.pitch,
                                  min(127, forca)) for n in notas)
            p.place(pat, t[chave], base)

    # o sub drop marca a queda para a QUEBRA: é o corte, não a chegada
    pat = p.new_pattern("Sub drop → QUEBRA", bars=2)
    pat.notes.append(Note(0, COMPASSO, N("D2"), 116))
    p.place(pat, t["subdrop"], INICIO["QUEBRA"])

    return p


# --------------------------------------------------------------------------
def main() -> None:
    out = Path(__file__).parent
    ref_path = sys.argv[1] if len(sys.argv) > 1 else None

    print("Banco de instrumentos...")
    sf2 = L.ensure("generaluser_gs",
                   lambda g, tt: print(f"  {g/1e6:5.1f} MB", end="\r", flush=True))
    print(f"  {SF.open_soundfont(sf2).name}\n")

    perfil_alvo = None
    alvo_lufs = -14.0
    if ref_path:
        from chipforge.analysis import reference as REF
        print(f"Medindo a referência: {Path(ref_path).name}")
        r = REF.analisar(ref_path)
        perfil_alvo = r.perfil_alvo()
        alvo_lufs = r.alvo_de_loudness()
        print(f"  alvo: {alvo_lufs:.1f} LUFS · {perfil_alvo}\n")

    p = build(sf2)
    if perfil_alvo:
        p.master_alvo = perfil_alvo

    dur = p.ticks_to_seconds(p.length_ticks)
    n_notas = sum(len(x.notes) for x in p.patterns.values())
    print(f"{p.name} — {TOTAL} compassos, {BPM} BPM, "
          f"{int(dur//60)}:{int(dur%60):02d}")
    print(f"{len(p.tracks)} tracks · {len(p.patterns)} patterns · {n_notas} notas")
    print(f"mapa de tempo: {len(p.tempo_map)} marcas "
          f"({min(b for _, b in p.tempo_map):.0f}–"
          f"{max(b for _, b in p.tempo_map):.0f} BPM)\n")

    secoes = [(n, INICIO[n] * 4.0, (INICIO[n] + k) * 4.0) for n, k, _ in SECOES]
    reps = H.analyze_sections(p.harmony_events(), secoes, beats_per_chord=4.0)
    print(H.sections_table_pt(reps))

    c = INT.calcular(p)
    print("\nIntensidade (o que o Palco recebe):")
    print("  " + c.resumo_pt(86))

    print("\nRenderizando (voz, coro por dobra e sala por convolução)...")
    t0 = time.time()
    audio = engine.render_project(p, apply_master=False)
    print(f"  mixagem crua em {time.time()-t0:.1f}s")

    m = MST.masterizar(audio, p.sample_rate, "cinematico",
                       alvo_lufs=alvo_lufs, perfil_espectral=perfil_alvo)
    print(f"  master em {time.time()-t0:.1f}s no total\n")
    print(m.relatorio_pt(alvo_lufs))

    if ref_path:
        from chipforge.analysis import reference as REF
        r = REF.analisar(ref_path)
        print("\nDistância para a referência (banda a banda, em dB):")
        for k, v in r.comparar(m.audio, p.sample_rate).items():
            seta = "acima" if v > 0 else "abaixo"
            print(f"  {k:12s} {v:+5.1f} dB {seta if abs(v) > 1.5 else 'no alvo'}")

    wav = engine.export_wav(m.audio, out / "cantiga_da_torre_afogada",
                            p.sample_rate, 16)
    chip = p.save(out / "cantiga_da_torre_afogada")
    mid = smf.write_midi(smf.project_to_midi(p), out / "cantiga_da_torre_afogada")
    js = c.exportar_json(out / "cantiga_da_torre_afogada_intensidade", p.name)
    print(f"\nWAV     : {wav}  ({len(m.audio)/p.sample_rate:.1f}s)")
    print(f"CHIP    : {chip}")
    print(f"MIDI    : {mid}")
    print(f"SPRITES : {js}")


if __name__ == "__main__":
    main()
