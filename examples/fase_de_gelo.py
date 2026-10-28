#!/usr/bin/env python3
"""
Fase de Gelo — peca de referencia do modo PS1.

Composicao original no espirito das fases de PlayStation da era Mega Man 8:
lead de onda quadrada sampleada, baixo slap sincopado, sinos de vidro em
arpejo, pad aberto e o reverb "Studio Large" do SPU fazendo o frio.

Ela e escrita INTEIRA pela API do Estudio — as mesmas chamadas que o servidor
MCP expoe. Ou seja: isto e exatamente o que um assistente faz quando compoe
pelo MCP, so que salvo num arquivo para voce ler e mexer.

    python3 examples/fase_de_gelo.py            # gera fase_de_gelo.wav/.chip
    python3 examples/fase_de_gelo.py --tocar    # e toca no alto-falante

Forma (32 compassos, ~50 s, loop volta ao compasso 5):
    1-4    Intro      sinos + pad, a bateria entra no 4
    5-12   A          tema
    13-20  A'         tema com final que abre para B
    21-28  B          relativa maior, sinos dobram o lead uma oitava acima
    29-32  Ponte      quebra: so pad e sinos, a bateria reconstroi
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chipforge.estudio import Estudio  # noqa: E402

AQUI = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Harmonia (cifras, um compasso por '|')
# --------------------------------------------------------------------------
INTRO = "Em9 | Cmaj7 | Am7 | B7sus4 B7"
A = "Em9 | Cmaj7 | Am7 | B7sus4 B7 | Em9 | Cmaj7 | Am7 D | G"
A2 = "Em9 | Cmaj7 | Am7 | B7sus4 B7 | Em9 | Cmaj7 | Am7 B7 | B7"
B = "Cmaj7 | D | Bm7 | Em7 | Am7 | D | Gmaj7 | B7"
PONTE = "Cmaj7 | Cmaj7 | D | Bsus4 B7"

# --------------------------------------------------------------------------
# Melodia (MML: l8 = colcheia padrao, > < mudam oitava, | confere compasso)
# --------------------------------------------------------------------------
TEMA_A = """
o5 l8
e r e f+ g4 b4        | a g f+ g e2              |
r e a b > c4 < b a    | b4. a g f+ d+4           |
e r e f+ g4 b4        | > c < b a b g4 e4        |
a b > c d c < b a f+  | g2 r d e f+              |
"""
TEMA_A2 = """
o5 l8
e r e f+ g4 b4        | a g f+ g e2              |
r e a b > c4 < b a    | b4. a g f+ d+4           |
e r e f+ g4 b4        | > c < b a b > d4 c4      |
< a > c e d < b a g f+ | d+4 f+4 b2              |
"""
TEMA_B = """
l8
g4. e g b > c4        | d4 < a4 f+4 a4           |
b4. a f+ d f+4        | e2 r e f+ g              |
a4. g a b > c4        | d c < b a f+4 d4         |
g a b > d f+4 d4      | d+2 < b4 a4              |
"""

# --------------------------------------------------------------------------
# Bateria (16 passos por compasso)
# --------------------------------------------------------------------------
GROOVE = """
k: x.....x.x.....x.|x.....x.x.x..x..
s: ....X.......X...|....X.......X.oX
h: x.x.x.x.x.x.x.xo|x.x.x.x.x.x.x.x.
"""
GROOVE_CRASH = """
k: x.....x.x.....x.|x.....x.x.x..x..
s: ....X.......X...|....X.......X.oX
h: ..x.x.x.x.x.x.xo|x.x.x.x.x.x.x.x.
c: X...............|................
"""
VIRADA = """
k: x.......x.......
s: ....X.......xxXX
t1: ........X.x.....
t2: ............x...
h: x.x.x.x.........
"""
INTRO_HATS = """
k: x...............|x...............|x.......x.......
h: x.x.x.x.x.x.x.x.|x.x.x.x.x.x.x.x.|x.x.x.x.x.x.x.x.
"""
PONTE_BAT = """
h: ..x...x...x...x.|..x...x...x...x.|..x.x.x.x.x.x.x.|................
k: ................|................|x...x...x...x...|x...x...x...x...
s: ................|................|............x.x.|xxxxxxxxXXXXXXXX
"""


def compor(e: Estudio) -> Estudio:
    e.novo("Fase de Gelo", bpm=152, tom="E", escala="natural_minor",
           console="ps1")

    # --- orquestra de PS1 ---------------------------------------------
    e.criar_track("Lead", "ps1:lead_quadrado", volume=0.7, pan=0.0,
                  reverb_send=0.32)
    e.criar_track("Sinos", "ps1:sino", volume=0.46, pan=0.55, reverb_send=0.55)
    e.criar_track("Teclado", "ps1:epiano", volume=0.5, pan=-0.5,
                  reverb_send=0.25)
    e.criar_track("Pad", "ps1:pad_halo", volume=0.34, pan=-0.12,
                  reverb_send=0.5)
    e.criar_track("Baixo", "ps1:baixo_slap", volume=0.6, reverb_send=0.04)
    e.criar_track("Bateria", "ps1:bateria", volume=0.8, reverb_send=0.14)
    # brilho de prato e chimbal: o banco GM e escuro, o PS1 mais ainda
    e.ajustar_track("Bateria", efeitos=[{"tipo": "eq", "modo": "highshelf",
                                         "freq": 6000, "ganho_db": 4.0,
                                         "q": 0.7}])

    # --- intro (1-4) ------------------------------------------------------
    e.acompanhar("Sinos", 1, INTRO, "arpejo16", oitava=5, velocity=78)
    e.acompanhar("Pad", 1, INTRO, "pad", oitava=4, velocity=80, vozes=3)
    e.acompanhar("Baixo", 3, "Am7 | B7sus4 B7", "baixo_longo", oitava=2)
    e.bateria("Bateria", 1, INTRO_HATS)
    e.bateria("Bateria", 4, VIRADA)

    # --- A (5-12) e A' (13-20) ----------------------------------------------
    e.escrever("Lead", 5, TEMA_A, velocity=104)
    e.escrever("Lead", 13, TEMA_A2, velocity=108)
    for ini, cif in ((5, A), (13, A2)):
        e.acompanhar("Baixo", ini, cif, "baixo_funk", oitava=2, velocity=100)
        e.acompanhar("Teclado", ini, cif, "stabs", oitava=4, velocity=82,
                     teto="C5")
        e.acompanhar("Pad", ini, cif, "pad", oitava=4, velocity=70, vozes=3,
                     teto="B4")
        e.bateria("Bateria", ini, GROOVE_CRASH)
        e.bateria("Bateria", ini + 2, GROOVE, repetir=3)
    e.bateria("Bateria", 20, VIRADA)

    # --- B (21-28): relativa maior, sinos dobram o lead 8va ------------------
    e.escrever("Lead", 21, "o5" + TEMA_B, velocity=110)
    e.escrever("Sinos", 21, "o6" + TEMA_B, velocity=64)
    e.acompanhar("Baixo", 21, B, "baixo_oitava", oitava=2, velocity=98)
    e.acompanhar("Teclado", 21, B, "galope", oitava=4, velocity=70,
                 vozes=2, teto="B4")
    e.acompanhar("Pad", 21, B, "pad", oitava=4, velocity=76, vozes=3,
                 teto="B4")
    e.bateria("Bateria", 21, GROOVE_CRASH)
    e.bateria("Bateria", 23, GROOVE, repetir=3)

    # --- ponte (29-32): quebra e reconstroi ----------------------------------
    e.acompanhar("Sinos", 29, PONTE, "arpejo16", oitava=5, velocity=74)
    e.acompanhar("Pad", 29, PONTE, "pad", oitava=4, velocity=86, vozes=3)
    e.acompanhar("Baixo", 29, PONTE, "baixo_longo", oitava=2, velocity=90)
    e.acompanhar("Baixo", 31, "D | Bsus4 B7", "baixo8", oitava=2, velocity=96)
    e.bateria("Bateria", 29, PONTE_BAT)
    # crescendo do pad na reconstrucao
    e.ajustar_track("Pad", automacao_volume=[[1, 0.34], [29, 0.34], [31, 0.34],
                                             [33, 0.6]])

    # --- sala e master -----------------------------------------------------
    e.sala("studio_large", wet=0.55)
    e.master("jogo")
    return e


if __name__ == "__main__":
    e = compor(Estudio())
    print(e.resumo())
    print()
    print(e.analisar(secoes=[["Intro", 1, 4], ["A", 5, 12], ["A'", 13, 20],
                             ["B", 21, 28], ["Ponte", 29, 32]]))
    print()
    print(e.renderizar(str(AQUI / "fase_de_gelo.wav")))
    print(e.salvar(str(AQUI / "fase_de_gelo.chip")))
    if "--tocar" in sys.argv:
        import time
        print(e.tocar(1))
        time.sleep(e.p.ticks_to_seconds(e.p.length_ticks) + 3)
