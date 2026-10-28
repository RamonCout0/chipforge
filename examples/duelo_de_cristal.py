#!/usr/bin/env python3
"""
Duelo de Cristal — tema de chefe no modo PS1.

Rock sintetico da era PlayStation: duas guitarras (modelagem fisica de corda
+ simulador de amplificador) abertas em L/R, baixo tocando a MESMA figura
ritmica do riff, lead de serra com naipe desafinado e eco, metais em stab,
coro e cordas na ponte, e uma bateria humanizada que puxa um pouco a frente.

Escrita inteira pela API do Estudio (as mesmas chamadas do servidor MCP).

    python3 examples/duelo_de_cristal.py          # gera .wav e .chip
    python3 examples/duelo_de_cristal.py --tocar

Forma (36 compassos, ~50 s, loop volta ao compasso 5):
    1-4    Intro     golpes da banda inteira, virada de tons
    5-8    Riff      o riff sozinho: e o que gruda
    9-16   Tema A    lead entra com o motivo de salto (Re-La-Re)
    17-24  Tema B    melodia longa, abre para cima; metais em stab
    25-32  Ponte     meio-tempo, coro, napolitano (Mib) e a volta pelo La
    33-36  Riff      de novo, para fechar o loop
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chipforge.core.theory import name_to_midi  # noqa: E402
from chipforge.estudio import Estudio  # noqa: E402

AQUI = Path(__file__).resolve().parent
NOMES = ["c", "c+", "d", "d+", "e", "f", "f+", "g", "g+", "a", "a+", "b"]


def nota(m: int) -> str:
    """MIDI -> token MML com oitava explicita ('o2 d')."""
    return f"o{m // 12 - 1} {NOMES[m % 12]}"


def quinta(m: int) -> str:
    """Power chord (fundamental + quinta). Duas vozes e nao tres: no SPU cada
    nota e uma voz das 24, e a oitava de cima o amplificador ja cria."""
    return f"[{nota(m)} {nota(m + 7)}]"


def _dur(n16: int) -> str:
    """Duracao em semicolcheias -> MML ('8.', '2', '4^16'...)."""
    tabela = {16: "1", 12: "2.", 8: "2", 6: "4.", 4: "4", 3: "8.", 2: "8", 1: "16"}
    partes = []
    for v in sorted(tabela, reverse=True):
        while n16 >= v:
            partes.append(tabela[v])
            n16 -= v
    return "^".join(partes)


def chug(raizes: list[str], padrao: str, acorde: bool = True) -> str:
    """Compassos de base a partir das fundamentais e de um padrao de 16 passos.
    Cada caractere e uma semicolcheia:
        X chug acentuado, x chug, P power chord acentuado, p power chord,
        - segura o anterior, . pausa
    Com `acorde=False` (o baixo) todo P/p vira nota simples.
    """
    out = []
    for r in raizes:
        m = name_to_midi(r)
        tokens = []
        k = 0
        while k < 16:
            ch = padrao[k]
            dur = 1
            while k + dur < 16 and padrao[k + dur] == "-":
                dur += 1
            if ch == ".":
                tokens.append("r16")
            else:
                acc = "!" if ch in "XP" else ""
                som = quinta(m) if (ch in "Pp" and acorde) else nota(m)
                tokens.append(f"{acc}{som}{_dur(dur)}")
            k += dur
        out.append(" ".join(tokens) + " |")
    return "\n".join(out)


def baixo(raizes: list[str], padrao: str) -> str:
    """O baixo segue o MESMO ritmo da guitarra, uma oitava abaixo: e isso que
    faz baixo e guitarra soarem como um bloco so."""
    return chug([_nome(name_to_midi(r) - 12) for r in raizes], padrao,
                acorde=False)


def _nome(m: int) -> str:
    return f"{NOMES[m % 12].replace('+', '#').upper()}{m // 12 - 1}"


# ==========================================================================
# Partitura
# ==========================================================================
# padroes de 16 passos
GALOPE = "Xx.xX.xxP-.xP-xx"
EMPURRA = "Xx.xx.xxP-P-P-x."
CORRIDA = "XxxxXxxxXxxxP---"

# Riff (5-8): sobe cromatico Fa-Sol-Sol# ate o La — o "gancho" da guitarra
RIFF = """
o2 l16 !d d r d d r !d d  [f o3 c f]8 [f o3 c f]8 o2 [g o3 d g]8 [g+ o3 d+ g+]8 |
o2 !d d r d d r !d d  !a8 r8 [g o3 d g]8 [f o3 c f]8 |
o2 !d d r d d r !d d  [f o3 c f]8 [f o3 c f]8 o2 [g o3 d g]8 [g+ o3 d+ g+]8 |
o2 !d d r d d r !d d  [a+ o3 f a+]4 [a o3 e a]4 |
"""
RIFF_BAIXO = """
o1 l16 !d d r d d r !d d  f8 f8 g8 g+8 |
o1 !d d r d d r !d d  !a8 r8 g8 f8 |
o1 !d d r d d r !d d  f8 f8 g8 g+8 |
o1 !d d r d d r !d d  a+4 a4 |
"""

# Tema A (9-16): motivo de salto Re-La-Re, respondido em sequencia
TEMA_A = """
o4 l8 d a > !d4 c d e f    |  e4. d c+4 < a4         |
o4 b- > f b-4 a g f e      |  f4. e !d4 c4           |
o4 d a > !d4 c d e f       |  g4. f e f g a          |
o5 !b-4 a g f e c+ e       |  d2 r4 r8 < a8          |
"""
HARMONIA_A = ["D3", "A2", "A#2", "C3", "D3", "G2", "A2", "D3"]

# Tema B (17-24): notas longas, a melodia abre para cima
TEMA_B = """
o5 l4 f2. d             |  e2 g2              |
o5 a2. g8 f8            |  e f d2             |
o5 d f !b-2             |  a4. g8 e2          |
o5 c+ e a g             |  f e !c+ e          |
"""
HARMONIA_B = ["A#2", "C3", "D3", "D3", "A#2", "C3", "A2", "A2"]
CIFRA_B = "Bb | C | Dm | Dm | Bb | C | A | A"

# Ponte (25-32): meio-tempo, Gm -> Bb -> Eb (napolitano) -> A
PONTE = """
o5 l1 d | c2 < b-2 | > d | f2 d2 |
o5 e- | g2 f2 | e | c+2. r4 |
"""
CIFRA_PONTE = "Gm | Gm | Bb | Bb | Eb | Eb | A | A"
PONTE_RAIZ = ["G2", "G2", "A#2", "A#2", "D#2", "D#2", "A2", "A2"]

# --------------------------------------------------------------------------
# Bateria
# --------------------------------------------------------------------------
BAT_RIFF = """
k: x.xx..x.x.xx..x.|x.xx..x.x.x..xx.
s: ....X.......X...|....X.......X.oX
h: x.x.x.x.x.x.x.x.|x.x.x.x.x.x.x.x.
"""
BAT_RIFF_CRASH = BAT_RIFF.replace("h: x.x.x.x", "h: ..x.x.x") + \
    "c: X...............|................\n"
BAT_TEMA = """
k: x..x..x.x.x..x..|x..x..x.x.x..xx.
s: ....X..o....X...|....X..o....X.oX
r: x.x.x.x.x.x.x.x.|x.x.x.x.x.x.x.x.
"""
BAT_TEMA_CRASH = BAT_TEMA.replace("r: x.x.x.x", "r: ..x.x.x") + \
    "c: X...............|................\n"
BAT_MEIO = """
k: x.......x.x.....|x.......x..x....
s: ........X.......|........X......o
r: x.x.x.x.x.x.x.x.|x.x.x.x.x.x.x.x.
"""
BAT_CONSTROI = """
k: x...x...x...x...|x.x.x.x.x.x.x.x.
s: ....x.......x.x.|xxxxxxxxXXXXXXXX
c: X...............|................
"""
VIRADA = """
k: x.......x.......
s: X...o.xx........
t1: ........X.x.....
t3: ............X.X.
"""
INTRO_BAT = """
k: X..X..X.........|X..X..X.........|X..X..X.....X...
c: X...............|X...............|X...........X...
"""


def compor(e: Estudio) -> Estudio:
    e.novo("Duelo de Cristal", bpm=172, tom="D", escala="natural_minor",
           console="ps1")

    # ---- banda --------------------------------------------------------
    for lado, pan in (("E", -0.7), ("D", 0.7)):
        nome = f"Guitarra {lado}"
        e.criar_track(nome, "corda:Guitarra (chug pesado)", volume=0.62,
                      reverb_send=0.08)
        # passa-alta depois do amp: o grave e do baixo e do bumbo; guitarra
        # com sub embola a mixagem inteira
        e.ajustar_track(nome, efeitos=[{"tipo": "amp", "preset": "metal",
                                        "ganho": 13.0, "pan": pan},
                                       {"tipo": "eq", "modo": "highpass",
                                        "freq": 110, "q": 0.7},
                                       {"tipo": "eq", "freq": 2800,
                                        "ganho_db": 2.0, "q": 0.9}])
    e.criar_track("Baixo", "corda:Baixo (palheta)", volume=0.58, reverb_send=0.0)
    e.ajustar_track("Baixo", efeitos=[
        {"tipo": "eq", "modo": "highpass", "freq": 38, "q": 0.7},
        {"tipo": "eq", "freq": 900, "ganho_db": 3.0, "q": 1.0},
        {"tipo": "amp", "ganho": 2.2, "medio": 2.0, "cab": 6000,
         "presenca": 1.0, "nivel": 0.55, "pan": 0.0},
        {"tipo": "comp", "limiar_db": -18.0, "razao": 4.5, "attack_ms": 8.0,
         "release_ms": 110.0, "makeup_db": 3.5}])
    e.criar_track("Bateria", "ps1:bateria_power", volume=0.6, reverb_send=0.14)
    # o bumbo do kit Power vem com sub demais: medido trilha a trilha, a
    # bateria ficava 10 dB acima de tudo no grave e escondia a melodia
    e.ajustar_track("Bateria", efeitos=[
        {"tipo": "eq", "modo": "lowshelf", "freq": 90, "ganho_db": -7.0},
        {"tipo": "comp", "limiar_db": -15.0, "razao": 3.5, "attack_ms": 5.0,
         "release_ms": 120.0, "makeup_db": 2.5},
        {"tipo": "eq", "modo": "highshelf", "freq": 5000, "ganho_db": 3.5}])

    # ---- lead de serra: 2 vozes desafinadas + eco = parede de synth ---
    e.criar_track("Lead", "chip:Saw Hero", volume=0.95, reverb_send=0.3)
    e.ajustar_track("Lead", naipe=2, efeitos=[
        {"tipo": "eq", "modo": "highpass", "freq": 180},
        {"tipo": "eco", "tempo_s": 60 / 172 * 0.75, "feedback": 0.3,
         "mix": 0.22}])
    lead = e._track("Lead").instrument
    lead.vibrato_depth_cents, lead.vibrato_delay_s = 18.0, 0.22
    lead.lowpass_hz = 7500.0

    # ---- orquestra de PS1 -------------------------------------------
    e.criar_track("Metais", "ps1:brass_synth", volume=0.5, pan=-0.25,
                  reverb_send=0.25)
    e.criar_track("Cordas", "ps1:cordas", volume=0.42, pan=0.3,
                  reverb_send=0.45)
    e.criar_track("Coro", "ps1:coro", volume=0.5, pan=-0.1, reverb_send=0.55)
    e.criar_track("Hit", "ps1:orquestra_hit", volume=0.55, reverb_send=0.3)

    # ================================================================
    # Intro (1-4): golpes
    # ================================================================
    golpes = """
    o2 [d a o3 d]8. [d a o3 d]8. [d a o3 d]8 r2 |
    o2 [c g o3 c]8. [c g o3 c]8. [c+ g+ o3 c+]8 r2 |
    o2 [d a o3 d]8. [d a o3 d]8. [f o3 c f]8 r4 [g o3 d g]4 |
    o2 !g+16 g+16 a16 a16 a4 [a o3 e a]2 |
    """
    for lado in "ED":
        e.escrever(f"Guitarra {lado}", 1, golpes, velocity=108, humano=4)
    e.escrever("Baixo", 1, """
    o1 d8. d8. d8 r2 | c8. c8. c+8 r2 | d8. d8. f8 r4 g4 | g+16 g+16 a16 a16 a2. |
    """, velocity=110)
    e.escrever("Hit", 1, """
    o4 !d8. d8. d8 r2 | c8. c8. c+8 r2 | d8. d8. f8 r4 g4 | r1 |
    """, velocity=100)
    e.bateria("Bateria", 1, INTRO_BAT, humano=3)
    e.bateria("Bateria", 4, VIRADA, humano=3)

    # ================================================================
    # Riff (5-8) e de novo no fim (33-36)
    # ================================================================
    for ini in (5, 33):
        for k, lado in enumerate("ED"):
            # as duas guitarras tocam o mesmo riff com pequena diferenca de
            # tempo (seed diferente): e o que faz o dobro soar como duas
            e.escrever(f"Guitarra {lado}", ini, RIFF, velocity=104,
                       humano=5 + k, tendencia_ms=-2 + 4 * k)
        e.escrever("Baixo", ini, RIFF_BAIXO, velocity=108, humano=4,
                   tendencia_ms=2)
        e.bateria("Bateria", ini, BAT_RIFF_CRASH, humano=4, tendencia_ms=-3)
        e.bateria("Bateria", ini + 2, BAT_RIFF, humano=4, tendencia_ms=-3)
        e.escrever("Hit", ini, "o4 !d1 |", velocity=96)

    # ================================================================
    # Tema A (9-16)
    # ================================================================
    for k, lado in enumerate("ED"):
        e.escrever(f"Guitarra {lado}", 9,
                   chug(HARMONIA_A[:7], GALOPE) + "\n" +
                   chug(["D3"], CORRIDA), velocity=100, humano=5 + k,
                   tendencia_ms=-2 + 4 * k)
    e.escrever("Baixo", 9, baixo(HARMONIA_A[:7], GALOPE) + "\n" +
               baixo(["D3"], CORRIDA), velocity=104, humano=4, tendencia_ms=2)
    e.escrever("Lead", 9, TEMA_A, velocity=104, humano=6)
    e.bateria("Bateria", 9, BAT_TEMA_CRASH, humano=4, tendencia_ms=-3)
    e.bateria("Bateria", 11, BAT_TEMA, repetir=3, humano=4, tendencia_ms=-3)
    e.acompanhar("Cordas", 9, "Dm | A | Bb | C | Dm | Gm | A7 | Dm", "pad",
                  oitava=4, velocity=64, vozes=3, teto="A4")

    # ================================================================
    # Tema B (17-24)
    # ================================================================
    for k, lado in enumerate("ED"):
        e.escrever(f"Guitarra {lado}", 17, chug(HARMONIA_B, EMPURRA),
                   velocity=98, humano=5 + k, tendencia_ms=-2 + 4 * k)
    e.escrever("Baixo", 17, baixo(HARMONIA_B, EMPURRA), velocity=102,
               humano=4, tendencia_ms=2)
    e.escrever("Lead", 17, TEMA_B, velocity=108, humano=6)
    e.acompanhar("Metais", 17, CIFRA_B, "stabs", oitava=4, velocity=96,
                 vozes=3, teto="A4", humano=5)
    e.acompanhar("Cordas", 17, CIFRA_B, "pad", oitava=4, velocity=70,
                 vozes=2, teto="A4")
    e.bateria("Bateria", 17, BAT_TEMA_CRASH, humano=4, tendencia_ms=-3)
    e.bateria("Bateria", 19, BAT_TEMA, repetir=3, humano=4, tendencia_ms=-3)
    e.bateria("Bateria", 24, VIRADA, humano=3)

    # ================================================================
    # Ponte (25-32): meio-tempo, coro, napolitano
    # ================================================================
    for k, lado in enumerate("ED"):
        e.escrever(f"Guitarra {lado}", 25,
                   chug(PONTE_RAIZ[:6], "P---------------") + "\n" +
                   chug(PONTE_RAIZ[6:], "XxxxXxxxXxxxXxxx"),
                   velocity=96, humano=5 + k)
    e.escrever("Baixo", 25, baixo(PONTE_RAIZ[:6], "X---------------") + "\n" +
               baixo(PONTE_RAIZ[6:], "XxXxXxXxXxXxXxXx"), velocity=100, humano=4)
    e.escrever("Lead", 25, PONTE, velocity=100, humano=4)
    e.acompanhar("Coro", 25, CIFRA_PONTE, "pad", oitava=3, velocity=84,
                 vozes=3, teto="G4")
    # as cordas saem na reconstrucao (31-32): a caixa em semicolcheia e os
    # metais precisam das vozes do SPU
    e.acompanhar("Cordas", 25, "Gm | Gm | Bb | Bb | Eb | Eb", "arpejo8",
                 oitava=3, velocity=72, vozes=3, humano=5)
    e.acompanhar("Metais", 29, "Eb | Eb | A | A", "bloco", oitava=4,
                  velocity=88, vozes=3, teto="A4")
    e.bateria("Bateria", 25, BAT_MEIO, repetir=3, humano=4)
    e.bateria("Bateria", 31, BAT_CONSTROI, humano=3)
    e.escrever("Hit", 29, "o4 !e-1 | r1 | !a1 |", velocity=96)

    # ---- sala e master -----------------------------------------------
    e.sala("studio_medium", wet=0.5)
    e.master("rock")
    return e


if __name__ == "__main__":
    e = compor(Estudio())
    print(e.resumo())
    print()
    print(e.analisar(secoes=[["Intro", 1, 4], ["Riff", 5, 8], ["Tema A", 9, 16],
                             ["Tema B", 17, 24], ["Ponte", 25, 32]]))
    print()
    print(e.renderizar(str(AQUI / "duelo_de_cristal.wav")))
    print(e.salvar(str(AQUI / "duelo_de_cristal.chip")))
    if "--tocar" in sys.argv:
        import time
        print(e.tocar(1))
        time.sleep(e.p.ticks_to_seconds(e.p.length_ticks) + 3)
