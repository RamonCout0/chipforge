"""
"O Ultimo Farol" — tema principal, ~3:40, so instrumentos reais.

Musica original. Nenhuma track de chiptune: piano, violino solo, violao de
nylon, bateria, contrabaixo, naipe de cordas, cello, trompas, oboe e celesta,
todos do banco GeneralUser GS.

O ponto desta peca e responder a duas criticas de uma vez:

1. "soa sintetico" — o sampler agora le o ENVELOPE que vem dentro do .sf2.
   O piano decai como piano, o violino sustenta como violino. Antes tudo
   sustentava igual, e por isso tudo soava como teclado barato.

2. "ficou repetitivo" — o tema aparece SETE vezes e nunca do mesmo jeito:

       PIANO SO       tema A cru, piano sozinho, sem bateria
       VIOLINO        tema A no violino, piano acompanhando em arpejo
       VIOLAO         tema B, novo, no violao com cordas leves
       CONJUNTO       tema A ORNAMENTADO (notas de passagem) + banda
       DESENVOLVIMENTO tema A fragmentado e SEQUENCIADO subindo por graus,
                      com contracanto em movimento contrario
       CLIMAX         tema A AUMENTADO (notas em dobro) nas cordas e trompas,
                      com o tema B correndo por cima como contracanto
       EPILOGO        tema A de volta ao piano, resolvendo em Re maior

O arco tonal e o mesmo dos quatro acordes: Si menor e Re maior usam Bm-G-D-A
lidos de dois jeitos. A peca comeca ouvindo como menor e termina ouvindo como
maior, sem trocar um acorde sequer.

    python3 examples/o_ultimo_farol.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from chipforge.audio import engine
from chipforge.audio import fx
from chipforge.audio import sampler as SF
from chipforge.core import expressao as E
from chipforge.core import harmony as H
from chipforge.core import intensity as INT
from chipforge.core import theory as T
from chipforge.core import variacao as V
from chipforge.core.project import Note, Project
from chipforge.midiio import library as L
from chipforge.midiio import smf

BPM = 104
Q = 480
BAR = Q * 4
N = T.name_to_midi
TONICA, MODO = 11, "natural_minor"        # Si menor


def ac(raiz, qual="min", oit=4):
    r = N(f"{raiz}{oit}")
    return [r + i for i in T.CHORDS[qual]]


def prog(seq, vezes=1):
    return [ac(*c) for c in seq] * vezes


CICLO = [("B", "min"), ("G", "maj"), ("D", "maj"), ("A", "maj")]
CICLO_B = [("G", "maj"), ("D", "maj"), ("E", "min"), ("A", "maj")]

SECOES = [
    ("PIANO SO",       12, prog(CICLO, 3)),
    ("VIOLINO",        16, prog(CICLO, 4)),
    ("VIOLAO",         12, prog(CICLO_B, 3)),
    ("CONJUNTO",       16, prog(CICLO, 4)),
    ("DESENVOLVIMENTO", 16, prog([("B", "min"), ("F#", "min"), ("G", "maj"),
                                  ("D", "maj"), ("E", "min"), ("A", "maj"),
                                  ("G", "maj"), ("A", "maj")], 2)),
    ("CLIMAX",         16, prog(CICLO, 4)),
    ("EPILOGO",         8, prog([("G", "maj"), ("D", "maj"),
                                 ("A", "maj"), ("D", "maj")], 2)),
]
INICIO, _b = {}, 0
for _n, _k, _ in SECOES:
    INICIO[_n] = _b
    _b += _k
TOTAL = _b

# --------------------------------------------------------------------------
TEMA_A = [
    ("F#5", 0, 0.0, 1.5), ("A5", 0, 1.5, 0.5), ("B5", 0, 2.0, 1.0),
    ("F#5", 0, 3.0, 1.0),
    ("G5", 1, 0.0, 1.0), ("B5", 1, 1.0, 1.0), ("A5", 1, 2.0, 2.0),
    ("F#5", 2, 0.0, 1.0), ("A5", 2, 1.0, 1.0), ("D6", 2, 2.0, 1.5),
    ("C#6", 2, 3.5, 0.5),
    ("B5", 3, 0.0, 1.0), ("A5", 3, 1.0, 1.0), ("F#5", 3, 2.0, 2.0),
    ("F#5", 4, 0.0, 1.5), ("B5", 4, 1.5, 0.5), ("D6", 4, 2.0, 1.0),
    ("B5", 4, 3.0, 1.0),
    ("G5", 5, 0.0, 2.0), ("A5", 5, 2.0, 2.0),
    ("F#5", 6, 0.0, 0.5), ("A5", 6, 0.5, 0.5), ("D6", 6, 1.0, 1.0),
    ("E6", 6, 2.0, 1.0), ("D6", 6, 3.0, 1.0),
    ("C#6", 7, 0.0, 1.0), ("B5", 7, 1.0, 1.0), ("A5", 7, 2.0, 2.0),
]

TEMA_B = [
    ("B4", 0, 0.0, 1.0), ("D5", 0, 1.0, 1.0), ("G5", 0, 2.0, 2.0),
    ("A4", 1, 0.0, 1.0), ("D5", 1, 1.0, 1.0), ("F#5", 1, 2.0, 2.0),
    ("G4", 2, 0.0, 1.0), ("B4", 2, 1.0, 1.0), ("E5", 2, 2.0, 1.5),
    ("D5", 2, 3.5, 0.5),
    ("C#5", 3, 0.0, 1.0), ("E5", 3, 1.0, 1.0), ("A5", 3, 2.0, 2.0),
]


def escrever(spec, transpor=0, oitava=0, vel=96, corte=25, vezes=1, comp=8):
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
# FAIXAS DE REGISTRO — a decisao de orquestracao que mais importa
#
# Toda camada precisa da PROPRIA faixa. Quando pad, violao, guitarra e
# contracanto sao todos escritos entre Sol3 e La4, eles brigam de segunda o
# tempo inteiro e o meio da mixagem vira lama — mesmo com cada linha correta
# isolada. Aqui cada papel tem a sua altura, de baixo para cima.
# --------------------------------------------------------------------------
FAIXA = {
    "grave":        N("D2"),    # contrabaixo, o fundo de tudo
    "cello":        N("D3"),    # o cello senta UMA OITAVA acima do contrabaixo,
                                # como numa orquestra. Os dois no mesmo lugar
                                # nao somam peso, viram lama.
    "sustentacao":  N("D3"),    # pad de cordas, largo e embaixo
    "harmonia":     N("G3"),    # violao dedilhado, guitarra
    "contracanto":  N("D5"),    # segunda voz, acima da harmonia
    # melodia vive de F#5 para cima, escrita a mao nos temas
}


def na_oitava(ch, low):
    from chipforge.core.generators import voice_in_octave
    return voice_in_octave(triade(ch), low)


# --------------------------------------------------------------------------
# Camadas
# --------------------------------------------------------------------------
def piano_mao_esquerda(acordes, low=N("D2"), padrao="arpejo", vel=74):
    """Baixo do piano. 'arpejo' quebra o acorde (fundamental, quinta, decima),
    'blocos' segura, 'balanco' alterna fundamental e quinta."""
    from chipforge.core.generators import voice_in_octave
    out = []
    for i, ch in enumerate(acordes):
        raiz = voice_in_octave([min(ch)], low)[0]
        if padrao == "blocos":
            out.append(Note(i * BAR, BAR - 60, raiz, vel))
            out.append(Note(i * BAR, BAR - 60, raiz + 7, vel - 8))
        elif padrao == "balanco":
            for j, t in enumerate((0.0, 2.0)):
                out.append(Note(i * BAR + int(t * Q), Q * 2 - 40,
                                raiz + (0 if j == 0 else 7), vel))
        else:
            terca = sorted({p % 12 for p in ch})
            alvo = raiz + 12
            while alvo % 12 not in terca or alvo <= raiz + 9:
                alvo += 1
            for j, t in enumerate((0.0, 1.0, 2.0, 3.0)):
                nota = (raiz, raiz + 7, alvo, raiz + 7)[j]
                out.append(Note(i * BAR + int(t * Q), Q - 40, nota,
                                vel + (8 if j == 0 else -4)))
    return out


def piano_mao_direita(acordes, low=None, vel=64, rate=2):
    low = FAIXA["harmonia"] if low is None else low
    passo = Q // rate
    out = []
    for i, ch in enumerate(acordes):
        v = na_oitava(ch, low)
        for j in range(4 * rate):
            out.append(Note(i * BAR + j * passo, passo - 20,
                            v[j % len(v)], vel + (6 if j % rate == 0 else -3)))
    return out


def pad(acordes, low=None, vel=62, sem_terca=False):
    low = FAIXA["sustentacao"] if low is None else low
    from chipforge.core.generators import voice_in_octave
    out = []
    for i, ch in enumerate(acordes):
        if sem_terca:
            raiz = voice_in_octave([min(ch)], low)[0]
            notas = [raiz, raiz - 5]
        else:
            notas = na_oitava(ch, low)
        for n in notas:
            out.append(Note(i * BAR, BAR - 50, n, vel))
    return out


def baixo(acordes, low=N("D2"), celula=(0.0, 2.0), vel=96, oitavar=False):
    from chipforge.core.generators import voice_in_octave
    out = []
    for i, ch in enumerate(acordes):
        raiz = voice_in_octave([min(ch)], low)[0]
        for j, b in enumerate(celula):
            nota = raiz + (12 if oitavar and j % 2 else 0)
            out.append(Note(i * BAR + int(b * Q),
                            int(Q * (0.9 if len(celula) <= 2 else 0.45)),
                            nota, vel))
    return out


def violao_dedilhado(acordes, low=N("G3"), vel=80):
    """Dedilhado de violao: baixo no tempo, cordas agudas no contratempo."""
    out = []
    for i, ch in enumerate(acordes):
        v = na_oitava(ch, low)
        ordem = [0, 2, 1, 2, 0, 1, 2, 1]
        for j, idx in enumerate(ordem):
            out.append(Note(i * BAR + j * (Q // 2), Q - 40,
                            v[idx % len(v)] + (0 if j % 4 else -12),
                            vel + (10 if j % 4 == 0 else -6)))
    return out


def bateria(compassos, estilo="suave", virada_a_cada=8):
    """36 bumbo · 38 caixa · 42 chimbal · 46 aberto · 49 prato · 45/47 toms"""
    out = []
    for c in range(compassos):
        b = c * BAR
        if estilo == "suave":
            out.append(Note(b, 100, 36, 96))
            out.append(Note(b + Q * 2, 100, 38, 84))
            for j in range(8):
                out.append(Note(b + j * (Q // 2), 80, 42,
                                58 if j % 2 else 72))
        elif estilo == "cheio":
            for x in (0.0, 1.5, 2.0):
                out.append(Note(b + int(x * Q), 100, 36, 110))
            for x in (1.0, 3.0):
                out.append(Note(b + int(x * Q), 100, 38, 104))
            for j in range(8):
                out.append(Note(b + j * (Q // 2), 80, 42,
                                62 if j % 2 else 84))
            # fantasmas na caixa: o que faz a levada respirar
            for x in (0.75, 2.75):
                out.append(Note(b + int(x * Q), 70, 38, 42))
        elif estilo == "grande":
            for x in (0.0, 1.0, 2.0, 3.0):
                out.append(Note(b + int(x * Q), 100, 36, 114))
            for x in (1.0, 3.0):
                out.append(Note(b + int(x * Q), 110, 38, 112))
            for j in range(4):
                out.append(Note(b + j * Q, 120, 46, 76))
        if c == 0:
            out.append(Note(b, 240, 49, 108))
        if virada_a_cada and c % virada_a_cada == virada_a_cada - 1:
            for j, p in enumerate((47, 45, 43, 38)):
                out.append(Note(b + int((3.0 + j * 0.25) * Q), 90, p, 100 + j * 4))
    return out


# --------------------------------------------------------------------------
def build(sf2: Path) -> Project:
    p = Project("O Ultimo Farol", bpm=BPM, bit_depth=24)
    p.key_root, p.scale = TONICA, MODO
    p.master_gain = 0.95
    banco = SF.open_soundfont(sf2)

    def gm(nome, **kw):
        return SF.instrument(banco, nome, **kw)

    p.master_reverb = {"sala": 0.7, "damp": 0.4, "largura": 1.0,
                       "predelay_ms": 24.0}
    p.master_inserts = [
        {"tipo": "eq", "modo": "lowshelf", "freq": 120, "ganho_db": 1.5},
        {"tipo": "eq", "modo": "highshelf", "freq": 6000, "ganho_db": 1.0},
        {"tipo": "comp", "limiar_db": -16.0, "razao": 2.0,
         "attack_ms": 18.0, "release_ms": 240.0, "makeup_db": 1.5},
    ]

    t = {}
    t["piano"] = p.new_track("Piano", gm("Stereo Grand", gain=0.85))
    t["piano"].reverb_send, t["piano"].volume = 0.3, 0.7

    t["violino"] = p.new_track("Violino solo", gm("Violin", gain=0.85))
    t["violino"].reverb_send, t["violino"].volume, t["violino"].pan = 0.4, 0.62, -0.18

    t["violao"] = p.new_track("Violao", gm("Nylon Guitar", gain=0.85))
    t["violao"].reverb_send, t["violao"].volume, t["violao"].pan = 0.3, 0.6, 0.3

    t["eletrica"] = p.new_track("Guitarra", gm("Clean Guitar", gain=0.85))
    t["eletrica"].inserts = [dict(fx.PRESETS_AMP["crunch"], ganho=3.2, pan=-0.45),
                             {"tipo": "chorus", "profundidade_ms": 6.0,
                              "taxa": 0.35, "mix": 0.35}]
    t["eletrica"].reverb_send, t["eletrica"].volume = 0.3, 0.42

    t["cordas"] = p.new_track("Cordas", gm("Stereo Strings Slow", gain=0.5))
    t["cordas"].section = {"vozes": 4, "detune": 11, "spread_ms": 30, "pan": 0.8}
    t["cordas"].reverb_send, t["cordas"].volume = 0.6, 0.46

    t["cordas_mov"] = p.new_track("Cordas movimento",
                                  gm("Stereo Strings Fast", gain=0.5))
    t["cordas_mov"].section = {"vozes": 3, "detune": 8, "spread_ms": 20, "pan": 0.6}
    t["cordas_mov"].reverb_send, t["cordas_mov"].volume = 0.45, 0.42

    t["cello"] = p.new_track("Cello", gm("Cello", gain=0.85))
    t["cello"].section = {"vozes": 3, "detune": 7, "spread_ms": 16, "pan": 0.35}
    t["cello"].reverb_send, t["cello"].volume = 0.4, 0.55

    t["trompas"] = p.new_track("Trompas", gm("French Horns", gain=0.7))
    t["trompas"].section = {"vozes": 3, "detune": 8, "spread_ms": 16, "pan": 0.35}
    t["trompas"].reverb_send, t["trompas"].volume = 0.45, 0.5

    t["oboe"] = p.new_track("Oboe", gm("Oboe", gain=0.8))
    t["oboe"].reverb_send, t["oboe"].volume, t["oboe"].pan = 0.4, 0.42, 0.22

    t["celesta"] = p.new_track("Celesta", gm("Celeste", gain=0.7))
    t["celesta"].reverb_send, t["celesta"].volume, t["celesta"].pan = 0.5, 0.36, 0.42

    t["baixo"] = p.new_track("Contrabaixo", gm("Acoustic Bass", gain=0.95))
    t["baixo"].inserts = [{"tipo": "eq", "modo": "lowshelf", "freq": 110,
                           "ganho_db": 3.0},
                          {"tipo": "comp", "limiar_db": -20.0, "razao": 3.5,
                           "attack_ms": 8.0, "release_ms": 130.0, "makeup_db": 2.5}]
    t["baixo"].reverb_send, t["baixo"].volume = 0.15, 0.66

    t["bateria"] = p.new_track("Bateria", gm("Standard", gain=0.9))
    t["bateria"].instrument.is_drum = True
    t["bateria"].inserts = [{"tipo": "comp", "limiar_db": -18.0, "razao": 3.0,
                             "attack_ms": 6.0, "release_ms": 140.0,
                             "makeup_db": 2.0}]
    t["bateria"].reverb_send, t["bateria"].volume = 0.2, 0.58

    acordes = {n: c for n, _, c in SECOES}
    ncomp = {n: k for n, k, _ in SECOES}

    # cada familia chega ao tempo de um jeito: e esse atraso relativo que faz
    # soar gente tocando junto, e nao um sequenciador disparando tudo igual
    ATRASO = {"bateria": -6.0, "baixo": -3.0, "piano": 0.0, "violao": 2.0,
              "violino": 6.0, "cordas": 9.0, "cordas_mov": 7.0, "cello": 8.0,
              "trompas": 8.0, "oboe": 5.0, "celesta": 3.0, "eletrica": 1.0}

    def por(track, nome, secao, notas, humano=True, semente=3):
        if not notas:
            return
        if humano:
            notas = E.humanizar(notas, tempo_ms=11.0, velocity=8, ppq=Q,
                                bpm=BPM, seed=semente,
                                tendencia_ms=ATRASO.get(track, 0.0))
        pat = p.new_pattern(f"{nome} · {secao}", bars=ncomp[secao])
        pat.notes.extend(notas)
        p.place(pat, t[track], INICIO[secao])

    def frase(notas, curva="sino", forca=0.26):
        return E.arco(notas, curva, forca)

    # ==================================================== PIANO SO (1-12)
    s, ch = "PIANO SO", acordes["PIANO SO"]
    tema_piano = escrever(TEMA_A, oitava=-1, vel=84)
    tema_piano = E.articular(frase(tema_piano), "legato", folga=18)
    por("piano", "Tema A", s, tema_piano + piano_mao_esquerda(ch, vel=66))
    por("celesta", "Brilho", s,
        [n for n in escrever(TEMA_A, vel=46) if n.start % BAR == 0][:6])

    # ==================================================== VIOLINO (13-28)
    s, ch = "VIOLINO", acordes["VIOLINO"]
    tema_violino = E.respirar(E.articular(frase(escrever(TEMA_A, vel=92),
                                                "sino", 0.3), "legato"),
                              a_cada=BAR * 2, folga=110)
    por("violino", "Tema A", s, tema_violino)
    por("piano", "Arpejo", s,
        piano_mao_esquerda(ch, padrao="arpejo", vel=66)
        + piano_mao_direita(ch, vel=54, rate=2))
    por("cordas", "Pad", s, pad(ch[8:], vel=50))
    por("cello", "Cello", s, baixo(ch, low=FAIXA["cello"], celula=(0.0,), vel=76))

    # ==================================================== VIOLAO (29-40)
    s, ch = "VIOLAO", acordes["VIOLAO"]
    por("violao", "Dedilhado", s, violao_dedilhado(ch, vel=82))
    tema_b = E.articular(frase(escrever(TEMA_B, vel=88, vezes=3, comp=4),
                               "onda", 0.2), "legato")
    por("oboe", "Tema B", s, tema_b)
    por("cordas", "Pad", s, pad(ch, vel=48, sem_terca=True))
    por("baixo", "Baixo", s, baixo(ch, celula=(0.0, 2.0), vel=84))
    # sem contracanto aqui: entre o violao e o oboe nao sobra faixa livre,
    # e secao intimista pede MENOS vozes, nao mais
    por("piano", "Sustentado", s,
        piano_mao_esquerda(ch, padrao="blocos", vel=56))

    # ==================================================== CONJUNTO (41-56)
    s, ch = "CONJUNTO", acordes["CONJUNTO"]
    # segunda vez que o tema A aparece inteiro: agora ORNAMENTADO
    tema_orn = V.ornamentar(escrever(TEMA_A, vel=98, vezes=2), TONICA, MODO,
                            densidade=0.5, min_dur=Q // 2, seed=4)
    por("violino", "Tema A ornamentado", s, frase(tema_orn, "sino", 0.22))
    por("trompas", "Tema A", s, frase(escrever(TEMA_A, oitava=-1, vel=76,
                                               vezes=2), "sino", 0.2))
    por("violao", "Dedilhado", s, violao_dedilhado(ch, vel=74))
    por("cordas", "Pad", s, pad(ch, vel=54, sem_terca=True))
    por("cello", "Cello", s, baixo(ch, low=FAIXA["cello"], celula=(0.0, 2.0), vel=86))
    por("baixo", "Baixo", s, baixo(ch, celula=(0.0, 1.5, 2.0, 3.0), vel=92,
                                   oitavar=True))
    por("bateria", "Bateria", s, bateria(16, "suave"), semente=9)
    por("piano", "Blocos", s, piano_mao_esquerda(ch, padrao="balanco", vel=62))
    por("celesta", "Brilho", s,
        [n for n in escrever(TEMA_A, oitava=1, vel=40, vezes=2)
         if n.start % BAR == 0])

    # =============================================== DESENVOLVIMENTO (57-72)
    s, ch = "DESENVOLVIMENTO", acordes["DESENVOLVIMENTO"]
    # dois compassos do tema viram material: sequencia subindo por graus
    frag = V.fragmentar(escrever(TEMA_A, oitava=-1, vel=96), 0, 2,
                        ticks_por_compasso=BAR)
    seq = V.sequenciar(frag, [0, 3, 5, 7, 8, 10, 12, 14], BAR * 2,
                       escala=(TONICA, MODO), decaimento=0)
    seq = E.rampa_velocity(seq, 78, 112)
    por("violino", "Sequencia", s, seq)
    # oboe dobra o violino uma terca diatonica abaixo — naipe, nao briga
    por("oboe", "Tercas", s,
        frase(V.harmonizar(seq, TONICA, MODO, -2), "crescendo", 0.2))
    # tres vozes melodicas no mesmo palmo de registro nao e contraponto, e
    # colisao. O duo violino+oboe basta; o piano volta para a harmonia.
    por("piano", "Arpejo", s,
        piano_mao_esquerda(ch, padrao="arpejo", vel=70))
    por("cordas", "Pad", s, pad(ch, vel=56))
    por("cello", "Cello", s, E.rampa_velocity(
        baixo(ch, low=FAIXA["cello"], celula=(0.0, 1.0, 2.0, 3.0), vel=88),
        74, 104))
    por("baixo", "Baixo", s, baixo(ch, celula=(0.0, 1.5, 2.0, 3.0), vel=94,
                                   oitavar=True))
    por("bateria", "Bateria", s, bateria(16, "cheio"), semente=12)
    # a guitarra entra so aqui, e embaixo: fundamental sustentada, sem terca
    por("eletrica", "Sustentado", s,
        pad(ch, low=N("G3"), vel=66, sem_terca=True))

    # ==================================================== CLIMAX (73-88)
    s, ch = "CLIMAX", acordes["CLIMAX"]
    # tema A AUMENTADO: mesma melodia com o dobro da duracao, nas cordas
    tema_grande = V.aumentar(escrever(TEMA_A, oitava=-1, vel=104), 2.0)
    tema_grande = [n for n in tema_grande if n.start < 16 * BAR]
    por("cordas", "Tema A aumentado", s, frase(tema_grande, "sino", 0.18))
    por("trompas", "Tema A aumentado", s,
        frase(V.transpor(tema_grande, -12), "sino", 0.18))
    # e o tema B corre por cima como contracanto
    por("violino", "Tema B", s, frase(escrever(TEMA_B, oitava=1, vel=100,
                                               vezes=4, comp=4), "onda", 0.2))
    por("celesta", "Brilho", s, escrever(TEMA_B, oitava=1, vel=54,
                                         vezes=4, comp=4))
    por("cordas_mov", "Movimento", s,
        piano_mao_direita(ch, low=FAIXA["harmonia"], vel=64, rate=4))
    por("cello", "Cello", s, baixo(ch, low=FAIXA["cello"], celula=(0.0, 1.0, 2.0, 3.0), vel=98))
    por("baixo", "Baixo", s, baixo(ch, celula=(0.0, 1.5, 2.0, 3.0), vel=100,
                                   oitavar=True))
    por("bateria", "Bateria", s, bateria(16, "grande"), semente=15)
    por("piano", "Blocos", s, piano_mao_esquerda(ch, padrao="arpejo", vel=70))

    # =================================================== EPILOGO (89-96)
    s, ch = "EPILOGO", acordes["EPILOGO"]
    # a mesma frase do inicio, agora ouvida em Re maior — o farol acende
    fim = V.fragmentar(escrever(TEMA_A, oitava=-1, vel=76), 4, 4,
                       ticks_por_compasso=BAR)
    fim = E.articular(E.arco(V.aumentar(fim, 2.0), "diminuendo", 0.2), "legato")
    fim = [n for n in fim if n.start < 8 * BAR]
    por("piano", "Epilogo", s, fim + piano_mao_esquerda(ch, padrao="blocos",
                                                        vel=54))
    por("cordas", "Pad", s, pad(ch, vel=44, sem_terca=True))
    por("celesta", "Brilho", s,
        [n for n in escrever(TEMA_B, oitava=1, vel=44, vezes=2, comp=4)
         if n.start % BAR == 0])
    por("cello", "Cello", s, baixo(ch, low=FAIXA["cello"], celula=(0.0,), vel=64))
    return p


# --------------------------------------------------------------------------
def main():
    out = Path(__file__).parent
    sf2 = L.ensure("generaluser_gs")
    banco = SF.open_soundfont(sf2)
    print(f"Banco: {banco.name}\n")

    p = build(sf2)
    dur = p.ticks_to_seconds(p.length_ticks)
    print(f"O Ultimo Farol — {TOTAL} compassos, {p.bpm} BPM, "
          f"{int(dur//60)}:{int(dur%60):02d}")
    print(f"{len(p.tracks)} tracks · {len(p.patterns)} patterns · "
          f"{sum(len(x.notes) for x in p.patterns.values())} notas")
    print("Instrumentos:", ", ".join(tr.name for tr in p.tracks))
    print()

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
        print(f"  {nome:16s} nivel {c.niveis[sel].mean():.1f} "
              f"(pico {c.niveis[sel].max()})")

    print("\nRenderizando...")
    audio = engine.render_project(p)
    wav = engine.export_wav(audio, out / "o_ultimo_farol", p.sample_rate, 16)
    chip = p.save(out / "o_ultimo_farol")
    mid = smf.write_midi(smf.project_to_midi(p), out / "o_ultimo_farol")
    js = c.exportar_json(out / "o_ultimo_farol_intensidade", p.name)
    print(f"WAV     : {wav}  ({len(audio)/p.sample_rate:.1f}s, "
          f"pico {float(np.abs(audio).max()):.3f})")
    print(f"CHIP    : {chip}")
    print(f"MIDI    : {mid}")
    print(f"SPRITES : {js}")


if __name__ == "__main__":
    main()
