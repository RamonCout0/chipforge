"""
"Maré de Ferro" — o confronto da Guardiã dos Mares, versão rock. ~6 min.

Esta peça existe para consertar três coisas que a primeira versão errou:

1. **A guitarra era um sample.** Agora é modelada corda por corda
   (`audio/corda.py`): guia de onda com ataque de palheta, posição da palheta,
   palm mute de verdade e afinação compensada. O que entra no amplificador é
   uma guitarra limpa — não um sample já distorcido sendo distorcido de novo.

2. **A guitarra tocava arpejo.** Agora toca riff (`core/banda.py`): acorde de
   potência numa grade de semicolcheias, com a nota parada antes da próxima
   palhetada. É esse silêncio entre as batidas que produz o chug.

3. **Não havia bateria.** Agora há kit completo — bumbo, caixa com notas
   fantasma, chimbal, condução, tons, prato e virada a cada quatro compassos.

E a paleta triplicou: sete guitarras, dois baixos, kit + taikos, órgão,
naipes de sopro e metal, sinos, piano, harpa, coro SATB e a sereia.

FORMA — 202 compassos, Mi menor, modulando para Fá# menor no segundo refrão:

    INTRO       1-8    limpa e voz, sem bateria
    RIFF 1      9-24   a banda entra
    VERSO      25-40   chug de semicolcheia, a sereia canta
    PRÉ        41-48   abre, condução no prato
    REFRÃO     49-64   tema + coro + acordes abertos
    RIFF 2     65-80   o trítono (Si bemol) — a cor do metal
    VERSO 2    81-96   galope duplo
    PONTE      97-108  breakdown: metade do tempo, o peso todo
    SOLO      109-136  guitarra solando sobre harmonia nova
    QUEBRA    137-144  tudo para; só a sereia
    PRÉ 2     145-152  empurra para Fá# menor
    REFRÃO 2  153-168  um tom acima
    FINAL     169-192  galope, tutti
    CODA      193-202  desce de volta a Mi menor, para o loop

    python examples/mare_de_ferro.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from chipforge.audio import cinema as CIN
from chipforge.audio import corda as CD
from chipforge.audio import engine
from chipforge.audio import fx
from chipforge.audio import jogo
from chipforge.audio import master as MST
from chipforge.audio import sampler as SF
from chipforge.audio import voz as VOZ
from chipforge.core import banda as BD
from chipforge.core import dinamica as DIN
from chipforge.core import expressao as EXP
from chipforge.core import harmony as H
from chipforge.core import intensity as INT
from chipforge.core import orquestra as ORQ
from chipforge.core import theory as T
from chipforge.core.project import Note, Project
from chipforge.midiio import library as L
from chipforge.midiio import smf

Q = 480
COMPASSO = Q * 4
N = T.name_to_midi
BPM = 140


def ac(raiz: str, qual: str = "min", oit: int = 3) -> list[int]:
    return [N(f"{raiz}{oit}") + i for i in T.CHORDS[qual]]


def prog(seq, vezes: int = 1) -> list[list[int]]:
    return [ac(*c) for c in seq] * vezes


Em, C, G, D_, Am, Bm, B_ = ("E", "min"), ("C", "maj"), ("G", "maj"), \
    ("D", "maj"), ("A", "min"), ("B", "min"), ("B", "maj")
Sib5 = ("A#", "5")                       # o trítono de Mi — a cor do metal
Fsm, A_, Csm, Dm_ = ("F#", "min"), ("A", "maj"), ("C#", "maj"), ("D", "maj")

SECOES = [
    ("INTRO", 8, prog([Em, Em, C, C, Am, Am, Bm, Bm])),
    ("RIFF1", 16, prog([Em, Em, C, D_, Em, Em, G, D_], 2)),
    ("VERSO", 16, prog([Em, G, D_, Am, Em, G, C, B_], 2)),
    ("PRE", 8, prog([C, D_, Em, Em, C, D_, B_, B_])),
    ("REFRAO", 16, prog([Em, C, G, D_, Am, C, Em, D_], 2)),
    ("RIFF2", 16, prog([Em, Em, Sib5, Sib5, C, C, D_, D_], 2)),
    ("VERSO2", 16, prog([Em, G, D_, Am, Em, G, C, B_], 2)),
    ("PONTE", 12, prog([Em, Em, C, C, Am, Am, B_, B_, Em, Em, D_, D_])),
    ("SOLO", 28, prog([Am, C, G, D_, Am, C, D_, B_], 3)
     + prog([Am, C, G, D_])),
    ("QUEBRA", 8, prog([Em, Em, Am, Am, C, C, B_, B_])),
    ("PRE2", 8, prog([C, D_, Csm, Csm, D_, ("E", "maj"), Csm, Csm])),
    ("REFRAO2", 16, prog([Fsm, Dm_, A_, ("E", "maj"),
                          Bm, Dm_, Fsm, ("E", "maj")], 2)),
    ("FINAL", 24, prog([Fsm, Dm_, ("E", "maj"), Fsm,
                        Dm_, A_, Bm, Csm], 3)),
    ("CODA", 10, prog([Fsm, ("E", "maj"), Dm_, C, Bm, Am, Em, Em, Em, Em])),
]

INICIO: dict[str, int] = {}
_b = 0
for _n, _k, _ in SECOES:
    INICIO[_n] = _b
    _b += _k
TOTAL = _b

NIVEIS = {
    "INTRO": 0.30, "RIFF1": 0.88, "VERSO": 0.72, "PRE": 0.80, "REFRAO": 1.00,
    "RIFF2": 0.92, "VERSO2": 0.76, "PONTE": 0.86, "SOLO": 0.90,
    "QUEBRA": 0.28, "PRE2": 0.78, "REFRAO2": 1.00, "FINAL": 1.00, "CODA": 0.52,
}

# --------------------------------------------------------------------------
# Temas
# --------------------------------------------------------------------------
TEMA_SEREIA = [
    ("B4", 0, 0.0, 2.0), ("A4", 0, 2.0, 1.0), ("G4", 0, 3.0, 1.0),
    ("A4", 1, 0.0, 3.0),
    ("E5", 2, 0.0, 1.5), ("D5", 2, 1.5, 0.5), ("B4", 2, 2.0, 2.0),
    ("A4", 3, 0.0, 2.0), ("F#4", 3, 2.0, 2.0),
    ("G4", 4, 0.0, 1.0), ("A4", 4, 1.0, 1.0), ("B4", 4, 2.0, 2.0),
    ("D5", 5, 0.0, 1.5), ("E5", 5, 1.5, 0.5), ("D5", 5, 2.0, 2.0),
    ("B4", 6, 0.0, 1.0), ("A4", 6, 1.0, 1.0), ("G4", 6, 2.0, 1.0),
    ("F#4", 6, 3.0, 1.0),
    ("E4", 7, 0.0, 3.5),
]

# O refrão: notas longas e altas. Um refrão não é o verso mais alto — é uma
# melodia com menos notas e mais fôlego, para caber o coro por baixo.
REFRAO_MEL = [
    ("E5", 0, 0.0, 3.0), ("D5", 0, 3.0, 1.0),
    ("B4", 1, 0.0, 2.0), ("D5", 1, 2.0, 2.0),
    ("G5", 2, 0.0, 3.0), ("F#5", 2, 3.0, 1.0),
    ("E5", 3, 0.0, 4.0),
    ("D5", 4, 0.0, 2.0), ("E5", 4, 2.0, 2.0),
    ("G5", 5, 0.0, 2.0), ("E5", 5, 2.0, 2.0),
    ("B4", 6, 0.0, 1.0), ("D5", 6, 1.0, 1.0), ("E5", 6, 2.0, 2.0),
    ("B4", 7, 0.0, 3.5),
]

MOTIVO_LENTO = [
    ("E2", 0, 0.0, 4.0), ("D2", 1, 0.0, 4.0),
    ("C2", 2, 0.0, 4.0), ("B1", 3, 0.0, 4.0),
]

CONTRA = [
    ("E4", 0, 0.0, 1.0), ("F#4", 0, 1.0, 1.0), ("G4", 0, 2.0, 2.0),
    ("B4", 1, 0.0, 2.0), ("A4", 1, 2.0, 2.0),
    ("G4", 2, 0.0, 1.0), ("B4", 2, 1.0, 1.0), ("D5", 2, 2.0, 2.0),
    ("B4", 3, 0.0, 4.0),
]


def mel(linha, **kw) -> list[Note]:
    return ORQ.escrever(linha, Q, 4, **kw)


# --------------------------------------------------------------------------
def build(sf2: Path) -> Project:
    sf = SF.open_soundfont(sf2)

    def gm(nome: str, **kw):
        return SF.instrument(sf, nome, **kw)

    p = Project(name="Mare de Ferro", bpm=BPM, ppq=Q, bit_depth=24,
                master_gain=0.9, key_root=N("E4") % 12, scale="natural_minor")

    # Sala de estúdio, não catedral. Rock precisa que o transiente da caixa e
    # o corte do chug cheguem secos; sala longa em cima de guitarra distorcida
    # é o jeito mais rápido de perder o peso que a distorção acabou de dar.
    p.master_reverb = {"sala": "camara", "tamanho": 0.9, "wet": 0.55,
                       "corte_grave": 220.0, "corte_agudo": 8000.0}
    p.master_perfil = "rock"

    mapa = DIN.MapaDeTempo(ppq=Q, bpm_base=BPM)
    mapa.marcar(0, BPM * 0.90)
    mapa.rampa(INICIO["INTRO"] * COMPASSO, INICIO["RIFF1"] * COMPASSO,
               BPM * 0.90, BPM)
    mapa.marcar(INICIO["SOLO"] * COMPASSO, BPM)
    mapa.rampa(INICIO["PONTE"] * COMPASSO, (INICIO["PONTE"] + 4) * COMPASSO,
               BPM, BPM * 0.94)
    mapa.rampa((INICIO["PONTE"] + 8) * COMPASSO, INICIO["SOLO"] * COMPASSO,
               BPM * 0.94, BPM)
    mapa.rampa(INICIO["PRE2"] * COMPASSO, INICIO["REFRAO2"] * COMPASSO,
               BPM, BPM * 1.04)
    mapa.marcar(INICIO["FINAL"] * COMPASSO, BPM * 1.05)
    mapa.ritardando(TOTAL * COMPASSO, 4, COMPASSO, fator=0.74)
    p.tempo_map = mapa.to_list()

    t: dict[str, object] = {}

    # ==================================================== GUITARRAS
    # Base: duas guitarras de chug pesado, uma em cada lado. É a gravação dupla
    # de sempre — mas agora as duas são MODELADAS, então cada uma tem palhetada
    # própria e elas não são a mesma onda deslocada.
    for lado, pan, sem in (("esq", -0.92, 11), ("dir", 0.92, 29)):
        tr = p.new_track(f"Guitarra ritmo {lado}",
                         CD.corda("Guitarra (chug pesado)", seed=sem))
        tr.inserts = [dict(fx.PRESETS_AMP["metal"], ganho=13.0, pan=pan)]
        tr.reverb_send = 0.10
        tr.volume = 0.62
        t[f"gtr_{lado}"] = tr

    # Segundo par, de corda solta, tocando acordes sustentados por baixo do
    # chug. É esse par que dá altura ao acorde; o chug sozinho é só ritmo.
    for lado, pan, sem in (("esq", -0.55, 43), ("dir", 0.55, 61)):
        tr = p.new_track(f"Guitarra aberta {lado}",
                         CD.corda("Guitarra (aberta)", seed=sem))
        tr.inserts = [dict(fx.PRESETS_AMP["rock"], ganho=8.0, pan=pan)]
        tr.reverb_send = 0.18
        tr.volume = 0.44
        t[f"gtra_{lado}"] = tr

    t["gtr_lead"] = p.new_track("Guitarra lead", CD.corda("Guitarra (lead)", seed=7))
    t["gtr_lead"].inserts = [dict(fx.PRESETS_AMP["crunch"], ganho=9.0, pan=0.10),
                             {"tipo": "eq", "freq": 1600, "ganho_db": 3.0, "q": 0.9},
                             {"tipo": "eco", "tempo_s": 0.26, "feedback": 0.28,
                              "mix": 0.18}]
    t["gtr_lead"].reverb_send = 0.28
    t["gtr_lead"].volume = 0.52

    t["gtr_limpa"] = p.new_track("Guitarra limpa", CD.corda("Guitarra limpa", seed=17))
    t["gtr_limpa"].inserts = [dict(fx.PRESETS_AMP["limpo"], pan=-0.25),
                              {"tipo": "chorus", "profundidade_ms": 7.0,
                               "taxa": 0.35, "mix": 0.45}]
    t["gtr_limpa"].reverb_send = 0.40
    t["gtr_limpa"].volume = 0.48

    t["gtr_harm"] = p.new_track("Guitarra harmônicos",
                                CD.corda("Guitarra (harmonico)", seed=23))
    t["gtr_harm"].inserts = [dict(fx.PRESETS_AMP["metal"], ganho=10.0, pan=0.35)]
    t["gtr_harm"].reverb_send = 0.35
    t["gtr_harm"].volume = 0.30

    # ==================================================== BAIXO
    t["baixo"] = p.new_track("Baixo", CD.corda("Baixo (palheta)", seed=5))
    t["baixo"].inserts = [
        {"tipo": "eq", "modo": "lowshelf", "freq": 100, "ganho_db": 3.0},
        {"tipo": "eq", "freq": 800, "ganho_db": 3.5, "q": 1.0},
        {"tipo": "amp", "ganho": 2.2, "medio": 2.0, "cab": 6000,
         "presenca": 1.0, "nivel": 0.55, "pan": 0.0},
        {"tipo": "comp", "limiar_db": -18.0, "razao": 4.5, "attack_ms": 8.0,
         "release_ms": 110.0, "makeup_db": 3.5}]
    t["baixo"].volume = 0.66

    t["sub"] = p.new_track("Sub", gm("Synth Bass 1", gain=0.8))
    t["sub"].inserts = [{"tipo": "eq", "modo": "lowpass", "freq": 140, "q": 0.7}]
    t["sub"].volume = 0.34

    # ==================================================== BATERIA
    t["bateria"] = p.new_track("Bateria", gm("Power", gain=0.9))
    t["bateria"].instrument.is_drum = True
    t["bateria"].inserts = [
        {"tipo": "comp", "limiar_db": -15.0, "razao": 3.5, "attack_ms": 5.0,
         "release_ms": 120.0, "makeup_db": 2.5},
        {"tipo": "eq", "freq": 4200, "ganho_db": 2.5, "q": 0.8}]
    t["bateria"].reverb_send = 0.16
    t["bateria"].volume = 0.72

    t["taiko"] = p.new_track("Taiko", CIN.instrumento("Taiko"))
    t["taiko"].reverb_send = 0.20
    t["taiko"].volume = 0.52

    t["taiko_alto"] = p.new_track("Taiko pequeno",
                                  CIN.instrumento("Taiko pequeno", pan=-0.4))
    t["taiko_alto"].reverb_send = 0.26
    t["taiko_alto"].volume = 0.34

    # ==================================================== VOZ
    t["voz"] = p.new_track("Sereia", VOZ.voz("Solista feminina", pan=-0.05))
    t["voz"].volume = 0.86
    t["voz"].reverb_send = 0.34
    t["voz"].inserts = [
        {"tipo": "eq", "modo": "highpass", "freq": 140.0, "q": 0.7},
        {"tipo": "eq", "freq": 2600.0, "ganho_db": 3.5, "q": 0.9},
        {"tipo": "comp", "limiar_db": -19.0, "razao": 3.5, "attack_ms": 10.0,
         "release_ms": 150.0, "makeup_db": 4.0}]

    t["voz_fundo"] = p.new_track("Sereia (fundo)",
                                 VOZ.voz("Lamento (u)", pan=0.35, gain=0.30))
    t["voz_fundo"].volume = 0.30
    t["voz_fundo"].reverb_send = 0.80

    for naipe, preset, pan in (
            ("soprano", "Coro soprano (aah)", -0.48),
            ("contralto", "Coro contralto (ooh)", 0.42),
            ("tenor", "Coro tenor (aah)", -0.32),
            ("baixo", "Coro baixo (ooh)", 0.30)):
        tr = p.new_track(f"Coro {naipe}",
                         VOZ.voz(preset, pan=pan, seed=abs(hash(naipe)) % 999))
        tr.section = dict(VOZ.SECAO_CORO_INTIMO)
        tr.reverb_send = 0.55
        tr.volume = 0.44
        t[f"coro_{naipe}"] = tr

    t["hum"] = p.new_track("Coro (boca fechada)",
                           VOZ.voz("Coro de boca fechada"))
    t["hum"].section = dict(VOZ.SECAO_CORO)
    t["hum"].reverb_send = 0.7
    t["hum"].volume = 0.36

    # ==================================================== ORQUESTRA
    t["cordas_ost"] = p.new_track("Cordas (ostinato)",
                                  gm("Stereo Strings Fast", gain=0.85))
    t["cordas_ost"].section = {"vozes": 3, "detune": 8, "spread_ms": 15, "pan": 0.55}
    t["cordas_ost"].reverb_send = 0.30
    t["cordas_ost"].volume = 0.42

    t["cordas_pad"] = p.new_track("Cordas (pad)", gm("Stereo Strings Slow", gain=0.85))
    t["cordas_pad"].section = {"vozes": 4, "detune": 10, "spread_ms": 26, "pan": 0.7}
    t["cordas_pad"].reverb_send = 0.48
    t["cordas_pad"].volume = 0.40

    t["tremolo"] = p.new_track("Cordas (tremolo)", gm("Stereo Strings Trem", gain=0.85))
    t["tremolo"].section = {"vozes": 2, "detune": 9, "spread_ms": 18, "pan": 0.6}
    t["tremolo"].reverb_send = 0.45
    t["tremolo"].volume = 0.30

    t["celos"] = p.new_track("Violoncelos", gm("Cello", gain=0.9))
    t["celos"].section = {"vozes": 3, "detune": 7, "spread_ms": 20, "pan": 0.4}
    t["celos"].reverb_send = 0.38
    t["celos"].volume = 0.46

    t["trompas"] = p.new_track("Trompas", gm("French Horns", gain=0.9))
    t["trompas"].section = {"vozes": 3, "detune": 8, "spread_ms": 20, "pan": 0.35}
    t["trompas"].reverb_send = 0.42
    t["trompas"].volume = 0.50

    t["metais"] = p.new_track("Metais", gm("Brass Section", gain=0.9))
    t["metais"].section = {"vozes": 2, "detune": 6, "spread_ms": 14, "pan": 0.45}
    t["metais"].reverb_send = 0.32
    t["metais"].volume = 0.46

    t["trombones"] = p.new_track("Trombones", gm("Trombone", gain=0.9))
    t["trombones"].section = {"vozes": 2, "detune": 7, "spread_ms": 16, "pan": 0.3}
    t["trombones"].reverb_send = 0.30
    t["trombones"].volume = 0.48

    t["orgao"] = p.new_track("Órgão", gm("Rock Organ", gain=0.8))
    t["orgao"].reverb_send = 0.30
    t["orgao"].volume = 0.32
    t["orgao"].pan = 0.4

    t["timpani"] = p.new_track("Timpani", gm("Timpani", gain=0.9))
    t["timpani"].reverb_send = 0.35
    t["timpani"].volume = 0.52

    t["sinos"] = p.new_track("Sinos", gm("Tubular Bells", gain=0.9))
    t["sinos"].reverb_send = 0.55
    t["sinos"].volume = 0.42
    t["sinos"].pan = -0.3
    t["sinos"].inserts = [{"tipo": "eq", "modo": "highpass", "freq": 180.0, "q": 0.7}]

    t["glock"] = p.new_track("Glockenspiel", gm("Glockenspiel", gain=0.8))
    t["glock"].reverb_send = 0.55
    t["glock"].volume = 0.26
    t["glock"].pan = 0.45

    t["piano"] = p.new_track("Piano", gm("Stereo Grand", gain=0.85))
    t["piano"].reverb_send = 0.34
    t["piano"].volume = 0.40
    t["piano"].pan = -0.18

    t["harpa"] = p.new_track("Harpa", gm("Orchestral Harp", gain=0.9))
    t["harpa"].reverb_send = 0.48
    t["harpa"].volume = 0.34
    t["harpa"].pan = 0.30

    t["flauta"] = p.new_track("Flauta", gm("Flute", gain=0.85))
    t["flauta"].reverb_send = 0.50
    t["flauta"].volume = 0.32
    t["flauta"].pan = 0.22

    t["oboe"] = p.new_track("Oboé", gm("Oboe", gain=0.85))
    t["oboe"].reverb_send = 0.45
    t["oboe"].volume = 0.30
    t["oboe"].pan = -0.3

    t["violao"] = p.new_track("Violão", CD.corda("Violao (aco)", seed=31))
    t["violao"].reverb_send = 0.42
    t["violao"].volume = 0.36
    t["violao"].pan = 0.35

    # ==================================================== DESENHO DE SOM
    for chave, preset, vol in (("impacto", "Impacto grave", 0.66),
                               ("riser", "Riser longo", 0.32),
                               ("swell", "Swell invertido", 0.28),
                               ("braam", "Braam", 0.40),
                               ("subdrop", "Sub drop", 0.50)):
        tr = p.new_track(preset, CIN.instrumento(preset))
        tr.reverb_send = 0.34
        tr.volume = vol
        t[chave] = tr

    # ---- arco -----------------------------------------------------------
    arco = DIN.arco_de_secoes([(n, k) for n, k, _ in SECOES], COMPASSO, NIVEIS)
    sem_arco = ("impacto", "subdrop", "braam", "riser", "swell")
    for chave, tr in t.items():
        if chave not in sem_arco:
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

    def humano(notas, espalhar=18.0, vel=8):
        return ORQ.humanizar_naipe(notas, Q, espalhar, BPM, vel)

    def parede(secao, padrao, vel=104, abertas=True, harm=False):
        """As guitarras de ritmo, o baixo e o sub tocando a MESMA grade.

        Tocar a mesma figura rítmica é o que faz guitarra e baixo soarem como
        uma coisa só. Se cada um tem o próprio ritmo, o ouvido separa os dois
        e o peso se divide em dois pesos menores.
        """
        ch = acordes[secao]
        for lado, sem in (("esq", 101), ("dir", 202)):
            por(f"gtr_{lado}", "Riff", secao,
                BD.riff(ch, Q, padrao, 1, vel, True, 40, 52, 0.52, True, sem))
        if abertas:
            for lado, sem in (("esq", 303), ("dir", 404)):
                por(f"gtra_{lado}", "Aberta", secao,
                    BD.riff(ch, Q, "meio", 1, int(vel * 0.8), True, 45, 59,
                            1.0, False, sem))
        por("baixo", "Baixo", secao,
            BD.linha_de_baixo(ch, Q, padrao, 1, vel + 4, 28, 45))
        por("sub", "Sub", secao,
            BD.linha_de_baixo(ch, Q, "meio", 1, 90, 28, 40))
        if harm:
            por("gtr_harm", "Harmônicos", secao,
                BD.riff(ch, Q, "quebrado", 1, 92, False, 45, 59, 0.8, False, 55))

    # ============================================================ INTRO
    s, ch = "INTRO", acordes["INTRO"]
    por("gtr_limpa", "Arpejo", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=2, low=52, high=76, velocity=62), 24, 10))
    por("violao", "Arpejo", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=4, low=45, high=69, velocity=54,
                     contorno="sobe"), 22, 10))
    por("voz", "Tema", s, EXP.arco(mel(TEMA_SEREIA, velocity=72, compassos=8)[:12],
                                   "sino", 0.24))
    por("cordas_pad", "Pad", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=55, high=76, velocity=44), 30))
    por("piano", "Onda", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=2, low=48, high=72, velocity=48), 30, 11))
    por("celos", "Pedal", s, humano(ORQ.pedal(ch, Q, 1, 4, "violoncelo", 52), 30))

    # ============================================================ RIFF 1
    s, ch = "RIFF1", acordes["RIFF1"]
    parede(s, "chug8", 106, True, True)
    por("bateria", "Kit", s, BD.bateria(16, Q, "rock", 110, 4, "curta"))
    por("taiko", "Marcha", s, ORQ.taiko(16, Q, "marcha", 4, 36, 92))
    por("cordas_ost", "Ostinato", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=4, low=64, high=86, velocity=70), 12, 8))
    por("trompas", "Bloco", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=48, high=69, velocity=80), 20))
    por("orgao", "Sustentado", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=52, high=74, velocity=60), 22))
    por("timpani", "Marca", s, ORQ.taiko(16, Q, "meio", 4,
                                         ORQ.encaixar(N("E2"), "timpani"), 92))

    # ============================================================ VERSO
    s, ch = "VERSO", acordes["VERSO"]
    parede(s, "chug16", 96, False)
    por("bateria", "Kit", s, BD.bateria(16, Q, "rock", 102, 4, "curta"))
    por("voz", "Tema", s, EXP.arco(mel(TEMA_SEREIA, velocity=100, vezes=2),
                                   "sino", 0.18))
    # Uníssono, não oitava abaixo. Dobrar a melodia uma oitava abaixo joga a
    # cópia no registro onde metais, trompas e o coro estão tocando — ela deixa
    # de ser dobra e vira uma voz a mais brigando com o acompanhamento (eram
    # 56 choques só neste par). Em uníssono ela engrossa sem ocupar lugar novo.
    por("voz_fundo", "Tema", s, mel(TEMA_SEREIA, velocity=60, vezes=2))
    por("cordas_ost", "Ostinato", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=4, low=64, high=86, velocity=64), 12, 8))
    por("hum", "Hum", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=4, low=45, high=64, velocity=58), 38, 10))

    # ============================================================ PRE
    s, ch = "PRE", acordes["PRE"]
    parede(s, "colcheias", 102, True)
    por("bateria", "Kit", s, BD.bateria(8, Q, "conducao", 106, 4, "longa"))
    por("cordas_ost", "Ostinato", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=4, low=64, high=88, velocity=78), 12, 8))
    por("metais", "Bloco", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=52, high=76, velocity=90), 18))
    por("voz", "Sobe", s, EXP.arco(mel(CONTRA, velocity=98, vezes=2),
                                   "crescendo", 0.20))
    por("tremolo", "Tensao", s, humano(
        ORQ.tremolo(ch, Q, 1, 4, low=64, high=84, velocity=44, divisao=8), 10, 6))

    # ============================================================ REFRÃO
    s, ch = "REFRAO", acordes["REFRAO"]
    parede(s, "colcheias", 108, True, True)
    por("bateria", "Kit", s, BD.bateria(16, Q, "rock_aberto", 112, 4, "curta"))
    por("voz", "Refrão", s, EXP.arco(mel(REFRAO_MEL, velocity=112, vezes=2),
                                     "crescendo", 0.14))
    por("voz_fundo", "Refrão", s, mel(REFRAO_MEL, velocity=66, vezes=2))
    for naipe, low, high, vel in (("soprano", 67, 81, 92), ("contralto", 58, 72, 88),
                                  ("tenor", 50, 67, 88), ("baixo", 43, 60, 92)):
        por(f"coro_{naipe}", "Refrão", s, humano(
            ORQ.bloco(ch, Q, 1, 4, n_vozes=1, low=low, high=high, velocity=vel), 32, 9))
    por("cordas_ost", "Ostinato", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=4, low=64, high=88, velocity=82), 12, 8))
    por("metais", "Bloco", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=52, high=76, velocity=96), 18))
    por("trompas", "Bloco", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=45, high=67, velocity=94), 20))
    por("sinos", "Badalo", s, humano(
        ORQ.pedal(ch, Q, 1, 4, "harpa", 92, oitava_dobrada=False), 24, 9))
    por("glock", "Brilho", s, humano(
        mel(REFRAO_MEL, velocity=58, oitava=1, vezes=2), 22, 10))
    por("taiko", "Marcha", s, ORQ.taiko(16, Q, "marcha", 4, 36, 100))

    # ============================================================ RIFF 2
    s, ch = "RIFF2", acordes["RIFF2"]
    parede(s, "galope", 108, False, True)
    por("bateria", "Kit", s, BD.bateria(16, Q, "galope", 112, 4, "curta"))
    por("gtr_lead", "Contra", s, EXP.arco(mel(CONTRA, velocity=104, vezes=4),
                                          "crescendo", 0.16))
    por("trombones", "Leviata", s, humano(
        mel(MOTIVO_LENTO, velocity=100, vezes=4, compassos=4), 20))
    por("cordas_ost", "Ostinato", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=4, low=64, high=86, velocity=80), 12, 8))
    por("taiko", "Galope", s, ORQ.taiko(16, Q, "galope", 4, 36, 104))
    por("taiko_alto", "Marcha", s, ORQ.taiko(16, Q, "marcha", 4, 36, 86))

    # ============================================================ VERSO 2
    s, ch = "VERSO2", acordes["VERSO2"]
    parede(s, "galope_duplo", 100, False)
    por("bateria", "Kit", s, BD.bateria(16, Q, "duplo", 106, 4, "curta"))
    por("voz", "Tema", s, EXP.arco(mel(TEMA_SEREIA, velocity=104, vezes=2),
                                   "sino", 0.16))
    por("cordas_ost", "Ostinato", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=4, low=64, high=86, velocity=72), 12, 8))
    por("orgao", "Sustentado", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=52, high=74, velocity=66), 22))
    por("oboe", "Contra", s, humano(mel(CONTRA, velocity=76, vezes=2), 26))

    # ============================================================ PONTE
    s, ch = "PONTE", acordes["PONTE"]
    parede(s, "quebrado", 112, False, True)
    por("bateria", "Kit", s, BD.bateria(12, Q, "quebrado", 114, 4, "parada"))
    por("trombones", "Leviata", s, humano(
        mel(MOTIVO_LENTO, velocity=106, vezes=3, compassos=4), 20))
    por("celos", "Leviata", s, humano(
        mel(MOTIVO_LENTO, velocity=96, vezes=3, compassos=4), 24))
    por("hum", "Hum", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=4, low=48, high=70, velocity=70), 38, 10))
    por("taiko", "Guerra", s, ORQ.taiko(12, Q, "guerra", 4, 36, 112))
    por("braam", "Golpe", s, [Note(c * COMPASSO, COMPASSO * 2, N("E2"), 112)
                              for c in (0, 8)])

    # ============================================================ SOLO
    s, ch = "SOLO", acordes["SOLO"]
    parede(s, "chug8", 100, True)
    por("bateria", "Kit", s, BD.bateria(28, Q, "rock_aberto", 108, 4, "curta"))
    por("gtr_lead", "Solo", s, EXP.arco(
        BD.solo_pentatonico(28, Q, N("E4") % 12, "menor", 5, 106, 0.62, 19),
        "crescendo", 0.16))
    por("cordas_ost", "Ostinato", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=4, low=64, high=86, velocity=74), 12, 8))
    por("orgao", "Sustentado", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=52, high=74, velocity=68), 22))
    por("trompas", "Bloco", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=45, high=67, velocity=82), 20))

    # ============================================================ QUEBRA
    s, ch = "QUEBRA", acordes["QUEBRA"]
    por("voz", "Sozinha", s, EXP.arco(mel(TEMA_SEREIA, velocity=80, compassos=8)[:12],
                                      "sino", 0.26))
    por("gtr_limpa", "Arpejo", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=2, low=52, high=76, velocity=58), 24, 10))
    por("piano", "Onda", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=2, low=48, high=72, velocity=48), 30, 11))
    por("harpa", "Onda", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=4, low=55, high=84, velocity=44,
                     contorno="sobe"), 26, 10))
    por("cordas_pad", "Pad", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=55, high=76, velocity=42), 30))
    por("flauta", "Ar", s, humano(mel(CONTRA, velocity=58, vezes=2), 28))

    # ============================================================ PRE 2
    s, ch = "PRE2", acordes["PRE2"]
    parede(s, "colcheias", 104, True)
    por("bateria", "Kit", s, BD.bateria(8, Q, "conducao", 108, 4, "longa"))
    por("cordas_ost", "Ostinato", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=4, low=64, high=88, velocity=80), 12, 8))
    por("metais", "Bloco", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=52, high=76, velocity=94), 18))
    por("tremolo", "Tensao", s, humano(
        ORQ.tremolo(ch, Q, 1, 4, low=64, high=84, velocity=48, divisao=8), 10, 6))
    por("timpani", "Rufo", s, ORQ.taiko(8, Q, "final", 4,
                                        ORQ.encaixar(N("C#2"), "timpani"), 98))

    # ============================================================ REFRÃO 2
    s, ch = "REFRAO2", acordes["REFRAO2"]
    parede(s, "colcheias", 112, True, True)
    por("bateria", "Kit", s, BD.bateria(16, Q, "rock_aberto", 116, 4, "curta"))
    por("voz", "Refrão", s, EXP.arco(mel(REFRAO_MEL, velocity=116, transpor=2,
                                         vezes=2), "crescendo", 0.12))
    por("voz_fundo", "Refrão", s, mel(REFRAO_MEL, velocity=70, transpor=2, vezes=2))
    for naipe, low, high, vel in (("soprano", 67, 83, 96), ("contralto", 58, 74, 92),
                                  ("tenor", 50, 69, 92), ("baixo", 43, 62, 96)):
        por(f"coro_{naipe}", "Refrão", s, humano(
            ORQ.bloco(ch, Q, 1, 4, n_vozes=1, low=low, high=high, velocity=vel), 32, 9))
    por("cordas_ost", "Ostinato", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=4, low=66, high=88, velocity=86), 12, 8))
    por("metais", "Bloco", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=54, high=78, velocity=100), 18))
    por("trompas", "Bloco", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=47, high=69, velocity=98), 20))
    por("sinos", "Badalo", s, humano(
        ORQ.pedal(ch, Q, 1, 4, "harpa", 96, oitava_dobrada=False), 24, 9))
    por("taiko", "Marcha", s, ORQ.taiko(16, Q, "marcha", 4, 36, 104))

    # ============================================================ FINAL
    s, ch = "FINAL", acordes["FINAL"]
    parede(s, "galope", 114, True, True)
    por("bateria", "Kit", s, BD.bateria(24, Q, "galope", 118, 4, "curta"))
    por("gtr_lead", "Tema", s, EXP.arco(mel(TEMA_SEREIA, velocity=112, transpor=2,
                                            vezes=2), "crescendo", 0.14))
    por("voz", "Tema", s, EXP.arco(mel(TEMA_SEREIA, velocity=114, transpor=2,
                                       vezes=2), "crescendo", 0.14))
    for naipe, low, high, vel in (("soprano", 67, 83, 98), ("contralto", 58, 74, 94),
                                  ("tenor", 50, 69, 94), ("baixo", 43, 62, 98)):
        por(f"coro_{naipe}", "Tutti", s, humano(
            ORQ.bloco(ch, Q, 1, 4, n_vozes=1, low=low, high=high, velocity=vel), 32, 9))
    por("cordas_ost", "Ostinato", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=4, low=66, high=88, velocity=88), 12, 8))
    por("metais", "Bloco", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=54, high=78, velocity=104), 18))
    por("trombones", "Bloco", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=2, low=42, high=64, velocity=100), 20))
    por("taiko", "Guerra", s, ORQ.taiko(24, Q, "guerra", 4, 36, 116))
    por("taiko_alto", "Galope", s, ORQ.taiko(24, Q, "galope", 4, 36, 96))
    por("timpani", "Rufo", s, ORQ.taiko(24, Q, "marcha", 4,
                                        ORQ.encaixar(N("F#2"), "timpani"), 108))

    # ============================================================ CODA
    s, ch = "CODA", acordes["CODA"]
    parede(s, "meio", 96, True)
    por("bateria", "Kit", s, BD.bateria(10, Q, "meio", 96, 4, "longa"))
    por("voz", "Adeus", s, EXP.arco(mel(TEMA_SEREIA, velocity=88,
                                        compassos=8)[:12], "diminuendo", 0.24))
    por("cordas_pad", "Pad", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=4, low=55, high=79, velocity=60), 28))
    por("celos", "Pedal", s, humano(ORQ.pedal(ch, Q, 1, 4, "violoncelo", 66), 30))
    por("hum", "Hum", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=50, high=72, velocity=54), 40, 10))
    por("gtr_limpa", "Arpejo", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=2, low=52, high=76, velocity=54), 24, 10))
    por("sinos", "Badalo", s, humano(
        ORQ.pedal(ch, Q, 1, 4, "harpa", 76, oitava_dobrada=False), 26, 9))

    # ---- preparação das entradas ---------------------------------------
    for alvo, forca in (("RIFF1", 116), ("REFRAO", 118), ("PONTE", 122),
                        ("REFRAO2", 120), ("FINAL", 122)):
        ev = ORQ.preparacao(INICIO[alvo], Q, 4, 2.0, True, True,
                            pitch_riser=N("E3"), pitch_impacto=N("E1"))
        for chave, notas in ev.items():
            if not notas:
                continue
            base = int(max(0, min(n.start for n in notas) // COMPASSO))
            pat = p.new_pattern(f"{chave} -> {alvo}", bars=3)
            pat.notes.extend(Note(n.start - base * COMPASSO, n.dur, n.pitch,
                                  min(127, forca)) for n in notas)
            p.place(pat, t[chave], base)

    for alvo in ("QUEBRA",):
        pat = p.new_pattern(f"Sub drop -> {alvo}", bars=2)
        pat.notes.append(Note(0, COMPASSO, N("E2"), 116))
        p.place(pat, t["subdrop"], INICIO[alvo])

    return p


def regioes() -> list[jogo.Regiao]:
    """A peça como a engine a vê: uma faixa de combate que repete inteira,
    com o loop pulando a introdução limpa."""
    return [jogo.Regiao(
        nome="COMBATE", inicio_compasso=0, compassos=TOTAL,
        loop_inicio_compasso=INICIO["RIFF1"], nivel=5, proxima="COMBATE",
        descricao="Confronto — rock com orquestra e coro")]


# --------------------------------------------------------------------------
def main() -> None:
    out = Path(__file__).parent
    print("Banco de instrumentos...")
    sf2 = L.ensure("generaluser_gs",
                   lambda g, tt: print(f"  {g/1e6:5.1f} MB", end="\r", flush=True))
    print(f"  {SF.open_soundfont(sf2).name}\n")

    p = build(sf2)
    dur = p.ticks_to_seconds(p.length_ticks)
    print(f"{p.name} — {TOTAL} compassos, {BPM} BPM, "
          f"{int(dur//60)}:{int(dur%60):02d}")
    print(f"{len(p.tracks)} tracks · {len(p.patterns)} patterns · "
          f"{sum(len(x.notes) for x in p.patterns.values())} notas\n")

    secoes = [(n, INICIO[n] * 4.0, (INICIO[n] + k) * 4.0) for n, k, _ in SECOES]
    print(H.sections_table_pt(
        H.analyze_sections(p.harmony_events(), secoes, beats_per_chord=4.0)))

    c = INT.calcular(p)
    print("\nIntensidade:")
    print("  " + c.resumo_pt(96))

    print("\nRenderizando (7 guitarras modeladas, kit, orquestra e coro)...")
    t0 = time.time()
    audio = engine.render_project(p, apply_master=False)
    print(f"  mixagem crua em {time.time()-t0:.0f}s")

    m = MST.masterizar(audio, p.sample_rate, "rock")
    del audio
    print(f"  master em {time.time()-t0:.0f}s no total\n")
    print(m.relatorio_pt(-11.0))

    man = jogo.exportar(p, m.audio, regioes(), out, "mare_de_ferro",
                        p.sample_rate, 16)
    p.save(out / "mare_de_ferro")
    smf.write_midi(smf.project_to_midi(p), out / "mare_de_ferro")
    c.exportar_json(out / "mare_de_ferro_intensidade", p.name)
    print(f"\n  {man['arquivo_completo']}  ({len(m.audio)/p.sample_rate/60:.2f} min)")
    print(f"  loop de {man['fases'][0]['segundos']['duracao_loop']:.1f}s "
          f"(a partir de {man['fases'][0]['segundos']['loop_inicio']:.1f}s)")


if __name__ == "__main__":
    main()
