"""
"Guardiã dos Mares" — tema de raid em três ciclos, ~12 min, com loop.

Três fases que se conectam e repetem, para a engine trocar conforme a luta anda:

    FASE 1 · A MARÉ DORME      c.  1- 72   72 BPM   nível 1-3
        A sereia canta. Piano suave, harpa, cordas em respiração lenta e a
        voz solista sozinha. O motivo do Leviatã aparece uma vez, no grave,
        e ninguém percebe que era um aviso.

    FASE 2 · OS SINOS DE FERRO c. 73-168  104 BPM   nível 3-5
        Os sinos começam. O mesmo tema da sereia, agora urgente, com
        violinos em ostinato e guitarras. A maré vira.

    FASE 3 · O LEVIATÃ DESPERTA c.169-296 132 BPM   nível 2-5
        Tudo junto: coro, metais, guitarras, taikos. O tema da sereia volta
        heroico por cima do ostinato do Leviatã — o ataque em conjunto. E a
        ressaca devolve a peça ao mar calmo, de onde ela recomeça.

O material temático é o que costura os três ciclos:

  * TEMA DA SEREIA   — lírico na fase 1, urgente na 2, heroico na 3
  * MOTIVO DO LEVIATÃ — quatro notas descendentes no grave; presságio na
    fase 1, ameaça na 2, ostinato que carrega a fase 3
  * TEMA DOS SINOS   — nasce na fase 2 e volta na coda da fase 3

A saída não é um arquivo, são cinco: a faixa inteira, um stem por fase (cada
um com o loop costurado, ver `audio/jogo.py`) e o manifesto JSON com os pontos
de troca em segundos, amostras e compassos.

    python examples/guardia_dos_mares.py
    python examples/guardia_dos_mares.py "referencia.mp3"
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from chipforge.audio import cinema as CIN
from chipforge.audio import engine
from chipforge.audio import fx
from chipforge.audio import jogo
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

Q = 480
COMPASSO = Q * 4
N = T.name_to_midi

BPM_BASE = 72          # a fase 1 manda no andamento declarado do projeto
BPM_F2 = 104
BPM_F3 = 132


def ac(raiz: str, qual: str = "min", oit: int = 3) -> list[int]:
    return [N(f"{raiz}{oit}") + i for i in T.CHORDS[qual]]


def prog(seq, vezes: int = 1) -> list[list[int]]:
    return [ac(*c) for c in seq] * vezes


# --------------------------------------------------------------------------
# Harmonia — Mi menor do começo ao fim; o que muda é a densidade e o andamento
# --------------------------------------------------------------------------
Em, C, G, D_, Am, Bm, B_ = ("E", "min"), ("C", "maj"), ("G", "maj"), \
    ("D", "maj"), ("A", "min"), ("B", "min"), ("B", "maj")

# A harmonia DO TEMA. Toda seção que enuncia o tema da sereia usa esta, e não
# uma parecida: um tema é uma melodia mais a harmonia contra a qual ela foi
# escrita. Reaproveitar a melodia sobre outra progressão é o erro que o medidor
# pega como "choque duro sustentado" — e ele tem razão, porque a nota longa da
# melodia passa a cair na nota errada do acorde.
HARM_SEREIA = [Em, C, G, D_, Am, Bm, Em, Em]

SECOES = [
    # ---------------- FASE 1 · A MARÉ DORME ----------------
    ("MARE", 12, prog([Em, Em, C, C, G, G, Am, Am, Em, C, Am, Bm])),
    ("CANTO", 16, prog(HARM_SEREIA, 2)),
    ("CHAMADO", 12, prog([C, G, Am, Em, C, G, D_, Bm, C, Am, Bm, Em])),
    ("PRESSAGIO", 16, prog(HARM_SEREIA, 2)),
    ("ARRASTO", 16, prog(HARM_SEREIA, 2)),

    # ---------------- FASE 2 · OS SINOS DE FERRO ----------------
    ("SINOS", 16, prog([Em, C, G, D_], 4)),
    ("INVESTIDA", 16, prog(HARM_SEREIA, 2)),
    ("CORRENTE", 16, prog([Em, D_, C, B_], 4)),
    ("OLHO", 12, prog([Am, Em, C, G, Am, Em, Bm, B_, C, G, Am, Em])),
    ("MARULHO", 20, prog(HARM_SEREIA, 2) + prog([Em, C, G, D_])),
    ("FENDA", 16, prog([Em, C, Am, B_], 4)),

    # ---------------- FASE 3 · O LEVIATÃ DESPERTA ----------------
    ("DESPERTAR", 16, prog([Em, C, D_, Em], 4)),
    ("ASSALTO", 16, prog(HARM_SEREIA, 2)),
    ("CANTO_FINAL", 16, prog([C, G, Em, D_, C, G, Am, B_], 2)),
    ("QUEBRA_MAR", 12, prog([Em, C, Am, Em, C, G, Bm, Em, Am, Em, B_, Em])),
    ("ULTIMO_FOLEGO", 24, prog([Em, C, G, D_], 6)),
    ("TUTTI", 24, prog(HARM_SEREIA, 3)),
    ("RESSACA", 20, prog(HARM_SEREIA, 2) + prog([Em, Am, Bm, Em])),
]

INICIO: dict[str, int] = {}
_b = 0
for _n, _k, _ in SECOES:
    INICIO[_n] = _b
    _b += _k
TOTAL = _b

FASES = [
    ("A MARE DORME", ["MARE", "CANTO", "CHAMADO", "PRESSAGIO", "ARRASTO"],
     "Fase 1 — a sereia canta. Piano, harpa e voz.", 2, "OS SINOS DE FERRO"),
    ("OS SINOS DE FERRO", ["SINOS", "INVESTIDA", "CORRENTE", "OLHO",
                           "MARULHO", "FENDA"],
     "Fase 2 — sinos, violinos e guitarras. A mare vira.", 4,
     "O LEVIATA DESPERTA"),
    ("O LEVIATA DESPERTA", ["DESPERTAR", "ASSALTO", "CANTO_FINAL", "QUEBRA_MAR",
                            "ULTIMO_FOLEGO", "TUTTI", "RESSACA"],
     "Fase 3 — ataque em conjunto, e a ressaca devolve ao mar.", 5,
     "A MARE DORME"),
]

# Volume relativo de cada seção: é este dicionário que vira o arco de 12 min.
NIVEIS = {
    "MARE": 0.26, "CANTO": 0.40, "CHAMADO": 0.48, "PRESSAGIO": 0.55, "ARRASTO": 0.66,
    "SINOS": 0.62, "INVESTIDA": 0.80, "CORRENTE": 0.84, "OLHO": 0.36,
    "MARULHO": 0.86, "FENDA": 0.96,
    "DESPERTAR": 0.98, "ASSALTO": 0.98, "CANTO_FINAL": 0.92, "QUEBRA_MAR": 0.34,
    "ULTIMO_FOLEGO": 0.82, "TUTTI": 1.00, "RESSACA": 0.42,
}

# --------------------------------------------------------------------------
# Temas — (nota, compasso, tempo, duração em semínimas)
# --------------------------------------------------------------------------
# O tema da sereia. Grau conjunto com um salto só, para caber numa garganta:
# é uma canção, não um exercício.
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

# Presságio: quatro notas descendo no grave. Na fase 3 vira ostinato.
MOTIVO_LEVIATA = [
    ("E2", 0, 0.0, 2.0), ("D2", 0, 2.0, 2.0),
    ("C2", 1, 0.0, 2.0), ("B1", 1, 2.0, 2.0),
    ("C2", 2, 0.0, 1.0), ("D2", 2, 1.0, 1.0), ("E2", 2, 2.0, 2.0),
    ("B1", 3, 0.0, 4.0),
]

# O mesmo motivo, uma nota por compasso. A CORRENTE troca de acorde a cada
# compasso (Mi-Re-Do-Si), e a versao de duas notas por compasso colocava a
# segunda nota de cada par sobre o acorde seguinte — 27 choques com os metais
# vinham so disso. Motivo e harmonia tem que andar no mesmo passo.
MOTIVO_LENTO = [
    ("E2", 0, 0.0, 4.0),
    ("D2", 1, 0.0, 4.0),
    ("C2", 2, 0.0, 4.0),
    ("B1", 3, 0.0, 4.0),
]

TEMA_SINOS = [
    ("B4", 0, 0.0, 2.0), ("E5", 0, 2.0, 2.0),
    ("B4", 1, 0.0, 4.0),
    ("G4", 2, 0.0, 2.0), ("B4", 2, 2.0, 2.0),
    ("E5", 3, 0.0, 4.0),
]

# Contracanto da fase 2: sobe onde o tema desce. É o que dá a sensação de duas
# forças puxando em direções opostas.
CONTRA_MARE = [
    ("E4", 0, 0.0, 1.0), ("F#4", 0, 1.0, 1.0), ("G4", 0, 2.0, 2.0),
    ("B4", 1, 0.0, 2.0), ("A4", 1, 2.0, 2.0),
    ("G4", 2, 0.0, 1.0), ("B4", 2, 1.0, 1.0), ("D5", 2, 2.0, 2.0),
    ("B4", 3, 0.0, 4.0),
    ("C5", 4, 0.0, 2.0), ("B4", 4, 2.0, 2.0),
    ("A4", 5, 0.0, 2.0), ("G4", 5, 2.0, 2.0),
    ("F#4", 6, 0.0, 1.0), ("A4", 6, 1.0, 1.0), ("B4", 6, 2.0, 2.0),
    ("E5", 7, 0.0, 3.5),
]

# A sereia na fase 3: mesmas alturas, valores dobrados. Aumentação é como se
# escreve "a mesma melodia, agora enorme" sem escrever outra melodia.
LAMENTO_FINAL = [
    ("E5", 0, 0.0, 3.0), ("D5", 0, 3.0, 1.0),
    ("B4", 1, 0.0, 2.0), ("A4", 1, 2.0, 2.0),
    ("G4", 2, 0.0, 3.0), ("A4", 2, 3.0, 1.0),
    ("B4", 3, 0.0, 4.0),
]


def mel(linha, **kw) -> list[Note]:
    return ORQ.escrever(linha, Q, 4, **kw)


# --------------------------------------------------------------------------
def build(sf2: Path) -> Project:
    sf = SF.open_soundfont(sf2)

    def gm(nome: str, **kw):
        return SF.instrument(sf, nome, **kw)

    p = Project(name="Guardia dos Mares", bpm=BPM_BASE, ppq=Q,
                bit_depth=24, master_gain=0.9,
                key_root=N("E4") % 12, scale="natural_minor")

    # A sala. Hall e não catedral: doze minutos de cauda de catedral por cima
    # de guitarra e taiko viram névoa. O corte de grave em 160 Hz é o que
    # mantém o Leviatã definido enquanto a sereia flutua.
    p.master_reverb = {"sala": "hall", "tamanho": 0.9, "wet": 0.85,
                       "corte_grave": 160.0, "corte_agudo": 9000.0}
    p.master_perfil = "cinematico"

    # ---- mapa de tempo: cada fase tem o seu pulso ----------------------
    mapa = DIN.MapaDeTempo(ppq=Q, bpm_base=BPM_BASE)
    mapa.marcar(0, BPM_BASE * 0.95)
    mapa.rampa(INICIO["CANTO"] * COMPASSO, INICIO["CHAMADO"] * COMPASSO,
               BPM_BASE * 0.95, BPM_BASE)
    # o ARRASTO puxa para o andamento da fase 2 antes dela começar: a troca de
    # fase não pode soar como alguém trocando de faixa
    mapa.rampa(INICIO["ARRASTO"] * COMPASSO, INICIO["SINOS"] * COMPASSO,
               BPM_BASE, BPM_F2)
    mapa.marcar(INICIO["INVESTIDA"] * COMPASSO, BPM_F2)
    mapa.rampa(INICIO["OLHO"] * COMPASSO, (INICIO["OLHO"] + 4) * COMPASSO,
               BPM_F2, BPM_F2 * 0.92)
    mapa.rampa(INICIO["MARULHO"] * COMPASSO, INICIO["FENDA"] * COMPASSO,
               BPM_F2 * 0.92, BPM_F2 * 1.04)
    mapa.rampa(INICIO["FENDA"] * COMPASSO, INICIO["DESPERTAR"] * COMPASSO,
               BPM_F2 * 1.04, BPM_F3)
    mapa.marcar(INICIO["CANTO_FINAL"] * COMPASSO, BPM_F3)
    mapa.rampa(INICIO["QUEBRA_MAR"] * COMPASSO,
               (INICIO["QUEBRA_MAR"] + 3) * COMPASSO, BPM_F3, BPM_F3 * 0.80)
    mapa.rampa(INICIO["ULTIMO_FOLEGO"] * COMPASSO, INICIO["TUTTI"] * COMPASSO,
               BPM_F3 * 0.80, BPM_F3 * 1.03)
    # a ressaca desacelera até o andamento da fase 1 — é o que faz a peça
    # inteira poder voltar ao começo sem emenda audível
    mapa.rampa(INICIO["RESSACA"] * COMPASSO, TOTAL * COMPASSO,
               BPM_F3 * 1.03, BPM_BASE * 0.95)
    p.tempo_map = mapa.to_list()

    t: dict[str, object] = {}

    # =================================================== A SEREIA
    t["voz"] = p.new_track("Sereia", VOZ.voz("Solista feminina", pan=-0.05))
    t["voz"].volume = 0.82
    t["voz"].reverb_send = 0.44
    t["voz"].inserts = [
        {"tipo": "eq", "modo": "highpass", "freq": 130.0, "q": 0.7},
        {"tipo": "eq", "freq": 2500.0, "ganho_db": 2.5, "q": 0.9},
        {"tipo": "comp", "limiar_db": -20.0, "razao": 3.0,
         "attack_ms": 12.0, "release_ms": 160.0, "makeup_db": 3.0},
    ]

    # A voz da profundeza: a mesma frase, uma oitava abaixo e com vogal escura,
    # bem molhada. É o que faz a sereia soar como duas coisas ao mesmo tempo.
    t["voz_fundo"] = p.new_track("Sereia (fundo)",
                                 VOZ.voz("Lamento (u)", pan=0.35, gain=0.30))
    t["voz_fundo"].volume = 0.36
    t["voz_fundo"].reverb_send = 0.88

    for naipe, preset, pan in (
            ("soprano", "Coro soprano (aah)", -0.45),
            ("contralto", "Coro contralto (ooh)", 0.40),
            ("tenor", "Coro tenor (aah)", -0.30),
            ("baixo", "Coro baixo (ooh)", 0.28)):
        tr = p.new_track(f"Coro {naipe}",
                         VOZ.voz(preset, pan=pan, seed=abs(hash(naipe)) % 999))
        tr.section = dict(VOZ.SECAO_CORO_INTIMO)
        tr.reverb_send = 0.72
        tr.volume = 0.46
        t[f"coro_{naipe}"] = tr

    t["hum"] = p.new_track("Coro (boca fechada)",
                           VOZ.voz("Coro de boca fechada", pan=0.0))
    t["hum"].section = dict(VOZ.SECAO_CORO)
    t["hum"].reverb_send = 0.8
    t["hum"].volume = 0.44

    # =================================================== O MAR (fase 1)
    t["piano"] = p.new_track("Piano", gm("Stereo Grand", gain=0.85))
    t["piano"].reverb_send = 0.42
    t["piano"].volume = 0.56
    t["piano"].pan = -0.15
    t["piano"].inserts = [{"tipo": "eq", "modo": "highpass", "freq": 55.0, "q": 0.7}]

    t["harpa"] = p.new_track("Harpa", gm("Orchestral Harp", gain=0.9))
    t["harpa"].reverb_send = 0.52
    t["harpa"].volume = 0.48
    t["harpa"].pan = 0.28

    t["flauta"] = p.new_track("Flauta", gm("Flute", gain=0.85))
    t["flauta"].reverb_send = 0.55
    t["flauta"].volume = 0.40
    t["flauta"].pan = 0.2

    # =================================================== OS SINOS (fase 2)
    t["sinos"] = p.new_track("Sinos", gm("Tubular Bells", gain=0.9))
    t["sinos"].reverb_send = 0.62
    t["sinos"].volume = 0.52
    t["sinos"].pan = -0.3
    # o sino tem fundamental grave e um agudo que corta tudo: o corte embaixo
    # tira o ronco e deixa só o badalo
    t["sinos"].inserts = [{"tipo": "eq", "modo": "highpass", "freq": 180.0, "q": 0.7},
                          {"tipo": "eq", "freq": 3600.0, "ganho_db": 2.0, "q": 0.8}]

    t["glock"] = p.new_track("Glockenspiel", gm("Glockenspiel", gain=0.8))
    t["glock"].reverb_send = 0.6
    t["glock"].volume = 0.32
    t["glock"].pan = 0.45

    # =================================================== CORDAS
    t["cordas_pad"] = p.new_track("Cordas (pad)", gm("Stereo Strings Slow", gain=0.85))
    t["cordas_pad"].section = {"vozes": 4, "detune": 10, "spread_ms": 28, "pan": 0.7}
    t["cordas_pad"].reverb_send = 0.55
    t["cordas_pad"].volume = 0.50

    t["cordas_ost"] = p.new_track("Cordas (ostinato)",
                                  gm("Stereo Strings Fast", gain=0.85))
    t["cordas_ost"].section = {"vozes": 3, "detune": 8, "spread_ms": 16, "pan": 0.55}
    t["cordas_ost"].reverb_send = 0.34
    t["cordas_ost"].volume = 0.46
    t["cordas_ost"].inserts = [{"tipo": "eq", "freq": 3200.0, "ganho_db": 2.0, "q": 0.8}]

    t["tremolo"] = p.new_track("Cordas (tremolo)",
                               gm("Stereo Strings Trem", gain=0.85))
    t["tremolo"].section = {"vozes": 2, "detune": 9, "spread_ms": 20, "pan": 0.6}
    t["tremolo"].reverb_send = 0.5
    t["tremolo"].volume = 0.34

    t["violino"] = p.new_track("Violino solo", gm("Violin", gain=0.9))
    t["violino"].reverb_send = 0.45
    t["violino"].volume = 0.48
    t["violino"].pan = -0.35
    t["violino"].instrument.vibrato_depth_cents = 22

    t["celos"] = p.new_track("Violoncelos", gm("Cello", gain=0.9))
    t["celos"].section = {"vozes": 3, "detune": 7, "spread_ms": 22, "pan": 0.4}
    t["celos"].reverb_send = 0.42
    t["celos"].volume = 0.58

    t["baixos"] = p.new_track("Contrabaixos", gm("Double Bass", gain=0.95))
    t["baixos"].reverb_send = 0.16
    t["baixos"].volume = 0.64
    t["baixos"].inserts = [{"tipo": "eq", "modo": "lowshelf", "freq": 90.0,
                            "ganho_db": 2.5}]

    # =================================================== METAIS
    t["trompas"] = p.new_track("Trompas", gm("French Horns", gain=0.9))
    t["trompas"].section = {"vozes": 3, "detune": 8, "spread_ms": 20, "pan": 0.35}
    t["trompas"].reverb_send = 0.48
    t["trompas"].volume = 0.54

    t["metais"] = p.new_track("Metais", gm("Brass Section", gain=0.9))
    t["metais"].section = {"vozes": 2, "detune": 6, "spread_ms": 14, "pan": 0.45}
    t["metais"].reverb_send = 0.36
    t["metais"].volume = 0.48

    t["trombones"] = p.new_track("Trombones", gm("Trombone", gain=0.9))
    t["trombones"].section = {"vozes": 2, "detune": 7, "spread_ms": 16, "pan": 0.3}
    t["trombones"].reverb_send = 0.34
    t["trombones"].volume = 0.50

    t["timpani"] = p.new_track("Timpani", gm("Timpani", gain=0.9))
    t["timpani"].reverb_send = 0.4
    t["timpani"].volume = 0.60

    # =================================================== A BANDA (fases 2 e 3)
    # Gravação dupla: a mesma parte tocada duas vezes, uma em cada lado. É o
    # truque mais antigo do rock e continua sendo o que dá largura de verdade.
    for lado, pan in (("esq", -0.85), ("dir", 0.85)):
        tr = p.new_track(f"Guitarra {lado}", gm("Distortion Guitar", gain=0.9))
        tr.inserts = [dict(fx.PRESETS_AMP["rock"], pan=pan)]
        tr.reverb_send = 0.14
        tr.volume = 0.56
        t[f"gtr_{lado}"] = tr

    t["gtr_lead"] = p.new_track("Guitarra lead", gm("Overdrive Guitar", gain=0.9))
    t["gtr_lead"].inserts = [dict(fx.PRESETS_AMP["crunch"], ganho=7.5, pan=0.12),
                             {"tipo": "eq", "freq": 1500, "ganho_db": 2.5, "q": 0.8}]
    t["gtr_lead"].reverb_send = 0.3
    t["gtr_lead"].volume = 0.52

    t["baixo_el"] = p.new_track("Baixo elétrico", gm("Finger Bass", gain=0.95))
    t["baixo_el"].inserts = [
        {"tipo": "eq", "modo": "lowshelf", "freq": 110, "ganho_db": 3.5},
        {"tipo": "eq", "freq": 900, "ganho_db": 3.0, "q": 1.1},
        {"tipo": "comp", "limiar_db": -20.0, "razao": 4.0, "attack_ms": 6.0,
         "release_ms": 110.0, "makeup_db": 3.0}]
    t["baixo_el"].volume = 0.60

    # =================================================== DESENHO DE SOM
    t["taiko"] = p.new_track("Taiko", CIN.instrumento("Taiko"))
    t["taiko"].reverb_send = 0.22
    t["taiko"].volume = 0.66

    t["taiko_alto"] = p.new_track("Taiko pequeno",
                                  CIN.instrumento("Taiko pequeno", pan=-0.4))
    t["taiko_alto"].reverb_send = 0.3
    t["taiko_alto"].volume = 0.42

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
    t["braam"].volume = 0.44

    t["subdrop"] = p.new_track("Sub drop", CIN.instrumento("Sub drop"))
    t["subdrop"].reverb_send = 0.1
    t["subdrop"].volume = 0.55

    # ---- arco de dinâmica ---------------------------------------------
    arco = DIN.arco_de_secoes([(n, k) for n, k, _ in SECOES], COMPASSO, NIVEIS)
    sem_arco = ("impacto", "subdrop", "braam", "riser", "swell")
    for chave, tr in t.items():
        if chave in sem_arco:
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
        return ORQ.humanizar_naipe(notas, Q, espalhar, BPM_BASE, vel)

    def badalada(ch, vel=96, oitava=0):
        """Sino seguindo o acorde: fundamental e quinta, uma por compasso.

        O tema dos sinos é fixo, e um sino tem cauda de vários segundos — se
        ele segurar uma nota que não pertence ao acorde seguinte, ela continua
        soando por cima dele. Nas seções em que a harmonia não é a do tema dos
        sinos, o badalo segue os acordes em vez de repetir o tema.
        """
        notas = []
        for i, acorde in enumerate(ch):
            raiz = ORQ.encaixar(acorde[0], "harpa")
            while raiz < N("E4"):
                raiz += 12
            while raiz > N("E5"):
                raiz -= 12
            notas.append(Note(i * COMPASSO, COMPASSO, raiz + 12 * oitava, vel))
            if i % 2 == 1:
                quinta = raiz + 7
                notas.append(Note(i * COMPASSO + Q * 2, COMPASSO // 2,
                                  quinta + 12 * oitava, max(1, vel - 18)))
        return notas

    def onda(ch, low=52, high=79, vel=54, div=2, contorno="sobe_desce"):
        """A figura de balanço do mar: arpejo lento subindo e descendo."""
        return ORQ.ostinato(ch, Q, 1, 4, divisao=div, low=low, high=high,
                            velocity=vel, contorno=contorno)

    # ============================================================ FASE 1
    # --- MARE: só o mar. Piano, cordas em respiração, nenhuma voz.
    s, ch = "MARE", acordes["MARE"]
    por("piano", "Onda", s, humano(onda(ch, 48, 76, 50), 34, 11))
    por("cordas_pad", "Respiro", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=55, high=76, velocity=42), 34))
    por("baixos", "Pedal", s, humano(
        ORQ.pedal(ch, Q, 1, 4, "contrabaixo", 58, oitava_dobrada=False), 36))
    por("harpa", "Onda", s, humano(onda(ch[6:], 55, 84, 44, 4, "sobe"), 28, 10))

    # --- CANTO: a sereia entra.
    s, ch = "CANTO", acordes["CANTO"]
    por("voz", "Tema", s, EXP.arco(mel(TEMA_SEREIA, velocity=74, vezes=2),
                                   "sino", 0.24))
    por("piano", "Onda", s, humano(onda(ch, 48, 76, 54), 32, 11))
    por("harpa", "Onda", s, humano(onda(ch, 55, 84, 48, 4, "sobe_desce"), 26, 10))
    por("cordas_pad", "Respiro", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=55, high=79, velocity=46), 32))
    por("celos", "Pedal", s, humano(ORQ.pedal(ch, Q, 1, 4, "violoncelo", 56), 32))
    por("baixos", "Pedal", s, humano(
        ORQ.pedal(ch, Q, 1, 4, "contrabaixo", 62, oitava_dobrada=False), 36))

    # --- CHAMADO: o coro de boca fechada responde à sereia.
    s, ch = "CHAMADO", acordes["CHAMADO"]
    por("voz", "Chamado", s, EXP.arco(
        mel(TEMA_SEREIA, velocity=82, compassos=8)[:12], "sino", 0.22))
    por("voz_fundo", "Chamado", s, mel(TEMA_SEREIA, velocity=58,
                                       oitava=-1, compassos=8)[:12])
    por("hum", "Hum", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=4, low=50, high=74, velocity=58), 40, 10))
    por("piano", "Onda", s, humano(onda(ch, 48, 79, 56), 32, 11))
    por("cordas_pad", "Respiro", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=57, high=81, velocity=52), 32))
    por("flauta", "Ar", s, humano(mel(CONTRA_MARE, velocity=58, compassos=8)[:10], 30))
    por("celos", "Pedal", s, humano(ORQ.pedal(ch, Q, 1, 4, "violoncelo", 62), 32))
    por("baixos", "Pedal", s, humano(ORQ.pedal(ch, Q, 1, 4, "contrabaixo", 68), 36))

    # --- PRESSAGIO: o Leviatã aparece pela primeira vez, no grave.
    s, ch = "PRESSAGIO", acordes["PRESSAGIO"]
    por("trombones", "Leviata", s, humano(
        mel(MOTIVO_LEVIATA, velocity=64, vezes=4, compassos=4), 30))
    por("baixos", "Leviata", s, humano(
        mel(MOTIVO_LEVIATA, velocity=72, vezes=4, compassos=4), 34))
    por("voz", "Tema", s, EXP.arco(mel(TEMA_SEREIA, velocity=86, vezes=2),
                                   "sino", 0.20))
    por("tremolo", "Tensao", s, humano(
        ORQ.tremolo(ch, Q, 1, 4, low=62, high=81, velocity=38, divisao=8), 12, 6))
    por("cordas_pad", "Respiro", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=57, high=79, velocity=54), 30))
    por("piano", "Onda", s, humano(onda(ch, 48, 76, 52), 32, 11))
    por("timpani", "Rufo", s, ORQ.taiko(16, Q, "meio", 4,
                                        ORQ.encaixar(N("E2"), "timpani"), 72))

    # --- ARRASTO: a maré puxa. Tudo cresce até os sinos.
    s, ch = "ARRASTO", acordes["ARRASTO"]
    por("cordas_ost", "Ostinato", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=2, low=62, high=84, velocity=62), 16, 8))
    por("hum", "Hum", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=4, low=50, high=74, velocity=66), 40, 10))
    por("trompas", "Bloco", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=48, high=69, velocity=70), 24))
    por("celos", "Linha", s, humano(ORQ.pedal(ch, Q, 1, 4, "violoncelo", 76), 28))
    por("baixos", "Pedal", s, humano(ORQ.pedal(ch, Q, 1, 4, "contrabaixo", 80), 32))
    por("taiko", "Distante", s, ORQ.taiko(16, Q, "meio", 4, 36, 82))
    por("piano", "Onda", s, humano(onda(ch, 48, 79, 60), 30, 11))
    por("voz", "Tema", s, EXP.arco(mel(TEMA_SEREIA, velocity=94, vezes=2),
                                   "crescendo", 0.18))

    # ============================================================ FASE 2
    # --- SINOS: o badalo. A mesma harmonia, outro mundo.
    s, ch = "SINOS", acordes["SINOS"]
    por("sinos", "Badalo", s, humano(
        mel(TEMA_SINOS, velocity=96, vezes=4, compassos=4), 26, 9))
    por("cordas_ost", "Ostinato", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=4, low=62, high=84, velocity=68), 14, 7))
    por("cordas_pad", "Pad", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=4, low=57, high=81, velocity=56), 26))
    por("celos", "Linha", s, humano(ORQ.pedal(ch, Q, 1, 4, "violoncelo", 78), 28))
    por("baixos", "Pedal", s, humano(ORQ.pedal(ch, Q, 1, 4, "contrabaixo", 82), 32))
    por("taiko", "Marcha", s, ORQ.taiko(16, Q, "marcha", 4, 36, 92))
    por("timpani", "Marca", s, ORQ.taiko(16, Q, "meio", 4,
                                         ORQ.encaixar(N("E2"), "timpani"), 88))
    por("glock", "Brilho", s, humano(
        mel(TEMA_SINOS, velocity=64, oitava=1, vezes=4, compassos=4), 24, 10))

    # --- INVESTIDA: guitarras e violino. O tema da sereia vira urgência.
    s, ch = "INVESTIDA", acordes["INVESTIDA"]
    for lado in ("esq", "dir"):
        por(f"gtr_{lado}", "Riff", s, humano(
            ORQ.ostinato(ch, Q, 1, 4, divisao=4, low=40, high=59,
                         velocity=92, contorno="pendulo"), 10, 7))
    por("violino", "Tema", s, EXP.arco(mel(TEMA_SEREIA, velocity=100, vezes=2),
                                       "crescendo", 0.16))
    por("voz", "Tema", s, EXP.arco(mel(TEMA_SEREIA, velocity=96, vezes=2),
                                   "sino", 0.18))
    por("baixo_el", "Baixo", s, humano(
        ORQ.pedal(ch, Q, 1, 4, "contrabaixo", 96, oitava_dobrada=False), 14))
    por("cordas_ost", "Ostinato", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=4, low=64, high=86, velocity=76), 13, 7))
    por("taiko", "Galope", s, ORQ.taiko(16, Q, "galope", 4, 36, 100))
    por("taiko_alto", "Marcha", s, ORQ.taiko(16, Q, "marcha", 4, 36, 84))
    por("sinos", "Badalo", s, humano(
        mel(TEMA_SINOS, velocity=88, vezes=4, compassos=4), 26, 9))

    # --- CORRENTE: a tetracorde descendente. É o Leviatã puxando.
    s, ch = "CORRENTE", acordes["CORRENTE"]
    for lado in ("esq", "dir"):
        por(f"gtr_{lado}", "Riff", s, humano(
            ORQ.ostinato(ch, Q, 1, 4, divisao=4, low=40, high=59,
                         velocity=96, contorno="pendulo"), 10, 7))
    # A progressão desta seção É o motivo do Leviatã harmonizado (Mi-Ré-Dó-Si).
    # Quem toca por cima dela tem que ser o próprio motivo: o contracanto foi
    # escrito para Em-C-G-D e batia de segunda com os metais e o coro.
    por("gtr_lead", "Leviata", s, EXP.arco(
        mel(MOTIVO_LENTO, velocity=102, oitava=2, vezes=4, compassos=4),
        "crescendo", 0.16))
    por("metais", "Bloco", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=52, high=76, velocity=88), 20))
    por("trombones", "Leviata", s, humano(
        mel(MOTIVO_LENTO, velocity=94, vezes=4, compassos=4), 22))
    por("baixo_el", "Baixo", s, humano(
        ORQ.pedal(ch, Q, 1, 4, "contrabaixo", 100, oitava_dobrada=False), 14))
    por("cordas_ost", "Ostinato", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=4, low=64, high=86, velocity=80), 13, 7))
    por("taiko", "Guerra", s, ORQ.taiko(16, Q, "guerra", 4, 36, 106))
    for naipe, low, high, vel in (("soprano", 67, 81, 82), ("contralto", 58, 72, 80),
                                  ("tenor", 50, 67, 80), ("baixo", 43, 60, 84)):
        por(f"coro_{naipe}", "Corrente", s, humano(
            ORQ.bloco(ch, Q, 1, 4, n_vozes=1, low=low, high=high, velocity=vel), 34, 9))

    # --- OLHO: o olho da tempestade. Tudo cai; só a sereia e um sino.
    s, ch = "OLHO", acordes["OLHO"]
    por("voz", "Sozinha", s, EXP.arco(
        mel(TEMA_SEREIA, velocity=78, compassos=8)[:12], "sino", 0.26))
    por("piano", "Onda", s, humano(onda(ch, 48, 76, 48), 32, 11))
    por("sinos", "Badalo", s, humano(
        mel(TEMA_SINOS, velocity=62, vezes=3, compassos=4), 30, 9))
    por("cordas_pad", "Respiro", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=57, high=79, velocity=40), 32))
    por("baixos", "Pedal", s, humano(
        ORQ.pedal(ch, Q, 1, 4, "contrabaixo", 58, oitava_dobrada=False), 36))

    # --- MARULHO: volta a subir, acelerando.
    s, ch = "MARULHO", acordes["MARULHO"]
    por("cordas_ost", "Ostinato", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=4, low=64, high=86, velocity=82), 12, 8))
    for lado in ("esq", "dir"):
        por(f"gtr_{lado}", "Riff", s, humano(
            ORQ.ostinato(ch, Q, 1, 4, divisao=4, low=40, high=59,
                         velocity=98, contorno="pendulo"), 10, 7))
    por("violino", "Tema", s, EXP.arco(
        mel(TEMA_SEREIA, velocity=104, vezes=2, compassos=8)[:36], "crescendo", 0.15))
    por("voz", "Tema", s, EXP.arco(
        mel(TEMA_SEREIA, velocity=100, vezes=2, compassos=8)[:36], "crescendo", 0.15))
    por("baixo_el", "Baixo", s, humano(
        ORQ.pedal(ch, Q, 1, 4, "contrabaixo", 102, oitava_dobrada=False), 14))
    por("trompas", "Bloco", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=48, high=69, velocity=88), 22))
    por("taiko", "Adensa", s, ORQ.taiko(20, Q, "adensa", 4, 36, 108))
    por("taiko_alto", "Galope", s, ORQ.taiko(20, Q, "galope", 4, 36, 90))
    por("sinos", "Badalo", s, humano(
        mel(TEMA_SINOS, velocity=94, vezes=5, compassos=4), 26, 9))

    # --- FENDA: o clímax da fase 2. A fenda no mar se abre.
    s, ch = "FENDA", acordes["FENDA"]
    for lado in ("esq", "dir"):
        por(f"gtr_{lado}", "Riff", s, humano(
            ORQ.ostinato(ch, Q, 1, 4, divisao=4, low=40, high=59,
                         velocity=104, contorno="pendulo"), 10, 7))
    por("metais", "Bloco", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=52, high=76, velocity=100), 20))
    por("trompas", "Bloco", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=45, high=67, velocity=98), 22))
    por("trombones", "Leviata", s, humano(
        mel(MOTIVO_LEVIATA, velocity=104, vezes=4, compassos=4), 22))
    por("cordas_ost", "Ostinato", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=4, low=64, high=88, velocity=88), 12, 8))
    por("baixo_el", "Baixo", s, humano(
        ORQ.pedal(ch, Q, 1, 4, "contrabaixo", 106, oitava_dobrada=False), 14))
    por("taiko", "Guerra", s, ORQ.taiko(16, Q, "guerra", 4, 36, 114))
    por("timpani", "Rufo", s, ORQ.taiko(16, Q, "marcha", 4,
                                        ORQ.encaixar(N("E2"), "timpani"), 104))
    for naipe, low, high, vel in (("soprano", 67, 81, 92), ("contralto", 58, 72, 88),
                                  ("tenor", 50, 67, 88), ("baixo", 43, 60, 92)):
        por(f"coro_{naipe}", "Fenda", s, humano(
            ORQ.bloco(ch, Q, 1, 4, n_vozes=1, low=low, high=high, velocity=vel), 34, 9))
    por("braam", "Golpe", s, [Note(c * COMPASSO, COMPASSO * 2, N("E2"), 112)
                              for c in (0, 8)])

    # ============================================================ FASE 3
    # --- DESPERTAR: o Leviatã vira ostinato. Tudo em cima dele.
    s, ch = "DESPERTAR", acordes["DESPERTAR"]
    por("trombones", "Ostinato", s, humano(
        mel(MOTIVO_LEVIATA, velocity=108, vezes=4, compassos=4), 20))
    por("baixos", "Ostinato", s, humano(
        mel(MOTIVO_LEVIATA, velocity=104, vezes=4, compassos=4), 26))
    por("metais", "Bloco", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=52, high=76, velocity=104), 20))
    for lado in ("esq", "dir"):
        por(f"gtr_{lado}", "Riff", s, humano(
            ORQ.ostinato(ch, Q, 1, 4, divisao=4, low=40, high=59,
                         velocity=104, contorno="pendulo"), 10, 7))
    por("cordas_ost", "Ostinato", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=4, low=64, high=88, velocity=90), 12, 8))
    por("baixo_el", "Baixo", s, humano(
        ORQ.pedal(ch, Q, 1, 4, "contrabaixo", 108, oitava_dobrada=False), 14))
    por("taiko", "Guerra", s, ORQ.taiko(16, Q, "guerra", 4, 36, 118))
    por("taiko_alto", "Galope", s, ORQ.taiko(16, Q, "galope", 4, 36, 96))
    for naipe, low, high, vel in (("soprano", 67, 81, 94), ("contralto", 58, 72, 90),
                                  ("tenor", 50, 67, 90), ("baixo", 43, 60, 96)):
        por(f"coro_{naipe}", "Despertar", s, humano(
            ORQ.bloco(ch, Q, 1, 4, n_vozes=1, low=low, high=high, velocity=vel), 34, 9))

    # --- ASSALTO: o ataque em conjunto. Guitarra e violino no mesmo tema.
    s, ch = "ASSALTO", acordes["ASSALTO"]
    por("gtr_lead", "Tema", s, EXP.arco(mel(TEMA_SEREIA, velocity=110, vezes=2),
                                        "crescendo", 0.14))
    por("violino", "Tema", s, EXP.arco(mel(TEMA_SEREIA, velocity=106, vezes=2),
                                       "crescendo", 0.14))
    por("voz", "Tema", s, EXP.arco(mel(TEMA_SEREIA, velocity=108, vezes=2),
                                   "crescendo", 0.14))
    for lado in ("esq", "dir"):
        por(f"gtr_{lado}", "Riff", s, humano(
            ORQ.ostinato(ch, Q, 1, 4, divisao=4, low=40, high=59,
                         velocity=104, contorno="pendulo"), 10, 7))
    por("metais", "Bloco", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=52, high=76, velocity=100), 20))
    por("trompas", "Bloco", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=45, high=67, velocity=98), 22))
    por("baixo_el", "Baixo", s, humano(
        ORQ.pedal(ch, Q, 1, 4, "contrabaixo", 108, oitava_dobrada=False), 14))
    por("cordas_ost", "Ostinato", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=4, low=64, high=88, velocity=90), 12, 8))
    por("taiko", "Guerra", s, ORQ.taiko(16, Q, "guerra", 4, 36, 118))
    por("timpani", "Rufo", s, ORQ.taiko(16, Q, "marcha", 4,
                                        ORQ.encaixar(N("E2"), "timpani"), 108))

    # --- CANTO_FINAL: o tema da sereia em aumentação, heroico.
    s, ch = "CANTO_FINAL", acordes["CANTO_FINAL"]
    por("voz", "Lamento", s, EXP.arco(
        mel(LAMENTO_FINAL, velocity=112, vezes=4, compassos=4), "sino", 0.18))
    por("voz_fundo", "Lamento", s, mel(LAMENTO_FINAL, velocity=70, oitava=-1,
                                       vezes=4, compassos=4))
    for naipe, low, high, vel in (("soprano", 67, 81, 96), ("contralto", 58, 72, 92),
                                  ("tenor", 50, 67, 92), ("baixo", 43, 60, 96)):
        por(f"coro_{naipe}", "Canto", s, humano(
            ORQ.bloco(ch, Q, 1, 4, n_vozes=1, low=low, high=high, velocity=vel), 34, 9))
    por("metais", "Bloco", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=52, high=76, velocity=96), 20))
    por("cordas_pad", "Pad", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=4, low=57, high=81, velocity=76), 24))
    por("cordas_ost", "Ostinato", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=4, low=64, high=88, velocity=86), 12, 8))
    por("celos", "Linha", s, humano(ORQ.pedal(ch, Q, 1, 4, "violoncelo", 92), 26))
    por("baixos", "Pedal", s, humano(ORQ.pedal(ch, Q, 1, 4, "contrabaixo", 96), 30))
    por("taiko", "Marcha", s, ORQ.taiko(16, Q, "marcha", 4, 36, 110))
    por("sinos", "Badalo", s, humano(badalada(ch, 98), 26, 9))

    # --- QUEBRA_MAR: o silêncio antes do fim. Volta o piano da fase 1.
    s, ch = "QUEBRA_MAR", acordes["QUEBRA_MAR"]
    por("voz", "Sozinha", s, EXP.arco(
        mel(TEMA_SEREIA, velocity=76, compassos=8)[:12], "sino", 0.26))
    por("piano", "Onda", s, humano(onda(ch, 48, 76, 50), 32, 11))
    por("harpa", "Onda", s, humano(onda(ch, 55, 84, 46, 4, "sobe"), 28, 10))
    por("cordas_pad", "Respiro", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=57, high=79, velocity=40), 32))
    por("baixos", "Pedal", s, humano(
        ORQ.pedal(ch, Q, 1, 4, "contrabaixo", 56, oitava_dobrada=False), 36))
    por("glock", "Gota", s, humano(
        mel(TEMA_SINOS, velocity=52, oitava=1, vezes=3, compassos=4), 26, 10))

    # --- ULTIMO_FOLEGO: reconstrói, acelerando até o tutti.
    s, ch = "ULTIMO_FOLEGO", acordes["ULTIMO_FOLEGO"]
    por("cordas_ost", "Ostinato", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=4, low=64, high=86, velocity=80), 13, 8))
    por("hum", "Hum", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=4, low=50, high=74, velocity=70), 40, 10))
    por("trompas", "Bloco", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=48, high=69, velocity=86), 22))
    for lado in ("esq", "dir"):
        por(f"gtr_{lado}", "Riff", s, humano(
            ORQ.ostinato(ch, Q, 1, 4, divisao=4, low=40, high=59,
                         velocity=94, contorno="pendulo"), 10, 7))
    por("baixo_el", "Baixo", s, humano(
        ORQ.pedal(ch, Q, 1, 4, "contrabaixo", 100, oitava_dobrada=False), 14))
    por("violino", "Contra", s, EXP.arco(
        mel(CONTRA_MARE, velocity=98, vezes=3), "crescendo", 0.16))
    por("taiko", "Adensa", s, ORQ.taiko(24, Q, "adensa", 4, 36, 110))
    por("celos", "Linha", s, humano(ORQ.pedal(ch, Q, 1, 4, "violoncelo", 88), 26))

    # --- TUTTI: tudo, ao mesmo tempo. O ataque em conjunto.
    s, ch = "TUTTI", acordes["TUTTI"]
    por("voz", "Tema", s, EXP.arco(mel(TEMA_SEREIA, velocity=115, vezes=3),
                                   "crescendo", 0.12))
    por("gtr_lead", "Tema", s, EXP.arco(mel(TEMA_SEREIA, velocity=112, vezes=3),
                                        "crescendo", 0.12))
    por("violino", "Tema", s, EXP.arco(mel(TEMA_SEREIA, velocity=108, oitava=1,
                                           vezes=3), "crescendo", 0.12))
    por("trombones", "Ostinato", s, humano(
        mel(MOTIVO_LEVIATA, velocity=110, vezes=6, compassos=4), 20))
    por("metais", "Bloco", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=52, high=76, velocity=106), 20))
    por("trompas", "Bloco", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=45, high=67, velocity=104), 22))
    for lado in ("esq", "dir"):
        por(f"gtr_{lado}", "Riff", s, humano(
            ORQ.ostinato(ch, Q, 1, 4, divisao=4, low=40, high=59,
                         velocity=106, contorno="pendulo"), 10, 7))
    por("baixo_el", "Baixo", s, humano(
        ORQ.pedal(ch, Q, 1, 4, "contrabaixo", 110, oitava_dobrada=False), 14))
    por("cordas_ost", "Ostinato", s, humano(
        ORQ.ostinato(ch, Q, 1, 4, divisao=4, low=64, high=88, velocity=92), 12, 8))
    por("celos", "Linha", s, humano(ORQ.pedal(ch, Q, 1, 4, "violoncelo", 96), 26))
    por("baixos", "Pedal", s, humano(ORQ.pedal(ch, Q, 1, 4, "contrabaixo", 100), 30))
    por("taiko", "Guerra", s, ORQ.taiko(24, Q, "guerra", 4, 36, 120))
    por("taiko_alto", "Galope", s, ORQ.taiko(24, Q, "galope", 4, 36, 100))
    por("timpani", "Rufo", s, ORQ.taiko(24, Q, "marcha", 4,
                                        ORQ.encaixar(N("E2"), "timpani"), 110))
    por("sinos", "Badalo", s, humano(badalada(ch, 104), 26, 9))
    for naipe, low, high, vel in (("soprano", 67, 81, 100), ("contralto", 58, 72, 96),
                                  ("tenor", 50, 67, 96), ("baixo", 43, 60, 100)):
        por(f"coro_{naipe}", "Tutti", s, humano(
            ORQ.bloco(ch, Q, 1, 4, n_vozes=1, low=low, high=high, velocity=vel), 34, 9))

    # --- RESSACA: o mar se fecha. Volta ao começo.
    s, ch = "RESSACA", acordes["RESSACA"]
    por("voz", "Adeus", s, EXP.arco(
        mel(TEMA_SEREIA, velocity=88, vezes=2), "diminuendo", 0.22))
    por("voz_fundo", "Adeus", s, mel(TEMA_SEREIA, velocity=58, oitava=-1, vezes=2))
    por("cordas_pad", "Respiro", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=4, low=55, high=79, velocity=54), 30))
    por("piano", "Onda", s, humano(onda(ch, 48, 76, 52), 32, 11))
    por("harpa", "Onda", s, humano(onda(ch, 55, 84, 46, 4, "sobe"), 28, 10))
    por("hum", "Hum", s, humano(
        ORQ.bloco(ch, Q, 1, 4, n_vozes=3, low=50, high=72, velocity=52), 42, 10))
    por("celos", "Pedal", s, humano(ORQ.pedal(ch, Q, 1, 4, "violoncelo", 60), 32))
    por("baixos", "Pedal", s, humano(
        ORQ.pedal(ch, Q, 1, 4, "contrabaixo", 62, oitava_dobrada=False), 36))
    por("flauta", "Ar", s, humano(mel(CONTRA_MARE, velocity=54, vezes=2), 30))

    # ---- preparação das entradas ---------------------------------------
    # As duas trocas de fase e os dois clímaxes. Riser terminando no downbeat,
    # impacto começando nele: é o que faz a fase CHEGAR em vez de só começar.
    for alvo, forca in (("SINOS", 112), ("DESPERTAR", 122), ("TUTTI", 120),
                        ("CANTO_FINAL", 110)):
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

    # o sub drop marca as duas quedas — o olho da tempestade e a quebra-mar
    for alvo in ("OLHO", "QUEBRA_MAR"):
        pat = p.new_pattern(f"Sub drop -> {alvo}", bars=2)
        pat.notes.append(Note(0, COMPASSO, N("E2"), 116))
        p.place(pat, t["subdrop"], INICIO[alvo])

    return p


def regioes() -> list[jogo.Regiao]:
    """As três fases, como a engine as enxerga.

    O `loop_inicio` de cada fase pula a primeira seção: ela é a entrada, toca
    uma vez, e a repetição acontece só do corpo em diante. É o padrão de
    trilha de jogo — a fase pode durar dez segundos ou dez minutos, e nenhum
    dos dois soa errado.
    """
    out = []
    for nome, secoes, desc, nivel, proxima in FASES:
        ini = INICIO[secoes[0]]
        fim = INICIO[secoes[-1]] + dict((n, k) for n, k, _ in SECOES)[secoes[-1]]
        out.append(jogo.Regiao(
            nome=nome, inicio_compasso=ini, compassos=fim - ini,
            loop_inicio_compasso=INICIO[secoes[1]], nivel=nivel,
            proxima=proxima, descricao=desc))
    return out


# --------------------------------------------------------------------------
def main() -> None:
    out = Path(__file__).parent
    ref_path = sys.argv[1] if len(sys.argv) > 1 else None

    print("Banco de instrumentos...")
    sf2 = L.ensure("generaluser_gs",
                   lambda g, tt: print(f"  {g/1e6:5.1f} MB", end="\r", flush=True))
    print(f"  {SF.open_soundfont(sf2).name}\n")

    perfil_alvo, alvo_lufs = None, -14.0
    if ref_path:
        from chipforge.analysis import reference as REF
        print(f"Medindo a referência: {Path(ref_path).name}")
        r = REF.analisar(ref_path)
        perfil_alvo, alvo_lufs = r.perfil_alvo(), r.alvo_de_loudness()
        print(f"  alvo: {alvo_lufs:.1f} LUFS · {perfil_alvo}\n")

    p = build(sf2)
    if perfil_alvo:
        p.master_alvo = perfil_alvo

    dur = p.ticks_to_seconds(p.length_ticks)
    n_notas = sum(len(x.notes) for x in p.patterns.values())
    print(f"{p.name} — {TOTAL} compassos, {int(dur//60)}:{int(dur%60):02d}")
    print(f"{len(p.tracks)} tracks · {len(p.patterns)} patterns · {n_notas} notas")
    print(f"mapa de tempo: {len(p.tempo_map)} marcas "
          f"({min(b for _, b in p.tempo_map):.0f}–"
          f"{max(b for _, b in p.tempo_map):.0f} BPM)\n")

    regs = regioes()
    print(jogo.relatorio_pt(jogo.manifesto(p, regs)))

    secoes = [(n, INICIO[n] * 4.0, (INICIO[n] + k) * 4.0) for n, k, _ in SECOES]
    reps = H.analyze_sections(p.harmony_events(), secoes, beats_per_chord=4.0)
    print()
    print(H.sections_table_pt(reps))

    c = INT.calcular(p)
    print("\nIntensidade (o que o Palco recebe):")
    print("  " + c.resumo_pt(110))

    print("\nRenderizando 12 minutos (voz, coro, guitarras e sala)...")
    t0 = time.time()
    audio = engine.render_project(p, apply_master=False)
    print(f"  mixagem crua em {time.time()-t0:.0f}s")

    m = MST.masterizar(audio, p.sample_rate, "cinematico",
                       alvo_lufs=alvo_lufs, perfil_espectral=perfil_alvo)
    del audio
    print(f"  master em {time.time()-t0:.0f}s no total\n")
    print(m.relatorio_pt(alvo_lufs))

    if ref_path:
        from chipforge.analysis import reference as REF
        r = REF.analisar(ref_path)
        print("\nDistância para a referência (banda a banda, em dB):")
        for k, v in r.comparar(m.audio, p.sample_rate).items():
            print(f"  {k:12s} {v:+5.1f} dB {'acima' if v > 1.5 else 'abaixo' if v < -1.5 else 'no alvo'}")

    print("\nExportando faixa, stems por fase e manifesto...")
    man = jogo.exportar(p, m.audio, regs, out, "guardia_dos_mares",
                        p.sample_rate, 16)
    p.save(out / "guardia_dos_mares")
    smf.write_midi(smf.project_to_midi(p), out / "guardia_dos_mares")
    c.exportar_json(out / "guardia_dos_mares_intensidade", p.name)

    print(f"  {man['arquivo_completo']}  ({len(m.audio)/p.sample_rate/60:.1f} min)")
    for f in man["fases"]:
        print(f"  {f['arquivo']}  (loop de {f['segundos']['duracao_loop']:.1f}s)")
    print("  guardia_dos_mares_fases.json  ← pontos de troca para a engine")
    print("  guardia_dos_mares.chip / .mid / _intensidade.json")


if __name__ == "__main__":
    main()
