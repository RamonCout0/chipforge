"""
"A Hora Marcada" — terror e suspense, ~3:15. Alguém perdido na mata.

═══════════════════════════════════════════════════════════════════════════
PARTITURA
═══════════════════════════════════════════════════════════════════════════

Esta peça foi escrita no papel antes de virar código, e as decisões abaixo são
todas anteriores ao primeiro render.

── AS TRÊS DECISÕES QUE DEFINEM A PEÇA ────────────────────────────────────

**1. O andamento é 60 BPM porque um relógio bate uma vez por segundo.**
A semínima e o ponteiro dos segundos são a mesma coisa. Durante dois minutos o
ouvinte é ensinado, sem perceber, que este pulso é o tempo real. Não é um
detalhe de produção: é a armadilha que a seção V vai usar.

**2. O baixo é a tetracorde cromática descendente — Ré, Dó♯, Dó, Si♭, Lá.**
É a figura que a música europeia usa há quatro séculos para dizer luto, e
funciona porque cada nota desce meio tom: não há escolha, não há desvio, só a
próxima nota mais baixa. É melancolia e inevitabilidade na mesma linha, que é
exatamente o que o roteiro pede.

O baixo tem **cinco compassos**, não quatro. Uma frase ímpar nunca deixa o
ouvinte assentar numa grade — ele conta até quatro, e vem mais um.

    Ré m │ Lá/Dó♯ │ Fá/Dó │ Si♭ │ Lá
     Ré2 │   Dó♯2 │   Dó2 │ Si♭1│ Lá1

**3. O relógio para.** No último compasso o tique-taque simplesmente não vem.
Nenhum acorde final, nenhum golpe: o fim é a ausência do som que estava lá
desde o primeiro segundo. É o que "aguardar pelo fim inevitável" soa quando o
fim chega.

── POR QUE ESTA PEÇA USA POUCOS INSTRUMENTOS ──────────────────────────────

Na peça anterior a nota foi "lotar de instrumento", e estava certa — para
rock. Aqui é o contrário, e de propósito: medo mora no espaço vazio. Quatorze
tracks, quase nunca tocando ao mesmo tempo. O que assusta na seção V não é a
quantidade de som, é o fato de o silêncio ter acabado.

── FORMA · 49 compassos · Ré menor · 60 BPM ───────────────────────────────

  I.   O RELÓGIO      c. 1-4    Só o tique-taque e a mata. Um sino ao longe.
  II.  A MATA         c. 5-14   Piano entra com o baixo cromático. Nenhuma
                                melodia ainda — só o chão se movendo.
  III. A PERGUNTA     c.15-24   O violino apresenta o tema. Duas voltas: a
                                primeira pergunta, a segunda tenta subir e cai.
  IV.  A VOZ          c.25-34   A voz assume o tema, em valores largos. Entra
                                a caixinha de música — alegre e fora de lugar.
  V.   O QUE VEM      c.35-44   Adensa. O relógio ACELERA de 60 para 69 BPM.
                                Clímax por sufocamento, não por volume.
  VI.  A HORA         c.45-49   Tudo cai. O sino bate cinco vezes, cada uma
                                mais fraca. O relógio para no compasso 49.

── O TEMA (violino, 10 compassos = duas voltas do baixo) ──────────────────

  c.1  Lá4  Fá4          sobre Ré m      — a pergunta
  c.2  Mi4  Fá4          sobre Lá/Dó♯
  c.3  Mi4  Dó4          sobre Fá/Dó     — o Mi é a sétima maior de Fá
  c.4  Ré4  Fá4          sobre Si♭
  c.5  Dó♯4 (silêncio)   sobre Lá        — para na sensível, sem resolver
  c.6  Ré4 Fá4 Lá4       sobre Ré m      — a tentativa de subir
  c.7  Si♭4 Lá4          sobre Lá/Dó♯    — o Si♭ é a nona menor: a ferroada
  c.8  Lá4  Sol4         sobre Fá/Dó
  c.9  Fá4  Ré4          sobre Si♭       — e cai
  c.10 Mi4  Dó♯4         sobre Lá        — termina pendurado

A voz canta o mesmo desenho em valores dobrados, na vogal /u/ que abre para
/o/. Quem espera não tem pressa.

    python examples/a_hora_marcada.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from chipforge.audio import cinema as CIN
from chipforge.audio import engine
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
BPM = 60                       # uma semínima = um segundo = um tique


def ac(raiz: str, qual: str = "min", oit: int = 3) -> list[int]:
    return [N(f"{raiz}{oit}") + i for i in T.CHORDS[qual]]


# --------------------------------------------------------------------------
# O baixo obstinado: cinco compassos, um acorde por compasso.
# --------------------------------------------------------------------------
GROUND = [ac("D", "min"), ac("A", "maj"), ac("F", "maj"),
          ac("A#", "maj"), ac("A", "maj")]

# A linha do baixo é escrita à parte porque ela NÃO é a fundamental dos
# acordes: em Lá/Dó♯ e Fá/Dó o baixo está na terça e na quinta. É a inversão
# que mantém o movimento cromático — e o movimento cromático é a peça.
BAIXO_CROMATICO = [
    ("D2", 0, 0.0, 4.0), ("C#2", 1, 0.0, 4.0), ("C2", 2, 0.0, 4.0),
    ("A#1", 3, 0.0, 4.0), ("A1", 4, 0.0, 4.0),
]


def voltas(n: int) -> list[list[int]]:
    return GROUND * n


SECOES = [
    ("RELOGIO", 4, [GROUND[0]] * 4),
    ("MATA", 10, voltas(2)),
    ("PERGUNTA", 10, voltas(2)),
    ("VOZ", 10, voltas(2)),
    ("VEM", 10, voltas(2)),
    ("HORA", 5, voltas(1)),
]

INICIO: dict[str, int] = {}
_b = 0
for _n, _k, _ in SECOES:
    INICIO[_n] = _b
    _b += _k
TOTAL = _b

# Dinâmica: a peça vai de quase inaudível a mezzo-forte. Nunca a fortíssimo —
# terror que grita deixa de ser terror e vira susto.
NIVEIS = {"RELOGIO": 0.16, "MATA": 0.30, "PERGUNTA": 0.44, "VOZ": 0.56,
          "VEM": 0.92, "HORA": 0.34}

# --------------------------------------------------------------------------
# Temas
# --------------------------------------------------------------------------
TEMA = [
    ("A4", 0, 0.0, 2.0), ("F4", 0, 2.0, 2.0),
    ("E4", 1, 0.0, 3.0), ("F4", 1, 3.0, 1.0),
    ("E4", 2, 0.0, 2.0), ("C4", 2, 2.0, 2.0),
    ("D4", 3, 0.0, 2.0), ("F4", 3, 2.0, 2.0),
    ("C#4", 4, 0.0, 3.0),
    ("D4", 5, 0.0, 1.0), ("F4", 5, 1.0, 1.0), ("A4", 5, 2.0, 2.0),
    ("A#4", 6, 0.0, 3.0), ("A4", 6, 3.0, 1.0),
    ("A4", 7, 0.0, 2.0), ("G4", 7, 2.0, 2.0),
    ("F4", 8, 0.0, 2.0), ("D4", 8, 2.0, 2.0),
    ("E4", 9, 0.0, 2.0), ("C#4", 9, 2.0, 2.0),
]

# A voz: o mesmo desenho, em valores largos. Quem espera não tem pressa.
VOZ_TEMA = [
    ("A4", 0, 0.0, 4.0),
    ("E4", 1, 0.0, 4.0),
    ("E4", 2, 0.0, 2.0), ("C4", 2, 2.0, 2.0),
    ("D4", 3, 0.0, 4.0),
    ("C#4", 4, 0.0, 4.0),
    ("D4", 5, 0.0, 2.0), ("F4", 5, 2.0, 2.0),
    ("A4", 6, 0.0, 4.0),
    ("G4", 7, 0.0, 4.0),
    ("F4", 8, 0.0, 2.0), ("D4", 8, 2.0, 2.0),
    ("E4", 9, 0.0, 4.0),
]

# A caixinha de música. Quatro notas, agudas, simples demais para o que está
# acontecendo — e é isso que assusta.
CAIXINHA = [
    ("D6", 0, 0.0, 2.0), ("F6", 0, 2.0, 1.0), ("E6", 0, 3.0, 1.0),
    ("D6", 1, 0.0, 3.0),
]


def mel(linha, **kw) -> list[Note]:
    return ORQ.escrever(linha, Q, 4, **kw)


def relogio_notas(de_compasso: int, ate_compasso: int,
                  parar_no_tempo: int | None = None) -> list[Note]:
    """Uma batida por semínima, alternando tique e taque.

    As duas alturas não são decoração: o escape de um relógio passa por dois
    dentes diferentes, e por isso tique e taque não são o mesmo som. Escrever
    a mesma altura duas vezes por segundo produz um metrônomo, não um relógio.
    """
    out: list[Note] = []
    for c in range(de_compasso, ate_compasso):
        for b in range(4):
            if parar_no_tempo is not None and c == ate_compasso - 1 \
                    and b >= parar_no_tempo:
                break
            alt = N("C5") if b % 2 == 0 else N("A#4")
            # o relógio não é perfeito: alguns cents de variação em cada
            # batida é o que separa um mecanismo de um arquivo colado
            v = 74 + (3 if b == 0 else 0) + ((c * 7 + b * 13) % 5) - 2
            out.append(Note(c * COMPASSO + b * Q, Q // 2, alt, v))
    return out


# --------------------------------------------------------------------------
def build(sf2: Path) -> Project:
    sf = SF.open_soundfont(sf2)

    def gm(nome: str, **kw):
        return SF.instrument(sf, nome, **kw)

    p = Project(name="A Hora Marcada", bpm=BPM, ppq=Q, bit_depth=24,
                master_gain=0.9, key_root=N("D4") % 12, scale="natural_minor")

    # Caverna: grande, escura e com o agudo morto. É a sala mais parecida com
    # uma mata à noite que o catálogo tem — e o agudo morto importa mais que o
    # tamanho, porque é a falta de brilho que faz um lugar soar fechado.
    p.master_reverb = {"sala": "caverna", "tamanho": 0.95, "wet": 0.85,
                       "corte_grave": 130.0, "corte_agudo": 6500.0}
    p.master_perfil = "cinematico"

    # ---- mapa de tempo -------------------------------------------------
    # Sessenta BPM do começo ao fim, EXCETO na seção V. Depois de dois minutos
    # o ouvinte já tomou este pulso como o tempo real; acelerá-lo é trair essa
    # certeza, e é isso que produz o pânico — não o volume.
    mapa = DIN.MapaDeTempo(ppq=Q, bpm_base=BPM)
    mapa.marcar(0, BPM)
    mapa.marcar(INICIO["VEM"] * COMPASSO, BPM)
    mapa.rampa(INICIO["VEM"] * COMPASSO, (INICIO["VEM"] + 8) * COMPASSO,
               BPM, BPM * 1.15)
    # e no fim ele desacelera abaixo do normal: o mecanismo está morrendo
    mapa.rampa(INICIO["HORA"] * COMPASSO, TOTAL * COMPASSO, BPM * 1.15, BPM * 0.72)
    p.tempo_map = mapa.to_list()

    t: dict[str, object] = {}

    # ==================================================== O RELÓGIO E A MATA
    # O relógio é a única coisa PERTO. Todo o resto está longe. Esse contraste
    # de distância — um objeto no seu ouvido, o mundo inteiro a trinta metros —
    # é o que coloca o ouvinte dentro da cena em vez de na plateia.
    t["relogio"] = p.new_track("Relógio", CIN.instrumento("Relogio grande",
                                                          pan=-0.12))
    t["relogio"].reverb_send = 0.06
    t["relogio"].volume = 0.82
    # Realce estreito na banda do clique. O par deste realce sao os cortes na
    # mesma frequencia no vento, no tremolo e no coro, logo abaixo: quando dois
    # sons disputam a mesma faixa, levantar um so nao resolve — o que resolve e
    # abrir espaco no outro.
    t["relogio"].inserts = [{"tipo": "eq", "modo": "highpass", "freq": 160.0,
                             "q": 0.7},
                            {"tipo": "eq", "freq": 2200.0, "ganho_db": 5.0,
                             "q": 1.4}]

    t["mata"] = p.new_track("Mata", CIN.instrumento("Mata", pan=0.1))
    t["mata"].reverb_send = 0.55
    t["mata"].volume = 0.20
    t["mata"].inserts = [{"tipo": "eq", "freq": 2200.0, "ganho_db": -3.5,
                          "q": 1.0}]

    # Centrado em 3,4 kHz, e nao em 1,6: o vento agudo estava exatamente em
    # cima do clique do relogio, e mediu 18 dB de mascaramento na secao V.
    t["vento"] = p.new_track("Vento", CIN.instrumento(
        "Vento agudo", pan=-0.35, centro_hz=3400.0, largura=1.3))
    t["vento"].reverb_send = 0.7
    t["vento"].volume = 0.13
    t["vento"].inserts = [{"tipo": "eq", "freq": 2200.0, "ganho_db": -5.0,
                           "q": 1.2}]

    # ==================================================== PIANO E CORDAS
    t["piano"] = p.new_track("Piano", gm("Stereo Grand", gain=0.85))
    t["piano"].reverb_send = 0.46
    t["piano"].volume = 1.0
    t["piano"].pan = -0.16
    # O realce em 1,8 kHz e o que faz o martelo aparecer. Sem ele o piano fica
    # so com o corpo, e corpo de piano ocupa a mesma faixa do violoncelo e do
    # baixo — some. Medido antes: 20 dB abaixo da mixagem.
    t["piano"].inserts = [{"tipo": "eq", "modo": "highpass", "freq": 60.0, "q": 0.7},
                          {"tipo": "eq", "freq": 1800.0, "ganho_db": 3.0, "q": 0.8},
                          {"tipo": "eq", "freq": 4600.0, "ganho_db": -2.0, "q": 0.8}]

    # O violino solo. O vibrato entra 0,4 s depois do ataque, como um
    # violinista instala: primeiro o arco pega, depois a mão esquerda treme.
    t["violino"] = p.new_track("Violino", gm("Violin", gain=0.9))
    t["violino"].instrument.vibrato_depth_cents = 26.0
    t["violino"].instrument.vibrato_hz = 5.0
    t["violino"].instrument.vibrato_atraso_s = 0.40
    t["violino"].reverb_send = 0.52
    t["violino"].volume = 0.72
    t["violino"].pan = 0.22

    t["viola"] = p.new_track("Viola", gm("Viola", gain=0.85))
    t["viola"].instrument.vibrato_depth_cents = 18.0
    t["viola"].reverb_send = 0.5
    t["viola"].volume = 0.34
    t["viola"].pan = -0.3

    t["celo"] = p.new_track("Violoncelo", gm("Cello", gain=0.9))
    t["celo"].instrument.vibrato_depth_cents = 16.0
    t["celo"].inserts = [{"tipo": "eq", "freq": 1800.0, "ganho_db": -3.0,
                          "q": 1.1}]
    t["celo"].reverb_send = 0.44
    t["celo"].volume = 0.42
    t["celo"].pan = 0.28

    t["tremolo"] = p.new_track("Cordas (tremolo)", gm("Stereo Strings Trem",
                                                      gain=0.85))
    t["tremolo"].section = {"vozes": 2, "detune": 11, "spread_ms": 24, "pan": 0.7}
    t["tremolo"].reverb_send = 0.6
    t["tremolo"].volume = 0.30
    t["tremolo"].inserts = [{"tipo": "eq", "freq": 2200.0, "ganho_db": -4.0,
                             "q": 1.2},
                            {"tipo": "eq", "freq": 1800.0, "ganho_db": -3.0,
                             "q": 1.2}]

    t["baixos"] = p.new_track("Contrabaixos", gm("Double Bass", gain=0.95))
    t["baixos"].reverb_send = 0.2
    t["baixos"].volume = 0.44

    # ==================================================== A VOZ
    t["voz"] = p.new_track("A que espera",
                           VOZ.voz("Lamento (u)", pan=0.06))
    t["voz"].volume = 0.44
    t["voz"].reverb_send = 0.62
    t["voz"].inserts = [
        {"tipo": "eq", "modo": "highpass", "freq": 130.0, "q": 0.7},
        {"tipo": "eq", "freq": 2400.0, "ganho_db": 1.5, "q": 0.9},
        {"tipo": "comp", "limiar_db": -22.0, "razao": 2.8, "attack_ms": 18.0,
         "release_ms": 200.0, "makeup_db": 3.0}]

    t["sussurro"] = p.new_track("Sussurro", VOZ.voz("Sussurro", pan=-0.5,
                                                    gain=0.22))
    t["sussurro"].reverb_send = 0.8
    t["sussurro"].volume = 0.26

    t["hum"] = p.new_track("Coro (boca fechada)",
                           VOZ.voz("Coro de boca fechada"))
    t["hum"].section = dict(VOZ.SECAO_CORO)
    t["hum"].reverb_send = 0.85
    t["hum"].volume = 0.22
    t["hum"].inserts = [{"tipo": "eq", "freq": 2000.0, "ganho_db": -4.0,
                         "q": 1.0}]

    # ==================================================== METAL E VIDRO
    t["sinos"] = p.new_track("Sinos", gm("Tubular Bells", gain=0.9))
    t["sinos"].reverb_send = 0.7
    t["sinos"].volume = 0.46
    t["sinos"].pan = 0.38
    t["sinos"].inserts = [{"tipo": "eq", "modo": "highpass", "freq": 170.0,
                           "q": 0.7}]

    t["caixinha"] = p.new_track("Caixinha de música", gm("Music Box", gain=0.85))
    # meio quarto de tom abaixo: a caixinha está velha, e o ouvido percebe que
    # há algo errado antes de conseguir dizer o quê
    t["caixinha"].instrument.detune_cents = -22.0
    t["caixinha"].reverb_send = 0.62
    t["caixinha"].volume = 0.34
    t["caixinha"].pan = -0.42

    t["celesta"] = p.new_track("Celesta", gm("Celeste", gain=0.8))
    t["celesta"].reverb_send = 0.66
    t["celesta"].volume = 0.24
    t["celesta"].pan = 0.45

    t["subdrop"] = p.new_track("Sub drop", CIN.instrumento("Sub drop"))
    t["subdrop"].reverb_send = 0.2
    t["subdrop"].volume = 0.42

    t["swell"] = p.new_track("Swell", CIN.instrumento("Swell invertido"))
    t["swell"].reverb_send = 0.6
    t["swell"].volume = 0.26

    # ---- arco -----------------------------------------------------------
    # O relógio fica FORA da automação: um relógio não fica mais alto porque a
    # música cresceu. Ele é a única coisa constante da peça, e a constância
    # dele é o que faz o resto parecer instável.
    arco = DIN.arco_de_secoes([(n, k) for n, k, _ in SECOES], COMPASSO, NIVEIS)
    for chave, tr in t.items():
        if chave not in ("relogio", "subdrop", "swell"):
            tr.automacao = {"volume": arco.to_list()}

    # O relógio tem a própria curva. Ele não segue o arco da peça — mas também
    # não pode ficar fixo: medido com nível fixo, ele estava 29 dB abaixo da
    # mixagem na seção V, ou seja, tinha sumido exatamente no compasso em que o
    # ouvinte precisa notar que ele acelerou. Aqui ele sobe só o suficiente
    # para atravessar o adensamento, e volta a cair na última seção.
    rel = DIN.Automacao(padrao=1.0)
    rel.ponto(0, 1.0)
    rel.ponto(INICIO["VEM"] * COMPASSO, 1.0)
    rel.ponto((INICIO["VEM"] + 6) * COMPASSO, 2.6)
    rel.ponto(INICIO["HORA"] * COMPASSO, 1.35)
    rel.ponto(TOTAL * COMPASSO, 1.0)
    t["relogio"].automacao = {"volume": rel.to_list()}

    # ================================================================
    acordes = {n: c for n, _, c in SECOES}
    ncomp = {n: k for n, k, _ in SECOES}

    def por(track: str, nome: str, secao: str, notas):
        if not notas:
            return
        pat = p.new_pattern(f"{nome} · {secao}", bars=ncomp[secao])
        pat.notes.extend(notas)
        p.place(pat, t[track], INICIO[secao])

    def humano(notas, espalhar=40.0, vel=7):
        return ORQ.humanizar_naipe(notas, Q, espalhar, BPM, vel)

    def baixo(voltas_n: int, vel: int = 62) -> list[Note]:
        return mel(BAIXO_CROMATICO, velocity=vel, vezes=voltas_n, compassos=5)

    def acordes_piano(secao: str, vel: int = 46, n_vozes: int = 3,
                      low: int = 52, high: int = 74) -> list[Note]:
        return ORQ.bloco(acordes[secao], Q, 1, 4, n_vozes=n_vozes,
                         low=low, high=high, velocity=vel)

    # ======================================================== I. O RELÓGIO
    s = "RELOGIO"
    por("mata", "Ambiente", s, [Note(0, COMPASSO * 4, 60, 74)])
    por("vento", "Ar", s, [Note(0, COMPASSO * 4, 60, 66)])
    # um sino, longe, no terceiro compasso — para o ouvinte saber que há algo
    # construído por gente naquela mata
    por("sinos", "Ao longe", s, [Note(COMPASSO * 2, COMPASSO * 2, N("D3"), 52)])

    # ======================================================== II. A MATA
    s = "MATA"
    por("mata", "Ambiente", s, [Note(0, COMPASSO * 10, 60, 78)])
    por("piano", "Baixo", s, humano(baixo(2, 58), 44, 8))
    por("piano", "Acordes", s, humano(acordes_piano(s, 40, 3, 55, 72), 46, 8))
    por("celo", "Baixo", s, humano(baixo(2, 54), 42, 8))
    # o violino ainda não toca melodia: sustenta o Lá, a única nota comum a
    # quase todos os acordes do baixo. É um fio, não uma frase.
    por("violino", "Fio", s, [Note(COMPASSO * 5, COMPASSO * 5, N("A4"), 46)])

    # ======================================================== III. A PERGUNTA
    s = "PERGUNTA"
    por("mata", "Ambiente", s, [Note(0, COMPASSO * 10, 60, 72)])
    por("violino", "Tema", s, EXP.arco(mel(TEMA, velocity=74), "sino", 0.26))
    por("piano", "Baixo", s, humano(baixo(2, 62), 42, 8))
    por("piano", "Acordes", s, humano(acordes_piano(s, 44, 3, 55, 72), 44, 8))
    por("celo", "Baixo", s, humano(baixo(2, 58), 40, 8))
    por("viola", "Contra", s, humano(
        ORQ.bloco(acordes[s], Q, 1, 4, n_vozes=1, low=48, high=64,
                  velocity=42), 44, 7))

    # ======================================================== IV. A VOZ
    s = "VOZ"
    por("mata", "Ambiente", s, [Note(0, COMPASSO * 10, 60, 66)])
    por("voz", "Espera", s, EXP.arco(mel(VOZ_TEMA, velocity=78), "sino", 0.22))
    por("violino", "Contra", s, EXP.arco(
        mel(TEMA, velocity=84)[10:], "sino", 0.24))
    por("piano", "Baixo", s, humano(baixo(2, 60), 42, 8))
    por("piano", "Acordes", s, humano(acordes_piano(s, 42, 3, 55, 72), 44, 8))
    por("celo", "Baixo", s, humano(baixo(2, 60), 40, 8))
    por("caixinha", "Volta", s, humano(
        mel(CAIXINHA, velocity=56, vezes=3, compassos=2), 60, 9))
    por("sinos", "Marca", s, [Note(COMPASSO * c, COMPASSO * 2, N("D3"), 58)
                              for c in (0, 5)])
    por("sussurro", "Perto", s, [Note(COMPASSO * 7, COMPASSO * 2, N("D4"), 60)])

    # ======================================================== V. O QUE VEM
    s = "VEM"
    por("mata", "Ambiente", s, [Note(0, COMPASSO * 10, 60, 92)])
    por("vento", "Ar", s, [Note(0, COMPASSO * 10, 60, 88)])
    por("tremolo", "Tensão", s, humano(
        ORQ.tremolo(acordes[s], Q, 1, 4, low=62, high=81, velocity=46,
                    divisao=8), 14, 6))
    por("violino", "Tema", s, EXP.arco(mel(TEMA, velocity=96), "crescendo", 0.20))
    por("voz", "Espera", s, EXP.arco(mel(VOZ_TEMA, velocity=92), "crescendo", 0.18))
    por("hum", "Coro", s, humano(
        [Note(n.start + COMPASSO * 5, n.dur, n.pitch, n.velocity)
         for n in ORQ.bloco(acordes[s][:5], Q, 1, 4, n_vozes=3, low=45,
                            high=62, velocity=52)], 48, 10))
    por("piano", "Baixo", s, humano(baixo(2, 82), 36, 8))
    por("piano", "Acordes", s, humano(acordes_piano(s, 62, 4, 52, 76), 40, 8))
    por("celo", "Baixo", s, humano(baixo(2, 84), 36, 8))
    por("baixos", "Baixo", s, humano(baixo(2, 86), 38, 8))
    # Viola e celesta SAEM daqui. Depois de cinco rodadas de EQ e fader eu
    # tinha ganho quatro decibeis; o que estava enterrando o relogio e o piano
    # nao era o nivel deles, era o numero de coisas tocando ao mesmo tempo.
    # A partitura dizia "medo mora no espaco vazio" e o arranjo tinha doze
    # camadas simultaneas. Tirar duas resolve o que nenhum filtro resolvia.
    # o swell termina exatamente no compasso 5 da seção — o meio, não o fim.
    # Preparar o clímax e depois continuar subindo é mais cruel que resolvê-lo.
    por("swell", "Sobe", s, [Note(COMPASSO * 3, COMPASSO * 2, N("D4"), 86)])

    # ======================================================== VI. A HORA
    s = "HORA"
    por("mata", "Ambiente", s, [Note(0, COMPASSO * 5, 60, 70)])
    # o sub drop é o instante em que a personagem entende
    por("subdrop", "Queda", s, [Note(0, COMPASSO, N("D2"), 104)])
    por("piano", "Baixo", s, humano(baixo(1, 54), 46, 8))
    por("celo", "Baixo", s, humano(baixo(1, 50), 44, 8))
    # cinco badaladas, cada uma mais fraca. A última fica sozinha.
    por("sinos", "A hora", s, [Note(COMPASSO * c, COMPASSO, N("D3"), v)
                               for c, v in zip(range(5), (92, 80, 68, 56, 44))])
    por("voz", "Fim", s, EXP.arco(
        mel(VOZ_TEMA, velocity=64)[:5], "diminuendo", 0.28))
    por("caixinha", "Resto", s, humano(
        mel(CAIXINHA, velocity=38, compassos=2), 80, 9))

    # ---- o relógio, contínuo, do primeiro compasso até parar --------------
    # Ele não pertence a nenhuma seção: atravessa a peça inteira. E no último
    # compasso ele simplesmente não vem — o fim da peça é a ausência do som
    # que estava lá desde o primeiro segundo.
    pat = p.new_pattern("Tique-taque", bars=TOTAL)
    pat.notes.extend(relogio_notas(0, TOTAL, parar_no_tempo=2))
    p.place(pat, t["relogio"], 0)

    return p


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

    print("\nRenderizando...")
    t0 = time.time()
    audio = engine.render_project(p, apply_master=False)
    print(f"  mixagem crua em {time.time()-t0:.0f}s")

    # Alvo baixo de propósito. Uma peça que vai do quase inaudível ao
    # mezzo-forte precisa da faixa dinâmica intacta; empurrá-la para -14 LUFS
    # levantaria o silêncio junto e apagaria a peça.
    m = MST.masterizar(audio, p.sample_rate, "cinematico", alvo_lufs=-18.0)
    del audio
    print(f"  master em {time.time()-t0:.0f}s no total\n")
    print(m.relatorio_pt(-18.0))

    wav = engine.export_wav(m.audio, out / "a_hora_marcada", p.sample_rate, 16)
    p.save(out / "a_hora_marcada")
    smf.write_midi(smf.project_to_midi(p), out / "a_hora_marcada")
    c.exportar_json(out / "a_hora_marcada_intensidade", p.name)
    print(f"\nWAV : {wav}  ({len(m.audio)/p.sample_rate/60:.2f} min)")


if __name__ == "__main__":
    main()
