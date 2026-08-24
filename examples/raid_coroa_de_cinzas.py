"""
"Coroa de Cinzas" — tema de raid lendario, 112 compassos, ~3 minutos.

O teste de estresse do Chipforge: 15 tracks, sete secoes, tres modulacoes,
orquestra do banco GM misturada com os canais do NES. Musica original.

    PORTAO      c.  1-8    Re menor    sino, timpano, coro, sem tema ainda
    DESPERTAR   c.  9-24   Re menor    Tema A entra nas trompas e no cello
    FASE I      c. 25-44   Re menor    combate: bateria, guitarra, chip lead
    ENRAGE      c. 45-60   Re menor    napolitana (Mib) e tremolo — a virada
    FASE II     c. 61-84   Fa menor    Tema B, cadencia andaluza, mais rapido
    CLIMAX      c. 85-104  Sol menor   Tema A de volta, um tom acima, tudo junto
    CINZAS      c.105-112  Re menor    resolve e volta pro comeco (loop)

    python3 examples/raid_coroa_de_cinzas.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chipforge.audio import engine
from chipforge.audio import sampler as SF
from chipforge.audio.instrument import factory_by_name
from chipforge.core import generators as G
from chipforge.core import harmony as H
from chipforge.core import theory as T
from chipforge.core.project import Note, Project
from chipforge.midiio import library as L
from chipforge.midiio import smf

BPM = 150
Q = 480
BAR = Q * 4

N = T.name_to_midi
GRAVE = N("D2")            # teto do baixo
ACOMP = N("G3")            # oitava onde vive todo o acompanhamento
MEL_MIN = N("C5")          # a melodia nunca desce abaixo disso


# --------------------------------------------------------------------------
# Acordes
# --------------------------------------------------------------------------
def ac(raiz: str, qualidade: str = "min", oitava: int = 4) -> list[int]:
    r = N(f"{raiz}{oitava}")
    return [r + i for i in T.CHORDS[qualidade]]


def prog(seq, repeticoes=1):
    return [ac(*c) for c in seq] * repeticoes


SECOES = [
    # nome,        compassos, acordes (um por compasso)
    ("PORTAO",   8,  prog([("D", "min"), ("D", "min"), ("A#", "maj"), ("A#", "maj"),
                           ("D", "min"), ("D", "min"), ("A", "maj"), ("A", "maj")])),
    ("DESPERTAR", 16, prog([("D", "min"), ("A#", "maj"), ("F", "maj"), ("C", "maj")], 4)),
    # o dominante entra como power chord: a terca maior (Do#) fica por conta
    # dos metais e da melodia. Do# no pad brigava de nona menor com o Re do tema
    ("FASE I",   20, prog([("D", "min"), ("A#", "maj"), ("G", "min"), ("A", "5")], 5)),
    ("ENRAGE",   16, prog([("D", "min"), ("D#", "maj"), ("D", "min"), ("A", "5")], 4)),
    ("FASE II",  24, prog([("F", "min"), ("D#", "maj"), ("C#", "maj"), ("C", "maj")], 6)),
    ("CLIMAX",   20, prog([("G", "min"), ("D#", "maj"), ("A#", "maj"), ("F", "maj")], 5)),
    ("CINZAS",    8, prog([("D", "min"), ("D", "min"), ("G", "min"), ("G", "min"),
                           ("D", "min"), ("D", "min"), ("D", "min"), ("D", "min")])),
]

INICIO = {}
_b = 0
for _nome, _n, _ in SECOES:
    INICIO[_nome] = _b
    _b += _n
TOTAL_COMPASSOS = _b


# --------------------------------------------------------------------------
# Temas — escritos nota a nota
# (nome da nota, compasso dentro da frase, batida, duracao em batidas)
# --------------------------------------------------------------------------
TEMA_A = [
    ("A4", 0, 0.0, 1.5), ("D5", 0, 1.5, 0.5), ("F5", 0, 2.0, 1.0), ("E5", 0, 3.0, 1.0),
    ("D5", 1, 0.0, 2.0), ("A#4", 1, 2.0, 1.0), ("C5", 1, 3.0, 1.0),
    ("A4", 2, 0.0, 1.0), ("C5", 2, 1.0, 1.0), ("F5", 2, 2.0, 1.5), ("E5", 2, 3.5, 0.5),
    ("D5", 3, 0.0, 1.0), ("C5", 3, 1.0, 1.0), ("A4", 3, 2.0, 2.0),
    ("D5", 4, 0.0, 0.5), ("E5", 4, 0.5, 0.5), ("F5", 4, 1.0, 1.0),
    ("A5", 4, 2.0, 1.5), ("G5", 4, 3.5, 0.5),
    ("F5", 5, 0.0, 1.0), ("D5", 5, 1.0, 1.0), ("A#4", 5, 2.0, 2.0),
    ("C5", 6, 0.0, 1.0), ("F5", 6, 1.0, 1.0), ("A5", 6, 2.0, 1.0), ("G5", 6, 3.0, 1.0),
    ("E5", 7, 0.0, 1.0), ("D5", 7, 1.0, 1.0), ("C5", 7, 2.0, 2.0),
]

TEMA_B = [
    ("C5", 0, 0.0, 0.5), ("F5", 0, 0.5, 0.5), ("G#5", 0, 1.0, 1.0),
    ("G5", 0, 2.0, 1.0), ("F5", 0, 3.0, 1.0),
    ("D#5", 1, 0.0, 1.0), ("G5", 1, 1.0, 1.0), ("A#5", 1, 2.0, 2.0),
    ("G#5", 2, 0.0, 1.0), ("F5", 2, 1.0, 1.0), ("C#5", 2, 2.0, 1.0), ("D#5", 2, 3.0, 1.0),
    ("C5", 3, 0.0, 2.0), ("E5", 3, 2.0, 2.0),
    ("F5", 4, 0.0, 0.5), ("G#5", 4, 0.5, 0.5), ("C6", 4, 1.0, 1.0),
    ("A#5", 4, 2.0, 1.0), ("G#5", 4, 3.0, 1.0),
    ("G5", 5, 0.0, 1.0), ("A#5", 5, 1.0, 1.0), ("D#6", 5, 2.0, 2.0),
    ("G#5", 6, 0.0, 1.0), ("C#6", 6, 1.0, 1.0), ("G#5", 6, 2.0, 1.0), ("F5", 6, 3.0, 1.0),
    ("G5", 7, 0.0, 1.5), ("F5", 7, 1.5, 0.5), ("E5", 7, 2.0, 2.0),
]

# ENRAGE nao tem tema: tem um motivo de dois compassos que roi por dentro
MOTIVO_ENRAGE = [
    ("D5", 0, 0.0, 0.5), ("D#5", 0, 0.5, 0.5), ("D5", 0, 1.0, 0.5),
    ("D#5", 0, 1.5, 0.5), ("D5", 0, 2.0, 2.0),
    ("D#5", 1, 0.0, 0.5), ("F5", 1, 0.5, 0.5), ("D#5", 1, 1.0, 0.5),
    ("F5", 1, 1.5, 0.5), ("D#5", 1, 2.0, 2.0),
    ("D5", 2, 0.0, 0.5), ("D#5", 2, 0.5, 0.5), ("D5", 2, 1.0, 0.5),
    ("D#5", 2, 1.5, 0.5), ("D5", 2, 2.0, 2.0),
    ("C#5", 3, 0.0, 1.0), ("D5", 3, 1.0, 0.5), ("C#5", 3, 1.5, 0.5),
    ("E5", 3, 2.0, 1.0), ("C#5", 3, 3.0, 1.0),
]

FIM = [   # CINZAS: as duas primeiras frases do tema, alargadas, e o repouso
    ("A4", 0, 0.0, 2.0), ("D5", 0, 2.0, 2.0),
    ("F5", 1, 0.0, 2.0), ("E5", 1, 2.0, 2.0),
    ("D5", 2, 0.0, 4.0),
    ("A#4", 3, 0.0, 4.0),
    ("A4", 4, 0.0, 4.0),
    ("D5", 6, 0.0, 8.0),
]


def escrever(spec, transpor=0, oitava=0, velocity=104, corte=25,
             repeticoes=1, compassos_por_repeticao=8) -> list[Note]:
    out: list[Note] = []
    for r in range(repeticoes):
        base = r * compassos_por_repeticao * BAR
        for nome, compasso, batida, dur in spec:
            out.append(Note(base + compasso * BAR + int(batida * Q),
                            max(60, int(dur * Q) - corte),
                            N(nome) + transpor + 12 * oitava,
                            velocity))
    return out


# --------------------------------------------------------------------------
# Camadas de acompanhamento — todas presas abaixo da melodia
# --------------------------------------------------------------------------
def triade(ch):
    """So fundamental, terca e quinta.

    Um acorde de setima espremido dentro de UMA oitava sempre cria uma segunda
    entre a setima e a fundamental — no registro do acompanhamento isso nao e
    cor, e sujeira. A setima fica por conta da melodia e dos metais.
    """
    return sorted(ch)[:3]


def pad(acordes, low=ACOMP, sem_terca=False, velocity=72, oitava=0) -> list[Note]:
    out = []
    for i, ch in enumerate(acordes):
        if sem_terca:
            raiz = G.voice_in_octave([min(ch)], low)[0]
            notas = [raiz, raiz - 5]
        else:
            notas = G.voice_in_octave(triade(ch), low)
        for n in notas:
            out.append(Note(i * BAR, BAR - 40, n + 12 * oitava, velocity))
    return out


def ostinato(acordes, rate=2, velocity=84, low=ACOMP) -> list[Note]:
    """Repeticao motora nas cordas: a fundamental pulsando, com a quinta no
    contratempo. E o que da a sensacao de que o chao esta tremendo."""
    step = Q // rate
    out = []
    for i, ch in enumerate(acordes):
        v = G.voice_in_octave(triade(ch), low)
        for j in range(4 * rate):
            nota = v[0] if j % 2 == 0 else v[min(2, len(v) - 1)]
            out.append(Note(i * BAR + j * step, step - 12, nota,
                            velocity + (12 if j % rate == 0 else 0)))
    return out


def baixo(acordes, oitava_grave=GRAVE, estilo="raiz_quinta", velocity=112) -> list[Note]:
    out = []
    for i, ch in enumerate(acordes):
        raiz = G.voice_in_octave([min(ch)], oitava_grave)[0]
        if estilo == "pulso":
            for j in range(8):
                out.append(Note(i * BAR + j * (Q // 2), Q // 2 - 20, raiz, velocity))
        elif estilo == "longo":
            out.append(Note(i * BAR, BAR - 30, raiz, velocity))
        else:
            out.append(Note(i * BAR, Q * 2 - 30, raiz, velocity))
            out.append(Note(i * BAR + Q * 2, Q - 30, raiz, velocity - 8))
            out.append(Note(i * BAR + Q * 3, Q - 30, raiz + 7, velocity - 8))
    return out


def golpes(acordes, batidas=(0.0, 2.5), velocity=110, low=ACOMP) -> list[Note]:
    """Naipe de metais dando pancada — so fundamental e quinta, sem terca,
    porque terca de metal no meio de tudo vira briga com a melodia."""
    out = []
    for i, ch in enumerate(acordes):
        raiz = G.voice_in_octave([min(ch)], low)[0]
        quinta = raiz + 7
        if quinta > N("G4"):          # nunca invade o registro da melodia
            quinta -= 12
        for b in batidas:
            for n in (raiz, quinta):
                out.append(Note(i * BAR + int(b * Q), Q // 2, n, velocity))
    return out


def percussao_unica(compassos, pitch, batidas=(0.0,), velocity=118,
                    dur=Q // 2) -> list[Note]:
    return [Note(i * BAR + int(b * Q), dur, pitch, velocity)
            for i in range(compassos) for b in batidas]


def sino(acordes, velocity=88, dur=Q * 2) -> list[Note]:
    """O sino dobra a FUNDAMENTAL do acorde, em unissono com o pad.

    Sino numa altura fixa parece boa ideia ate a harmonia andar por baixo dele.
    Dobrando a fundamental ele so pode formar intervalo consonante com o
    proprio acorde — e continua sendo a pancada metalica que a seção pede.
    """
    return [Note(i * BAR, dur, G.voice_in_octave([min(ch)], ACOMP)[0], velocity)
            for i, ch in enumerate(acordes)]


def timpano(acordes, velocity=115) -> list[Note]:
    out = []
    for i, ch in enumerate(acordes):
        raiz = G.voice_in_octave([min(ch)], N("D2"))[0]
        out.append(Note(i * BAR, Q, raiz, velocity))
        if i % 2 == 1:
            out.append(Note(i * BAR + Q * 3, Q // 2, raiz, velocity - 20))
    return out


def rufo(acordes, velocity_ini=50, velocity_fim=125) -> list[Note]:
    """Rufo de timpano em crescendo — o aviso de que a coisa vai virar.

    Segue a fundamental de cada acorde. Um pedal fixo em Re parecia boa ideia
    ate a napolitana entrar em Mib: Re e Mib a dois passos no sub-grave nao e
    tensao, e lama.
    """
    out = []
    passos = len(acordes) * 8
    for i, ch in enumerate(acordes):
        raiz = G.voice_in_octave([min(ch)], N("D2"))[0]
        for j in range(8):
            k = i * 8 + j
            v = int(velocity_ini + (velocity_fim - velocity_ini) * k / max(1, passos - 1))
            out.append(Note(i * BAR + j * (Q // 2), Q // 2 - 10, raiz, v))
    return out


# --------------------------------------------------------------------------
def build(sf2: Path) -> Project:
    p = Project("Coroa de Cinzas", bpm=BPM, bit_depth=16)
    p.key_root, p.scale = 2, "natural_minor"
    banco = SF.open_soundfont(sf2)

    def gm(nome, **kw):
        return SF.instrument(banco, nome, **kw)

    # ------------------------------------------------------------ tracks
    tr = {}
    tr["trompas"] = p.new_track("Trompas", gm("French Horns", gain=0.8))
    tr["cordas_ost"] = p.new_track("Cordas ostinato", gm("Stereo Strings Fast", gain=0.6))
    tr["cordas_pad"] = p.new_track("Cordas pad", gm("Stereo Strings Slow", gain=0.55))
    tr["cello"] = p.new_track("Cellos", gm("Cello", gain=0.85))
    tr["coro"] = p.new_track("Coro", gm("Concert Choir", gain=0.6))
    tr["metais"] = p.new_track("Metais", gm("Brass Section", gain=0.7))
    tr["trompete"] = p.new_track("Trompete", gm("Trumpet", gain=0.7))
    tr["sinos"] = p.new_track("Sinos", gm("Tubular Bells", gain=0.7))
    tr["timpani"] = p.new_track("Timpani", gm("Timpani", gain=0.9))
    tr["taiko"] = p.new_track("Taiko", gm("Taiko Drum", gain=0.85))
    # o taiko e percussao, nao harmonia: fica de fora da analise harmonica
    tr["taiko"].instrument.is_drum = True
    tr["guitarra"] = p.new_track("Guitarra", gm("Distortion Guitar", gain=0.5))
    tr["chip_lead"] = p.new_track("Chip Lead", factory_by_name("NES Lead 25%"))
    tr["chip_arp"] = p.new_track("Chip Arpejo", factory_by_name("Pluck Harp"))
    tr["chip_bass"] = p.new_track("Chip Baixo", factory_by_name("NES Triangle Bass"))
    tr["bateria"] = p.new_track("Bateria chip", factory_by_name("Kick"))
    tr["bateria"].instrument.is_drum = True

    tr["chip_lead"].instrument.vibrato_depth_cents = 20
    tr["cordas_ost"].pan, tr["cordas_pad"].pan = -0.28, 0.22
    tr["trompas"].pan, tr["metais"].pan = -0.16, 0.18
    tr["chip_arp"].pan, tr["guitarra"].pan = 0.36, -0.4
    tr["coro"].volume = 0.5
    tr["chip_arp"].volume = 0.4
    tr["guitarra"].volume = 0.45
    tr["bateria"].volume = 0.7
    tr["cordas_ost"].volume = 0.55

    acordes = {nome: ch for nome, _, ch in SECOES}
    n_comp = {nome: n for nome, n, _ in SECOES}

    def por(nome_track, nome_pattern, secao, notas):
        if not notas:
            return
        pat = p.new_pattern(f"{nome_pattern} · {secao}", bars=n_comp[secao])
        pat.notes.extend(notas)
        p.place(pat, tr[nome_track], INICIO[secao])

    # ------------------------------------------------------- PORTAO (1-8)
    s, ch = "PORTAO", acordes["PORTAO"]
    por("cordas_pad", "Pad", s, pad(ch, sem_terca=True, velocity=58))
    por("coro", "Coro", s, pad(ch, sem_terca=True, velocity=54))
    por("cello", "Cello", s, baixo(ch, N("D2"), "longo", 92))
    por("sinos", "Sino", s, sino(ch, 96, Q * 3))
    por("timpani", "Timpani", s, timpano(ch, 96))

    # ---------------------------------------------------- DESPERTAR (9-24)
    s, ch = "DESPERTAR", acordes["DESPERTAR"]
    por("trompas", "Tema A", s, escrever(TEMA_A, velocity=104, repeticoes=2))
    por("cello", "Cello", s, baixo(ch, N("D2"), "raiz_quinta", 100))
    por("cordas_pad", "Pad", s, pad(ch, velocity=64))
    por("cordas_ost", "Ostinato", s, ostinato(ch[8:], rate=2, velocity=72))
    por("timpani", "Timpani", s, timpano(ch, 100))
    # o chip lead so entra na SEGUNDA metade do Despertar — pattern de 8
    # compassos posicionado no compasso 17
    eco = p.new_pattern("Tema A eco · DESPERTAR", bars=8)
    eco.notes.extend(escrever(TEMA_A, velocity=74))
    p.place(eco, tr["chip_lead"], INICIO[s] + 8)

    # ------------------------------------------------------ FASE I (25-44)
    s, ch = "FASE I", acordes["FASE I"]
    por("chip_lead", "Tema A", s, escrever(TEMA_A, velocity=112, repeticoes=2))
    por("trompas", "Tema A", s, escrever(TEMA_A, velocity=98, repeticoes=2))
    por("cordas_ost", "Ostinato", s, ostinato(ch, rate=4, velocity=80))
    por("cordas_pad", "Pad", s, pad(ch, velocity=62))
    por("cello", "Cello", s, baixo(ch, N("D2"), "pulso", 104))
    por("chip_bass", "Baixo", s, baixo(ch, N("D2"), "pulso", 112))
    por("chip_arp", "Arpejo", s,
        G.arpeggiate([G.voice_in_octave(triade(c), ACOMP) for c in ch], Q,
                     rate=4, mode="updown", octaves=1, velocity=72))
    por("guitarra", "Riff", s, baixo(ch, N("D3"), "pulso", 96))
    por("taiko", "Taiko", s, percussao_unica(20, N("D2"), (0.0, 1.5, 2.0, 3.5), 104, Q // 2))
    por("bateria", "Bateria", s, G.drums(20, Q, style="batalha"))
    por("timpani", "Timpani", s, timpano(ch, 108))

    # ------------------------------------------------------ ENRAGE (45-60)
    s, ch = "ENRAGE", acordes["ENRAGE"]
    por("cordas_ost", "Tremolo", s, ostinato(ch, rate=4, velocity=74))
    por("cordas_pad", "Pad", s, pad(ch, velocity=66))
    por("coro", "Coro", s, pad(ch, sem_terca=True, velocity=70))
    por("trompas", "Motivo", s, escrever(MOTIVO_ENRAGE, velocity=98,
                                         repeticoes=4, compassos_por_repeticao=4))
    por("chip_lead", "Motivo", s, escrever(MOTIVO_ENRAGE, velocity=98,
                                           repeticoes=4, compassos_por_repeticao=4))
    por("cello", "Cello", s, baixo(ch, N("D2"), "pulso", 106))
    por("chip_bass", "Baixo", s, baixo(ch, N("D2"), "pulso", 110))
    por("metais", "Golpes", s, golpes(ch, (0.0,), 106))
    por("bateria", "Bateria", s, G.drums(16, Q, style="corrida"))
    por("timpani", "Rufo", s, rufo(ch))

    # ----------------------------------------------------- FASE II (61-84)
    s, ch = "FASE II", acordes["FASE II"]
    por("chip_lead", "Tema B", s, escrever(TEMA_B, velocity=114, repeticoes=3))
    por("trompete", "Tema B", s, escrever(TEMA_B, velocity=94, repeticoes=3))
    por("cordas_ost", "Ostinato", s, ostinato(ch, rate=4, velocity=84))
    por("cordas_pad", "Pad", s, pad(ch, velocity=62))
    por("cello", "Cello", s, baixo(ch, N("D2"), "pulso", 106))
    por("chip_bass", "Baixo", s, baixo(ch, N("D2"), "pulso", 112))
    por("chip_arp", "Arpejo", s,
        G.arpeggiate([G.voice_in_octave(triade(c), ACOMP) for c in ch], Q,
                     rate=4, mode="up", octaves=1, velocity=74))
    por("guitarra", "Riff", s, baixo(ch, N("D3"), "pulso", 98))
    por("taiko", "Taiko", s, percussao_unica(24, N("F2"), (0.0, 1.5, 2.0, 3.5), 106, Q // 2))
    por("bateria", "Bateria", s, G.drums(24, Q, style="corrida"))

    # ------------------------------------------------------ CLIMAX (85-104)
    s, ch = "CLIMAX", acordes["CLIMAX"]
    sub = 5      # Tema A sobe uma quarta: Re menor -> Sol menor
    por("chip_lead", "Tema A+", s, escrever(TEMA_A, transpor=sub, velocity=118,
                                            repeticoes=2))
    por("trompete", "Tema A+", s, escrever(TEMA_A, transpor=sub, velocity=104,
                                           repeticoes=2))
    por("trompas", "Tema A+", s, escrever(TEMA_A, transpor=sub,
                                          velocity=100, repeticoes=2))
    por("cordas_ost", "Ostinato", s, ostinato(ch, rate=4, velocity=86))
    por("cordas_pad", "Pad", s, pad(ch, velocity=66))
    por("coro", "Coro", s, pad(ch, sem_terca=True, velocity=76))
    por("cello", "Cello", s, baixo(ch, N("D2"), "pulso", 108))
    por("chip_bass", "Baixo", s, baixo(ch, N("D2"), "pulso", 114))
    por("chip_arp", "Arpejo", s,
        G.arpeggiate([G.voice_in_octave(triade(c), ACOMP) for c in ch], Q,
                     rate=4, mode="updown", octaves=1, velocity=76))
    por("metais", "Golpes", s, golpes(ch, (0.0, 2.5), 108))
    por("guitarra", "Riff", s, baixo(ch, N("D3"), "pulso", 100))
    por("taiko", "Taiko", s, percussao_unica(20, N("G2"), (0.0, 1.5, 2.0, 3.5), 110, Q // 2))
    por("bateria", "Bateria", s, G.drums(20, Q, style="batalha"))
    por("sinos", "Sino", s, sino(ch, 84, Q * 2))
    por("timpani", "Timpani", s, timpano(ch, 112))

    # ------------------------------------------------------ CINZAS (105-112)
    s, ch = "CINZAS", acordes["CINZAS"]
    por("trompas", "Fim", s, escrever(FIM, velocity=88))
    por("cordas_pad", "Pad", s, pad(ch, velocity=60))
    por("coro", "Coro", s, pad(ch, sem_terca=True, velocity=62))
    por("cello", "Cello", s, baixo(ch, N("D2"), "longo", 92))
    por("sinos", "Sino", s, sino(ch, 80, Q * 3))
    por("timpani", "Timpani", s, timpano(ch, 92))
    return p


# --------------------------------------------------------------------------
def main():
    out = Path(__file__).parent
    print("Banco de instrumentos...")
    sf2 = L.ensure("generaluser_gs",
                   lambda g, t: print(f"  {g/1e6:5.1f} MB", end="\r", flush=True))
    banco = SF.open_soundfont(sf2)
    print(f"  {banco.name} — {len(banco.presets)} presets\n")

    p = build(sf2)
    dur = p.ticks_to_seconds(p.length_ticks)
    notas = sum(len(x.notes) for x in p.patterns.values())
    print(f"Coroa de Cinzas — {TOTAL_COMPASSOS} compassos, {p.bpm:.0f} BPM, "
          f"{int(dur//60)}:{int(dur%60):02d}")
    print(f"{len(p.tracks)} tracks · {len(p.patterns)} patterns · "
          f"{notas} notas escritas\n")

    secoes = [(nome, INICIO[nome] * 4.0, (INICIO[nome] + n) * 4.0)
              for nome, n, _ in SECOES]
    reps = H.analyze_sections(p.harmony_events(), secoes, beats_per_chord=4.0)
    print(H.sections_table_pt(reps))

    print("\nO que o medidor ainda reclama, por seção:")
    for r in reps:
        graves = [i for i in r.report.issues if i.severity == 2]
        if graves:
            print(f"  {r.name}: {len(graves)} choque(s) sustentado(s) — "
                  f"ex.: {graves[0].message}")
        else:
            print(f"  {r.name}: nenhum choque sustentado")

    print("\nRenderizando...")
    audio = engine.render_project(p)
    wav = engine.export_wav(audio, out / "coroa_de_cinzas", p.sample_rate, 16)
    chip = p.save(out / "coroa_de_cinzas")
    mid = smf.write_midi(smf.project_to_midi(p), out / "coroa_de_cinzas")
    print(f"WAV  : {wav}  ({len(audio)/p.sample_rate:.1f}s)")
    print(f"CHIP : {chip}")
    print(f"MIDI : {mid}")


if __name__ == "__main__":
    main()
