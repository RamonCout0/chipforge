"""
Testes do desenvolvimento tematico, da expressao e do envelope do SoundFont.

Roda com: python3 -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chipforge.audio import sampler as SF
from chipforge.core import expressao as E
from chipforge.core import theory as T
from chipforge.core import variacao as V
from chipforge.core.project import Note

SR = 44100
Q = 480
BAR = Q * 4


def tema() -> list[Note]:
    alturas = [69, 72, 76, 74, 72, 69, 71, 72]
    return [Note(i * Q, Q - 20, p, 100) for i, p in enumerate(alturas)]


# ------------------------------------------------------------- variacao
def test_inverter_espelha_o_desenho():
    t = tema()
    inv = V.inverter(t, escala=(9, "natural_minor"))
    subidas_orig = [b.pitch - a.pitch for a, b in zip(t, t[1:])]
    subidas_inv = [b.pitch - a.pitch for a, b in zip(inv, inv[1:])]
    assert all(o * i <= 0 for o, i in zip(subidas_orig, subidas_inv))


def test_inversao_diatonica_fica_na_escala():
    permitidas = T.scale_pitch_classes(9, "natural_minor")
    for n in V.inverter(tema(), escala=(9, "natural_minor")):
        assert n.pitch % 12 in permitidas


def test_retrogradar_inverte_a_ordem_das_alturas():
    t = tema()
    r = V.retrogradar(t)
    assert [n.pitch for n in r] == [n.pitch for n in reversed(t)]
    assert r[0].start == t[0].start


def test_aumentar_e_diminuir_escalam_o_tempo():
    t = tema()
    grande = V.aumentar(t, 2.0)
    pequeno = V.diminuir(t, 0.5)
    assert grande[-1].start - grande[0].start == \
        pytest.approx((t[-1].start - t[0].start) * 2)
    assert pequeno[-1].start - pequeno[0].start == \
        pytest.approx((t[-1].start - t[0].start) * 0.5)
    assert [n.pitch for n in grande] == [n.pitch for n in t]


def test_fragmentar_recorta_e_rebaseia():
    t = [Note(i * Q, Q - 20, 60 + i, 100) for i in range(8)]
    f = V.fragmentar(t, 1, 1, ticks_por_compasso=BAR)
    assert len(f) == 4
    assert f[0].start == 0
    assert [n.pitch for n in f] == [64, 65, 66, 67]


def test_sequenciar_repete_subindo_e_fica_na_escala():
    frag = V.fragmentar(tema(), 0, 1, ticks_por_compasso=BAR)
    seq = V.sequenciar(frag, [0, 2, 4], BAR, escala=(9, "natural_minor"))
    assert len(seq) == len(frag) * 3
    permitidas = T.scale_pitch_classes(9, "natural_minor")
    assert all(n.pitch % 12 in permitidas for n in seq)
    blocos = [[n.pitch for n in seq if n.start // BAR == b] for b in range(3)]
    assert blocos[1][0] > blocos[0][0] and blocos[2][0] > blocos[1][0]


def test_ornamentar_adiciona_notas_sem_sair_do_tom():
    orn = V.ornamentar(tema(), 9, "natural_minor", densidade=1.0, min_dur=200)
    assert len(orn) > len(tema())
    permitidas = T.scale_pitch_classes(9, "natural_minor")
    assert all(n.pitch % 12 in permitidas for n in orn)


def test_harmonizar_em_tercas_fica_consonante_e_abaixo():
    t = tema()
    terca = V.harmonizar(t, 9, "natural_minor", -2)
    assert len(terca) == len(t)
    for a, b in zip(t, terca):
        assert b.pitch < a.pitch
        assert (a.pitch - b.pitch) in (3, 4)      # terca menor ou maior
    permitidas = T.scale_pitch_classes(9, "natural_minor")
    assert all(n.pitch % 12 in permitidas for n in terca)


def test_contracanto_anda_ao_contrario_e_respeita_a_faixa():
    t = tema()
    acordes = [[57, 60, 64], [53, 57, 60], [60, 64, 67], [52, 56, 59]] * 2
    cc = V.contracanto(t, acordes, ticks_por_compasso=Q,
                       oitava_base=60, minimo=57, maximo=69)
    assert cc
    assert all(57 <= n.pitch <= 69 for n in cc)
    passos = min(len(t), len(cc)) - 1
    contrarios = sum(1 for i in range(passos)
                     if (t[i + 1].pitch - t[i].pitch) *
                        (cc[i + 1].pitch - cc[i].pitch) < 0)
    assert contrarios >= passos * 0.6


def test_variar_recusa_modo_desconhecido():
    with pytest.raises(ValueError):
        V.variar(tema(), "girar")


# ------------------------------------------------------------ expressao
def test_humanizar_tira_do_grid_sem_baguncar():
    t = tema()
    h = E.humanizar(t, tempo_ms=14, velocity=10, ppq=Q, bpm=120, seed=1)
    assert [n.pitch for n in h] == [n.pitch for n in t]
    desvios = [abs(a.start - b.start) for a, b in zip(t, h)]
    assert any(d > 0 for d in desvios)
    assert max(desvios) < Q // 2
    assert all(1 <= n.velocity <= 127 for n in h)


def test_humanizar_com_tendencia_desloca_a_camada_toda():
    t = tema()
    atras = E.humanizar(t, tempo_ms=0.0, velocity=0, ppq=Q, bpm=120,
                        seed=1, tendencia_ms=20.0)
    frente = E.humanizar(t, tempo_ms=0.0, velocity=0, ppq=Q, bpm=120,
                         seed=1, tendencia_ms=-20.0)
    assert all(a.start > f.start for a, f in zip(atras[1:], frente[1:]))


def test_arco_de_sino_tem_pico_no_meio():
    v = [n.velocity for n in E.arco(tema(), "sino", 0.3)]
    meio = len(v) // 2
    assert v[meio] > v[0] and v[meio] > v[-1]


def test_crescendo_e_diminuendo():
    cre = [n.velocity for n in E.arco(tema(), "crescendo", 0.3)]
    dim = [n.velocity for n in E.arco(tema(), "diminuendo", 0.3)]
    assert cre[-1] > cre[0]
    assert dim[-1] < dim[0]


def test_articular_legato_cola_e_staccato_encurta():
    t = tema()
    leg = E.articular(t, "legato", folga=10)
    sta = E.articular(t, "staccato", fator=0.5)
    assert leg[0].dur >= t[0].dur
    assert sta[0].dur < t[0].dur
    for a, b in zip(leg, leg[1:]):
        assert a.start + a.dur <= b.start + 1


def test_respirar_abre_espaco_no_fim_da_frase():
    t = [Note(i * Q, Q, 60, 100) for i in range(8)]
    r = E.respirar(t, a_cada=BAR, folga=100, ppq=Q)
    ultima_do_compasso = r[3]
    assert ultima_do_compasso.dur < Q


def test_rampa_velocity_vai_de_um_valor_a_outro():
    r = E.rampa_velocity(tema(), 60, 120)
    assert r[0].velocity == 60
    assert r[-1].velocity == 120


def test_acentuar_marca_o_tempo_forte():
    t = [Note(i * Q, Q - 20, 60, 90) for i in range(8)]
    a = E.acentuar(t, ppq=Q, fortes=(0.0,), ganho=15, fracos=-5)
    assert a[0].velocity > a[1].velocity


# ------------------------------------------- envelope do SoundFont
@pytest.fixture(scope="module")
def banco():
    from chipforge.midiio import library as L
    return SF.open_soundfont(L.ensure())


def _perfil(inst, dur=3.0):
    s = inst.render_note(60, dur, SR, 110)
    janelas = [float(np.sqrt((s[int(t * SR):int(t * SR) + SR // 4] ** 2).mean()))
               for t in (0.05, 1.0, 2.0)]
    topo = max(janelas) or 1.0
    return [j / topo for j in janelas]


@pytest.mark.parametrize("nome", ["Stereo Grand", "Marimba", "Music Box"])
def test_instrumento_percutido_decai(banco, nome):
    """Piano, marimba e caixinha PRECISAM morrer sozinhos. Enquanto o sampler
    ignorava o envelope do banco, todos sustentavam — e por isso soavam
    sinteticos."""
    hits = banco.find(nome)
    assert hits, nome
    p = _perfil(SF.SamplerInstrument(banco, hits[0]))
    assert p[1] < p[0] * 0.75
    assert p[2] < p[0] * 0.4


@pytest.mark.parametrize("nome", ["Violin", "Flute"])
def test_instrumento_de_sopro_ou_arco_sustenta(banco, nome):
    hits = banco.find(nome)
    assert hits, nome
    p = _perfil(SF.SamplerInstrument(banco, hits[0]))
    assert p[2] > 0.6


def test_zona_traz_envelope_do_banco(banco):
    piano = banco.find("Stereo Grand")[0]
    z = [x for x in piano.zones if x.key_lo <= 60 <= x.key_hi][0]
    assert z.decay_s > 1.0            # piano tem decay longo
    assert z.sustain < 0.05           # e sustain praticamente zero
    violino = banco.find("Violin")[0]
    zv = [x for x in violino.zones if x.key_lo <= 60 <= x.key_hi][0]
    assert zv.sustain > 0.5           # arco segura


def test_envelope_manual_ainda_funciona(banco):
    """Quem mexer no ADSR na interface tem de continuar mandando."""
    from chipforge.audio.envelope import ADSR
    piano = banco.find("Stereo Grand")[0]
    inst = SF.SamplerInstrument(banco, piano, env_do_banco=False,
                                env=ADSR(0.01, 0.05, 0.95, 0.2))
    p = _perfil(inst)
    assert p[2] > 0.5                 # agora sustenta, porque foi mandado
