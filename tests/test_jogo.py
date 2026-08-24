"""
Testes da exportacao para engine de jogo: regioes, costura de loop, stems e
manifesto.

O que estes testes protegem: uma trilha de raid so serve se a repeticao for
inaudivel e se os pontos de troca estiverem certos em AMOSTRAS, nao em
milissegundos arredondados. Erro de meio compasso aqui nao aparece no editor —
aparece no jogador, uma vez por volta, para sempre.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chipforge.audio import jogo
from chipforge.audio.instrument import factory_by_name
from chipforge.core.project import Note, Project

SR = 44100


def _projeto(tempo_map=None) -> Project:
    p = Project(name="Raid", bpm=120.0, bit_depth=24)
    if tempo_map:
        p.tempo_map = tempo_map
    tr = p.new_track("lead", factory_by_name("NES Lead 25%"))
    pat = p.new_pattern("p", bars=4)
    for i in range(16):
        pat.notes.append(Note(i * 480, 400, 60 + (i % 5), 100))
    for c in range(0, 12, 4):
        p.place(pat, tr, c)
    return p


REGIOES = [
    jogo.Regiao("FASE1", 0, 4, nivel=1, proxima="FASE2", descricao="calma"),
    jogo.Regiao("FASE2", 4, 8, loop_inicio_compasso=6, nivel=4, proxima="FASE1"),
]


# ==========================================================================
# Costura do loop
# ==========================================================================
def test_costura_traz_a_cauda_de_volta():
    """A ideia inteira do modulo. O trecho cortado termina seco; a cauda que
    vinha depois dele e somada sobre o comeco, para que na segunda volta ela
    chegue como teria chegado se a musica nao tivesse sido cortada."""
    n = 1000
    x = np.zeros((n, 2), dtype=np.float32)
    x[400:600] = 1.0          # o corpo
    x[600:700] = 0.5          # a cauda que vem depois do corte

    corpo = jogo.costurar_loop(x, 400, 600, cauda=100)
    assert len(corpo) == 200
    # as primeiras 100 amostras receberam a cauda por cima
    assert np.allclose(corpo[:100], 1.5)
    assert np.allclose(corpo[100:], 1.0)


def test_costura_sem_cauda_nao_muda_nada():
    x = np.linspace(0, 1, 500, dtype=np.float32)[:, None] * np.ones((1, 2), np.float32)
    assert np.allclose(jogo.costurar_loop(x, 100, 300, cauda=0), x[100:300])


def test_costura_respeita_os_limites_do_buffer():
    """Pedir cauda que nao existe nao pode estourar o indice: o ultimo loop da
    peca sempre pede mais do que ha depois dele."""
    x = np.ones((300, 2), dtype=np.float32)
    corpo = jogo.costurar_loop(x, 100, 300, cauda=5000)
    assert len(corpo) == 200
    assert np.isfinite(corpo).all()


def test_janela_de_cruzamento_zera_as_pontas():
    x = np.ones((1000, 2), dtype=np.float32)
    corpo = jogo.costurar_loop(x, 100, 600, cauda=0, janela=50)
    assert corpo[0].max() < 0.05
    assert corpo[-1].max() < 0.05
    assert corpo[250].min() > 0.9


# ==========================================================================
# Regioes
# ==========================================================================
def test_loop_comeca_no_inicio_quando_nao_declarado():
    r = jogo.Regiao("X", 8, 16)
    assert r.loop_de == 8
    assert r.fim_compasso == 24


def test_loop_pula_a_entrada_quando_declarado():
    """O padrao de trilha de jogo: a introducao toca uma vez e a repeticao
    acontece so do corpo em diante."""
    r = jogo.Regiao("X", 8, 16, loop_inicio_compasso=12)
    assert r.loop_de == 12


# ==========================================================================
# Manifesto
# ==========================================================================
def test_manifesto_concorda_consigo_mesmo():
    """Segundos e amostras tem que descrever o MESMO instante: a engine usa
    amostras no callback de audio e segundos na logica de jogo, e as duas
    precisam apontar para o mesmo lugar."""
    p = _projeto()
    m = jogo.manifesto(p, REGIOES, SR)
    for f in m["fases"]:
        for chave in ("inicio", "fim", "loop_inicio"):
            assert f["segundos"][chave] == pytest.approx(
                f["amostras"][chave] / SR, abs=1e-3)
        assert (f["amostras"]["duracao_loop"]
                == f["amostras"]["fim"] - f["amostras"]["loop_inicio"])


def test_manifesto_respeita_o_mapa_de_tempo():
    """Com accelerando, o compasso 8 nao esta em 8 * 4 * 60 / bpm. So o projeto
    sabe onde ele esta — e o manifesto precisa perguntar a ele."""
    reto = jogo.manifesto(_projeto(), REGIOES, SR)
    veloz = jogo.manifesto(
        _projeto(tempo_map=[[0, 120.0], [480 * 4 * 12, 240.0]]), REGIOES, SR)
    assert veloz["fases"][1]["amostras"]["fim"] < \
        reto["fases"][1]["amostras"]["fim"] * 0.85
    assert veloz["tem_mapa_de_tempo"] is True


def test_bordas_de_compasso_sao_crescentes():
    """A engine so pode trocar de fase numa borda de compasso. Se a lista nao
    for monotona, ela troca para tras."""
    b = jogo.manifesto(_projeto(tempo_map=[[0, 90.0], [9600, 160.0]]),
                       REGIOES, SR)["bordas_de_compasso_s"]
    assert all(y > x for x, y in zip(b, b[1:]))


def test_fases_encadeiam_ate_o_comeco():
    m = jogo.manifesto(_projeto(), REGIOES, SR)
    nomes = {f["nome"] for f in m["fases"]}
    for f in m["fases"]:
        assert f["proxima"] in nomes, "fase apontando para lugar nenhum"


# ==========================================================================
# Fatiamento e exportacao
# ==========================================================================
def test_fatiar_da_um_stem_por_fase_com_a_duracao_do_loop():
    p = _projeto()
    audio = np.random.default_rng(2).normal(0, 0.1, (SR * 30, 2)).astype(np.float32)
    stems = jogo.fatiar(p, audio, REGIOES, SR, cauda_s=1.0)
    m = jogo.manifesto(p, REGIOES, SR)
    assert set(stems) == {"FASE1", "FASE2"}
    for f in m["fases"]:
        assert len(stems[f["nome"]]) == f["amostras"]["duracao_loop"]


def test_exportar_escreve_faixa_stems_e_manifesto(tmp_path):
    p = _projeto()
    audio = np.random.default_rng(3).normal(0, 0.1, (SR * 30, 2)).astype(np.float32)
    m = jogo.exportar(p, audio, REGIOES, tmp_path, "raid", SR, 16, 1.0)

    assert (tmp_path / "raid.wav").exists()
    assert (tmp_path / "raid_fases.json").exists()
    for f in m["fases"]:
        assert (tmp_path / f["arquivo"]).exists(), f["nome"]

    gravado = json.loads((tmp_path / "raid_fases.json").read_text(encoding="utf-8"))
    assert gravado["arquivo_completo"] == "raid.wav"
    assert len(gravado["fases"]) == 2


def test_exportar_sem_stems_ainda_gera_manifesto(tmp_path):
    p = _projeto()
    audio = np.zeros((SR * 30, 2), dtype=np.float32)
    m = jogo.exportar(p, audio, REGIOES, tmp_path, "raid", SR, 16, stems=False)
    assert (tmp_path / "raid.wav").exists()
    assert all(f["arquivo"] is None for f in m["fases"])


def test_relatorio_lista_todas_as_fases():
    texto = jogo.relatorio_pt(jogo.manifesto(_projeto(), REGIOES, SR))
    for r in REGIOES:
        assert r.nome in texto
