"""
Testes da cadeia de estudio, do mixer e da curva de intensidade.

Roda com: python3 -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chipforge.audio import engine, fx
from chipforge.audio.instrument import factory_by_name
from chipforge.core import intensity as INT
from chipforge.core.project import Note, Project

SR = 44100


def _impulso(n=8192):
    x = np.zeros(n, dtype=np.float32)
    x[0] = 1.0
    return x


def _resposta_db(y, sr, freq):
    h = np.abs(np.fft.rfft(y))
    fr = np.fft.rfftfreq(len(y), 1 / sr)
    i = int(np.argmin(np.abs(fr - freq)))
    return 20 * np.log10(max(float(h[i]), 1e-12))


# ----------------------------------------------------------------- biquads
def test_lowpass_corta_agudo_e_passa_grave():
    y = fx.biquad(_impulso(), SR, "lowpass", 2000.0, 0.9)
    assert _resposta_db(y, SR, 200) > -1.5
    assert _resposta_db(y, SR, 8000) < -20


def test_highpass_corta_grave():
    y = fx.biquad(_impulso(), SR, "highpass", 400.0, 0.9)
    assert _resposta_db(y, SR, 60) < -20
    assert _resposta_db(y, SR, 4000) > -1.5


def test_peak_levanta_so_a_banda_pedida():
    y = fx.biquad(_impulso(), SR, "peak", 1000.0, 1.0, 6.0)
    assert 5.0 < _resposta_db(y, SR, 1000) < 7.0
    assert abs(_resposta_db(y, SR, 100)) < 1.0
    assert abs(_resposta_db(y, SR, 12000)) < 1.0


# ------------------------------------------------------------------ reverb
def _comb_referencia(x, D, f, d):
    b = np.zeros(len(x) + D)
    s = 0.0
    for n in range(len(x)):
        s = (1 - d) * b[n] + d * s
        b[n + D] = x[n] + f * s
    return b[D:len(x) + D]


def _allpass_referencia(x, D, g):
    b = np.zeros(len(x) + D)
    y = np.zeros(len(x))
    for n in range(len(x)):
        b[n + D] = x[n] + g * b[n]
        y[n] = -x[n] + b[n]
    return y


@pytest.mark.parametrize("D,f,d", [(97, 0.8, 0.4), (1116, 0.84, 0.2), (233, 0.7, 0.0)])
def test_comb_em_blocos_bate_com_a_referencia(D, f, d):
    """O comb roda em blocos do tamanho do atraso, por velocidade.
    O resultado tem de ser IDENTICO ao laco amostra a amostra."""
    x = np.random.default_rng(3).standard_normal(4000)
    assert np.allclose(fx._comb_damped(x, D, f, d), _comb_referencia(x, D, f, d),
                       atol=1e-9)


@pytest.mark.parametrize("D,g", [(556, 0.5), (225, 0.6)])
def test_allpass_em_blocos_bate_com_a_referencia(D, g):
    x = np.random.default_rng(4).standard_normal(3000)
    assert np.allclose(fx._allpass(x, D, g), _allpass_referencia(x, D, g),
                       atol=1e-9)


def test_reverb_tem_cauda_que_decai_e_nao_explode():
    x = np.zeros(SR * 3, dtype=np.float32)
    x[100] = 1.0
    r = fx.reverb(x, SR, sala=0.8, damp=0.4)
    assert np.isfinite(r).all()
    janelas = [float(np.sqrt((r[int(t * SR):int(t * SR) + 2205] ** 2).mean()))
               for t in (0.05, 0.3, 0.8, 1.5, 2.5)]
    assert all(janelas[i] > janelas[i + 1] for i in range(len(janelas) - 1))
    assert janelas[0] > 0.0
    assert janelas[-1] < janelas[0] * 0.02


def test_reverb_alarga_o_estereo():
    mono = np.random.default_rng(9).standard_normal((SR, 2)).astype(np.float32) * 0.1
    mono[:, 1] = mono[:, 0]                     # entrada perfeitamente mono
    r = fx.reverb(mono, SR, sala=0.8, largura=1.0)
    correl = float(np.corrcoef(r[:, 0], r[:, 1])[0, 1])
    assert correl < 0.98                        # os canais deixaram de ser iguais


# --------------------------------------------------------------------- amp
def test_amp_gera_harmonicos():
    t = np.arange(SR) / SR
    x = (0.6 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    y = fx.amp(x, SR, ganho=10.0)
    sp = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    fr = np.fft.rfftfreq(len(y), 1 / SR)

    def pico(f):
        m = np.abs(fr - f) < 30
        return float(sp[m].max())

    assert pico(880) > pico(440) * 1e-4          # 2a harmonica (assimetria)
    assert pico(1320) > pico(440) * 1e-3         # 3a harmonica


def test_oversampling_reduz_aliasing():
    """Clipping cria harmonicos acima de Nyquist. Sem oversampling eles voltam
    rebatidos em frequencias que nao pertencem a serie harmonica."""
    t = np.arange(SR) / SR
    f0 = 1760.0
    x = (0.6 * np.sin(2 * np.pi * f0 * t)).astype(np.float32)

    def razao(ov):
        y = fx.amp(x, SR, ganho=12.0, oversample=ov)
        sp = np.abs(np.fft.rfft(y * np.hanning(len(y))))
        fr = np.fft.rfftfreq(len(y), 1 / SR)
        harm = np.zeros_like(fr, dtype=bool)
        for k in range(1, 13):
            harm |= np.abs(fr - k * f0) < 25
        return float((sp[harm] ** 2).sum() / max((sp[~harm] ** 2).sum(), 1e-20))

    assert razao(4) > razao(1) * 3.0


def test_presets_de_amp_nao_estouram():
    t = np.arange(SR // 2) / SR
    x = (0.6 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    for nome, p in fx.PRESETS_AMP.items():
        q = {k: v for k, v in p.items() if k != "tipo"}
        y = fx.amp(x, SR, **q)
        assert np.isfinite(y).all(), nome
        assert np.abs(y).max() < 1.0, nome


# ------------------------------------------------------------- compressor
def test_compressor_reduz_a_faixa_dinamica():
    alto = np.ones(SR // 2, dtype=np.float32) * 0.9
    baixo = np.ones(SR // 2, dtype=np.float32) * 0.05
    x = np.concatenate([alto, baixo])
    y = fx.compressor(x, SR, limiar_db=-20.0, razao=6.0)[:, 0]
    antes = np.abs(x[SR // 4]) / np.abs(x[SR - 100])
    depois = np.abs(y[SR // 4]) / max(np.abs(y[SR - 100]), 1e-9)
    assert depois < antes


def test_limiter_respeita_o_teto():
    x = (np.random.default_rng(2).standard_normal((SR, 2)) * 3).astype(np.float32)
    y = fx.limiter(x, 0.97)
    assert np.abs(y).max() <= 0.99


# ------------------------------------------------------------------ cadeia
def test_cadeia_declarativa_roda_na_ordem():
    x = np.zeros((SR, 2), dtype=np.float32)
    x[0] = 1.0
    y = fx.aplicar(x, SR, [{"tipo": "eq", "freq": 1000, "ganho_db": 6.0},
                           {"tipo": "comp", "limiar_db": -30.0}])
    assert y.shape == x.shape
    assert np.isfinite(y).all()


def test_cadeia_recusa_efeito_desconhecido():
    with pytest.raises(ValueError):
        fx.aplicar(np.zeros((100, 2), np.float32), SR, [{"tipo": "banana"}])


# ------------------------------------------------------------------ mixer
def _projeto_simples(bit_depth=24) -> Project:
    p = Project("mix", bpm=120, bit_depth=bit_depth)
    tr = p.new_track("Lead", factory_by_name("NES Lead 25%"))
    pat = p.new_pattern("A", bars=1)
    for i, n in enumerate([60, 64, 67, 72]):
        pat.add(n, i * 480, 460)
    p.place(pat, tr, 0)
    return p


def test_mixer_serializa_inserts_send_e_secao(tmp_path):
    p = _projeto_simples()
    p.tracks[0].inserts = [{"tipo": "eq", "freq": 2000, "ganho_db": 3.0}]
    p.tracks[0].reverb_send = 0.35
    p.tracks[0].section = {"vozes": 3, "detune": 8, "spread_ms": 20, "pan": 0.6}
    p.master_reverb = {"sala": 0.7, "damp": 0.4}
    q = Project.load(p.save(tmp_path / "x"))
    assert q.tracks[0].inserts == p.tracks[0].inserts
    assert q.tracks[0].reverb_send == pytest.approx(0.35)
    assert q.tracks[0].section == p.tracks[0].section
    assert q.master_reverb == p.master_reverb


def test_envio_de_reverb_alonga_a_cauda():
    p = _projeto_simples()
    seco = engine.render_project(p)

    p.tracks[0].reverb_send = 0.6
    p.master_reverb = {"sala": 0.8, "damp": 0.35}
    molhado = engine.render_project(p)

    def fim(a):
        lim = np.abs(a).max() * 0.002
        idx = np.where(np.abs(a).max(axis=1) > lim)[0]
        return idx[-1] / p.sample_rate if len(idx) else 0.0

    assert fim(molhado) > fim(seco) + 0.5


def test_dobra_de_secao_alarga_o_estereo():
    p = _projeto_simples()
    p.tracks[0].pan = 0.0
    mono = engine.render_project(p)
    assert abs(float(np.corrcoef(mono[:, 0], mono[:, 1])[0, 1])) > 0.99

    p.tracks[0].section = {"vozes": 4, "detune": 10, "spread_ms": 25, "pan": 0.8}
    naipe = engine.render_project(p)
    assert abs(float(np.corrcoef(naipe[:, 0], naipe[:, 1])[0, 1])) < 0.95


def test_insert_de_track_muda_o_som():
    p = _projeto_simples()
    limpo = engine.render_project(p)
    p.tracks[0].inserts = [{"tipo": "eq", "modo": "lowpass",
                            "freq": 600, "q": 0.8}]
    filtrado = engine.render_project(p)

    def energia_aguda(a):
        sp = np.abs(np.fft.rfft(a[:, 0]))
        fr = np.fft.rfftfreq(len(a), 1 / p.sample_rate)
        return float((sp[fr > 3000] ** 2).sum())

    assert energia_aguda(filtrado) < energia_aguda(limpo) * 0.5


def test_track_muda_nao_entra_no_mix():
    p = _projeto_simples()
    p.tracks[0].mute = True
    a = engine.render_project(p)
    assert float(np.abs(a).max()) < 1e-6


# ------------------------------------------------------------- intensidade
def _projeto_com_arco() -> Project:
    """Comeca com uma nota longa sozinha e termina com bateria e acorde cheio."""
    p = Project("arco", bpm=120)
    calmo = p.new_track("Pad", factory_by_name("Warm Pad"))
    denso = p.new_track("Lead", factory_by_name("NES Lead 25%"))
    bat = p.new_track("Bateria", factory_by_name("Kick"))
    bat.instrument.is_drum = True

    pa = p.new_pattern("calmo", bars=4)
    pa.notes.append(Note(0, 480 * 16, 60, 70))
    pb = p.new_pattern("denso", bars=4)
    for i in range(64):
        pb.notes.append(Note(i * 120, 110, 60 + (i % 12), 110))
    pc = p.new_pattern("bateria", bars=4)
    for i in range(16):
        pc.notes.append(Note(i * 480, 120, 36, 120))

    p.place(pa, calmo, 0)
    p.place(pb, denso, 4)
    p.place(pc, bat, 4)
    return p


def test_intensidade_sobe_do_trecho_calmo_para_o_denso():
    p = _projeto_com_arco()
    c = INT.calcular(p)
    metade = p.ticks_to_seconds(p.ticks_per_bar * 4)
    calmo = c.valores[c.tempos < metade].mean()
    denso = c.valores[c.tempos >= metade].mean()
    assert denso > calmo * 1.5
    assert c.niveis.max() >= 4
    assert c.niveis.min() >= 1


def test_intensidade_desce_devagar_e_sobe_rapido():
    """Sprite que treme fica feio: a curva sobe na pancada e desce com calma."""
    p = _projeto_com_arco()
    c = INT.calcular(p)
    d = np.diff(c.valores)
    subidas = d[d > 0]
    descidas = -d[d < 0]
    assert len(subidas) and len(descidas)
    assert subidas.max() > descidas.max()


def test_intensidade_consulta_por_tempo():
    p = _projeto_com_arco()
    c = INT.calcular(p)
    assert 1 <= c.nivel_em(0.0) <= c.n_niveis
    assert 1 <= c.nivel_em(999.0) <= c.n_niveis      # alem do fim nao quebra
    assert 0.0 <= c.em(5.0) <= 1.0


def test_exporta_json_para_a_engine(tmp_path):
    import json
    p = _projeto_com_arco()
    c = INT.calcular(p)
    caminho = c.exportar_json(tmp_path / "curva", "arco")
    d = json.loads(caminho.read_text(encoding="utf-8"))
    assert d["niveis"] == 5
    assert len(d["pontos"]) == len(c.valores)
    assert all(1 <= ponto["n"] <= 5 for ponto in d["pontos"])
    assert all(0.0 <= ponto["v"] <= 1.0 for ponto in d["pontos"])


def test_resumo_desenha_a_curva():
    p = _projeto_com_arco()
    linha = INT.calcular(p).resumo_pt(20)
    assert len(linha) == 20


def test_amp_aceita_pan_sem_vazar_para_o_amplificador():
    """`pan` e parametro do canal. Se vazar para amp() o render inteiro morre."""
    x = np.zeros((SR // 2, 2), dtype=np.float32)
    x[0] = 1.0
    y = fx.aplicar(x, SR, [dict(fx.PRESETS_AMP["rock"], pan=-0.85)])
    assert y.shape == x.shape
    assert np.isfinite(y).all()
    # panorama para a esquerda: canal esquerdo tem de ter mais energia
    assert (y[:, 0] ** 2).sum() > (y[:, 1] ** 2).sum() * 3


@pytest.mark.parametrize("pico_entrada", [0.3, 0.9, 1.4, 4.0])
def test_limiter_entrega_o_nivel_que_prometeu(pico_entrada):
    """Sinal abaixo do teto passa intacto; acima do teto para NO teto —
    nunca alguns dB abaixo dele."""
    t = np.arange(SR // 4) / SR
    x = (np.sin(2 * np.pi * 220 * t) * pico_entrada).astype(np.float32)
    y = fx.limiter(x, 0.97)
    esperado = min(pico_entrada, 0.97)
    assert float(np.abs(y).max()) == pytest.approx(esperado, rel=0.02)


def test_limiter_nao_amplifica_material_baixo():
    x = (np.random.default_rng(11).standard_normal((SR, 2)) * 0.05).astype(np.float32)
    y = fx.limiter(x, 0.97)
    assert float(np.abs(y).max()) <= float(np.abs(x).max()) * 1.01
