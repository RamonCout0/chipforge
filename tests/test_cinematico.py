"""
Testes do modo cinematografico: voz, sala por convolucao, master, mapa de
tempo, automacao e escrita orquestral.

A regra destes testes e a mesma do resto do projeto: nada de "roda sem
excecao". Cada um afirma uma propriedade AUDIVEL — que a voz tem energia onde
uma voz tem, que a sala decai no tempo que ela promete, que o medidor de LUFS
bate com a calibracao do padrao, que o accelerando realmente encurta o
compasso.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chipforge.audio import cinema as CIN
from chipforge.audio import engine
from chipforge.audio import espaco
from chipforge.audio import master as MST
from chipforge.audio import voz as VOZ
from chipforge.core import dinamica as DIN
from chipforge.core import orquestra as ORQ
from chipforge.core.project import Note, Project

SR = 44100


def bandas(y: np.ndarray, sr: int = SR) -> dict:
    """Energia por banda de um trecho, em dB."""
    y = np.asarray(y, np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1)
    n = min(16384, len(y))
    seg = y[len(y) // 4: len(y) // 4 + n]
    if len(seg) < n:
        seg = np.pad(seg, (0, n - len(seg)))
    S = np.abs(np.fft.rfft(seg * np.hanning(n)))
    f = np.fft.rfftfreq(n, 1 / sr)
    out = {}
    for nome, lo, hi in [("grave", 80, 250), ("medio_grave", 250, 800),
                         ("medio", 800, 2500), ("agudo", 2500, 7000),
                         ("brilho", 7000, 14000)]:
        m = (f >= lo) & (f < hi)
        out[nome] = float(10 * np.log10(max((S[m] ** 2).mean(), 1e-12)))
    return out


# ==========================================================================
# Voz
# ==========================================================================
def test_voz_poe_energia_na_faixa_da_melodia():
    """A razao de existir do sintetizador de voz e preencher 800-2500 Hz.

    Se a voz nao dominar ali, ela nao resolve o problema que motivou o modulo:
    mixagem so de sample fica magra exatamente nessa faixa.
    """
    y = VOZ.voz("Solista feminina").render_note(69, 1.2, SR, 100)
    b = bandas(y)
    assert b["medio"] > b["grave"] + 12
    assert b["medio"] > b["brilho"] + 12


def test_voz_tem_conteudo_de_ar_acima_de_7k():
    """O banco de formantes morre no F5 (~5 kHz). Sem o caminho de ar direto,
    a voz sai com o mesmo teto opaco que denuncia sample de teclado."""
    com = VOZ.voz("Solista feminina", ar_direto=0.02)
    sem = VOZ.voz("Solista feminina", ar_direto=0.0)
    assert bandas(com.render_note(69, 1.0, SR))["brilho"] > \
        bandas(sem.render_note(69, 1.0, SR))["brilho"] + 10


@pytest.mark.parametrize("preset", VOZ.nomes_de_preset())
def test_presets_de_voz_estao_calibrados(preset):
    """Trocar de preset nao pode significar mexer no fader.

    Os ganhos foram calibrados um a um; este teste e o que impede que uma
    mudanca no banco de formantes desfaca a calibracao sem ninguem notar.
    """
    inst = VOZ.voz(preset)
    rep = {"soprano": 69, "contralto": 62, "tenor": 57, "baixo": 48}[inst.naipe]
    rms = np.mean([float(np.sqrt((inst.render_note(p, 1.0, SR, 100) ** 2).mean()))
                   for p in (rep - 5, rep, rep + 5)])
    assert 0.20 < rms < 0.45, f"{preset} saiu em rms={rms:.3f}"


def test_dobra_de_secao_produz_vozes_diferentes():
    """Quatro copias identicas em unissono continuam sendo uma voz, so mais
    alta. O que faz virar coro e cada garganta tremer por conta propria — e
    aqui o engine so muda `detune_cents`, entao e ele que precisa mudar o
    jitter junto."""
    from dataclasses import replace
    base = VOZ.voz("Coro soprano (aah)")
    a = base.render_note(69, 1.0, SR, 90)
    b = replace(base, detune_cents=9.0).render_note(69, 1.0, SR, 90)
    n = min(len(a), len(b))
    corr = float(np.corrcoef(a[:n], b[:n])[0, 1])
    assert corr < 0.9, f"as duas vozes saíram quase idênticas (corr={corr:.3f})"


def test_voz_e_reprodutivel():
    """Duas exportacoes do mesmo projeto tem que dar o mesmo arquivo."""
    a = VOZ.voz("Solista feminina").render_note(67, 0.8, SR, 100)
    b = VOZ.voz("Solista feminina").render_note(67, 0.8, SR, 100)
    assert np.array_equal(a, b)


def test_velocity_abre_o_timbre_e_nao_so_o_volume():
    """Cantar forte gera mais harmonico agudo. Se so mudasse o nivel, o
    fortissimo soaria como o pianissimo com o fader levantado."""
    inst = VOZ.voz("Solista feminina")
    forte = inst.render_note(69, 1.0, SR, 120)
    fraco = inst.render_note(69, 1.0, SR, 45)

    def tilt(y):
        b = bandas(y)
        return b["agudo"] - b["medio"]

    assert tilt(forte) > tilt(fraco) + 1.0


def test_satb_respeita_a_tessitura():
    acordes = [[62, 65, 69], [58, 62, 65], [57, 60, 64]]
    vozes = VOZ.satb([81, 77, 76], acordes)
    for naipe, notas in vozes.items():
        lo, hi = VOZ.TESSITURA[naipe]
        if naipe == "soprano":
            continue                      # a melodia manda no soprano
        assert all(lo <= p <= hi for p in notas), (naipe, notas)


# ==========================================================================
# Sala
# ==========================================================================
@pytest.mark.parametrize("nome", ["camara", "salao", "hall", "catedral"])
def test_sala_decai_no_tempo_que_promete(nome):
    """T20 medido por integracao de Schroeder, comparado ao RT declarado.

    Uma sala que promete 5 s e entrega 2 nao e uma catedral — e um salao com
    nome errado, e quem escolhe pelo nome fica sem o som que queria.
    """
    ir = espaco.impulso(nome, SR)
    e = (ir.astype(np.float64) ** 2).mean(axis=1)
    s = np.cumsum(e[::-1])[::-1]
    db = 10 * np.log10(np.maximum(s / s[0], 1e-12))

    def cruza(alvo):
        i = int(np.argmax(db < alvo))
        return i / SR

    t20 = (cruza(-25) - cruza(-5)) * 3.0
    esperado = espaco.SALAS[nome].rt_medio
    assert 0.5 * esperado < t20 < 1.2 * esperado, f"{nome}: T20={t20:.2f}s"


def test_sala_alonga_o_sinal_e_espalha_no_estereo():
    x = np.zeros((SR, 2), np.float32)
    x[100] = 1.0
    y = espaco.reverb(x, SR, "hall", wet=1.0, dry=0.0)
    assert float(np.abs(y[SR // 2:]).max()) > 1e-4     # ainda soa meio segundo depois
    # os dois canais nao podem ser a mesma coisa, senao a sala colapsa no centro
    assert float(np.corrcoef(y[:, 0], y[:, 1])[0, 1]) < 0.85


def test_largura_mantem_o_grave_no_centro():
    """Alargar o grave e o erro classico: soma errado no mono e faz o pico do
    limitador disparar sem nenhum ganho de volume percebido."""
    t = np.arange(SR) / SR
    grave = np.sin(2 * np.pi * 60 * t).astype(np.float32)
    agudo = np.sin(2 * np.pi * 3000 * t).astype(np.float32)
    st = np.stack([grave + agudo, grave - agudo * 0.5], axis=1)
    y = espaco.largura(st, 1.8, 140.0, SR)
    lado = (y[:, 0] - y[:, 1]) * 0.5
    b = bandas(np.stack([lado, lado], axis=1))
    assert b["grave"] < b["agudo"] - 20


# ==========================================================================
# Master
# ==========================================================================
def test_lufs_bate_com_a_calibracao_do_padrao():
    """Senoide de 1 kHz a -20 dBFS: -20 LUFS em dual mono, -23 num canal so.
    Sao os dois valores de calibracao do BS.1770 — se estes andarem, qualquer
    alvo de loudness do app esta errado junto."""
    t = np.arange(SR * 5) / SR
    s = (0.1 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    assert abs(MST.lufs_integrado(np.stack([s, s], 1), SR) + 20.0) < 0.6
    assert abs(MST.lufs_integrado(np.stack([s, np.zeros_like(s)], 1), SR) + 23.0) < 0.6


def test_pico_real_ve_o_que_o_pico_de_amostra_nao_ve():
    """Uma senoide desalinhada com a grade de amostragem passa por cima do que
    as amostras mostram. E esse excedente que distorce no conversor."""
    t = np.arange(SR) / SR
    s = (0.98 * np.sin(2 * np.pi * 7000 * t + 0.7)).astype(np.float32)
    st = np.stack([s, s], 1)
    amostra = 20 * np.log10(float(np.abs(st).max()))
    assert MST.pico_real_db(st, SR) > amostra - 0.01


def test_master_atinge_o_alvo_de_loudness_e_o_teto():
    rng = np.random.default_rng(4)
    n = SR * 12
    env = np.interp(np.arange(n), [0, n * 0.4, n * 0.7, n], [0.08, 0.7, 0.3, 0.05])
    x = (rng.normal(0, 0.25, (n, 2)) * env[:, None]).astype(np.float32)
    r = MST.masterizar(x, SR, "cinematico", alvo_lufs=-14.0, teto_db=-1.0)
    assert abs(r.depois.lufs + 14.0) < 1.0
    assert r.depois.pico_dbtp <= -0.9, f"pico real {r.depois.pico_dbtp:.2f} dBTP"


def test_master_nao_esmaga_o_arco():
    """Um master que zera o LRA transformou a musica num tijolo. A cadeia pode
    apertar; nao pode apagar a diferenca entre o pianissimo e o tutti."""
    rng = np.random.default_rng(9)
    n = SR * 20
    env = np.interp(np.arange(n), [0, n * 0.3, n * 0.6, n], [0.05, 0.9, 0.2, 0.05])
    x = (rng.normal(0, 0.25, (n, 2)) * env[:, None]).astype(np.float32)
    r = MST.masterizar(x, SR, "cinematico")
    assert r.depois.lra > r.antes.lra * 0.55


def test_excitador_cria_conteudo_que_nao_existia():
    """EQ so multiplica o que ja esta la. Num mix de SoundFont nao ha nada
    acima de 8 kHz para multiplicar — quem cria e a saturacao."""
    from chipforge.audio import fx
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.2, SR * 3).astype(np.float32)
    escuro = fx.biquad(x, SR, "lowpass", 5000.0, 0.7)
    escuro = fx.biquad(escuro, SR, "lowpass", 5000.0, 0.7)
    st = np.stack([escuro, escuro], 1)
    y = MST.excitador(st, SR, 2400.0, 0.5, 0.5)
    assert bandas(y)["brilho"] > bandas(st)["brilho"] + 6


def test_casar_perfil_aproxima_do_alvo():
    from chipforge.audio import fx
    rng = np.random.default_rng(6)
    x = rng.normal(0, 0.2, SR * 4).astype(np.float32)
    escuro = fx.biquad(x, SR, "lowshelf", 300.0, 0.7, 12.0)
    st = np.stack([escuro, escuro], 1)
    alvo = {"sub": -3.0, "grave": -1.0, "medio-grave": 0.0,
            "medio": -1.0, "agudo": -3.0, "brilho": -6.0}
    antes = MST.balanco_espectral(st, SR)
    depois = MST.balanco_espectral(MST.casar_perfil(st, SR, alvo), SR)
    erro_antes = sum(abs(antes[k] - alvo[k]) for k in alvo)
    erro_depois = sum(abs(depois[k] - alvo[k]) for k in alvo)
    assert erro_depois < erro_antes


def test_limitador_com_antecipacao_nao_deixa_passar():
    rng = np.random.default_rng(11)
    x = rng.normal(0, 0.4, (SR * 2, 2)).astype(np.float32)
    x[SR] = 3.0                                   # um transiente absurdo
    y = MST.limitador(x, SR, -1.0, 3.0, 90.0)
    assert float(np.abs(y).max()) <= 10 ** (-1.0 / 20) + 1e-6


# ==========================================================================
# Mapa de tempo e automacao
# ==========================================================================
def test_accelerando_encurta_o_compasso():
    m = DIN.MapaDeTempo(ppq=480, bpm_base=100.0)
    m.rampa(0, 480 * 16, 100.0, 200.0)
    primeiro = m.duracao_de(0, 480 * 4)
    ultimo = m.duracao_de(480 * 12, 480 * 4)
    assert ultimo < primeiro * 0.75


def test_mapa_de_tempo_e_monotono():
    m = DIN.MapaDeTempo(ppq=480, bpm_base=90.0)
    m.rampa(0, 4800, 90.0, 140.0).rampa(4800, 9600, 140.0, 60.0)
    t = [m.segundos(x) for x in range(0, 12000, 137)]
    assert all(b > a for a, b in zip(t, t[1:]))


def test_ritardando_estica_o_fim():
    m = DIN.MapaDeTempo(ppq=480, bpm_base=120.0)
    m.marcar(0, 120.0)
    sem = m.segundos(480 * 32)
    m.ritardando(480 * 32, 4, 480 * 4, fator=0.6)
    assert m.segundos(480 * 32) > sem * 1.05


def test_automacao_interpola_sem_degrau():
    a = DIN.Automacao().rampa(0, 480 * 8, 0.2, 1.0)
    c = a.curva(SR * 2, None, SR, 60.0 / 120.0 / 480)
    assert c[0] == pytest.approx(0.2, abs=0.02)
    assert np.all(np.diff(c) >= -1e-6)            # so sobe
    assert float(np.abs(np.diff(c)).max()) < 0.01  # sem salto


def test_arco_de_secoes_da_niveis_diferentes():
    secoes = [("A", 8), ("B", 8), ("C", 8)]
    a = DIN.arco_de_secoes(secoes, 1920, {"A": 0.3, "B": 1.0, "C": 0.5})
    assert a.valor(1920 * 4) < a.valor(1920 * 12)
    assert a.valor(1920 * 20) < a.valor(1920 * 12)


# ==========================================================================
# Escrita orquestral
# ==========================================================================
def test_encaixar_traz_para_a_tessitura():
    for inst, (lo, hi) in ORQ.TESSITURA.items():
        for p in (12, 40, 64, 100, 126):
            assert lo <= ORQ.encaixar(p, inst) <= hi


def test_limpar_grave_abre_intervalo_apertado():
    """Duas notas a uma terca menor abaixo de Do3 sao lama em qualquer sala."""
    saida = ORQ.limpar_grave([36, 39, 43])
    graves = [p for p in saida if p < ORQ.LIMITE_GRAVE]
    for a, b in zip(graves, graves[1:]):
        assert b - a >= ORQ.INTERVALO_MINIMO_GRAVE


def test_bloco_conduz_as_vozes_com_pouco_movimento():
    """A diferenca entre naipe e teclado: as vozes andam por segundas em vez de
    saltarem juntas para a posicao fundamental de cada acorde."""
    acordes = [[50, 53, 57], [46, 50, 53], [53, 57, 60], [48, 52, 55]]
    notas = ORQ.bloco(acordes, 480, 1, 4, n_vozes=3, low=55, high=79)
    por_tempo: dict[int, list[int]] = {}
    for n in notas:
        por_tempo.setdefault(n.start, []).append(n.pitch)
    seq = [sorted(v) for _, v in sorted(por_tempo.items())]
    movimento = [abs(b[i] - a[i]) for a, b in zip(seq, seq[1:])
                 for i in range(min(len(a), len(b)))]
    assert max(movimento) <= 7, f"salto de {max(movimento)} semitons"


def test_ostinato_acentua_o_primeiro_de_cada_tempo():
    notas = ORQ.ostinato([[50, 53, 57]], 480, 1, 4, divisao=4)
    fortes = [n.velocity for n in notas if n.start % 480 == 0]
    fracos = [n.velocity for n in notas if n.start % 480 != 0]
    assert min(fortes) > max(fracos)


def test_taiko_varia_o_quarto_compasso():
    """Dezesseis compassos do mesmo padrao soam como loop — e loop e a
    impressao que uma trilha nao pode dar."""
    notas = ORQ.taiko(8, 480, "marcha", 4, 36, 100)
    def compasso(c):
        return sorted(n.start - c * 1920 for n in notas
                      if c * 1920 <= n.start < (c + 1) * 1920)
    assert compasso(3) != compasso(0)
    assert compasso(1) == compasso(0)


def test_humanizar_espalha_dentro_do_acorde():
    """Mover o acorde inteiro nao muda nada perceptivel; o que engrossa o naipe
    e as vozes DENTRO dele chegarem em instantes diferentes."""
    acorde = [Note(0, 480, p, 80) for p in (48, 55, 60, 64, 67)]
    saida = ORQ.humanizar_naipe(acorde, 480, 24.0, 100.0)
    assert len({n.start for n in saida}) >= 4


def test_humanizar_atrasa_o_grave():
    """Um contrabaixo leva mais tempo para a corda falar que um violino."""
    notas = [Note(0, 480, p, 80) for p in (36, 36, 36, 84, 84, 84)]
    saida = ORQ.humanizar_naipe(notas, 480, 24.0, 100.0)
    graves = [n.start for n in saida if n.pitch == 36]
    agudos = [n.start for n in saida if n.pitch == 84]
    assert np.mean(graves) > np.mean(agudos)


def test_preparacao_termina_no_downbeat():
    ev = ORQ.preparacao(16, 480, 4, 2.0)
    alvo = 16 * 1920
    riser = ev["riser"][0]
    assert riser.start + riser.dur == alvo
    assert ev["impacto"][0].start == alvo


# ==========================================================================
# Desenho de som
# ==========================================================================
@pytest.mark.parametrize("preset", CIN.nomes_de_preset())
def test_instrumentos_de_cinema_soam_e_nao_estouram(preset):
    inst = CIN.instrumento(preset)
    y = inst.render_note(40, 1.5, SR, 110)
    # O relogio e a excecao proposital: um tique tem que durar 50 a 160 ms e
    # calar. Se ele durasse os 0,4 s que os outros duram, nao seria um tique —
    # seria um bloco de madeira com cauda, e o ouvido nao o usaria para cravar
    # o instante, que e a unica funcao dele.
    minimo = SR * 0.04 if "Relogio" in preset else SR * 0.4
    assert len(y) > minimo
    assert float(np.abs(y).max()) <= 1.7
    assert float(np.sqrt((y ** 2).mean())) > 1e-4


def test_tique_e_curto_e_taque_e_diferente():
    """Duas exigencias que fazem um relogio: o tique cala rapido (senao vira
    um bloco de madeira ressoando) e as duas batidas nao sao o mesmo som."""
    inst = CIN.instrumento("Relogio")
    tique = inst.render_note(72, 0.5, SR, 100)
    taque = inst.render_note(70, 0.5, SR, 100)
    assert len(tique) / SR < 0.25
    n = min(len(tique), len(taque))
    assert abs(float(np.corrcoef(tique[:n], taque[:n])[0, 1])) < 0.6


def test_vento_respira():
    """Ruido estatico o ouvido descarta como defeito de gravacao; ruido que
    varia devagar ele aceita como ambiente. A envoltoria tem que se mexer."""
    y = CIN.instrumento("Mata").render_note(60, 12.0, SR, 100)
    jan = int(SR * 0.25)
    env = np.array([float(np.sqrt((y[i:i + jan] ** 2).mean()))
                    for i in range(0, len(y) - jan, jan)])
    meio = env[len(env) // 6: -len(env) // 6]
    assert meio.std() / max(meio.mean(), 1e-9) > 0.08


def test_impacto_e_taiko_moram_no_grave():
    for preset in ("Impacto grave", "Taiko"):
        b = bandas(CIN.instrumento(preset).render_note(36, 1.0, SR, 120))
        assert b["grave"] > b["agudo"] + 15, preset


def test_riser_sobe():
    """A energia da segunda metade tem que estar mais alta em frequencia que a
    da primeira — senao nao e um riser, e um ruido."""
    y = CIN.instrumento("Riser longo").render_note(48, 3.0, SR, 110)
    meio = len(y) // 2

    def centroide(seg):
        S = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        f = np.fft.rfftfreq(len(seg), 1 / SR)
        return float((f * S).sum() / max(S.sum(), 1e-9))

    assert centroide(y[meio:]) > centroide(y[:meio]) * 1.3


def test_riser_corta_no_fim():
    y = CIN.instrumento("Riser longo").render_note(48, 2.0, SR, 110)
    assert float(np.abs(y[-200:]).max()) < float(np.abs(y[-SR // 4:-SR // 8]).max()) * 0.3


def test_cinema_serializa_e_volta():
    inst = CIN.instrumento("Braam", gain=0.66, drive=1.3)
    volta = CIN._Base.from_dict(inst.to_dict())
    assert type(volta) is type(inst)
    assert volta.gain == pytest.approx(0.66)
    n = min(len(inst.render_note(45, 0.5, SR)), len(volta.render_note(45, 0.5, SR)))
    assert np.allclose(inst.render_note(45, 0.5, SR)[:n],
                       volta.render_note(45, 0.5, SR)[:n])


# ==========================================================================
# Integracao com o projeto
# ==========================================================================
def _projeto_simples(tempo_map=None, automacao=None) -> Project:
    p = Project(name="t", bpm=120.0, bit_depth=24)
    if tempo_map:
        p.tempo_map = tempo_map
    tr = p.new_track("voz", VOZ.voz("Solista feminina"))
    if automacao:
        tr.automacao = {"volume": automacao}
    pat = p.new_pattern("p", bars=4)
    for i in range(4):
        pat.notes.append(Note(i * 480 * 4, 480 * 3, 65, 100))
    p.place(pat, tr, 0)
    return p


def test_tempo_map_muda_a_duracao_da_peca():
    normal = _projeto_simples()
    lento = _projeto_simples(tempo_map=[[0, 120.0], [480 * 16, 60.0]])
    assert lento.ticks_to_seconds(480 * 16) > normal.ticks_to_seconds(480 * 16) * 1.2


def test_automacao_de_volume_chega_no_audio():
    """Um crescendo de doze compassos nao existe em velocity de nota; ele
    precisa sair no fader, e o fader precisa chegar ao render."""
    p = _projeto_simples()
    plano = engine.render_project(p, apply_master=False)
    cresc = engine.render_project(
        _projeto_simples(automacao=[[0, 0.15], [480 * 16, 1.0]]),
        apply_master=False)
    # o buffer tem cauda de sala depois da ultima nota; a comparacao precisa
    # cair dentro da MUSICA, senao mede silencio contra silencio
    musica = int(p.ticks_to_seconds(480 * 16) * SR)
    n = musica // 4
    assert float(np.abs(cresc[:n]).max()) < float(np.abs(plano[:n]).max()) * 0.6
    fim = slice(musica - n, musica)
    assert float(np.abs(cresc[fim]).max()) > float(np.abs(cresc[:n]).max()) * 1.5


def test_projeto_com_tempo_e_automacao_sobrevive_ao_disco(tmp_path):
    p = _projeto_simples(tempo_map=[[0, 100.0], [9600, 140.0]],
                         automacao=[[0, 0.3], [9600, 1.0]])
    p.master_perfil = "cinematico"
    p.master_alvo = {"medio": -12.0}
    volta = Project.load(p.save(tmp_path / "x"))
    assert volta.tempo_map == [[0, 100.0], [9600, 140.0]]
    assert volta.tracks[0].automacao["volume"] == [[0, 0.3], [9600, 1.0]]
    assert volta.master_perfil == "cinematico"
    assert volta.master_alvo == {"medio": -12.0}


def test_voz_serializa_dentro_do_projeto(tmp_path):
    p = _projeto_simples()
    volta = Project.load(p.save(tmp_path / "y"))
    inst = volta.tracks[0].instrument
    assert isinstance(inst, VOZ.VozInstrument)
    assert inst.naipe == "soprano"


def test_sala_por_convolucao_no_master_do_projeto():
    p = _projeto_simples()
    p.tracks[0].reverb_send = 0.8
    p.master_reverb = {"sala": "catedral", "wet": 1.0}
    seco = engine.render_project(_projeto_simples(), apply_master=False)
    molhado = engine.render_project(p, apply_master=False)
    # a cauda depois da ultima nota so pode vir da sala
    corte = min(len(seco), len(molhado)) - int(SR * 1.2)
    assert float(np.abs(molhado[corte:]).max()) > \
        float(np.abs(seco[corte:]).max()) * 3
