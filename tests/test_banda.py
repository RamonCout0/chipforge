"""
Testes da corda modelada e da escrita de banda.

O que estes testes protegem e o que a primeira tentativa de rock errou: a
guitarra tem que AFINAR (o filtro de perda desafinava ate 35 cents conforme o
palm mute fechava), o palm mute tem que ser curto E escuro (nao apenas mais
baixo), e o riff tem que deixar silencio entre as palhetadas — sem esse
silencio a distorcao nunca respira e o riff vira parede.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chipforge.audio import corda as CD
from chipforge.core import banda as BD
from chipforge.core import theory as T
from chipforge.core.project import Note, Project

SR = 44100


def _f0_cents(y: np.ndarray, alvo_hz: float) -> float:
    """Desvio de afinacao em cents, pelo pico do espectro perto do alvo.

    O refino parabolico entre os tres bins em volta do pico nao e luxo: com
    32768 pontos a 44,1 kHz um bin vale 1,35 Hz, o que em Mi2 (82 Hz) da 28
    cents. Sem interpolar, este teste mediria o proprio erro de quantizacao e
    reprovaria qualquer modelo, por mais afinado que estivesse.
    """
    n = 1 << 15
    seg = y[int(0.03 * SR): int(0.03 * SR) + n]
    if len(seg) < n:
        seg = np.pad(seg, (0, n - len(seg)))
    S = np.abs(np.fft.rfft(seg * np.hanning(n)))
    f = np.fft.rfftfreq(n, 1 / SR)
    m = (f > alvo_hz * 0.8) & (f < alvo_hz * 1.2)
    if not m.any() or S[m].max() <= 0:
        return float("nan")
    idx = int(np.where(m)[0][np.argmax(S[m])])
    d = 0.0
    if 0 < idx < len(S) - 1:
        a, b, c = (float(np.log(S[i] + 1e-12)) for i in (idx - 1, idx, idx + 1))
        den = a - 2 * b + c
        if abs(den) > 1e-9:
            d = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    return float(1200 * np.log2(((idx + d) * SR / n) / alvo_hz))


def _bandas(y: np.ndarray) -> dict:
    n = min(1 << 14, len(y))
    S = np.abs(np.fft.rfft(y[:n] * np.hanning(n)))
    f = np.fft.rfftfreq(n, 1 / SR)
    return {k: float(10 * np.log10(max((S[(f >= lo) & (f < hi)] ** 2).mean(), 1e-12)))
            for k, (lo, hi) in {"grave": (60, 250), "medio": (250, 2500),
                                "agudo": (2500, 9000)}.items()}


def _duracao_util(y: np.ndarray, limiar: float = 0.02) -> float:
    env = np.abs(y)
    acima = np.where(env > env.max() * limiar)[0]
    return (acima[-1] - acima[0]) / SR if len(acima) else 0.0


# ==========================================================================
# A corda
# ==========================================================================
@pytest.mark.parametrize("preset", [p for p in CD.nomes_de_preset()
                                    if "harmonico" not in p])
def test_corda_afina(preset):
    """O laco nao tem o comprimento da linha de atraso: tem o comprimento da
    linha MAIS o atraso do filtro de perda. Sem compensar isso a corda sai
    grave, e sai grave por uma quantidade que MUDA com o palm mute — medido em
    -9 cents com a corda solta e -35 cents no chug pesado."""
    inst = CD.corda(preset)
    base = 28 if "aixo" in preset else 40
    for off in (0, 7, 12, 19, 24):
        p = base + off
        y = inst.render_note(p, 0.9, SR, 105)
        desvio = _f0_cents(y, 440 * 2 ** ((p - 69) / 12))
        assert abs(desvio) < 6.0, f"{preset} em {p}: {desvio:+.1f} cents"


def test_palm_mute_encurta_e_escurece():
    """Palm mute nao e "mais baixo". E a corda perdendo o agudo em dezenas de
    milissegundos enquanto a fundamental continua — e isso que soa percussivo."""
    solta = CD.corda("Guitarra (aberta)")
    presa = CD.corda("Guitarra (chug pesado)")
    ys, yp = solta.render_note(45, 1.0, SR, 105), presa.render_note(45, 1.0, SR, 105)
    assert _duracao_util(yp) < _duracao_util(ys) * 0.6
    bs, bp = _bandas(ys), _bandas(yp)
    assert (bp["agudo"] - bp["grave"]) < (bs["agudo"] - bs["grave"]) - 3.0


def test_posicao_da_palheta_muda_o_timbre():
    """Perto do cavalete apaga os harmonicos graves e soa nasal; no meio do
    braco soa redondo. E uma subtracao, nao um filtro."""
    cavalete = CD.corda("Guitarra (aberta)", pos_palheta=0.05)
    meio = CD.corda("Guitarra (aberta)", pos_palheta=0.45)
    a = _bandas(cavalete.render_note(45, 1.0, SR, 105))
    b = _bandas(meio.render_note(45, 1.0, SR, 105))
    assert (a["agudo"] - a["grave"]) > (b["agudo"] - b["grave"])


def test_harmonico_mata_a_fundamental():
    """Harmonico natural: a palheta encosta no no do parcial. A fundamental
    simplesmente nao e excitada."""
    normal = CD.corda("Guitarra (aberta)").render_note(40, 1.0, SR, 105)
    harm = CD.corda("Guitarra (harmonico)").render_note(40, 1.0, SR, 105)
    f0 = 440 * 2 ** ((40 - 69) / 12)
    n = 1 << 15

    def energia_em(y, hz):
        seg = y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))
        S = np.abs(np.fft.rfft(seg * np.hanning(n)))
        f = np.fft.rfftfreq(n, 1 / SR)
        m = (f > hz * 0.94) & (f < hz * 1.06)
        return float(S[m].max()) if m.any() else 0.0

    assert energia_em(harm, f0) < energia_em(normal, f0) * 0.2


def test_velocity_muda_ataque_e_nivel():
    inst = CD.corda("Guitarra (aberta)")
    forte = inst.render_note(45, 1.0, SR, 120)
    fraco = inst.render_note(45, 1.0, SR, 40)
    assert np.abs(forte).max() > np.abs(fraco).max() * 1.5


def test_corda_e_reprodutivel():
    a = CD.corda("Guitarra (chug)").render_note(43, 0.4, SR, 100)
    b = CD.corda("Guitarra (chug)").render_note(43, 0.4, SR, 100)
    assert np.array_equal(a, b)


def test_cordas_diferentes_soam_diferente():
    """Duas guitarras dobradas so dao largura se forem execucoes DIFERENTES.
    Com a mesma semente elas sao a mesma onda, e dobrar vira so +6 dB."""
    a = CD.corda("Guitarra (chug pesado)", seed=11).render_note(43, 0.4, SR, 100)
    b = CD.corda("Guitarra (chug pesado)", seed=29).render_note(43, 0.4, SR, 100)
    n = min(len(a), len(b))
    assert abs(float(np.corrcoef(a[:n], b[:n])[0, 1])) < 0.9


def test_nota_e_parada_no_fim():
    """Sem cortar a nota, a anterior continua soando por baixo da proxima e o
    riff vira nuvem."""
    y = CD.corda("Guitarra (aberta)").render_note(45, 0.25, SR, 105)
    fim = int(0.25 * SR)
    cauda = y[fim + int(0.15 * SR):]
    assert len(cauda) == 0 or float(np.abs(cauda).max()) < float(np.abs(y).max()) * 0.02


def test_escolhe_a_corda_que_alcanca():
    inst = CD.corda("Guitarra (aberta)")
    assert inst._corda_usada(40) == 40          # Mi grave solta
    assert inst._corda_usada(64) == 64          # Mi agudo solta
    assert inst._corda_usada(52) in CD.AFINACAO_GUITARRA


def test_corda_serializa_e_volta():
    inst = CD.corda("Guitarra (chug)", gain=0.66, abafamento=0.5)
    volta = CD.CordaInstrument.from_dict(inst.to_dict())
    assert volta.gain == pytest.approx(0.66)
    assert volta.abafamento == pytest.approx(0.5)
    assert tuple(volta.afinacao) == tuple(inst.afinacao)
    a = inst.render_note(45, 0.3, SR, 100)
    b = volta.render_note(45, 0.3, SR, 100)
    assert np.allclose(a, b)


def test_corda_dentro_do_projeto(tmp_path):
    p = Project(name="r", bpm=140.0, bit_depth=24)
    tr = p.new_track("gtr", CD.corda("Guitarra (chug pesado)"))
    pat = p.new_pattern("p", bars=1)
    pat.notes.append(Note(0, 240, 40, 105))
    p.place(pat, tr, 0)
    volta = Project.load(p.save(tmp_path / "x"))
    assert isinstance(volta.tracks[0].instrument, CD.CordaInstrument)


# ==========================================================================
# Riff
# ==========================================================================
def _acordes():
    N = T.name_to_midi
    return [[N("E3"), N("G3"), N("B3")], [N("C3"), N("E3"), N("G3")]]


def test_riff_e_acorde_de_potencia_sem_terca():
    """A terca e o que vira lama quando distorcida: com tres notas os produtos
    de intermodulacao caem fora do acorde. Fundamental e quinta reforcam."""
    notas = BD.riff(_acordes(), 480, "chug8", 1, 100)
    por_tempo: dict[int, set] = {}
    for n in notas:
        por_tempo.setdefault(n.start, set()).add(n.pitch)
    for alturas in por_tempo.values():
        intervalos = {(p - min(alturas)) % 12 for p in alturas}
        assert 3 not in intervalos and 4 not in intervalos, intervalos


def test_riff_deixa_silencio_entre_as_palhetadas():
    """O parametro `curta`. Em 1.0 as notas se emendam, a distorcao nunca
    respira e o riff vira uma parede continua — o erro mais comum de guitarra
    programada."""
    chug = BD.riff(_acordes(), 480, "chug16", 1, 100, curta=0.5)
    passo = 480 // 4
    curtas = [n for n in chug if n.dur < passo]
    assert len(curtas) > len(chug) * 0.7


def test_riff_liga_as_notas_marcadas_com_traco():
    """`-` na grade prolonga a nota anterior em vez de tocar de novo."""
    sustentado = BD.riff(_acordes()[:1], 480, "parada", 1, 100,
                         variar_ultimo=False)
    assert len(sustentado) == 3            # so o acorde, uma vez
    assert all(n.dur >= 480 * 3 for n in sustentado)


def test_riff_varia_o_ultimo_acorde():
    a = BD.riff(_acordes(), 480, "chug8", 1, 100, variar_ultimo=True)
    b = BD.riff(_acordes(), 480, "chug8", 1, 100, variar_ultimo=False)
    ini_a = sorted({n.start for n in a if n.start >= 480 * 4})
    ini_b = sorted({n.start for n in b if n.start >= 480 * 4})
    assert ini_a != ini_b


def test_potencia_respeita_a_faixa():
    for raiz in (28, 40, 52, 64, 76):
        for p in BD.potencia(raiz, True, 40, 52):
            assert 40 <= p <= 64


def test_baixo_segue_a_mesma_grade_do_riff():
    """Tocar a mesma figura ritmica e o que faz guitarra e baixo soarem como
    uma coisa so."""
    r = BD.riff(_acordes(), 480, "galope", 1, 100, variar_ultimo=False)
    b = BD.linha_de_baixo(_acordes(), 480, "galope", 1, 100)
    assert {n.start for n in b} <= {n.start for n in r}


# ==========================================================================
# Bateria
# ==========================================================================
def test_bateria_usa_as_alturas_do_padrao_gm():
    notas = BD.bateria(4, 480, "rock")
    alturas = {n.pitch for n in notas}
    assert BD.BUMBO in alturas and BD.CAIXA in alturas and BD.CHIMBAL in alturas
    assert alturas <= set(BD._VOZES[k][i] for k in BD._VOZES for i in (0, 1))


def test_bateria_vira_a_cada_quatro_compassos():
    """Sem virada, dezesseis compassos sao o mesmo compasso dezesseis vezes, e
    o ouvido registra 'loop' em menos de trinta segundos."""
    notas = BD.bateria(8, 480, "rock", virada_a_cada=4)
    bar = 480 * 4

    def perfil(c):
        return sorted((n.start - c * bar, n.pitch) for n in notas
                      if c * bar <= n.start < (c + 1) * bar
                      and n.pitch in (BD.CAIXA, BD.TOM_MEDIO, BD.TOM_GRAVE))

    assert perfil(3) != perfil(0)
    assert perfil(1) != perfil(3)


def test_bateria_marca_o_inicio_da_frase_com_prato():
    notas = BD.bateria(8, 480, "rock")
    pratos = sorted(n.start for n in notas if n.pitch == BD.PRATO)
    assert pratos and pratos[0] == 0


def test_notas_fantasma_sao_fracas():
    """A caixa nao toca so em 2 e 4: entre eles ela da toques quase inaudiveis,
    e e isso que da liga ao groove."""
    notas = BD.bateria(8, 480, "rock", fantasma=True)
    caixas = [n for n in notas if n.pitch == BD.CAIXA]
    fortes = [n.velocity for n in caixas if n.velocity > 60]
    fracas = [n.velocity for n in caixas if n.velocity <= 60]
    assert fracas and fortes and max(fracas) < min(fortes)


def test_sem_fantasma_a_caixa_fica_so_nos_tempos():
    notas = BD.bateria(4, 480, "rock", fantasma=False, virada_a_cada=0)
    caixas = {n.start % (480 * 4) for n in notas if n.pitch == BD.CAIXA}
    assert caixas == {480, 480 * 3}


@pytest.mark.parametrize("estilo", list(BD.ESTILOS_BATERIA))
def test_todos_os_estilos_produzem_bateria(estilo):
    notas = BD.bateria(4, 480, estilo)
    assert len(notas) > 8
    assert all(0 <= n.start < 480 * 16 for n in notas)
    assert all(1 <= n.velocity <= 127 for n in notas)


@pytest.mark.parametrize("padrao", list(BD.RIFFS))
def test_todos_os_riffs_produzem_notas(padrao):
    notas = BD.riff(_acordes(), 480, padrao, 1, 100)
    assert notas
    assert all(n.dur > 0 for n in notas)
    assert all(1 <= n.velocity <= 127 for n in notas)


def test_solo_fica_na_pentatonica():
    """Na pentatonica nao existe nota errada sobre harmonia de rock: faltam
    justamente os dois graus que brigam."""
    tonica = T.name_to_midi("E4") % 12
    notas = BD.solo_pentatonico(8, 480, tonica, "menor", 5, 104)
    graus = {(n.pitch - tonica) % 12 for n in notas}
    assert graus <= {0, 3, 5, 7, 10}, graus
    assert 55 <= min(n.pitch for n in notas) and max(n.pitch for n in notas) <= 88


def test_notas_geradas_sao_serializaveis():
    """Regressao: `rng.integers()` devolve int64, e int64 atravessa o render
    inteiro sem reclamar — o erro so aparece ao salvar o .chip, DEPOIS de meia
    hora de sintese. Toda nota que sai de um gerador tem que ser int de Python.
    """
    import json
    geradas = (BD.solo_pentatonico(8, 480, 4, "menor", 5, 100)
               + BD.bateria(8, 480, "rock")
               + BD.riff(_acordes(), 480, "galope", 1, 100)
               + BD.linha_de_baixo(_acordes(), 480, "colcheias", 1, 100))
    for n in geradas:
        for campo in (n.start, n.dur, n.pitch, n.velocity):
            assert type(campo) is int, f"{type(campo).__name__} vazou numa nota"
    json.dumps([[n.start, n.dur, n.pitch, n.velocity] for n in geradas])


def test_note_converte_numpy_na_entrada():
    """A blindagem: mesmo que um gerador novo esqueca de converter, a nota
    guarda int de Python."""
    n = Note(np.int64(480), np.int64(240), np.int32(60), np.int64(100))
    assert all(type(c) is int for c in (n.start, n.dur, n.pitch, n.velocity))
