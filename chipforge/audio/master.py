"""
chipforge.audio.master
======================
Medicao e masterizacao: LUFS, faixa de loudness, pico real, compressor
multibanda, excitador de harmonicos, meio/lado e limitador com antecipacao.

Por que isto existe: o `limiter` de `fx.py` segura o pico e so. Pico nao e
volume — duas faixas com o mesmo pico podem ter 8 dB de diferenca no que o
ouvido chama de alto. Sem medir em LUFS nao da para dizer "esta faixa esta no
nivel de uma trilha comercial", so "esta faixa nao esta estourando".

E ha o efeito colateral que aparece em qualquer render deste app: uma
mixagem so de samples de SoundFont nao tem energia acima de 7 kHz, porque o
banco ja veio filtrado. Ganho de EQ ali nao resolve — nao da para levantar o
que nao existe. Quem cria conteudo naquela faixa e o excitador: ele gera
harmonicos NOVOS a partir do medio-agudo que ja esta la.

Referencias de medicao: ITU-R BS.1770 (LUFS com portao) e EBU R128 (LRA).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import fx

_LOG = 10.0


def _stereo(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x if x.ndim == 2 else np.stack([x, x], axis=1)


# ==========================================================================
# Medicao — ITU-R BS.1770
# ==========================================================================
def _k_weight(x: np.ndarray, sr: int) -> np.ndarray:
    """Filtro K: um shelf que imita a cabeca e um corte de grave.

    A ponderacao K existe porque o ouvido nao conta grave e agudo igual. Sem
    ela, um bumbo forte mede como faixa alta e um vocal brilhante mede como
    faixa baixa — exatamente ao contrario do que se escuta.
    """
    y = fx.biquad(x, sr, "highshelf", 1681.97, 0.7071752, 3.999)
    return fx.biquad(y, sr, "highpass", 38.13, 0.5003271)


_PESO_CANAL = (1.0, 1.0)          # esquerdo e direito; surround teria mais


def _blocos_loudness(st: np.ndarray, sr: int, janela_s: float,
                     sobreposicao: float = 0.75):
    """Loudness por bloco, em LKFS, sem portao."""
    n_j = max(1, int(janela_s * sr))
    passo = max(1, int(n_j * (1.0 - sobreposicao)))
    if len(st) < n_j:
        st = np.vstack([st, np.zeros((n_j - len(st), 2), np.float32)])
    k = np.stack([_k_weight(st[:, c], sr) for c in (0, 1)], axis=1)
    quadrado = k.astype(np.float64) ** 2
    inicios = np.arange(0, len(k) - n_j + 1, passo)
    if not len(inicios):
        inicios = np.array([0])
    soma = np.cumsum(np.vstack([np.zeros((1, 2)), quadrado]), axis=0)
    media = (soma[inicios + n_j] - soma[inicios]) / n_j
    z = _PESO_CANAL[0] * media[:, 0] + _PESO_CANAL[1] * media[:, 1]
    return -0.691 + 10 * np.log10(np.maximum(z, 1e-12))


def lufs_integrado(x: np.ndarray, sr: int = 44100) -> float:
    """LUFS integrado com os dois portoes do BS.1770.

    O portao relativo e o que impede que o silencio entre as frases puxe a
    medida para baixo: so entra na conta o que esta a menos de 10 LU do nivel
    medio da propria faixa.
    """
    st = _stereo(x)
    l = _blocos_loudness(st, sr, 0.400)
    absoluto = l[l > -70.0]
    if not len(absoluto):
        return -np.inf
    energia = 10 ** ((absoluto + 0.691) / 10)
    limiar = -0.691 + 10 * np.log10(energia.mean()) - 10.0
    sel = absoluto[absoluto > limiar]
    if not len(sel):
        sel = absoluto
    e = (10 ** ((sel + 0.691) / 10)).mean()
    return float(-0.691 + 10 * np.log10(max(e, 1e-12)))


def faixa_de_loudness(x: np.ndarray, sr: int = 44100) -> float:
    """LRA (EBU R128): quantos LU separam os trechos baixos dos altos.

    E o numero que diz se a musica tem ARCO. Trilha cinematica costuma ficar
    entre 6 e 12 LU; abaixo de 4, tudo soa no mesmo volume do inicio ao fim.
    """
    st = _stereo(x)
    l = _blocos_loudness(st, sr, 3.0, 0.90)
    l = l[l > -70.0]
    if len(l) < 4:
        return 0.0
    energia = 10 ** ((l + 0.691) / 10)
    limiar = -0.691 + 10 * np.log10(energia.mean()) - 20.0
    sel = l[l > limiar]
    if len(sel) < 4:
        sel = l
    return float(np.percentile(sel, 95) - np.percentile(sel, 10))


def pico_real_db(x: np.ndarray, sr: int = 44100, fator: int = 4) -> float:
    """Pico REAL (dBTP): o sinal entre as amostras pode passar do que as
    amostras mostram. Um mix que mede -0.1 dBFS pode bater +0.6 dBTP e distorcer
    no conversor de qualquer aparelho — e no encoder de MP3, sempre."""
    st = _stereo(x)
    picos = []
    for c in (0, 1):
        canal = st[:, c].astype(np.float64)
        n = len(canal)
        if n < 8:
            picos.append(np.abs(canal).max() if n else 0.0)
            continue
        # sobreamostragem por zero-padding no dominio da frequencia: e
        # interpolacao sinc exata, que e justamente o que o conversor faz
        bloco = 1 << 16
        maior = 0.0
        for ini in range(0, n, bloco):
            trecho = canal[ini:ini + bloco]
            if len(trecho) < 8:
                maior = max(maior, float(np.abs(trecho).max()))
                continue
            F = np.fft.rfft(trecho)
            G = np.zeros(len(trecho) * fator // 2 + 1, dtype=complex)
            G[:len(F)] = F
            maior = max(maior, float(np.abs(np.fft.irfft(G, len(trecho) * fator)).max() * fator))
        picos.append(maior)
    p = max(picos) if picos else 0.0
    return float(20 * np.log10(max(p, 1e-9)))


BANDAS = [
    ("sub", 20, 80), ("grave", 80, 250), ("medio-grave", 250, 800),
    ("medio", 800, 2500), ("agudo", 2500, 7000), ("brilho", 7000, 11000),
]


def balanco_espectral(x: np.ndarray, sr: int = 44100) -> dict[str, float]:
    """Nivel medio por banda, em dB relativos a banda mais forte."""
    mono = _stereo(x).mean(axis=1)
    n = 4096
    if len(mono) < n:
        mono = np.pad(mono, (0, n - len(mono)))
    salto = n // 2
    quadros = np.lib.stride_tricks.sliding_window_view(mono, n)[::salto]
    if len(quadros) > 900:
        quadros = quadros[:: len(quadros) // 900 + 1]
    esp = np.abs(np.fft.rfft(quadros * np.hanning(n), axis=1))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    cru = {}
    for nome, lo, hi in BANDAS:
        m = (freqs >= lo) & (freqs < hi)
        cru[nome] = float((esp[:, m] ** 2).mean())
    topo = max(cru.values()) or 1.0
    return {k: round(10 * np.log10(max(v, 1e-12) / topo), 1) for k, v in cru.items()}


def correlacao(x: np.ndarray) -> float:
    """+1 = mono, 0 = largo, negativo = os canais se cancelam no mono."""
    st = _stereo(x)
    a, b = st[:, 0].astype(np.float64), st[:, 1].astype(np.float64)
    d = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / d) if d > 1e-12 else 1.0


@dataclass
class Medida:
    lufs: float
    lra: float
    pico_dbtp: float
    pico_dbfs: float
    correlacao: float
    bandas: dict = field(default_factory=dict)
    duracao: float = 0.0

    def relatorio_pt(self, alvo_lufs: float = -14.0) -> str:
        L = [
            f"Duracao      : {int(self.duracao // 60)}:{self.duracao % 60:04.1f}",
            f"Loudness     : {self.lufs:6.1f} LUFS   "
            f"(alvo {alvo_lufs:.0f}; diferenca {self.lufs - alvo_lufs:+.1f} LU)",
            f"Faixa (LRA)  : {self.lra:6.1f} LU     "
            f"{'— arco de verdade' if self.lra >= 6 else '— achatado, falta arco' if self.lra < 4 else '— aceitavel'}",
            f"Pico real    : {self.pico_dbtp:6.1f} dBTP   "
            f"{'ESTOURANDO' if self.pico_dbtp > -0.9 else 'ok'}",
            f"Pico amostra : {self.pico_dbfs:6.1f} dBFS",
            f"Correlacao   : {self.correlacao:6.2f}      "
            f"{'estreito demais' if self.correlacao > 0.95 else 'cuidado com o mono' if self.correlacao < 0.1 else 'ok'}",
            "",
            "Balanco espectral:",
        ]
        for nome, db in self.bandas.items():
            L.append(f"  {nome:12s} {db:6.1f} dB  {'█' * int(np.clip(48 + db, 0, 48))}")
        return "\n".join(L)


def medir(x: np.ndarray, sr: int = 44100) -> Medida:
    st = _stereo(x)
    return Medida(
        lufs=lufs_integrado(st, sr),
        lra=faixa_de_loudness(st, sr),
        pico_dbtp=pico_real_db(st, sr),
        pico_dbfs=float(20 * np.log10(max(float(np.abs(st).max()), 1e-9))),
        correlacao=correlacao(st),
        bandas=balanco_espectral(st, sr),
        duracao=len(st) / sr,
    )


# ==========================================================================
# Processadores
# ==========================================================================
def _lr4(x: np.ndarray, sr: int, tipo: str, freq: float) -> np.ndarray:
    """Linkwitz-Riley de 4a ordem = dois Butterworth em cascata.

    A propriedade que importa: as duas metades somadas de volta dao resposta
    plana. Um crossover qualquer deixa um buraco ou um pico na frequencia de
    corte, e num multibanda esse erro aparece na faixa inteira.
    """
    y = fx.biquad(x, sr, tipo, freq, 0.7071)
    return fx.biquad(y, sr, tipo, freq, 0.7071)


def dividir_bandas(x: np.ndarray, sr: int, cortes=(220.0, 2200.0)):
    """Divide em N+1 bandas com crossovers Linkwitz-Riley."""
    st = _stereo(x)
    bandas, resto = [], st
    for f in cortes:
        baixa = np.stack([_lr4(resto[:, c], sr, "lowpass", f) for c in (0, 1)], axis=1)
        alta = np.stack([_lr4(resto[:, c], sr, "highpass", f) for c in (0, 1)], axis=1)
        bandas.append(baixa.astype(np.float32))
        resto = alta
    bandas.append(resto.astype(np.float32))
    return bandas


def compressor_multibanda(x: np.ndarray, sr: int = 44100,
                          cortes=(220.0, 2200.0), ajustes=None) -> np.ndarray:
    """Compressor por banda.

    O que um compressor de banda larga faz de errado numa trilha: o bumbo
    abaixa a orquestra inteira a cada tempo. Separando as bandas, o grave
    controla o grave e o medio segue livre — e a orquestra para de "respirar"
    junto com a percussao.
    """
    ajustes = ajustes or [
        {"limiar_db": -22.0, "razao": 2.6, "attack_ms": 22.0, "release_ms": 180.0},
        {"limiar_db": -20.0, "razao": 2.0, "attack_ms": 14.0, "release_ms": 150.0},
        {"limiar_db": -24.0, "razao": 1.8, "attack_ms": 6.0, "release_ms": 110.0},
    ]
    bandas = dividir_bandas(x, sr, cortes)
    saida = np.zeros_like(bandas[0])
    for banda, cfg in zip(bandas, ajustes):
        saida += fx.compressor(banda, sr, **cfg)
    return saida.astype(np.float32)


def excitador(x: np.ndarray, sr: int = 44100, freq: float = 3800.0,
              quantidade: float = 0.35, mix: float = 0.30) -> np.ndarray:
    """Gera harmonicos ACIMA de `freq` a partir do que ja existe abaixo dela.

    Isto nao e um EQ. EQ so multiplica o que ja esta la — e um banco de
    samples de SoundFont simplesmente nao TEM conteudo acima de 8 kHz para
    multiplicar. O excitador cria: satura uma copia da banda de cima, o que
    produz harmonicos em 2f e 3f, e devolve so a parte nova.

    A saturacao e assimetrica de proposito: assimetria gera harmonico de
    ordem PAR, que o ouvido lê como "brilho", enquanto so ordem impar soa
    como distorcao.
    """
    st = _stereo(x)
    saida = np.empty_like(st)
    for c in (0, 1):
        alto = fx.biquad(st[:, c], sr, "highpass", freq, 0.7)
        gerado = np.tanh(alto * (1.0 + 14.0 * quantidade)) - np.tanh(alto * 0.15)
        gerado = fx.biquad(gerado, sr, "highpass", freq * 1.5, 0.7)
        gerado = fx.biquad(gerado, sr, "lowpass", min(sr * 0.45, 17000.0), 0.7)
        saida[:, c] = st[:, c] + gerado * mix
    return saida.astype(np.float32)


def eq_meio_lado(x: np.ndarray, sr: int = 44100,
                 meio=(), lado=()) -> np.ndarray:
    """EQ separado para o que esta no centro e o que esta nas bordas.

    Serve para o problema mais comum de mixagem densa: o centro (voz, baixo,
    caixa) precisa de espaco no medio, e as bordas (sala, cordas, coro)
    precisam de brilho. Fazer os dois no estereo junto e impossivel — um
    desfaz o outro.
    """
    st = _stereo(x)
    m = (st[:, 0] + st[:, 1]) * 0.5
    s = (st[:, 0] - st[:, 1]) * 0.5
    for spec in meio:
        m = fx.biquad(m, sr, spec.get("modo", "peak"), spec["freq"],
                      spec.get("q", 0.9), spec.get("ganho_db", 0.0))
    for spec in lado:
        s = fx.biquad(s, sr, spec.get("modo", "peak"), spec["freq"],
                      spec.get("q", 0.9), spec.get("ganho_db", 0.0))
    return np.stack([m + s, m - s], axis=1).astype(np.float32)


def limitador(x: np.ndarray, sr: int = 44100, teto_db: float = -1.0,
              antecipacao_ms: float = 3.0, release_ms: float = 90.0,
              ganho_db: float = 0.0) -> np.ndarray:
    """Limitador com ANTECIPACAO: ve o pico chegar antes de ele chegar.

    Um limitador sem antecipacao so reage depois que o pico ja passou, e o
    jeito de esconder isso e um ataque tao rapido que distorce o grave. Com
    3 ms de antecipacao o ganho ja esta abaixado quando o transiente chega, e
    o abaixamento pode ser suave — que e o que separa "alto" de "esmagado".
    """
    st = _stereo(x) * (10 ** (ganho_db / 20))
    teto = 10 ** (teto_db / 20)
    n = len(st)
    la = max(1, int(antecipacao_ms * sr / 1000))

    det = np.abs(st).max(axis=1)
    # maximo deslizante na janela de antecipacao
    pad = np.concatenate([det, np.full(la, det[-1] if n else 0.0)])
    janela = np.lib.stride_tricks.sliding_window_view(pad, la + 1)[:n]
    alvo = np.maximum(janela.max(axis=1), 1e-9)

    ganho = np.minimum(1.0, teto / alvo)
    # suaviza so a VOLTA. Descer rapido e o trabalho do limitador; subir rapido
    # e o que gera a modulacao de grave que soa como bombeamento.
    a = float(np.exp(-1.0 / (sr * release_ms / 1000)))
    from .fx import _filt
    suave = _filt([1 - a], [1.0, -a], ganho)
    ganho = np.minimum(ganho, suave)
    # a janela curta de antecipacao ainda deixa degraus; um passa-baixa curto
    # arredonda a curva de ganho sem deixar o pico escapar
    k = max(3, la // 2)
    ganho = np.convolve(ganho, np.ones(k) / k, mode="same")
    ganho = np.minimum(ganho, np.minimum(1.0, teto / alvo))

    y = st * ganho[:, None]
    return np.clip(y, -teto, teto).astype(np.float32)


def casar_perfil(x: np.ndarray, sr: int = 44100, alvo: dict | None = None,
                 maximo_db: float = 4.5, passadas: int = 3) -> np.ndarray:
    """Aproxima o balanco espectral do mix de um perfil de referencia.

    Nao e "clonar a referencia": sao seis filtros largos, limitados a poucos
    dB, corrigindo o que uma medicao ja mostrou estar fora — a mesma coisa que
    se faz de ouvido comparando com uma faixa que se conhece, so que sem o
    cansaco de ouvido que faz a gente exagerar depois de dez minutos.
    """
    if not alvo:
        return _stereo(x)
    centros = {"sub": 50, "grave": 140, "medio-grave": 450,
               "medio": 1400, "agudo": 4200, "brilho": 8800}
    modos = {"sub": "lowshelf", "brilho": "highshelf"}
    y = _stereo(x)
    for _ in range(max(1, passadas)):
        y, mexeu = _uma_passada(y, sr, alvo, centros, modos, maximo_db)
        if not mexeu:
            break
    return y.astype(np.float32)


def _uma_passada(y, sr, alvo, centros, modos, maximo_db):
    """Uma correcao, com o teto de dB aplicado por filtro.

    O teto existe para que cada filtro continue suave; a consequencia e que um
    desvio maior que ele nao cabe numa passada so. Medir de novo e corrigir o
    residuo resolve sem afiar nenhum filtro: num caso real o erro total caiu de
    10,8 dB para 5,1 em tres passadas, com o brilho indo de +7,8 para +3,6.
    """
    atual = balanco_espectral(y, sr)
    mexeu = False
    for nome, f in centros.items():
        if nome not in alvo or nome not in atual:
            continue
        delta = float(np.clip(alvo[nome] - atual[nome], -maximo_db, maximo_db))
        if abs(delta) < 0.4:
            continue
        mexeu = True
        modo = modos.get(nome, "peak")
        # Q baixo de proposito: cada banda destas cobre mais de uma oitava, e
        # um filtro estreito corrige o centro dela e deixa as pontas como
        # estavam — a medida melhora no papel e o som nao muda.
        q = 0.7 if modo != "peak" else 0.6
        y = np.stack([fx.biquad(y[:, c], sr, modo, f, q, delta)
                      for c in (0, 1)], axis=1)
    return y.astype(np.float32), mexeu


# ==========================================================================
# A cadeia inteira
# ==========================================================================
@dataclass
class Masterizacao:
    audio: np.ndarray
    antes: Medida
    depois: Medida
    ganho_aplicado_db: float

    def relatorio_pt(self, alvo_lufs: float = -14.0) -> str:
        return (f"ANTES\n{self.antes.relatorio_pt(alvo_lufs)}\n\n"
                f"(ganho de master: {self.ganho_aplicado_db:+.1f} dB)\n\n"
                f"DEPOIS\n{self.depois.relatorio_pt(alvo_lufs)}")


# `freq_excitador` e a decisao mais consequente de todo o perfil. Um banco de
# SoundFont ja vem filtrado por volta de 6-8 kHz, entao a faixa que falta numa
# mixagem so de samples nao e o "ar" de 10 kHz — e a PRESENCA, de 2,5 a 7 kHz,
# onde moram o ataque do arco, o metal do prato e a consoante da voz. Excitar
# a partir de 2,4 kHz gera harmonicos justamente ali; excitar a partir de
# 3,8 kHz joga tudo acima de 7 kHz e produz uma faixa chiada e ainda opaca.
PERFIS = {
    "cinematico": dict(
        excitar=0.34, mix_excitador=0.20, freq_excitador=2400.0, largura=1.25,
        multibanda=True, ar_db=0.5, presenca_db=2.5, corpo_db=0.8,
        alvo_lufs=-14.0, teto_db=-1.0),
    "trailer": dict(
        excitar=0.45, mix_excitador=0.40, freq_excitador=2600.0, largura=1.35,
        multibanda=True, ar_db=1.5, presenca_db=3.0, corpo_db=1.5,
        alvo_lufs=-11.0, teto_db=-1.0),
    "jogo": dict(
        excitar=0.28, mix_excitador=0.26, freq_excitador=2800.0, largura=1.15,
        multibanda=True, ar_db=0.5, presenca_db=1.5, corpo_db=0.5,
        alvo_lufs=-16.0, teto_db=-1.5),
    # Rock nao e trilha orquestral com guitarra em cima. A diferenca de perfil
    # e concreta: menos largura (guitarra dobrada ja e larga, e alargar mais
    # esvazia o centro onde moram voz, caixa e baixo), mais medio, e o alvo de
    # loudness mais alto — ninguem ouve rock a -14 LUFS.
    "rock": dict(
        excitar=0.30, mix_excitador=0.22, freq_excitador=2800.0, largura=1.08,
        multibanda=True, ar_db=0.5, presenca_db=2.0, corpo_db=1.2,
        alvo_lufs=-11.0, teto_db=-1.0),
    "chiptune": dict(
        excitar=0.0, mix_excitador=0.0, largura=1.0,
        multibanda=False, ar_db=0.0, corpo_db=0.0,
        alvo_lufs=-14.0, teto_db=-1.0),
    "cru": dict(
        excitar=0.0, mix_excitador=0.0, largura=1.0,
        multibanda=False, ar_db=0.0, corpo_db=0.0,
        alvo_lufs=None, teto_db=-1.0),
}


def masterizar(x: np.ndarray, sr: int = 44100, perfil: str = "cinematico",
               alvo_lufs: float | None = None, teto_db: float | None = None,
               perfil_espectral: dict | None = None,
               medir_antes: bool = True) -> Masterizacao:
    """Cadeia de master, na ordem que faz sentido.

    A ordem nao e negociavel e e a razao de isto ser uma funcao e nao seis:
    casar o espectro ANTES de comprimir (senao o compressor reage ao erro),
    excitar antes de alargar (para o brilho novo tambem ganhar largura), e o
    limitador por ultimo, sempre — qualquer coisa depois dele desfaz o teto.
    """
    cfg = dict(PERFIS.get(perfil, PERFIS["cinematico"]))
    if alvo_lufs is not None:
        cfg["alvo_lufs"] = alvo_lufs
    if teto_db is not None:
        cfg["teto_db"] = teto_db

    st = _stereo(x)
    antes = medir(st, sr) if medir_antes else Medida(0, 0, 0, 0, 0, {}, len(st) / sr)

    # Espaco de manobra na entrada. Uma mixagem que chega colada em 0 dBFS ja
    # distorceu antes de qualquer coisa aqui, e todo processador nao-linear
    # abaixo (multibanda, excitador) reagiria a distorcao em vez do sinal. Como
    # o alvo e LUFS, e nao pico, baixar aqui nao custa volume nenhum no fim.
    pico_entrada = float(np.abs(st).max())
    if pico_entrada > 0.5:
        st = st * (0.5 / pico_entrada)

    y = st
    if cfg.get("corpo_db"):
        y = eq_meio_lado(y, sr,
                         meio=({"modo": "lowshelf", "freq": 110, "q": 0.7,
                                "ganho_db": cfg["corpo_db"]},),
                         lado=({"modo": "highshelf", "freq": 6000, "q": 0.7,
                                "ganho_db": cfg.get("ar_db", 0.0)},))

    if cfg.get("multibanda"):
        y = compressor_multibanda(y, sr)

    if cfg.get("excitar"):
        y = excitador(y, sr, cfg.get("freq_excitador", 2600.0),
                      cfg["excitar"], cfg["mix_excitador"])
    if cfg.get("presenca_db"):
        y = np.stack([fx.biquad(y[:, c], sr, "peak", 3400.0, 0.8,
                                cfg["presenca_db"]) for c in (0, 1)], axis=1)

    if cfg.get("largura", 1.0) != 1.0:
        from . import espaco
        y = espaco.largura(y, cfg["largura"], 140.0, sr)

    # O casamento de perfil vem DEPOIS do excitador, e nao antes.
    #
    # Antes era o contrario, e o resultado media 13 dB de brilho acima da
    # referencia: o casamento equilibrava o espectro, o excitador despejava
    # harmonicos novos por cima e desfazia o equilibrio. A ordem certa segue a
    # dependencia — o excitador CRIA conteudo onde nao havia, o casamento
    # ESCULPE o que existe, e nao da para esculpir o que ainda nao chegou.
    if perfil_espectral:
        y = casar_perfil(y, sr, perfil_espectral, maximo_db=7.0)

    ganho_db = 0.0
    if cfg.get("alvo_lufs") is not None:
        atual = lufs_integrado(y, sr)
        if np.isfinite(atual):
            # o limitador ainda vai comer 1 a 2 dB do que for empurrado;
            # deixar 1 dB de margem evita a segunda passada de ajuste
            ganho_db = float(np.clip(cfg["alvo_lufs"] - atual - 1.0, -24.0, 24.0))

    y = limitador(y, sr, cfg["teto_db"], 3.0, 90.0, ganho_db)

    # uma correcao final so se ainda estiver longe: sem isto, faixa muito
    # comprimida na entrada erra o alvo por 2 LU
    if cfg.get("alvo_lufs") is not None:
        atual = lufs_integrado(y, sr)
        erro = cfg["alvo_lufs"] - atual
        if np.isfinite(atual) and abs(erro) > 0.5:
            y = limitador(y, sr, cfg["teto_db"], 3.0, 90.0, float(np.clip(erro, -6, 6)))
            ganho_db += float(np.clip(erro, -6, 6))

    # Teto REAL, nao teto de amostra. O limitador segura o que ele consegue
    # ver — as amostras — mas a onda reconstruida entre elas passa por cima, e
    # e essa onda que o conversor e o encoder de MP3 vao produzir.
    #
    # A correcao NAO e abaixar a faixa inteira pelo excedente: isso custou
    # 3 LU de volume numa medicao real, porque paga com a musica toda o que
    # apenas alguns transientes deviam. Em vez disso, o limitador roda de novo
    # com o teto interno mais baixo — ele desce so onde ha pico, e o resto do
    # material fica onde estava.
    interno = cfg["teto_db"]
    for _ in range(6):
        tp = pico_real_db(y, sr)
        if not np.isfinite(tp) or tp <= cfg["teto_db"]:
            break
        # desce o teto INTERNO de forma acumulativa. Recalcular a partir do
        # teto original a cada volta fazia o laco andar em circulo e sair
        # ainda por cima do limite; o excedente de cada passada tem que somar
        # com o que ja foi descontado nas anteriores.
        interno -= (tp - cfg["teto_db"]) + 0.15
        y = limitador(y, sr, interno, 3.0, 90.0, 0.0)

    return Masterizacao(y.astype(np.float32), antes, medir(y, sr), ganho_db)
