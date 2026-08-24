"""
chipforge.audio.espaco
======================
Sala por CONVOLUCAO, largura estereo e eco.

Por que isto existe: o Freeverb de `fx.reverb` e oito filtros pente em
paralelo. Isso funciona para colar uma mixagem, mas oito pentes tem oito
frequencias de ressonancia, e o ouvido as escuta como um timbre metalico —
o "chiiii" que denuncia reverb barato assim que a orquestra abre um acorde
longo. Sala de concerto de verdade nao tem oito ressonancias: tem milhares,
densas demais para serem separadas.

O caminho para isso e convolucao. A resposta ao impulso aqui e SINTETIZADA,
nao gravada — ninguem precisa baixar 200 MB de IR:

    reflexoes iniciais discretas   <- dizem o TAMANHO da sala
    +
    cauda de ruido com decaimento  <- diz o MATERIAL da sala
      exponencial POR BANDA           (pedra segura o agudo, cortina nao)

Reflexoes iniciais separadas entre esquerda e direita sao o que da largura;
a cauda com decaimento por banda e o que diferencia catedral de estudio.

Convolucao roda por FFT com sobreposicao e soma: uma faixa de tres minutos
com cauda de tres segundos passa em poucos segundos.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import fx


def _stereo(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x if x.ndim == 2 else np.stack([x, x], axis=1)


# --------------------------------------------------------------------------
# Perfis de sala
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Sala:
    """Descreve uma sala pelo que se ouve nela, nao por coeficiente de filtro.

    rt_grave/rt_medio/rt_agudo: quantos segundos cada faixa leva para cair
    60 dB. Pedra nua segura o agudo (rt_agudo alto); madeira e publico comem
    o agudo primeiro (rt_agudo baixo) — e essa razao entre as tres, muito mais
    que o numero absoluto, e o que faz uma sala soar como um lugar.
    """
    nome: str
    predelay_ms: float
    rt_grave: float
    rt_medio: float
    rt_agudo: float
    densidade: float          # reflexoes iniciais por segundo
    espalhamento: float       # 0..1, quanto as iniciais se abrem no estereo
    brilho: float = 0.0       # dB de shelf no agudo da cauda
    descricao: str = ""


SALAS: dict[str, Sala] = {
    "estudio": Sala("Estudio", 8.0, 0.9, 0.75, 0.5, 900, 0.35, -1.0,
                    "Seco e curto. Cola sem colorir."),
    "camara": Sala("Camara", 14.0, 1.7, 1.45, 0.95, 700, 0.55, 0.0,
                   "Sala pequena de gravacao classica. Quarteto, harpa, solo."),
    "salao": Sala("Salao", 20.0, 2.4, 2.1, 1.3, 550, 0.7, 0.5,
                  "Sala de concerto media. O padrao para orquestra."),
    "hall": Sala("Hall", 26.0, 3.2, 2.8, 1.6, 450, 0.82, 0.5,
                 "Sala grande. Cauda longa sem virar nevoa."),
    "catedral": Sala("Catedral", 42.0, 6.0, 5.0, 2.2, 300, 0.95, -2.0,
                     "Pedra. Cauda enorme e escura — coro mora aqui."),
    "caverna": Sala("Caverna", 55.0, 4.5, 3.4, 1.1, 220, 1.0, -6.0,
                    "Grande, escura e irregular. Tensao, nao beleza."),
    "placa": Sala("Placa", 6.0, 2.2, 2.4, 2.0, 1600, 0.6, 3.5,
                  "Reverb de placa: denso desde o primeiro milissegundo, "
                  "brilhante. Voz e caixa."),
    "epico": Sala("Epico", 30.0, 4.0, 3.4, 1.9, 500, 0.9, 2.0,
                  "Sala grande com brilho empurrado. Trailer, nao concerto."),
}

SALAS_PT = {k: f"{v.nome} — {v.descricao}" for k, v in SALAS.items()}


# --------------------------------------------------------------------------
# Sintese da resposta ao impulso
# --------------------------------------------------------------------------
def _decaimento_por_banda(n: int, sr: int, rt: float) -> np.ndarray:
    """Envelope exponencial que cai 60 dB em `rt` segundos."""
    t = np.arange(n) / sr
    return np.exp(-6.907755 * t / max(rt, 0.05)).astype(np.float32)


def impulso(sala: str | Sala = "salao", sr: int = 44100,
            tamanho: float = 1.0, semente: int = 7) -> np.ndarray:
    """Sintetiza a resposta ao impulso estereo de uma sala.

    `tamanho` escala a cauda inteira: 0.5 e a mesma sala pela metade.
    A semente fixa deixa o render REPRODUTIVEL — duas exportacoes do mesmo
    projeto tem que dar o mesmo arquivo.
    """
    s = SALAS.get(sala, SALAS["salao"]) if isinstance(sala, str) else sala
    rng = np.random.default_rng(semente)

    rt_max = max(s.rt_grave, s.rt_medio, s.rt_agudo) * tamanho
    n = int(min(rt_max * 1.15, 8.0) * sr)
    pre = int(s.predelay_ms * sr / 1000)
    total = n + pre

    ir = np.zeros((total, 2), dtype=np.float32)

    # ---- reflexoes iniciais ------------------------------------------
    # Distribuidas com espacamento crescente: as primeiras chegam separadas
    # (paredes proximas), depois adensam ate virarem cauda. Se fossem
    # igualmente espacadas, o resultado seria um pente — de novo o metalico.
    n_ini = int(s.densidade * 0.09 * tamanho) + 8
    tempos = np.cumsum(rng.exponential(1.0 / s.densidade, n_ini)) * tamanho
    tempos = tempos[tempos < min(0.12 * tamanho + 0.02, n / sr)]
    for t in tempos:
        i = pre + int(t * sr)
        if i >= total - 1:
            break
        g = float(np.exp(-3.0 * t / max(0.09 * tamanho, 0.01)))
        lado = rng.uniform(-1.0, 1.0) * s.espalhamento
        sinal = 1.0 if rng.random() > 0.35 else -1.0
        ir[i, 0] += g * sinal * (0.5 + 0.5 * (1 - lado)) * 0.5
        # a mesma reflexao chega no outro ouvido alguns samples depois
        j = min(total - 1, i + int(abs(lado) * 0.0009 * sr))
        ir[j, 1] += g * sinal * (0.5 + 0.5 * (1 + lado)) * 0.5

    # ---- cauda difusa -------------------------------------------------
    # Ruido independente por canal. Independente e o ponto: se os dois lados
    # fossem o mesmo ruido, a cauda colapsaria no centro e a sala sumiria.
    env_g = _decaimento_por_banda(n, sr, s.rt_grave * tamanho)
    env_m = _decaimento_por_banda(n, sr, s.rt_medio * tamanho)
    env_a = _decaimento_por_banda(n, sr, s.rt_agudo * tamanho)
    # a cauda nao comeca cheia: cresce enquanto as reflexoes ainda estao
    # discretas, e e essa subida que faz a sala "abrir" em vez de aparecer
    subida = np.clip(np.arange(n) / max(1, int(0.05 * tamanho * sr)), 0, 1)

    for canal in (0, 1):
        ruido = rng.normal(0, 1, n).astype(np.float32)
        grave = fx.biquad(ruido, sr, "lowpass", 400.0, 0.7)
        agudo = fx.biquad(ruido, sr, "highpass", 3000.0, 0.7)
        medio = ruido - grave - agudo
        cauda = grave * env_g + medio * env_m + agudo * env_a
        cauda = cauda * subida * 0.32
        ir[pre:pre + n, canal] += cauda

    if s.brilho:
        for canal in (0, 1):
            ir[:, canal] = fx.biquad(ir[:, canal], sr, "highshelf",
                                     4200.0, 0.7, s.brilho)

    # tira o DC que o ruido filtrado deixa: DC numa IR vira um "puf" grave
    ir -= ir.mean(axis=0, keepdims=True)
    for canal in (0, 1):
        ir[:, canal] = fx.biquad(ir[:, canal], sr, "highpass", 28.0, 0.7)

    # normaliza por ENERGIA, nao por pico: e a energia que decide o quanto a
    # sala vai empurrar o mix, e ela varia dez vezes entre estudio e catedral
    energia = float(np.sqrt((ir.astype(np.float64) ** 2).sum() / 2))
    if energia > 1e-9:
        ir /= energia
    return ir.astype(np.float32)


# --------------------------------------------------------------------------
# Convolucao
# --------------------------------------------------------------------------
def convolver(x: np.ndarray, ir: np.ndarray, bloco: int = 32768) -> np.ndarray:
    """Convolucao estereo por FFT com sobreposicao e soma.

    Convolucao direta de 8 milhoes de amostras por 130 mil seria 10^12
    multiplicacoes. Por FFT, em blocos, sao alguns segundos.
    """
    st = _stereo(x).astype(np.float32)
    ir = _stereo(ir).astype(np.float32)
    n, m = len(st), len(ir)
    if m < 2:
        return st

    nfft = 1
    while nfft < bloco + m - 1:
        nfft *= 2
    passo = nfft - m + 1

    saida = np.zeros((n + m - 1, 2), dtype=np.float32)
    for canal in (0, 1):
        H = np.fft.rfft(ir[:, canal], nfft)
        for ini in range(0, n, passo):
            trecho = st[ini:ini + passo, canal]
            y = np.fft.irfft(np.fft.rfft(trecho, nfft) * H, nfft)
            fim = min(ini + len(y), len(saida))
            saida[ini:fim, canal] += y[:fim - ini].astype(np.float32)
    return saida[:n + m - 1]


_CACHE_IR: dict[tuple, np.ndarray] = {}


def reverb(x: np.ndarray, sr: int = 44100, sala: str = "salao",
           tamanho: float = 1.0, wet: float = 1.0, dry: float = 0.0,
           corte_grave: float = 120.0, corte_agudo: float = 9000.0,
           semente: int = 7) -> np.ndarray:
    """Sala por convolucao, pronta para o barramento de envio.

    `corte_grave` e o parametro mais subestimado de uma mixagem: mandar o
    contrabaixo inteiro para a sala e o que transforma orquestra em lama. Aqui
    o grave entra na sala filtrado, o que preserva a definicao do grave seco.
    """
    st = _stereo(x)
    chave = (sala, sr, round(tamanho, 3), semente)
    ir = _CACHE_IR.get(chave)
    if ir is None:
        ir = impulso(sala, sr, tamanho, semente)
        _CACHE_IR[chave] = ir

    entrada = st
    if corte_grave > 20:
        entrada = np.stack([fx.biquad(entrada[:, c], sr, "highpass",
                                      corte_grave, 0.7) for c in (0, 1)], axis=1)
    if corte_agudo < sr * 0.45:
        entrada = np.stack([fx.biquad(entrada[:, c], sr, "lowpass",
                                      corte_agudo, 0.7) for c in (0, 1)], axis=1)

    molhado = convolver(entrada, ir)[:len(st)]
    if len(molhado) < len(st):
        molhado = np.vstack([molhado,
                             np.zeros((len(st) - len(molhado), 2), np.float32)])
    return (dry * st + wet * molhado).astype(np.float32)


# --------------------------------------------------------------------------
# Largura estereo
# --------------------------------------------------------------------------
def largura(x: np.ndarray, fator: float = 1.4,
            monofonico_abaixo: float = 140.0, sr: int = 44100) -> np.ndarray:
    """Alarga o estereo em meio/lado, mantendo o grave no centro.

    Alargar o grave junto e o erro classico: grave fora de fase soma errado no
    mono (e todo alto-falante de celular e mono) e faz o pico do limitador
    disparar sem nenhum ganho de volume percebido. Aqui o que esta abaixo de
    `monofonico_abaixo` fica intocado.
    """
    st = _stereo(x)
    if monofonico_abaixo > 20:
        grave = np.stack([fx.biquad(st[:, c], sr, "lowpass",
                                    monofonico_abaixo, 0.7) for c in (0, 1)], axis=1)
        alto = st - grave
        grave_mono = grave.mean(axis=1)
        grave = np.stack([grave_mono, grave_mono], axis=1)
    else:
        alto, grave = st, np.zeros_like(st)

    meio = (alto[:, 0] + alto[:, 1]) * 0.5
    lado = (alto[:, 0] - alto[:, 1]) * 0.5 * float(fator)
    return (np.stack([meio + lado, meio - lado], axis=1) + grave).astype(np.float32)


def haas(x: np.ndarray, sr: int = 44100, ms: float = 14.0,
         lado: int = 1, mix: float = 0.5) -> np.ndarray:
    """Copia atrasada num canal so. Um som mono vira largo sem nenhum filtro —
    e o efeito de precedencia: o ouvido localiza pela PRIMEIRA chegada e usa
    a segunda so para julgar tamanho."""
    st = _stereo(x)
    d = max(1, int(ms * sr / 1000))
    y = st.copy()
    atrasado = np.vstack([np.zeros((d, 2), np.float32), st])[:len(st)]
    c = 1 if lado >= 0 else 0
    y[:, c] = (1 - mix) * st[:, c] + mix * atrasado[:, c]
    return y.astype(np.float32)


def eco(x: np.ndarray, sr: int = 44100, tempo_s: float = 0.35,
        feedback: float = 0.38, mix: float = 0.28,
        pingue_pongue: bool = True, corte_agudo: float = 5200.0) -> np.ndarray:
    """Eco estereo. Cada repeticao volta mais escura, como no mundo real —
    eco que repete com o mesmo brilho soa como falha de fita digital."""
    st = _stereo(x)
    n = len(st)
    d = max(1, int(tempo_s * sr))
    saida = st.copy()
    tap = st.mean(axis=1)
    g = 1.0
    for k in range(1, 12):
        g *= feedback
        if g < 0.004:
            break
        tap = fx.biquad(tap, sr, "lowpass", corte_agudo, 0.7)
        off = d * k
        if off >= n:
            break
        c = (k % 2) if pingue_pongue else None
        trecho = tap[:n - off] * g * mix
        if c is None:
            saida[off:, 0] += trecho
            saida[off:, 1] += trecho
        else:
            saida[off:, c] += trecho * 1.4
    return saida.astype(np.float32)
