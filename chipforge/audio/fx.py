"""
chipforge.audio.fx
==================
A cadeia de estudio: filtros biquad, reverb, simulador de amplificador,
chorus e compressor.

Por que isto existe: o sampler entrega a nota crua, mono e seca. Orquestra
sem sala soa como MIDI de 1998, e guitarra sem amplificador soa como uma
guitarra desplugada — nao e o banco de samples que falta, e o ar em volta.

Cada efeito e descrito por um dicionario serializavel, entao a cadeia inteira
entra no arquivo de projeto:

    inserts = [{"tipo": "amp", "ganho": 9.0, "cab": 4200},
               {"tipo": "eq", "freq": 2500, "ganho_db": 3.5, "q": 0.9}]
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.signal import lfilter as _lfilter
except Exception:                       # pragma: no cover
    _lfilter = None


def _filt(b, a, x: np.ndarray) -> np.ndarray:
    """Aplica um filtro racional. Sem scipy cai num laco direto (lento, mas
    o resultado e identico)."""
    x = np.asarray(x, dtype=np.float64)
    if _lfilter is not None:
        return _lfilter(np.asarray(b, float), np.asarray(a, float), x, axis=0)
    b = np.asarray(b, float) / a[0]
    a = np.asarray(a, float) / a[0]
    y = np.zeros_like(x)
    for n in range(len(x)):
        acc = 0.0
        for i in range(len(b)):
            if n - i >= 0:
                acc += b[i] * x[n - i]
        for j in range(1, len(a)):
            if n - j >= 0:
                acc -= a[j] * y[n - j]
        y[n] = acc
    return y


def _stereo(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 2 else np.stack([x, x], axis=1)


# ==========================================================================
# Biquads (formulas do cookbook de Robert Bristow-Johnson)
# ==========================================================================
def _biquad(tipo: str, sr: int, freq: float, q: float = 0.707,
            ganho_db: float = 0.0):
    w0 = 2 * np.pi * max(1.0, min(freq, sr * 0.49)) / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2 * q)
    A = 10 ** (ganho_db / 40)

    if tipo == "lowpass":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif tipo == "highpass":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif tipo == "peak":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    elif tipo == "lowshelf":
        s = 2 * np.sqrt(A) * alpha
        b = [A * ((A + 1) - (A - 1) * cw + s),
             2 * A * ((A - 1) - (A + 1) * cw),
             A * ((A + 1) - (A - 1) * cw - s)]
        a = [(A + 1) + (A - 1) * cw + s,
             -2 * ((A - 1) + (A + 1) * cw),
             (A + 1) + (A - 1) * cw - s]
    elif tipo == "highshelf":
        s = 2 * np.sqrt(A) * alpha
        b = [A * ((A + 1) + (A - 1) * cw + s),
             -2 * A * ((A - 1) + (A + 1) * cw),
             A * ((A + 1) + (A - 1) * cw - s)]
        a = [(A + 1) - (A - 1) * cw + s,
             2 * ((A - 1) - (A + 1) * cw),
             (A + 1) - (A - 1) * cw - s]
    else:
        raise ValueError(f"biquad desconhecido: {tipo}")
    return b, a


def biquad(x, sr, tipo, freq, q=0.707, ganho_db=0.0) -> np.ndarray:
    b, a = _biquad(tipo, sr, freq, q, ganho_db)
    return _filt(b, a, x).astype(np.float32)


# ==========================================================================
# Reverb (topologia Freeverb: 8 combs com damping + 4 allpass, por canal)
# ==========================================================================
_COMBS = (1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617)
_ALLPASS = (556, 441, 341, 225)
_SPREAD = 23


# Cada comb do Freeverb tem atraso de ~1200 amostras. Escrever isso como um
# filtro racional daria ordem ~1200 — matematicamente correto e inutil na
# pratica: `lfilter` custa O(N x ordem), e tres minutos de audio virariam
# bilhoes de operacoes.
#
# A saida e processar em BLOCOS do tamanho do atraso: dentro de um bloco de D
# amostras, tudo que depende de y[n-D] ja esta pronto do bloco anterior. Sobra
# no maximo uma recursao de primeira ordem (o damping), que e barata.
def _comb_damped(x, delay, feedback, damp):
    D = max(2, int(delay))
    n = len(x)
    x = np.asarray(x, dtype=np.float64)
    b = np.zeros(n + D)               # linha de atraso: b[i] = amostra i - D
    s_ini = 0.0
    for ini in range(0, n, D):
        fim = min(ini + D, n)
        atrasado = b[ini:fim]         # ja escrito num bloco anterior
        if damp:
            s = _filt([1 - damp], [1.0, -damp],
                      np.concatenate([[s_ini / max(1 - damp, 1e-9)], atrasado]))[1:]
        else:
            s = atrasado
        s_ini = s[-1] if len(s) else s_ini
        b[ini + D:fim + D] = x[ini:fim] + feedback * s
    return b[D:n + D]


def _allpass(x, delay, g=0.5):
    D = max(2, int(delay))
    n = len(x)
    x = np.asarray(x, dtype=np.float64)
    b = np.zeros(n + D)
    y = np.zeros(n)
    for ini in range(0, n, D):
        fim = min(ini + D, n)
        atrasado = b[ini:fim]
        b[ini + D:fim + D] = x[ini:fim] + g * atrasado
        y[ini:fim] = -x[ini:fim] + atrasado
    return y


def reverb(x: np.ndarray, sr: int = 44100, sala: float = 0.72,
           damp: float = 0.42, largura: float = 1.0, wet: float = 1.0,
           dry: float = 0.0, predelay_ms: float = 18.0) -> np.ndarray:
    """Reverb estereo. `sala` 0..1 controla o tamanho; `damp` 0..1 o quanto o
    agudo morre antes do grave (sala com cortina x sala de pedra)."""
    st = _stereo(np.asarray(x, dtype=np.float32))
    escala = sr / 44100.0
    fb = 0.70 + 0.28 * float(np.clip(sala, 0, 1))
    damp = float(np.clip(damp, 0, 0.95))

    pre = int(predelay_ms * sr / 1000)
    entrada = st.mean(axis=1)
    if pre:
        entrada = np.concatenate([np.zeros(pre, np.float32), entrada])[:len(st)]
    entrada = entrada * 0.030          # ganho de entrada do Freeverb

    saidas = []
    for canal in (0, 1):
        offset = _SPREAD if canal else 0
        acc = np.zeros(len(entrada))
        for c in _COMBS:
            acc += _comb_damped(entrada, round((c + offset) * escala), fb, damp)
        for ap in _ALLPASS:
            acc = _allpass(acc, round((ap + offset) * escala), 0.5)
        saidas.append(acc)

    l, r = saidas
    w = float(np.clip(largura, 0, 1))
    wl = l * (0.5 + 0.5 * w) + r * (0.5 - 0.5 * w)
    wr = r * (0.5 + 0.5 * w) + l * (0.5 - 0.5 * w)
    molhado = np.stack([wl, wr], axis=1)
    return (dry * st + wet * molhado).astype(np.float32)


# ==========================================================================
# Amplificador de guitarra
# ==========================================================================
def _upsample(x, fator):
    n = len(x)
    idx = np.arange(n * fator) / fator
    i0 = np.floor(idx).astype(int)
    frac = (idx - i0).astype(np.float32)
    i1 = np.minimum(i0 + 1, n - 1)
    return x[i0] * (1 - frac) + x[i1] * frac


def _tube(x, bias=0.06):
    """Clipping assimetrico. A assimetria e o que gera harmonicos PARES —
    a diferenca entre 'distorcido' e 'quente'."""
    y = np.tanh(x + bias)
    return y - np.tanh(bias)


def amp(x: np.ndarray, sr: int = 44100, ganho: float = 8.0,
        grave: float = 0.0, medio: float = 3.0, agudo: float = 0.0,
        cab: float = 4600.0, presenca: float = 2.0,
        nivel: float = 0.5, oversample: int = 4) -> np.ndarray:
    """Cadeia: corte de grave -> dois estagios de valvula -> tonestack -> cabine.

    Os estagios de distorcao rodam com oversampling porque clipping cria
    harmonicos acima de Nyquist, e sem oversampling eles voltam rebatidos
    como aquele chiado metalico que denuncia distorcao digital barata.
    """
    mono = np.asarray(x, np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=1)

    # o grave entra na valvula e vira lama: corta antes
    y = biquad(mono, sr, "highpass", 95.0, 0.7)

    if oversample > 1:
        y = _upsample(y, oversample)
        sr_os = sr * oversample
    else:
        sr_os = sr

    y = _tube(y * ganho, 0.06)
    y = biquad(y, sr_os, "lowpass", 6500.0, 0.7)        # acoplamento
    y = _tube(y * (1.0 + ganho * 0.25), -0.03)

    if oversample > 1:
        y = biquad(y, sr_os, "lowpass", sr * 0.45, 0.7)  # anti-imagem
        y = y[::oversample]

    # tonestack
    if grave:
        y = biquad(y, sr, "lowshelf", 160.0, 0.7, grave)
    if medio:
        y = biquad(y, sr, "peak", 900.0, 0.9, medio)
    if agudo:
        y = biquad(y, sr, "highshelf", 3200.0, 0.7, agudo)

    # cabine: passa-alta seca + queda no agudo + ressonancia de falante
    y = biquad(y, sr, "highpass", 110.0, 0.8)
    y = biquad(y, sr, "peak", 2100.0, 1.2, presenca)
    y = biquad(y, sr, "lowpass", cab, 0.9)
    y = biquad(y, sr, "lowpass", cab * 1.6, 0.7)
    return (y * nivel).astype(np.float32)


# ==========================================================================
# Chorus / largura
# ==========================================================================
def chorus(x: np.ndarray, sr: int = 44100, profundidade_ms: float = 6.0,
           taxa: float = 0.45, mix: float = 0.4,
           espalhar: bool = True) -> np.ndarray:
    st = _stereo(np.asarray(x, np.float32))
    n = len(st)
    t = np.arange(n) / sr
    saida = np.empty_like(st)
    for canal in (0, 1):
        fase = 0.0 if canal == 0 else np.pi / 2
        atraso = (profundidade_ms * (1 + np.sin(2 * np.pi * taxa * t + fase)) / 2
                  + 4.0) * sr / 1000
        idx = np.arange(n) - atraso
        idx = np.clip(idx, 0, n - 1)
        i0 = np.floor(idx).astype(int)
        frac = (idx - i0).astype(np.float32)
        i1 = np.minimum(i0 + 1, n - 1)
        fonte = st[:, canal if espalhar else 0]
        molhado = fonte[i0] * (1 - frac) + fonte[i1] * frac
        saida[:, canal] = (1 - mix) * st[:, canal] + mix * molhado
    return saida.astype(np.float32)


# ==========================================================================
# Compressor
# ==========================================================================
def compressor(x: np.ndarray, sr: int = 44100, limiar_db: float = -18.0,
               razao: float = 3.0, attack_ms: float = 8.0,
               release_ms: float = 140.0, makeup_db: float = 0.0) -> np.ndarray:
    st = _stereo(np.asarray(x, np.float32))
    det = np.abs(st).max(axis=1)
    a_at = float(np.exp(-1.0 / (sr * attack_ms / 1000)))
    a_re = float(np.exp(-1.0 / (sr * release_ms / 1000)))

    # envelope com ataque e release separados: nao da pra vetorizar num
    # filtro so, mas um passa-baixa por trecho aproxima bem e roda rapido
    env = _filt([1 - a_re], [1, -a_re], det)
    rapido = _filt([1 - a_at], [1, -a_at], det)
    env = np.maximum(env, rapido)

    env_db = 20 * np.log10(np.maximum(env, 1e-9))
    excesso = np.maximum(0.0, env_db - limiar_db)
    reducao_db = excesso * (1 - 1 / max(razao, 1.001))
    ganho = 10 ** ((makeup_db - reducao_db) / 20)
    return (st * ganho[:, None]).astype(np.float32)


def limiter(x: np.ndarray, teto: float = 0.97) -> np.ndarray:
    """Segura o pico no teto, sem roubar volume de quem ja estava abaixo dele.

    A versao anterior aplicava tanh DEPOIS de normalizar e nao devolvia o nivel:
    qualquer mix que passasse do teto saia sempre em 0.746, dois decibeis e meio
    mais baixo do que devia. Limitador que abaixa 2,5 dB sem avisar nao e
    limitador, e um fader escondido.
    """
    st = _stereo(np.asarray(x, np.float32))
    pico = float(np.abs(st).max())
    if pico <= 1e-9:
        return st
    alvo = min(pico, teto)
    if pico > teto:
        st = st * (teto / pico)
    y = np.tanh(st * 1.05)
    p2 = float(np.abs(y).max())
    if p2 > 1e-9:
        y = y * (alvo / p2)
    return y.astype(np.float32)


# ==========================================================================
# Cadeia declarativa
# ==========================================================================
def aplicar(stereo: np.ndarray, sr: int, inserts) -> np.ndarray:
    """Roda a lista de efeitos de uma track, na ordem."""
    y = _stereo(np.asarray(stereo, np.float32))
    for spec in inserts or ():
        t = spec.get("tipo")
        p = {k: v for k, v in spec.items() if k != "tipo"}
        if t == "amp":
            # `pan` e do canal, nao do amplificador: sai dos kwargs antes
            pan = float(p.pop("pan", 0.0))
            mono = amp(y, sr, **p)
            ang = (pan + 1.0) * np.pi / 4.0
            y = np.stack([mono * np.cos(ang), mono * np.sin(ang)], axis=1)
        elif t == "eq":
            y = np.stack([biquad(y[:, c], sr, p.get("modo", "peak"),
                                 p["freq"], p.get("q", 0.9),
                                 p.get("ganho_db", 0.0)) for c in (0, 1)], axis=1)
        elif t == "chorus":
            y = chorus(y, sr, **p)
        elif t == "comp":
            y = compressor(y, sr, **p)
        elif t == "reverb":
            y = reverb(y, sr, **p)
        elif t == "sala":
            from . import espaco
            y = espaco.reverb(y, sr, **p)
        elif t == "largura":
            from . import espaco
            y = espaco.largura(y, sr=sr, **p)
        elif t == "haas":
            from . import espaco
            y = espaco.haas(y, sr, **p)
        elif t == "eco":
            from . import espaco
            y = espaco.eco(y, sr, **p)
        elif t == "excitador":
            from . import master
            y = master.excitador(y, sr, **p)
        elif t == "multibanda":
            from . import master
            y = master.compressor_multibanda(y, sr, **p)
        elif t == "ms":
            from . import master
            y = master.eq_meio_lado(y, sr, **p)
        elif t == "ganho":
            y = y * float(p.get("db", 0.0) and 10 ** (p["db"] / 20) or 1.0)
        else:
            raise ValueError(f"efeito desconhecido: {t}")
    return y.astype(np.float32)


PRESETS_AMP = {
    "crunch":   {"tipo": "amp", "ganho": 5.0, "medio": 3.0, "cab": 5200,
                 "presenca": 2.0, "nivel": 0.42},
    "rock":     {"tipo": "amp", "ganho": 9.0, "medio": 2.5, "grave": -2.0,
                 "cab": 4600, "presenca": 3.0, "nivel": 0.36},
    "metal":    {"tipo": "amp", "ganho": 16.0, "medio": -3.0, "grave": 2.0,
                 "agudo": 3.0, "cab": 4200, "presenca": 4.0, "nivel": 0.30},
    "limpo":    {"tipo": "amp", "ganho": 1.6, "medio": 1.0, "cab": 7000,
                 "presenca": 1.0, "nivel": 0.55},
}
