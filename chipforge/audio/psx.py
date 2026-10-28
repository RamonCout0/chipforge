"""
chipforge.audio.psx
===================
O chip de som do PlayStation (SPU) — o "32 bits" do Chipforge.

A trilha de Mega Man 8, Castlevania SotN ou Final Fantasy VII nao e chiptune:
e sequencia MIDI tocando samples. O que da a cara da epoca sao as limitacoes
do SPU, e cada uma delas mora aqui:

1. **512 KB de RAM de som** para todos os samples do jogo. Por isso os samples
   eram curtos, em loop e gravados em taxa baixa (11-22 kHz) — o timbre
   levemente abafado e "sintetico" que o review da trilha do MM8 chama de vazio.
2. **ADPCM de 4 bits** (formato VAG): blocos de 28 amostras com um de 5 filtros
   preditores e um fator de escala. Deixa um granulado proprio, diferente do
   bitcrush do NES — o erro acompanha o sinal em vez de ser um degrau fixo.
3. **Interpolacao gaussiana** de 4 pontos na hora de mudar a afinacao: arredonda
   o agudo, e e por isso que nada no PS1 soa "cristalino".
4. **24 vozes** no total. Arranjo de PS1 e economico porque precisava ser.
5. **Reverb de hardware** com presets fixos (Room, Studio, Hall, Space Echo...).
   E o som mais reconhecivel do console: um eco metalico, granulado, rodando a
   22 kHz dentro da mesma RAM dos samples.

Fonte da formula do reverb e dos presets: psx-spx (Martin Korth),
"SPU Reverb Formula" e "SPU Reverb Examples". Os enderecos dos registradores
estao em unidades de 8 bytes (= 4 amostras de 16 bits) e o buffer anda uma
amostra por tick de 22050 Hz.
"""
from __future__ import annotations

from contextlib import contextmanager
from fractions import Fraction

import numpy as np

try:
    from scipy.signal import lfilter as _lfilter, resample_poly as _resample_poly
except Exception:                       # pragma: no cover
    _lfilter = None
    _resample_poly = None


# ==========================================================================
# Modo ativo durante o render
# ==========================================================================
# O render liga o modo com `with ativo(...)`; o SamplerInstrument consulta
# `ESTADO` a cada nota. Global de proposito: o contrato render_note() e o mesmo
# para todo instrumento, e passar o console por ele mudaria 6 classes.
ESTADO = {"ligado": False, "taxa": 22050}


@contextmanager
def ativo(taxa: int = 22050):
    antes = dict(ESTADO)
    ESTADO.update(ligado=True, taxa=int(taxa))
    try:
        yield
    finally:
        ESTADO.clear()
        ESTADO.update(antes)


TAXAS_TIPICAS = {
    # quanto cada familia costumava ocupar na RAM de som
    "padrao": 22050, "baixo": 11025, "bateria": 22050, "pad": 16000,
}


# ==========================================================================
# ADPCM (VAG) — codifica e decodifica, para o sample carregar o erro do PS1
# ==========================================================================
FILTROS = np.array([[0.0, 0.0],
                    [60 / 64, 0.0],
                    [115 / 64, -52 / 64],
                    [98 / 64, -55 / 64],
                    [122 / 64, -60 / 64]])


def adpcm_ida_e_volta(x: np.ndarray) -> np.ndarray:
    """Passa o sinal (float -1..1) pelo codec VAG e devolve o que o console
    tocaria. Escolhe filtro e escala por bloco de 28 amostras, como o
    conversor da Sony fazia; o preditor roda sobre o sinal JA decodificado,
    entao o erro se propaga do jeito certo."""
    x = np.asarray(x, dtype=np.float64) * 32767.0
    n = len(x)
    pad = (-n) % 28
    if pad:
        x = np.concatenate([x, np.zeros(pad)])
    out = np.empty_like(x)
    s1 = s2 = 0.0
    for b0 in range(0, len(x), 28):
        bloco = x[b0:b0 + 28]
        # 1) filtro: o que deixa o menor residuo, estimado no sinal original
        h1 = np.concatenate([[s1], bloco[:-1]])
        h2 = np.concatenate([[s2, s1], bloco[:-2]])
        melhor, melhor_pico = 0, np.inf
        for f, (p, q) in enumerate(FILTROS):
            pico = np.abs(bloco - p * h1 - q * h2).max()
            if pico < melhor_pico:
                melhor, melhor_pico = f, pico
        p, q = FILTROS[melhor]
        # 2) escala: o menor passo que ainda cabe em 4 bits (-8..7)
        shift = 12
        while shift > 0 and melhor_pico > 7.0 * (1 << (12 - shift)):
            shift -= 1
        passo = float(1 << (12 - shift))
        # 3) quantiza em malha fechada (o preditor ve o sinal decodificado)
        for i in range(28):
            pred = p * s1 + q * s2
            nib = round((bloco[i] - pred) / passo)
            nib = -8 if nib < -8 else (7 if nib > 7 else nib)
            v = nib * passo + pred
            v = -32768.0 if v < -32768.0 else (32767.0 if v > 32767.0 else v)
            v = float(int(v))                   # o SPU guarda inteiros
            out[b0 + i] = v
            s2, s1 = s1, v
    return (out[:n] / 32767.0).astype(np.float32)


def reamostrar(x: np.ndarray, de: int, para: int) -> np.ndarray:
    if de == para or len(x) < 4:
        return np.asarray(x, dtype=np.float32)
    if _resample_poly is not None:
        fr = Fraction(int(para), int(de)).limit_denominator(1000)
        return _resample_poly(x, fr.numerator, fr.denominator).astype(np.float32)
    idx = np.arange(0, len(x) - 1, de / para)           # pragma: no cover
    i0 = idx.astype(int)
    f = idx - i0
    return (x[i0] * (1 - f) + x[i0 + 1] * f).astype(np.float32)


_CACHE: dict = {}


def amostra_na_ram(chave, data: np.ndarray, taxa_original: int,
                   taxa: int) -> tuple[np.ndarray, int]:
    """O sample como estaria na RAM do SPU: reamostrado e com o erro do ADPCM.
    Cacheado — o mesmo sample e tocado milhares de vezes numa musica."""
    taxa = int(min(taxa, taxa_original))
    k = (chave, taxa)
    if k not in _CACHE:
        _CACHE[k] = adpcm_ida_e_volta(reamostrar(data, taxa_original, taxa))
    return _CACHE[k], taxa


# ==========================================================================
# Interpolacao gaussiana
# ==========================================================================
# O SPU usa uma tabela de 512 posicoes x 4 pontos (a mesma do SNES). No ponto
# exato da amostra (frac = 0) os pesos sao 372, 1305, 374 e 0 (/2048): uma
# gaussiana de sigma ~0,63 amostra. A tabela abaixo e gerada dessa gaussiana —
# bate com os pesos medidos nas pontas e devolve o mesmo abafamento do agudo.
_SIGMA = 0.632


def pesos_gaussianos(frac: np.ndarray) -> tuple[np.ndarray, ...]:
    d = np.stack([1.0 + frac, frac, 1.0 - frac, 2.0 - frac])
    w = np.exp(-(d * d) / (2 * _SIGMA * _SIGMA))
    w /= w.sum(axis=0, keepdims=True)
    return tuple(w.astype(np.float32))


def ler_gaussiano(data: np.ndarray, idx: np.ndarray) -> np.ndarray:
    i0 = np.floor(idx).astype(np.int64)
    frac = (idx - i0).astype(np.float64)
    n = len(data)
    w0, w1, w2, w3 = pesos_gaussianos(frac)

    def at(i):
        return data[np.clip(i, 0, n - 1)]

    return (w0 * at(i0 - 1) + w1 * at(i0) + w2 * at(i0 + 1) + w3 * at(i0 + 2))


PASSO_MAXIMO = 0x3FFF / 0x1000      # registrador de pitch: ~4x a 44,1 kHz


# ==========================================================================
# Limite de vozes
# ==========================================================================
VOZES = 24


def limitar_vozes(eventos: list[list], vozes: int = VOZES) -> int:
    """`eventos`: listas mutaveis [inicio, fim_soando, peso, nota]. Quando passa
    de `vozes`, a voz mais antiga e cortada (roubo de voz, como o driver de som
    do console fazia). Corta `nota.dur` no lugar. Devolve quantas foram roubadas.

    `peso` e quantas vozes do SPU a nota ocupa (dobra de secao = N)."""
    ativos: list[list] = []
    roubadas = 0
    for ev in sorted(eventos, key=lambda e: e[0]):
        ini = ev[0]
        ativos = [a for a in ativos if a[1] > ini]
        ocupadas = sum(a[2] for a in ativos)
        while ativos and ocupadas + ev[2] > vozes:
            velho = min(ativos, key=lambda a: a[0])
            ativos.remove(velho)
            ocupadas -= velho[2]
            nota = velho[3]
            nota.dur = max(1, int(ini - nota.start)) if ini > nota.start else 1
            roubadas += 1
        ativos.append(ev)
    return roubadas


# ==========================================================================
# Reverb do SPU
# ==========================================================================
# Ordem dos campos (igual ao psx-spx / registradores 1F801DC0h..):
_CAMPOS = ("tamanho dAPF1 dAPF2 vIIR vCOMB1 vCOMB2 vCOMB3 vCOMB4 vWALL "
           "vAPF1 vAPF2 mLSAME mRSAME mLCOMB1 mRCOMB1 mLCOMB2 mRCOMB2 "
           "dLSAME dRSAME mLDIFF mRDIFF mLCOMB3 mRCOMB3 mLCOMB4 mRCOMB4 "
           "dLDIFF dRDIFF mLAPF1 mRAPF1 mLAPF2 mRAPF2").split()

_TABELA = {
    "room": (0x26C0,
             0x007D, 0x005B, 0x6D80, 0x54B8, 0xBED0, 0x0000, 0x0000, 0xBA80,
             0x5800, 0x5300, 0x04D6, 0x0333, 0x03F0, 0x0227, 0x0374, 0x01EF,
             0x0334, 0x01B5, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
             0x0000, 0x0000, 0x01B4, 0x0136, 0x00B8, 0x005C),
    "studio_small": (0x1F40,
             0x0033, 0x0025, 0x70F0, 0x4FA8, 0xBCE0, 0x4410, 0xC0F0, 0x9C00,
             0x5280, 0x4EC0, 0x03E4, 0x031B, 0x03A4, 0x02AF, 0x0372, 0x0266,
             0x031C, 0x025D, 0x025C, 0x018E, 0x022F, 0x0135, 0x01D2, 0x00B7,
             0x018F, 0x00B5, 0x00B4, 0x0080, 0x004C, 0x0026),
    "studio_medium": (0x4840,
             0x00B1, 0x007F, 0x70F0, 0x4FA8, 0xBCE0, 0x4510, 0xBEF0, 0xB4C0,
             0x5280, 0x4EC0, 0x0904, 0x076B, 0x0824, 0x065F, 0x07A2, 0x0616,
             0x076C, 0x05ED, 0x05EC, 0x042E, 0x050F, 0x0305, 0x0462, 0x02B7,
             0x042F, 0x0265, 0x0264, 0x01B2, 0x0100, 0x0080),
    "studio_large": (0x6FE0,
             0x00E3, 0x00A9, 0x6F60, 0x4FA8, 0xBCE0, 0x4510, 0xBEF0, 0xA680,
             0x5680, 0x52C0, 0x0DFB, 0x0B58, 0x0D09, 0x0A3C, 0x0BD9, 0x0973,
             0x0B59, 0x08DA, 0x08D9, 0x05E9, 0x07EC, 0x04B0, 0x06EF, 0x03D2,
             0x05EA, 0x031D, 0x031C, 0x0238, 0x0154, 0x00AA),
    "hall": (0xADE0,
             0x01A5, 0x0139, 0x6000, 0x5000, 0x4C00, 0xB800, 0xBC00, 0xC000,
             0x6000, 0x5C00, 0x15BA, 0x11BB, 0x14C2, 0x10BD, 0x11BC, 0x0DC1,
             0x11C0, 0x0DC3, 0x0DC0, 0x09C1, 0x0BC4, 0x07C1, 0x0A00, 0x06CD,
             0x09C2, 0x05C1, 0x05C0, 0x041A, 0x0274, 0x013A),
    "space_echo": (0xF6C0,
             0x033D, 0x0231, 0x7E00, 0x5000, 0xB400, 0xB000, 0x4C00, 0xB000,
             0x6000, 0x5400, 0x1ED6, 0x1A31, 0x1D14, 0x183B, 0x1BC2, 0x16B2,
             0x1A32, 0x15EF, 0x15EE, 0x1055, 0x1334, 0x0F2D, 0x11F6, 0x0C5D,
             0x1056, 0x0AE1, 0x0AE0, 0x07A2, 0x0464, 0x0232),
    "half_echo": (0x3C00,
             0x0017, 0x0013, 0x70F0, 0x4FA8, 0xBCE0, 0x4510, 0xBEF0, 0x8500,
             0x5F80, 0x54C0, 0x0371, 0x02AF, 0x02E5, 0x01DF, 0x02B0, 0x01D7,
             0x0358, 0x026A, 0x01D6, 0x011E, 0x012D, 0x00B1, 0x011F, 0x0059,
             0x01A0, 0x00E3, 0x0058, 0x0040, 0x0028, 0x0014),
}

PRESETS_REVERB = {
    "room": "Sala pequena — o padrao de muito jogo de acao",
    "studio_small": "Estudio pequeno — brilho curto, bom para chip/lead",
    "studio_medium": "Estudio medio — o 'ar' de fase de plataforma",
    "studio_large": "Estudio grande — o vazio frio de fase de gelo",
    "hall": "Hall — cauda longa, chefoes e cenas",
    "space_echo": "Space Echo — eco metalico enorme, cavernas e espaco",
    "half_echo": "Half Echo — eco curto e repetido, slapback de PS1",
}


def _preset(nome: str) -> dict:
    vals = _TABELA[nome]
    d = dict(zip(_CAMPOS, vals))
    out = {"N": d["tamanho"] // 2}             # bytes -> amostras de 16 bits
    for k, v in d.items():
        if k.startswith("v"):                  # volumes: 16 bits com sinal
            out[k] = (v - 0x10000 if v >= 0x8000 else v) / 32768.0
        elif k != "tamanho":                   # enderecos: unidades de 8 bytes
            out[k] = v * 4
    return out


def reverb(x: np.ndarray, sr: int = 44100, preset: str = "studio_large",
           wet: float = 0.5, dry: float = 0.0) -> np.ndarray:
    """Reverb do SPU sobre um sinal estereo. Roda a 22050 Hz como o console.

    Implementacao: a formula do psx-spx e um laco por amostra num buffer
    circular. Aqui cada endereco escrito vira uma linha de atraso, cada leitura
    vira "a escrita mais recente naquele endereco", e o laco anda em blocos do
    tamanho do menor atraso de realimentacao — mesmo resultado, rapido o
    bastante para render de musica inteira em Python.
    """
    if preset not in _TABELA:
        raise ValueError(f"preset de reverb PS1 desconhecido: {preset}. "
                         f"Opcoes: {', '.join(_TABELA)}")
    st = np.asarray(x, dtype=np.float64)
    if st.ndim == 1:
        st = np.stack([st, st], axis=1)
    p = _preset(preset)
    N = p["N"]

    # --- entrada a 22050 Hz -------------------------------------------------
    sr_rev = 22050
    entrada = np.stack([reamostrar(st[:, c].astype(np.float32), sr, sr_rev)
                        for c in (0, 1)], axis=1).astype(np.float64)
    # meio segundo de cauda alem do fim da musica + o tamanho do buffer
    T = len(entrada) + N + sr_rev // 2
    Lin = np.zeros(T); Rin = np.zeros(T)
    Lin[:len(entrada)] = entrada[:, 0]
    Rin[:len(entrada)] = entrada[:, 1]

    # --- linhas (escritores) na ordem em que o hardware escreve -----------
    escritores = ["mLSAME", "mRSAME", "mLDIFF", "mRDIFF",
                  "mLAPF1", "mRAPF1", "mLAPF2", "mRAPF2"]
    ordem = {w: i for i, w in enumerate(escritores)}
    m = {w: p[w] for w in escritores}

    def resolver(endereco: int, leitor_ordem: float):
        """Quem escreveu por ultimo em [endereco] e ha quantas amostras."""
        melhor = None
        for w in escritores:
            D = (m[w] - endereco) % N
            if D == 0 and ordem[w] >= leitor_ordem:
                D = N                       # escreve depois de ler: ciclo antigo
            if melhor is None or D < melhor[1]:
                melhor = (w, D)
        return melhor

    linhas = {w: np.zeros(T) for w in escritores}

    # leituras de cada etapa
    leituras = {}
    for lado, outro in (("L", "R"), ("R", "L")):
        leituras[f"m{lado}SAME"] = (resolver(p[f"d{lado}SAME"], ordem[f"m{lado}SAME"]),
                                    resolver(p[f"m{lado}SAME"] - 1, ordem[f"m{lado}SAME"]))
        leituras[f"m{lado}DIFF"] = (resolver(p[f"d{outro}DIFF"], ordem[f"m{lado}DIFF"]),
                                    resolver(p[f"m{lado}DIFF"] - 1, ordem[f"m{lado}DIFF"]))
        leituras[f"comb{lado}"] = [(p[f"vCOMB{k}"],
                                    resolver(p[f"m{lado}COMB{k}"], 3.5))
                                   for k in (1, 2, 3, 4) if p[f"vCOMB{k}"] != 0]
        for a in (1, 2):
            w = f"m{lado}APF{a}"
            leituras[w] = resolver(p[w] - p[f"dAPF{a}"], ordem[w])

    # tamanho do bloco: o menor atraso das leituras que voltam no grafo
    def para_tras(src, leitor_ord):
        return ordem[src] >= leitor_ord
    atrasos = []
    for w in ("mLSAME", "mRSAME", "mLDIFF", "mRDIFF"):
        (sw, Dw), (sp, Dp) = leituras[w]
        if para_tras(sw, ordem[w]):
            atrasos.append(Dw)
        if not (sp == w and Dp == 1) and para_tras(sp, ordem[w]):
            atrasos.append(Dp)
    for lado in "LR":
        for _, (s, D) in leituras[f"comb{lado}"]:
            if para_tras(s, 3.5):
                atrasos.append(D)
        for a in (1, 2):
            w = f"m{lado}APF{a}"
            s, D = leituras[w]
            if para_tras(s, ordem[w]):
                atrasos.append(D)
    B = max(1, min([a for a in atrasos if a > 0] or [256]))
    B = min(B, 4096)

    vIIR, vWALL = p["vIIR"], p["vWALL"]
    zi = {w: 0.0 for w in ("mLSAME", "mRSAME", "mLDIFF", "mRDIFF")}
    saida = np.zeros((T, 2))

    def ler(src_D, t0, t1):
        src, D = src_D
        a, b = t0 - D, t1 - D
        lin = linhas[src]
        if a >= 0:
            return lin[a:b]
        r = np.zeros(t1 - t0)
        if b > 0:
            r[-b:] = lin[0:b]
        return r

    for t0 in range(0, T, B):
        t1 = min(T, t0 + B)
        # 1) reflexoes (mesmo lado e lado oposto) — filtro IIR de 1 polo
        for w, entrada_c in (("mLSAME", Lin), ("mRSAME", Rin),
                             ("mLDIFF", Lin), ("mRDIFF", Rin)):
            (src_d, prev) = leituras[w]
            u = entrada_c[t0:t1] + ler(src_d, t0, t1) * vWALL
            if prev[0] == w and prev[1] == 1:
                # [m] = (u - [m-1]) * vIIR + [m-1]
                if _lfilter is not None:
                    y, zf = _lfilter([vIIR], [1.0, -(1.0 - vIIR)], u,
                                     zi=[(1.0 - vIIR) * zi[w]])
                else:                                   # pragma: no cover
                    y = np.empty_like(u); s = zi[w]
                    for i, ui in enumerate(u):
                        s = (ui - s) * vIIR + s; y[i] = s
                zi[w] = float(y[-1])
            else:
                y = (u - ler(prev, t0, t1)) * vIIR + ler(prev, t0, t1)
            linhas[w][t0:t1] = np.clip(y, -1.0, 1.0)
        # 2) eco inicial (combs) -> 3) dois all-pass em serie
        for c, lado in ((0, "L"), (1, "R")):
            out = np.zeros(t1 - t0)
            for v, src in leituras[f"comb{lado}"]:
                out += v * ler(src, t0, t1)
            for a in (1, 2):
                w = f"m{lado}APF{a}"
                va = p[f"vAPF{a}"]
                atras = ler(leituras[w], t0, t1)
                escrito = np.clip(out - va * atras, -1.0, 1.0)
                linhas[w][t0:t1] = escrito
                out = atras + escrito * va
            saida[t0:t1, c] = out

    molhado = np.stack([reamostrar(saida[:, c].astype(np.float32), sr_rev, sr)
                        for c in (0, 1)], axis=1)
    n = len(st)
    molhado = molhado[:n]
    if len(molhado) < n:
        molhado = np.vstack([molhado, np.zeros((n - len(molhado), 2))])
    return (dry * st + wet * molhado).astype(np.float32)
