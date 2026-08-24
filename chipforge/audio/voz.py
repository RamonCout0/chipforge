"""
chipforge.audio.voz
===================
Voz cantada e coro por sintese de formantes.

Por que isto existe: um banco GM traz "Choir Aahs", e todo mundo reconhece o
som — e reconhece como sample de teclado dos anos 90. Uma trilha cinematica e
carregada pela VOZ, e voz nao e um sample sustentado: e uma fonte glotal com
tremor proprio passando por um trato vocal que muda de forma enquanto a frase
anda.

Modelo fonte-filtro, o mesmo do Klatt:

    pulso glotal (Rosenberg)  +  ruido de ar
             |
        jitter e shimmer            <- o que nenhum humano consegue evitar
             |
        banco de 5 formantes        <- a vogal
             |
        formante do cantor (3 kHz)  <- o que faz a voz atravessar a orquestra
             |
        vibrato com entrada atrasada

As tabelas de formante sao as medidas classicas de vogal CANTADA, separadas
por naipe: a mesma vogal tem formante diferente num soprano e num baixo,
porque o trato vocal tem comprimento diferente.

O ganho pratico: 800-2500 Hz e a faixa onde a melodia principal mora, e e
exatamente onde uma mixagem so de samples fica magra. A voz preenche ali.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    from scipy.signal import lfilter as _lfilter
except Exception:                       # pragma: no cover
    _lfilter = None

from . import effects as FX
from .envelope import ADSR

# --------------------------------------------------------------------------
# Tabelas de formante — vogal cantada, por naipe.
# Cada entrada: (frequencias Hz, amplitudes dB, larguras de banda Hz)
# --------------------------------------------------------------------------
FORMANTES: dict[str, dict[str, tuple]] = {
    "soprano": {
        "a": ((800, 1150, 2900, 3900, 4950), (0, -6, -32, -20, -50), (80, 90, 120, 130, 140)),
        "e": ((350, 2000, 2800, 3600, 4950), (0, -20, -15, -40, -56), (60, 100, 120, 150, 200)),
        "i": ((270, 2140, 2950, 3900, 4950), (0, -12, -26, -26, -44), (60, 90, 100, 120, 120)),
        "o": ((450, 800, 2830, 3800, 4950), (0, -11, -22, -22, -50), (70, 80, 100, 130, 135)),
        "u": ((325, 700, 2700, 3800, 4950), (0, -16, -35, -40, -60), (50, 60, 170, 180, 200)),
    },
    "contralto": {
        "a": ((800, 1150, 2800, 3500, 4950), (0, -4, -20, -36, -60), (80, 90, 120, 130, 140)),
        "e": ((400, 1600, 2700, 3300, 4950), (0, -24, -30, -35, -60), (60, 80, 120, 150, 200)),
        "i": ((350, 1700, 2700, 3700, 4950), (0, -20, -30, -36, -60), (50, 100, 120, 150, 200)),
        "o": ((450, 800, 2830, 3500, 4950), (0, -9, -16, -28, -55), (70, 80, 100, 130, 135)),
        "u": ((325, 700, 2530, 3500, 4950), (0, -12, -30, -40, -64), (50, 60, 170, 180, 200)),
    },
    "tenor": {
        "a": ((650, 1080, 2650, 2900, 3250), (0, -6, -7, -8, -22), (80, 90, 120, 130, 140)),
        "e": ((400, 1700, 2600, 3200, 3580), (0, -14, -12, -14, -20), (70, 80, 100, 120, 120)),
        "i": ((290, 1870, 2800, 3250, 3540), (0, -15, -18, -20, -30), (40, 90, 100, 120, 120)),
        "o": ((400, 800, 2600, 2800, 3000), (0, -10, -12, -12, -26), (40, 80, 100, 120, 120)),
        "u": ((350, 600, 2700, 2900, 3300), (0, -20, -17, -14, -26), (40, 60, 100, 120, 120)),
    },
    "baixo": {
        "a": ((600, 1040, 2250, 2450, 2750), (0, -7, -9, -9, -20), (60, 70, 110, 120, 130)),
        "e": ((400, 1620, 2400, 2800, 3100), (0, -12, -9, -12, -18), (40, 80, 100, 120, 120)),
        "i": ((250, 1750, 2600, 3050, 3340), (0, -30, -16, -22, -28), (60, 90, 100, 120, 120)),
        "o": ((400, 750, 2400, 2600, 2900), (0, -11, -21, -20, -40), (40, 80, 100, 120, 120)),
        "u": ((350, 600, 2400, 2675, 2950), (0, -20, -32, -28, -36), (40, 80, 100, 120, 120)),
    },
}

# Boca fechada. Nao esta nas tabelas classicas porque nao e vogal — mas
# "coro de boca fechada" e metade do vocabulario de trilha, entao entra aqui
# como um /u/ muito abafado, com o agudo derrubado pela nasal.
_HUM = ((280, 1100, 2100, 2900, 3600), (0, -26, -40, -48, -60), (60, 130, 200, 240, 260))

VOGAIS_PT = {
    "a": "aberto (aah) — o mais brilhante, corta a orquestra",
    "e": "meio fechado (eeh)",
    "i": "fechado (iih) — agudo, agressivo",
    "o": "arredondado (ooh) — o som de coro epico",
    "u": "muito fechado (uuh) — escuro, com sopro",
    "m": "boca fechada (hum) — coro sem palavra",
}

NAIPES_PT = {
    "soprano": "Soprano (feminino agudo)",
    "contralto": "Contralto (feminino grave)",
    "tenor": "Tenor (masculino agudo)",
    "baixo": "Baixo (masculino grave)",
}

# faixa confortavel de cada naipe, em MIDI
TESSITURA = {
    "soprano": (60, 84),
    "contralto": (53, 77),
    "tenor": (48, 72),
    "baixo": (40, 64),
}

_BLOCO = 512                # blocos para o filtro variante no tempo

# O banco paralelo sai baixo porque quase toda a tabela de amplitudes e
# negativa em relacao ao F1. Este fator so recoloca a saida numa faixa util
# (gain=1.0 -> pico perto de 0,7), sem mexer na dinamica de velocity.
_NIVEL = 3.0


def _tabela(naipe: str, vogal: str):
    if vogal == "m":
        return _HUM
    naipe = naipe if naipe in FORMANTES else "soprano"
    tab = FORMANTES[naipe]
    return tab.get(vogal, tab["a"])


def _interp_vogal(naipe: str, va: str, vb: str, k: float):
    """Mistura duas vogais. Interpola frequencia em LOG (o ouvido ouve razao,
    nao diferenca) e amplitude em dB — que ja e log."""
    fa, aa, ba = _tabela(naipe, va)
    fb, ab, bb = _tabela(naipe, vb)
    k = float(np.clip(k, 0.0, 1.0))
    f = np.exp((1 - k) * np.log(np.array(fa, float)) + k * np.log(np.array(fb, float)))
    a = (1 - k) * np.array(aa, float) + k * np.array(ab, float)
    b = (1 - k) * np.array(ba, float) + k * np.array(bb, float)
    return f, a, b


# --------------------------------------------------------------------------
# Fonte glotal
# --------------------------------------------------------------------------
def _pulso_glotal(fase: np.ndarray, quociente_aberto: float = 0.62,
                  fecho: float = 0.16) -> np.ndarray:
    """Pulso de Rosenberg, avaliado a partir da FASE acumulada.

    Trabalhar por fase (e nao gerando periodo a periodo) e o que deixa o pitch
    variar DENTRO da nota: vibrato, jitter e portamento saem de graca, e a
    forma de onda nao pula quando a frequencia muda.

    O que se ouve e a DERIVADA do fluxo glotal; por isso o retorno e
    diferenciado no fim — e a diferenca entre um orgao e uma voz.
    """
    t = np.mod(fase, 1.0)
    tp = float(np.clip(quociente_aberto, 0.2, 0.9))
    tn = float(np.clip(fecho, 0.05, 1.0 - tp - 0.02))
    g = np.zeros_like(t)

    subida = t < tp
    x = t[subida] / tp
    g[subida] = 3 * x ** 2 - 2 * x ** 3          # abertura suave

    descida = (t >= tp) & (t < tp + tn)
    x = (t[descida] - tp) / tn
    g[descida] = 1.0 - x ** 2                    # fecho rapido: o "chute"

    d = np.diff(g, prepend=g[0])
    d[0] = 0.0
    # Normaliza por RMS, nao por pico. Normalizando por pico, uma nota grave
    # sai muito mais alta que uma aguda: o periodo e mais longo, o "chute" de
    # fecho se espalha por mais amostras e a energia por segundo dispara. Com
    # RMS, o mesmo preset canta no mesmo volume em qualquer registro.
    m = float(np.sqrt((d.astype(np.float64) ** 2).mean()))
    return (d / m if m > 1e-9 else d).astype(np.float32)


def _ressoador(f: float, bw: float, sr: int, amp: float = 1.0):
    """Ressoador de 2 polos normalizado no PICO (nao em DC).

    Klatt normaliza em DC; num banco paralelo isso faz o F1 engolir todos os
    outros, porque um polo grave tem ganho enorme em DC. Normalizando no pico,
    a tabela de amplitudes da vogal passa a significar o que diz.
    """
    f = float(np.clip(f, 20.0, sr * 0.48))
    bw = float(np.clip(bw, 10.0, sr * 0.2))
    r = np.exp(-np.pi * bw / sr)
    theta = 2 * np.pi * f / sr
    a1 = 2 * r * np.cos(theta)
    a2 = -r * r
    z = np.exp(1j * theta)
    h = 1.0 / (1.0 - a1 / z - a2 / (z * z))
    b0 = amp / max(abs(h), 1e-9)
    return np.array([b0, 0.0, 0.0]), np.array([1.0, -a1, -a2])


def _banco_formantes(fonte: np.ndarray, sr: int, naipe: str,
                     vogal_de: str, vogal_para: str, morph: np.ndarray,
                     escala: float = 1.0, dispersao: np.ndarray | None = None,
                     ganho_formante: float = 1.0) -> np.ndarray:
    """Banco paralelo de 5 formantes, com coeficientes recalculados por bloco.

    Recalcular por bloco (e nao por amostra) e a escolha que torna isto viavel
    em Python: 512 amostras compartilham coeficiente, e a vogal leva dezenas de
    milissegundos para mudar de verdade — ninguem ouve o degrau.
    """
    n = len(fonte)
    saida = np.zeros(n, dtype=np.float64)
    estados: list = [None] * 5
    if dispersao is None:
        dispersao = np.ones(5)

    for ini in range(0, n, _BLOCO):
        fim = min(ini + _BLOCO, n)
        k = float(morph[min(ini, n - 1)])
        f, adb, bw = _interp_vogal(naipe, vogal_de, vogal_para, k)
        trecho = fonte[ini:fim].astype(np.float64)
        for i in range(5):
            amp = (10 ** (adb[i] / 20.0)) * ganho_formante
            b, a = _ressoador(f[i] * escala * dispersao[i], bw[i] * escala, sr, amp)
            if estados[i] is None:
                estados[i] = np.zeros(2)
            if _lfilter is not None:
                y, estados[i] = _lfilter(b, a, trecho, zi=estados[i])
            else:                       # pragma: no cover
                y = _filtro_direto(b, a, trecho)
            # sinais alternados: e o que o Klatt faz no ramo paralelo, para que
            # dois formantes vizinhos nao se cancelem no vale entre eles
            saida[ini:fim] += y if i % 2 == 0 else -y
    return saida.astype(np.float32)


def _filtro_direto(b, a, x):            # pragma: no cover — so sem scipy
    y = np.zeros_like(x)
    for n in range(len(x)):
        acc = b[0] * x[n]
        if n >= 1:
            acc -= a[1] * y[n - 1]
        if n >= 2:
            acc -= a[2] * y[n - 2]
        y[n] = acc
    return y


# --------------------------------------------------------------------------
# Instrumento
# --------------------------------------------------------------------------
@dataclass
class VozInstrument:
    """Um cantor. Mesmo contrato de Instrument e SamplerInstrument."""
    naipe: str = "soprano"
    vogal: str = "a"
    vogal_final: str = ""          # "" = nao muda; senao a vogal morfa ate ela
    morph_s: float = 0.55          # em quanto tempo a vogal vira a final
    name: str = ""
    gain: float = 0.55
    pan: float = 0.0
    env: ADSR = field(default_factory=lambda: ADSR(0.09, 0.18, 0.88, 0.42))

    # --- o que separa "voz" de "orgao com formante" ----------------------
    vibrato_hz: float = 5.4
    vibrato_depth_cents: float = 42.0
    vibrato_atraso_s: float = 0.32     # cantor nao entra com vibrato ligado
    jitter: float = 0.006              # tremor de periodo (0.5-1% no humano)
    shimmer: float = 0.05              # tremor de amplitude
    sopro: float = 0.055               # quanto ar escapa junto
    formante_cantor_db: float = 7.0    # o pico de 3 kHz do canto treinado
    quociente_aberto: float = 0.60     # voz apertada x voz aberta
    portamento_s: float = 0.07         # ligadura entre notas proximas
    escala_trato: float = 1.0          # <1 = trato mais longo (voz maior)
    dispersao_formante: float = 0.0    # espalha os formantes: coro, nao solo
    ar_direto: float = 0.020           # aspiracao que NAO passa pelo trato

    # --- compatibilidade com o resto do app ------------------------------
    detune_cents: float = 0.0
    is_drum: bool = False
    osc: str = "voz"
    wavetable: str = "gb_sine"
    arpeggio: tuple = ()
    bits: int = 0
    sample_rate_crush: int = 0
    lowpass_hz: float = 0.0
    drive: float = 1.0
    seed: int = 0

    def __post_init__(self):
        if not self.name:
            self.name = f"{NAIPES_PT.get(self.naipe, self.naipe)} · {self.vogal}"

    # ----------------------------------------------------------------
    def _rng(self, pitch: float) -> np.random.Generator:
        """Semente derivada de tudo que distingue esta voz desta nota.

        E o detalhe que faz a dobra de secao virar coro: o engine clona o
        instrumento mudando so `detune_cents`, e por isso cada copia precisa
        tremer diferente. Se o jitter fosse igual, quatro vozes em unissono
        continuariam sendo uma voz — so mais alta.
        """
        chave = (int(self.seed) * 1000003
                 + int(round(self.detune_cents * 97))
                 + int(round(pitch * 13))
                 + int(round(self.pan * 211)))
        return np.random.default_rng(abs(chave) % (2 ** 31))

    def _curva_pitch(self, pitch: float, n: int, sr: int,
                     prev_pitch, rng) -> np.ndarray:
        semis = np.full(n, float(pitch), dtype=np.float64)

        # portamento: so entre notas proximas. Salto de oitava o cantor ataca
        # direto; segunda maior ele liga.
        if prev_pitch is not None and self.portamento_s > 0:
            salto = abs(float(prev_pitch) - float(pitch))
            if 0 < salto <= 5:
                ns = min(n, max(1, int(self.portamento_s * sr * (salto / 5.0 + 0.4))))
                semis[:ns] = np.linspace(float(prev_pitch), float(pitch), ns)

        t = np.arange(n) / sr
        # vibrato entra depois do ataque e cresce; e assim que se canta
        if self.vibrato_depth_cents > 0:
            sobe = np.clip((t - self.vibrato_atraso_s) / 0.45, 0.0, 1.0)
            taxa = self.vibrato_hz * (1.0 + 0.05 * np.sin(2 * np.pi * 0.31 * t
                                                          + rng.uniform(0, 6.28)))
            fase = 2 * np.pi * np.cumsum(taxa) / sr + rng.uniform(0, 6.28)
            semis += sobe * (self.vibrato_depth_cents / 100.0) * np.sin(fase)

        # jitter: ruido LENTO, nao branco. Voz treme devagar; ruido branco em
        # cima do pitch soa como fita estragada, nao como voz.
        if self.jitter > 0:
            passos = max(2, int(n / (sr * 0.045)))
            bruto = rng.normal(0, 1, passos)
            liso = np.interp(np.linspace(0, passos - 1, n),
                             np.arange(passos), bruto)
            semis += liso * self.jitter * 3.0

        semis += self.detune_cents / 100.0
        return semis

    # ----------------------------------------------------------------
    def render_note(self, pitch: float, dur_s: float, sr: int,
                    velocity: int = 100, prev_pitch=None,
                    release: bool = True) -> np.ndarray:
        n = max(1, int(dur_s * sr))
        rel_n = max(1, int(self.env.release * sr)) if release else 0
        total = n + rel_n
        rng = self._rng(pitch)

        semis = self._curva_pitch(pitch, total, sr, prev_pitch, rng)
        hz = 440.0 * (2.0 ** ((semis - 69.0) / 12.0))
        fase = np.cumsum(hz) / sr

        # Velocity manda na PRESSAO, nao so no volume: cantar forte soa
        # brilhante, e nao apenas alto.
        #
        # O que controla esse brilho e a duracao do FECHO da glote, nao o
        # quociente aberto. O fecho e a descontinuidade do pulso — o "chute" —
        # e a largura de uma descontinuidade e que decide ate onde vai o
        # espectro dela. Variar so o quociente aberto muda a forma da parte
        # suave do pulso e deixa a inclinacao espectral praticamente intacta
        # (medido: 0,2 dB entre 0,40 e 0,70). Encurtar o fecho de 22% para 8%
        # do periodo levanta o agudo em varios decibeis, que e o efeito real.
        v = float(np.clip(velocity / 127.0, 0.05, 1.0))
        fecho = float(np.clip(0.24 - 0.17 * v, 0.06, 0.30))
        fonte = _pulso_glotal(fase, self.quociente_aberto, fecho)

        if self.shimmer > 0:
            passos = max(2, int(total / (sr * 0.06)))
            bruto = rng.normal(0, 1, passos)
            liso = np.interp(np.linspace(0, passos - 1, total),
                             np.arange(passos), bruto)
            fonte = fonte * (1.0 + self.shimmer * liso).astype(np.float32)

        # ar: sopro passando pela mesma boca. Filtrado alto, porque aspiracao e
        # ruido de turbulencia, nao ruido branco.
        if self.sopro > 0:
            ar = rng.normal(0, 1, total).astype(np.float32)
            ar = ar - FX.lowpass(ar, sr, 700.0)
            fonte = fonte + ar * self.sopro * (1.25 - 0.5 * v)

        alvo = self.vogal_final or self.vogal
        if alvo != self.vogal and self.morph_s > 0:
            ns = max(1, int(self.morph_s * sr))
            morph = np.clip(np.arange(total) / ns, 0.0, 1.0)
        else:
            morph = np.zeros(total)

        disp = np.ones(5)
        if self.dispersao_formante > 0:
            disp = 1.0 + rng.normal(0, self.dispersao_formante, 5)

        y = _banco_formantes(fonte, sr, self.naipe, self.vogal, alvo, morph,
                             escala=self.escala_trato, dispersao=disp) * _NIVEL

        # Um caminho de ar que NAO passa pelo trato. Fisicamente e a turbulencia
        # que escapa nos labios; praticamente, e a unica fonte de energia acima
        # de 7 kHz que a voz tem — o banco de formantes morre no F5, por volta
        # de 5 kHz, e sem isto o canto sai com a mesma falta de brilho que
        # denuncia sample de teclado.
        if self.ar_direto > 0:
            sopro = rng.normal(0, 1, total).astype(np.float32)
            sopro = sopro - FX.lowpass(sopro, sr, 2600.0)
            # cresce com o esforco, ao contrario do `sopro` que passa pelo
            # trato: aquele e a fenda que nao fecha (voz fraca e mais soprada),
            # este e a turbulencia nos labios, que aumenta quando se canta forte
            y = y + sopro * self.ar_direto * (0.7 + 0.5 * v)

        # formante do cantor: a ressonancia de 2.8-3.2 kHz que um cantor
        # treinado produz e que faz a voz passar POR CIMA de oitenta musicos
        # sem gritar. Sem isso, voz somada a orquestra some.
        if self.formante_cantor_db:
            from . import fx as FXX
            centro = 2900.0 if self.naipe in ("soprano", "contralto") else 2600.0
            y = FXX.biquad(y, sr, "peak", centro * self.escala_trato, 2.2,
                           self.formante_cantor_db * (0.55 + 0.45 * v))

        env = self.env.render(n, sr, release_tail=rel_n)
        if len(env) < total:
            env = np.pad(env, (0, total - len(env)))
        y = y[:total] * env[:total]

        # a boca nao abre instantaneamente: uma rampa curta na entrada tira o
        # clique de sintetizador do ataque de cada nota
        n_ab = min(total, max(1, int(0.035 * sr)))
        abertura = np.ones(total, dtype=np.float32)
        abertura[:n_ab] = np.linspace(0.35, 1.0, n_ab)
        y = y * abertura

        y = y * (0.35 + 0.65 * v) * self.gain
        if self.lowpass_hz:
            y = FX.lowpass(y, sr, self.lowpass_hz)
        if self.drive != 1.0:
            y = FX.soft_clip(y, self.drive)
        if self.sample_rate_crush:
            y = FX.downsample(y, sr, self.sample_rate_crush)
        if self.bits:
            y = FX.bitcrush(y, self.bits)
        pico = float(np.abs(y).max())
        if pico > 4.0:                  # rede de seguranca do banco paralelo
            y = y * (4.0 / pico)
        return y.astype(np.float32)

    # ----------------------------------------------------------------
    def to_dict(self) -> dict:
        from dataclasses import asdict
        d = asdict(self)
        d["kind"] = "voz"
        d["env"] = asdict(self.env)
        d["arpeggio"] = list(self.arpeggio)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "VozInstrument":
        d = dict(d)
        d.pop("kind", None)
        env = d.pop("env", None)
        obj = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if env:
            obj.env = ADSR(**env)
        return obj


# --------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------
PRESETS_VOZ: dict[str, dict] = {
    "Solista feminina": dict(
        naipe="soprano", vogal="a", vibrato_depth_cents=48, vibrato_hz=5.5,
        sopro=0.06, formante_cantor_db=8.0, gain=0.821, jitter=0.007),
    "Solista feminina (escura)": dict(
        naipe="contralto", vogal="o", vibrato_depth_cents=38, vibrato_hz=5.0,
        sopro=0.07, formante_cantor_db=6.0, escala_trato=0.96, gain=0.481),
    "Solista masculino": dict(
        naipe="tenor", vogal="a", vibrato_depth_cents=40, vibrato_hz=5.2,
        sopro=0.05, formante_cantor_db=7.5, gain=0.448),
    "Lamento (u)": dict(
        naipe="soprano", vogal="u", vogal_final="o", morph_s=0.9,
        vibrato_depth_cents=30, sopro=0.11, formante_cantor_db=4.0, gain=0.519),
    "Coro soprano (aah)": dict(
        naipe="soprano", vogal="a", vibrato_depth_cents=26, jitter=0.010,
        dispersao_formante=0.035, formante_cantor_db=3.0, gain=0.916),
    "Coro contralto (ooh)": dict(
        naipe="contralto", vogal="o", vibrato_depth_cents=22, jitter=0.010,
        dispersao_formante=0.035, formante_cantor_db=2.0, gain=0.504),
    "Coro tenor (aah)": dict(
        naipe="tenor", vogal="a", vibrato_depth_cents=22, jitter=0.011,
        dispersao_formante=0.04, formante_cantor_db=3.0, gain=0.49),
    "Coro baixo (ooh)": dict(
        naipe="baixo", vogal="o", vibrato_depth_cents=18, jitter=0.011,
        dispersao_formante=0.04, formante_cantor_db=1.0, gain=0.608),
    "Coro de boca fechada": dict(
        naipe="contralto", vogal="m", vibrato_depth_cents=16, jitter=0.009,
        dispersao_formante=0.03, sopro=0.02, formante_cantor_db=0.0, gain=0.464),
    "Sussurro": dict(
        naipe="contralto", vogal="u", sopro=0.85, jitter=0.02,
        vibrato_depth_cents=8, formante_cantor_db=0.0, gain=0.602),
}


def voz(preset: str = "Solista feminina", **kw) -> VozInstrument:
    """Cria um cantor a partir de um preset, sobrescrevendo o que quiser."""
    base = dict(PRESETS_VOZ.get(preset, PRESETS_VOZ["Solista feminina"]))
    base.update(kw)
    inst = VozInstrument(**base)
    if "name" not in kw:
        inst.name = preset
    return inst


def nomes_de_preset() -> list[str]:
    return list(PRESETS_VOZ)


# Dobra pronta para Track.section: quantas gargantas, quanto desafinadas e
# quanto espalhadas. Coro de verdade nao e unissono — e um enxame estreito.
SECAO_CORO = {"vozes": 5, "detune": 14.0, "spread_ms": 42.0, "pan": 0.75}
SECAO_CORO_INTIMO = {"vozes": 3, "detune": 9.0, "spread_ms": 26.0, "pan": 0.45}
SECAO_SOLISTA_DOBRADA = {"vozes": 2, "detune": 5.0, "spread_ms": 14.0, "pan": 0.18}


def satb(pitches_sopr, acordes) -> dict[str, list[int]]:
    """Distribui cada acorde nas quatro vozes respeitando a tessitura.

    Regra de coral: o soprano recebe a melodia, o baixo recebe a fundamental,
    e o que sobra vai para contralto e tenor sem sair da faixa de cada naipe.
    """
    out: dict[str, list[int]] = {"soprano": [], "contralto": [], "tenor": [], "baixo": []}
    for sop, acorde in zip(pitches_sopr, acordes):
        pcs = sorted({p % 12 for p in acorde})
        baixo = acorde[0]
        while baixo > TESSITURA["baixo"][1]:
            baixo -= 12
        while baixo < TESSITURA["baixo"][0]:
            baixo += 12
        internas = []
        for alvo, naipe in ((sop - 3, "contralto"), (sop - 8, "tenor")):
            lo, hi = TESSITURA[naipe]
            cand = [pc + 12 * o for pc in pcs for o in range(1, 9)
                    if lo <= pc + 12 * o <= hi]
            internas.append(min(cand, key=lambda p: abs(alvo - p)) if cand else alvo)
        out["soprano"].append(int(sop))
        out["contralto"].append(int(internas[0]))
        out["tenor"].append(int(internas[1]))
        out["baixo"].append(int(baixo))
    return out
