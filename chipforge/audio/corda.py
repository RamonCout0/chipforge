"""
chipforge.audio.corda
=====================
Corda dedilhada e palhetada por modelagem fisica (guitarra e baixo).

Por que isto existe: um banco GM tem "Distortion Guitar", e ele e um sample
sustentado de uma nota ja distorcida. Passar isso pelo simulador de
amplificador e distorcer o que ja estava distorcido — o resultado nao e uma
guitarra alta, e uma lama alta. E, pior, faltam nele as tres coisas que fazem
uma guitarra soar como guitarra:

  * o ATAQUE DA PALHETA, que e um transiente de alguns milissegundos e muda
    completamente com a forca e o angulo;
  * o PALM MUTE, que nao e "mais baixo": e a corda perdendo o agudo em 40 ms
    enquanto a fundamental continua — o "chug";
  * a POSICAO em que a palheta bate, que apaga os harmonicos cujo NO cai
    naquele ponto, e e por isso que perto do cavalete soa nasal e no meio do
    braco soa redondo.

Nenhuma das tres esta num sample estatico, e nenhuma delas se equaliza depois.

O modelo e um guia de onda digital (Karplus-Strong estendido):

    excitacao (palheta)  ->  [ linha de atraso de N amostras ]  ->  saida
                                     ^                    |
                                     |   filtro de perda  |
                                     +--- (passa-baixa) <-+

A altura sai do comprimento da linha; o timbre sai do filtro de perda; o
ataque sai da excitacao; a posicao da palheta sai de um pente somado na
saida. Tudo que uma corda faz, o loop faz.

**Como isto roda rapido em Python.** A recursao e y[n] = f(y[n-N]), que parece
obrigar um laco amostra a amostra. Nao obriga: dentro de um bloco de N
amostras, tudo de que ele depende ja foi escrito no bloco anterior. Processando
em blocos do tamanho do atraso, sobra apenas uma recursao de PRIMEIRA ordem (o
filtro de perda), que o `lfilter` resolve vetorizado. E o mesmo truque que o
reverb usa nos filtros pente.
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

# Afinacao padrao, da corda mais grave para a mais aguda (MIDI).
AFINACAO_GUITARRA = (40, 45, 50, 55, 59, 64)          # E2 A2 D3 G3 B3 E4
AFINACAO_DROP_D = (38, 45, 50, 55, 59, 64)
AFINACAO_BAIXO = (28, 33, 38, 43)                     # E1 A1 D2 G2


def _hz(pitch: float) -> float:
    return 440.0 * (2.0 ** ((pitch - 69.0) / 12.0))


def _rng(seed: int, pitch: float, dur: float, extra: float = 0.0):
    return np.random.default_rng(
        abs(int(seed) * 7919 + int(pitch * 131) + int(dur * 977)
            + int(extra * 31)) % (2 ** 31))


# --------------------------------------------------------------------------
# O guia de onda
# --------------------------------------------------------------------------
def _atraso_do_filtro(a: float, f0: float, sr: int) -> float:
    """Atraso de fase que o filtro de perda acrescenta ao laco, em amostras.

    O laco nao tem o comprimento da linha de atraso: tem o comprimento da linha
    MAIS o atraso do filtro que esta dentro dele. Ignorar isso deixa a corda
    grave — e pior, deixa grave por uma quantidade que MUDA com o abafamento,
    porque o palm mute abaixa o corte e aumenta o atraso. Medido antes desta
    correcao: -9 cents com a corda solta e -35 cents no chug pesado. Um riff
    que desafina conforme a mao direita fecha nao e um efeito, e um defeito.
    """
    w = 2.0 * np.pi * max(f0, 1.0) / sr
    if a <= 1e-9 or w <= 1e-9:
        return 0.0
    return float(np.arctan2(a * np.sin(w), 1.0 - a * np.cos(w)) / w)


def _laco_de_corda(excitacao: np.ndarray, atraso_total: float, perda: float,
                   corte: float, sr: int, f0: float) -> np.ndarray:
    """A corda propriamente dita: atraso realimentado com perda no agudo.

    `perda` decide quanto tempo a nota dura; `corte` decide quao rapido o
    agudo morre ANTES do grave — que e o que separa uma corda de nylon de uma
    de aco, e a corda solta do palm mute.

    `atraso_total` e o periodo em amostras. O que sobra depois de descontar o
    atraso do filtro de perda vira linha de atraso inteira mais um allpass de
    primeira ordem para a parte fracionaria. Sem o allpass a afinacao erra ate
    25 cents no agudo: 44100/1318 da 33,5 amostras, e arredondar para 33 sobe
    a nota quase um quarto de tom.
    """
    n = len(excitacao)
    a = float(np.clip(np.exp(-2.0 * np.pi * corte / sr), 0.0, 0.999))

    alvo = max(2.5, atraso_total - _atraso_do_filtro(a, f0, sr))
    N = max(2, int(np.floor(alvo)))
    frac = alvo - N
    # o allpass de primeira ordem so e bem-comportado com atraso fracionario
    # entre 0.1 e 1.1; abaixo disso ele passa a ressoar
    if frac < 0.1:
        N -= 1
        frac += 1.0

    buf = np.zeros(n + N + 2, dtype=np.float64)

    # filtro de perda: um polo, com o ganho ajustado para que o ganho em DC
    # seja exatamente `perda` (senao a nota some ou explode conforme a nota)
    b_lp = np.array([(1.0 - a) * perda])
    a_lp = np.array([1.0, -a])
    zi_lp = np.zeros(1)

    # allpass de atraso fracionario: c = (1-frac)/(1+frac)
    c = (1.0 - frac) / (1.0 + frac) if frac > 1e-6 else 0.0
    b_ap = np.array([c, 1.0])
    a_ap = np.array([1.0, c])
    zi_ap = np.zeros(1)

    for ini in range(0, n, N):
        fim = min(ini + N, n)
        atrasado = buf[ini:fim]
        if _lfilter is not None:
            y, zi_lp = _lfilter(b_lp, a_lp, atrasado, zi=zi_lp)
            if c:
                y, zi_ap = _lfilter(b_ap, a_ap, y, zi=zi_ap)
        else:                            # pragma: no cover
            y = atrasado * perda
        buf[ini + N:fim + N] = y + excitacao[ini:fim]
    return buf[N:n + N].astype(np.float32)


def _palheta(n: int, sr: int, rng, dureza: float = 0.55,
             forca: float = 1.0) -> np.ndarray:
    """O transiente da palheta.

    Nao e um impulso e nao e ruido branco: e um ruido curto passado por uma
    passa-baixa cuja frequencia sobe com a DUREZA da palheta. Palheta fina e
    macia empurra a corda devagar e excita pouco harmonico; palheta grossa e
    dura solta a corda de uma vez e excita tudo. E por isso que a mesma
    guitarra soa doce ou agressiva sem mexer em nenhum botao.
    """
    dur = int(sr * (0.004 + 0.010 * (1.0 - dureza)))
    dur = max(8, dur)
    x = rng.normal(0.0, 1.0, dur).astype(np.float32)
    corte = 900.0 + 6500.0 * dureza
    x = FX.lowpass(x, sr, corte)
    # janela assimetrica: sobe em duas amostras, cai devagar. E o formato do
    # empurrao de uma palheta que escorrega da corda.
    jan = np.exp(-np.linspace(0.0, 5.0, dur)).astype(np.float32)
    jan[:2] *= np.linspace(0.0, 1.0, 2)
    x = x * jan
    pico = float(np.abs(x).max())
    if pico > 1e-9:
        x = x / pico
    saida = np.zeros(n, dtype=np.float32)
    saida[:min(dur, n)] = x[:min(dur, n)] * forca
    return saida


def _pente(x: np.ndarray, atraso: int, mistura: float = 1.0) -> np.ndarray:
    """Pente por subtracao: apaga os harmonicos cujo no cai no ponto tocado.

    E o modelo da posicao da palheta e do captador, e e uma subtracao mesmo —
    um harmonico que tem um no exatamente ali simplesmente nao e excitado.
    """
    if atraso < 1:
        return x
    atrasado = np.concatenate([np.zeros(atraso, np.float32), x])[:len(x)]
    return (x - mistura * atrasado).astype(np.float32)


# --------------------------------------------------------------------------
# Instrumento
# --------------------------------------------------------------------------
@dataclass
class CordaInstrument:
    """Uma corda palhetada. Mesmo contrato de Instrument e SamplerInstrument."""
    name: str = ""
    gain: float = 0.8
    pan: float = 0.0
    env: ADSR = field(default_factory=lambda: ADSR(0.0005, 0.02, 1.0, 0.10))

    # --- a corda ---------------------------------------------------------
    decaimento_s: float = 2.6          # quanto tempo a nota leva para sumir
    brilho: float = 3200.0             # corte do filtro de perda, em Hz
    pos_palheta: float = 0.14          # 0.5 = meio do braco, 0.08 = no cavalete
    dureza_palheta: float = 0.62
    pos_captador: float = 0.11
    # --- a mao direita ---------------------------------------------------
    abafamento: float = 0.0            # 0 = corda solta, 1 = palm mute fechado
    harmonico: int = 0                 # 0 = normal; 2..5 = harmonico natural
    ruido_de_dedo: float = 0.035       # o rangido ao trocar de casa
    # --- o instrumento ---------------------------------------------------
    afinacao: tuple = AFINACAO_GUITARRA
    corpo_hz: float = 0.0              # >0 liga uma ressonancia de caixa
    detune_cents: float = 0.0

    # --- compatibilidade com o resto do app ------------------------------
    is_drum: bool = False
    osc: str = "corda"
    wavetable: str = "gb_sine"
    arpeggio: tuple = ()
    vibrato_depth_cents: float = 0.0
    bits: int = 0
    sample_rate_crush: int = 0
    lowpass_hz: float = 0.0
    drive: float = 1.0
    seed: int = 0

    def __post_init__(self):
        if not self.name:
            self.name = "Corda"

    # ----------------------------------------------------------------
    def _corda_usada(self, pitch: int) -> int:
        """Qual corda toca esta nota — a mais aguda que a alcanca.

        Importa porque a mesma altura soa diferente em cordas diferentes: um
        Mi3 na corda de Mi grave (casa 12) e brilhante e curto; o mesmo Mi3 na
        corda de Re (casa 2) e grosso e longo. Guitarrista escolhe; aqui a
        regra e a mesma que a mao usa por reflexo: a corda mais grave que
        alcanca sem subir demais no braco.
        """
        candidatas = [c for c in self.afinacao if c <= pitch <= c + 22]
        return max(candidatas) if candidatas else self.afinacao[0]

    def render_note(self, pitch: float, dur_s: float, sr: int,
                    velocity: int = 100, prev_pitch=None,
                    release: bool = True) -> np.ndarray:
        v = float(np.clip(velocity / 127.0, 0.05, 1.0))
        rng = _rng(self.seed, pitch, dur_s, self.pan)

        # o palm mute encurta a nota de verdade: nao adianta so abaixar
        cauda = self.decaimento_s * (1.0 - 0.88 * self.abafamento)
        n = max(64, int((dur_s + (cauda if release else 0.0)) * sr))

        f0 = _hz(pitch + self.detune_cents / 100.0)
        if self.harmonico >= 2:
            f0 *= self.harmonico            # harmonico natural: toca o parcial

        atraso_exato = sr / max(f0, 20.0)
        atraso = max(2, int(atraso_exato))

        # perda por periodo para que caia 60 dB em `cauda` segundos
        periodos = max(1.0, cauda * f0)
        perda = float(np.clip(10 ** (-3.0 / periodos), 0.80, 0.9999))

        # o abafamento derruba o agudo muito antes do grave: e ISSO que faz o
        # "chug". Um palm mute equalizado depois soa abafado, nao percussivo.
        corte = self.brilho * (1.0 - 0.86 * self.abafamento) + 120.0
        if self.harmonico >= 2:
            corte *= 1.6                    # harmonico e mais brilhante

        exc = _palheta(n, sr, rng, self.dureza_palheta,
                       0.35 + 0.65 * v) * (0.5 + 0.5 * v)

        # harmonico natural: a palheta encosta no no do parcial e mata a
        # fundamental. O pente exatamente no no e o modelo disso.
        if self.harmonico >= 2:
            exc = _pente(exc, max(1, atraso // self.harmonico), 0.95)
        else:
            pos = float(np.clip(self.pos_palheta, 0.02, 0.5))
            exc = _pente(exc, max(1, int(atraso * pos)), 0.85)

        y = _laco_de_corda(exc, atraso_exato, perda, corte, sr, f0)

        if self.pos_captador > 0:
            y = _pente(y, max(1, int(atraso * float(np.clip(self.pos_captador,
                                                            0.02, 0.5)))), 0.7)

        # rangido de dedo na troca de casa — so quando a mao realmente anda
        if self.ruido_de_dedo > 0 and prev_pitch is not None \
                and abs(float(prev_pitch) - float(pitch)) >= 2:
            nr = min(n, int(0.035 * sr))
            r = rng.normal(0, 1, nr).astype(np.float32)
            r = r - FX.lowpass(r, sr, 1800.0)
            y[:nr] += r * self.ruido_de_dedo * v * np.linspace(1.0, 0.0, nr)

        if self.corpo_hz:
            from . import fx as FXX
            y = FXX.biquad(y, sr, "peak", self.corpo_hz, 1.1, 5.0)

        # corte da nota: a mao direita para a corda. Sem isso a nota anterior
        # continua soando por baixo da proxima e o riff vira nuvem.
        n_nota = int(dur_s * sr)
        if release and n_nota < n:
            queda = int(min(n - n_nota, max(0.012 * sr,
                                            0.09 * sr * (1 - self.abafamento))))
            if queda > 1:
                y[n_nota:n_nota + queda] *= np.linspace(1.0, 0.0, queda)
                y[n_nota + queda:] = 0.0

        y = y * self.gain * (0.4 + 0.6 * v)
        if self.lowpass_hz:
            y = FX.lowpass(y, sr, self.lowpass_hz)
        if self.drive != 1.0:
            y = FX.soft_clip(y, self.drive)
        if self.sample_rate_crush:
            y = FX.downsample(y, sr, self.sample_rate_crush)
        if self.bits:
            y = FX.bitcrush(y, self.bits)
        pico = float(np.abs(y).max())
        if pico > 2.0:
            y = y * (2.0 / pico)
        return y.astype(np.float32)

    # ----------------------------------------------------------------
    def to_dict(self) -> dict:
        from dataclasses import asdict
        d = asdict(self)
        d["kind"] = "corda"
        d["env"] = asdict(self.env)
        d["arpeggio"] = list(self.arpeggio)
        d["afinacao"] = list(self.afinacao)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CordaInstrument":
        d = dict(d)
        d.pop("kind", None)
        env = d.pop("env", None)
        if "afinacao" in d:
            d["afinacao"] = tuple(d["afinacao"])
        if "arpeggio" in d:
            d["arpeggio"] = tuple(d["arpeggio"])
        obj = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if env:
            obj.env = ADSR(**env)
        return obj


# --------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------
PRESETS_CORDA: dict[str, dict] = {
    # --- guitarra eletrica: a saida daqui e LIMPA, para entrar no amp -----
    "Guitarra (aberta)": dict(
        decaimento_s=3.2, brilho=3400.0, pos_palheta=0.13, dureza_palheta=0.66,
        pos_captador=0.10, gain=0.75),
    "Guitarra (chug)": dict(
        decaimento_s=2.4, brilho=3600.0, pos_palheta=0.09, dureza_palheta=0.78,
        pos_captador=0.09, abafamento=0.72, gain=0.85),
    "Guitarra (chug pesado)": dict(
        decaimento_s=2.2, brilho=3800.0, pos_palheta=0.07, dureza_palheta=0.85,
        pos_captador=0.08, abafamento=0.86, gain=0.9),
    "Guitarra (lead)": dict(
        decaimento_s=4.5, brilho=3000.0, pos_palheta=0.22, dureza_palheta=0.55,
        pos_captador=0.16, gain=0.7),
    "Guitarra (harmonico)": dict(
        decaimento_s=3.8, brilho=3200.0, pos_palheta=0.10, dureza_palheta=0.8,
        harmonico=4, gain=0.65),
    "Guitarra limpa": dict(
        decaimento_s=4.0, brilho=4200.0, pos_palheta=0.18, dureza_palheta=0.5,
        pos_captador=0.14, corpo_hz=210.0, gain=0.62),
    "Violao (aco)": dict(
        decaimento_s=5.0, brilho=4800.0, pos_palheta=0.20, dureza_palheta=0.72,
        pos_captador=0.0, corpo_hz=190.0, ruido_de_dedo=0.06, gain=0.6),
    "Violao (nylon)": dict(
        decaimento_s=4.2, brilho=2600.0, pos_palheta=0.26, dureza_palheta=0.34,
        pos_captador=0.0, corpo_hz=170.0, ruido_de_dedo=0.05, gain=0.6),
    # --- baixo -----------------------------------------------------------
    "Baixo (dedo)": dict(
        decaimento_s=3.4, brilho=1500.0, pos_palheta=0.22, dureza_palheta=0.28,
        pos_captador=0.14, afinacao=AFINACAO_BAIXO, gain=0.85),
    "Baixo (palheta)": dict(
        decaimento_s=2.8, brilho=2600.0, pos_palheta=0.10, dureza_palheta=0.72,
        pos_captador=0.11, afinacao=AFINACAO_BAIXO, gain=0.85),
    "Baixo (abafado)": dict(
        decaimento_s=2.2, brilho=1800.0, pos_palheta=0.16, dureza_palheta=0.5,
        abafamento=0.55, afinacao=AFINACAO_BAIXO, gain=0.9),
}


def corda(preset: str = "Guitarra (aberta)", **kw) -> CordaInstrument:
    base = dict(PRESETS_CORDA.get(preset, PRESETS_CORDA["Guitarra (aberta)"]))
    base.update(kw)
    inst = CordaInstrument(**base)
    if "name" not in kw:
        inst.name = preset
    return inst


def nomes_de_preset() -> list[str]:
    return list(PRESETS_CORDA)
