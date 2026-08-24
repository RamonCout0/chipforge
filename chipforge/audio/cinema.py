"""
chipforge.audio.cinema
======================
Instrumentos de desenho de som: impacto, riser, sub drop, braam e swell
invertido.

Por que isto existe: o que separa "orquestra tocando" de "trilha de trailer"
quase nunca esta na partitura. Esta nos eventos que nenhum instrumento produz —
o BUM que abre a secao, o ruido que sobe durante quatro compassos e some
exatamente no tempo 1, o naco de grave que cai quando a imagem corta. Um banco
GM nao tem nada disso, e nao deveria ter: nao sao instrumentos, sao efeitos.

Todos aqui respeitam o mesmo contrato de `Instrument`, entao entram numa track
normal, aceitam inserts e envio de sala, e aparecem no piano roll. A nota diz
QUANDO e por QUANTO tempo; a altura diz a afinacao do evento, e a velocity diz
o tamanho.

O detalhe que faz funcionar: a duracao da nota MODELA o efeito, nao apenas o
corta. Um riser de dois compassos e um riser de dois compassos, nao um riser de
um compasso com silencio depois — a subida inteira se estica.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import effects as FX
from . import fx
from .envelope import ADSR


def _rng(seed: int, pitch: float, dur: float) -> np.random.Generator:
    return np.random.default_rng(
        abs(int(seed) * 7919 + int(pitch * 31) + int(dur * 1000)) % (2 ** 31))


def _hz(pitch: float) -> float:
    return 440.0 * (2.0 ** ((pitch - 69.0) / 12.0))


def _curva(n: int, forma: str = "exp", k: float = 4.0) -> np.ndarray:
    t = np.linspace(0.0, 1.0, max(n, 1), dtype=np.float32)
    if forma == "exp":
        return np.exp(-k * t)
    if forma == "sobe":
        return (t ** k).astype(np.float32)
    if forma == "sino":
        return np.sin(np.pi * t) ** 1.5
    return np.ones_like(t)


@dataclass
class _Base:
    """Campos que o resto do app espera encontrar em qualquer instrumento."""
    name: str = ""
    gain: float = 0.8
    pan: float = 0.0
    env: ADSR = field(default_factory=lambda: ADSR(0.001, 0.05, 0.0, 0.25))
    is_drum: bool = False
    osc: str = "cinema"
    wavetable: str = "gb_sine"
    arpeggio: tuple = ()
    vibrato_depth_cents: float = 0.0
    detune_cents: float = 0.0
    bits: int = 0
    sample_rate_crush: int = 0
    lowpass_hz: float = 0.0
    drive: float = 1.0
    seed: int = 0

    def _pos(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32) * self.gain
        if self.lowpass_hz:
            y = FX.lowpass(y, 44100, self.lowpass_hz)
        if self.drive != 1.0:
            y = FX.soft_clip(y, self.drive)
        if self.bits:
            y = FX.bitcrush(y, self.bits)
        pico = float(np.abs(y).max())
        if pico > 1.6:
            y = y * (1.6 / pico)
        return y.astype(np.float32)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        d = asdict(self)
        d["kind"] = "cinema"
        d["classe"] = type(self).__name__
        d["env"] = asdict(self.env)
        d["arpeggio"] = list(self.arpeggio)
        return d

    @classmethod
    def from_dict(cls, d: dict):
        d = dict(d)
        d.pop("kind", None)
        alvo = CLASSES.get(d.pop("classe", cls.__name__), cls)
        env = d.pop("env", None)
        obj = alvo(**{k: v for k, v in d.items() if k in alvo.__dataclass_fields__})
        if env:
            obj.env = ADSR(**env)
        return obj


# ==========================================================================
@dataclass
class Impacto(_Base):
    """O BUM. Tres camadas, porque impacto de verdade tem tres.

    1. SUB — uma senoide que cai uma oitava e meia em 200 ms. E ela que se
       sente no peito, e ela sozinha soa como nada.
    2. CORPO — ruido filtrado em torno de 120 Hz, decaindo rapido: a pele do
       tambor, a madeira, o chao.
    3. METAL — uma faisca de agudo com 40 ms de vida. Sem ela o impacto soa
       abafado; com ela, o ouvido localiza o evento no tempo com precisao.

    O sub e um pouco ATRASADO em relacao ao metal, uns 12 ms. Isso e o que
    acontece num tambor grande de verdade (a membrana leva tempo para
    desenvolver o modo fundamental) e e o que faz o impacto soar GRANDE em vez
    de apenas alto.
    """
    queda_semitons: float = 19.0
    tempo_queda_s: float = 0.22
    cauda_s: float = 1.6
    metal: float = 0.35
    corpo: float = 0.8
    atraso_sub_ms: float = 12.0

    def render_note(self, pitch: float, dur_s: float, sr: int,
                    velocity: int = 100, prev_pitch=None,
                    release: bool = True) -> np.ndarray:
        v = float(np.clip(velocity / 127.0, 0.05, 1.0))
        n = max(int(sr * max(dur_s, 0.05)), int(sr * self.cauda_s * (0.5 + 0.5 * v)))
        rng = _rng(self.seed, pitch, dur_s)
        t = np.arange(n) / sr

        f0 = _hz(pitch + self.detune_cents / 100.0)
        nq = max(1, int(self.tempo_queda_s * sr))
        semis = np.concatenate([
            np.linspace(self.queda_semitons, 0.0, min(nq, n)) ** 1.0,
            np.zeros(max(0, n - nq))])[:n]
        hz = f0 * (2.0 ** (semis / 12.0))
        sub = np.sin(2 * np.pi * np.cumsum(hz) / sr) * _curva(n, "exp", 3.2)

        atraso = int(self.atraso_sub_ms * sr / 1000)
        if atraso:
            sub = np.concatenate([np.zeros(atraso, np.float32), sub])[:n]

        ruido = rng.normal(0, 1, n).astype(np.float32)
        corpo = fx.biquad(ruido, sr, "lowpass", 190.0, 1.1) * _curva(n, "exp", 9.0)
        corpo = corpo + fx.biquad(ruido, sr, "peak", 320.0, 1.6, 6.0) * _curva(n, "exp", 16.0) * 0.5

        metal = fx.biquad(ruido, sr, "highpass", 3200.0, 0.7) * _curva(n, "exp", 90.0)

        y = (sub * 1.0 + corpo * self.corpo * 0.9 + metal * self.metal)
        y = np.tanh(y * (1.1 + 1.4 * v)) * (0.35 + 0.65 * v)
        return self._pos(y)


@dataclass
class Riser(_Base):
    """Ruido e altura subindo durante a nota inteira, e corte seco no fim.

    O corte e o efeito. Um riser que decai desperdica a tensao que acabou de
    construir: o silencio de 30 ms entre o topo do riser e o downbeat e o que
    faz o downbeat parecer duas vezes maior do que e.
    """
    subida_semitons: float = 28.0
    corte_ms: float = 30.0
    ruido: float = 0.65
    tom: float = 0.45
    tremolo_hz: float = 0.0

    def render_note(self, pitch: float, dur_s: float, sr: int,
                    velocity: int = 100, prev_pitch=None,
                    release: bool = True) -> np.ndarray:
        v = float(np.clip(velocity / 127.0, 0.05, 1.0))
        n = max(64, int(dur_s * sr))
        rng = _rng(self.seed, pitch, dur_s)
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)
        forma = t ** 2.1                       # acelera perto do fim

        f0 = _hz(pitch)
        hz = f0 * (2.0 ** (forma * self.subida_semitons / 12.0))
        tonal = np.sin(2 * np.pi * np.cumsum(hz) / sr, dtype=np.float64).astype(np.float32)
        tonal += 0.5 * np.sin(2 * np.pi * np.cumsum(hz * 1.5) / sr).astype(np.float32)

        br = rng.normal(0, 1, n).astype(np.float32)
        # filtro que abre por blocos: bandpass caminhando de 400 Hz ao agudo
        saida = np.zeros(n, dtype=np.float32)
        passo = max(1024, n // 96)
        for i in range(0, n, passo):
            j = min(i + passo, n)
            k = float(forma[min(i, n - 1)])
            centro = 400.0 * (2.0 ** (k * 5.2))
            trecho = fx.biquad(br[i:j], sr, "highpass", min(centro, sr * 0.45), 0.8)
            saida[i:j] = fx.biquad(trecho, sr, "lowpass",
                                   min(centro * 3.0, sr * 0.46), 0.9)

        env = (0.05 + 0.95 * forma)
        if self.tremolo_hz:
            eixo = np.arange(n) / sr
            env = env * (0.75 + 0.25 * np.sin(2 * np.pi * self.tremolo_hz * eixo
                                              * (0.5 + forma)))
        y = (saida * self.ruido + tonal * self.tom * 0.5) * env

        corte = max(1, int(self.corte_ms * sr / 1000))
        if corte < n:
            y[-corte:] *= np.linspace(1.0, 0.0, corte) ** 2
        return self._pos(y * (0.4 + 0.6 * v))


@dataclass
class SubDrop(_Base):
    """A queda de grave. Serve para marcar corte, nao para tocar nota."""
    queda_semitons: float = 26.0
    cauda_s: float = 2.2

    def render_note(self, pitch: float, dur_s: float, sr: int,
                    velocity: int = 100, prev_pitch=None,
                    release: bool = True) -> np.ndarray:
        v = float(np.clip(velocity / 127.0, 0.05, 1.0))
        n = max(int(sr * max(dur_s, 0.2)), int(sr * self.cauda_s))
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)
        hz = _hz(pitch) * (2.0 ** (-(t ** 0.6) * self.queda_semitons / 12.0))
        hz = np.maximum(hz, 18.0)
        y = np.sin(2 * np.pi * np.cumsum(hz) / sr).astype(np.float32)
        y *= np.exp(-2.6 * t)
        # um pouco de segunda harmonica: sub puro some em caixa de som pequena,
        # e a harmonica e o que faz o grave ser "ouvido" mesmo sem ser sentido
        y = y + 0.22 * np.sin(4 * np.pi * np.cumsum(hz) / sr).astype(np.float32) * np.exp(-3.4 * t)
        return self._pos(np.tanh(y * 1.4) * (0.4 + 0.6 * v))


@dataclass
class Braam(_Base):
    """O acorde de metais distorcido de trailer.

    Nao e um metal de verdade: sao seis serras desafinadas passando por um
    filtro que abre no ataque e por saturacao. Metal de verdade num banco GM
    soa educado; isto soa como ameaca. Sao coisas diferentes e as duas tem uso.
    """
    vozes: int = 6
    detune: float = 22.0
    abertura_hz: float = 5200.0
    corte_base: float = 320.0
    sujeira: float = 2.6

    def render_note(self, pitch: float, dur_s: float, sr: int,
                    velocity: int = 100, prev_pitch=None,
                    release: bool = True) -> np.ndarray:
        v = float(np.clip(velocity / 127.0, 0.05, 1.0))
        rel = max(1, int(0.35 * sr))
        n = max(64, int(dur_s * sr))
        total = n + rel
        rng = _rng(self.seed, pitch, dur_s)
        f0 = _hz(pitch + self.detune_cents / 100.0)

        y = np.zeros(total, dtype=np.float32)
        for i in range(self.vozes):
            k = (i / max(1, self.vozes - 1)) - 0.5
            cents = k * self.detune * 2 + rng.normal(0, 2.5)
            f = f0 * (2.0 ** (cents / 1200.0))
            fase = np.arange(total) * f / sr + rng.random()
            y += (2.0 * np.mod(fase, 1.0) - 1.0).astype(np.float32)
        y /= self.vozes

        ataque = max(1, int(0.09 * sr))
        env = np.ones(total, dtype=np.float32)
        env[:ataque] = np.linspace(0.0, 1.0, ataque) ** 0.7
        env[n:] = np.linspace(1.0, 0.0, total - n) ** 1.6
        y *= env

        # o filtro abre junto com o ataque: e o "aaaAAA" do braam
        passo = max(512, total // 40)
        aberto = np.zeros(total, dtype=np.float32)
        for i in range(0, total, passo):
            j = min(i + passo, total)
            k = min(1.0, (i / sr) / 0.5)
            corte = self.corte_base + (self.abertura_hz - self.corte_base) * (k ** 0.6)
            aberto[i:j] = fx.biquad(y[i:j], sr, "lowpass", corte, 1.3)
        y = np.tanh(aberto * self.sujeira * (0.6 + 0.8 * v))
        y = fx.biquad(y, sr, "highpass", 55.0, 0.7)
        return self._pos(y * (0.35 + 0.65 * v) * 0.5)


@dataclass
class SwellInvertido(_Base):
    """Prato invertido: cresce do nada e some no downbeat.

    O truque de estudio mais antigo que existe para preparar uma entrada, e o
    unico jeito de dar ao ouvinte um aviso de que algo vai mudar sem tocar uma
    unica nota.
    """
    brilho_final: float = 9000.0
    corte_ms: float = 40.0

    def render_note(self, pitch: float, dur_s: float, sr: int,
                    velocity: int = 100, prev_pitch=None,
                    release: bool = True) -> np.ndarray:
        v = float(np.clip(velocity / 127.0, 0.05, 1.0))
        n = max(64, int(dur_s * sr))
        rng = _rng(self.seed, pitch, dur_s)
        br = rng.normal(0, 1, n).astype(np.float32)
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)

        saida = np.zeros(n, dtype=np.float32)
        passo = max(1024, n // 64)
        for i in range(0, n, passo):
            j = min(i + passo, n)
            k = float(t[min(i, n - 1)])
            corte = 900.0 + (self.brilho_final - 900.0) * (k ** 1.4)
            saida[i:j] = fx.biquad(br[i:j], sr, "highpass",
                                   min(corte, sr * 0.45), 0.7)
        y = saida * (t ** 2.4)
        corte = max(1, int(self.corte_ms * sr / 1000))
        if corte < n:
            y[-corte:] *= np.linspace(1.0, 0.0, corte) ** 2
        return self._pos(y * (0.4 + 0.6 * v) * 0.7)


@dataclass
class Taiko(_Base):
    """Tambor grande de pele. Nao e o bumbo de bateria e nao e o timpani.

    Timpani tem afinacao clara (e por isso toca a fundamental do acorde);
    taiko tem PESO e um decaimento curto de madeira. Numa secao rapida, o
    timpani vira lama e o taiko continua legivel.
    """
    corpo_hz: float = 78.0
    pele: float = 0.55
    decaimento: float = 6.5

    def render_note(self, pitch: float, dur_s: float, sr: int,
                    velocity: int = 100, prev_pitch=None,
                    release: bool = True) -> np.ndarray:
        v = float(np.clip(velocity / 127.0, 0.05, 1.0))
        n = max(int(sr * 0.55), int(dur_s * sr))
        rng = _rng(self.seed, pitch, dur_s)
        t = np.arange(n) / sr
        f = self.corpo_hz * (2.0 ** ((pitch - 36) / 12.0))
        # a afinacao cai um pouco no ataque: e a pele esticando
        hz = f * (1.0 + 0.35 * np.exp(-40.0 * t))
        corpo = np.sin(2 * np.pi * np.cumsum(hz) / sr).astype(np.float32)
        corpo *= np.exp(-self.decaimento * t)
        corpo += 0.35 * np.sin(2 * np.pi * np.cumsum(hz * 1.58) / sr).astype(np.float32) \
            * np.exp(-self.decaimento * 2.4 * t)

        br = rng.normal(0, 1, n).astype(np.float32)
        batida = fx.biquad(br, sr, "peak", 900.0, 1.0, 8.0)
        batida = fx.biquad(batida, sr, "highpass", 400.0, 0.7) * np.exp(-55.0 * t)

        y = corpo + batida * self.pele
        return self._pos(np.tanh(y * (1.2 + 0.9 * v)) * (0.3 + 0.7 * v))


@dataclass
class Relogio(_Base):
    """Tique-taque. Duas camadas, porque o escape de um relogio tem duas.

    1. O CLIQUE — ruido de banda estreita com 3 ms de vida. E ele que o ouvido
       usa para cravar o instante, e por isso ele nao pode ter cauda nenhuma.
    2. A RESSONANCIA — a caixa de madeira respondendo, uma senoide curta em
       torno de 400 Hz. Sem ela o tique soa como um estalo digital; com ela,
       soa como um objeto num comodo.

    A altura da nota controla o brilho do clique, entao escrever alturas
    ALTERNADAS produz o tique-taque: dois sons parecidos e nao iguais, que e o
    que o escape de verdade faz — a roda passa por dois dentes diferentes.
    """
    corpo_hz: float = 380.0
    aspereza: float = 0.55          # quanto de madeira contra metal
    cauda_s: float = 0.09

    def render_note(self, pitch: float, dur_s: float, sr: int,
                    velocity: int = 100, prev_pitch=None,
                    release: bool = True) -> np.ndarray:
        v = float(np.clip(velocity / 127.0, 0.05, 1.0))
        n = max(64, int(sr * self.cauda_s))
        rng = _rng(self.seed, pitch, dur_s)
        t = np.arange(n) / sr

        brilho = 1400.0 * (2.0 ** ((pitch - 60) / 12.0))
        br = rng.normal(0, 1, n).astype(np.float32)
        clique = fx.biquad(br, sr, "peak", float(np.clip(brilho, 300, 12000)),
                           2.5, 12.0)
        clique = fx.biquad(clique, sr, "highpass", 700.0, 0.7)
        clique *= np.exp(-380.0 * t)

        f = self.corpo_hz * (2.0 ** ((pitch - 60) / 24.0))
        corpo = np.sin(2 * np.pi * f * t).astype(np.float32) * np.exp(-70.0 * t)

        y = clique * self.aspereza + corpo * (1.0 - self.aspereza) * 0.8
        return self._pos(y * (0.45 + 0.55 * v))


@dataclass
class Vento(_Base):
    """Vento e mata: ruido de banda larga que RESPIRA.

    O que faz uma cama de ruido soar como um lugar em vez de um chiado e a
    variacao lenta — o filtro andando e o nivel subindo e descendo em ciclos de
    varios segundos, sem periodo audivel. Ruido estatico o ouvido descarta como
    defeito de gravacao em poucos segundos; ruido que respira ele aceita como
    ambiente e para de notar, que e exatamente o que se quer de um fundo.

    A nota nao tem altura: a duracao e o unico parametro que importa.
    """
    centro_hz: float = 620.0
    largura: float = 2.2            # oitavas de excursao do filtro
    respiracao_s: float = 7.0
    galhos: float = 0.0             # estalos ocasionais de madeira

    def render_note(self, pitch: float, dur_s: float, sr: int,
                    velocity: int = 100, prev_pitch=None,
                    release: bool = True) -> np.ndarray:
        v = float(np.clip(velocity / 127.0, 0.05, 1.0))
        n = max(1024, int(dur_s * sr))
        rng = _rng(self.seed, pitch, dur_s)
        br = rng.normal(0, 1, n).astype(np.float32)

        # tres osciladores lentos incomensuraveis: a soma nunca se repete
        t = np.arange(n) / sr
        lento = (np.sin(2 * np.pi * t / self.respiracao_s)
                 + 0.6 * np.sin(2 * np.pi * t / (self.respiracao_s * 1.61) + 1.1)
                 + 0.4 * np.sin(2 * np.pi * t / (self.respiracao_s * 2.73) + 2.3))
        lento = lento / 2.0

        saida = np.zeros(n, dtype=np.float32)
        passo = max(2048, int(sr * 0.08))
        for i in range(0, n, passo):
            j = min(i + passo, n)
            k = float(lento[min(i, n - 1)])
            centro = self.centro_hz * (2.0 ** (k * self.largura * 0.5))
            centro = float(np.clip(centro, 60.0, sr * 0.4))
            trecho = fx.biquad(br[i:j], sr, "highpass", centro * 0.6, 0.7)
            saida[i:j] = fx.biquad(trecho, sr, "lowpass", centro * 2.4, 0.8)

        env = 0.45 + 0.55 * (0.5 + 0.5 * lento)
        # entra e sai por rampa longa: vento que comeca de repente e um efeito
        borda = min(n // 2, int(sr * 1.2))
        if borda > 1:
            env[:borda] *= np.linspace(0.0, 1.0, borda)
            env[-borda:] *= np.linspace(1.0, 0.0, borda)
        y = saida * env

        if self.galhos > 0:
            quantos = int(dur_s * self.galhos)
            for _ in range(quantos):
                i = int(rng.integers(0, max(1, n - 2000)))
                dur = int(rng.integers(200, 1400))
                estalo = rng.normal(0, 1, dur).astype(np.float32)
                estalo = fx.biquad(estalo, sr, "peak",
                                   float(rng.uniform(900, 4200)), 3.0, 10.0)
                estalo *= np.exp(-np.linspace(0, 14, dur)) * rng.uniform(0.2, 0.7)
                y[i:i + dur] += estalo

        return self._pos(y * (0.4 + 0.6 * v))


CLASSES = {c.__name__: c for c in
           (Impacto, Riser, SubDrop, Braam, SwellInvertido, Taiko, Relogio,
            Vento)}


# ==========================================================================
PRESETS_CINEMA: dict[str, tuple] = {
    "Impacto grave": (Impacto, dict(queda_semitons=22, cauda_s=2.2, metal=0.25,
                                    corpo=0.9, gain=0.85)),
    "Impacto seco": (Impacto, dict(queda_semitons=14, cauda_s=0.9, metal=0.5,
                                   corpo=0.6, gain=0.8)),
    "Impacto metalico": (Impacto, dict(queda_semitons=10, cauda_s=2.6, metal=1.0,
                                       corpo=0.4, gain=0.75)),
    "Riser longo": (Riser, dict(subida_semitons=30, ruido=0.7, tom=0.35, gain=0.55)),
    "Riser tenso": (Riser, dict(subida_semitons=20, ruido=0.45, tom=0.7,
                                tremolo_hz=7.0, gain=0.55)),
    "Sub drop": (SubDrop, dict(queda_semitons=26, cauda_s=2.4, gain=0.9)),
    "Braam": (Braam, dict(gain=0.8)),
    "Braam sujo": (Braam, dict(sujeira=4.5, detune=32, gain=0.75)),
    "Swell invertido": (SwellInvertido, dict(gain=0.6)),
    "Taiko": (Taiko, dict(gain=0.85)),
    "Taiko pequeno": (Taiko, dict(corpo_hz=112, decaimento=9.0, gain=0.8)),
    "Relogio": (Relogio, dict(gain=0.55)),
    "Relogio grande": (Relogio, dict(corpo_hz=210.0, aspereza=0.42,
                                     cauda_s=0.16, gain=0.6)),
    "Relogio de bolso": (Relogio, dict(corpo_hz=900.0, aspereza=0.8,
                                       cauda_s=0.05, gain=0.5)),
    "Vento": (Vento, dict(gain=0.5)),
    "Mata": (Vento, dict(centro_hz=480.0, largura=2.6, respiracao_s=9.0,
                         galhos=0.35, gain=0.5)),
    "Vento agudo": (Vento, dict(centro_hz=1600.0, largura=1.6,
                                respiracao_s=5.0, gain=0.42)),
}


def instrumento(preset: str, **kw):
    """Cria um instrumento de desenho de som pelo nome do preset."""
    if preset not in PRESETS_CINEMA:
        raise KeyError(f"preset de cinema desconhecido: {preset}")
    classe, base = PRESETS_CINEMA[preset]
    cfg = dict(base)
    cfg.update(kw)
    obj = classe(**cfg)
    if not obj.name:
        obj.name = preset
    return obj


def nomes_de_preset() -> list[str]:
    return list(PRESETS_CINEMA)
