"""
chipforge.core.dinamica
=======================
Mapa de tempo e automacao de parametro.

Por que isto existe: o projeto tinha UM bpm e UM volume por track, do primeiro
ao ultimo compasso. Com isso, qualquer render mede faixa de loudness perto de
zero — tudo soa no mesmo volume o tempo inteiro. E a diferenca mais audivel
entre uma trilha e um loop: trilha tem ARCO.

Duas coisas separadas moram aqui:

* **Mapa de tempo** — o andamento muda ao longo da peca. Accelerando na
  aproximacao do climax, ritardando na ultima frase, e a fermata antes do
  ultimo acorde. Sem isso a musica anda no metronomo, e metronomo e a
  assinatura sonora de MIDI.

* **Automacao** — uma curva por parametro, desenhada em ticks e lida em
  amostras. Volume de track e a mais usada (o crescendo de doze compassos que
  nenhuma velocity de nota consegue fazer sozinha), mas serve para qualquer
  numero.

Ambas sao listas de pares serializaveis: entram no .chip sem tratamento
especial.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# ==========================================================================
# Mapa de tempo
# ==========================================================================
@dataclass
class MapaDeTempo:
    """Andamento em funcao do tick, com interpolacao linear entre marcas.

    Interpolar o BPM linearmente (e nao em degraus) e o que faz um accelerando
    soar como accelerando. Em degraus, o ouvido escuta cada troca como um
    tropeco.

    A conversao tick -> segundos e a integral de 1/bpm, e ela e resolvida uma
    vez so, numa tabela cumulativa: chamar `segundos()` cinco mil vezes durante
    o render nao pode custar uma integracao cada vez.
    """
    ppq: int = 480
    bpm_base: float = 120.0
    marcas: list = field(default_factory=list)      # [(tick, bpm), ...]
    _tabela: tuple | None = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------
    def marcar(self, tick: int, bpm: float) -> "MapaDeTempo":
        self.marcas.append((int(tick), float(bpm)))
        self.marcas.sort(key=lambda m: m[0])
        self._tabela = None
        return self

    def rampa(self, de_tick: int, ate_tick: int, de_bpm: float,
              ate_bpm: float, passos: int = 8) -> "MapaDeTempo":
        """Accelerando ou ritardando entre dois pontos."""
        for i in range(passos + 1):
            k = i / passos
            self.marcar(int(de_tick + (ate_tick - de_tick) * k),
                        de_bpm + (ate_bpm - de_bpm) * k)
        return self

    def ritardando(self, ate_tick: int, compassos: int, ticks_por_compasso: int,
                   fator: float = 0.72) -> "MapaDeTempo":
        """Segura o andamento nos ultimos compassos. `fator` 0.72 = termina a
        72% da velocidade — o freio de fim de peca."""
        ini = ate_tick - compassos * ticks_por_compasso
        base = self.bpm_em(ini)
        return self.rampa(ini, ate_tick, base, base * fator)

    def accelerando(self, de_tick: int, compassos: int, ticks_por_compasso: int,
                    fator: float = 1.10) -> "MapaDeTempo":
        base = self.bpm_em(de_tick)
        return self.rampa(de_tick, de_tick + compassos * ticks_por_compasso,
                          base, base * fator)

    # ------------------------------------------------------------------
    def bpm_em(self, tick: float) -> float:
        if not self.marcas:
            return self.bpm_base
        ts = [m[0] for m in self.marcas]
        vs = [m[1] for m in self.marcas]
        if tick <= ts[0]:
            return vs[0]
        if tick >= ts[-1]:
            return vs[-1]
        return float(np.interp(tick, ts, vs))

    def _construir(self):
        """Tabela cumulativa de segundos, amostrada densamente em ticks."""
        fim = max([m[0] for m in self.marcas], default=0)
        passo = max(1, self.ppq // 16)
        n = int(fim / passo) + 2
        ticks = np.arange(n + 1) * passo
        bpm = np.array([self.bpm_em(t) for t in ticks])
        # dt de cada intervalo = passo_em_beats * 60 / bpm_medio do intervalo
        beats = passo / self.ppq
        medio = (bpm[:-1] + bpm[1:]) * 0.5
        dt = beats * 60.0 / np.maximum(medio, 1e-6)
        segs = np.concatenate([[0.0], np.cumsum(dt)])
        self._tabela = (ticks, segs, float(60.0 / max(bpm[-1], 1e-6) / self.ppq))

    def segundos(self, tick: float) -> float:
        """Instante, em segundos, do tick pedido."""
        if not self.marcas:
            return tick / self.ppq * (60.0 / self.bpm_base)
        if self._tabela is None:
            self._construir()
        ticks, segs, spt_final = self._tabela
        if tick <= ticks[-1]:
            return float(np.interp(tick, ticks, segs))
        return float(segs[-1] + (tick - ticks[-1]) * spt_final)

    def duracao_de(self, tick: float, dur_ticks: float) -> float:
        """Quanto dura, em segundos, uma nota que comeca em `tick`."""
        return max(1e-4, self.segundos(tick + dur_ticks) - self.segundos(tick))

    # ------------------------------------------------------------------
    def to_list(self) -> list:
        return [[int(t), float(b)] for t, b in self.marcas]

    @classmethod
    def from_list(cls, dados, ppq: int = 480, bpm_base: float = 120.0):
        m = cls(ppq=ppq, bpm_base=bpm_base)
        for t, b in dados or ():
            m.marcas.append((int(t), float(b)))
        m.marcas.sort(key=lambda x: x[0])
        return m


# ==========================================================================
# Automacao
# ==========================================================================
@dataclass
class Automacao:
    """Curva de um parametro no tempo, em ticks.

    `curva` e interpolada em amostras porque volume que muda em degraus produz
    um estalo a cada degrau: multiplicar um sinal por uma escada e modulacao em
    anel de banda larga, e o ouvido escuta isso como clique.
    """
    pontos: list = field(default_factory=list)      # [(tick, valor), ...]
    padrao: float = 1.0

    def ponto(self, tick: int, valor: float) -> "Automacao":
        self.pontos.append((int(tick), float(valor)))
        self.pontos.sort(key=lambda p: p[0])
        return self

    def rampa(self, de_tick: int, ate_tick: int, de: float,
              ate: float) -> "Automacao":
        self.ponto(de_tick, de)
        self.ponto(ate_tick, ate)
        return self

    def patamar(self, de_tick: int, ate_tick: int, valor: float) -> "Automacao":
        self.ponto(de_tick, valor)
        self.ponto(ate_tick, valor)
        return self

    def crescendo(self, de_tick: int, ate_tick: int, de: float = 0.35,
                  ate: float = 1.0) -> "Automacao":
        return self.rampa(de_tick, ate_tick, de, ate)

    def diminuendo(self, de_tick: int, ate_tick: int, de: float = 1.0,
                   ate: float = 0.3) -> "Automacao":
        return self.rampa(de_tick, ate_tick, de, ate)

    def valor(self, tick: float) -> float:
        if not self.pontos:
            return self.padrao
        ts = [p[0] for p in self.pontos]
        vs = [p[1] for p in self.pontos]
        return float(np.interp(tick, ts, vs))

    def curva(self, n_amostras: int, mapa: MapaDeTempo | None = None,
              sr: int = 44100, sec_per_tick: float = 0.0) -> np.ndarray:
        """Expande a curva para uma envoltoria com uma amostra por sample."""
        if not self.pontos:
            return np.full(n_amostras, self.padrao, dtype=np.float32)
        if mapa is not None:
            tempos = np.array([mapa.segundos(t) for t, _ in self.pontos])
        else:
            spt = sec_per_tick or 60.0 / 120.0 / 480
            tempos = np.array([t * spt for t, _ in self.pontos])
        vals = np.array([v for _, v in self.pontos], dtype=np.float64)
        eixo = np.arange(n_amostras) / sr
        return np.interp(eixo, tempos, vals).astype(np.float32)

    def to_list(self) -> list:
        return [[int(t), float(v)] for t, v in self.pontos]

    @classmethod
    def from_list(cls, dados, padrao: float = 1.0) -> "Automacao":
        a = cls(padrao=padrao)
        for t, v in dados or ():
            a.pontos.append((int(t), float(v)))
        a.pontos.sort(key=lambda p: p[0])
        return a


# ==========================================================================
# Atalhos de arco
# ==========================================================================
def arco_de_secoes(secoes: Sequence[tuple], ticks_por_compasso: int,
                   niveis: dict[str, float] | None = None,
                   transicao_compassos: float = 2.0) -> Automacao:
    """Monta a automacao de volume a partir da lista de secoes da peca.

    `secoes` e a mesma lista que os exemplos ja usam: [(nome, compassos), ...].
    `niveis` da o volume relativo de cada secao. A transicao entre elas nao e
    um degrau — sao dois compassos de rampa, que e como um regente faz.
    """
    niveis = niveis or {}
    a = Automacao()
    compasso = 0
    trans = transicao_compassos
    for nome, n_comp, *_ in secoes:
        v = float(niveis.get(nome, 1.0))
        ini = compasso * ticks_por_compasso
        fim = (compasso + n_comp) * ticks_por_compasso
        rampa = int(min(trans, n_comp * 0.4) * ticks_por_compasso)
        a.ponto(ini + rampa, v)
        a.ponto(max(ini + rampa, fim - rampa), v)
        compasso += n_comp
    if a.pontos:
        primeiro = a.pontos[0]
        a.ponto(0, primeiro[1])
    return a


def hairpin(notas, de: int = 60, ate: int = 110):
    """Crescendo escrito nas velocities de uma frase (nao no fader).

    Fader e sala: sobe o volume de tudo, inclusive do reverb ja gravado.
    Velocity e execucao: o instrumento toca com mais forca, e num sampler ou
    numa voz isso muda o TIMBRE, nao so o nivel. Um crescendo de verdade
    precisa dos dois, e este cuida do lado da execucao.
    """
    notas = list(notas)
    if not notas:
        return notas
    ini = min(n.start for n in notas)
    fim = max(n.start + n.dur for n in notas)
    span = max(1, fim - ini)
    saida = []
    for n in notas:
        k = (n.start - ini) / span
        v = int(np.clip(de + (ate - de) * k, 1, 127))
        saida.append(type(n)(n.start, n.dur, n.pitch, v))
    return saida
