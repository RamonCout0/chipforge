"""
chipforge.core.expressao
========================
O que separa "MIDI tocando" de "alguem tocando".

Um arranjo escrito no grid tem todas as notas exatamente no tempo, todas com a
mesma intensidade e todas com a mesma duracao. Nenhum musico faz isso, e o
ouvido percebe na hora — mesmo sem saber apontar o motivo, soa mecanico.

Este modulo aplica as tres coisas que um interprete faz sem pensar:

  * chega alguns milissegundos antes ou depois do tempo exato
  * toca mais forte no alto da frase e alivia no fim
  * liga umas notas e destaca outras

Tudo funciona sobre listas de Note e devolve listas novas — da para encadear.
"""
from __future__ import annotations

import random
from typing import Iterable, Sequence

from .project import Note


def _clone(n: Note) -> Note:
    return Note(n.start, n.dur, n.pitch, n.velocity)


def humanizar(notas: Iterable[Note], tempo_ms: float = 14.0,
              velocity: int = 9, ppq: int = 480, bpm: float = 120.0,
              seed: int | None = 7, tendencia_ms: float = 0.0) -> list[Note]:
    """Tira as notas do grid perfeito.

    `tendencia_ms` desloca a track inteira: negativo soa "na frente" (bateria de
    rock costuma puxar), positivo soa "atras" (naipe de cordas costuma arrastar
    alguns milissegundos). E esse atraso relativo entre camadas que da a
    sensacao de gente tocando junto em vez de um sequenciador disparando tudo
    no mesmo instante.
    """
    rng = random.Random(seed)
    ticks_por_ms = ppq * bpm / 60000.0
    desloc = tendencia_ms * ticks_por_ms
    espalho = tempo_ms * ticks_por_ms
    out = []
    for n in notas:
        c = _clone(n)
        c.start = max(0, int(round(c.start + desloc + rng.gauss(0, espalho / 2))))
        c.velocity = max(1, min(127, c.velocity + int(rng.gauss(0, velocity / 2))))
        out.append(c)
    return out


CURVAS = ("sino", "crescendo", "diminuendo", "onda")


def arco(notas: Sequence[Note], curva: str = "sino", intensidade: float = 0.28,
         por_compasso: int | None = None) -> list[Note]:
    """Dinamica de frase: nenhuma frase e plana do comeco ao fim.

    'sino' cresce ate o meio e alivia (o padrao de quase toda frase melodica),
    'crescendo' e 'diminuendo' fazem o obvio, 'onda' respira a cada compasso.
    `intensidade` e quanto da velocity original pode variar, em fracao.
    """
    notas = list(notas)
    if not notas:
        return []
    ini = min(n.start for n in notas)
    fim = max(n.start + n.dur for n in notas)
    span = max(1, fim - ini)
    out = []
    for n in notas:
        c = _clone(n)
        x = (c.start - ini) / span
        if curva == "crescendo":
            f = -1.0 + 2.0 * x
        elif curva == "diminuendo":
            f = 1.0 - 2.0 * x
        elif curva == "onda" and por_compasso:
            import math
            f = math.sin(2 * math.pi * (c.start - ini) / por_compasso)
        else:                                   # sino
            f = 1.0 - abs(2.0 * x - 1.0) * 2.0
        c.velocity = max(1, min(127, int(round(c.velocity * (1 + intensidade * f)))))
        out.append(c)
    return out


def acentuar(notas: Iterable[Note], ppq: int = 480, beats_por_compasso: int = 4,
             fortes: Sequence[float] = (0.0,), ganho: int = 14,
             fracos: int = -6) -> list[Note]:
    """Marca os tempos fortes e alivia o resto. Sem isso o compasso some."""
    bar = ppq * beats_por_compasso
    alvos = {int(b * ppq) for b in fortes}
    out = []
    for n in notas:
        c = _clone(n)
        pos = c.start % bar
        perto = min((abs(pos - a) for a in alvos), default=10 ** 9)
        delta = ganho if perto <= ppq // 16 else fracos
        c.velocity = max(1, min(127, c.velocity + delta))
        out.append(c)
    return out


def articular(notas: Sequence[Note], modo: str = "legato",
              folga: int = 12, fator: float = 0.55) -> list[Note]:
    """'legato' estica cada nota ate a proxima da MESMA linha; 'staccato'
    encurta; 'tenuto' segura quase tudo. Articulacao muda a leitura da frase
    tanto quanto as alturas."""
    notas = sorted(notas, key=lambda n: (n.start, n.pitch))
    out = [_clone(n) for n in notas]
    if modo == "staccato":
        for c in out:
            c.dur = max(30, int(c.dur * fator))
        return out
    if modo == "tenuto":
        for c in out:
            c.dur = max(30, int(c.dur * 1.08))
        return out
    for i, c in enumerate(out):
        prox = next((o for o in out[i + 1:] if o.start > c.start), None)
        if prox:
            c.dur = max(30, prox.start - c.start - folga)
    return out


def respirar(notas: Sequence[Note], a_cada: int = 1920, folga: int = 90,
             ppq: int = 480) -> list[Note]:
    """Abre um respiro no fim de cada frase encurtando a ultima nota dela.

    Cantor e sopro precisam respirar; cordas trocam o arco. Uma linha que nunca
    para de soar denuncia que nao tem ninguem tocando."""
    notas = sorted(notas, key=lambda n: n.start)
    out = [_clone(n) for n in notas]
    for i, c in enumerate(out):
        prox = out[i + 1] if i + 1 < len(out) else None
        if prox and (prox.start // a_cada) > (c.start // a_cada):
            c.dur = max(60, min(c.dur, prox.start - c.start - folga))
    return out


def rampa_velocity(notas: Iterable[Note], de: int, para: int) -> list[Note]:
    """Crescendo/diminuendo explicito ao longo de um trecho inteiro."""
    notas = sorted(notas, key=lambda n: n.start)
    if not notas:
        return []
    ini, fim = notas[0].start, max(n.start for n in notas)
    span = max(1, fim - ini)
    out = []
    for n in notas:
        c = _clone(n)
        x = (c.start - ini) / span
        c.velocity = max(1, min(127, int(round(de + (para - de) * x))))
        out.append(c)
    return out


def aplicar(notas: Sequence[Note], *passos) -> list[Note]:
    """Encadeia: aplicar(tema, lambda n: arco(n), lambda n: humanizar(n))."""
    resultado = list(notas)
    for passo in passos:
        resultado = passo(resultado)
    return resultado
