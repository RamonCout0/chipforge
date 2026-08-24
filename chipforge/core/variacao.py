"""
chipforge.core.variacao
=======================
Desenvolvimento tematico — o antidoto para "ficou repetitivo".

Repetir um tema identico quatro vezes cansa em trinta segundos. O que os
arranjos que a gente admira fazem e outra coisa: o tema volta, mas volta
MUDADO. Este modulo traz as transformacoes classicas, as mesmas que Bach,
Beethoven e qualquer compositor de trilha usa:

    inverter     o desenho de cima para baixo
    retrogradar  de tras para frente
    aumentar     mesma melodia, notas duas vezes mais longas (pesa, agiganta)
    diminuir     notas pela metade (acelera, agita)
    fragmentar   pegar dois compassos e trabalhar so eles
    sequenciar   repetir um fragmento subindo por graus
    ornamentar   preencher os saltos com notas de passagem
    contracanto  uma segunda linha que anda ao contrario da primeira

Todas trabalham sobre listas de Note e devolvem listas novas.
"""
from __future__ import annotations

import random
from typing import Sequence

from . import theory as T
from .project import Note


def _clone(n: Note, **kw) -> Note:
    c = Note(n.start, n.dur, n.pitch, n.velocity)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def transpor(notas: Sequence[Note], semitons: int) -> list[Note]:
    return [_clone(n, pitch=max(0, min(127, n.pitch + semitons))) for n in notas]


def deslocar(notas: Sequence[Note], ticks: int) -> list[Note]:
    return [_clone(n, start=max(0, n.start + ticks)) for n in notas]


def inverter(notas: Sequence[Note], eixo: int | None = None,
             escala: tuple[int, str] | None = None) -> list[Note]:
    """Espelha as alturas em torno de um eixo. Subiu vira desceu.

    Com `escala=(tonica, modo)` a inversao e DIATONICA: as notas espelhadas
    caem de volta na escala, entao o resultado continua soando no tom em vez
    de virar uma linha cromatica estranha.
    """
    notas = sorted(notas, key=lambda n: n.start)
    if not notas:
        return []
    eixo = notas[0].pitch if eixo is None else eixo
    out = []
    for n in notas:
        p = 2 * eixo - n.pitch
        if escala:
            p = T.quantize_to_scale(p, escala[0], escala[1])
        out.append(_clone(n, pitch=max(0, min(127, p))))
    return out


def retrogradar(notas: Sequence[Note]) -> list[Note]:
    """De tras para frente, preservando o ritmo espelhado."""
    notas = sorted(notas, key=lambda n: n.start)
    if not notas:
        return []
    ini = notas[0].start
    fim = max(n.start + n.dur for n in notas)
    return sorted((_clone(n, start=ini + (fim - (n.start + n.dur)))
                   for n in notas), key=lambda n: n.start)


def escalar_tempo(notas: Sequence[Note], fator: float) -> list[Note]:
    """fator 2.0 = aumentacao (dobra tudo), 0.5 = diminuicao."""
    notas = sorted(notas, key=lambda n: n.start)
    if not notas:
        return []
    ini = notas[0].start
    return [_clone(n, start=int(ini + (n.start - ini) * fator),
                   dur=max(40, int(n.dur * fator))) for n in notas]


def aumentar(notas, fator: float = 2.0):
    return escalar_tempo(notas, fator)


def diminuir(notas, fator: float = 0.5):
    return escalar_tempo(notas, fator)


def fragmentar(notas: Sequence[Note], compasso_ini: int, compassos: int,
               ticks_por_compasso: int = 1920, rebase: bool = True) -> list[Note]:
    """Recorta N compassos do tema — a materia-prima do desenvolvimento."""
    a = compasso_ini * ticks_por_compasso
    b = a + compassos * ticks_por_compasso
    pedaco = [n for n in notas if a <= n.start < b]
    if rebase:
        pedaco = [_clone(n, start=n.start - a) for n in pedaco]
    return sorted(pedaco, key=lambda n: n.start)


def sequenciar(fragmento: Sequence[Note], intervalos: Sequence[int],
               passo_ticks: int, escala: tuple[int, str] | None = None,
               decaimento: int = 0) -> list[Note]:
    """Repete o fragmento subindo (ou descendo) a cada repeticao.

    E o motor de qualquer ponte que "vai crescendo": o mesmo gesto, cada vez
    um grau acima. Com `escala` a transposicao e diatonica.
    """
    out: list[Note] = []
    for i, iv in enumerate(intervalos):
        for n in fragmento:
            p = n.pitch + iv
            if escala:
                p = T.quantize_to_scale(p, escala[0], escala[1])
            out.append(Note(n.start + i * passo_ticks, n.dur,
                            max(0, min(127, p)),
                            max(1, n.velocity - i * decaimento)))
    return sorted(out, key=lambda n: n.start)


def ornamentar(notas: Sequence[Note], tonica: int, modo: str,
               densidade: float = 0.45, min_dur: int = 240,
               seed: int | None = 11) -> list[Note]:
    """Preenche saltos com notas de passagem e quebra notas longas.

    E o que um solista faz na segunda vez que toca o tema: a linha e a mesma,
    mas ela ganha movimento. Só mexe em notas longas o suficiente para caber
    a divisao, e as notas novas saem da escala do trecho.
    """
    rng = random.Random(seed)
    notas = sorted(notas, key=lambda n: n.start)
    graus = sorted(T.scale_pitch_classes(tonica, modo))
    out: list[Note] = []
    for i, n in enumerate(notas):
        prox = notas[i + 1] if i + 1 < len(notas) else None
        if (prox is None or n.dur < min_dur or rng.random() > densidade):
            out.append(_clone(n))
            continue
        salto = prox.pitch - n.pitch
        meio = n.dur // 2
        if abs(salto) >= 3:
            # nota de passagem no caminho entre as duas alturas
            alvo = n.pitch + (2 if salto > 0 else -2)
            alvo = T.quantize_to_scale(alvo, tonica, modo)
        else:
            # bordadura: sobe um grau e volta
            alvo = T.quantize_to_scale(n.pitch + 2, tonica, modo)
        out.append(_clone(n, dur=max(60, meio - 10)))
        out.append(Note(n.start + meio, max(60, n.dur - meio - 10),
                        max(0, min(127, alvo)), max(1, n.velocity - 12)))
    return sorted(out, key=lambda n: n.start)


def harmonizar(notas: Sequence[Note], tonica: int, modo: str,
               graus: int = -2) -> list[Note]:
    """Segunda voz em intervalo DIATONICO paralelo (tercas, sextas).

    `graus=-2` da a terca abaixo, `-5` a sexta abaixo, `+2` a terca acima.
    Como o intervalo e contado em graus da escala e nao em semitons, a segunda
    voz cai sempre dentro do tom — consonancia por construcao. E o naipe de
    sopros dobrando a melodia de qualquer trilha.
    """
    escala = sorted(T.scale_pitch_classes(tonica, modo))
    n_graus = len(escala)
    out = []
    for n in notas:
        pc = n.pitch % 12
        if pc in escala:
            idx = escala.index(pc)
        else:
            idx = min(range(n_graus), key=lambda i: abs(escala[i] - pc))
        novo_idx = idx + graus
        oitava_off = novo_idx // n_graus
        alvo_pc = escala[novo_idx % n_graus]
        base = n.pitch - pc + alvo_pc + 12 * oitava_off
        while base > n.pitch and graus < 0:
            base -= 12
        while base < n.pitch and graus > 0:
            base += 12
        out.append(_clone(n, pitch=max(0, min(127, base))))
    return out


def contracanto(tema: Sequence[Note], acordes: Sequence[Sequence[int]],
                ticks_por_compasso: int = 1920, oitava_base: int = 60,
                seed: int | None = 5,
                minimo: int | None = None, maximo: int | None = None) -> list[Note]:
    """Gera uma segunda linha que anda AO CONTRARIO do tema.

    Movimento contrario e o truque mais antigo do contraponto: quando a melodia
    sobe, a outra voz desce. Duas linhas independentes soam muito mais ricas do
    que melodia com bloco de acordes por baixo — e e exatamente o que falta num
    arranjo que soa repetitivo.

    A linha resultante cai sempre em nota do acorde do compasso, entao ela
    combina por construcao.
    """
    rng = random.Random(seed)
    tema = sorted(tema, key=lambda n: n.start)
    if not tema:
        return []
    out: list[Note] = []
    anterior = None
    for i, n in enumerate(tema):
        compasso = n.start // ticks_por_compasso
        if compasso >= len(acordes):
            compasso = len(acordes) - 1
        pcs = sorted({p % 12 for p in acordes[compasso]})
        if anterior is None:
            alvo = oitava_base
        else:
            direcao = -1 if n.pitch >= anterior[0] else 1
            alvo = anterior[1] + direcao * rng.choice((2, 3, 4))
        # A faixa e ESTREITA de proposito. Deixar a linha vagar meia oitava
        # para baixo faz ela cair em cima do pad e brigar de segunda o trecho
        # inteiro — foi exatamente assim que "O Ultimo Farol" desandou.
        lo = oitava_base - 3 if minimo is None else minimo
        hi = oitava_base + 9 if maximo is None else maximo
        cands = [oitava_base + 12 * o + pc for pc in pcs for o in (-1, 0, 1, 2)]
        cands = [c for c in cands if lo <= c <= hi]
        if not cands:
            continue
        escolha = min(cands, key=lambda c: abs(c - alvo))
        anterior = (n.pitch, escolha)
        out.append(Note(n.start, n.dur, escolha, max(1, n.velocity - 18)))

    # junta notas repetidas seguidas numa so — contracanto martelando cansa
    limpo: list[Note] = []
    for n in out:
        if limpo and limpo[-1].pitch == n.pitch and \
                limpo[-1].start + limpo[-1].dur >= n.start - 20:
            limpo[-1].dur = n.start + n.dur - limpo[-1].start
        else:
            limpo.append(n)
    return limpo


def variar(tema: Sequence[Note], modo: str, tonica: int = 0,
           escala: str = "natural_minor", **kw) -> list[Note]:
    """Atalho: variar(tema, 'ornamentar', 9, 'natural_minor')."""
    tabela = {
        "inverter": lambda: inverter(tema, escala=(tonica, escala), **kw),
        "retrogradar": lambda: retrogradar(tema),
        "aumentar": lambda: aumentar(tema, **kw),
        "diminuir": lambda: diminuir(tema, **kw),
        "ornamentar": lambda: ornamentar(tema, tonica, escala, **kw),
    }
    if modo not in tabela:
        raise ValueError(f"variacao desconhecida: {modo}")
    return tabela[modo]()
