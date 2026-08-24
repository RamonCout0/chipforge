"""
chipforge.core.orquestra
========================
Escrita orquestral: tessitura, condução de vozes, ostinato, divisi e
preparação de seção.

Por que isto existe: os geradores de `generators.py` foram feitos para
chiptune, onde qualquer nota cabe em qualquer canal. Orquestra não funciona
assim. Um acorde de quatro notas escrito abaixo de Dó3 vira lama em qualquer
sala; a mesma progressão tocada com inversões escolhidas ao acaso troca de
posição a cada compasso e soa como um teclado, não como um naipe; e ostinato
de cordas em bloco quantizado perfeito soa como sequenciador, porque dezesseis
músicos nunca atacam no mesmo milissegundo.

Este módulo é sobre as regras que nenhum banco de samples resolve por você.

Nada aqui produz áudio: entra acorde, sai lista de `Note`. É o mesmo contrato
de `generators`, e por isso tudo continua passando pelo medidor de harmonia.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .project import Note

# --------------------------------------------------------------------------
# Tessitura — a faixa em que cada família soa como ela mesma
# --------------------------------------------------------------------------
# Não são os limites físicos do instrumento: são os limites de onde ele soa
# BEM. Um violino toca até Lá7, mas escrever ali é pedir estridência.
TESSITURA = {
    "contrabaixo": (28, 55),
    "violoncelo": (36, 69),
    "viola": (48, 76),
    "violino": (55, 91),
    "violino2": (55, 84),
    "trompa": (41, 72),
    "trompete": (55, 84),
    "trombone": (40, 72),
    "tuba": (28, 58),
    "flauta": (60, 93),
    "oboe": (58, 86),
    "clarinete": (50, 84),
    "fagote": (34, 69),
    "harpa": (24, 96),
    "piano": (21, 108),
    "coro_soprano": (60, 81),
    "coro_contralto": (53, 74),
    "coro_tenor": (48, 69),
    "coro_baixo": (40, 62),
    "timpani": (36, 55),
}

# Abaixo daqui, dois sons a menos de uma sexta viram lama. É a regra prática
# mais útil de orquestração e a que mais se ignora ao escrever no piano roll.
LIMITE_GRAVE = 48          # Dó3
INTERVALO_MINIMO_GRAVE = 7  # quinta justa


def encaixar(pitch: int, instrumento: str) -> int:
    """Traz uma nota para dentro da tessitura, por oitavas."""
    lo, hi = TESSITURA.get(instrumento, (0, 127))
    while pitch < lo:
        pitch += 12
    while pitch > hi:
        pitch -= 12
    return int(np.clip(pitch, lo, hi))


def limpar_grave(acorde: Sequence[int],
                 limite: int = LIMITE_GRAVE,
                 minimo: int = INTERVALO_MINIMO_GRAVE) -> list[int]:
    """Abre os intervalos apertados abaixo do limite, subindo a nota de cima.

    É a correção que o medidor de harmonia já aponta como "grave sujo", feita
    antes de escrever em vez de depois de ouvir.
    """
    notas = sorted(set(int(p) for p in acorde))

    # Sobe a nota de cima do par apertado uma oitava por vez, reordenando a
    # cada passo. Reordenar e o que importa: comparar cada nota so com a
    # ANTERIOR de uma lista que ja foi alterada faz o erro cascatear — uma
    # triade de tres notas saia espalhada por tres oitavas, porque cada
    # correcao empurrava a proxima para mais longe ainda.
    #
    # A subida tambem para assim que a nota passa do limite: acima de Do3 dois
    # sons a uma terca nao sao lama, sao um acorde normal, e continuar abrindo
    # so destruiria a voicing.
    for _ in range(len(notas) * 4):
        for i in range(len(notas) - 1):
            baixo, alto = notas[i], notas[i + 1]
            if baixo < limite and alto - baixo < minimo:
                notas[i + 1] = alto + 12
                notas.sort()
                break
        else:
            break
    return notas


# --------------------------------------------------------------------------
# Condução de vozes
# --------------------------------------------------------------------------
def _melhor_inversao(alvo: Sequence[int], pcs: Sequence[int],
                     n_vozes: int, low: int, high: int) -> list[int]:
    """Escolhe as alturas do próximo acorde que MENOS mexem as vozes."""
    candidatos: list[list[int]] = []
    todas = sorted({pc + 12 * o for pc in set(p % 12 for p in pcs)
                    for o in range(0, 11) if low <= pc + 12 * o <= high})
    if not todas:
        return list(alvo)
    for base in range(len(todas)):
        escolha = todas[base:base + n_vozes]
        if len(escolha) == n_vozes:
            candidatos.append(escolha)
    if not candidatos:
        return list(alvo)

    def custo(c):
        pares = min(len(c), len(alvo))
        mov = sum(abs(c[i] - alvo[i]) for i in range(pares))
        # penaliza abrir demais entre vozes vizinhas: naipe é um bloco
        abertura = sum(max(0, (c[i + 1] - c[i]) - 12) for i in range(len(c) - 1))
        return mov + abertura * 1.5

    return min(candidatos, key=custo)


def bloco(acordes: Sequence[Sequence[int]], ppq: int,
          compassos_por_acorde: int = 1, beats_por_compasso: int = 4,
          n_vozes: int = 4, low: int = 55, high: int = 84,
          velocity: int = 72, ligado: bool = True) -> list[Note]:
    """Acompanhamento em bloco com condução de vozes.

    A diferença entre isto e empilhar a tríade em posição fundamental toda vez:
    aqui cada acorde escolhe a inversão que exige o menor movimento a partir do
    acorde anterior. É o que faz um naipe de cordas soar como um naipe — as
    vozes andam por segundas, não saltam juntas em paralelo.

    `ligado` amarra cada acorde ao próximo (sem buraco entre eles), que é como
    se toca um pad sustentado.

    **Cuidado com `compassos_por_acorde`.** Ele diz quanto tempo cada acorde da
    LISTA dura — não decima a lista. Passar 2 com a progressão inteira faz esta
    voz andar na metade da velocidade das outras: no compasso 2 ela ainda
    segura o acorde 1 enquanto o resto da orquestra já foi para o 2, e o
    resultado é uma parede de segundas maiores. Se a intenção é harmonia mais
    lenta nesta voz, passe também a lista decimada — `acordes[::2]`.
    """
    passo = ppq * beats_por_compasso * compassos_por_acorde
    notas: list[Note] = []
    anterior: list[int] = []
    for i, acorde in enumerate(acordes):
        if anterior:
            vozes = _melhor_inversao(anterior, acorde, n_vozes, low, high)
        else:
            todas = sorted({pc % 12 + 12 * o for pc in acorde
                            for o in range(0, 11)
                            if low <= pc % 12 + 12 * o <= high})
            vozes = todas[:n_vozes] or list(acorde[:n_vozes])
        vozes = limpar_grave(vozes)
        dur = passo if ligado else int(passo * 0.92)
        for p in vozes:
            notas.append(Note(i * passo, dur, int(p), velocity))
        anterior = vozes
    return notas


def divisi(acorde: Sequence[int], n_vozes: int = 5,
           low: int = 48, high: int = 88) -> list[int]:
    """Espalha um acorde por várias oitavas, com o grave aberto.

    Divisi de verdade não dobra a mesma nota em uníssono: distribui, deixando
    as fundamentais embaixo e as tensões em cima. Escrever a sétima no
    violoncelo e a fundamental no violino é o caminho mais curto para uma
    orquestração que não abre.
    """
    pcs = sorted({p % 12 for p in acorde})
    raiz = acorde[0] % 12
    ordem = [raiz] + [pc for pc in pcs if pc != raiz]
    saida: list[int] = []
    oit = low
    for i in range(n_vozes):
        pc = ordem[i % len(ordem)]
        p = pc + 12 * ((oit - pc) // 12 + (1 if pc < oit % 12 else 0))
        while p < oit:
            p += 12
        while p > high:
            p -= 12
        saida.append(int(p))
        oit += 12 if i % len(ordem) == len(ordem) - 1 else 4
    return limpar_grave(sorted(set(saida)))


# --------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------
def ostinato(acordes: Sequence[Sequence[int]], ppq: int,
             compassos_por_acorde: int = 1, beats_por_compasso: int = 4,
             divisao: int = 4, low: int = 62, high: int = 84,
             velocity: int = 74, contorno: str = "sobe_desce") -> list[Note]:
    """A figura repetida que carrega a energia da peça.

    `divisao` é quantas notas por tempo: 2 = colcheia, 4 = semicolcheia.
    `contorno` decide o desenho dentro do acorde. Ostinato bom não é escala
    nem arpejo completo — são três ou quatro alturas que voltam, porque é a
    REPETIÇÃO que hipnotiza, e uma figura complexa demais nunca é reconhecida
    como figura.
    """
    passo_nota = max(1, ppq // divisao)
    por_acorde = ppq * beats_por_compasso * compassos_por_acorde
    n_por_acorde = max(1, por_acorde // passo_nota)
    notas: list[Note] = []
    for i, acorde in enumerate(acordes):
        alturas = [p for p in divisi(acorde, 4, low, high) if low <= p <= high]
        if not alturas:
            alturas = [encaixar(acorde[0], "violino")]
        if contorno == "sobe":
            seq = alturas
        elif contorno == "desce":
            seq = alturas[::-1]
        elif contorno == "pendulo":
            seq = [alturas[0], alturas[-1], alturas[1 % len(alturas)], alturas[-1]]
        else:                                   # sobe_desce
            seq = alturas + alturas[-2:0:-1]
        base = i * por_acorde
        for k in range(n_por_acorde):
            p = seq[k % len(seq)]
            # o primeiro de cada tempo é mais forte: é o que dá pulso ao
            # ostinato em vez de uma esteira uniforme
            forte = (k % divisao) == 0
            v = int(np.clip(velocity + (8 if forte else -4), 1, 127))
            notas.append(Note(base + k * passo_nota, max(1, int(passo_nota * 0.92)),
                              int(p), v))
    return notas


def tremolo(acordes: Sequence[Sequence[int]], ppq: int,
            compassos_por_acorde: int = 1, beats_por_compasso: int = 4,
            low: int = 60, high: int = 84, velocity: int = 66,
            divisao: int = 8) -> list[Note]:
    """Tremolo de cordas: a mesma altura repetida muito rápido.

    É tensão pura e o recurso mais barato de toda a orquestração — e o que
    engana quando feito com sample: repetir a MESMA amostra trinta vezes por
    segundo soa como um motor. Aqui as velocities oscilam de propósito.
    """
    passo = max(1, ppq // divisao)
    por_acorde = ppq * beats_por_compasso * compassos_por_acorde
    notas: list[Note] = []
    for i, acorde in enumerate(acordes):
        alturas = divisi(acorde, 3, low, high)
        base = i * por_acorde
        for k in range(max(1, por_acorde // passo)):
            v = int(np.clip(velocity + (6 if k % 2 == 0 else -6)
                            + (k % 5) - 2, 1, 127))
            for p in alturas:
                notas.append(Note(base + k * passo, passo, int(p), v))
    return notas


def pedal(acordes: Sequence[Sequence[int]], ppq: int,
          compassos_por_acorde: int = 1, beats_por_compasso: int = 4,
          instrumento: str = "contrabaixo", velocity: int = 84,
          oitava_dobrada: bool = True) -> list[Note]:
    """Fundamental sustentada no grave, opcionalmente dobrada na oitava.

    A oitava dobrada é o que os contrabaixos e violoncelos fazem por padrão na
    orquestra clássica, e é o que dá definição a um grave que, sozinho, some
    em alto-falante pequeno.
    """
    passo = ppq * beats_por_compasso * compassos_por_acorde
    notas: list[Note] = []
    for i, acorde in enumerate(acordes):
        raiz = encaixar(acorde[0], instrumento)
        notas.append(Note(i * passo, passo, raiz, velocity))
        if oitava_dobrada:
            alto = raiz + 12
            lo, hi = TESSITURA.get(instrumento, (0, 127))
            if alto <= hi + 12:
                notas.append(Note(i * passo, passo, alto,
                                  max(1, velocity - 12)))
    return notas


# --------------------------------------------------------------------------
# Percussão cinematográfica
# --------------------------------------------------------------------------
PADROES_TAIKO = {
    # cada string é um compasso em semicolcheias: X forte, x fraco, . silêncio
    "marcha":    "X...x...X...x...",
    "galope":    "X.xxX.xxX.xxX.xx",
    "tribal":    "X..xX...x.X.X...",
    "guerra":    "X.X.x.X.X.X.xxXx",
    "meio":      "X.......X.......",
    "adensa":    "X...X...X.X.XxXx",
    "final":     "XxXxXxXxXxXxXxXx",
}


def taiko(compassos: int, ppq: int, padrao: str = "marcha",
          beats_por_compasso: int = 4, pitch: int = 36,
          velocity: int = 108, variar: bool = True) -> list[Note]:
    """Escreve a linha de tambor a partir de um padrão em texto.

    `variar` troca o último compasso de cada quatro por uma virada. Sem isso,
    dezesseis compassos do mesmo padrão soam como um loop — e loop é
    exatamente a impressão que uma trilha não pode dar.
    """
    grade = PADROES_TAIKO.get(padrao, PADROES_TAIKO["marcha"])
    passo = max(1, ppq // 4)
    notas: list[Note] = []
    for c in range(compassos):
        linha = grade
        if variar and c % 4 == 3:
            linha = PADROES_TAIKO["adensa"] if padrao != "final" else PADROES_TAIKO["final"]
        base = c * ppq * beats_por_compasso
        for k, ch in enumerate(linha[:beats_por_compasso * 4]):
            if ch == ".":
                continue
            v = velocity if ch == "X" else max(1, int(velocity * 0.62))
            notas.append(Note(base + k * passo, passo, pitch, v))
    return notas


# --------------------------------------------------------------------------
# Preparação de seção
# --------------------------------------------------------------------------
def preparacao(compasso_alvo: int, ppq: int, beats_por_compasso: int = 4,
               compassos_riser: float = 2.0, impacto: bool = True,
               swell: bool = True, pitch_riser: int = 48,
               pitch_impacto: int = 36) -> dict[str, list[Note]]:
    """Os eventos que preparam a entrada de uma seção.

    Devolve um dicionário com uma lista por track: `riser`, `swell`, `impacto`.
    O riser e o swell TERMINAM no downbeat; o impacto começa nele. É essa
    coincidência exata que faz a mudança de seção soar decidida em vez de
    apenas acontecer.
    """
    compasso = ppq * beats_por_compasso
    alvo = compasso_alvo * compasso
    dur = int(compassos_riser * compasso)
    saida: dict[str, list[Note]] = {}
    if dur > 0:
        saida["riser"] = [Note(max(0, alvo - dur), dur, pitch_riser, 104)]
        if swell:
            meio = int(dur * 0.7)
            saida["swell"] = [Note(max(0, alvo - meio), meio, 72, 96)]
    if impacto:
        saida["impacto"] = [Note(alvo, compasso, pitch_impacto, 120)]
    return saida


# --------------------------------------------------------------------------
# Realismo de execução
# --------------------------------------------------------------------------
def humanizar_naipe(notas: Sequence[Note], ppq: int = 480,
                    espalhar_ms: float = 22.0, bpm: float = 100.0,
                    velocity_desvio: int = 9, semente: int = 11,
                    ataque_do_grave: float = 1.4) -> list[Note]:
    """Desalinha um bloco como um naipe real se desalinha.

    Dois detalhes que uma humanização genérica erra:

    * As notas de um MESMO acorde precisam receber desvios DIFERENTES. Mover o
      acorde inteiro 12 ms para a frente não muda nada perceptível; espalhar as
      vozes dentro dele é o que produz a "grossura" de um naipe.
    * O grave chega DEPOIS. Um contrabaixo leva mais tempo para a corda falar
      que um violino — é física, não estilo — e respeitar isso é metade da
      diferença entre orquestra e teclado.
    """
    rng = np.random.default_rng(semente)
    ticks_por_ms = ppq * bpm / 60000.0
    saida: list[Note] = []
    for n in notas:
        desvio = rng.normal(0.0, espalhar_ms * 0.4)
        if n.pitch < 55:
            desvio += (55 - n.pitch) / 27.0 * espalhar_ms * ataque_do_grave * 0.5
        desloc = int(round(desvio * ticks_por_ms))
        v = int(np.clip(n.velocity + rng.normal(0, velocity_desvio), 1, 127))
        saida.append(Note(max(0, n.start + desloc), n.dur, n.pitch, v))
    return saida


def respiro(notas: Sequence[Note], ppq: int = 480,
            frase_ticks: int | None = None, folga_ticks: int = 90) -> list[Note]:
    """Encurta a última nota de cada frase para caber uma respirada.

    Frase de sopro ou de voz sem respiro não é frase: é uma linha contínua que
    o ouvido não consegue segmentar, e por isso soa como sequenciador.
    """
    frase = frase_ticks or ppq * 8
    saida: list[Note] = []
    for n in notas:
        fim_frase = ((n.start // frase) + 1) * frase
        if n.start + n.dur >= fim_frase - 1:
            saida.append(Note(n.start, max(ppq // 8, n.dur - folga_ticks),
                              n.pitch, n.velocity))
        else:
            saida.append(n)
    return saida


def escrever(linha: Sequence[tuple], ppq: int, beats_por_compasso: int = 4,
             velocity: int = 96, transpor: int = 0, oitava: int = 0,
             vezes: int = 1, compassos: int | None = None,
             nome_para_midi=None) -> list[Note]:
    """Escreve uma melodia a partir de tuplas (nota, compasso, tempo, duração).

    É o formato que os exemplos já usam à mão; centralizar aqui evita que cada
    peça reescreva a mesma conversão — e, mais importante, dá um lugar único
    para consertar quando o cálculo de compasso mudar.
    """
    from . import theory as T
    nome_para_midi = nome_para_midi or T.name_to_midi
    if compassos is None:
        compassos = int(max(c for _, c, _, _ in linha)) + 1 if linha else 1
    passo = ppq * beats_por_compasso * compassos
    notas: list[Note] = []
    for r in range(vezes):
        for nome, comp, beat, dur in linha:
            p = nome_para_midi(nome) + transpor + 12 * oitava
            inicio = r * passo + int(comp) * ppq * beats_por_compasso \
                + int(round(beat * ppq))
            notas.append(Note(inicio, max(1, int(round(dur * ppq))), int(p),
                              velocity))
    return notas
