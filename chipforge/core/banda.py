"""
chipforge.core.banda
====================
Escrita de banda: riff de acorde de potência e bateria de kit.

Por que isto existe: `orquestra.py` escreve para naipe — bloco, ostinato,
pedal, divisi. Nada disso é rock. Uma guitarra de rock não toca arpejo no
grave: ela toca o MESMO acorde muitas vezes, e o que carrega a música é o
ritmo com que ela repete. Escrever guitarra com `ostinato()` produz
exatamente o que produziu na primeira tentativa — algo que tem as notas
certas e não soa como uma banda.

Duas ideias, e as duas são de ritmo, não de altura:

**Acorde de potência.** Fundamental e quinta (mais a oitava quando se quer
peso), no registro grave da guitarra. Sem terça — é ela que faz o acorde
soar maior ou menor, e é exatamente por não ter terça que um riff distorcido
não vira lama: dois intervalos puros distorcidos geram harmônicos que caem
nos mesmos lugares, e três não.

**Grade de dezesseis avos.** Riff e bateria são escritos em texto, um
caractere por semicolcheia. É como um baterista lê e é como um tracker
sempre escreveu. `X` acentua, `x` toca normal, `-` deixa soar (liga na
anterior) e `.` cala.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .project import Note

# --------------------------------------------------------------------------
# Bateria — alturas do padrão GM, que é o que um kit de SoundFont espera
# --------------------------------------------------------------------------
BUMBO = 36
CAIXA = 38
CAIXA_ARO = 37
CHIMBAL = 42
CHIMBAL_PEDAL = 44
CHIMBAL_ABERTO = 46
PRATO = 49
PRATO_2 = 57
CONDUCAO = 51
CAMPANA = 53
TOM_AGUDO = 48
TOM_MEDIO = 45
TOM_GRAVE = 41

# Cada string é um compasso de 4/4 em semicolcheias.
#   X = acento · x = normal · o = variação (chimbal aberto, aro) · . = silêncio
ESTILOS_BATERIA: dict[str, dict[str, str]] = {
    "balada": {
        "bumbo": "X.......x.......",
        "caixa": "....X.......X...",
        "conducao": "x..x..x..x..x..x",
    },
    "meio": {                       # half time: pesado, espaçado
        "bumbo": "X.....x.X.......",
        "caixa": "........X.......",
        "chimbal": "x.x.x.x.x.x.x.x.",
    },
    "rock": {
        "bumbo": "X..x..X...x.....",
        "caixa": "....X.......X...",
        "chimbal": "x.x.x.x.x.x.x.x.",
    },
    "rock_aberto": {
        "bumbo": "X..x..X...x.....",
        "caixa": "....X.......X...",
        "chimbal": "x.x.x.x.x.x.x.o.",
    },
    "conducao": {                   # o mesmo groove, mas no prato de condução
        "bumbo": "X..x..X...x.....",
        "caixa": "....X.......X...",
        "conducao": "x.x.x.x.x.x.x.x.",
    },
    "galope": {
        "bumbo": "X.xxX.xxX.xxX.xx",
        "caixa": "....X.......X...",
        "chimbal": "x.x.x.x.x.x.x.x.",
    },
    "duplo": {                      # double time: o motor da seção rápida
        "bumbo": "X.x.X.x.X.x.X.x.",
        "caixa": "....X.......X...",
        "chimbal": "xxxxxxxxxxxxxxxx",
    },
    "quebrado": {                   # breakdown: espaço e peso
        "bumbo": "X...X...X.X.....",
        "caixa": "........X.......",
        "chimbal": "x.......x.......",
    },
    "marcha": {
        "bumbo": "X...X...X...X...",
        "caixa": "....X.......X...",
        "chimbal": "x.x.x.x.x.x.x.x.",
    },
    "blast": {                      # o mais denso: só para o clímax
        "bumbo": "xxxxxxxxxxxxxxxx",
        "caixa": "..x...x...x...x.",
        "chimbal": "x.x.x.x.x.x.x.x.",
    },
}

# Viradas. A virada é o que impede dezesseis compassos iguais de soarem como
# um loop — e é onde o baterista avisa que a seção vai mudar.
VIRADAS: dict[str, dict[str, str]] = {
    "curta": {
        "caixa": "........xxxx....",
        "tom_medio": "............xx..",
        "tom_grave": "..............xx",
    },
    "longa": {
        "caixa": "xxxx............",
        "tom_agudo": "....xx..........",
        "tom_medio": "......xx........",
        "tom_grave": "........xxxxxxxx",
    },
    "caixa": {
        "caixa": "xxxxxxxxxxxxxxxx",
    },
    "parada": {                     # tudo para; só o último tempo volta
        "caixa": "............X...",
        "bumbo": "..............X.",
    },
}

_VOZES = {
    "bumbo": (BUMBO, BUMBO), "caixa": (CAIXA, CAIXA_ARO),
    "chimbal": (CHIMBAL, CHIMBAL_ABERTO), "conducao": (CONDUCAO, CAMPANA),
    "tom_agudo": (TOM_AGUDO, TOM_AGUDO), "tom_medio": (TOM_MEDIO, TOM_MEDIO),
    "tom_grave": (TOM_GRAVE, TOM_GRAVE), "prato": (PRATO, PRATO_2),
}

_FORCA = {"bumbo": 1.0, "caixa": 0.98, "chimbal": 0.55, "conducao": 0.58,
          "tom_agudo": 0.85, "tom_medio": 0.85, "tom_grave": 0.88, "prato": 0.9}


def bateria(compassos: int, ppq: int, estilo: str = "rock",
            velocity: int = 108, virada_a_cada: int = 4,
            virada: str = "curta", prato_no_inicio: bool = True,
            fantasma: bool = True, semente: int = 5) -> list[Note]:
    """Escreve a bateria a partir das grades de texto.

    Três coisas que separam isto de um metrônomo com timbre:

    * **Virada periódica.** Sem ela, dezesseis compassos são o mesmo compasso
      dezesseis vezes, e o ouvido registra "loop" em menos de trinta segundos.
    * **Prato na primeira.** É o que marca onde a frase começa. Sem prato, uma
      mudança de seção passa despercebida mesmo com tudo mais mudando.
    * **Notas fantasma.** A caixa não toca só nos tempos 2 e 4: entre eles ela
      dá toques quase inaudíveis que é o que dá liga ao groove. Tirá-las deixa
      a bateria correta e morta.
    """
    grade = ESTILOS_BATERIA.get(estilo, ESTILOS_BATERIA["rock"])
    rng = np.random.default_rng(semente)
    passo = max(1, ppq // 4)
    bar = ppq * 4
    out: list[Note] = []

    for c in range(compassos):
        base = c * bar
        ultima = (c == compassos - 1)
        vira = virada_a_cada and (c % virada_a_cada == virada_a_cada - 1)
        linhas = dict(grade)
        if vira and not ultima:
            for voz, linha in VIRADAS.get(virada, {}).items():
                # a virada substitui a caixa e acrescenta os tons; o bumbo e o
                # chimbal continuam, senão o compasso inteiro desaparece
                linhas[voz] = linha
            linhas.pop("chimbal", None)
            linhas.pop("conducao", None)

        for voz, linha in linhas.items():
            alt_normal, alt_var = _VOZES.get(voz, (CAIXA, CAIXA))
            forca = _FORCA.get(voz, 0.8)
            for k, ch in enumerate(linha[:16]):
                if ch == ".":
                    continue
                alt = alt_var if ch == "o" else alt_normal
                v = velocity * forca * (1.0 if ch in "Xo" else 0.72)
                v = int(np.clip(v + rng.normal(0, 4), 1, 127))
                out.append(Note(base + k * passo, passo, alt, v))

        if fantasma and "caixa" in linhas and not vira:
            for k in (2, 6, 10, 14):
                if linhas["caixa"][k] == "." and rng.random() < 0.45:
                    out.append(Note(base + k * passo, passo, CAIXA,
                                    int(np.clip(velocity * 0.22, 1, 127))))

        if prato_no_inicio and c % 8 == 0:
            out.append(Note(base, passo * 4, PRATO,
                            int(np.clip(velocity * 0.92, 1, 127))))
    return out


# --------------------------------------------------------------------------
# Riff
# --------------------------------------------------------------------------
RIFFS: dict[str, str] = {
    "quartos": "X...x...x...x...",
    "colcheias": "X.x.x.x.x.x.x.x.",
    "chug8": "X.x.X.x.X.x.X.x.",
    "chug16": "XxxxXxxxXxxxXxxx",
    "galope": "X.xxX.xxX.xxX.xx",
    "galope_duplo": "Xxx.Xxx.Xxx.Xxx.",
    "sincopado": "X..x..X..x..X...",
    "empurrado": "X..x.X..x.X..x..",
    "parada": "X---------------",
    "meio": "X-------X-------",
    "quebrado": "X..X..X...X.....",
    "marulho": "X.x.X..xX.x.X.x.",
    "tresilho": "X..X..X..X..X...",
}


def potencia(raiz: int, oitava_dobrada: bool = True,
             baixo: int = 40, alto: int = 62) -> list[int]:
    """Acorde de potência: fundamental e quinta, opcionalmente mais a oitava.

    Sem terça, de propósito. A terça é o que faz o acorde soar maior ou menor,
    e é o que vira lama quando distorcido: distorção gera soma e diferença de
    cada par de frequências, e com três notas os produtos caem em lugares que
    não pertencem ao acorde. Com fundamental e quinta, quase tudo o que a
    distorção gera reforça as duas notas que já estão lá. É por isso que rock
    pesado é escrito assim, e não por preguiça.
    """
    r = int(raiz)
    while r < baixo:
        r += 12
    while r > alto:
        r -= 12
    notas = [r, r + 7]
    if oitava_dobrada and r + 12 <= alto + 12:
        notas.append(r + 12)
    return notas


def riff(acordes: Sequence[Sequence[int]], ppq: int, padrao: str = "chug8",
         compassos_por_acorde: int = 1, velocity: int = 100,
         oitava_dobrada: bool = True, baixo: int = 40, alto: int = 62,
         curta: float = 0.55, variar_ultimo: bool = True,
         semente: int = 3) -> list[Note]:
    """Riff de acorde de potência a partir de uma grade de semicolcheias.

    `curta` é a fração da subdivisão que a nota realmente dura. É o parâmetro
    mais importante aqui e não parece: em `0.55`, a corda é parada antes da
    próxima palhetada, e é esse silêncio entre as batidas que produz o "chug".
    Em `1.0` as notas se emendam, a distorção nunca respira e o riff vira uma
    parede contínua — o erro mais comum de guitarra programada.
    """
    grade = RIFFS.get(padrao, RIFFS["chug8"])
    rng = np.random.default_rng(semente)
    passo = max(1, ppq // 4)
    bar = ppq * 4
    out: list[Note] = []

    for i, acorde in enumerate(acordes):
        notas_ac = potencia(acorde[0], oitava_dobrada, baixo, alto)
        for c in range(compassos_por_acorde):
            base = (i * compassos_por_acorde + c) * bar
            linha = grade
            if variar_ultimo and i == len(acordes) - 1 and c == compassos_por_acorde - 1:
                linha = RIFFS.get("quebrado", grade)

            k = 0
            while k < 16:
                ch = linha[k]
                if ch == ".":
                    k += 1
                    continue
                # conta quantos '-' seguem: é a duração da nota ligada
                dur_passos = 1
                while k + dur_passos < 16 and linha[k + dur_passos] == "-":
                    dur_passos += 1
                if ch == "-":
                    k += 1
                    continue
                ligada = dur_passos > 1
                dur = int(passo * dur_passos * (1.0 if ligada else curta))
                v = velocity * (1.0 if ch == "X" else 0.78)
                for p in notas_ac:
                    vv = int(np.clip(v + rng.normal(0, 3.5), 1, 127))
                    out.append(Note(base + k * passo, max(passo // 4, dur), p, vv))
                k += dur_passos
    return out


def linha_de_baixo(acordes: Sequence[Sequence[int]], ppq: int,
                   padrao: str = "colcheias", compassos_por_acorde: int = 1,
                   velocity: int = 104, baixo: int = 28, alto: int = 48,
                   oitavar: bool = False, semente: int = 7) -> list[Note]:
    """Baixo elétrico seguindo a mesma grade do riff.

    Tocar a mesma figura rítmica da guitarra é o que faz baixo e guitarra
    soarem como uma coisa só em vez de duas tocando junto. O baixo fica na
    fundamental; quem dá a quinta é a guitarra.
    """
    grade = RIFFS.get(padrao, RIFFS["colcheias"])
    rng = np.random.default_rng(semente)
    passo = max(1, ppq // 4)
    bar = ppq * 4
    out: list[Note] = []
    for i, acorde in enumerate(acordes):
        r = int(acorde[0])
        while r < baixo:
            r += 12
        while r > alto:
            r -= 12
        for c in range(compassos_por_acorde):
            base = (i * compassos_por_acorde + c) * bar
            for k, ch in enumerate(grade[:16]):
                if ch in ".-":
                    continue
                p = r + (12 if oitavar and k % 8 == 4 else 0)
                v = int(np.clip(velocity * (1.0 if ch == "X" else 0.8)
                                + rng.normal(0, 3), 1, 127))
                out.append(Note(base + k * passo, int(passo * 0.9), p, v))
    return out


def solo_pentatonico(compassos: int, ppq: int, tonica: int, modo: str = "menor",
                     oitava: int = 5, velocity: int = 104,
                     densidade: float = 0.7, semente: int = 11) -> list[Note]:
    """Uma linha de solo na pentatônica, com bends implícitos por repetição.

    Não substitui um solo escrito à mão — serve para preencher uma seção com
    algo idiomático em vez de uma escala corrida. A pentatônica é a escolha
    porque nela não existe nota errada sobre a harmonia de rock: faltam
    justamente os dois graus que brigam.
    """
    graus = (0, 3, 5, 7, 10) if modo == "menor" else (0, 2, 4, 7, 9)
    rng = np.random.default_rng(semente)
    base_midi = tonica % 12 + 12 * oitava
    escala = [base_midi + g + 12 * o for o in (-1, 0, 1) for g in graus]
    escala = sorted(p for p in escala if 55 <= p <= 88)
    passo = ppq // 4
    out: list[Note] = []
    idx = len(escala) // 2
    for c in range(compassos):
        base = c * ppq * 4
        k = 0
        while k < 16:
            if rng.random() > densidade:
                k += int(rng.integers(1, 3))
                continue
            # caminha por graus vizinhos, com salto ocasional: é o desenho de
            # uma frase, não de um sorteio
            idx = int(np.clip(idx + rng.choice([-2, -1, -1, 1, 1, 2, 3]),
                              0, len(escala) - 1))
            dur_passos = int(rng.choice([1, 1, 2, 2, 4]))
            out.append(Note(base + k * passo, int(passo * dur_passos * 0.9),
                            escala[idx],
                            int(np.clip(velocity + rng.normal(0, 6), 1, 127))))
            k += dur_passos
    return out
