"""
chipforge.core.mml
==================
Partitura em texto. Tres notacoes, todas pensadas para serem escritas rapido
e lidas sem ambiguidade — por voce, ou por um assistente compondo pelo MCP.

1. **MML** (Music Macro Language) — a notacao que os compositores de PC-88,
   MSX e muito tracker de chiptune usavam. Uma frase inteira cabe numa linha:

       o5 l8 e g a > c < b4 a | g4. e8 d2 |

   ============  =========================================================
   c d e f g a b  notas. `+` ou `#` sustenido, `-` bemol:  c+  b-  f#
   4 8 16 . ^     duracao (1=semibreve, 4=seminima, 8=colcheia, 12=tercina
                  de colcheia...). `.` pontua, `^` liga: c4^16   g2.
   r              pausa (aceita duracao: r8, r2.)
   o4  > <        oitava (o4: c = Do4 = MIDI 60). `>` sobe, `<` desce
   l8             duracao padrao das notas sem numero
   v100 v+8 v-8   velocity (0-127), absoluta ou relativa
   q7             gate em oitavos: q8 = legato, q4 = staccato (padrao q7)
   [c e g]2       acorde (as notas soam juntas; oitava muda so dentro)
   c & c          `&` liga com a proxima nota de mesma altura
   !c  ?c         `!` acento (+20 de velocity so nesta nota); `?` nota
                  fantasma (metade da velocity) — o balanco mora aqui
   ( ... )3       repete o trecho 3 vezes (pode aninhar)
   |              barra de compasso: confere se o compasso fechou
   ; texto        comentario ate o fim da linha
   ============  =========================================================

2. **Cifras** — `Am7 | F | C G | Dm/F E7` (um compasso por `|`, acordes do
   mesmo compasso dividem o tempo igualmente; `%` repete o compasso anterior).

3. **Grade de bateria** — uma linha por peca, um caractere por passo:

       k: x...x...x..x.x..
       s: ....X.......X..o
       h: x.x.x.x.x.x.x.xx

   `X` acento, `x` normal, `o` fantasma, `.` ou `-` silencio. 16 passos por
   compasso (semicolcheias) a menos que se peca outra resolucao.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .project import Note

_PC = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}


class ErroMML(ValueError):
    """Erro de sintaxe com posicao — para quem escreveu achar rapido."""

    def __init__(self, msg: str, texto: str, pos: int):
        linha = texto.count("\n", 0, pos) + 1
        trecho = texto[max(0, pos - 12):pos + 12].replace("\n", " ")
        super().__init__(f"{msg} (linha {linha}, perto de '...{trecho}...')")


@dataclass
class ResultadoMML:
    notas: list[Note]
    fim: int                    # tick em que a escrita terminou (inclui pausas)
    avisos: list[str] = field(default_factory=list)

    @property
    def extensao(self) -> tuple[int, int] | None:
        if not self.notas:
            return None
        ps = [n.pitch for n in self.notas]
        return min(ps), max(ps)


def _ticks(den: int, pontos: int, ppq: int) -> int:
    if den <= 0:
        raise ValueError("duracao precisa ser positiva")
    base = ppq * 4 / den
    total, add = base, base
    for _ in range(pontos):
        add /= 2
        total += add
    return int(round(total))


def _expandir_repeticoes(txt: str) -> str:
    """( ... )n -> o trecho n vezes, de dentro para fora."""
    padrao = re.compile(r"\(([^()]*)\)(\d*)")
    for _ in range(50):
        novo = padrao.sub(lambda m: " ".join([m.group(1)] * int(m.group(2) or 2)),
                          txt)
        if novo == txt:
            break
        txt = novo
    if "(" in txt or ")" in txt:
        raise ErroMML("parenteses de repeticao desbalanceados", txt,
                      max(txt.find("("), txt.find(")")))
    return txt


def parse(texto: str, ppq: int = 480, beats_por_compasso: int = 4,
          oitava: int = 4, duracao: int = 8, velocity: int = 96,
          gate: int = 7, inicio: int = 0) -> ResultadoMML:
    """Converte MML em notas. Ticks relativos a `inicio`."""
    limpo = "\n".join(l.split(";", 1)[0] for l in texto.splitlines())
    s = _expandir_repeticoes(limpo).lower()
    i, n = 0, len(s)
    t = inicio
    notas: list[Note] = []
    avisos: list[str] = []
    compasso_t = ppq * beats_por_compasso
    num_compasso = 1
    inicio_compasso = inicio
    ligar = False
    acento = 0                       # +20 (!) ou -metade (?) na proxima nota
    est = {"o": oitava, "l": (duracao, 0), "v": velocity, "q": gate}

    def numero(j):
        k = j
        while k < n and s[k].isdigit():
            k += 1
        return (int(s[j:k]), k) if k > j else (None, j)

    def duracao_em(j):
        den, j = numero(j)
        pontos = 0
        while j < n and s[j] == ".":
            pontos += 1
            j += 1
        if den is None:
            den, p0 = est["l"]
            pontos = pontos or p0
        tk = _ticks(den, pontos, ppq)
        while j < n and s[j] == "^":           # ligaduras: c4^16^32
            d2, j = numero(j + 1)
            p2 = 0
            while j < n and s[j] == ".":
                p2 += 1
                j += 1
            tk += _ticks(d2 if d2 else est["l"][0], p2, ppq)
        return tk, j

    def altura(j, oit):
        pc = _PC[s[j]]
        j += 1
        while j < n and s[j] in "+#-":
            pc += -1 if s[j] == "-" else 1
            j += 1
        return (oit + 1) * 12 + pc, j

    def vel_agora() -> int:
        nonlocal acento
        v = est["v"]
        if acento > 0:
            v = min(127, v + 20)
        elif acento < 0:
            v = max(1, v // 2)
        acento = 0
        return v

    def soltar(pitch, dur):
        nonlocal ligar
        som = dur if est["q"] >= 8 else max(1, dur * est["q"] // 8)
        if ligar and notas and notas[-1].pitch == pitch and notas[-1].start < t:
            ult = notas[-1]
            ult.dur = (t - ult.start) + som
        else:
            if not 0 <= pitch <= 127:
                raise ErroMML(f"nota fora do MIDI ({pitch})", texto, min(i, len(texto)))
            notas.append(Note(t, som, pitch, vel_agora()))
        ligar = False

    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
        elif c == "|":
            dentro = t - inicio_compasso
            if dentro != compasso_t:
                avisos.append(f"compasso {num_compasso} tem {dentro / ppq:g} "
                              f"tempos (esperado {beats_por_compasso})")
            inicio_compasso = t
            num_compasso += 1
            i += 1
        elif c in _PC:
            p, i = altura(i, est["o"])
            d, i = duracao_em(i)
            soltar(p, d)
            t += d
        elif c == "r":
            d, i = duracao_em(i + 1)
            ligar = False
            t += d
        elif c == "[":
            fim = s.find("]", i)
            if fim < 0:
                raise ErroMML("acorde sem ']'", texto, i)
            corpo, oit = s[i + 1:fim], est["o"]
            alturas, j = [], 0
            while j < len(corpo):
                ch = corpo[j]
                if ch in _PC:
                    k = j + 1
                    pc = _PC[ch]
                    while k < len(corpo) and corpo[k] in "+#-":
                        pc += -1 if corpo[k] == "-" else 1
                        k += 1
                    alturas.append((oit + 1) * 12 + pc)
                    j = k
                elif ch == ">":
                    oit += 1; j += 1
                elif ch == "<":
                    oit -= 1; j += 1
                elif ch == "o":
                    v, k = 0, j + 1
                    while k < len(corpo) and corpo[k].isdigit():
                        k += 1
                    oit = int(corpo[j + 1:k]); j = k
                elif ch.isspace():
                    j += 1
                else:
                    raise ErroMML(f"'{ch}' nao vale dentro de acorde", texto, i + 1 + j)
            d, i = duracao_em(fim + 1)
            som = d if est["q"] >= 8 else max(1, d * est["q"] // 8)
            v_ac = vel_agora()
            for p in alturas:
                notas.append(Note(t, som, p, v_ac))
            ligar = False
            t += d
        elif c == "&":
            ligar = True
            i += 1
        elif c == "!":
            acento = 1
            i += 1
        elif c == "?":
            acento = -1
            i += 1
        elif c == "o":
            v, i = numero(i + 1)
            if v is None:
                raise ErroMML("'o' precisa de numero (o4)", texto, i)
            est["o"] = v
        elif c == ">":
            est["o"] += 1; i += 1
        elif c == "<":
            est["o"] -= 1; i += 1
        elif c == "l":
            v, j = numero(i + 1)
            if v is None:
                raise ErroMML("'l' precisa de numero (l8)", texto, i)
            pontos = 0
            while j < n and s[j] == ".":
                pontos += 1; j += 1
            est["l"] = (v, pontos)
            i = j
        elif c == "v":
            sinal = 0
            if i + 1 < n and s[i + 1] in "+-":
                sinal = 1 if s[i + 1] == "+" else -1
                i += 1
            v, i = numero(i + 1)
            if v is None:
                raise ErroMML("'v' precisa de numero (v100)", texto, i)
            est["v"] = max(1, min(127, est["v"] + sinal * v if sinal else v))
        elif c == "q":
            v, i = numero(i + 1)
            est["q"] = max(1, min(8, v or 7))
        else:
            raise ErroMML(f"simbolo desconhecido '{c}'", texto, i)

    return ResultadoMML(notas, t, avisos)


def para_mml(notas: list[Note], ppq: int = 480, beats_por_compasso: int = 4) -> str:
    """Caminho de volta: notas -> MML aproximado (monofonico por cima).
    Serve para mostrar ao compositor o que ja esta escrito num pattern."""
    if not notas:
        return ""
    from fractions import Fraction
    dens = {Fraction(ppq * 4, d): f"{d}" for d in (1, 2, 4, 8, 16, 32)}
    dens.update({Fraction(ppq * 6, d): f"{d}." for d in (1, 2, 4, 8, 16)})
    dens.update({Fraction(ppq * 4, d): f"{d}" for d in (3, 6, 12, 24)})

    def dur_txt(tk):
        partes, resto = [], tk
        for v in sorted(dens, reverse=True):
            while resto >= v and v > 0:
                partes.append(dens[v]); resto -= v
        return "^".join(partes) if partes else "64"

    nomes = ["c", "c+", "d", "d+", "e", "f", "f+", "g", "g+", "a", "a+", "b"]
    por_inicio: dict[int, list[Note]] = {}
    for nt in notas:
        por_inicio.setdefault(nt.start, []).append(nt)
    out, t, oit = [], 0, None
    barra = ppq * beats_por_compasso
    for ini in sorted(por_inicio):
        if ini > t:
            out.append("r" + dur_txt(ini - t))
        grupo = por_inicio[ini]
        soa = max(nt.dur for nt in grupo)
        prox = min([k for k in por_inicio if k > ini], default=ini + round(soa * 8 / 7))
        ate_prox = prox - ini
        # o gate (q7) encurta a nota que soa; para a escrita, a nota vai ate a
        # proxima se o buraco for pequeno — senao vira pausa de verdade
        if soa <= ate_prox and ate_prox - soa <= max(ate_prox // 4, 30):
            passo = ate_prox
        else:
            passo = max(1, min(ate_prox, soa))
            cheio = round(soa * 8 / 7)          # desfaz o gate padrao q7
            limpos = [int(v) for v in dens]
            if cheio <= ate_prox and any(abs(cheio - v) <= 2 for v in limpos):
                passo = min(limpos, key=lambda v: abs(cheio - v))
        o = grupo[0].pitch // 12 - 1
        if o != oit:
            out.append(f"o{o}"); oit = o
        if len(grupo) == 1:
            out.append(nomes[grupo[0].pitch % 12] + dur_txt(passo))
        else:
            corpo = []
            base = o
            for nt in sorted(grupo, key=lambda x: x.pitch):
                oo = nt.pitch // 12 - 1
                corpo.append((">" * (oo - base)) + nomes[nt.pitch % 12])
                base = oo
            out.append("[" + " ".join(corpo) + "]" + dur_txt(passo))
        t = ini + passo
        if t % barra == 0:
            out.append("|")
    return " ".join(out)


# ==========================================================================
# Cifras
# ==========================================================================
_QUALIDADES = {
    "": (0, 4, 7), "maj": (0, 4, 7), "m": (0, 3, 7), "min": (0, 3, 7),
    "dim": (0, 3, 6), "aug": (0, 4, 8), "+": (0, 4, 8),
    "sus2": (0, 2, 7), "sus4": (0, 5, 7), "sus": (0, 5, 7), "5": (0, 7),
    "6": (0, 4, 7, 9), "m6": (0, 3, 7, 9),
    "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11), "m7": (0, 3, 7, 10),
    "mmaj7": (0, 3, 7, 11), "m7b5": (0, 3, 6, 10), "dim7": (0, 3, 6, 9),
    "7sus4": (0, 5, 7, 10), "add9": (0, 4, 7, 14), "madd9": (0, 3, 7, 14),
    "9": (0, 4, 7, 10, 14), "maj9": (0, 4, 7, 11, 14), "m9": (0, 3, 7, 10, 14),
    "11": (0, 4, 7, 10, 14, 17), "m11": (0, 3, 7, 10, 14, 17),
    "7b9": (0, 4, 7, 10, 13), "7#9": (0, 4, 7, 10, 15),
}
_RE_CIFRA = re.compile(r"^([A-G])([#b]?)([^/]*)(?:/([A-G][#b]?))?$")


@dataclass
class Cifra:
    nome: str
    raiz: int            # classe de altura 0..11
    intervalos: tuple
    baixo: int           # classe de altura do baixo (inversao / slash)

    def alturas(self, oitava: int = 4) -> list[int]:
        base = (oitava + 1) * 12 + self.raiz
        return [base + i for i in self.intervalos]

    @property
    def classes(self) -> list[int]:
        return sorted({(self.raiz + i) % 12 for i in self.intervalos})


def _pc_nome(letra: str, acid: str) -> int:
    return (_PC[letra.lower()] + (1 if acid == "#" else -1 if acid == "b" else 0)) % 12


def cifra(nome: str) -> Cifra:
    m = _RE_CIFRA.match(nome.strip())
    if not m:
        raise ValueError(f"cifra invalida: '{nome}'")
    letra, acid, qual, baixo = m.groups()
    q = qual.replace("M7", "maj7").replace("°", "dim").replace("ø", "m7b5")
    q = q.replace("min7", "m7")
    if q not in _QUALIDADES:
        raise ValueError(f"cifra '{nome}': nao conheco '{qual}'. Uso: "
                         + ", ".join(k or "(maior)" for k in _QUALIDADES))
    raiz = _pc_nome(letra, acid)
    b = _pc_nome(baixo[0], baixo[1:]) if baixo else raiz
    return Cifra(nome.strip(), raiz, _QUALIDADES[q], b)


def parse_cifras(texto: str, ppq: int = 480,
                 beats_por_compasso: int = 4) -> list[tuple[int, int, Cifra]]:
    """'Am | F G | %' -> [(inicio, duracao, Cifra), ...] em ticks."""
    barra = ppq * beats_por_compasso
    out: list[tuple[int, int, Cifra]] = []
    anterior: list[str] = []
    compassos = [c.strip() for c in texto.strip().strip("|").split("|")]
    for k, comp in enumerate(compassos):
        nomes = comp.split()
        if nomes == ["%"]:
            nomes = anterior
        if not nomes:
            continue
        anterior = nomes
        passo = barra // len(nomes)
        for j, nm in enumerate(nomes):
            if nm in ("-", "."):            # segura o acorde anterior
                if out:
                    s0, d0, c0 = out[-1]
                    out[-1] = (s0, d0 + passo, c0)
                continue
            out.append((k * barra + j * passo, passo, cifra(nm)))
    return out


# ==========================================================================
# Grade de bateria
# ==========================================================================
BATERIA_GM = {
    "k": 36, "kick": 36, "bumbo": 36,
    "s": 38, "snare": 38, "caixa": 38,
    "p": 39, "clap": 39, "palma": 39,
    "rs": 37, "aro": 37,
    "h": 42, "hat": 42, "chimbal": 42,
    "ph": 44, "pedal": 44,
    "oh": 46, "H": 46, "aberto": 46,
    "t1": 50, "t2": 47, "t3": 45, "t4": 43, "t5": 41,
    "c": 49, "crash": 49, "c2": 57,
    "r": 51, "ride": 51, "rb": 53, "sino": 53,
    "sp": 55, "splash": 55, "ch": 52, "china": 52,
    "tb": 54, "pandeiro": 54, "cb": 56, "cowbell": 56,
}
_VEL = {"X": 124, "x": 100, "o": 58, "g": 58}


def parse_bateria(texto: str, ppq: int = 480, beats_por_compasso: int = 4,
                  passos_por_tempo: int = 4, velocity_escala: float = 1.0
                  ) -> tuple[list[Note], int]:
    """Linhas 'peca: x...x...' -> notas GM. Devolve (notas, ticks totais)."""
    passo = ppq // passos_por_tempo
    notas: list[Note] = []
    fim = 0
    for linha in texto.strip().splitlines():
        linha = linha.split(";", 1)[0].strip()
        if not linha:
            continue
        if ":" not in linha:
            raise ValueError(f"linha de bateria sem 'peca:': '{linha}'")
        peca, grade = linha.split(":", 1)
        peca = peca.strip()
        nota = BATERIA_GM.get(peca, BATERIA_GM.get(peca.lower()))
        if nota is None:
            if peca.isdigit():
                nota = int(peca)
            else:
                raise ValueError(f"peca de bateria desconhecida '{peca}'. Use: "
                                 + ", ".join(sorted(set(BATERIA_GM))))
        grade = grade.replace("|", "").replace(" ", "")
        for k, ch in enumerate(grade):
            if ch in _VEL:
                v = int(min(127, _VEL[ch] * velocity_escala))
                notas.append(Note(k * passo, max(30, passo - 10), nota, v))
        fim = max(fim, len(grade) * passo)
    return sorted(notas, key=lambda n: n.start), fim
