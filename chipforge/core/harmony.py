"""
chipforge.core.harmony
======================
O "medidor de harmonia": olha um trecho de musica e responde, em portugues,
se aquilo esta soando bem — e QUANDO e POR QUE nao esta.

Nao e um juiz de gosto. E um conjunto de metricas classicas de acustica e
contraponto, cada uma com peso, somadas num score 0..100:

  1. Consonancia vertical   — os intervalos que soam JUNTOS batem de frente?
  2. Aderencia tonal        — as notas pertencem a tonalidade percebida?
  3. Conducao de vozes      — quintas/oitavas paralelas, saltos absurdos
  4. Limite grave           — tercas empilhadas no subgrave viram lama
  5. Coerencia harmonica    — a sequencia de acordes tem funcao?

Cada problema vira um Issue com posicao em beats, para a UI pintar a regiao.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, NamedTuple, Sequence

from . import theory as T


class NoteEvent(NamedTuple):
    start: float     # em beats
    dur: float       # em beats
    pitch: int       # MIDI
    velocity: int = 100
    track: str = "main"


# Consonancia sensorial por intervalo (classe de intervalo, 0..11).
# Baseado na ordem classica de aspereza (Helmholtz / Plomp-Levelt simplificado).
CONSONANCE = {
    0: 1.00,   # unissono / oitava
    1: 0.05,   # segunda menor  — o choque classico
    2: 0.28,   # segunda maior
    3: 0.86,   # terca menor
    4: 0.90,   # terca maior
    5: 0.80,   # quarta justa
    6: 0.16,   # tritono
    7: 0.97,   # quinta justa
    8: 0.84,   # sexta menor
    9: 0.88,   # sexta maior
    10: 0.38,  # setima menor
    11: 0.12,  # setima maior
}

# Abaixo desta nota, intervalos apertados viram lama (regra do "low interval limit")
MUD_ZONE = T.name_to_midi("C3")      # 48
MUD_MIN_INTERVAL = 7                 # abaixo de C3, so quinta ou mais aberto

# Sobreposicoes muito curtas (rabo de uma nota encostando na proxima) sao
# inaudiveis como dissonancia. Elas ainda entram na media ponderada por duracao,
# mas nao viram alerta na lista — senao a UI enche de falso positivo.
MIN_REPORT_BEATS = 0.06

# Uma dissonancia so e "problema" quando ela SUSTENTA. Nota curta de passagem
# roçando na melodia e vocabulario normal — arpejo de semicolcheia vive disso.
# Abaixo desta duracao (da mais curta das duas notas), o choque vira aviso.
SUSTAINED_CLASH_BEATS = 0.3

# Uma setima ou nona entre uma voz interna e a melodia, com as duas notas
# dentro da tonalidade e separadas por pelo menos uma setima, e COR — o
# Dm(add9) de qualquer trilha de aventura. O que continua sendo erro e o
# aglomerado apertado e a nota cromatica batendo de frente. Abaixo desta
# distancia o choque volta a valer severidade maxima.
SPREAD_SEMITONES = 10

SEVERITY_LABEL = {0: "dica", 1: "atencao", 2: "problema"}


@dataclass
class Issue:
    beat: float
    kind: str
    severity: int          # 0 dica, 1 atencao, 2 problema
    message: str
    pitches: tuple[int, ...] = ()
    tracks: tuple[str, ...] = ()

    @property
    def culpados(self) -> str:
        """Quais tracks estao brigando. Num projeto de 15 tracks, saber QUE
        duas notas batem nao ajuda — o que resolve e saber de quem sao."""
        return " x ".join(dict.fromkeys(self.tracks)) if self.tracks else ""

    def __str__(self) -> str:
        quem = f"  [{self.culpados}]" if self.tracks else ""
        return (f"[{SEVERITY_LABEL[self.severity]}] compasso {self.beat/4+1:.2f}"
                f" — {self.message}{quem}")


@dataclass
class HarmonyReport:
    score: float                      # 0..100
    key: T.KeyGuess | None
    chords: list[tuple[float, T.ChordMatch]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        s = self.score
        if s >= 88:
            return "Harmonico e limpo"
        if s >= 75:
            return "Soa bem, com tensoes controladas"
        if s >= 60:
            return "Funciona, mas tem pontos ariscos"
        if s >= 45:
            return "Tenso — provavelmente nao foi de proposito"
        return "Dissonante / fora da tonalidade"

    @property
    def color(self) -> str:
        """Cor do LED do medidor na UI."""
        if self.score >= 75:
            return "#3ddc84"
        if self.score >= 55:
            return "#ffcc33"
        return "#ff4d5e"

    def summary_pt(self) -> str:
        lines = [f"{self.verdict}  —  {self.score:.0f}/100"]
        if self.key:
            lines.append(f"Tonalidade percebida: {self.key.name} "
                         f"(confianca {self.key.confidence*100:.0f}%)")
        if self.chords:
            seq = " | ".join(c.name for _, c in self.chords[:8])
            lines.append(f"Acordes: {seq}")
        for k, v in self.metrics.items():
            lines.append(f"  {k}: {v*100:.0f}%")
        if self.issues:
            lines.append("Pontos de atencao:")
            for i in self.issues[:10]:
                lines.append("  " + str(i))
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Fatiamento temporal
# --------------------------------------------------------------------------
def _slices(notes: Sequence[NoteEvent]) -> list[tuple[float, float, list[NoteEvent]]]:
    """Corta a linha do tempo nos pontos onde a textura muda."""
    if not notes:
        return []
    bounds = sorted({round(n.start, 6) for n in notes} |
                    {round(n.start + n.dur, 6) for n in notes})
    out = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a < 1e-6:
            continue
        sounding = [n for n in notes if n.start <= a + 1e-6 and n.start + n.dur >= b - 1e-6]
        if sounding:
            out.append((a, b - a, sounding))
    return out


def _pc_weights(notes: Sequence[NoteEvent]) -> list[float]:
    w = [0.0] * 12
    for n in notes:
        w[n.pitch % 12] += n.dur * (n.velocity / 127.0)
    return w


# --------------------------------------------------------------------------
# Metricas
# --------------------------------------------------------------------------
def _vertical_consonance(slices, issues: list[Issue],
                         key: T.KeyGuess | None = None) -> float:
    escala = T.scale_pitch_classes(key.root, key.mode) if key else None
    total_w = 0.0
    total_s = 0.0
    for start, dur, sounding in slices:
        pitches = sorted({n.pitch for n in sounding})
        longest = {}
        dono: dict[int, str] = {}
        for n in sounding:
            if n.dur > longest.get(n.pitch, 0.0):
                longest[n.pitch] = n.dur
                dono[n.pitch] = n.track
        on_beat = abs(start - round(start)) < 0.03
        if len(pitches) < 2:
            continue
        pair_scores = []
        for i in range(len(pitches)):
            for j in range(i + 1, len(pitches)):
                a, b = pitches[i], pitches[j]
                ic = (b - a) % 12
                c = CONSONANCE[ic]
                # intervalos apertados doem mais quando as notas estao perto
                if (b - a) < 12 and ic in (1, 2, 11):
                    c *= 0.7
                # ... e doem menos quando estao a mais de duas oitavas
                if (b - a) >= 24:
                    c = min(1.0, c + 0.25)
                pair_scores.append(c)
                if dur < MIN_REPORT_BEATS:
                    continue
                shortest = min(longest.get(a, dur), longest.get(b, dur))
                sustained = shortest >= SUSTAINED_CLASH_BEATS
                diatonico = escala is not None and \
                    a % 12 in escala and b % 12 in escala
                cor = diatonico and (b - a) >= SPREAD_SEMITONES
                if c < 0.2 and (b - a) < 15:
                    if cor:
                        issues.append(Issue(
                            start, "cor", 1,
                            f"{T.midi_to_name(a)}/{T.midi_to_name(b)}: "
                            f"{_interval_name(b - a)} entre vozes afastadas — "
                            "tensao de cor, dentro da tonalidade",
                            (a, b), (dono.get(a, ""), dono.get(b, ""))))
                    elif sustained and on_beat:
                        issues.append(Issue(
                            start, "dissonancia", 2,
                            f"{T.midi_to_name(a)} e {T.midi_to_name(b)} formam "
                            f"{_interval_name(b - a)} e ficam segurando — choque duro",
                            (a, b), (dono.get(a, ""), dono.get(b, ""))))
                    elif sustained:
                        issues.append(Issue(
                            start, "dissonancia", 1,
                            f"{T.midi_to_name(a)} e {T.midi_to_name(b)}: "
                            f"{_interval_name(b - a)} fora do tempo forte",
                            (a, b), (dono.get(a, ""), dono.get(b, ""))))
                    else:
                        issues.append(Issue(
                            start, "passagem", 0,
                            f"{T.midi_to_name(a)}/{T.midi_to_name(b)}: "
                            f"{_interval_name(b - a)} de passagem — normal se a "
                            "nota curta seguir andando por grau",
                            (a, b), (dono.get(a, ""), dono.get(b, ""))))
                elif c < 0.4 and (b - a) < 13 and sustained:
                    issues.append(Issue(
                        start, "tensao", 1,
                        f"{T.midi_to_name(a)}/{T.midi_to_name(b)}: "
                        f"{_interval_name(b - a)} cria tensao (ok se for proposital)",
                        (a, b), (dono.get(a, ""), dono.get(b, ""))))
        s = sum(pair_scores) / len(pair_scores)
        total_s += s * dur
        total_w += dur
    return total_s / total_w if total_w else 1.0


def _interval_name(semitones: int) -> str:
    names = ["unissono", "segunda menor", "segunda maior", "terca menor",
             "terca maior", "quarta justa", "tritono", "quinta justa",
             "sexta menor", "sexta maior", "setima menor", "setima maior"]
    base = names[semitones % 12]
    octs = semitones // 12
    return base if octs == 0 else f"{base} + {octs} oitava(s)"


def _tonal_fit(notes, key: T.KeyGuess, issues: list[Issue]) -> float:
    allowed = T.scale_pitch_classes(key.root, key.mode)
    # graus emprestados muito comuns nao sao "erro", so custam menos ponto
    borrowed = T.scale_pitch_classes(key.root, "harmonic_minor") | \
        T.scale_pitch_classes(key.root, "dorian" if key.mode != "major" else "mixolydian")
    inside = 0.0
    total = 0.0
    for n in notes:
        w = n.dur
        total += w
        pc = n.pitch % 12
        if pc in allowed:
            inside += w
        elif pc in borrowed:
            inside += w * 0.6
        else:
            if w >= 0.25:
                issues.append(Issue(
                    n.start, "fora_da_escala", 1,
                    f"{T.midi_to_name(n.pitch)} esta fora de {key.name}",
                    (n.pitch,)))
    return inside / total if total else 1.0


def _voice_leading(slices, issues: list[Issue], style: str = "chiptune") -> float:
    """Paralelas de verdade: MESMO par de vozes, mesmo intervalo, mesma direcao.

    Em estilo 'strict' (contraponto) isso pesa; em 'chiptune'/'pop' e so uma dica,
    porque quinta paralela e vocabulario do genero, nao erro.
    """
    strict = style == "strict"
    par_weight = 0.5 if strict else 0.12
    par_sev = 1 if strict else 0
    penalties = 0.0
    checks = 0
    prev = None
    for start, dur, sounding in slices:
        cur = sorted({n.pitch for n in sounding})
        if prev and len(cur) >= 2 and len(prev) >= 2 and cur != prev:
            checks += 1
            if len(cur) == len(prev):
                for i in range(len(cur)):
                    for j in range(i + 1, len(cur)):
                        p_iv = (prev[j] - prev[i]) % 12
                        c_iv = (cur[j] - cur[i]) % 12
                        if p_iv != c_iv or p_iv not in (0, 7):
                            continue
                        di, dj = cur[i] - prev[i], cur[j] - prev[j]
                        if di == 0 or dj == 0 or (di > 0) != (dj > 0):
                            continue
                        penalties += par_weight
                        issues.append(Issue(
                            start, "paralelas", par_sev,
                            f"{'Quintas' if p_iv == 7 else 'Oitavas'} paralelas entre "
                            f"{T.midi_to_name(prev[i])}/{T.midi_to_name(prev[j])} e "
                            f"{T.midi_to_name(cur[i])}/{T.midi_to_name(cur[j])}"
                            + ("" if strict else " — no chiptune isso costuma ser proposital"),
                            (cur[i], cur[j])))
                        break
            leap = abs(cur[-1] - prev[-1])
            if leap > 12:
                penalties += 0.35
                issues.append(Issue(
                    start, "salto", 0,
                    f"Salto de {leap} semitons na voz aguda — considere um caminho "
                    "mais suave",
                    (prev[-1], cur[-1])))
        if len(cur) >= 2:
            prev = cur
    if not checks:
        return 1.0
    return max(0.0, 1.0 - penalties / checks)


def _low_register(slices, issues: list[Issue]) -> float:
    bad = 0.0
    total = 0.0
    seen = set()
    for start, dur, sounding in slices:
        pitches = sorted({n.pitch for n in sounding})
        low = [p for p in pitches if p < MUD_ZONE]
        total += dur
        if len(low) >= 2:
            iv = low[1] - low[0]
            if 0 < iv < MUD_MIN_INTERVAL:
                bad += dur
                key = (low[0], low[1])
                if key not in seen:
                    seen.add(key)
                    issues.append(Issue(
                        start, "grave_embolado", 1,
                        f"{T.midi_to_name(low[0])} e {T.midi_to_name(low[1])} "
                        "muito perto no grave — vira lama. Abra para quinta ou "
                        "suba uma oitava.",
                        (low[0], low[1])))
    return 1.0 - (bad / total if total else 0.0)


def _progression_coherence(chords, key: T.KeyGuess, issues: list[Issue]) -> float:
    if len(chords) < 2:
        return 1.0
    allowed = T.scale_pitch_classes(key.root, key.mode)
    good = 0
    for beat, ch in chords:
        diat = ch.root % 12 in allowed
        # dominante secundaria e emprestimo modal contam como intencionais
        if diat:
            good += 1
        elif ch.quality in ("7", "dim7", "m7b5"):
            good += 0.7
        else:
            issues.append(Issue(
                beat, "acorde_estranho", 0,
                f"{ch.name} nao e diatonico em {key.name} — pode ser cor, "
                "pode ser engano",
                ()))
    return good / len(chords)


# --------------------------------------------------------------------------
# API principal
# --------------------------------------------------------------------------
WEIGHTS = {
    "consonancia": 0.40,
    "tonalidade": 0.22,
    "conducao": 0.10,
    "grave_limpo": 0.16,
    "progressao": 0.12,
}

# Quanto cada choque duro distinto tira do score final, ate um teto.
CLASH_PENALTY = 4.0
CLASH_PENALTY_CAP = 18.0


# Peso de cada pista harmonica no desempate de tonalidade.
TONIC_HINT_FIRST = 0.16
TONIC_HINT_LAST = 0.07
TONIC_HINT_COMMON = 0.05


def _tonic_hints(slices) -> dict[int, float]:
    """Onde a harmonia diz que fica a tonica.

    O sinal mais confiavel e o BAIXO, nao o rotulo do acorde: numa textura com
    arpejo rapido o identificador de acorde erra a fundamental com facilidade,
    mas a nota mais grave de cada fatia quase sempre e a fundamental real.
    Pontuamos o baixo da primeira fatia, o da ultima, e o mais presente ao
    longo do trecho. Sem isso Re menor vira Fa maior — mesmas sete notas.
    """
    hints: dict[int, float] = {}
    if not slices:
        return hints

    def add(pc: int, w: float) -> None:
        hints[pc] = hints.get(pc, 0.0) + w

    bass = [(start, dur, min(n.pitch for n in sounding) % 12)
            for start, dur, sounding in slices]
    add(bass[0][2], TONIC_HINT_FIRST)
    add(bass[-1][2], TONIC_HINT_LAST)
    peso: dict[int, float] = {}
    for _, dur, pc in bass:
        peso[pc] = peso.get(pc, 0.0) + dur
    add(max(peso, key=lambda k: (peso[k], k == bass[0][2])), TONIC_HINT_COMMON)
    return hints


def analyze(notes: Iterable[NoteEvent],
            key: T.KeyGuess | None = None,
            beats_per_chord: float = 1.0,
            style: str = "chiptune") -> HarmonyReport:
    """style: 'chiptune'/'pop' (padrao) ou 'strict' para regras de contraponto."""
    notes = [n for n in notes if n.dur > 0]
    if not notes:
        return HarmonyReport(100.0, None, metrics={}, issues=[])

    issues: list[Issue] = []
    sl = _slices(notes)

    # acordes por janela
    chords: list[tuple[float, T.ChordMatch]] = []
    end = max(n.start + n.dur for n in notes)
    b = 0.0
    last_name = None
    while b < end:
        window = [n.pitch for n in notes
                  if n.start < b + beats_per_chord and n.start + n.dur > b]
        if len(set(p % 12 for p in window)) >= 2:
            ch = T.detect_chord(window)
            if ch and ch.name != last_name:
                chords.append((b, ch))
                last_name = ch.name
        b += beats_per_chord

    if key is None:
        key = T.detect_key(_pc_weights(notes), _tonic_hints(sl))

    # Aderencia tonal so vale muito se existe MESMO uma tonalidade clara.
    # Sem isso, um cluster atonal "encaixa" em qualquer tom e ganha nota cheia.
    fit = _tonal_fit(notes, key, issues)
    conf_factor = 0.55 + 0.45 * key.confidence

    m = {
        "consonancia": _vertical_consonance(sl, issues, key),
        "tonalidade": fit * conf_factor,
        "conducao": _voice_leading(sl, issues, style),
        "grave_limpo": _low_register(sl, issues),
        "progressao": _progression_coherence(chords, key, issues),
    }
    score = sum(m[k] * WEIGHTS[k] for k in WEIGHTS) * 100.0

    hard = {i.pitches for i in issues if i.severity == 2}
    score -= min(CLASH_PENALTY_CAP, CLASH_PENALTY * len(hard))
    score = max(0.0, min(100.0, score))

    issues.sort(key=lambda i: (-i.severity, i.beat))
    return HarmonyReport(round(score, 1), key, chords, m, issues)


@dataclass
class SectionReport:
    name: str
    start_beat: float
    end_beat: float
    report: "HarmonyReport"


def analyze_sections(notes: Iterable[NoteEvent],
                     sections: Sequence[tuple[str, float, float]],
                     **kw) -> list[SectionReport]:
    """Um laudo por trecho, em vez de um laudo so para a musica inteira.

    Musica de jogo modula. Uma faixa que vai de Re menor para Fa menor e
    depois para Sol menor nao TEM uma tonalidade unica — analisar tudo junto
    faz a metrica tonal despencar por um motivo que nao e defeito. Rodando
    trecho a trecho, cada seção e julgada no proprio tom.
    """
    notes = list(notes)
    out: list[SectionReport] = []
    for name, a, b in sections:
        chunk = [n for n in notes if n.start < b and n.start + n.dur > a]
        out.append(SectionReport(name, a, b, analyze(chunk, **kw)))
    return out


def sections_table_pt(reports: Sequence[SectionReport],
                      beats_per_bar: int = 4) -> str:
    """Tabela pronta para o terminal — e para a barra de status da interface."""
    w = max((len(r.name) for r in reports), default=6)
    linhas = [f"{'SEÇÃO'.ljust(w)}  COMP.     SCORE  TOM              VEREDITO",
              "-" * (w + 62)]
    for r in reports:
        c0 = int(r.start_beat // beats_per_bar) + 1
        c1 = int(r.end_beat // beats_per_bar)
        tom = r.report.key.name if r.report.key else "—"
        linhas.append(f"{r.name.ljust(w)}  {c0:3d}-{c1:<4d}  "
                      f"{r.report.score:5.1f}  {tom:15s}  {r.report.verdict}")
    media = sum(r.report.score for r in reports) / max(1, len(reports))
    linhas.append("-" * (w + 62))
    linhas.append(f"{'MÉDIA'.ljust(w)}            {media:5.1f}")
    return "\n".join(linhas)


def suggest_fix(report: HarmonyReport) -> list[str]:
    """Traduz as metricas mais fracas em conselhos acionaveis."""
    tips = []
    m = report.metrics
    if m.get("consonancia", 1) < 0.7:
        tips.append("Troque as segundas/setimas empilhadas por tercas e sextas, "
                    "ou afaste as vozes em uma oitava.")
    if m.get("tonalidade", 1) < 0.75 and report.key:
        tips.append(f"Ative o Scale Lock em {report.key.name} para prender as "
                    "notas soltas na escala.")
    if m.get("conducao", 1) < 0.7:
        tips.append("Mova as vozes pelo caminho mais curto entre acordes: "
                    "segure as notas comuns e ande de meio tom onde der.")
    if m.get("grave_limpo", 1) < 0.85:
        tips.append("No grave use so a fundamental e a quinta. Deixe as tercas "
                    "acima de C3.")
    if m.get("progressao", 1) < 0.7:
        tips.append("Volte a progressoes diatonicas (I-V-vi-IV, i-VI-III-VII) e "
                    "use os acordes de fora como tempero pontual.")
    if not tips:
        tips.append("Esta soando bem. Se quiser mais tensao proposital, "
                    "experimente um tritono resolvendo por meio tom.")
    return tips
