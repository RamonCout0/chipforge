"""
chipforge.core.intensity
========================
Curva de intensidade da musica — o quanto esta "pegando fogo" a cada instante.

Serve para duas coisas:

1. O palco de sprites do app: quanto mais acao, mais agitado fica o bicho
   na tela. Cinco niveis, do parado ao surto.
2. A engine do jogo: `exportar_json()` cospe tempo -> nivel, entao o mesmo
   personagem que dancou aqui dentro dança igual dentro do jogo, sem precisar
   analisar audio em tempo de execucao.

A intensidade NAO e o volume. Volume voce consegue com um fader; intensidade
e densidade de notas, polifonia, quantas camadas estao tocando, se a bateria
entrou e quanto do registro esta ocupado. E por isso que um trecho baixinho e
frenetico marca alto, e um acorde longo e fortissimo marca baixo.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# peso de cada fator na soma final
PESOS = {
    "densidade": 0.28,     # quantas notas comecam por batida
    "polifonia": 0.22,     # quantas soam ao mesmo tempo
    "camadas": 0.20,       # quantas tracks diferentes estao ativas
    "energia": 0.15,       # velocity somada
    "percussao": 0.09,     # a bateria entrou?
    "registro": 0.06,      # o quanto do grave ao agudo esta ocupado
}

NIVEIS_PADRAO = 5


@dataclass
class CurvaIntensidade:
    tempos: np.ndarray          # em segundos
    valores: np.ndarray         # 0..1
    niveis: np.ndarray          # inteiros 1..n_niveis
    n_niveis: int
    fatores: dict

    def em(self, segundos: float) -> float:
        if len(self.tempos) == 0:
            return 0.0
        i = int(np.searchsorted(self.tempos, segundos, side="right") - 1)
        return float(self.valores[max(0, min(i, len(self.valores) - 1))])

    def nivel_em(self, segundos: float) -> int:
        if len(self.tempos) == 0:
            return 1
        i = int(np.searchsorted(self.tempos, segundos, side="right") - 1)
        return int(self.niveis[max(0, min(i, len(self.niveis) - 1))])

    def exportar_json(self, caminho: str | Path, nome: str = "") -> Path:
        caminho = Path(caminho).with_suffix(".json")
        caminho.write_text(json.dumps({
            "nome": nome,
            "formato": "chipforge.intensidade/1",
            "niveis": self.n_niveis,
            "passo_s": float(self.tempos[1] - self.tempos[0])
                       if len(self.tempos) > 1 else 0.0,
            "pontos": [{"t": round(float(t), 3),
                        "v": round(float(v), 3),
                        "n": int(n)}
                       for t, v, n in zip(self.tempos, self.valores, self.niveis)],
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        return caminho

    def resumo_pt(self, largura: int = 56) -> str:
        if not len(self.valores):
            return "(sem notas)"
        passos = min(largura, len(self.valores))
        blocos = np.array_split(self.valores, passos)
        escala = " ▁▂▃▄▅▆▇█"
        linha = "".join(escala[int(np.clip(b.mean(), 0, 1) * (len(escala) - 1))]
                        for b in blocos)
        return linha


def _normalizar(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if not len(v):
        return v
    topo = np.percentile(v, 95)
    if topo <= 1e-9:
        return np.zeros_like(v)
    return np.clip(v / topo, 0.0, 1.0)


def calcular(project, resolucao_beats: float = 0.5,
             n_niveis: int = NIVEIS_PADRAO,
             ataque: float = 0.55, relaxamento: float = 0.12) -> CurvaIntensidade:
    """ataque/relaxamento: o quanto a curva sobe rapido e desce devagar.

    Sprite que acompanha musica precisa reagir na hora da pancada e descer
    com calma — se subir e descer na mesma velocidade, treme e fica feio.
    """
    ppq = project.ppq
    passo = max(1, int(resolucao_beats * ppq))
    fim = project.length_ticks or passo
    n = max(1, int(np.ceil(fim / passo)))

    inicios = np.zeros(n)
    soando = np.zeros(n)
    energia = np.zeros(n)
    perc = np.zeros(n)
    grave = np.full(n, 127.0)
    agudo = np.zeros(n)
    camadas = [set() for _ in range(n)]

    for c in project.arrangement:
        pat = project.patterns.get(c.pattern_id)
        tr = project.track(c.track_id)
        if not pat or not tr or tr.mute:
            continue
        percussivo = bool(getattr(tr.instrument, "is_drum", False))
        for nota in pat.notes:
            ini = c.start + nota.start
            fim_n = ini + nota.dur
            i0 = min(n - 1, ini // passo)
            inicios[i0] += 1
            for i in range(int(i0), min(n, int(fim_n // passo) + 1)):
                soando[i] += 1
                energia[i] += nota.velocity / 127.0
                camadas[i].add(tr.id)
                if percussivo:
                    perc[i] += 1
                else:
                    grave[i] = min(grave[i], nota.pitch)
                    agudo[i] = max(agudo[i], nota.pitch)

    espalhamento = np.where(agudo > 0, np.maximum(0.0, agudo - grave), 0.0)
    fatores = {
        "densidade": _normalizar(inicios),
        "polifonia": _normalizar(soando),
        "camadas": _normalizar(np.array([len(s) for s in camadas], dtype=float)),
        "energia": _normalizar(energia),
        "percussao": _normalizar(perc),
        "registro": _normalizar(espalhamento),
    }
    bruto = sum(fatores[k] * PESOS[k] for k in PESOS)
    bruto = _normalizar(bruto)

    # envelope assimetrico
    suave = np.empty_like(bruto)
    acc = 0.0
    for i, v in enumerate(bruto):
        coef = ataque if v > acc else relaxamento
        acc += (v - acc) * coef
        suave[i] = acc

    limites = np.linspace(0, 1, n_niveis + 1)[1:-1]
    niveis = np.searchsorted(limites, suave, side="right") + 1

    dt = project.ticks_to_seconds(passo)
    tempos = np.arange(n) * dt
    return CurvaIntensidade(tempos, suave, niveis.astype(int), n_niveis, fatores)
