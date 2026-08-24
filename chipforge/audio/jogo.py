"""
chipforge.audio.jogo
====================
Exportacao para engine de jogo: regioes, pontos de loop, stems e manifesto.

Por que isto existe: uma trilha de raid nao e um arquivo. E tres arquivos que
a engine troca conforme a luta anda, cada um capaz de repetir indefinidamente
sem que o jogador perceba a emenda, e com um ponto combinado onde a troca pode
acontecer sem cortar a musica no meio de uma frase.

O `export_wav` do engine entrega um arquivo unico e pronto. Isso serve para
ouvir; nao serve para jogar.

Tres problemas concretos, resolvidos aqui:

1. **A emenda do loop.** Cortar no compasso certo e necessario e insuficiente:
   no ultimo compasso ainda esta soando a cauda da sala e o release das notas.
   Cortar ali deixa um silencio abrupto a cada volta. A solucao e renderizar
   ALEM do fim e dobrar essa sobra de volta sobre o inicio do loop — a cauda
   passa a chegar por cima da repeticao, exatamente como chegaria se a musica
   tivesse continuado.

2. **O ponto de troca.** A engine nao pode trocar de fase no meio de um
   compasso. O manifesto diz, em segundos e em amostras, onde estao as bordas
   de compasso seguras.

3. **A conversao de tempo.** Com mapa de tempo, compasso 32 nao esta em
   `32 * 4 * 60 / bpm`. So o projeto sabe onde ele esta, e e ele que responde.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .engine import export_wav


@dataclass
class Regiao:
    """Um trecho tocavel da peca — tipicamente uma fase da luta."""
    nome: str
    inicio_compasso: int
    compassos: int
    # onde a repeticao volta. None = volta ao inicio da regiao. Apontar para
    # depois do inicio e o padrao de trilha de jogo: a introducao toca uma vez
    # e o loop acontece so do corpo em diante.
    loop_inicio_compasso: int | None = None
    nivel: int = 3                       # intensidade 1..5, para o Palco
    proxima: str | None = None           # para onde a luta vai depois
    descricao: str = ""

    @property
    def fim_compasso(self) -> int:
        return self.inicio_compasso + self.compassos

    @property
    def loop_de(self) -> int:
        return (self.loop_inicio_compasso
                if self.loop_inicio_compasso is not None else self.inicio_compasso)


def _amostras(projeto, compasso: int, sr: int) -> int:
    """Compasso -> amostra, respeitando o mapa de tempo do projeto."""
    return int(round(projeto.ticks_to_seconds(compasso * projeto.ticks_per_bar) * sr))


def costurar_loop(audio: np.ndarray, inicio: int, fim: int,
                  cauda: int, janela: int = 0) -> np.ndarray:
    """Corta [inicio, fim) e dobra a cauda de volta sobre o comeco do trecho.

    Esta e a funcao inteira do modulo em uma ideia. O trecho cortado termina
    seco; a cauda que vinha depois dele (sala, release, prato aberto) e somada
    sobre as primeiras amostras. Na segunda volta o ouvinte escuta a cauda da
    primeira chegando por cima — que e o que ele escutaria se a musica nunca
    tivesse sido cortada.

    `janela` aplica um cruzamento curto na emenda. Serve para o caso em que a
    cauda somada ainda deixa um degrau de forma de onda; em material com sala
    ela costuma ser desnecessaria, e por isso o padrao e zero.
    """
    audio = np.asarray(audio, dtype=np.float32)
    n = len(audio)
    inicio = max(0, min(inicio, n))
    fim = max(inicio + 1, min(fim, n))
    corpo = audio[inicio:fim].copy()

    sobra = audio[fim:min(fim + cauda, n)]
    if len(sobra):
        k = min(len(sobra), len(corpo))
        corpo[:k] += sobra[:k]

    if janela > 0 and len(corpo) > janela * 2:
        rampa = np.linspace(0.0, 1.0, janela, dtype=np.float32)
        if corpo.ndim == 2:
            rampa = rampa[:, None]
        corpo[:janela] *= rampa
        corpo[-janela:] *= rampa[::-1]
    return corpo


def fatiar(projeto, audio: np.ndarray, regioes, sr: int | None = None,
           cauda_s: float = 4.0) -> dict[str, np.ndarray]:
    """Corta o render completo em um stem por regiao, ja com o loop costurado."""
    sr = sr or projeto.sample_rate
    cauda = int(cauda_s * sr)
    saida: dict[str, np.ndarray] = {}
    for r in regioes:
        ini = _amostras(projeto, r.loop_de, sr)
        fim = _amostras(projeto, r.fim_compasso, sr)
        saida[r.nome] = costurar_loop(audio, ini, fim, cauda)
    return saida


def manifesto(projeto, regioes, sr: int | None = None,
              stems: dict[str, str] | None = None) -> dict:
    """O JSON que a engine le: onde cada fase comeca, termina e repete.

    Tudo em tres unidades de proposito. Segundos para quem programa a logica,
    amostras para quem faz a troca no callback de audio (e precisa de precisao
    de amostra, nao de milissegundo), e compassos para quem for abrir o projeto
    e conferir contra a partitura.
    """
    sr = sr or projeto.sample_rate
    fases = []
    for r in regioes:
        a_ini = _amostras(projeto, r.inicio_compasso, sr)
        a_loop = _amostras(projeto, r.loop_de, sr)
        a_fim = _amostras(projeto, r.fim_compasso, sr)
        fases.append({
            "nome": r.nome,
            "descricao": r.descricao,
            "nivel": r.nivel,
            "proxima": r.proxima,
            "arquivo": (stems or {}).get(r.nome),
            "compassos": {"inicio": r.inicio_compasso, "fim": r.fim_compasso,
                          "loop_inicio": r.loop_de},
            "segundos": {"inicio": round(a_ini / sr, 4),
                         "fim": round(a_fim / sr, 4),
                         "loop_inicio": round(a_loop / sr, 4),
                         "duracao_loop": round((a_fim - a_loop) / sr, 4)},
            "amostras": {"inicio": a_ini, "fim": a_fim, "loop_inicio": a_loop,
                         "duracao_loop": a_fim - a_loop},
        })

    # bordas de compasso: onde a engine pode trocar de fase sem cortar frase
    bordas = []
    total = max((r.fim_compasso for r in regioes), default=0)
    for c in range(total + 1):
        bordas.append(round(projeto.ticks_to_seconds(c * projeto.ticks_per_bar), 4))

    return {
        "nome": projeto.name,
        "sample_rate": sr,
        "bpm_base": projeto.bpm,
        "compasso": [projeto.beats_per_bar, projeto.beat_unit],
        "tem_mapa_de_tempo": bool(getattr(projeto, "tempo_map", None)),
        "fases": fases,
        "bordas_de_compasso_s": bordas,
    }


def exportar(projeto, audio: np.ndarray, regioes, pasta: str | Path,
             nome_base: str = "", sr: int | None = None,
             bit_depth: int = 16, cauda_s: float = 4.0,
             stems: bool = True) -> dict:
    """Escreve a faixa inteira, um stem por fase e o manifesto.

    Devolve o proprio manifesto, para quem quiser continuar em codigo em vez de
    reabrir o arquivo.
    """
    sr = sr or projeto.sample_rate
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    base = nome_base or projeto.name.lower().replace(" ", "_")

    export_wav(audio, pasta / base, sr, bit_depth)

    nomes: dict[str, str] = {}
    if stems:
        for nome, trecho in fatiar(projeto, audio, regioes, sr, cauda_s).items():
            arq = f"{base}_{nome.lower().replace(' ', '_')}"
            export_wav(trecho, pasta / arq, sr, bit_depth)
            nomes[nome] = f"{arq}.wav"

    m = manifesto(projeto, regioes, sr, nomes)
    m["arquivo_completo"] = f"{base}.wav"
    (pasta / f"{base}_fases.json").write_text(
        json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    return m


def relatorio_pt(m: dict) -> str:
    """O manifesto em texto, para conferir sem abrir o JSON."""
    L = [f"{m['nome']} — {len(m['fases'])} fases, {m['sample_rate']} Hz"]
    L.append(f"  {'FASE':16s} {'COMP.':>9s} {'ENTRA':>8s} {'LOOP DE':>8s} "
             f"{'DURACAO':>8s}  NIVEL  SEGUE PARA")
    for f in m["fases"]:
        c, s = f["compassos"], f["segundos"]
        L.append(f"  {f['nome']:16s} {c['inicio']:4d}-{c['fim']:<4d} "
                 f"{s['inicio']:7.1f}s {s['loop_inicio']:7.1f}s "
                 f"{s['duracao_loop']:7.1f}s  {f['nivel']:5d}  "
                 f"{f['proxima'] or '(fim)'}")
    return "\n".join(L)
