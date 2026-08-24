"""
chipforge.analysis.reference
============================
Analise de faixa de referencia.

Voce ouve uma trilha que gostaria de alcancar e quer saber o que, tecnicamente,
faz ela funcionar: em que andamento esta, em que tom, quantas secoes tem e
quanto tempo cada uma dura, como a energia se distribui entre grave, medio e
agudo, e qual o arco de dinamica.

Isto NAO transcreve a musica e nao serve para copiar: o que sai daqui sao
medidas de forma e de mixagem — a materia que qualquer compositor observa
numa referencia antes de escrever a PROPRIA peca.

Depende so de numpy e do ffmpeg para decodificar.
"""
from __future__ import annotations

import subprocess
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core import theory as T

SR_ANALISE = 22050
JANELA = 2048
SALTO = 512

BANDAS = [
    ("sub", 20, 80),
    ("grave", 80, 250),
    ("medio-grave", 250, 800),
    ("medio", 800, 2500),
    ("agudo", 2500, 7000),
    ("brilho", 7000, 11000),
]


@dataclass
class Secao:
    inicio: float
    fim: float
    nivel: int
    rms_db: float

    @property
    def duracao(self) -> float:
        return self.fim - self.inicio


@dataclass
class Referencia:
    caminho: Path
    duracao: float
    bpm: float
    bpm_alternativos: list[float]
    tom: T.KeyGuess
    chroma: np.ndarray
    secoes: list[Secao]
    bandas: dict
    envelope_db: np.ndarray
    tempos: np.ndarray
    faixa_dinamica_db: float
    pico_db: float
    lufs: float = -99.0
    lra: float = 0.0
    notas: list[str] = field(default_factory=list)

    def relatorio_pt(self) -> str:
        L = []
        L.append(f"Arquivo   : {self.caminho.name[:64]}")
        L.append(f"Duracao   : {int(self.duracao//60)}:{self.duracao%60:04.1f}")
        L.append(f"Andamento : {self.bpm:.1f} BPM "
                 f"(candidatos: {', '.join(f'{b:.0f}' for b in self.bpm_alternativos)})")
        rel = ((self.tom.root + 9) % 12, "menor") if self.tom.mode == "major" \
            else ((self.tom.root + 3) % 12, "maior")
        L.append(f"Tom       : {self.tom.name}  "
                 f"(confianca {self.tom.confidence*100:.0f}%; "
                 f"relativa: {T.SHARP_NAMES[rel[0]]} {rel[1]})")
        L.append(f"Dinamica  : pico {self.pico_db:.1f} dBFS · "
                 f"faixa util {self.faixa_dinamica_db:.1f} dB")
        L.append(f"Loudness  : {self.lufs:.1f} LUFS · faixa (LRA) {self.lra:.1f} LU")
        L.append("")
        L.append("Balanco espectral (nivel medio por banda, relativo ao pico):")
        topo = max(self.bandas.values()) or 1.0
        for nome, v in self.bandas.items():
            db = 10 * np.log10(max(v, 1e-12) / topo)
            barra = int(np.clip(48 + db, 0, 48))
            L.append(f"  {nome:12s} {db:6.1f} dB  {'█' * barra}")
        L.append("")
        L.append("Forma (secoes por mudanca de textura):")
        L.append(f"  {'#':>2}  {'INICIO':>7}  {'DUR':>6}  {'NIVEL':>5}  RMS")
        for i, s in enumerate(self.secoes, 1):
            L.append(f"  {i:2d}  {int(s.inicio//60):>2d}:{s.inicio%60:04.1f}  "
                     f"{s.duracao:5.1f}s  {s.nivel:5d}  {s.rms_db:6.1f} dB")
        L.append("")
        L.append("Arco de intensidade:")
        L.append("  " + self.sparkline(84))
        for n in self.notas:
            L.append(f"  · {n}")
        return "\n".join(L)

    def sparkline(self, largura: int = 80) -> str:
        v = self.envelope_db
        if not len(v):
            return ""
        lo, hi = np.percentile(v, 8), np.percentile(v, 99)
        v = np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)
        blocos = np.array_split(v, min(largura, len(v)))
        escala = " ▁▂▃▄▅▆▇█"
        return "".join(escala[int(np.clip(b.mean(), 0, 1) * (len(escala) - 1))]
                       for b in blocos)

    def comparar(self, audio: np.ndarray, sr: int) -> dict:
        """Compara o balanco espectral de um render com o da referencia.

        Devolve, por banda, quantos dB o SEU material esta acima ou abaixo da
        referencia (depois de alinhar o nivel geral). Serve para responder a
        unica pergunta que importa numa mixagem: 'esta faltando grave ou esta
        sobrando agudo?'
        """
        x = np.asarray(audio, dtype=np.float32)
        if x.ndim == 2:
            x = x.mean(axis=1)
        if sr != SR_ANALISE:
            passo = sr / SR_ANALISE
            idx = (np.arange(int(len(x) / passo)) * passo).astype(int)
            x = x[np.clip(idx, 0, len(x) - 1)]
        esp = _espectrograma(x)
        freqs = np.fft.rfftfreq(JANELA, 1 / SR_ANALISE)
        minhas = {}
        for nome, lo, hi in BANDAS:
            m = (freqs >= lo) & (freqs < hi)
            minhas[nome] = float((esp[:, m] ** 2).mean())

        def db(d):
            topo = max(d.values()) or 1.0
            return {k: 10 * np.log10(max(v, 1e-12) / topo) for k, v in d.items()}

        a, b = db(minhas), db(self.bandas)
        return {k: round(a[k] - b[k], 1) for k in b}

    def perfil_alvo(self) -> dict:
        """Balanco espectral da referencia, em dB relativos a banda mais forte.

        E o formato que `Project.master_alvo` e `master.casar_perfil` esperam:
        depois de medir uma faixa que voce quer alcancar, a cadeia de master
        passa a puxar o SEU mix na direcao daquele equilibrio — sem copiar uma
        unica nota dela.
        """
        topo = max(self.bandas.values()) or 1.0
        return {k: round(float(10 * np.log10(max(v, 1e-12) / topo)), 1)
                for k, v in self.bandas.items()}

    def alvo_de_loudness(self) -> float:
        """LUFS integrado da referencia — o alvo honesto para o seu master."""
        return self.lufs

    def niveis_por_segundo(self, passo: float = 0.5) -> list[tuple[float, int]]:
        """Mesmo formato que o Palco consome: tempo -> nivel 1..5."""
        out = []
        for s in self.secoes:
            t = s.inicio
            while t < s.fim:
                out.append((round(t, 2), s.nivel))
                t += passo
        return out


# --------------------------------------------------------------------------
def _decodificar(caminho: Path, sr: int = SR_ANALISE) -> np.ndarray:
    """ffmpeg -> WAV mono. Nao guarda o audio decodificado em disco."""
    tmp = Path(tempfile.gettempdir()) / f"_ref_{abs(hash(str(caminho))) % 10**8}.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(caminho),
         "-ac", "1", "-ar", str(sr), str(tmp)],
        check=True)
    with wave.open(str(tmp)) as w:
        dados = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    tmp.unlink(missing_ok=True)
    return (dados.astype(np.float32) / 32768.0)


def _decodificar_estereo(caminho: Path, sr: int = 44100) -> np.ndarray:
    """Segunda decodificacao, agora em estereo e na taxa cheia.

    A analise de forma roda em mono a 22 kHz porque e o suficiente para tempo,
    tonalidade e secoes — e dez vezes mais rapido. Loudness nao: LUFS depende
    dos dois canais e da ponderacao ate o agudo, entao esta medida precisa do
    arquivo como ele e.
    """
    tmp = Path(tempfile.gettempdir()) / f"_refst_{abs(hash(str(caminho))) % 10**8}.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(caminho),
         "-ac", "2", "-ar", str(sr), str(tmp)],
        check=True)
    with wave.open(str(tmp)) as w:
        dados = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    tmp.unlink(missing_ok=True)
    return (dados.astype(np.float32) / 32768.0).reshape(-1, 2)


def _espectrograma(x: np.ndarray, janela=JANELA, salto=SALTO):
    n = 1 + max(0, (len(x) - janela) // salto)
    jan = np.hanning(janela).astype(np.float32)
    quadros = np.lib.stride_tricks.sliding_window_view(x, janela)[::salto][:n]
    esp = np.abs(np.fft.rfft(quadros * jan, axis=1))
    return esp.astype(np.float32)


def _tempo(esp: np.ndarray, sr: int, salto: int):
    """Andamento por autocorrelacao do fluxo espectral (onset strength)."""
    fluxo = np.maximum(0.0, np.diff(esp, axis=0)).sum(axis=1)
    fluxo = fluxo - fluxo.mean()
    if fluxo.std() > 0:
        fluxo /= fluxo.std()
    ac = np.correlate(fluxo, fluxo, mode="full")[len(fluxo) - 1:]
    quadros_por_s = sr / salto
    candidatos = []
    for bpm in np.arange(60.0, 200.0, 0.25):
        lag = int(round(60.0 / bpm * quadros_por_s))
        if 1 < lag < len(ac):
            candidatos.append((float(ac[lag]), float(bpm)))
    candidatos.sort(reverse=True)
    melhor = candidatos[0][1]
    # descarta candidatos que sao o mesmo pulso em outra subdivisao
    alt = []
    for _, bpm in candidatos:
        if all(abs(bpm - b) > 6 and abs(bpm - b * 2) > 6 and abs(bpm * 2 - b) > 6
               for b in alt):
            alt.append(bpm)
        if len(alt) >= 4:
            break
    return melhor, alt


def _chroma(esp: np.ndarray, sr: int, janela: int,
            f_min: float = 55.0, f_max: float = 2200.0) -> np.ndarray:
    freqs = np.fft.rfftfreq(janela, 1 / sr)
    validas = (freqs > f_min) & (freqs < f_max)
    f = freqs[validas]
    pcs = np.mod(np.round(12 * np.log2(f / 440.0) + 69).astype(int), 12)
    peso = esp[:, validas]
    saida = np.zeros(12)
    for pc in range(12):
        saida[pc] = float(peso[:, pcs == pc].sum())
    return saida


def _novidade(esp: np.ndarray, kernel: int = 32) -> np.ndarray:
    """Curva de novidade: onde a TEXTURA muda, nao onde ha um ataque."""
    log = np.log1p(esp)
    log = log / (np.linalg.norm(log, axis=1, keepdims=True) + 1e-9)
    n = len(log)
    nov = np.zeros(n)
    for i in range(kernel, n - kernel):
        antes = log[i - kernel:i]
        depois = log[i:i + kernel]
        nov[i] = 1.0 - float((antes.mean(axis=0) * depois.mean(axis=0)).sum())
    return nov


def analisar(caminho: str | Path, n_secoes_max: int = 12,
             n_niveis: int = 5) -> Referencia:
    caminho = Path(caminho)
    x = _decodificar(caminho)
    sr = SR_ANALISE
    dur = len(x) / sr

    esp = _espectrograma(x)
    tempos = np.arange(len(esp)) * SALTO / sr

    bpm, alt = _tempo(esp, sr, SALTO)

    chroma = _chroma(esp, sr, JANELA)
    # Relativa maior e relativa menor tem o mesmo chroma. Quem desempata, aqui
    # como no medidor de harmonia, e o GRAVE: a fundamental da tonalidade e a
    # nota que o baixo mais sustenta.
    chroma_grave = _chroma(esp, sr, JANELA, f_min=55.0, f_max=260.0)
    if chroma_grave.max() > 0:
        cg = chroma_grave / chroma_grave.max()
        ordem = np.argsort(cg)[::-1]
        dica = {int(ordem[0]): 0.16, int(ordem[1]): 0.06}
    else:
        dica = {}
    tom = T.detect_key(list(chroma / (chroma.max() or 1.0)), dica)

    freqs = np.fft.rfftfreq(JANELA, 1 / sr)
    bandas = {}
    for nome, lo, hi in BANDAS:
        m = (freqs >= lo) & (freqs < hi)
        bandas[nome] = float((esp[:, m] ** 2).mean())

    rms = np.sqrt(np.maximum((esp ** 2).mean(axis=1), 1e-12))
    env_db = 20 * np.log10(rms / rms.max())
    suave = np.convolve(env_db, np.ones(24) / 24, mode="same")

    nov = _novidade(esp)
    limiar = np.percentile(nov, 92)
    cortes = [0]
    minimo = int(6.0 * sr / SALTO)          # secao nao dura menos que 6 s
    for i in range(1, len(nov) - 1):
        if nov[i] >= limiar and nov[i] >= nov[i - 1] and nov[i] > nov[i + 1]:
            if i - cortes[-1] >= minimo:
                cortes.append(i)
    cortes.append(len(esp) - 1)
    if len(cortes) - 1 > n_secoes_max:
        forca = sorted(((nov[c], c) for c in cortes[1:-1]), reverse=True)
        manter = {c for _, c in forca[:n_secoes_max - 1]}
        cortes = [0] + sorted(manter) + [len(esp) - 1]

    picos = np.array([suave[a:b].mean() for a, b in zip(cortes, cortes[1:])])
    lo, hi = picos.min(), picos.max()
    secoes = []
    for (a, b), p in zip(zip(cortes, cortes[1:]), picos):
        frac = (p - lo) / max(hi - lo, 1e-9)
        secoes.append(Secao(float(tempos[a]), float(tempos[b]),
                            int(np.clip(1 + frac * (n_niveis - 1), 1, n_niveis)),
                            float(p)))

    try:
        from ..audio import master as MST
        st = _decodificar_estereo(caminho)
        lufs, lra = MST.lufs_integrado(st, 44100), MST.faixa_de_loudness(st, 44100)
    except Exception:
        lufs, lra = -99.0, 0.0

    notas = []
    total = sum(bandas.values()) or 1.0
    if bandas["sub"] / total > 0.25:
        notas.append("Muita energia no sub: percussao grave carrega a faixa.")
    if bandas["medio"] / total > 0.20:
        notas.append("Medio forte: e ali que a melodia principal vive.")
    if bandas["brilho"] / total < 0.02:
        notas.append("Pouco brilho acima de 7 kHz — som mais encorpado que aereo.")
    compasso = 60.0 / bpm * 4
    notas.append(f"Um compasso de 4/4 dura {compasso:.2f}s; "
                 f"a faixa tem cerca de {dur / compasso:.0f} compassos.")
    if secoes:
        media = sum(s.duracao for s in secoes) / len(secoes)
        notas.append(f"Secoes de {media:.0f}s em media "
                     f"(~{media / compasso:.0f} compassos cada).")

    return Referencia(
        caminho=caminho, duracao=dur, bpm=bpm, bpm_alternativos=alt,
        tom=tom, chroma=chroma, secoes=secoes, bandas=bandas,
        envelope_db=suave, tempos=tempos,
        faixa_dinamica_db=float(np.percentile(suave, 95) - np.percentile(suave, 5)),
        pico_db=float(20 * np.log10(max(np.abs(x).max(), 1e-9))),
        lufs=lufs, lra=lra,
        notas=notas)
