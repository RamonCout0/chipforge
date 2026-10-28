"""
chipforge.audio.engine
======================
Renderiza um Project (ou um Pattern solto) em audio, e toca em tempo real.

Estrategia: render offline em numpy, playback por stream do sounddevice lendo
de um buffer. E o modelo que os trackers usam — previsivel, sem xruns, e o
mesmo codigo serve para exportar WAV. Se depois voce quiser gravar automacao
ao vivo, o ponto de entrada e o callback em RealtimePlayer.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from ..audio.instrument import DRUM_BASE_PITCH, GM_DRUM_MAP, Instrument
from . import effects as FX


# --------------------------------------------------------------------------
# Render offline
# --------------------------------------------------------------------------
def render_notes(notes, instrument: Instrument, sr: int, sec_per_tick: float,
                 volume: float = 1.0, pan: float = 0.0,
                 total_samples: int | None = None,
                 mapa=None, automacao=None) -> np.ndarray:
    """notes: iteraveis com .start/.dur/.pitch/.velocity em ticks.

    `mapa` (MapaDeTempo) faz cada nota consultar o andamento do proprio ponto
    da peca, em vez de multiplicar por um `sec_per_tick` fixo. E o que permite
    accelerando e ritardando: sem ele, mudar o bpm no meio desalinharia tudo
    que vem depois.

    `automacao` (Automacao de volume) e aplicada como envoltoria continua no
    fim, depois da soma das notas — que e onde um fader de verdade fica.
    """
    notes = sorted(notes, key=lambda n: n.start)
    if not notes:
        return np.zeros((total_samples or 1, 2), dtype=np.float32)

    def inicio_s(t):
        return mapa.segundos(t) if mapa is not None else t * sec_per_tick

    def dur_s(t, d):
        return (mapa.duracao_de(t, d) if mapa is not None else d * sec_per_tick)

    end_tick = max(n.start + n.dur for n in notes)
    tail = int(sr * (instrument.env.release + 0.6))
    n_total = total_samples or int(inicio_s(end_tick) * sr) + tail
    buf = np.zeros(n_total + tail, dtype=np.float32)

    prev_pitch = None
    for n in notes:
        d = max(dur_s(n.start, n.dur), 0.01)
        off = int(inicio_s(n.start) * sr)
        sig = instrument.render_note(n.pitch, d, sr, n.velocity, prev_pitch)
        end = min(off + len(sig), len(buf))
        if end > off:
            buf[off:end] += sig[:end - off]
        prev_pitch = float(n.pitch)

    buf *= volume
    if automacao is not None:
        buf *= automacao.curva(len(buf), mapa, sr, sec_per_tick)
    st = FX.stereo_pan(buf, pan + instrument.pan)
    return st[:n_total] if total_samples else st


def _render_track(tr, flat, sr: int, spt: float, total: int,
                  mapa=None) -> np.ndarray:
    """Rende UMA track, ja resolvendo bateria, dobra de secao e automacao."""
    auto = None
    curva = (getattr(tr, "automacao", None) or {}).get("volume")
    if curva:
        from ..core.dinamica import Automacao
        auto = Automacao.from_list(curva)

    # Bateria sintetizada só quando o instrumento não é de amostra (SoundFont ou SFZ
    # trazem as próprias gravações de bumbo, caixa e pratos).
    chip_drums = getattr(tr.instrument, "is_drum", False) and \
        getattr(tr.instrument, "osc", "") != "sample"
    if chip_drums:
        return render_drum_notes(flat, sr, spt, volume=tr.volume,
                                 total_samples=total, mapa=mapa, automacao=auto,
                                 pan=tr.pan)

    sec = getattr(tr, "section", None)
    if not sec or int(sec.get("vozes", 1)) < 2:
        return render_notes(flat, tr.instrument, sr, spt,
                            volume=tr.volume, pan=tr.pan, total_samples=total,
                            mapa=mapa, automacao=auto)

    # --- dobra de secao ---------------------------------------------------
    # Um sample de violino tocado quatro vezes em uníssono perfeito continua
    # sendo UM violino, so mais alto. O que faz soar naipe e o que nenhum
    # musico consegue evitar: cada um entra alguns milissegundos depois,
    # afinado alguns cents diferente, e sentado noutro lugar do palco.
    from dataclasses import replace
    vozes = int(sec.get("vozes", 3))
    detune = float(sec.get("detune", 8.0))
    spread_ms = float(sec.get("spread_ms", 18.0))
    pan_spread = float(sec.get("pan", 0.5))
    base_detune = getattr(tr.instrument, "detune_cents", 0.0)

    acc = np.zeros((total, 2), dtype=np.float32)
    for v in range(vozes):
        k = (v / (vozes - 1)) - 0.5 if vozes > 1 else 0.0
        inst = replace(tr.instrument, detune_cents=base_detune + detune * 2 * k)
        pan = float(np.clip(tr.pan + pan_spread * 2 * k, -1.0, 1.0))
        voz = render_notes(flat, inst, sr, spt,
                           volume=tr.volume / np.sqrt(vozes),
                           pan=pan, total_samples=total,
                           mapa=mapa, automacao=auto)
        atraso = int(abs(spread_ms) * sr / 1000 * (v / max(1, vozes - 1)))
        if atraso:
            voz = np.vstack([np.zeros((atraso, 2), np.float32), voz])[:total]
        acc += voz
    return acc


def render_project(project, sr: int | None = None,
                   apply_master: bool = True,
                   progress=None) -> np.ndarray:
    """Devolve audio estereo float32 (-1..1) do arranjo inteiro.

    Sinal de cada canal: notas -> inserts da track -> soma no mix,
    e uma copia proporcional ao `reverb_send` vai para o barramento de sala.
    O reverb roda UMA vez no barramento somado, nao por track: e mais barato
    e, mais importante, e o que faz os instrumentos soarem no mesmo lugar.
    """
    from . import fx

    sr = sr or project.sample_rate
    spt = 60.0 / project.bpm / project.ppq
    mapa = project.mapa_de_tempo if getattr(project, "tempo_map", None) else None
    total = int(project.ticks_to_seconds(project.length_ticks) * sr) + int(sr * 3.5)
    mix = np.zeros((total, 2), dtype=np.float32)
    barramento = np.zeros((total, 2), dtype=np.float32)
    usou_reverb = False

    tracks = project.audible_tracks
    por_track = []
    for tr in tracks:
        flat = []
        for c in project.arrangement:
            if c.track_id != tr.id:
                continue
            pat = project.patterns.get(c.pattern_id)
            if not pat:
                continue
            for n in pat.notes:
                flat.append(type(n)(n.start + c.start, n.dur, n.pitch, n.velocity))
        if flat:
            por_track.append((tr, flat))

    ps1 = _e_ps1(project)
    relatorio = {"console": "ps1" if ps1 else (project.console or ""),
                 "vozes_roubadas": 0, "pico_de_vozes": 0}
    if ps1:
        relatorio.update(_limitar_vozes_ps1(project, por_track))
    from contextlib import nullcontext
    from . import psx as PSX

    for ti, (tr, flat) in enumerate(por_track):
        with (PSX.ativo(getattr(project, "psx_taxa", 22050) or 22050)
              if ps1 else nullcontext()):
            stereo = _render_track(tr, flat, sr, spt, total, mapa)
        inserts = getattr(tr, "inserts", None)
        if inserts:
            stereo = fx.aplicar(stereo, sr, inserts)[:total]
            if len(stereo) < total:
                stereo = np.vstack([stereo,
                                    np.zeros((total - len(stereo), 2), np.float32)])
        mix += stereo
        envio = float(getattr(tr, "reverb_send", 0.0) or 0.0)
        if envio > 0:
            barramento += stereo * envio
            usou_reverb = True
        if progress:
            progress((ti + 1) / len(por_track))

    sala = getattr(project, "master_reverb", None)
    if sala and usou_reverb:
        p = dict(sala)
        p.setdefault("wet", 1.0)
        p["dry"] = 0.0
        # Freeverb tambem tem um parametro chamado "sala", mas la ele e o
        # TAMANHO (0..1); aqui e o NOME da sala. O tipo do valor desempata.
        if "psx" in p:
            # o reverb de hardware do PlayStation, preset por nome
            from . import psx as PSX
            mix += PSX.reverb(barramento, sr, preset=p["psx"],
                              wet=float(p.get("wet", 1.0)))[:total]
        elif p.pop("conv", None) or isinstance(p.get("sala"), str):
            # sala por convolucao: densa desde o primeiro milissegundo, sem as
            # ressonancias de filtro pente que denunciam reverb algoritmico
            from . import espaco
            p.pop("largura", None)
            p.pop("damp", None)
            p.pop("predelay_ms", None)
            mix += espaco.reverb(barramento, sr, **p)[:total]
        else:
            mix += fx.reverb(barramento, sr, **p)[:total]

    if getattr(project, "master_inserts", None):
        mix = fx.aplicar(mix, sr, project.master_inserts)[:total]

    if apply_master:
        mix *= project.master_gain
        perfil = getattr(project, "master_perfil", None)
        if perfil:
            # cadeia de master de verdade: alvo de LUFS, pico real e brilho
            # gerado por excitador. `fx.limiter` sozinho so evitava o estouro.
            from . import master as MST
            mix = MST.masterizar(mix, sr, perfil,
                                 perfil_espectral=getattr(project, "master_alvo", None),
                                 medir_antes=False).audio[:total]
        else:
            mix = fx.limiter(mix, 0.97)
    peak = float(np.abs(mix).max())
    if peak > 1.0:
        mix /= peak
    try:
        project.ultimo_render = relatorio
    except Exception:                       # pragma: no cover
        pass
    # o bitcrush vem DEPOIS da normalizacao: dividir pelo pico tirava o audio
    # da grade de 8 bits e o "degrau" do NES virava so um chiado leve
    if apply_master and project.bit_depth and project.bit_depth < 24 \
            and not _e_ps1(project):
        mix = FX.bitcrush(mix, project.bit_depth)
    return mix.astype(np.float32)


def _limitar_vozes_ps1(project, por_track) -> dict:
    """Aplica o teto de 24 vozes do SPU antes de renderizar. Notas roubadas
    sao encurtadas no lugar (sao copias: o projeto nao muda)."""
    from . import psx as PSX
    tps = project.bpm / 60.0 * project.ppq            # ticks por segundo
    eventos = []
    for tr, flat in por_track:
        inst = tr.instrument
        sec = getattr(tr, "section", None) or {}
        peso = max(1, int(sec.get("vozes", 1)))
        if getattr(inst, "is_drum", False):
            rel = 0.12                                 # pele/prato abafa rapido
        elif getattr(inst, "osc", "") != "sample":
            rel = getattr(getattr(inst, "env", None), "release", 0.2)
        else:
            rel = 0.35                                 # cauda tipica de banco
        rel_t = int(min(rel, 1.0) * tps)
        # Driver de som de PS1: a mesma tecla de novo na mesma track reusa a
        # voz (o chimbal em semicolcheia nao empilha 4 vozes, ele se corta).
        proximo: dict[int, int] = {}
        evs_track = []
        for n in sorted(flat, key=lambda x: x.start, reverse=True):
            fim = n.start + n.dur + rel_t
            if n.pitch in proximo:
                fim = min(fim, proximo[n.pitch])
            proximo[n.pitch] = n.start
            evs_track.append([n.start, max(fim, n.start + 1), peso, n])
        eventos += evs_track
    # pico de vozes simultaneas ANTES do corte — diz quanto o arranjo pediu
    marcas = sorted([(e[0], e[2]) for e in eventos] +
                    [(e[1], -e[2]) for e in eventos], key=lambda m: (m[0], m[1]))
    atual = pico = 0
    for _, d in marcas:
        atual += d
        pico = max(pico, atual)
    roubadas = PSX.limitar_vozes(eventos)
    return {"vozes_roubadas": roubadas, "pico_de_vozes": pico}


def _e_ps1(project) -> bool:
    return (getattr(project, "console", "") or "").lower() == "ps1"


def render_pattern(project, pattern, track=None, sr: int | None = None) -> np.ndarray:
    sr = sr or project.sample_rate
    spt = 60.0 / project.bpm / project.ppq
    inst = track.instrument if track else Instrument()
    vol = track.volume if track else 0.8
    out = render_notes(pattern.notes, inst, sr, spt, volume=vol)
    if project.bit_depth < 24:
        out = FX.bitcrush(out, project.bit_depth)
    return out


def render_drum_notes(notes, sr: int, sec_per_tick: float,
                      volume: float = 1.0, total_samples: int | None = None,
                      mapa=None, automacao=None, pan: float = 0.0):
    """Bateria: cada pitch MIDI vira um preset diferente (mapa GM)."""
    from ..audio.instrument import factory_by_name

    def inicio_s(t):
        return mapa.segundos(t) if mapa is not None else t * sec_per_tick

    def dur_s(t, d):
        return mapa.duracao_de(t, d) if mapa is not None else d * sec_per_tick

    end_tick = max((n.start + n.dur for n in notes), default=0)
    n_total = total_samples or int(inicio_s(end_tick) * sr) + sr
    buf = np.zeros(n_total + sr, dtype=np.float32)
    cache: dict[str, Instrument] = {}
    for n in notes:
        pname = GM_DRUM_MAP.get(n.pitch, "Snare")
        inst = cache.get(pname) or factory_by_name(pname)
        cache[pname] = inst
        pitch = DRUM_BASE_PITCH.get(pname, 60)
        sig = inst.render_note(pitch, max(dur_s(n.start, n.dur), 0.05), sr, n.velocity)
        off = int(inicio_s(n.start) * sr)
        end = min(off + len(sig), len(buf))
        if end > off:
            buf[off:end] += sig[:end - off]
    buf *= volume
    if automacao is not None:
        buf *= automacao.curva(len(buf), mapa, sr, sec_per_tick)
    st = FX.stereo_pan(buf, pan)
    return st[:n_total] if total_samples else st


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
def export_wav(audio: np.ndarray, path: str | Path, sr: int = 44100,
               bit_depth: int = 16) -> Path:
    path = Path(path).with_suffix(".wav")
    a = np.clip(audio, -1.0, 1.0)
    if a.ndim == 1:
        a = a[:, None]
    if bit_depth == 8:
        data = ((a * 127) + 128).astype(np.uint8).tobytes()
        sampwidth = 1
    else:
        data = (a * 32767).astype("<i2").tobytes()
        sampwidth = 2
    with wave.open(str(path), "wb") as w:
        w.setnchannels(a.shape[1])
        w.setsampwidth(sampwidth)
        w.setframerate(sr)
        w.writeframes(data)
    return path


# --------------------------------------------------------------------------
# Playback em tempo real
# --------------------------------------------------------------------------
class RealtimePlayer:
    """Toca um buffer ja renderizado e informa a posicao para o playhead da UI."""

    def __init__(self, sr: int = 44100, blocksize: int = 512):
        self.sr = sr
        self.blocksize = blocksize
        self._buf: np.ndarray | None = None
        self._pos = 0
        self._stream = None
        self.loop = False
        self.on_position = None      # callback(frames_tocados)

    def _callback(self, outdata, frames, time_info, status):
        if self._buf is None:
            outdata[:] = 0
            return
        end = self._pos + frames
        chunk = self._buf[self._pos:end]
        if len(chunk) < frames:
            pad = frames - len(chunk)
            if self.loop:
                chunk = np.vstack([chunk, self._buf[:pad]])
                self._pos = pad
            else:
                chunk = np.vstack([chunk, np.zeros((pad, 2), np.float32)])
                self._pos = len(self._buf)
        else:
            self._pos = end
        outdata[:] = chunk
        if self.on_position:
            self.on_position(self._pos)

    def play(self, audio: np.ndarray, loop: bool = False, start_frame: int = 0):
        import sounddevice as sd
        self._buf = np.ascontiguousarray(audio, dtype=np.float32)
        self._pos = start_frame
        self.loop = loop
        if self._stream is None:
            self._stream = sd.OutputStream(
                samplerate=self.sr, channels=2, dtype="float32",
                blocksize=self.blocksize, callback=self._callback)
            self._stream.start()

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._pos = 0

    def pause(self):
        if self._stream:
            self._stream.stop()

    def resume(self):
        if self._stream:
            self._stream.start()

    @property
    def position_seconds(self) -> float:
        return self._pos / self.sr

    def preview(self, instrument: Instrument, pitch: int, dur: float = 0.5):
        """Toca uma nota so — usado ao clicar no piano roll."""
        sig = instrument.render_note(pitch, dur, self.sr)
        self.play(FX.stereo_pan(sig, instrument.pan))
