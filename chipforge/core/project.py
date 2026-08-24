"""
chipforge.core.project
======================
Modelo de dados do projeto. Tudo aqui e serializavel: um .chip e um ZIP com
project.json dentro (e espaco reservado para samples e wavetables custom).

Unidade de tempo: TICKS. ppq ticks por semininma (padrao 480, igual ao MIDI).
Isso mantem swing, tercinas e quantizacao exatos sem erro de ponto flutuante.
"""
from __future__ import annotations

import json
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..audio.instrument import Instrument
from .harmony import NoteEvent

FORMAT_VERSION = 1


@dataclass
class Note:
    start: int          # ticks
    dur: int            # ticks
    pitch: int          # MIDI 0..127
    velocity: int = 100

    def __post_init__(self):
        """Converte para int de Python. Nao e paranoia: um gerador que use
        numpy devolve int64 sem avisar, e int64 atravessa o render inteiro sem
        reclamar — o erro so aparece na hora de salvar o .chip, porque o JSON
        nao serializa esse tipo. Foi assim que um solo gerado com
        `rng.integers()` derrubou o salvamento de uma peca de seis minutos
        DEPOIS de trinta minutos de render. Converter aqui custa quatro
        chamadas por nota e fecha a familia inteira de defeito."""
        self.start = int(self.start)
        self.dur = int(self.dur)
        self.pitch = int(self.pitch)
        self.velocity = int(self.velocity)

    @property
    def end(self) -> int:
        return self.start + self.dur


@dataclass
class Pattern:
    """Um bloco de notas reutilizavel — o 'clip' do FL Studio."""
    id: str
    name: str = "Pattern 1"
    length_ticks: int = 480 * 16          # 4 compassos em 4/4
    notes: list[Note] = field(default_factory=list)
    color: str = "#4aa3ff"

    def add(self, pitch: int, start: int, dur: int, velocity: int = 100) -> Note:
        n = Note(start, dur, pitch, velocity)
        self.notes.append(n)
        return n

    def remove_at(self, tick: int, pitch: int) -> bool:
        for n in list(self.notes):
            if n.pitch == pitch and n.start <= tick < n.end:
                self.notes.remove(n)
                return True
        return False


@dataclass
class Track:
    id: str
    name: str = "Track"
    instrument: Instrument = field(default_factory=Instrument)
    volume: float = 0.8
    pan: float = 0.0
    mute: bool = False
    solo: bool = False
    # --- canal de mixer -------------------------------------------------
    # inserts: lista de efeitos, na ordem (ver chipforge.audio.fx.aplicar)
    inserts: list = field(default_factory=list)
    # quanto deste canal vai para o reverb da sala (0..1)
    reverb_send: float = 0.0
    # dobra de secao: {"vozes": 4, "detune": 9, "spread_ms": 22, "pan": 0.6}
    # um violino vira naipe de cordas; um cantor vira coro
    section: dict | None = None
    # curvas por parametro, em ticks: {"volume": [[tick, valor], ...]}
    # e o que faz um crescendo de doze compassos existir. Hoje o render le
    # "volume"; qualquer outra chave fica guardada para quem quiser usar.
    automacao: dict = field(default_factory=dict)


@dataclass
class ClipRef:
    """Uma instancia de um pattern na linha do tempo."""
    pattern_id: str
    track_id: str
    start: int          # ticks na timeline


@dataclass
class Project:
    name: str = "Novo projeto"
    bpm: float = 120.0
    ppq: int = 480
    beats_per_bar: int = 4
    beat_unit: int = 4
    sample_rate: int = 44100
    bit_depth: int = 8            # 8 ou 16 — a "cara" do projeto
    master_gain: float = 0.85
    # sala onde a musica toca — None deixa tudo seco
    master_reverb: dict | None = None
    master_inserts: list = field(default_factory=list)
    # perfil da cadeia de master (ver chipforge.audio.master.PERFIS):
    # "cinematico", "trailer", "jogo", "chiptune", "cru" — None mantem o
    # comportamento antigo (so limitador de pico, sem alvo de loudness)
    master_perfil: str | None = None
    # balanco espectral que o master deve perseguir, por banda, em dB.
    # Sai de chipforge.analysis.reference.Referencia.perfil_alvo()
    master_alvo: dict | None = None
    key_root: int = 0             # tonalidade declarada (para Scale Lock)
    scale: str = "major"
    scale_lock: bool = False
    # marcas de andamento [[tick, bpm], ...]. Vazio = `bpm` vale para a peca
    # inteira. Com marcas, accelerando e ritardando passam a existir — e sem
    # eles a musica anda no metronomo do inicio ao fim.
    tempo_map: list = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    patterns: dict[str, Pattern] = field(default_factory=dict)
    arrangement: list[ClipRef] = field(default_factory=list)

    # ---- conversoes de tempo -------------------------------------------
    @property
    def ticks_per_bar(self) -> int:
        return self.ppq * self.beats_per_bar

    @property
    def mapa_de_tempo(self):
        """MapaDeTempo do projeto, reconstruido quando as marcas mudam.

        O cache existe porque o render pergunta o instante de cada nota, uma a
        uma: numa peca de sete mil notas, reconstruir a tabela a cada pergunta
        custaria mais que sintetizar o audio.
        """
        from .dinamica import MapaDeTempo
        chave = (self.ppq, self.bpm, tuple(map(tuple, self.tempo_map)))
        cache = getattr(self, "_mapa_cache", None)
        if cache is None or cache[0] != chave:
            mapa = MapaDeTempo.from_list(self.tempo_map, self.ppq, self.bpm)
            object.__setattr__(self, "_mapa_cache", (chave, mapa))
            return mapa
        return cache[1]

    def ticks_to_seconds(self, ticks: int) -> float:
        if self.tempo_map:
            return self.mapa_de_tempo.segundos(ticks)
        return ticks / self.ppq * (60.0 / self.bpm)

    def seconds_to_ticks(self, s: float) -> int:
        if self.tempo_map:
            mapa = self.mapa_de_tempo
            lo, hi = 0, max(1, self.length_ticks * 2)
            for _ in range(40):                 # busca binaria: o mapa e monotono
                meio = (lo + hi) / 2
                if mapa.segundos(meio) < s:
                    lo = meio
                else:
                    hi = meio
            return int(round((lo + hi) / 2))
        return int(round(s * self.bpm / 60.0 * self.ppq))

    def ticks_to_beats(self, ticks: int) -> float:
        return ticks / self.ppq

    @property
    def length_ticks(self) -> int:
        end = 0
        for c in self.arrangement:
            pat = self.patterns.get(c.pattern_id)
            if pat:
                end = max(end, c.start + pat.length_ticks)
        return end or self.ticks_per_bar * 4

    # ---- helpers de edicao ---------------------------------------------
    def new_track(self, name: str, instrument: Instrument) -> Track:
        tid = f"t{len(self.tracks) + 1}"
        t = Track(tid, name, instrument)
        self.tracks.append(t)
        return t

    def new_pattern(self, name: str | None = None, bars: int = 4) -> Pattern:
        pid = f"p{len(self.patterns) + 1}"
        pat = Pattern(pid, name or f"Pattern {len(self.patterns) + 1}",
                      self.ticks_per_bar * bars)
        self.patterns[pid] = pat
        return pat

    def place(self, pattern: Pattern, track: Track, start_bar: int = 0) -> ClipRef:
        c = ClipRef(pattern.id, track.id, start_bar * self.ticks_per_bar)
        self.arrangement.append(c)
        return c

    def track(self, tid: str) -> Track | None:
        return next((t for t in self.tracks if t.id == tid), None)

    @property
    def audible_tracks(self) -> list[Track]:
        solos = [t for t in self.tracks if t.solo]
        pool = solos or self.tracks
        return [t for t in pool if not t.mute]

    # ---- ponte para o medidor de harmonia -------------------------------
    def harmony_events(self, only_pitched: bool = True) -> list[NoteEvent]:
        """Achata o arranjo inteiro em eventos em beats, para analise."""
        out: list[NoteEvent] = []
        for c in self.arrangement:
            pat = self.patterns.get(c.pattern_id)
            tr = self.track(c.track_id)
            if not pat or not tr:
                continue
            # Bateria nunca entrou no medidor. Desenho de som tambem nao pode:
            # um riser e ruido varrendo o espectro e um impacto e um transiente
            # — a "altura" da nota deles diz QUANDO e que tamanho, nao que nota.
            # Julgados como harmonia, eles apareciam brigando com o coro e
            # derrubavam o score de uma secao inteira por um choque que nao
            # existe.
            if only_pitched and (tr.instrument.is_drum
                                 or getattr(tr.instrument, "osc", "") == "cinema"):
                continue
            # um instrumento com arpejo toca um acorde inteiro a partir de
            # uma nota so — o medidor precisa enxergar as notas que soam
            offsets = tuple(getattr(tr.instrument, "arpeggio", ()) or (0,))
            for n in pat.notes:
                for off in offsets:
                    out.append(NoteEvent(
                        start=(c.start + n.start) / self.ppq,
                        dur=n.dur / self.ppq,
                        pitch=n.pitch + off,
                        velocity=n.velocity,
                        track=tr.name))
        return sorted(out, key=lambda e: e.start)

    def pattern_events(self, pattern: Pattern) -> list[NoteEvent]:
        return sorted((NoteEvent(n.start / self.ppq, n.dur / self.ppq,
                                 n.pitch, n.velocity)
                       for n in pattern.notes), key=lambda e: e.start)

    # ---- serializacao ---------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "format_version": FORMAT_VERSION,
            "name": self.name, "bpm": self.bpm, "ppq": self.ppq,
            "beats_per_bar": self.beats_per_bar, "beat_unit": self.beat_unit,
            "sample_rate": self.sample_rate, "bit_depth": self.bit_depth,
            "master_gain": self.master_gain,
            "master_reverb": self.master_reverb,
            "master_inserts": self.master_inserts,
            "master_perfil": self.master_perfil,
            "master_alvo": self.master_alvo,
            "key_root": self.key_root, "scale": self.scale,
            "scale_lock": self.scale_lock,
            "tempo_map": [[int(t), float(b)] for t, b in self.tempo_map],
            "tracks": [{**asdict(t), "instrument": t.instrument.to_dict()}
                       for t in self.tracks],
            "patterns": {k: asdict(v) for k, v in self.patterns.items()},
            "arrangement": [asdict(c) for c in self.arrangement],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Project":
        p = cls(
            name=d.get("name", "Projeto"), bpm=d.get("bpm", 120.0),
            ppq=d.get("ppq", 480), beats_per_bar=d.get("beats_per_bar", 4),
            beat_unit=d.get("beat_unit", 4),
            sample_rate=d.get("sample_rate", 44100),
            bit_depth=d.get("bit_depth", 8),
            master_gain=d.get("master_gain", 0.85),
            master_reverb=d.get("master_reverb"),
            master_inserts=d.get("master_inserts", []) or [],
            master_perfil=d.get("master_perfil"),
            master_alvo=d.get("master_alvo"),
            key_root=d.get("key_root", 0), scale=d.get("scale", "major"),
            scale_lock=d.get("scale_lock", False),
            tempo_map=[list(m) for m in d.get("tempo_map", []) or []])
        for td in d.get("tracks", []):
            td = dict(td)
            idict = td.pop("instrument")
            kind = idict.get("kind")
            if kind == "sampler":
                from ..audio.sampler import SamplerInstrument
                try:
                    inst = SamplerInstrument.from_dict(idict)
                except Exception as e:
                    # o banco pode nao estar nesta maquina: nao perde o projeto
                    print(f"[chipforge] banco indisponivel ({e}); "
                          f"usando instrumento chiptune no lugar")
                    inst = Instrument(name=idict.get("name", "Sample"))
            elif kind == "voz":
                from ..audio.voz import VozInstrument
                inst = VozInstrument.from_dict(idict)
            elif kind == "corda":
                from ..audio.corda import CordaInstrument
                inst = CordaInstrument.from_dict(idict)
            elif kind == "cinema":
                from ..audio.cinema import _Base
                inst = _Base.from_dict(idict)
            else:
                inst = Instrument.from_dict(idict)
            p.tracks.append(Track(instrument=inst, **td))
        for k, pd in d.get("patterns", {}).items():
            pd = dict(pd)
            notes = [Note(**n) for n in pd.pop("notes", [])]
            p.patterns[k] = Pattern(notes=notes, **pd)
        p.arrangement = [ClipRef(**c) for c in d.get("arrangement", [])]
        return p

    def save(self, path: str | Path) -> Path:
        path = Path(path).with_suffix(".chip")
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("project.json",
                       json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
            z.writestr("META-INF", f"chipforge {FORMAT_VERSION}")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        with zipfile.ZipFile(path) as z:
            return cls.from_dict(json.loads(z.read("project.json")))
