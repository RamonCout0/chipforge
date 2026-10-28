"""
chipforge.estudio
=================
O Chipforge sem interface: compor, mixar, ouvir e medir por chamadas de
funcao. E a camada que o servidor MCP expoe (chipforge.mcp_server), mas
funciona igual num script ou no terminal:

    from chipforge.estudio import Estudio
    e = Estudio()
    e.novo("Fase de gelo", bpm=150, tom="E", escala="natural_minor", console="ps1")
    e.criar_track("Lead", "ps1:lead_quadrado")
    e.escrever("Lead", 1, "o5 l8 b > e f+ g f+4 e d | e2. r4 |")
    print(e.renderizar("gelo.wav"))

Convencoes pensadas para quem compoe em texto:
  * compassos sao contados a partir de 1, como na partitura;
  * tracks sao chamadas pelo nome (sem diferenciar maiusculas);
  * toda funcao devolve um texto curto dizendo o que mudou e o que conferir.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .audio import engine
from .audio.instrument import FACTORY_BANK, Instrument, factory_by_name
from .core import harmony as H
from .core import mml as MML
from .core import theory as T
from .core.project import ClipRef, Note, Project

# ==========================================================================
# Paleta "PS1": timbres da era 32 bits, ja com a taxa de sample tipica
# ==========================================================================
# nome -> (busca no banco GM, banco, programa, ajustes)
PALETA_PS1: dict[str, tuple] = {
    "lead_quadrado": ("Square Lead", 0, 80, dict(gain=0.55, psx_taxa=22050)),
    "lead_serra": ("Saw Lead", 0, 81, dict(gain=0.5, psx_taxa=22050)),
    "lead_calliope": ("Synth Calliope", 0, 82, dict(gain=0.6)),
    "lead_quinta": ("5th Saw Wave", 0, 86, dict(gain=0.45)),
    "sino": ("Glockenspiel", 0, 9, dict(gain=0.55)),
    "celesta": ("Celeste", 0, 8, dict(gain=0.7)),
    "caixa_de_musica": ("Music Box", 0, 10, dict(gain=0.65)),
    "vibrafone": ("Vibraphone", 0, 11, dict(gain=0.65)),
    "marimba": ("Marimba", 0, 12, dict(gain=0.7)),
    "cristal": ("Crystal", 0, 98, dict(gain=0.6, psx_taxa=16000)),
    "chuva_de_gelo": ("Ice Rain", 0, 96, dict(gain=0.55, psx_taxa=16000)),
    "epiano": ("Tine Electric Piano", 0, 4, dict(gain=0.7)),
    "epiano_fm": ("FM Electric Piano", 0, 5, dict(gain=0.7)),
    "piano": ("Stereo Grand", 0, 0, dict(gain=0.75)),
    "orgao": ("Tonewheel Organ", 0, 16, dict(gain=0.5)),
    "baixo_slap": ("Slap Bass 1", 0, 36, dict(gain=0.85, psx_taxa=11025)),
    "baixo_finger": ("Finger Bass", 0, 33, dict(gain=0.9, psx_taxa=11025)),
    "baixo_synth": ("Synth Bass 1", 0, 38, dict(gain=0.75, psx_taxa=11025)),
    "baixo_synth2": ("Synth Bass 2", 0, 39, dict(gain=0.75, psx_taxa=11025)),
    "brass_synth": ("Synth Brass 1", 0, 62, dict(gain=0.55)),
    "brass": ("Brass Section", 0, 61, dict(gain=0.6)),
    "trompete": ("Trumpet", 0, 56, dict(gain=0.6)),
    "cordas": ("Stereo Strings Fast", 0, 48, dict(gain=0.6, psx_taxa=16000)),
    "cordas_synth": ("Synth Strings 1", 0, 50, dict(gain=0.55, psx_taxa=16000)),
    "pizzicato": ("Pizzicato Strings", 0, 45, dict(gain=0.75)),
    "pad_quente": ("Warm Pad", 0, 89, dict(gain=0.45, psx_taxa=16000)),
    "pad_halo": ("Halo Pad", 0, 94, dict(gain=0.45, psx_taxa=16000)),
    "pad_vidro": ("Bowed Glass", 0, 92, dict(gain=0.5, psx_taxa=16000)),
    "atmosfera": ("Atmosphere", 0, 99, dict(gain=0.5, psx_taxa=16000)),
    "coro": ("Voice Oohs", 0, 53, dict(gain=0.6, psx_taxa=16000)),
    "voz_synth": ("Synth Voice", 0, 54, dict(gain=0.55, psx_taxa=16000)),
    "guitarra_dist": ("Distortion Guitar", 0, 30, dict(gain=0.5)),
    "guitarra_over": ("Overdrive Guitar", 0, 29, dict(gain=0.5)),
    "guitarra_limpa": ("Clean Guitar", 0, 27, dict(gain=0.65)),
    "guitarra_abafada": ("Muted Guitar", 0, 28, dict(gain=0.7)),
    "flauta": ("Flute", 0, 73, dict(gain=0.6)),
    "ocarina": ("Ocarina", 0, 79, dict(gain=0.6)),
    "orquestra_hit": ("Orchestra Hit", 0, 55, dict(gain=0.6)),
    "bateria": ("Standard", 128, 0, dict(gain=0.9, psx_taxa=22050)),
    "bateria_power": ("Power", 128, 16, dict(gain=0.9)),
    "bateria_eletronica": ("Electronic", 128, 24, dict(gain=0.9)),
    "bateria_808": ("808/909", 128, 25, dict(gain=0.9)),
    "bateria_sala": ("Room", 128, 8, dict(gain=0.9)),
}

CONSOLES = {"nes": (8, ""), "snes": (16, ""), "limpo": (24, ""), "ps1": (24, "ps1")}

ESTILOS_ACOMPANHAMENTO = {
    "bloco": "acorde segurado o compasso todo (voicing com conducao suave)",
    "pad": "igual a bloco, mais aberto e legato — para pads e cordas",
    "stabs": "acorde curto nos contratempos (o 'tchá' de metais e guitarra)",
    "galope": "acorde em colcheias com acento no 1 e no 3 (rock de fase)",
    "arpejo8": "arpejo subindo em colcheias",
    "arpejo16": "arpejo sobe-e-desce em semicolcheias (energia de fase)",
    "baixo": "fundamental nos tempos 1 e 3, quinta no 4",
    "baixo8": "fundamental em colcheias (motor de rock)",
    "baixo_oitava": "fundamental alternando oitava em colcheias (disco/MM)",
    "baixo_funk": "fundamental + oitava sincopados (slap de PS1)",
    "baixo_longo": "fundamental segurada o compasso todo",
}


class ErroEstudio(ValueError):
    pass


class Estudio:
    def __init__(self, projeto: Project | None = None):
        self.projeto = projeto or Project("Sem titulo", bpm=120, bit_depth=24)
        self.caminho: Path | None = None
        self._audio_cache: tuple[str, np.ndarray] | None = None
        self._player = None

    # ------------------------------------------------------------------
    # util
    # ------------------------------------------------------------------
    @property
    def p(self) -> Project:
        return self.projeto

    def _barra(self) -> int:
        return self.p.ticks_per_bar

    def _track(self, nome: str):
        alvo = (nome or "").strip().lower()
        for t in self.p.tracks:
            if t.name.lower() == alvo or t.id == alvo:
                return t
        nomes = ", ".join(t.name for t in self.p.tracks) or "(nenhuma)"
        raise ErroEstudio(f"Track '{nome}' nao existe. Tracks: {nomes}")

    def _nome_nota(self, p: int) -> str:
        return T.midi_to_name(p)

    def _fora_da_escala(self, notas) -> list[int]:
        pcs = T.scale_pitch_classes(self.p.key_root, self.p.scale)
        return sorted({n.pitch for n in notas if n.pitch % 12 not in pcs})

    @staticmethod
    def _tom_para_pc(tom: str | int) -> int:
        if isinstance(tom, int):
            return tom % 12
        s = tom.strip()
        if s.isdigit():
            return int(s) % 12
        return T.name_to_midi(s[0].upper() + s[1:] + "4") % 12

    # ------------------------------------------------------------------
    # projeto
    # ------------------------------------------------------------------
    def novo(self, nome: str = "Sem titulo", bpm: float = 120, tom: str = "C",
             escala: str = "major", compasso: str = "4/4",
             console: str = "ps1") -> str:
        if escala not in T.SCALES:
            raise ErroEstudio(f"Escala '{escala}' desconhecida. Opcoes: "
                              + ", ".join(T.SCALES))
        if console not in CONSOLES:
            raise ErroEstudio(f"Console '{console}'. Opcoes: {', '.join(CONSOLES)}")
        num, den = (int(x) for x in compasso.split("/"))
        bits, cons = CONSOLES[console]
        self.projeto = Project(nome, bpm=float(bpm), beats_per_bar=num,
                               beat_unit=den, bit_depth=bits)
        self.p.console = cons
        self.p.key_root = self._tom_para_pc(tom)
        self.p.scale = escala
        if console == "ps1":
            self.p.master_reverb = {"psx": "studio_large", "wet": 0.55}
        self.caminho = None
        self._audio_cache = None
        return (f"Projeto '{nome}' criado: {bpm:g} BPM, {compasso}, "
                f"{T.SHARP_NAMES[self.p.key_root]} {T.SCALE_LABELS_PT.get(escala, escala)}, "
                f"console {console}."
                + (" Reverb do SPU 'studio_large' ligado no barramento "
                   "(use reverb_send nas tracks)." if console == "ps1" else ""))

    def abrir(self, caminho: str) -> str:
        self.projeto = Project.load(caminho)
        self.caminho = Path(caminho)
        self._audio_cache = None
        av = ("\nAvisos:\n" + "\n".join(self.p.avisos)) if self.p.avisos else ""
        return "Aberto.\n" + self.resumo() + av

    def salvar(self, caminho: str | None = None) -> str:
        alvo = caminho or self.caminho
        if not alvo:
            raise ErroEstudio("Informe o caminho do .chip na primeira vez.")
        self.caminho = self.p.save(alvo)
        return f"Salvo em {self.caminho}"

    def configurar(self, bpm: float | None = None, tom: str | None = None,
                   escala: str | None = None, console: str | None = None,
                   nome: str | None = None, tempo_map: list | None = None) -> str:
        if bpm:
            self.p.bpm = float(bpm)
        if tom is not None:
            self.p.key_root = self._tom_para_pc(tom)
        if escala:
            if escala not in T.SCALES:
                raise ErroEstudio(f"Escala desconhecida: {escala}")
            self.p.scale = escala
        if console:
            bits, cons = CONSOLES[console]
            self.p.bit_depth, self.p.console = bits, cons
        if nome:
            self.p.name = nome
        if tempo_map is not None:
            # [[compasso, bpm], ...] -> ticks
            self.p.tempo_map = [[int((c - 1) * self._barra()), float(b)]
                                for c, b in tempo_map]
        return self.resumo()

    def resumo(self) -> str:
        p = self.p
        comp = math.ceil(p.length_ticks / self._barra()) if p.arrangement else 0
        cons = p.console or {8: "nes", 16: "snes"}.get(p.bit_depth, "limpo")
        L = [f"'{p.name}' — {p.bpm:g} BPM, {p.beats_per_bar}/{p.beat_unit}, "
             f"{T.SHARP_NAMES[p.key_root]} {T.SCALE_LABELS_PT.get(p.scale, p.scale)}, "
             f"console {cons}, {comp} compassos "
             f"(~{p.ticks_to_seconds(p.length_ticks):.0f}s)"]
        if p.master_reverb:
            L.append(f"Sala: {p.master_reverb}")
        if p.master_perfil:
            L.append(f"Master: {p.master_perfil}")
        for t in p.tracks:
            clips = sorted((c for c in p.arrangement if c.track_id == t.id),
                           key=lambda c: c.start)
            ocup = []
            for c in clips:
                pat = p.patterns.get(c.pattern_id)
                a = c.start // self._barra() + 1
                b = (c.start + (pat.length_ticks if pat else 0)) // self._barra()
                ocup.append(f"{a}-{b}" if b > a else f"{a}")
            inst = t.instrument
            nome_inst = getattr(inst, "name", "?")
            extra = []
            if t.reverb_send:
                extra.append(f"rev {t.reverb_send:.2f}")
            if t.section:
                extra.append(f"naipe x{t.section.get('vozes')}")
            if t.mute:
                extra.append("MUDO")
            L.append(f"  • {t.name} [{nome_inst}] vol {t.volume:.2f} pan {t.pan:+.2f}"
                     f"{' ' + ', '.join(extra) if extra else ''} — compassos: "
                     f"{', '.join(ocup) or '(vazia)'}")
        return "\n".join(L)

    # ------------------------------------------------------------------
    # instrumentos
    # ------------------------------------------------------------------
    def _banco_gm(self):
        from .audio.sampler import open_soundfont
        from .midiio import library as L
        sf2 = L.ensure("generaluser_gs")
        if not sf2:
            raise ErroEstudio("Banco GeneralUser GS indisponivel (sem rede?). "
                              "Use instrumentos 'chip:'.")
        return open_soundfont(sf2)

    def _instrumento(self, spec: str):
        """'ps1:baixo_slap' | 'gm:Slap Bass 1' | 'chip:NES Lead 25%' |
        'voz:Coro soprano (aah)' | 'cinema:Impacto grave' | 'sf2:/c/x.sf2:Piano'"""
        from .audio import sampler as S
        if ":" not in spec:
            spec = ("ps1:" if spec in PALETA_PS1 else "chip:") + spec
        fonte, nome = spec.split(":", 1)
        fonte = fonte.lower()
        if fonte == "ps1":
            if nome not in PALETA_PS1:
                raise ErroEstudio(f"'{nome}' nao esta na paleta PS1. Opcoes: "
                                  + ", ".join(PALETA_PS1))
            busca, banco, prog, ajustes = PALETA_PS1[nome]
            sf = self._banco_gm()
            pre = S.find_preset(sf, banco, prog, busca)
            inst = S.SamplerInstrument(sf, pre, **ajustes)
            inst.name = nome
            return inst
        if fonte == "gm":
            sf = self._banco_gm()
            hits = sf.find(nome)
            if not hits:
                raise ErroEstudio(f"Nenhum preset GM com '{nome}'.")
            return S.SamplerInstrument(sf, hits[0])
        if fonte == "sf2":
            caminho, busca = nome.rsplit(":", 1)
            return S.instrument(caminho, busca)
        if fonte == "chip":
            inst = factory_by_name(nome)
            if inst is None:
                hits = [n for g in FACTORY_BANK.values() for n in
                        (i.name for i in g) if nome.lower() in n.lower()]
                if not hits:
                    raise ErroEstudio(f"Preset chip '{nome}' nao existe.")
                inst = factory_by_name(hits[0])
            return inst
        if fonte == "corda":
            from .audio import corda
            if nome not in corda.PRESETS_CORDA:
                raise ErroEstudio("Cordas: " + ", ".join(corda.PRESETS_CORDA))
            return corda.corda(nome)
        if fonte == "voz":
            from .audio import voz
            return voz.voz(nome)
        if fonte == "cinema":
            from .audio import cinema
            return cinema.instrumento(nome)
        raise ErroEstudio(f"Fonte '{fonte}' desconhecida (ps1, gm, chip, corda, "
                          "voz, cinema, sf2).")

    ALVO_RMS_DB = -20.0

    def _calibrar(self, inst) -> float:
        """Iguala o volume dos instrumentos: uma nota tipica de cada um sai a
        -20 dBFS RMS. Sem isso o sino da paleta saia 25 dB abaixo do baixo
        synth, e 'volume 0.8' significava uma coisa diferente em cada track —
        nenhuma mixagem fica de pe assim."""
        if not hasattr(inst, "gain") or getattr(inst, "osc", "") == "cinema":
            return 0.0
        nome = (getattr(inst, "name", "") or "").lower()
        if getattr(inst, "is_drum", False):
            if getattr(inst, "osc", "") != "sample":
                return 0.0                         # bateria chip: por preset
            notas = (36, 38, 42)
        elif any(k in nome for k in ("baixo", "bass")):
            notas = (36, 43)
        else:
            notas = (60, 67)
        sr = 22050
        try:
            sig = np.concatenate([inst.render_note(p, 0.5, sr, 100) for p in notas])
        except Exception:                          # pragma: no cover
            return 0.0
        rms = float(np.sqrt(np.mean(np.square(sig[: int(0.5 * sr) * len(notas)]))))
        if rms < 1e-6:
            return 0.0
        ganho_db = self.ALVO_RMS_DB - 20 * np.log10(rms)
        ganho_db = float(np.clip(ganho_db, -24.0, 30.0))
        inst.gain = float(inst.gain * 10 ** (ganho_db / 20))
        return ganho_db

    def listar_instrumentos(self, fonte: str = "ps1", busca: str = "") -> str:
        b = busca.lower()
        if fonte == "ps1":
            return "\n".join(f"ps1:{k}  ({v[0]})" for k, v in PALETA_PS1.items()
                             if b in k or b in v[0].lower())
        if fonte == "chip":
            return "\n".join(f"chip:{i.name}  [{g}]" for g, l in FACTORY_BANK.items()
                             for i in l if b in i.name.lower())
        if fonte == "gm":
            sf = self._banco_gm()
            return "\n".join(f"gm:{p.name}  (banco {p.bank}, prog {p.program})"
                             for p in sorted(sf.presets, key=lambda p: (p.bank, p.program))
                             if b in p.name.lower())
        if fonte == "corda":
            from .audio import corda
            return "\n".join(f"corda:{n}" for n in corda.nomes_de_preset()) + \
                "\n(guitarra sai LIMPA: ponha um amp com ajustar_track, " \
                "efeitos=[{'tipo':'amp','preset':'metal'}])"
        if fonte == "voz":
            from .audio import voz
            return "\n".join(f"voz:{n}" for n in voz.nomes_de_preset())
        if fonte == "cinema":
            from .audio import cinema
            return "\n".join(f"cinema:{n}" for n in cinema.nomes_de_preset())
        raise ErroEstudio("fonte: ps1, chip, gm, corda, voz ou cinema")

    def criar_track(self, nome: str, instrumento: str, volume: float = 0.8,
                    pan: float = 0.0, reverb_send: float = 0.25) -> str:
        if any(t.name.lower() == nome.lower() for t in self.p.tracks):
            raise ErroEstudio(f"Ja existe uma track '{nome}'.")
        inst = self._instrumento(instrumento)
        self._calibrar(inst)
        t = self.p.new_track(nome, inst)
        t.volume, t.pan, t.reverb_send = float(volume), float(pan), float(reverb_send)
        self._audio_cache = None
        return (f"Track '{nome}' criada com {getattr(inst, 'name', instrumento)}"
                f"{' (bateria: use a grade)' if getattr(inst, 'is_drum', False) else ''}.")

    def ajustar_track(self, nome: str, volume: float | None = None,
                      pan: float | None = None, reverb_send: float | None = None,
                      instrumento: str | None = None, mudo: bool | None = None,
                      solo: bool | None = None, naipe: int | None = None,
                      efeitos: list | None = None,
                      automacao_volume: list | None = None) -> str:
        t = self._track(nome)
        if volume is not None:
            t.volume = float(volume)
        if pan is not None:
            t.pan = float(np.clip(pan, -1, 1))
        if reverb_send is not None:
            t.reverb_send = float(reverb_send)
        if instrumento:
            t.instrument = self._instrumento(instrumento)
            self._calibrar(t.instrument)
        if mudo is not None:
            t.mute = bool(mudo)
        if solo is not None:
            t.solo = bool(solo)
        if naipe is not None:
            t.section = None if naipe < 2 else {
                "vozes": int(naipe), "detune": 8.0, "spread_ms": 18.0, "pan": 0.5}
        if efeitos is not None:
            from .audio import fx
            lista = []
            for spec in efeitos:
                spec = dict(spec)
                if spec.get("tipo") == "amp" and "preset" in spec:
                    base = dict(fx.PRESETS_AMP.get(spec.pop("preset"), {}))
                    base.update(spec)
                    spec = base
                lista.append(spec)
            t.inserts = lista
        if automacao_volume is not None:
            # [[compasso, valor], ...] — compasso pode ser fracionario
            t.automacao = dict(t.automacao or {})
            t.automacao["volume"] = [[int((c - 1) * self._barra()), float(v)]
                                     for c, v in automacao_volume]
        self._audio_cache = None
        return f"Track '{t.name}' ajustada: vol {t.volume:.2f}, pan {t.pan:+.2f}, " \
               f"rev {t.reverb_send:.2f}{', naipe x' + str(t.section['vozes']) if t.section else ''}" \
               f"{', MUDO' if t.mute else ''}{', SOLO' if t.solo else ''}."

    # ------------------------------------------------------------------
    # escrita
    # ------------------------------------------------------------------
    def _colocar(self, track, compasso: int, notas: list[Note], fim: int,
                 nome: str | None, substituir: bool) -> tuple[int, int]:
        barra = self._barra()
        compassos = max(1, math.ceil(fim / barra))
        inicio = (int(compasso) - 1) * barra
        if inicio < 0:
            raise ErroEstudio("compasso comeca em 1")
        if substituir:
            self._limpar(track, inicio, inicio + compassos * barra)
        pat = self.p.new_pattern(nome or f"{track.name} c{compasso}", bars=compassos)
        pat.notes = list(notas)
        self.p.arrangement.append(ClipRef(pat.id, track.id, inicio))
        self._audio_cache = None
        return compassos, inicio

    def _limpar(self, track, a: int, b: int) -> int:
        """Remove/corta o que a track tem entre os ticks a e b."""
        removidas = 0
        novos = []
        for c in self.p.arrangement:
            pat = self.p.patterns.get(c.pattern_id)
            if c.track_id != track.id or not pat:
                novos.append(c)
                continue
            c0, c1 = c.start, c.start + pat.length_ticks
            if c1 <= a or c0 >= b:
                novos.append(c)
                continue
            # sobreposicao: mantem so as notas fora de [a, b) num pattern novo
            fica = [n for n in pat.notes
                    if not (a <= c.start + n.start < b)]
            removidas += len(pat.notes) - len(fica)
            if fica:
                np_ = self.p.new_pattern(pat.name, bars=1)
                np_.length_ticks = pat.length_ticks
                np_.notes = [Note(n.start, min(n.dur, max(1, a - c.start - n.start))
                                  if c.start + n.start < a else n.dur,
                                  n.pitch, n.velocity) for n in fica]
                novos.append(ClipRef(np_.id, c.track_id, c.start))
        self.p.arrangement = novos
        usados = {c.pattern_id for c in self.p.arrangement}
        for k in [k for k in self.p.patterns if k not in usados]:
            del self.p.patterns[k]
        return removidas

    def _relatorio_escrita(self, track, compasso, compassos, notas, avisos) -> str:
        L = [f"'{track.name}': {len(notas)} notas nos compassos "
             f"{compasso}-{compasso + compassos - 1}."]
        if notas and not getattr(track.instrument, "is_drum", False):
            lo, hi = min(n.pitch for n in notas), max(n.pitch for n in notas)
            L.append(f"Extensao: {self._nome_nota(lo)} a {self._nome_nota(hi)}.")
            fora = self._fora_da_escala(notas)
            if fora:
                L.append("Fora da escala do projeto: "
                         + " ".join(self._nome_nota(x) for x in fora)
                         + " (ok se for cromatismo intencional).")
        for a in avisos:
            L.append("AVISO: " + a)
        return "\n".join(L)

    def _humano(self, notas, ms: float, seed: int, tendencia_ms: float = 0.0):
        if ms <= 0 and not tendencia_ms:
            return notas
        from .core import expressao as EX
        return EX.humanizar(notas, tempo_ms=ms, velocity=max(4, int(ms * 0.8)),
                            ppq=self.p.ppq, bpm=self.p.bpm, seed=seed,
                            tendencia_ms=tendencia_ms)

    def escrever(self, track: str, compasso: int, mml: str,
                 nome: str | None = None, substituir: bool = True,
                 oitava: int = 4, velocity: int = 96, humano: float = 0.0,
                 tendencia_ms: float = 0.0) -> str:
        """Escreve MML na track a partir do compasso (1 = primeiro).
        `humano`: espalhamento de tempo em ms (6-12 soa tocado; 0 = grade)."""
        tr = self._track(track)
        r = MML.parse(mml, self.p.ppq, self.p.beats_per_bar, oitava=oitava,
                      velocity=velocity)
        notas = self._humano(r.notas, humano, compasso * 31 + len(r.notas),
                             tendencia_ms)
        compassos, _ = self._colocar(tr, compasso, notas, r.fim, nome, substituir)
        return self._relatorio_escrita(tr, compasso, compassos, r.notas, r.avisos)

    def bateria(self, track: str, compasso: int, grade: str, repetir: int = 1,
                passos_por_tempo: int = 4, substituir: bool = True,
                humano: float = 0.0, tendencia_ms: float = 0.0) -> str:
        tr = self._track(track)
        notas, fim = MML.parse_bateria(grade, self.p.ppq, self.p.beats_per_bar,
                                       passos_por_tempo)
        tudo = []
        for k in range(max(1, repetir)):
            tudo += [Note(n.start + k * fim, n.dur, n.pitch, n.velocity) for n in notas]
        tudo = self._humano(tudo, humano, compasso * 17 + len(tudo), tendencia_ms)
        compassos, _ = self._colocar(tr, compasso, tudo, fim * max(1, repetir),
                                     None, substituir)
        avisos = []
        if fim % self._barra():
            avisos.append(f"a grade tem {fim / self.p.ppq:g} tempos — nao fecha "
                          "compasso inteiro")
        return self._relatorio_escrita(tr, compasso, compassos, tudo, avisos)

    def _voicing(self, cif: MML.Cifra, centro: int, anterior: list[int] | None,
                 n_vozes: int = 4, teto: int | None = None) -> list[int]:
        """Voicing de tecladista: escolhe QUAIS notas tocar e ONDE.

        1. Com menos vozes que notas no acorde, as tensoes certas ficam: terca e
           setima definem o acorde; a fundamental o baixo ja toca; a quinta e a
           primeira a sair; a nona entra se sobrar voz.
        2. Nada de segunda apertada entre vozes vizinhas (Mi-Fa#-Sol empilhado
           e o 'acorde embolado' que o medidor acusa). Vizinhas ficam a pelo
           menos uma terca menor.
        3. Entre os voicings validos, o que menos se move desde o anterior.
        """
        import itertools
        ints = sorted({i % 12 for i in cif.intervalos})
        prioridade = []
        for alvo in ((3, 4, 2, 5), (10, 11, 9), (0,), (1, 2, 3), (7, 6, 8), (5, 6)):
            for i in alvo:
                if i in ints and i not in prioridade:
                    prioridade.append(i)
                    break
        prioridade += [i for i in ints if i not in prioridade]
        # acompanhamento mora ABAIXO da melodia: o teto fica uma quinta acima
        # do centro (Re5 para centro Sol4) — o lead canta por cima disso
        lo, hi = centro - 9, centro + 7
        if teto is not None:
            hi = min(hi, teto)
            lo = min(lo, teto - 14)
        permite_b9 = 1 in ints
        faixas = [(lo, hi), (lo - 5, hi)] if teto is not None else \
            [(lo, hi), (lo, hi + 5), (lo, hi + 12)]
        validos: list[list[int]] = []
        # Sem segunda entre vozes vizinhas e sem 9a menor entre quaisquer duas
        # (o Si embaixo do Do num Cmaj7 e a "nota a evitar" classica). Se nao
        # couber, primeiro abre a faixa; depois tira a voz menos importante —
        # melhor 3 vozes limpas que 4 emboladas.
        for nv in range(max(3, n_vozes), 1, -1):
            pcs = [(cif.raiz + i) % 12 for i in prioridade[:nv]]
            for a_lo, a_hi in faixas:
                opcoes = [[p for p in range(a_lo, a_hi + 1) if p % 12 == pc]
                          for pc in pcs]
                for combo in itertools.product(*opcoes):
                    ac = sorted(combo)
                    if len(set(ac)) < len(ac):
                        continue
                    if min(y - x for x, y in zip(ac, ac[1:])) < 3 \
                            or ac[-1] - ac[0] > 19:
                        continue
                    if not permite_b9 and any(
                            (y - x) % 12 == 1
                            for x, y in itertools.combinations(ac, 2)):
                        continue
                    validos.append(ac)
                if validos:
                    break
            if validos:
                break
        if not validos:                      # acorde estranho: cai no fechado
            pcs = [(cif.raiz + i) % 12 for i in prioridade[:max(3, n_vozes)]]
            base = (centro // 12) * 12
            return sorted(base + pc for pc in pcs)

        def custo(ac):
            c = 0.35 * abs(sum(ac) / len(ac) - centro)
            if anterior:
                c += sum(min(abs(x - y) for y in anterior) for x in ac)
            return c
        return min(validos, key=custo)

    def acompanhar(self, track: str, compasso: int, cifras: str,
                   estilo: str = "bloco", oitava: int = 4, velocity: int = 90,
                   substituir: bool = True, vozes: int = 0,
                   teto: str | None = None, humano: float = 0.0) -> str:
        if estilo not in ESTILOS_ACOMPANHAMENTO:
            raise ErroEstudio("Estilos: " + "; ".join(
                f"{k} = {v}" for k, v in ESTILOS_ACOMPANHAMENTO.items()))
        tr = self._track(track)
        ppq = self.p.ppq
        lista = MML.parse_cifras(cifras, ppq, self.p.beats_per_bar)
        teto_midi = T.name_to_midi(teto) if teto else None
        notas: list[Note] = []
        anterior = None
        centro = (oitava + 1) * 12 + 7
        e8, e16 = ppq // 2, ppq // 4
        for ini, dur, cif in lista:
            raiz_b = (oitava + 1) * 12 + cif.baixo
            quinta = raiz_b + 7
            if estilo.startswith("baixo"):
                if estilo == "baixo_longo":
                    notas.append(Note(ini, dur - 20, raiz_b, velocity))
                elif estilo == "baixo8":
                    for k in range(dur // e8):
                        notas.append(Note(ini + k * e8, e8 - 30, raiz_b,
                                          velocity + (8 if k % 2 == 0 else 0)))
                elif estilo == "baixo_oitava":
                    for k in range(dur // e8):
                        notas.append(Note(ini + k * e8, e8 - 30,
                                          raiz_b + (12 if k % 2 else 0), velocity))
                elif estilo == "baixo_funk":
                    # 1 . . 1' . 1 8 . | posicoes em semicolcheias
                    padrao = [(0, 0, 3), (3, 12, 1), (6, 0, 1), (8, 0, 2),
                              (10, 12, 1), (11, 0, 1), (14, 12, 2)]
                    for k in range(max(1, dur // (16 * e16))):
                        for pos, sobe, ln in padrao:
                            t0 = ini + k * 16 * e16 + pos * e16
                            if t0 < ini + dur:
                                notas.append(Note(t0, ln * e16 - 20, raiz_b + sobe,
                                                  velocity + (10 if sobe else 0)))
                    if dur < 16 * e16:
                        for pos, sobe, ln in padrao:
                            t0 = ini + pos * e16
                            if t0 < ini + dur:
                                notas.append(Note(t0, ln * e16 - 20, raiz_b + sobe,
                                                  velocity))
                else:  # baixo
                    tempos = dur // ppq
                    for k in range(tempos):
                        if k % 2 == 0:
                            notas.append(Note(ini + k * ppq, ppq * (2 if k + 1 < tempos else 1) - 30,
                                              raiz_b, velocity))
                        elif k == 3:
                            notas.append(Note(ini + k * ppq, ppq - 30, quinta, velocity - 8))
                continue
            ac = self._voicing(cif, centro, anterior,
                               n_vozes=vozes or (4 if estilo in ("pad", "bloco") else 3),
                               teto=teto_midi)
            anterior = ac
            if estilo in ("bloco", "pad"):
                for x in ac:
                    notas.append(Note(ini, dur - (10 if estilo == "pad" else 40),
                                      x, velocity))
            elif estilo == "stabs":
                for k in range(dur // ppq):
                    for x in ac:
                        notas.append(Note(ini + k * ppq + e8, e16, x, velocity))
            elif estilo == "galope":
                for k in range(dur // e8):
                    v = velocity + (12 if k % 4 == 0 else 0)
                    for x in ac:
                        notas.append(Note(ini + k * e8, e8 - 60, x, v))
            elif estilo in ("arpejo8", "arpejo16"):
                passo = e8 if estilo == "arpejo8" else e16
                # arpejo8: sobe e recomeca; arpejo16: sobe e desce (1 3 5 8 5 3)
                seq = ac + [ac[0] + 12]
                if estilo == "arpejo16":
                    seq = seq + ac[:0:-1]
                for k in range(dur // passo):
                    notas.append(Note(ini + k * passo, passo - 15,
                                      seq[k % len(seq)],
                                      velocity + (10 if k % 4 == 0 else 0)))
        fim = max((i + d for i, d, _ in lista), default=self._barra())
        notas = self._humano(notas, humano, compasso * 13 + len(notas))
        compassos, _ = self._colocar(tr, compasso, notas, fim, None, substituir)
        return self._relatorio_escrita(tr, compasso, compassos, notas, [])

    def apagar(self, track: str, de_compasso: int, ate_compasso: int) -> str:
        tr = self._track(track)
        a = (de_compasso - 1) * self._barra()
        b = ate_compasso * self._barra()
        n = self._limpar(tr, a, b)
        self._audio_cache = None
        return f"'{tr.name}': {n} notas apagadas dos compassos {de_compasso}-{ate_compasso}."

    def copiar(self, de_compasso: int, ate_compasso: int, destino: int,
               tracks: list[str] | None = None, vezes: int = 1) -> str:
        """Copia um trecho (todas as tracks ou algumas) para outro compasso."""
        barra = self._barra()
        a, b = (de_compasso - 1) * barra, ate_compasso * barra
        largura = b - a
        alvos = [self._track(n) for n in tracks] if tracks else list(self.p.tracks)
        total = 0
        for tr in alvos:
            notas = []
            for c in self.p.arrangement:
                pat = self.p.patterns.get(c.pattern_id)
                if c.track_id != tr.id or not pat:
                    continue
                for n in pat.notes:
                    t0 = c.start + n.start
                    if a <= t0 < b:
                        notas.append(Note(t0 - a, n.dur, n.pitch, n.velocity))
            if not notas:
                continue
            for k in range(max(1, vezes)):
                self._colocar(tr, destino + k * (ate_compasso - de_compasso + 1),
                              [Note(n.start, n.dur, n.pitch, n.velocity) for n in notas],
                              largura, f"{tr.name} copia", True)
                total += len(notas)
        return (f"Compassos {de_compasso}-{ate_compasso} copiados para {destino}"
                f"{f' ({vezes}x)' if vezes > 1 else ''}: {total} notas.")

    def ver(self, track: str, de_compasso: int = 1, ate_compasso: int | None = None) -> str:
        """Mostra em MML o que a track tem no trecho — para revisar."""
        tr = self._track(track)
        barra = self._barra()
        a = (de_compasso - 1) * barra
        b = (ate_compasso * barra) if ate_compasso else 10 ** 12
        notas = []
        for c in self.p.arrangement:
            pat = self.p.patterns.get(c.pattern_id)
            if c.track_id != tr.id or not pat:
                continue
            for n in pat.notes:
                t0 = c.start + n.start
                if a <= t0 < b:
                    notas.append(Note(t0 - a, n.dur, n.pitch, n.velocity))
        if not notas:
            return f"'{tr.name}' vazia nesse trecho."
        if getattr(tr.instrument, "is_drum", False):
            return self._grade_de(notas)
        return MML.para_mml(sorted(notas, key=lambda n: (n.start, n.pitch)),
                            self.p.ppq, self.p.beats_per_bar)

    def _grade_de(self, notas) -> str:
        passo = self.p.ppq // 4
        fim = max(n.start for n in notas) // passo + 1
        fim = math.ceil(fim / 16) * 16
        inv = {}
        for k, v in MML.BATERIA_GM.items():
            inv.setdefault(v, k)
        linhas = {}
        for n in notas:
            nome = inv.get(n.pitch, str(n.pitch))
            g = linhas.setdefault(nome, ["."] * fim)
            g[n.start // passo] = "X" if n.velocity >= 115 else "x" if n.velocity >= 75 else "o"
        return "\n".join(f"{k}: " + "|".join("".join(g[i:i + 16]) for i in range(0, fim, 16))
                         for k, g in linhas.items())

    # ------------------------------------------------------------------
    # mix
    # ------------------------------------------------------------------
    def sala(self, preset: str = "studio_large", wet: float = 0.5) -> str:
        from .audio import psx
        if preset == "nenhuma":
            self.p.master_reverb = None
        elif preset in psx.PRESETS_REVERB:
            self.p.master_reverb = {"psx": preset, "wet": float(wet)}
        else:
            from .audio import espaco
            salas = list(getattr(espaco, "SALAS", {}))
            if preset not in salas:
                raise ErroEstudio("Salas PS1: " + ", ".join(psx.PRESETS_REVERB)
                                  + ". Convolucao: " + ", ".join(salas)
                                  + ". Ou 'nenhuma'.")
            self.p.master_reverb = {"sala": preset, "wet": float(wet)}
        self._audio_cache = None
        return f"Sala: {self.p.master_reverb}"

    def master(self, perfil: str | None = "jogo", ganho: float | None = None) -> str:
        from .audio import master as MST
        if perfil and perfil not in MST.PERFIS:
            raise ErroEstudio("Perfis: " + ", ".join(MST.PERFIS) + " ou None")
        self.p.master_perfil = perfil or None
        if ganho is not None:
            self.p.master_gain = float(ganho)
        self._audio_cache = None
        return f"Master: perfil {self.p.master_perfil}, ganho {self.p.master_gain:.2f}"

    # ------------------------------------------------------------------
    # ouvidos
    # ------------------------------------------------------------------
    def _tom_do_projeto(self):
        maiores = {"major", "lydian", "mixolydian", "major_pentatonic"}
        return T.KeyGuess(self.p.key_root,
                          "major" if self.p.scale in maiores else "natural_minor",
                          0.9)

    def analisar(self, de_compasso: int = 1, ate_compasso: int | None = None,
                 secoes: list | None = None, detectar_tom: bool = False) -> str:
        """Medidor de harmonia. `secoes`: [[nome, compasso_ini, compasso_fim], ...]

        Por padrao julga tudo no tom DECLARADO do projeto: um trecho na
        relativa maior (Sol maior dentro de Mi menor) usa as mesmas notas, e o
        detector automatico as vezes chuta um tom vizinho e acusa nota "fora".
        `detectar_tom=True` e para musica que modula de verdade."""
        bpb = self.p.beats_per_bar
        evs = self.p.harmony_events()
        kw = {} if detectar_tom else {"key": self._tom_do_projeto()}
        if secoes:
            rel = H.analyze_sections(
                evs, [(n, (a - 1) * bpb, b * bpb) for n, a, b in secoes], **kw)
            out = [H.sections_table_pt(rel, bpb)]
            for r in rel:
                ruins = [i for i in r.report.issues if i.severity >= 1][:4]
                if ruins:
                    out.append(f"\n{r.name}:")
                    out += ["  " + str(i) for i in ruins]
            return "\n".join(out)
        a = (de_compasso - 1) * bpb
        b = ate_compasso * bpb if ate_compasso else 10 ** 9
        trecho = [e for e in evs if e.start < b and e.start + e.dur > a]
        rep = H.analyze(trecho, **kw)
        return rep.summary_pt() + "\n\nSugestoes:\n" + "\n".join(
            "  - " + s for s in H.suggest_fix(rep))

    def _chave(self) -> str:
        d = self.p.to_dict()
        return hashlib.sha1(json.dumps(d, sort_keys=True, default=str)
                            .encode()).hexdigest()

    def _render(self) -> np.ndarray:
        k = self._chave()
        if self._audio_cache and self._audio_cache[0] == k:
            return self._audio_cache[1]
        audio = engine.render_project(self.p)
        self._audio_cache = (k, audio)
        return audio

    def renderizar(self, caminho: str, de_compasso: int | None = None,
                   ate_compasso: int | None = None) -> str:
        """Renderiza (.wav, ou .mp3 se houver ffmpeg) e mede o resultado."""
        from .audio import master as MST
        audio = self._render()
        sr = self.p.sample_rate
        if de_compasso or ate_compasso:
            a = int(self.p.ticks_to_seconds(((de_compasso or 1) - 1) * self._barra()) * sr)
            b = int(self.p.ticks_to_seconds((ate_compasso or 10 ** 6) * self._barra()) * sr) \
                if ate_compasso else len(audio)
            audio = audio[a:b + int(1.5 * sr)]
        destino = Path(caminho).expanduser()
        destino.parent.mkdir(parents=True, exist_ok=True)
        wav = engine.export_wav(audio, destino.with_suffix(".wav"), sr, 16)
        final = wav
        if destino.suffix.lower() == ".mp3":
            if shutil.which("ffmpeg"):
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
                                "-b:a", "192k", str(destino)], check=True)
                wav.unlink(missing_ok=True)
                final = destino
        m = MST.medir(audio, sr)
        rel = getattr(self.p, "ultimo_render", {}) or {}
        L = [f"Arquivo: {final}",
             f"Duracao {m.duracao:.1f}s · {m.lufs:.1f} LUFS · pico real "
             f"{m.pico_dbtp:.1f} dBTP · LRA {m.lra:.1f} LU · correlacao {m.correlacao:.2f}"]
        if rel.get("console") == "ps1":
            L.append(f"SPU: pico de {rel['pico_de_vozes']} vozes pedidas, "
                     f"{rel['vozes_roubadas']} roubadas (limite 24)."
                     + (" Afine o arranjo: notas foram cortadas." if rel["vozes_roubadas"] else ""))
        L.append("Balanco: " + "  ".join(f"{k} {v:.0f}" for k, v in m.bandas.items()))
        return "\n".join(L)

    def comparar(self, referencia: str) -> str:
        """Mede uma faixa de referencia e compara com o render atual."""
        from .analysis import reference as REF
        r = REF.analisar(referencia)
        audio = self._render()
        dif = r.comparar(audio, self.p.sample_rate)
        from .audio import master as MST
        lufs = MST.lufs_integrado(audio, self.p.sample_rate)
        L = [r.relatorio_pt(), "",
             "SEU RENDER vs REFERENCIA (dB por banda, nivel geral alinhado):"]
        for k, v in dif.items():
            dica = "" if abs(v) < 2 else (" ← sobrando" if v > 0 else " ← faltando")
            L.append(f"  {k:12s} {v:+5.1f}{dica}")
        L.append(f"Loudness: seu {lufs:.1f} LUFS vs referencia {r.lufs:.1f} LUFS")
        return "\n".join(L)

    def adotar_referencia(self, referencia: str) -> str:
        """Faz o master perseguir o balanco espectral da referencia."""
        from .analysis import reference as REF
        r = REF.analisar(referencia)
        self.p.master_alvo = r.perfil_alvo()
        if not self.p.master_perfil:
            self.p.master_perfil = "jogo"
        self._audio_cache = None
        return "Alvo do master: " + json.dumps(self.p.master_alvo)

    def tocar(self, de_compasso: int = 1) -> str:
        """Toca no alto-falante do computador (precisa de sounddevice)."""
        audio = self._render()
        sr = self.p.sample_rate
        ini = int(self.p.ticks_to_seconds((de_compasso - 1) * self._barra()) * sr)
        if self._player is None:
            self._player = engine.RealtimePlayer(sr)
        self._player.stop()
        self._player.play(audio, start_frame=min(ini, len(audio) - 1))
        return f"Tocando a partir do compasso {de_compasso} ({(len(audio) - ini) / sr:.0f}s)."

    def parar(self) -> str:
        if self._player:
            self._player.stop()
        return "Parado."

    def exportar_midi(self, caminho: str) -> str:
        from .midiio import smf
        out = smf.write_midi(smf.project_to_midi(self.p), caminho)
        return f"MIDI: {out}"
