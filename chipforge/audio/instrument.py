"""
chipforge.audio.instrument
==========================
Um Instrument descreve TUDO que transforma uma nota em som: forma de onda,
envelope, arpejo, vibrato, slide e efeitos. Serializa para dict, entao entra
direto no arquivo de projeto.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..core.theory import midi_to_hz
from . import effects as FX
from . import oscillators as OSC
from .envelope import ADSR, PRESET_ENVELOPES


@dataclass
class Instrument:
    name: str = "Pulse Lead"
    osc: str = "pulse25"            # ver OSC.OSC_LABELS_PT
    wavetable: str = "gb_sine"      # usado quando osc == "wave"
    env: ADSR = field(default_factory=lambda: ADSR(**asdict(PRESET_ENVELOPES["lead"])))
    gain: float = 0.8
    pan: float = 0.0
    detune_cents: float = 0.0
    # arpejo: offsets em semitons percorridos rapidamente (acorde em 1 canal)
    arpeggio: tuple[int, ...] = ()
    arp_rate_hz: float = 18.0
    # vibrato
    vibrato_depth_cents: float = 0.0
    vibrato_rate_hz: float = 5.5
    vibrato_delay_s: float = 0.08
    # slide de pitch (em semitons, do inicio ate o fim da nota)
    pitch_sweep: float = 0.0
    glide_s: float = 0.0            # portamento a partir da nota anterior
    # cor / lo-fi
    bits: int = 0                   # 0 = herda do projeto
    sample_rate_crush: int = 0      # 0 = desligado
    lowpass_hz: float = 0.0
    drive: float = 1.0
    is_drum: bool = False

    # ------------------------------------------------------------------
    def render_note(self, pitch: float, dur_s: float, sr: int,
                    velocity: int = 100, prev_pitch: float | None = None,
                    release: bool = True) -> np.ndarray:
        n = max(1, int(dur_s * sr))
        rel_n = max(1, int(self.env.release * sr)) if release else 0
        total = n + rel_n
        t = np.arange(total, dtype=np.float64) / sr

        base = float(pitch) + self.detune_cents / 100.0
        semis = np.full(total, base)

        if self.arpeggio:
            step = np.floor(t * self.arp_rate_hz).astype(int) % len(self.arpeggio)
            semis = semis + np.asarray(self.arpeggio, dtype=np.float64)[step]

        if self.pitch_sweep:
            semis = semis + self.pitch_sweep * np.clip(t / max(dur_s, 1e-6), 0, 1)

        if self.glide_s > 0 and prev_pitch is not None:
            k = np.clip(t / self.glide_s, 0, 1)
            semis = semis + (prev_pitch - base) * (1 - k)

        if self.vibrato_depth_cents:
            on = np.clip((t - self.vibrato_delay_s) * 6.0, 0, 1)
            semis = semis + on * (self.vibrato_depth_cents / 100.0) * \
                np.sin(2 * np.pi * self.vibrato_rate_hz * t)

        freq = np.asarray([midi_to_hz(s) for s in semis]) if total < 0 else \
            440.0 * (2.0 ** ((semis - 69.0) / 12.0))
        freq = np.clip(freq, 8.0, sr / 2.2)

        o = self.osc
        if o.startswith("pulse"):
            duty = {"pulse12": 0.125, "pulse25": 0.25,
                    "pulse50": 0.5, "pulse75": 0.75}.get(o, 0.5)
            sig = OSC.pulse(freq, sr, duty)
        elif o == "triangle":
            sig = OSC.triangle(freq, sr)
        elif o == "saw":
            sig = OSC.saw(freq, sr)
        elif o == "sine":
            sig = OSC.sine(freq, sr)
        elif o == "noise":
            sig = OSC.noise(freq, sr, short_period=False)
        elif o == "noise_metal":
            sig = OSC.noise(freq, sr, short_period=True)
        elif o == "wave":
            sig = OSC.wavetable(freq, sr, OSC.WAVETABLES[self.wavetable])
        else:
            sig = OSC.pulse(freq, sr, 0.5)

        env = self.env.render(n, sr, release_tail=rel_n)[:total]
        if len(env) < total:
            env = np.pad(env, (0, total - len(env)))
        sig = sig[:total] * env * (velocity / 127.0) * self.gain

        if self.lowpass_hz:
            sig = FX.lowpass(sig, sr, self.lowpass_hz)
        if self.drive != 1.0:
            sig = FX.soft_clip(sig, self.drive)
        if self.sample_rate_crush:
            sig = FX.downsample(sig, sr, self.sample_rate_crush)
        if self.bits:
            sig = FX.bitcrush(sig, self.bits)
        return sig.astype(np.float32)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["env"] = asdict(self.env)
        d["arpeggio"] = list(self.arpeggio)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Instrument":
        d = dict(d)
        d["env"] = ADSR(**d.get("env", {}))
        d["arpeggio"] = tuple(d.get("arpeggio", ()))
        return cls(**d)


def _env(name: str, **kw) -> ADSR:
    base = asdict(PRESET_ENVELOPES[name])
    base.update(kw)
    return ADSR(**base)


# --------------------------------------------------------------------------
# Banco de fabrica — o que aparece no browser de instrumentos da UI
# --------------------------------------------------------------------------
FACTORY_BANK: dict[str, list[Instrument]] = {
    "Leads": [
        Instrument("NES Lead 25%", "pulse25", env=_env("lead"),
                   vibrato_depth_cents=22, gain=0.55),
        Instrument("NES Lead 12.5%", "pulse12", env=_env("lead"),
                   vibrato_depth_cents=18, gain=0.5),
        Instrument("Square Fat", "pulse50", env=_env("lead"),
                   detune_cents=6, gain=0.5),
        Instrument("Saw Hero", "saw", env=_env("lead"), lowpass_hz=6000, gain=0.45),
        Instrument("Chip Bell", "wave", wavetable="gb_bell",
                   env=_env("pluck", decay=0.35), gain=0.55),
        Instrument("Arp Chord (maior)", "pulse25", env=_env("organ"),
                   arpeggio=(0, 4, 7), arp_rate_hz=20, gain=0.4),
        Instrument("Arp Chord (menor)", "pulse25", env=_env("organ"),
                   arpeggio=(0, 3, 7), arp_rate_hz=20, gain=0.4),
    ],
    "Baixos": [
        Instrument("NES Triangle Bass", "triangle", env=_env("bass"), gain=0.9),
        Instrument("Sub Sine", "sine", env=_env("bass", attack=0.004), gain=0.9),
        Instrument("Square Bass", "pulse50", env=_env("bass"),
                   lowpass_hz=1800, gain=0.7),
        Instrument("Growl Bass", "saw", env=_env("bass"), lowpass_hz=900,
                   drive=2.2, gain=0.6),
    ],
    "Pads e teclas": [
        Instrument("GB Organ", "wave", wavetable="gb_organ",
                   env=_env("organ"), gain=0.4),
        Instrument("Warm Pad", "wave", wavetable="gb_vocal",
                   env=_env("pad"), lowpass_hz=3200, gain=0.35),
        Instrument("Hollow Pad", "wave", wavetable="gb_hollow",
                   env=_env("pad"), gain=0.35),
        Instrument("Pluck Harp", "wave", wavetable="gb_ramp",
                   env=_env("pluck", decay=0.25), gain=0.5),
    ],
    "Bateria": [
        Instrument("Kick", "sine", env=_env("perc", decay=0.12),
                   pitch_sweep=-30, gain=1.0, is_drum=True),
        Instrument("Snare", "noise", env=_env("perc", decay=0.11),
                   gain=0.6, is_drum=True, lowpass_hz=7000),
        Instrument("Hi-hat fechado", "noise", env=_env("perc", decay=0.03),
                   gain=0.32, is_drum=True),
        Instrument("Hi-hat aberto", "noise", env=_env("perc", decay=0.22),
                   gain=0.28, is_drum=True),
        Instrument("Tom", "triangle", env=_env("perc", decay=0.18),
                   pitch_sweep=-8, gain=0.7, is_drum=True),
        Instrument("Crash", "noise", env=_env("perc", decay=0.7),
                   gain=0.3, is_drum=True),
        Instrument("Zap SFX", "noise_metal", env=_env("perc", decay=0.15),
                   pitch_sweep=-24, gain=0.5, is_drum=True),
    ],
}

# Mapa de bateria (General MIDI) -> nome do preset. Assim um MIDI de bateria
# baixado da internet ja toca certo.
GM_DRUM_MAP = {
    35: "Kick", 36: "Kick", 37: "Snare", 38: "Snare", 40: "Snare",
    41: "Tom", 43: "Tom", 45: "Tom", 47: "Tom", 48: "Tom", 50: "Tom",
    42: "Hi-hat fechado", 44: "Hi-hat fechado", 46: "Hi-hat aberto",
    49: "Crash", 51: "Crash", 52: "Crash", 55: "Crash", 57: "Crash",
}

DRUM_BASE_PITCH = {
    "Kick": 40, "Snare": 62, "Hi-hat fechado": 92, "Hi-hat aberto": 92,
    "Tom": 52, "Crash": 88, "Zap SFX": 70,
}


def factory_by_name(name: str) -> Instrument | None:
    for group in FACTORY_BANK.values():
        for inst in group:
            if inst.name == name:
                return Instrument.from_dict(inst.to_dict())
    return None


def all_factory_names() -> list[str]:
    return [i.name for g in FACTORY_BANK.values() for i in g]
