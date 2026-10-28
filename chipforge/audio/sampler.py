"""
chipforge.audio.sampler
=======================
Leitor de SoundFont (.sf2) e instrumento de samples.

Le o RIFF na mao e monta o mapa preset -> instrumento -> zonas -> sample,
percorrendo phdr/pbag/pgen/inst/ibag/igen/shdr. Com isso da para carregar
qualquer banco GM gratuito (GeneralUser GS, FluidR3, packs do archive.org)
e ter piano, cordas, metais e bateria de verdade ao lado dos canais chiptune.

Nao e um player SF2 completo (modulators, filtros e LFOs do padrao ficam de
fora); e um extrator de zonas com playback proprio — o suficiente para compor,
e o resultado passa pelo mesmo bitcrusher do resto, entao continua com a cara
do projeto.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import effects as FX
from .envelope import ADSR

# generator operators que nos interessam
GEN_KEY_RANGE = 43
GEN_VEL_RANGE = 44
GEN_INSTRUMENT = 41
GEN_SAMPLE_ID = 53
GEN_OVERRIDING_ROOT = 58
GEN_SAMPLE_MODES = 54
GEN_FINE_TUNE = 52
GEN_COARSE_TUNE = 51
GEN_DELAY_VOL = 33
GEN_ATTACK_VOL = 34
GEN_HOLD_VOL = 35
GEN_DECAY_VOL = 36
GEN_SUSTAIN_VOL = 37
GEN_RELEASE_VOL = 38
GEN_PAN = 17
GEN_ATTENUATION = 48
GEN_FILTER_FC = 8
GEN_FILTER_Q = 9


def _timecents(v: int | None, padrao: float) -> float:
    """Timecents -> segundos. -12000 tc = 1 ms, 0 tc = 1 s."""
    if v is None:
        return padrao
    v = _signed(v)
    if v <= -12000:
        return 0.001
    return float(2.0 ** (v / 1200.0))


def _centibeis_para_nivel(v: int | None) -> float:
    """Centibeis de ATENUACAO -> ganho linear. 0 cb = volume cheio."""
    if v is None:
        return 1.0
    cb = max(0, _signed(v))
    if cb >= 1000:
        return 0.0
    return float(10.0 ** (-cb / 200.0))


@dataclass
class SampleHeader:
    name: str
    start: int
    end: int
    loop_start: int
    loop_end: int
    sample_rate: int
    original_pitch: int
    pitch_correction: int
    sample_type: int


@dataclass
class Zone:
    sample_index: int
    key_lo: int = 0
    key_hi: int = 127
    vel_lo: int = 0
    vel_hi: int = 127
    root_override: int = -1
    loop_mode: int = 0
    coarse: int = 0
    fine: int = 0
    # --- envelope de volume declarado no proprio banco -------------------
    # Sem isto, todo instrumento sustenta igual: piano vira orgao, marimba
    # vira pad. O envelope certo e o que separa "sample de instrumento" de
    # "instrumento".
    delay_s: float = 0.001
    attack_s: float = 0.001
    hold_s: float = 0.001
    decay_s: float = 0.001
    sustain: float = 1.0
    release_s: float = 0.1
    atenuacao: float = 1.0
    filtro_hz: float = 20000.0


@dataclass
class Preset:
    name: str
    bank: int
    program: int
    zones: list[Zone] = field(default_factory=list)

    @property
    def is_drum(self) -> bool:
        return self.bank == 128


def _chunks(buf: bytes, start: int, end: int):
    i = start
    while i + 8 <= end:
        cid = buf[i:i + 4]
        size = struct.unpack("<I", buf[i + 4:i + 8])[0]
        yield cid, i + 8, size
        i += 8 + size + (size & 1)


class SoundFont:
    """Carrega um .sf2 e expõe presets prontos para tocar."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        raw = self.path.read_bytes()
        if raw[:4] != b"RIFF" or raw[8:12] != b"sfbk":
            raise ValueError("Arquivo nao e um SoundFont valido (.sf2)")
        total = struct.unpack("<I", raw[4:8])[0] + 8
        self.name = self.path.stem
        self._pcm: np.ndarray | None = None
        raw_chunks: dict[bytes, tuple[int, int]] = {}

        for cid, off, size in _chunks(raw, 12, total):
            if cid != b"LIST":
                continue
            list_type = raw[off:off + 4]
            for sub, soff, ssize in _chunks(raw, off + 4, off + size):
                raw_chunks[sub] = (soff, ssize)
                if list_type == b"INFO" and sub == b"INAM":
                    self.name = raw[soff:soff + ssize].split(b"\0")[0].decode(
                        "latin-1", "replace")

        if b"smpl" in raw_chunks:
            o, s = raw_chunks[b"smpl"]
            self._pcm = np.frombuffer(raw[o:o + s], dtype="<i2").astype(np.float32) / 32768.0

        self.samples = self._read_shdr(raw, raw_chunks)
        self.presets = self._read_presets(raw, raw_chunks)

    # ------------------------------------------------------------------
    @staticmethod
    def _read_shdr(raw, ch) -> list[SampleHeader]:
        if b"shdr" not in ch:
            return []
        o, s = ch[b"shdr"]
        out = []
        for i in range(s // 46):
            b = raw[o + i * 46: o + (i + 1) * 46]
            name = b[:20].split(b"\0")[0].decode("latin-1", "replace")
            (st, en, ls, le, sr) = struct.unpack("<5I", b[20:40])
            pitch, corr, _link, stype = struct.unpack("<BbHH", b[40:46])
            if name == "EOS":
                break
            out.append(SampleHeader(name, st, en, ls, le, sr or 44100,
                                    pitch if pitch < 128 else 60, corr, stype))
        return out

    @staticmethod
    def _read_bag(raw, ch, key) -> list[tuple[int, int]]:
        o, s = ch[key]
        return [struct.unpack("<HH", raw[o + i * 4:o + i * 4 + 4])
                for i in range(s // 4)]

    @staticmethod
    def _read_gen(raw, ch, key) -> list[tuple[int, int]]:
        o, s = ch[key]
        out = []
        for i in range(s // 4):
            op, amount = struct.unpack("<HH", raw[o + i * 4:o + i * 4 + 4])
            out.append((op, amount))
        return out

    def _read_presets(self, raw, ch) -> list[Preset]:
        need = (b"phdr", b"pbag", b"pgen", b"inst", b"ibag", b"igen")
        if not all(k in ch for k in need):
            return []
        o, s = ch[b"phdr"]
        phdr = []
        for i in range(s // 38):
            b = raw[o + i * 38:o + (i + 1) * 38]
            nm = b[:20].split(b"\0")[0].decode("latin-1", "replace")
            preset, bank, bag = struct.unpack("<HHH", b[20:26])
            phdr.append((nm, preset, bank, bag))

        pbag = self._read_bag(raw, ch, b"pbag")
        pgen = self._read_gen(raw, ch, b"pgen")
        ibag = self._read_bag(raw, ch, b"ibag")
        igen = self._read_gen(raw, ch, b"igen")

        o, s = ch[b"inst"]
        inst = []
        for i in range(s // 22):
            b = raw[o + i * 22:o + (i + 1) * 22]
            nm = b[:20].split(b"\0")[0].decode("latin-1", "replace")
            bag = struct.unpack("<H", b[20:22])[0]
            inst.append((nm, bag))

        def zones_of_instrument(idx: int) -> list[Zone]:
            if idx >= len(inst) - 1:
                return []
            b0, b1 = inst[idx][1], inst[idx + 1][1]
            out: list[Zone] = []
            glob: dict[int, int] = {}
            for bi in range(b0, min(b1, len(ibag) - 1)):
                g0, g1 = ibag[bi][0], ibag[bi + 1][0]
                gens = dict(igen[g0:g1])
                if GEN_SAMPLE_ID not in gens:
                    glob.update(gens)
                    continue
                merged = {**glob, **gens}
                kr = merged.get(GEN_KEY_RANGE, 0x7F00)
                vr = merged.get(GEN_VEL_RANGE, 0x7F00)
                fc = merged.get(GEN_FILTER_FC)
                filtro = 20000.0 if fc is None else \
                    float(8.176 * 2.0 ** (_signed(fc) / 1200.0))
                out.append(Zone(
                    sample_index=merged[GEN_SAMPLE_ID],
                    key_lo=kr & 0xFF, key_hi=(kr >> 8) & 0xFF,
                    vel_lo=vr & 0xFF, vel_hi=(vr >> 8) & 0xFF,
                    root_override=merged.get(GEN_OVERRIDING_ROOT, -1),
                    loop_mode=merged.get(GEN_SAMPLE_MODES, 0),
                    coarse=_signed(merged.get(GEN_COARSE_TUNE, 0)),
                    fine=_signed(merged.get(GEN_FINE_TUNE, 0)),
                    delay_s=_timecents(merged.get(GEN_DELAY_VOL), 0.001),
                    attack_s=_timecents(merged.get(GEN_ATTACK_VOL), 0.001),
                    hold_s=_timecents(merged.get(GEN_HOLD_VOL), 0.001),
                    decay_s=_timecents(merged.get(GEN_DECAY_VOL), 0.001),
                    sustain=_centibeis_para_nivel(merged.get(GEN_SUSTAIN_VOL)),
                    release_s=_timecents(merged.get(GEN_RELEASE_VOL), 0.1),
                    atenuacao=_centibeis_para_nivel(merged.get(GEN_ATTENUATION)),
                    filtro_hz=min(filtro, 20000.0)))
            return out

        presets: list[Preset] = []
        for i in range(len(phdr) - 1):
            nm, prog, bank, b0 = phdr[i]
            b1 = phdr[i + 1][3]
            p = Preset(nm, bank, prog)
            for bi in range(b0, min(b1, len(pbag) - 1)):
                g0, g1 = pbag[bi][0], pbag[bi + 1][0]
                gens = dict(pgen[g0:g1])
                if GEN_INSTRUMENT in gens:
                    p.zones.extend(zones_of_instrument(gens[GEN_INSTRUMENT]))
            if p.zones:
                presets.append(p)
        presets.sort(key=lambda p: (p.bank, p.program))
        return presets

    # ------------------------------------------------------------------
    def preset_names(self) -> list[str]:
        return [f"{p.bank:03d}:{p.program:03d}  {p.name}" for p in self.presets]

    def find(self, query: str) -> list[Preset]:
        q = query.lower()
        return [p for p in self.presets if q in p.name.lower()]

    def sample_audio(self, idx: int) -> tuple[np.ndarray, SampleHeader]:
        h = self.samples[idx]
        return self._pcm[h.start:h.end], h


def _signed(v: int) -> int:
    return v - 65536 if v > 32767 else v


PISO_DB = -96.0
PISO = 10.0 ** (PISO_DB / 20.0)


def envelope_da_zona(z: Zone, n: int, rel_n: int, sr: int) -> np.ndarray:
    """Constroi o envelope declarado no banco: delay, attack, hold, decay,
    sustain e release.

    O decay anda LINEAR EM DECIBEIS, que e como o padrao SF2 define e como o
    som se comporta: um piano perde metade da energia sempre no mesmo intervalo,
    nao numa reta. Quando o sustain e zero, o decay leva ate o silencio — e por
    isso que uma nota de piano morre sozinha e uma de violino nao.
    """
    total = n + rel_n
    env = np.zeros(total, dtype=np.float32)
    sus = max(float(z.sustain), PISO) if z.sustain > 0 else PISO
    pos = 0

    def por(cont: int, valores) -> None:
        nonlocal pos
        k = min(int(cont), total - pos)
        if k > 0:
            env[pos:pos + k] = valores[:k]
            pos += k

    por(int(z.delay_s * sr), np.zeros(total, dtype=np.float32))
    na = max(1, int(z.attack_s * sr))
    por(na, np.linspace(0.0, 1.0, na, endpoint=False, dtype=np.float32) ** 0.6)
    por(int(z.hold_s * sr), np.ones(total, dtype=np.float32))
    nd = int(z.decay_s * sr)
    if nd > 0:
        db = np.linspace(0.0, 20 * np.log10(sus), nd, dtype=np.float32)
        por(nd, (10.0 ** (db / 20.0)).astype(np.float32))
    if pos < total:
        env[pos:] = sus

    if rel_n > 0 and n < total:
        nivel = max(float(env[min(n, total - 1)]), PISO)
        db = np.linspace(20 * np.log10(nivel), PISO_DB, rel_n, dtype=np.float32)
        env[n:n + rel_n] = np.minimum(env[n:n + rel_n],
                                      (10.0 ** (db / 20.0)).astype(np.float32))
    return env


# --------------------------------------------------------------------------
@dataclass
class SamplerInstrument:
    """Toca um preset de SoundFont, com o mesmo contrato do Instrument chiptune."""
    soundfont: SoundFont
    preset: Preset
    name: str = ""
    gain: float = 0.9
    pan: float = 0.0
    env: ADSR = field(default_factory=lambda: ADSR(0.002, 0.05, 0.9, 0.15))
    bits: int = 0
    sample_rate_crush: int = 0
    lowpass_hz: float = 0.0
    drive: float = 1.0
    is_drum: bool = False
    # campos que existem so para o instrumento de sample responder a mesma
    # interface do chiptune — a UI mexe neles sem precisar saber a diferenca
    osc: str = "sample"
    wavetable: str = "gb_sine"
    arpeggio: tuple = ()
    vibrato_depth_cents: float = 0.0
    vibrato_hz: float = 5.2
    vibrato_atraso_s: float = 0.28     # o arco pega antes do vibrato entrar
    detune_cents: float = 0.0
    # True = usa o envelope e o filtro que vem dentro do .sf2 (o certo).
    # A interface poe False quando voce mexe nos sliders de ADSR na mao.
    env_do_banco: bool = True
    cauda_max_s: float = 4.0
    # modo PS1: taxa em que este sample "mora" na RAM do SPU. 0 = a do projeto
    # (22050). Baixo em 11025 e o truque classico para caber mais coisa.
    psx_taxa: int = 0

    def __post_init__(self):
        if not self.name:
            self.name = self.preset.name
        self.is_drum = self.is_drum or self.preset.is_drum

    def _effective_root(self, z: Zone) -> int:
        if z.root_override >= 0:
            return z.root_override
        return self.soundfont.samples[z.sample_index].original_pitch

    def _pick_zone(self, pitch: int, velocity: int) -> Zone | None:
        """Escolhe a zona certa entre as centenas que um banco de verdade tem.

        Um preset como o 'Stereo Grand' do GeneralUser traz mais de mil zonas:
        cada tecla, cada camada de velocity, canal esquerdo e direito. Pegar a
        primeira que serve devolve som errado — as vezes um layer de release,
        as vezes so o canal direito. A ordem de preferencia aqui e:
          1. respeita key range e velocity range
          2. prefere sample mono ou canal esquerdo (a saida daqui e mono)
          3. entre as que sobram, a de root mais perto da nota pedida —
             menos transposicao = menos artefato
        """
        zs = self.preset.zones
        if not zs:
            return None
        cands = [z for z in zs
                 if z.key_lo <= pitch <= z.key_hi and z.vel_lo <= velocity <= z.vel_hi]
        if not cands:
            cands = [z for z in zs if z.key_lo <= pitch <= z.key_hi]
        if not cands:
            cands = zs

        def rank(z: Zone):
            stype = self.soundfont.samples[z.sample_index].sample_type
            mono_or_left = 0 if stype in (1, 4) else 1
            return (mono_or_left, abs(self._effective_root(z) - pitch))

        return min(cands, key=rank)

    def render_note(self, pitch: float, dur_s: float, sr: int,
                    velocity: int = 100, prev_pitch=None,
                    release: bool = True) -> np.ndarray:
        z = self._pick_zone(int(pitch), velocity)
        n = max(1, int(dur_s * sr))
        if release:
            solta = z.release_s if (z and self.env_do_banco) else self.env.release
            rel_n = max(1, int(min(solta, self.cauda_max_s) * sr))
        else:
            rel_n = 0
        total = n + rel_n
        if z is None or self.soundfont._pcm is None:
            return np.zeros(total, dtype=np.float32)

        data, hdr = self.soundfont.sample_audio(z.sample_index)
        if len(data) < 4:
            return np.zeros(total, dtype=np.float32)

        from . import psx as PSX
        modo_psx = PSX.ESTADO["ligado"]
        taxa_dados = hdr.sample_rate
        ls = hdr.loop_start - hdr.start
        le = hdr.loop_end - hdr.start
        if modo_psx:
            # o sample como estaria na RAM do PlayStation: taxa baixa + ADPCM
            alvo = self.psx_taxa or PSX.ESTADO["taxa"]
            data, taxa_dados = PSX.amostra_na_ram(
                (str(self.soundfont.path), z.sample_index), data,
                hdr.sample_rate, alvo)
            esc = taxa_dados / hdr.sample_rate
            ls, le = ls * esc, le * esc

        root = self._effective_root(z)
        cents = ((pitch - root) * 100 + z.coarse * 100 + z.fine
                 + hdr.pitch_correction + self.detune_cents)
        ratio = (2.0 ** (cents / 1200.0)) * (taxa_dados / sr)
        if modo_psx:
            # o registrador de pitch do SPU para em ~4x: nota aguda demais
            # simplesmente nao sobe mais — limitacao que os compositores
            # contornavam gravando o sample uma oitava acima
            ratio = min(ratio, PSX.PASSO_MAXIMO * 44100.0 / sr)

        loop = z.loop_mode in (1, 3) and le > ls + 8

        if self.vibrato_depth_cents > 0:
            # Vibrato de verdade: a razao de leitura varia AO LONGO da nota, e
            # a posicao e a integral dela. Ate agora este campo existia no
            # dataclass e nao era lido por ninguem — um violino solo saia com
            # a afinacao travada, que e exatamente o que denuncia sample.
            #
            # A entrada e atrasada de proposito: violinista nao comeca a nota
            # com vibrato ligado, ele instala o vibrato depois que o arco pegou.
            t = np.arange(total, dtype=np.float64) / sr
            sobe = np.clip((t - self.vibrato_atraso_s) / 0.35, 0.0, 1.0)
            osc = np.sin(2 * np.pi * self.vibrato_hz * t)
            fator = 2.0 ** (sobe * self.vibrato_depth_cents * osc / 1200.0)
            idx = np.cumsum(ratio * fator) - ratio * fator[0]
        else:
            idx = np.arange(total, dtype=np.float64) * ratio
        if loop:
            loop_len = le - ls
            over = idx >= ls
            idx = np.where(over, ls + np.mod(idx - ls, loop_len), idx)
        else:
            idx = np.clip(idx, 0, len(data) - 1.001)

        if modo_psx:
            sig = PSX.ler_gaussiano(data, idx).astype(np.float32)
        else:
            i0 = idx.astype(np.int64)
            frac = (idx - i0).astype(np.float32)
            i1 = np.minimum(i0 + 1, len(data) - 1)
            sig = data[i0] * (1 - frac) + data[i1] * frac
        if not loop:
            sig[idx >= len(data) - 1.01] = 0.0

        if self.env_do_banco:
            env = envelope_da_zona(z, n, rel_n, sr)
            sig = sig * env * z.atenuacao
        else:
            env = self.env.render(n, sr, release_tail=rel_n)[:total]
            if len(env) < total:
                env = np.pad(env, (0, total - len(env)))
            sig = sig * env
        sig = sig * (velocity / 127.0) * self.gain

        # filtro declarado na zona: e ele que faz a camada de velocity baixa
        # soar abafada e a de fortissimo soar aberta, no mesmo instrumento
        if self.env_do_banco and z.filtro_hz < 18000.0:
            sig = FX.lowpass(sig, sr, z.filtro_hz)
        if self.lowpass_hz:
            sig = FX.lowpass(sig, sr, self.lowpass_hz)
        if self.drive != 1.0:
            sig = FX.soft_clip(sig, self.drive)
        if self.sample_rate_crush:
            sig = FX.downsample(sig, sr, self.sample_rate_crush)
        if self.bits:
            sig = FX.bitcrush(sig, self.bits)
        return sig.astype(np.float32)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return {"kind": "sampler", "sf2": str(self.soundfont.path),
                "bank": self.preset.bank, "program": self.preset.program,
                "preset_name": self.preset.name,
                "name": self.name, "gain": self.gain, "pan": self.pan,
                "env": asdict(self.env), "bits": self.bits,
                "sample_rate_crush": self.sample_rate_crush,
                "lowpass_hz": self.lowpass_hz, "drive": self.drive,
                "detune_cents": self.detune_cents, "is_drum": self.is_drum,
                "vibrato_depth_cents": self.vibrato_depth_cents,
                "vibrato_hz": self.vibrato_hz,
                "env_do_banco": self.env_do_banco,
                "psx_taxa": self.psx_taxa}

    @classmethod
    def from_dict(cls, d: dict) -> "SamplerInstrument":
        sf = open_soundfont(d["sf2"])
        preset = find_preset(sf, d.get("bank", 0), d.get("program", 0),
                             d.get("preset_name"))
        if preset is None:
            raise ValueError(
                f"O preset {d.get('preset_name')} nao existe em {d['sf2']}")
        return cls(sf, preset, name=d.get("name", ""),
                   gain=d.get("gain", 0.9), pan=d.get("pan", 0.0),
                   env=ADSR(**d.get("env", {})), bits=d.get("bits", 0),
                   sample_rate_crush=d.get("sample_rate_crush", 0),
                   lowpass_hz=d.get("lowpass_hz", 0.0),
                   drive=d.get("drive", 1.0),
                   detune_cents=d.get("detune_cents", 0.0),
                   is_drum=d.get("is_drum", False),
                   vibrato_depth_cents=d.get("vibrato_depth_cents", 0.0),
                   vibrato_hz=d.get("vibrato_hz", 5.2),
                   env_do_banco=d.get("env_do_banco", True),
                   psx_taxa=d.get("psx_taxa", 0))


# --------------------------------------------------------------------------
# Cache de bancos: um .sf2 de 30 MB nao pode ser lido duas vezes
# --------------------------------------------------------------------------
_SF_CACHE: dict[str, SoundFont] = {}


def open_soundfont(path: str | Path) -> SoundFont:
    key = str(Path(path).resolve())
    if key not in _SF_CACHE:
        _SF_CACHE[key] = SoundFont(key)
    return _SF_CACHE[key]


def find_preset(sf: SoundFont, bank: int, program: int,
                name: str | None = None) -> Preset | None:
    for p in sf.presets:
        if p.bank == bank and p.program == program:
            return p
    if name:
        hits = [p for p in sf.presets if p.name == name]
        if hits:
            return hits[0]
        hits = sf.find(name)
        if hits:
            return hits[0]
    return None


def instrument(sf: SoundFont | str | Path, query: str, **kw) -> SamplerInstrument:
    """Atalho: sampler.instrument(banco, "Nylon Guitar", gain=0.8)"""
    if not isinstance(sf, SoundFont):
        sf = open_soundfont(sf)
    hits = sf.find(query)
    if not hits:
        raise KeyError(f'Nenhum preset com "{query}" em {sf.name}')
    return SamplerInstrument(sf, hits[0], **kw)
