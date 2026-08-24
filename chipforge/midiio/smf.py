"""
chipforge.midiio.smf
====================
Leitor e escritor de Standard MIDI File (formato 0 e 1) em Python puro.

Sem dependencia externa de proposito: o formato e simples, e assim o Chipforge
nao quebra quando uma lib de terceiro muda. Suporta note on/off, program change,
set tempo, time signature, track name e control change (volume/pan).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------------
# Variable-length quantity
# --------------------------------------------------------------------------
def _read_vlq(data: bytes, i: int) -> tuple[int, int]:
    v = 0
    while True:
        b = data[i]
        i += 1
        v = (v << 7) | (b & 0x7F)
        if not b & 0x80:
            return v, i


def _write_vlq(v: int) -> bytes:
    if v < 0:
        v = 0
    out = bytearray([v & 0x7F])
    v >>= 7
    while v:
        out.insert(0, (v & 0x7F) | 0x80)
        v >>= 7
    return bytes(out)


# --------------------------------------------------------------------------
@dataclass
class MidiNote:
    start: int          # ticks absolutos
    dur: int
    pitch: int
    velocity: int
    channel: int = 0


@dataclass
class MidiTrack:
    name: str = ""
    program: int = 0           # instrumento GM
    channel: int = 0
    notes: list[MidiNote] = field(default_factory=list)

    @property
    def is_drum(self) -> bool:
        return self.channel == 9


@dataclass
class MidiFile:
    ppq: int = 480
    tempo_bpm: float = 120.0
    numerator: int = 4
    denominator: int = 4
    tracks: list[MidiTrack] = field(default_factory=list)

    @property
    def all_notes(self) -> list[MidiNote]:
        return sorted((n for t in self.tracks for n in t.notes),
                      key=lambda n: n.start)


# --------------------------------------------------------------------------
# Leitura
# --------------------------------------------------------------------------
def read_midi(path: str | Path) -> MidiFile:
    raw = Path(path).read_bytes()
    if raw[:4] != b"MThd":
        raise ValueError("Nao parece um arquivo MIDI (falta MThd)")
    hdr_len = struct.unpack(">I", raw[4:8])[0]
    fmt, ntrk, div = struct.unpack(">HHH", raw[8:14])
    if div & 0x8000:
        raise ValueError("MIDI em SMPTE timecode ainda nao suportado")
    mf = MidiFile(ppq=div)
    pos = 8 + hdr_len

    for _ in range(ntrk):
        if raw[pos:pos + 4] != b"MTrk":
            break
        tlen = struct.unpack(">I", raw[pos + 4:pos + 8])[0]
        chunk = raw[pos + 8:pos + 8 + tlen]
        pos += 8 + tlen
        mf.tracks.extend(_parse_track(chunk, mf))

    # tracks vazias (so meta) nao interessam
    mf.tracks = [t for t in mf.tracks if t.notes]
    return mf


def _parse_track(data: bytes, mf: MidiFile) -> list[MidiTrack]:
    i = 0
    t = 0
    running = 0
    per_channel: dict[int, MidiTrack] = {}
    open_notes: dict[tuple[int, int], tuple[int, int]] = {}
    track_name = ""

    def chan(c: int) -> MidiTrack:
        if c not in per_channel:
            per_channel[c] = MidiTrack(name=track_name, channel=c)
        return per_channel[c]

    while i < len(data):
        dt, i = _read_vlq(data, i)
        t += dt
        if i >= len(data):
            break
        status = data[i]
        if status & 0x80:
            i += 1
            running = status
        else:
            status = running
        ev, c = status & 0xF0, status & 0x0F

        if status == 0xFF:                       # meta
            mtype = data[i]; i += 1
            ln, i = _read_vlq(data, i)
            payload = data[i:i + ln]; i += ln
            if mtype == 0x51 and ln == 3:
                us = (payload[0] << 16) | (payload[1] << 8) | payload[2]
                mf.tempo_bpm = round(60_000_000 / us, 3)
            elif mtype == 0x58 and ln >= 2:
                mf.numerator = payload[0]
                mf.denominator = 2 ** payload[1]
            elif mtype == 0x03:
                track_name = payload.decode("latin-1", "replace").strip()
                for tr in per_channel.values():
                    if not tr.name:
                        tr.name = track_name
            continue

        if status in (0xF0, 0xF7):               # sysex
            ln, i = _read_vlq(data, i)
            i += ln
            continue

        if ev in (0x80, 0x90, 0xA0, 0xB0, 0xE0):
            d1, d2 = data[i], data[i + 1]; i += 2
            if ev == 0x90 and d2 > 0:
                open_notes[(c, d1)] = (t, d2)
            elif ev == 0x80 or (ev == 0x90 and d2 == 0):
                key = (c, d1)
                if key in open_notes:
                    st, vel = open_notes.pop(key)
                    chan(c).notes.append(MidiNote(st, max(1, t - st), d1, vel, c))
        elif ev in (0xC0, 0xD0):
            d1 = data[i]; i += 1
            if ev == 0xC0:
                chan(c).program = d1
        else:
            i += 1

    # notas presas: fecha no fim da track
    for (c, p), (st, vel) in open_notes.items():
        chan(c).notes.append(MidiNote(st, max(1, t - st), p, vel, c))

    for tr in per_channel.values():
        if not tr.name:
            tr.name = f"Canal {tr.channel + 1}"
    return list(per_channel.values())


# --------------------------------------------------------------------------
# Escrita
# --------------------------------------------------------------------------
# ordem importa: o primeiro termo encontrado no nome vence
GM_PROGRAM_HINTS = {
    "bass": 38, "sub": 39, "kick": 118, "snare": 118, "drum": 118,
    "organ": 19, "pad": 89, "bell": 10, "harp": 46, "pluck": 45,
    "lead": 80, "square": 80, "pulse": 80, "saw": 81, "triangle": 80,
}


def write_midi(mf: MidiFile, path: str | Path) -> Path:
    path = Path(path).with_suffix(".mid")
    chunks = [_build_meta_track(mf)]
    for tr in mf.tracks:
        chunks.append(_build_note_track(tr))
    hdr = b"MThd" + struct.pack(">IHHH", 6, 1, len(chunks), mf.ppq)
    body = b"".join(b"MTrk" + struct.pack(">I", len(c)) + c for c in chunks)
    path.write_bytes(hdr + body)
    return path


def _build_meta_track(mf: MidiFile) -> bytes:
    out = bytearray()
    us = int(round(60_000_000 / mf.tempo_bpm))
    out += _write_vlq(0) + b"\xFF\x51\x03" + us.to_bytes(3, "big")
    den_pow = max(0, int.bit_length(mf.denominator) - 1)
    out += _write_vlq(0) + b"\xFF\x58\x04" + bytes([mf.numerator, den_pow, 24, 8])
    out += _write_vlq(0) + b"\xFF\x2F\x00"
    return bytes(out)


def _build_note_track(tr: MidiTrack) -> bytes:
    out = bytearray()
    if tr.name:
        nb = tr.name.encode("latin-1", "replace")[:127]
        out += _write_vlq(0) + b"\xFF\x03" + _write_vlq(len(nb)) + nb
    out += _write_vlq(0) + bytes([0xC0 | (tr.channel & 0x0F), tr.program & 0x7F])

    events: list[tuple[int, int, int, int]] = []
    for n in tr.notes:
        events.append((n.start, 1, n.pitch, n.velocity))       # on
        events.append((n.start + n.dur, 0, n.pitch, 0))        # off
    events.sort(key=lambda e: (e[0], e[1]))

    last = 0
    for t, on, pitch, vel in events:
        out += _write_vlq(max(0, t - last))
        out += bytes([(0x90 if on else 0x80) | (tr.channel & 0x0F),
                      pitch & 0x7F, vel & 0x7F])
        last = t
    out += _write_vlq(0) + b"\xFF\x2F\x00"
    return bytes(out)


# --------------------------------------------------------------------------
# Pontes com o Project do Chipforge
# --------------------------------------------------------------------------
def project_to_midi(project) -> MidiFile:
    mf = MidiFile(ppq=project.ppq, tempo_bpm=project.bpm,
                  numerator=project.beats_per_bar, denominator=project.beat_unit)
    ch = 0
    for tr in project.tracks:
        clips = [c for c in project.arrangement if c.track_id == tr.id]
        notes: list[MidiNote] = []
        drum = tr.instrument.is_drum
        channel = 9 if drum else ch
        for c in clips:
            pat = project.patterns.get(c.pattern_id)
            if not pat:
                continue
            for n in pat.notes:
                notes.append(MidiNote(n.start + c.start, n.dur, n.pitch,
                                      n.velocity, channel))
        if not notes:
            continue
        prog = 0
        low = tr.instrument.name.lower() + " " + tr.instrument.osc
        for k, v in GM_PROGRAM_HINTS.items():
            if k in low:
                prog = v
                break
        mf.tracks.append(MidiTrack(tr.name, prog, channel, sorted(notes, key=lambda x: x.start)))
        if not drum:
            ch += 1
            if ch == 9:
                ch = 10
    return mf


def midi_to_project(mf: MidiFile, project_cls=None, instrument_for=None):
    """Importa um MIDI como projeto novo: 1 track MIDI = 1 track + 1 pattern."""
    from ..core.project import Note, Project
    from ..audio.instrument import factory_by_name
    P = project_cls or Project
    p = P(name="Importado", bpm=mf.tempo_bpm, ppq=mf.ppq,
          beats_per_bar=mf.numerator, beat_unit=mf.denominator)

    def default_instrument(mt: MidiTrack):
        if instrument_for:
            return instrument_for(mt)
        if mt.is_drum:
            return factory_by_name("Snare")
        pr = mt.program
        if pr in range(32, 40):
            return factory_by_name("NES Triangle Bass")
        if pr in range(16, 24):
            return factory_by_name("GB Organ")
        if pr in range(88, 96):
            return factory_by_name("Warm Pad")
        return factory_by_name("NES Lead 25%")

    for mt in mf.tracks:
        inst = default_instrument(mt)
        inst.name = mt.name or inst.name
        inst.is_drum = mt.is_drum
        track = p.new_track(mt.name or f"Track {len(p.tracks)+1}", inst)
        end = max((n.start + n.dur for n in mt.notes), default=0)
        bars = max(1, -(-end // p.ticks_per_bar))
        pat = p.new_pattern(f"{track.name}", bars=int(bars))
        for n in mt.notes:
            pat.notes.append(Note(n.start, n.dur, n.pitch, n.velocity))
        p.place(pat, track, 0)
    return p
