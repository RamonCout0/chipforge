"""Suite de testes do nucleo. Roda com: python3 -m pytest tests -q"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chipforge.audio import engine, oscillators as OSC
from chipforge.audio.instrument import Instrument, all_factory_names, factory_by_name
from chipforge.audio.sampler import SamplerInstrument, SoundFont
from chipforge.core import harmony as H
from chipforge.core import theory as T
from chipforge.core.project import Project
from chipforge.midiio import smf

SR = 44100


# ---------------------------------------------------------------- teoria
def test_note_names_roundtrip():
    for n in range(24, 108):
        assert T.name_to_midi(T.midi_to_name(n)) == n


def test_frequencies():
    assert T.midi_to_hz(69) == pytest.approx(440.0)
    assert T.midi_to_hz(60) == pytest.approx(261.6256, rel=1e-4)
    assert T.hz_to_midi(440.0) == pytest.approx(69.0)


@pytest.mark.parametrize("notes,expected", [
    ([60, 64, 67], "C"),
    ([60, 63, 67], "Cmin"),
    ([60, 64, 67, 70], "C7"),
    ([60, 64, 67, 71], "Cmaj7"),
    ([60, 63, 66], "Cdim"),
    ([60, 67], "C5"),
    ([62, 65, 69], "Dmin"),
])
def test_chord_detection(notes, expected):
    assert T.detect_chord(notes).name == expected


def test_key_detection_major_and_minor():
    w = [0.0] * 12
    for n in [60, 62, 64, 65, 67, 69, 71, 60, 67, 64, 60]:
        w[n % 12] += 1
    k = T.detect_key(w)
    assert k.root == 0 and k.mode == "major"

    w = [0.0] * 12
    for n in [57, 59, 60, 62, 64, 65, 67, 57, 64, 60, 57]:
        w[n % 12] += 1
    k = T.detect_key(w)
    assert k.root == 9 and k.mode == "natural_minor"


def test_scale_lock_pulls_into_scale():
    allowed = T.scale_pitch_classes(0, "major")
    for n in range(48, 84):
        q = T.quantize_to_scale(n, 0, "major")
        assert q % 12 in allowed
        assert abs(q - n) <= 1


def test_voice_leading_progression_is_smooth():
    chords = T.build_progression(0, "major", [0, 4, 5, 3])
    moves = [abs(sum(b) / 3 - sum(a) / 3) for a, b in zip(chords, chords[1:])]
    assert max(moves) < 7          # nada de pulo de oitava entre acordes


# -------------------------------------------------------------- harmonia
def _chord_notes(pitches, start=0.0, dur=2.0):
    return [H.NoteEvent(start, dur, p) for p in pitches]


def test_meter_rewards_clean_progression():
    notes = []
    for i, ch in enumerate(T.build_progression(0, "major", [0, 4, 5, 3])):
        notes += _chord_notes(ch, i * 2.0, 2.0)
    rep = H.analyze(notes, beats_per_chord=2.0)
    assert rep.score > 85
    assert rep.key.root == 0 and rep.key.mode == "major"
    assert not [i for i in rep.issues if i.severity == 2]


def test_meter_punishes_semitone_cluster():
    notes = _chord_notes([60, 61, 62, 63], 0.0, 4.0)
    rep = H.analyze(notes)
    assert rep.score < 55
    assert any(i.kind == "dissonancia" for i in rep.issues)


def test_meter_flags_muddy_low_thirds():
    notes = _chord_notes([36, 39, 43], 0.0, 4.0)
    rep = H.analyze(notes)
    assert rep.metrics["grave_limpo"] < 0.5
    assert any(i.kind == "grave_embolado" for i in rep.issues)


def test_meter_flags_out_of_key_note():
    notes = []
    for i, ch in enumerate(T.build_progression(0, "major", [0, 4, 0, 4])):
        notes += _chord_notes(ch, i * 2.0, 2.0)
    notes.append(H.NoteEvent(0.0, 2.0, 61))       # C# em Do maior
    rep = H.analyze(notes, key=T.KeyGuess(0, "major", 0.95))
    assert any(i.kind == "fora_da_escala" for i in rep.issues)


def test_meter_strict_mode_catches_parallel_fifths():
    a = _chord_notes([60, 67], 0.0, 1.0)
    b = _chord_notes([62, 69], 1.0, 1.0)
    rep = H.analyze(a + b, style="strict")
    assert any(i.kind == "paralelas" for i in rep.issues)
    assert rep.metrics["conducao"] < 1.0


def test_meter_empty_is_perfect():
    assert H.analyze([]).score == 100.0


# ------------------------------------------------------------- audio
def test_oscillators_are_bounded_and_right_length():
    freq = np.full(SR // 10, 440.0)
    for fn in (lambda f: OSC.pulse(f, SR, 0.25),
               lambda f: OSC.triangle(f, SR),
               lambda f: OSC.saw(f, SR),
               lambda f: OSC.sine(f, SR),
               lambda f: OSC.noise(f, SR),
               lambda f: OSC.wavetable(f, SR, OSC.WAVETABLES["gb_organ"])):
        s = fn(freq)
        assert len(s) == len(freq)
        assert np.all(np.abs(s) <= 1.6)
        assert np.abs(s).max() > 0.1


def test_pulse_frequency_is_accurate():
    freq = np.full(SR, 440.0)
    s = OSC.pulse(freq, SR, 0.5, antialias=False)
    crossings = np.sum(np.diff(np.sign(s)) > 0)
    assert crossings == pytest.approx(440, rel=0.02)


def test_triangle_is_quantized_like_the_nes():
    s = OSC.triangle(np.full(SR // 20, 220.0), SR, steps=16)
    assert len(np.unique(np.round(s, 5))) <= 17


def test_bitcrush_reduces_distinct_levels():
    from chipforge.audio.effects import bitcrush
    x = np.linspace(-1, 1, 5000, dtype=np.float32)
    assert len(np.unique(bitcrush(x, 4))) == 16
    assert len(np.unique(bitcrush(x, 8))) <= 256


def test_every_factory_instrument_renders():
    for name in all_factory_names():
        s = factory_by_name(name).render_note(60, 0.25, SR)
        assert len(s) > 1000
        assert np.isfinite(s).all()
        assert np.abs(s).max() > 0.0


def test_arpeggio_produces_multiple_pitches():
    inst = Instrument("arp", "pulse50", arpeggio=(0, 4, 7), arp_rate_hz=20)
    s = inst.render_note(60, 0.6, SR)
    # divide em blocos e conta cruzamentos por bloco: devem diferir
    blocks = [np.sum(np.diff(np.sign(s[i:i + 2000])) > 0)
              for i in (2000, 6000, 10000)]
    assert len(set(blocks)) > 1


# ------------------------------------------------------------- projeto
def _demo_project() -> Project:
    p = Project("t", bpm=120, bit_depth=8)
    lead = p.new_track("Lead", factory_by_name("NES Lead 25%"))
    bass = p.new_track("Bass", factory_by_name("NES Triangle Bass"))
    pl = p.new_pattern("L", bars=1)
    pb = p.new_pattern("B", bars=1)
    for i, n in enumerate([72, 74, 76, 79]):
        pl.add(n, i * 480, 460)
    for i, n in enumerate([48, 55]):
        pb.add(n, i * 960, 940)
    p.place(pl, lead, 0)
    p.place(pb, bass, 0)
    return p


def test_project_save_load_roundtrip(tmp_path):
    p = _demo_project()
    f = p.save(tmp_path / "x")
    q = Project.load(f)
    assert q.bpm == p.bpm and q.bit_depth == p.bit_depth
    assert len(q.tracks) == len(p.tracks)
    assert [n.pitch for n in q.patterns["p1"].notes] == \
           [n.pitch for n in p.patterns["p1"].notes]
    assert q.tracks[0].instrument.osc == p.tracks[0].instrument.osc


def test_tick_time_conversions():
    p = Project(bpm=120, ppq=480)
    assert p.ticks_to_seconds(480) == pytest.approx(0.5)
    assert p.seconds_to_ticks(1.0) == 960
    assert p.ticks_per_bar == 1920


def test_mute_and_solo():
    p = _demo_project()
    p.tracks[0].mute = True
    assert [t.name for t in p.audible_tracks] == ["Bass"]
    p.tracks[0].mute = False
    p.tracks[1].solo = True
    assert [t.name for t in p.audible_tracks] == ["Bass"]


def test_render_project_produces_audio():
    p = _demo_project()
    a = engine.render_project(p)
    assert a.ndim == 2 and a.shape[1] == 2
    assert np.isfinite(a).all()
    assert np.abs(a).max() > 0.05
    assert np.abs(a).max() <= 1.0


def test_export_wav_is_readable(tmp_path):
    import wave
    p = _demo_project()
    f = engine.export_wav(engine.render_project(p), tmp_path / "o", p.sample_rate, 16)
    with wave.open(str(f)) as w:
        assert w.getnchannels() == 2
        assert w.getframerate() == p.sample_rate
        assert w.getnframes() > 1000


def test_bit_depth_changes_the_signal():
    p = _demo_project()
    p.bit_depth = 24
    clean = engine.render_project(p)
    p.bit_depth = 4
    crushed = engine.render_project(p)
    assert len(np.unique(crushed)) < len(np.unique(clean))


# ---------------------------------------------------------------- MIDI
def test_midi_roundtrip_preserves_notes(tmp_path):
    p = _demo_project()
    mf = smf.project_to_midi(p)
    f = smf.write_midi(mf, tmp_path / "a")
    back = smf.read_midi(f)
    assert back.tempo_bpm == pytest.approx(p.bpm, rel=1e-3)
    assert back.ppq == p.ppq
    orig = sorted((n.start, n.pitch, n.dur) for t in mf.tracks for n in t.notes)
    got = sorted((n.start, n.pitch, n.dur) for t in back.tracks for n in t.notes)
    assert orig == got


def test_midi_import_builds_playable_project(tmp_path):
    p = _demo_project()
    f = smf.write_midi(smf.project_to_midi(p), tmp_path / "b")
    q = smf.midi_to_project(smf.read_midi(f))
    assert len(q.tracks) == 2
    a = engine.render_project(q)
    assert np.abs(a).max() > 0.01


def test_vlq_encoding():
    from chipforge.midiio.smf import _read_vlq, _write_vlq
    for v in (0, 1, 127, 128, 8192, 100000, 0x0FFFFFFF):
        assert _read_vlq(_write_vlq(v), 0)[0] == v


# ------------------------------------------------------------ soundfont
@pytest.fixture(scope="module")
def test_sf2(tmp_path_factory):
    sys.path.insert(0, str(Path(__file__).parent))
    from make_test_sf2 import build
    return build(tmp_path_factory.mktemp("sf") / "bank.sf2")


def test_soundfont_parses_presets_and_samples(test_sf2):
    sf = SoundFont(test_sf2)
    assert sf.name == "Chipforge Test Bank"
    assert [s.name for s in sf.samples] == ["Sine A4", "Noise Hit"]
    assert len(sf.presets) == 2
    assert sf.presets[1].is_drum


def test_sampler_pitches_correctly(test_sf2):
    sf = SoundFont(test_sf2)
    inst = SamplerInstrument(sf, sf.presets[0])
    s = inst.render_note(60, 0.4, SR)          # C4 = 261.63 Hz
    seg = s[3000:15000]
    crossings = np.sum(np.diff(np.sign(seg)) > 0)
    hz = crossings / (len(seg) / SR)
    assert hz == pytest.approx(261.63, rel=0.05)


def test_meter_treats_short_offbeat_clash_as_passing_tone():
    """Semicolcheia de arpejo roçando na melodia nao pode virar 'problema'."""
    held = [H.NoteEvent(0.0, 4.0, 60), H.NoteEvent(0.0, 4.0, 64)]
    passing = [H.NoteEvent(0.25, 0.25, 61)]          # curta, fora do tempo forte
    rep = H.analyze(held + passing)
    kinds = {i.kind for i in rep.issues}
    assert "passagem" in kinds
    assert not [i for i in rep.issues if i.severity == 2]


def test_meter_still_punishes_sustained_clash_on_the_beat():
    notes = [H.NoteEvent(0.0, 4.0, 60), H.NoteEvent(0.0, 4.0, 61)]
    rep = H.analyze(notes)
    assert [i for i in rep.issues if i.severity == 2]


def test_drum_track_maps_each_pitch_to_its_own_drum():
    """Bumbo, caixa e chimbal na mesma track precisam soar diferente."""
    from chipforge.core.project import Note
    p = Project("d", bpm=120)
    tr = p.new_track("Bateria", factory_by_name("Kick"))
    tr.instrument.is_drum = True
    pat = p.new_pattern("D", bars=1)
    pat.notes += [Note(0, 120, 36), Note(480, 120, 38), Note(960, 120, 42)]
    p.place(pat, tr, 0)
    a = engine.render_project(p)
    sr = p.sample_rate
    hits = [a[int(t * sr):int(t * sr) + 3000] for t in (0.0, 0.5, 1.0)]
    energias = [float(np.sqrt((h ** 2).mean())) for h in hits]
    assert all(e > 0.001 for e in energias)
    # o chimbal (ruido curto e agudo) nao pode ter a mesma energia do bumbo
    assert abs(energias[0] - energias[2]) > 0.01


def test_noise_is_periodic_and_deterministic():
    freq = np.full(SR // 4, 900.0)
    a = OSC.noise(freq, SR)
    b = OSC.noise(freq, SR)
    assert np.array_equal(a, b)
    assert set(np.unique(a).tolist()) == {-1.0, 1.0}


def test_analyze_sections_finds_a_key_per_section():
    ev = []
    for i, ch in enumerate(T.build_progression(2, "natural_minor", [0, 5, 2, 6])):
        ev += [H.NoteEvent(i * 4.0, 4.0, p) for p in ch]
    for i, ch in enumerate(T.build_progression(5, "natural_minor", [0, 5, 2, 6])):
        ev += [H.NoteEvent(16.0 + i * 4.0, 4.0, p) for p in ch]
    secs = H.analyze_sections(ev, [("A", 0.0, 16.0), ("B", 16.0, 32.0)],
                              beats_per_chord=4.0)
    assert len(secs) == 2
    assert secs[0].report.key.root == 2      # Re menor
    assert secs[1].report.key.root == 5      # Fa menor
    assert "A" in H.sections_table_pt(secs)


def test_meter_calls_spread_diatonic_seventh_colour_not_error():
    """Dm(add9): fa numa voz interna e mi na melodia. Cor, nao erro."""
    notes = [H.NoteEvent(0.0, 4.0, T.name_to_midi("D4")),
             H.NoteEvent(0.0, 4.0, T.name_to_midi("F4")),
             H.NoteEvent(0.0, 4.0, T.name_to_midi("A4")),
             H.NoteEvent(0.0, 4.0, T.name_to_midi("E5"))]
    rep = H.analyze(notes, key=T.KeyGuess(2, "natural_minor", 0.9))
    assert not [i for i in rep.issues if i.severity == 2]
    assert any(i.kind == "cor" for i in rep.issues)


def test_meter_still_rejects_tight_and_chromatic_clashes():
    apertado = [H.NoteEvent(0.0, 4.0, 65), H.NoteEvent(0.0, 4.0, 64)]
    assert [i for i in H.analyze(apertado).issues if i.severity == 2]
    cromatico = [H.NoteEvent(0.0, 4.0, T.name_to_midi("C#4")),
                 H.NoteEvent(0.0, 4.0, T.name_to_midi("D5"))]
    rep = H.analyze(cromatico, key=T.KeyGuess(2, "natural_minor", 0.9))
    assert [i for i in rep.issues if i.severity == 2]


def test_issues_name_the_guilty_tracks():
    notes = [H.NoteEvent(0.0, 4.0, 60, track="Cordas"),
             H.NoteEvent(0.0, 4.0, 61, track="Chip Lead")]
    rep = H.analyze(notes)
    duros = [i for i in rep.issues if i.severity == 2]
    assert duros and "Cordas" in duros[0].culpados
    assert "Chip Lead" in duros[0].culpados
