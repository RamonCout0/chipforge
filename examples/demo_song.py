"""
Monta um loop de trilha de jogo do zero, salva .chip, exporta WAV e MIDI,
e imprime o laudo do medidor de harmonia.

    python3 examples/demo_song.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chipforge.audio import engine
from chipforge.audio.instrument import factory_by_name
from chipforge.core import harmony as H
from chipforge.core import generators as G
from chipforge.core import theory as T
from chipforge.core.project import Project
from chipforge.midiio import smf

Q = 480          # ppq
BAR = Q * 4


def build() -> Project:
    p = Project("Vilarejo ao entardecer", bpm=124, bit_depth=8)
    p.key_root, p.scale = 9, "natural_minor"        # La menor

    lead = p.new_track("Lead", factory_by_name("NES Lead 25%"))
    lead.instrument.vibrato_depth_cents = 26
    harm = p.new_track("Arpejo", factory_by_name("Pluck Harp"))
    harm.volume = 0.5
    harm.pan = -0.35
    bass = p.new_track("Baixo", factory_by_name("NES Triangle Bass"))
    drums = p.new_track("Bateria", factory_by_name("Kick"))
    drums.instrument.is_drum = True

    # ---- progressao i - VI - III - VII (o classico "aventura melancolica")
    degrees = [0, 5, 2, 6]
    chords = T.build_progression(p.key_root, "natural_minor", degrees, octave=3)

    # Arpejo ESCRITO nota a nota: assim ele acompanha cada acorde de verdade
    # (um instrumento com arpejo fixo tocaria sempre a mesma qualidade).
    pat_h = p.new_pattern("Arpejo A", bars=4)
    G.fill_pattern(pat_h, G.arpeggiate(chords, Q, bars_per_chord=1,
                                       rate=4, mode="updown", octaves=2,
                                       velocity=78))

    pat_b = p.new_pattern("Baixo A", bars=4)
    G.fill_pattern(pat_b, G.bassline(chords, Q, style="root_fifth"))

    # ---- melodia: desce e resolve, com uma nota longa no fim de cada frase
    melody = [
        (69, 0, Q), (72, Q, Q // 2), (74, Q + Q // 2, Q // 2),
        (76, Q * 2, Q), (74, Q * 3, Q),
        (72, BAR, Q), (69, BAR + Q, Q), (67, BAR + Q * 2, Q * 2),
        (69, BAR * 2, Q // 2), (71, BAR * 2 + Q // 2, Q // 2),
        (72, BAR * 2 + Q, Q), (76, BAR * 2 + Q * 2, Q),
        (79, BAR * 2 + Q * 3, Q),
        (76, BAR * 3, Q), (74, BAR * 3 + Q, Q), (69, BAR * 3 + Q * 2, Q * 2),
    ]
    pat_l = p.new_pattern("Lead A", bars=4)
    for pitch, start, dur in melody:
        pat_l.add(pitch, start, dur - 25, 108)

    # ---- bateria: kick 1 e 3, caixa 2 e 4, chimbal em colcheias
    pat_d = p.new_pattern("Bateria A", bars=4)
    G.fill_pattern(pat_d, G.drums(4, Q, style="chip_rock"))

    for pat, tr in [(pat_l, lead), (pat_h, harm), (pat_b, bass), (pat_d, drums)]:
        p.place(pat, tr, 0)
        p.place(pat, tr, 4)          # dois ciclos = 8 compassos
    return p


def main():
    out = Path(__file__).parent
    p = build()

    rep = H.analyze(p.harmony_events(), beats_per_chord=4.0)
    print("=" * 62)
    print(rep.summary_pt())
    print("-" * 62)
    for tip in H.suggest_fix(rep):
        print("  ->", tip)
    print("=" * 62)

    audio = engine.render_project(p)
    wav = engine.export_wav(audio, out / "demo_vilarejo", p.sample_rate, 16)
    chip = p.save(out / "demo_vilarejo")
    mid = smf.write_midi(smf.project_to_midi(p), out / "demo_vilarejo")
    dur = len(audio) / p.sample_rate
    print(f"WAV  : {wav}  ({dur:.1f}s)")
    print(f"CHIP : {chip}")
    print(f"MIDI : {mid}")


if __name__ == "__main__":
    main()
