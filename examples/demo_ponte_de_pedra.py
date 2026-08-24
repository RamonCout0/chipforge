"""
"Ponte de Pedra" — tema de vilarejo, 24 compassos, em Re dorico.

O ponto da demo e mostrar os dois mundos do Chipforge na mesma faixa:

    compassos  1-8    so chiptune       — a versao "cartucho"
    compassos  9-16   so acustico       — os mesmos acordes com o banco GM
    compassos 17-24   tudo junto        — e assim que trilha de jogo cresce

O banco de instrumentos e baixado sozinho na primeira execucao
(GeneralUser GS, ~28 MB, licenca livre para uso musical).

    python3 examples/demo_ponte_de_pedra.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chipforge.audio import engine
from chipforge.audio import sampler as SF
from chipforge.audio.instrument import factory_by_name
from chipforge.core import generators as G
from chipforge.core import harmony as H
from chipforge.core import theory as T
from chipforge.core.project import Project
from chipforge.midiio import library as L
from chipforge.midiio import smf

Q = 480
BAR = Q * 4
D = T.name_to_midi("D3")          # tonica: Re


# ---------------------------------------------------------------- melodia
# (altura, compasso, batida, duracao em batidas) — escrita na mao, nota a nota
MELODIA = [
    # frase 1 — apresenta o tema
    ("D5", 0, 0.0, 1.0), ("F5", 0, 1.0, 0.5), ("E5", 0, 1.5, 0.5),
    ("D5", 0, 2.0, 1.0), ("A4", 0, 3.0, 1.0),
    ("B4", 1, 0.0, 1.0), ("D5", 1, 1.0, 1.0), ("C5", 1, 2.0, 2.0),
    ("A4", 2, 0.0, 0.5), ("C5", 2, 0.5, 0.5), ("D5", 2, 1.0, 1.0),
    ("F5", 2, 2.0, 1.0), ("E5", 2, 3.0, 1.0),
    ("D5", 3, 0.0, 1.5), ("C5", 3, 1.5, 0.5), ("A4", 3, 2.0, 2.0),
    # frase 2 — sobe, respira e resolve
    ("D5", 4, 0.0, 0.5), ("E5", 4, 0.5, 0.5), ("F5", 4, 1.0, 1.0),
    ("A5", 4, 2.0, 1.5), ("G5", 4, 3.5, 0.5),
    ("F5", 5, 0.0, 1.0), ("E5", 5, 1.0, 1.0), ("D5", 5, 2.0, 2.0),
    ("C5", 6, 0.0, 1.0), ("D5", 6, 1.0, 1.0), ("E5", 6, 2.0, 1.0),
    ("F5", 6, 3.0, 1.0),
    ("C5", 7, 0.0, 1.0), ("A4", 7, 1.0, 1.0), ("D5", 7, 2.0, 2.0),
]

# contracanto para a ultima secao: so as notas altas, bem espacado
CONTRACANTO = [
    ("A5", 0, 2.0, 2.0), ("D6", 2, 2.0, 2.0),
    ("A5", 4, 1.0, 1.0), ("D6", 4, 2.0, 2.0),
    ("C6", 6, 0.0, 2.0), ("A5", 7, 2.0, 2.0),
]


def escrever(pat, linhas, velocity=104, corte=25):
    for nome, compasso, batida, dur in linhas:
        pat.add(T.name_to_midi(nome),
                compasso * BAR + int(batida * Q),
                max(60, int(dur * Q) - corte), velocity)
    return pat


def build(sf2: Path) -> Project:
    p = Project("Ponte de Pedra", bpm=118, bit_depth=16)
    p.key_root, p.scale = 2, "dorian"          # Re dorico
    banco = SF.open_soundfont(sf2)

    def gm(nome, **kw):
        return SF.instrument(banco, nome, **kw)

    # ---- acordes: i - IV - i - VII (o IV maior e a assinatura do dorico)
    acordes = [
        [D, D + 3, D + 7],          # Dm
        [D + 5, D + 9, D + 12],     # G
        [D, D + 3, D + 7],          # Dm
        [D - 2, D + 2, D + 5],      # C
    ] * 2                            # oito compassos

    # O acompanhamento mora SEMPRE abaixo do tema: a melodia vive de A4 pra
    # cima, entao arpejo e coro ficam presos na oitava que comeca em Sol 3.
    acomp = [G.voice_in_octave(ch, T.name_to_midi("G3")) for ch in acordes]

    # ================================================================ tracks
    chip_lead = p.new_track("Chip Lead", factory_by_name("NES Lead 25%"))
    chip_lead.instrument.vibrato_depth_cents = 24
    chip_arp = p.new_track("Chip Arpejo", factory_by_name("Pluck Harp"))
    chip_arp.volume, chip_arp.pan = 0.45, -0.3
    chip_bass = p.new_track("Chip Baixo", factory_by_name("NES Triangle Bass"))
    bateria = p.new_track("Bateria", factory_by_name("Kick"))
    bateria.instrument.is_drum = True
    bateria.volume = 0.75

    flauta = p.new_track("Flauta", gm("Flute", gain=0.85))
    flauta.volume = 0.9
    violao = p.new_track("Violao", gm("Nylon Guitar", gain=0.8))
    violao.volume, violao.pan = 0.6, 0.32
    contrabaixo = p.new_track("Contrabaixo", gm("Acoustic Bass", gain=0.95))
    coro = p.new_track("Coro", gm("Concert Choir", gain=0.55))
    coro.volume, coro.pan = 0.4, -0.15
    sino = p.new_track("Glockenspiel", gm("Glockenspiel", gain=0.5))
    sino.volume, sino.pan = 0.45, 0.4

    # =============================================================== patterns
    pat_mel = escrever(p.new_pattern("Melodia", bars=8), MELODIA)
    pat_contra = escrever(p.new_pattern("Contracanto", bars=8), CONTRACANTO, 86)

    pat_arp_chip = G.fill_pattern(
        p.new_pattern("Arpejo chip", bars=8),
        G.arpeggiate(acomp, Q, rate=4, mode="updown", octaves=1, velocity=74))
    pat_arp_violao = G.fill_pattern(
        p.new_pattern("Arpejo violao", bars=8),
        G.arpeggiate(acomp, Q, rate=2, mode="up", octaves=1, velocity=88))

    pat_baixo = G.fill_pattern(
        p.new_pattern("Baixo", bars=8),
        G.bassline(acordes, Q, style="root_fifth"))
    pat_baixo_ac = G.fill_pattern(
        p.new_pattern("Baixo acustico", bars=8),
        G.bassline(acordes, Q, style="walking", octave_shift=-24, velocity=100))

    pat_bat = G.fill_pattern(p.new_pattern("Bateria", bars=8),
                             G.drums(8, Q, style="chip_rock"))

    # Coro sem terca — so fundamental, quinta e oitava. Pad embaixo de melodia
    # nao precisa da terca, e e ela que costuma bater de frente com o tema.
    pat_coro = p.new_pattern("Coro", bars=8)
    for i, ch in enumerate(acordes):
        raiz = G.voice_in_octave([min(ch)], T.name_to_midi("G3"))[0]
        # a quinta vai ABAIXO da fundamental, nunca acima: assim o pad nunca
        # sobe para o registro onde a flauta esta cantando
        for nota in (raiz, raiz - 5):
            pat_coro.add(nota, i * BAR, BAR - 60, 68)

    # ============================================================ arranjo
    #  A (1-8) chiptune | B (9-16) acustico | C (17-24) os dois
    A, B, C = 0, 8, 16
    p.place(pat_mel, chip_lead, A);        p.place(pat_mel, chip_lead, C)
    p.place(pat_arp_chip, chip_arp, A);    p.place(pat_arp_chip, chip_arp, C)
    p.place(pat_baixo, chip_bass, A);      p.place(pat_baixo, chip_bass, C)
    p.place(pat_bat, bateria, A);          p.place(pat_bat, bateria, C)

    p.place(pat_mel, flauta, B);           p.place(pat_mel, flauta, C)
    p.place(pat_arp_violao, violao, B);    p.place(pat_arp_violao, violao, C)
    p.place(pat_baixo_ac, contrabaixo, B); p.place(pat_baixo_ac, contrabaixo, C)
    p.place(pat_coro, coro, B);            p.place(pat_coro, coro, C)
    p.place(pat_contra, sino, C)
    return p


def main():
    out = Path(__file__).parent
    print("Buscando o banco de instrumentos...")
    sf2 = L.ensure("generaluser_gs",
                   lambda got, tot: print(f"  {got/1e6:5.1f} MB", end="\r", flush=True))
    banco = SF.open_soundfont(sf2)
    print(f"Banco: {banco.name} — {len(banco.presets)} presets, "
          f"{len(banco.samples)} samples")

    p = build(sf2)
    print(f"Projeto: {len(p.tracks)} tracks, {len(p.patterns)} patterns, "
          f"{p.length_ticks // p.ticks_per_bar} compassos, {p.bpm:.0f} BPM")

    rep = H.analyze(p.harmony_events(), beats_per_chord=4.0)
    print("\n" + "=" * 64)
    print(rep.summary_pt())
    print("-" * 64)
    for tip in H.suggest_fix(rep):
        print("  ->", tip)
    print("=" * 64 + "\n")

    audio = engine.render_project(p)
    wav = engine.export_wav(audio, out / "ponte_de_pedra", p.sample_rate, 16)
    chip = p.save(out / "ponte_de_pedra")
    mid = smf.write_midi(smf.project_to_midi(p), out / "ponte_de_pedra")
    print(f"WAV  : {wav}  ({len(audio)/p.sample_rate:.1f}s)")
    print(f"CHIP : {chip}")
    print(f"MIDI : {mid}")


if __name__ == "__main__":
    main()
