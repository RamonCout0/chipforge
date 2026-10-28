"""
Testes do modo PS1, da partitura em texto (MML), do Estudio sem interface e
das correcoes de robustez (projeto, biblioteca, render).

O que estes testes protegem:
  * o reverb do SPU em blocos tem que ser IDENTICO a formula amostra-a-amostra
    do psx-spx — e a otimizacao que permite renderizar musica inteira;
  * a MML tem que pegar compasso que nao fecha, que e o erro mais comum de
    quem escreve partitura em texto;
  * salvar um projeto aberto sem o banco de samples nao pode apagar o
    instrumento original;
  * apagar um pattern e criar outro nao pode sobrescrever um que ainda toca.
"""
from __future__ import annotations

import io
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chipforge.audio import engine, psx
from chipforge.audio.instrument import Instrument, factory_by_name
from chipforge.core import mml
from chipforge.core.project import Note, Project
from chipforge.estudio import Estudio, ErroEstudio

PPQ = 480


# ==========================================================================
# MML
# ==========================================================================
def test_mml_duracoes_pontos_e_ligaduras():
    r = mml.parse("o4 c4 d8. e16 f2^8 r8", PPQ)
    assert [(n.pitch, n.start) for n in r.notas] == [(60, 0), (62, 480),
                                                     (64, 840), (65, 960)]
    assert r.fim == 960 + 960 + 240 + 240
    # gate padrao q7: a nota soa 7/8 do valor escrito
    assert r.notas[0].dur == 420


def test_mml_oitavas_acidentes_e_acorde():
    r = mml.parse("o5 c+ > d- < < b- [c e g]2", PPQ, duracao=4)
    assert [n.pitch for n in r.notas[:3]] == [73, 85, 70]
    acorde = [n for n in r.notas if n.start == 3 * PPQ]
    assert sorted(n.pitch for n in acorde) == [60, 64, 67]


def test_mml_confere_compasso():
    r = mml.parse("o4 l8 c d e f g a b | c1 | c2 |", PPQ)
    assert len(r.avisos) == 2
    assert "compasso 1" in r.avisos[0] and "3.5" in r.avisos[0]
    assert "compasso 3" in r.avisos[1]


def test_mml_repeticao_e_ligadura_com_e_comercial():
    r = mml.parse("o4 (c8 d8)3 e4&e4", PPQ, gate=8)
    assert len(r.notas) == 7
    assert r.notas[-1].dur == 960


def test_mml_erro_mostra_onde():
    with pytest.raises(ValueError, match="perto de"):
        mml.parse("o4 c4 z4")


def test_mml_ida_e_volta():
    texto = "o5 b8 o6 e8 f+8 g8 f+4 e8 d8 | e2."
    r = mml.parse(texto, PPQ)
    assert mml.parse(mml.para_mml(r.notas, PPQ), PPQ).notas == r.notas


def test_cifras():
    lst = mml.parse_cifras("Am7 | F G | % | Dm/F -", PPQ)
    assert [c.nome for _, _, c in lst] == ["Am7", "F", "G", "F", "G", "Dm/F"]
    ini, dur, dmf = lst[-1]
    assert dur == 4 * PPQ and dmf.baixo == 5
    assert mml.cifra("B7").classes == [3, 6, 9, 11]
    with pytest.raises(ValueError):
        mml.cifra("Hm7")


def test_grade_de_bateria():
    notas, fim = mml.parse_bateria("k: x...x...|x...x...\ns: ..X.", PPQ)
    assert fim == 16 * PPQ // 4
    bumbos = [n for n in notas if n.pitch == 36]
    assert [n.start for n in bumbos] == [0, 480, 960, 1440]
    assert [n for n in notas if n.pitch == 38][0].velocity > 110


# ==========================================================================
# PS1 / SPU
# ==========================================================================
def _reverb_referencia(Lin, Rin, p):
    """A formula do psx-spx, literalmente, amostra por amostra."""
    N = p["N"]
    buf = np.zeros(N)
    pos = 0
    outL, outR = [], []

    def A(a):
        return (pos + a) % N
    for i in range(len(Lin)):
        for w, x, d in (("mLSAME", Lin[i], p["dLSAME"]), ("mRSAME", Rin[i], p["dRSAME"]),
                        ("mLDIFF", Lin[i], p["dRDIFF"]), ("mRDIFF", Rin[i], p["dLDIFF"])):
            prev = buf[A(p[w] - 1)]
            buf[A(p[w])] = np.clip((x + buf[A(d)] * p["vWALL"] - prev) * p["vIIR"]
                                   + prev, -1, 1)
        saida = []
        for s in "LR":
            out = sum(p[f"vCOMB{k}"] * buf[A(p[f"m{s}COMB{k}"])] for k in (1, 2, 3, 4))
            for a in (1, 2):
                w = f"m{s}APF{a}"
                at = buf[A(p[w] - p[f"dAPF{a}"])]
                buf[A(p[w])] = np.clip(out - p[f"vAPF{a}"] * at, -1, 1)
                out = at + buf[A(p[w])] * p[f"vAPF{a}"]
            saida.append(out)
        outL.append(saida[0]); outR.append(saida[1])
        pos = (pos + 1) % N
    return np.array(outL), np.array(outR)


@pytest.mark.parametrize("preset", ["room", "studio_small", "hall"])
def test_reverb_spu_em_blocos_igual_a_formula(preset, monkeypatch):
    monkeypatch.setattr(psx, "reamostrar", lambda x, de, para: np.asarray(x, np.float32))
    rng = np.random.default_rng(1)
    n = 12000
    x = np.zeros((n, 2))
    x[:1500] = rng.standard_normal((1500, 2)) * 0.15
    y = psx.reverb(x, 22050, preset, wet=1.0)
    L, R = _reverb_referencia(x[:, 0], x[:, 1], psx._preset(preset))
    assert np.abs(y[:, 0] - L).max() < 1e-5
    assert np.abs(y[:, 1] - R).max() < 1e-5


def test_reverb_spu_decai_e_e_estavel():
    x = np.zeros((44100 * 3, 2), np.float32)
    x[100] = 0.8
    for preset in psx.PRESETS_REVERB:
        y = psx.reverb(x, 44100, preset, wet=1.0)
        assert np.isfinite(y).all()
        assert np.abs(y[-4410:]).max() < np.abs(y).max()


def test_adpcm_qualidade_de_4_bits():
    t = np.arange(22050) / 22050
    x = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    y = psx.adpcm_ida_e_volta(x)
    snr = 10 * np.log10((x ** 2).mean() / ((y - x) ** 2).mean())
    assert 20 < snr < 60          # tem erro (e o ponto) mas e musical
    assert len(y) == len(x)


def test_gaussiana_soma_um_e_abafa():
    w = psx.pesos_gaussianos(np.linspace(0, 0.999, 50))
    assert np.allclose(sum(w), 1.0, atol=1e-5)
    w0 = psx.pesos_gaussianos(np.array([0.0]))
    # bate com a tabela do SPU nas pontas: 372, 1305, 374, 0 (/2048)
    assert abs(w0[1][0] - 1305 / 2051) < 0.02
    assert abs(w0[0][0] - 372 / 2051) < 0.02


def test_limite_de_24_vozes_rouba_a_mais_antiga():
    notas = [Note(i * 10, 10_000, 60 + i, 100) for i in range(30)]
    eventos = [[n.start, n.end, 1, n] for n in notas]
    roubadas = psx.limitar_vozes(eventos, 24)
    assert roubadas == 6
    assert notas[0].dur == 240 and notas[6].dur == 10_000


def test_render_ps1_relata_vozes():
    p = Project("t", bpm=120, bit_depth=24)
    p.console = "ps1"
    tr = p.new_track("pad", factory_by_name("Warm Pad"))
    pat = p.new_pattern("x", bars=1)
    for k in range(30):
        pat.add(48 + k, 0, 1800, 90)
    p.place(pat, tr, 0)
    audio = engine.render_project(p)
    assert np.isfinite(audio).all()
    assert p.ultimo_render["vozes_roubadas"] == 6
    assert p.ultimo_render["pico_de_vozes"] == 30


# ==========================================================================
# Correcoes
# ==========================================================================
def test_sampler_sem_banco_nao_some_ao_salvar(tmp_path):
    p = Project("t")
    p.new_track("piano", Instrument())
    d = p.to_dict()
    d["tracks"][0]["instrument"] = {"kind": "sampler", "sf2": "/nao/existe.sf2",
                                    "bank": 0, "program": 4,
                                    "preset_name": "Tine Electric Piano",
                                    "name": "EP"}
    p2 = Project.from_dict(d)
    assert p2.avisos and "indisponivel" in p2.avisos[0]
    caminho = p2.save(tmp_path / "x.chip")
    p3 = Project.load(caminho)
    assert p3.to_dict()["tracks"][0]["instrument"]["sf2"] == "/nao/existe.sf2"


def test_abrir_arquivo_invalido_da_erro_claro(tmp_path):
    ruim = tmp_path / "ruim.chip"
    ruim.write_bytes(b"isto nao e zip")
    with pytest.raises(ValueError, match="nao e um projeto"):
        Project.load(ruim)
    futuro = tmp_path / "futuro.chip"
    with zipfile.ZipFile(futuro, "w") as z:
        z.writestr("project.json", '{"format_version": 99}')
    with pytest.raises(ValueError, match="mais nova"):
        Project.load(futuro)


def test_id_de_pattern_nao_colide_depois_de_apagar():
    p = Project("t")
    a, b, c = (p.new_pattern(n) for n in "abc")
    del p.patterns[b.id]
    d = p.new_pattern("d")
    assert d.id not in (a.id, c.id)
    assert p.patterns[c.id] is c


def test_bateria_chip_respeita_pan():
    p = Project("t", bit_depth=24)
    tr = p.new_track("bat", factory_by_name("Kick"))
    tr.pan = -1.0
    pat = p.new_pattern("x", bars=1)
    pat.add(36, 0, 200, 120)
    p.place(pat, tr, 0)
    a = engine.render_project(p, apply_master=False)
    assert np.abs(a[:, 0]).max() > 10 * np.abs(a[:, 1]).max()


def test_bitcrush_depois_da_normalizacao():
    p = Project("t", bit_depth=8)
    tr = p.new_track("lead", factory_by_name("Square Fat"))
    tr.volume = 3.0
    pat = p.new_pattern("x", bars=1)
    for k in range(4):
        pat.add(60 + k * 4, 0, 1800, 127)
    p.place(pat, tr, 0)
    a = engine.render_project(p)
    niveis = np.unique(np.round(a[:, 0] * 128 - 0.5, 4))
    assert np.allclose(niveis, np.round(niveis))     # tudo na grade de 8 bits


def test_tar_com_caminho_para_fora_e_recusado(tmp_path):
    from chipforge.midiio import library as L
    arq = tmp_path / "mal.tar.gz"
    with tarfile.open(arq, "w:gz") as t:
        dado = b"x"
        info = tarfile.TarInfo("../../fora.txt")
        info.size = len(dado)
        t.addfile(info, io.BytesIO(dado))
    alvo = tmp_path / "dest"
    alvo.mkdir()
    with pytest.raises(RuntimeError):
        L._extract(arq, alvo)
    assert not (tmp_path / "fora.txt").exists()


# ==========================================================================
# Estudio (o que o MCP expoe)
# ==========================================================================
def _estudio() -> Estudio:
    e = Estudio()
    e.novo("Teste", bpm=140, tom="A", escala="natural_minor", console="nes")
    e.criar_track("Lead", "chip:NES Lead 25%")
    e.criar_track("Baixo", "chip:NES Triangle Bass")
    e.criar_track("Bat", "chip:Kick")
    return e


def test_estudio_escreve_le_e_apaga():
    e = _estudio()
    msg = e.escrever("lead", 3, "o5 l8 a > c e d c4 < a4 | g2 e2 |")
    assert "compassos 3-4" in msg and "AVISO" not in msg
    assert e.ver("Lead", 3, 4).startswith("o5 a8")
    assert "AVISO" in e.escrever("Lead", 5, "o5 c4 d4 e4 |")
    e.apagar("Lead", 5, 5)
    assert "vazia" in e.ver("Lead", 5, 5)
    with pytest.raises(ErroEstudio):
        e.escrever("Guitarra", 1, "c4")


def test_estudio_acompanhamento_sem_segunda_embolada():
    e = _estudio()
    e.criar_track("Pad", "chip:Warm Pad")
    e.acompanhar("Pad", 1, "Am9 | Fmaj7 | Dm7 | E7", "pad", vozes=4)
    notas = sorted((n for c in e.p.arrangement if c.track_id == e._track("Pad").id
                    for n in e.p.patterns[c.pattern_id].notes),
                   key=lambda n: (n.start, n.pitch))
    por_acorde: dict[int, list[int]] = {}
    for n in notas:
        por_acorde.setdefault(n.start, []).append(n.pitch)
    for ac in por_acorde.values():
        assert min(b - a for a, b in zip(ac, ac[1:])) >= 3, ac


def test_estudio_teto_mantem_acompanhamento_abaixo():
    e = _estudio()
    e.criar_track("Pad", "chip:Warm Pad")
    e.acompanhar("Pad", 1, "Em9 | Cmaj7 | Am7 | B7", "bloco", teto="B4")
    tr = e._track("Pad")
    alto = max(n.pitch for c in e.p.arrangement if c.track_id == tr.id
               for n in e.p.patterns[c.pattern_id].notes)
    assert alto <= 71


def test_estudio_copia_bateria_e_renderiza(tmp_path):
    e = _estudio()
    e.bateria("Bat", 1, "k: x...x...x...x...\ns: ....x.......x...")
    e.acompanhar("Baixo", 1, "Am | F | C | G", "baixo8", oitava=2)
    e.copiar(1, 1, 2, tracks=["Bat"], vezes=3)
    assert e.ver("Bat", 4, 4).startswith("k: x...x...x...x...")
    msg = e.renderizar(str(tmp_path / "x.wav"))
    assert "LUFS" in msg and (tmp_path / "x.wav").exists()
    e.salvar(str(tmp_path / "x.chip"))
    assert Estudio().abrir(str(tmp_path / "x.chip")).startswith("Aberto")


def test_estudio_analisa_no_tom_do_projeto():
    e = _estudio()
    e.escrever("Lead", 1, "o5 l4 a c+ e a | g+ b e2 |")
    assert "Tonalidade percebida: A menor" in e.analisar()


def test_medidor_aceita_tritono_do_dominante():
    from chipforge.core import harmony as H
    from chipforge.core.harmony import NoteEvent
    # B7 com a fundamental no baixo: La-Re# e tensao funcional
    ev = [NoteEvent(0, 4, 47, 100, "b"), NoteEvent(0, 4, 69, 90, "p"),
          NoteEvent(0, 4, 75, 90, "p")]
    rep = H.analyze(ev)
    assert not [i for i in rep.issues if i.severity == 2]


def test_servidor_mcp_registra_ferramentas():
    pytest.importorskip("mcp")
    import asyncio
    from chipforge import mcp_server as M
    nomes = {t.name for t in asyncio.run(M.mcp.list_tools())}
    assert {"escrever", "acompanhar", "bateria", "renderizar", "analisar",
            "tocar", "sala"} <= nomes


# ==========================================================================
# Timbre e groove
# ==========================================================================
def test_mml_acento_e_fantasma():
    r = mml.parse("o4 v90 !c d ?e ![c e]4", PPQ)
    assert [n.velocity for n in r.notas] == [110, 90, 45, 110, 110]


def test_calibracao_iguala_volumes():
    e = Estudio()
    e.novo("t", console="nes")
    for nome, inst in (("a", "chip:NES Lead 25%"), ("b", "chip:Warm Pad"),
                       ("c", "chip:NES Triangle Bass"),
                       ("d", "corda:Guitarra limpa")):
        e.criar_track(nome, inst)
    niveis = []
    for t in e.p.tracks:
        notas = (36, 43) if "Bass" in t.instrument.name else (60, 67)
        sig = np.concatenate([t.instrument.render_note(p, 0.5, 22050, 100)[:11025]
                              for p in notas])
        niveis.append(20 * np.log10(np.sqrt(np.mean(sig ** 2))))
    assert max(niveis) - min(niveis) < 3.0


def test_humano_tira_da_grade_de_forma_reprodutivel():
    e = _estudio()
    e.escrever("Lead", 1, "o5 l16 (c d e f)4 |", humano=10)
    tr = e._track("Lead")
    a = [n.start for c in e.p.arrangement if c.track_id == tr.id
         for n in e.p.patterns[c.pattern_id].notes]
    assert any(s % 120 for s in a)                 # saiu da grade
    e2 = _estudio()
    e2.escrever("Lead", 1, "o5 l16 (c d e f)4 |", humano=10)
    b = [n.start for c in e2.p.arrangement if c.track_id == e2._track("Lead").id
         for n in e2.p.patterns[c.pattern_id].notes]
    assert a == b                                   # mesma semente


def test_amp_por_preset_e_guitarra_corda():
    e = _estudio()
    e.criar_track("Gtr", "corda:Guitarra (chug pesado)")
    e.ajustar_track("Gtr", efeitos=[{"tipo": "amp", "preset": "metal", "pan": -0.5}])
    ins = e._track("Gtr").inserts[0]
    assert ins["ganho"] == 16.0 and ins["pan"] == -0.5 and "preset" not in ins
