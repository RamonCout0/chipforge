"""
chipforge.midiio.library
========================
Biblioteca de conteudo: baixa bancos de instrumentos (.sf2) e pacotes de MIDI,
guarda em cache local e indexa o que ja esta na maquina.

Politica de licenca: o catalogo so aponta para fontes com licenca declarada
(dominio publico, CC0/CC-BY ou redistribuicao livre). Cada entrada carrega o
campo `license` e a UI mostra isso antes de baixar — em musica de jogo, licenca
errada e problema de verdade na hora de publicar.

O catalogo fica em ~/.chipforge/sources.json e e editavel: se voce achar um
pack legal, e so acrescentar.
"""
from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

HOME = Path.home() / ".chipforge"
CACHE = HOME / "library"
SOURCES_FILE = HOME / "sources.json"
USER_AGENT = "Chipforge/0.1 (+music tool)"


@dataclass
class Source:
    id: str
    name: str
    kind: str                 # "soundfont" | "midi_pack" | "samples"
    url: str
    license: str
    homepage: str = ""
    size_mb: float = 0.0
    tags: list[str] = field(default_factory=list)
    notes: str = ""


# Catalogo inicial. URLs podem mudar com o tempo: a UI valida antes de baixar
# e o usuario pode editar sources.json.
DEFAULT_SOURCES: list[Source] = [
    Source("generaluser_gs", "GeneralUser GS 1.471 (banco GM completo)", "soundfont",
           "https://registry.npmjs.org/generaluser/-/generaluser-1.47.1.tgz",
           "Licenca propria: uso livre em musica, inclusive comercial",
           "https://www.schristiancollins.com/generaluser.php", 28.5,
           ["gm", "geral", "piano", "cordas", "metais", "recomendado"],
           "269 presets e 11 kits de bateria. Comece por aqui. Servido pelo "
           "registro do npm, que costuma passar por firewall corporativo."),
    Source("generaluser_gs_site", "GeneralUser GS (site oficial)", "soundfont",
           "https://www.schristiancollins.com/soundfonts/GeneralUser_GS_1.471.zip",
           "Licenca propria: uso livre em musica, inclusive comercial",
           "https://www.schristiancollins.com/generaluser.php", 30.0,
           ["gm", "geral", "espelho"],
           "Mesmo banco, direto do autor. Use se o npm estiver bloqueado."),
    Source("fluidr3_gm", "FluidR3 GM", "soundfont",
           "https://member.keymusician.com/Member/FluidR3_GM/FluidR3_GM.zip",
           "MIT",
           "https://member.keymusician.com/Member/FluidR3_GM/", 148.0,
           ["gm", "geral", "padrao"],
           "Banco padrao do FluidSynth. Pesado, som mais neutro. (Site do autor "
           "fora do ar em out/2026: prefira o MuseScore General, que e a versao melhorada.)"),
    Source("musescore_general", "MuseScore General", "soundfont",
           "https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf2",
           "MIT (manter os creditos de Frank Wen, Michael Cowgill e S. Christian Collins)",
           "https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/", 216.0,
           ["gm", "geral", "recomendado"],
           "FluidR3 revisado pelo autor do GeneralUser: synth brass, cordas por naipe, "
           "coro, cravo e kits Power/Room. Baixe com tools/baixar_bancos.py."),
    Source("salamander_drumkit", "Salamander Drumkit (SFZ)", "sfz",
           "https://archive.org/download/SalamanderDrumkit/salamanderDrumkit.tar.bz2",
           "CC BY-SA 3.0 (credito obrigatorio; o share-alike vale so para as amostras)",
           "https://archive.org/details/SalamanderDrumkit", 388.0,
           ["bateria", "rock", "acustico"],
           "Bateria acustica gravada, com varias gravacoes por batida (lorand/hirand). "
           "Use chipforge.audio.sfz com ALL.sfz e is_drum=True."),
    Source("vsco2_ce", "VSCO 2 Community Edition (SFZ, partes)", "sfz",
           "https://github.com/sgossner/VSCO-2-CE/tree/SFZ",
           "CC0 (dominio publico)",
           "https://github.com/sgossner/VSCO-2-CE", 530.0,
           ["orquestra", "cordas", "metais", "sinos"],
           "Orquestra de camara gravada. O tools/baixar_bancos.py baixa so os instrumentos "
           "das trilhas (cordas, metais, sinos, timpano, harpa, orgao, flauta, marimba)."),
    Source("archive_500_sf2", "500 SoundFonts Collection", "soundfont",
           "https://archive.org/details/500-soundfonts-full-gm-sets",
           "Ver pagina do item no Internet Archive",
           "https://archive.org/details/500-soundfonts-full-gm-sets", 0.0,
           ["colecao", "variado"],
           "Pagina de indice: escolha o arquivo e cole a URL direta."),
    Source("wikimedia_midi", "Wikimedia Commons — MIDI de dominio publico",
           "midi_pack",
           "https://commons.wikimedia.org/wiki/Commons:Free_media_resources/Music",
           "Dominio publico / CC",
           "https://commons.wikimedia.org/wiki/Commons:Free_media_resources/Music",
           0.0, ["classico", "dominio publico"],
           "Otimo para estudar arranjo: Bach e Mozart em MIDI limpo."),
]


def _ensure_dirs() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)


def load_sources() -> list[Source]:
    _ensure_dirs()
    if SOURCES_FILE.exists():
        try:
            data = json.loads(SOURCES_FILE.read_text(encoding="utf-8"))
            return [Source(**d) for d in data]
        except Exception:
            pass
    save_sources(DEFAULT_SOURCES)
    return list(DEFAULT_SOURCES)


def save_sources(sources: list[Source]) -> None:
    _ensure_dirs()
    SOURCES_FILE.write_text(
        json.dumps([asdict(s) for s in sources], indent=2, ensure_ascii=False),
        encoding="utf-8")


def add_source(src: Source) -> None:
    sources = load_sources()
    sources = [s for s in sources if s.id != src.id] + [src]
    save_sources(sources)


# --------------------------------------------------------------------------
def download(url: str, dest: Path | None = None, progress=None,
             timeout: int = 60) -> Path:
    """Baixa com barra de progresso. progress(recebido, total) em bytes."""
    _ensure_dirs()
    dest = dest or CACHE / Path(url.split("?")[0]).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r, \
                open(dest, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            got = 0
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if progress:
                    progress(got, total)
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Nao consegui baixar {url}\n{e}\n"
            "Se a rede estiver bloqueada, baixe pelo navegador e use "
            "'Importar arquivo local' na aba Biblioteca.") from e
    return dest


def install_source(src: Source, progress=None) -> list[Path]:
    """Baixa e, se for zip, extrai. Devolve os arquivos utilizaveis."""
    _ensure_dirs()
    target_dir = CACHE / src.id
    target_dir.mkdir(parents=True, exist_ok=True)
    archive = download(src.url, target_dir / Path(src.url.split("?")[0]).name,
                       progress)
    _extract(archive, target_dir)
    return scan(target_dir)


def _extract(archive: Path, target_dir: Path) -> None:
    """Aceita .zip e .tgz/.tar.gz — o npm serve tarball, os sites servem zip."""
    import tarfile
    name = archive.name.lower()
    try:
        if name.endswith(".zip"):
            with zipfile.ZipFile(archive) as z:
                z.extractall(target_dir)
        elif name.endswith((".tgz", ".tar.gz", ".tar")):
            with tarfile.open(archive) as t:
                members = [m for m in t.getmembers()
                           if not m.name.startswith(("/", ".."))]
                t.extractall(target_dir, members=members)
        else:
            return                      # arquivo solto (.sf2/.mid): fica onde esta
    except Exception:
        return
    archive.unlink(missing_ok=True)


def import_local(path: str | Path, kind: str = "soundfont") -> list[Path]:
    """Copia um .sf2/.mid/.zip que o usuario ja tem para dentro da biblioteca."""
    _ensure_dirs()
    src = Path(path)
    dest_dir = CACHE / ("local_" + kind)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in (".zip", ".tgz", ".gz", ".tar"):
        tmp = dest_dir / src.name
        shutil.copy2(src, tmp)
        _extract(tmp, dest_dir)
    else:
        shutil.copy2(src, dest_dir / src.name)
    return scan(dest_dir)


SF2_EXT = {".sf2", ".sf3"}
MIDI_EXT = {".mid", ".midi", ".smf"}


def scan(root: Path | None = None) -> list[Path]:
    root = root or CACHE
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*")
                  if p.suffix.lower() in SF2_EXT | MIDI_EXT)


def installed_soundfonts() -> list[Path]:
    return [p for p in scan() if p.suffix.lower() in SF2_EXT]


def installed_midis() -> list[Path]:
    return [p for p in scan() if p.suffix.lower() in MIDI_EXT]


def get_source(sid: str) -> Source | None:
    return next((s for s in load_sources() if s.id == sid), None)


def ensure(sid: str = "generaluser_gs", progress=None) -> Path | None:
    """Garante que o banco esta na maquina; baixa se precisar. Devolve o .sf2.

    E o atalho usado pelo botao 'Baixar banco de instrumentos' da interface e
    pelos exemplos: chamar duas vezes nao baixa duas vezes.
    """
    target_dir = CACHE / sid
    existing = [p for p in scan(target_dir) if p.suffix.lower() in SF2_EXT]
    if existing:
        return max(existing, key=lambda p: p.stat().st_size)
    src = get_source(sid)
    if not src:
        raise KeyError(f"Fonte desconhecida: {sid}")
    files = install_source(src, progress)
    sf2 = [p for p in files if p.suffix.lower() in SF2_EXT]
    return max(sf2, key=lambda p: p.stat().st_size) if sf2 else None


def catalog_summary() -> str:
    lines = ["Fontes no catalogo:"]
    for s in load_sources():
        size = f"{s.size_mb:.0f} MB" if s.size_mb else "?"
        lines.append(f"  [{s.kind}] {s.name} — {size} — {s.license}")
    sf, md = installed_soundfonts(), installed_midis()
    lines.append(f"Instalado: {len(sf)} soundfont(s), {len(md)} MIDI(s) "
                 f"em {CACHE}")
    return "\n".join(lines)
