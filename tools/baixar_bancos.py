"""
Baixa os bancos de instrumentos livres para o cache da biblioteca (~/.chipforge/library):

    MuseScore General .sf2 ... MIT (créditos: Frank Wen, Michael Cowgill, S. Christian Collins)
    FluidR3 GM .sf2 .......... MIT (Frank Wen)
    Salamander Drumkit .sfz .. CC BY-SA 3.0 (Alexander Holm) — crédito obrigatório
    VSCO 2 CE (partes) .sfz .. CC0 (Versilian Studios) — só os instrumentos usados nas trilhas

    python tools/baixar_bancos.py              # todos
    python tools/baixar_bancos.py vsco2        # só um

Os arquivos ficam em ~/.chipforge/library/<id>/. Já baixado não baixa de novo.
"""
from __future__ import annotations

import json
import re
import sys
import tarfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chipforge.midiio import library as L  # noqa: E402

UA = {"User-Agent": "Chipforge/1.0"}

ARQUIVOS = {
    "musescore_general": "https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf2",
    "fluidr3_gm": "https://member.keymusician.com/Member/FluidR3_GM/FluidR3_GM.zip",
    "salamander_drumkit": "https://archive.org/download/SalamanderDrumkit/salamanderDrumkit.tar.bz2",
}
LICENCAS = {
    "musescore_general": "https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General_License.md",
}

# VSCO 2 CE: só os SFZ que as trilhas usam (o pacote inteiro tem 2,3 GB).
VSCO_RAW = "https://raw.githubusercontent.com/sgossner/VSCO-2-CE/SFZ/"
VSCO_SFZ = [
    "TubularBells.sfz", "Glockenspiel.sfz", "Harp.sfz", "Timpani.sfz", "TimpaniRolls.sfz",
    "ViolinEnsSusVib.sfz", "ViolaEnsSusVib.sfz", "CelloEnsSusVib.sfz", "ViolinEnsSpic.sfz",
    "CelloEnsSpic.sfz", "ViolinEnsTrem.sfz", "CelloEnsTrem.sfz", "ContrabassSusVB.sfz",
    "FHornSus.sfz", "FHornStac.sfz", "TromboneSus.sfz", "TromboneStac.sfz", "TubaSus.sfz",
    "TrumpetSus.sfz", "OrganLoud.sfz", "OrganQuiet.sfz", "FluteSusVib.sfz", "Marimba.sfz",
]


def baixar(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".parcial")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if total:
                print(f"\r  {dest.name}: {got / 1e6:6.1f} / {total / 1e6:6.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)
    return dest


def extrair(arquivo: Path, pasta: Path) -> None:
    nome = arquivo.name.lower()
    if nome.endswith(".zip"):
        with zipfile.ZipFile(arquivo) as z:
            for m in z.namelist():
                if m.startswith(("/", "..")) or ".." in Path(m).parts:
                    continue
                z.extract(m, pasta)
    elif nome.endswith((".tar.bz2", ".tbz2", ".tar.gz", ".tgz", ".tar")):
        with tarfile.open(arquivo) as t:
            t.extractall(pasta, filter="data")
    else:
        return
    arquivo.unlink()


def arquivo_simples(id_: str) -> None:
    pasta = L.CACHE / id_
    marca = pasta / ".instalado"
    if marca.exists():
        print(f"{id_}: já instalado")
        return
    url = ARQUIVOS[id_]
    print(f"{id_}: baixando {url}")
    arq = baixar(url, pasta / Path(urllib.parse.urlparse(url).path).name)
    extrair(arq, pasta)
    if id_ in LICENCAS:
        baixar(LICENCAS[id_], pasta / "LICENCA.md")
    marca.write_text("ok", encoding="utf-8")
    print(f"{id_}: pronto em {pasta}")


def vsco2() -> None:
    pasta = L.CACHE / "vsco2_ce"
    for nome in VSCO_SFZ:
        sfz = baixar(VSCO_RAW + nome, pasta / nome)
        texto = sfz.read_text(encoding="utf-8", errors="replace")
        # As amostras ficam em default_path (do <control>) + o caminho de cada região.
        m = re.search(r"default_path=([^\r\n]*)", texto)
        prefixo = (m.group(1).strip() if m else "").replace("\\", "/")
        amostras = sorted(set(m.strip() for m in re.findall(r"sample=([^\r\n]+?\.(?:wav|flac|ogg))", texto, re.I)))
        print(f"vsco2_ce: {nome} ({len(amostras)} amostras)")
        for a in amostras:
            rel = (prefixo + a.replace("\\", "/")).lstrip("./")
            baixar(VSCO_RAW + urllib.parse.quote(rel), pasta / rel)
    baixar("https://raw.githubusercontent.com/sgossner/VSCO-2-CE/master/LICENSE", pasta / "LICENCA.txt")
    (pasta / ".instalado").write_text("ok", encoding="utf-8")
    print(f"vsco2_ce: pronto em {pasta}")


def main() -> None:
    escolha = sys.argv[1:] or ["musescore_general", "fluidr3_gm", "salamander_drumkit", "vsco2"]
    for id_ in escolha:
        try:
            vsco2() if id_ == "vsco2" else arquivo_simples(id_)
        except Exception as e:  # um banco fora do ar não impede os outros
            print(f"{id_}: FALHOU — {e}")


if __name__ == "__main__":
    main()
