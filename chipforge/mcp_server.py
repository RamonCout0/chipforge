"""
chipforge.mcp_server
====================
Servidor MCP do Chipforge: deixa um assistente (Claude, por exemplo) compor,
mixar, ouvir no seu alto-falante e medir musicas aqui na sua maquina.

Rodar:     python -m chipforge.mcp_server
Conectar:  veja a secao "Compor com o Claude (MCP)" do README.

Cada ferramenta e um metodo de chipforge.estudio.Estudio — o mesmo codigo que
funciona num script. O servidor guarda UM projeto aberto por vez.
"""
from __future__ import annotations

import functools
import sys

try:                                            # SDK mcp 2.x
    from mcp.server.mcpserver import MCPServer as _Servidor
except ImportError:                             # SDK mcp 1.x
    try:
        from mcp.server.fastmcp import FastMCP as _Servidor
    except ImportError:                         # pragma: no cover
        sys.exit("Falta o pacote 'mcp'. Instale com:  pip install mcp")

from .core.mml import __doc__ as _DOC_MML
from .estudio import ESTILOS_ACOMPANHAMENTO, Estudio

INSTRUCOES = """\
Chipforge Studio — DAW de musica de jogo em Python, com modo PS1 (chip de som
do PlayStation: samples ADPCM, interpolacao gaussiana, 24 vozes, reverb do SPU).

Fluxo de trabalho:
1. novo_projeto (console "ps1" para a era 32 bits; "nes"/"snes" para chiptune)
2. criar_track com instrumentos "ps1:<timbre>" (listar_instrumentos mostra a
   paleta), "chip:<preset>", "gm:<busca>", "voz:<preset>" ou "cinema:<preset>"
3. escrever melodias em MML; acompanhar com cifras; bateria em grade
4. analisar (medidor de harmonia) e renderizar (mede LUFS, pico, vozes do SPU)
5. tocar para o usuario ouvir no computador; ajustar mix; salvar_projeto

Compassos contam a partir de 1. Tracks sao chamadas pelo nome.
Escreva secao por secao (intro, A, B, ponte) e confira os avisos de compasso
que `escrever` devolve — eles pegam erro de duracao na hora.
Para o som de PS1: reverb_send 0.2-0.5 nas melodias, pouco no baixo e bumbo,
pan abrindo as camadas, e o reverb do SPU no barramento (ferramenta `sala`).
"""

estudio = Estudio()
mcp = _Servidor(name="chipforge", instructions=INSTRUCOES)


def _seguro(fn):
    """Erros viram texto: o assistente le, corrige a chamada e tenta de novo."""
    @functools.wraps(fn)
    def envolto(*a, **kw):
        try:
            return fn(*a, **kw)
        except Exception as e:                  # noqa: BLE001
            return f"ERRO: {e}"
    return envolto


def ferramenta(fn):
    return mcp.tool()(_seguro(fn))


# --------------------------------------------------------------------------
# projeto
# --------------------------------------------------------------------------
@ferramenta
def novo_projeto(nome: str, bpm: float = 120, tom: str = "C",
                 escala: str = "major", compasso: str = "4/4",
                 console: str = "ps1") -> str:
    """Cria um projeto novo (descarta o atual se nao foi salvo).

    tom: C, C#, Db, D... ; escala: major, natural_minor, harmonic_minor,
    dorian, phrygian, lydian, mixolydian, minor_pentatonic, blues...;
    console: ps1 | snes | nes | limpo."""
    return estudio.novo(nome, bpm, tom, escala, compasso, console)


@ferramenta
def abrir_projeto(caminho: str) -> str:
    """Abre um arquivo .chip."""
    return estudio.abrir(caminho)


@ferramenta
def salvar_projeto(caminho: str = "") -> str:
    """Salva o projeto em .chip (o usuario abre na interface do Chipforge)."""
    return estudio.salvar(caminho or None)


@ferramenta
def configurar_projeto(bpm: float = 0, tom: str = "", escala: str = "",
                       console: str = "", nome: str = "",
                       tempo_map: list[list[float]] | None = None) -> str:
    """Muda andamento, tom, escala, console ou nome. tempo_map: [[compasso, bpm],
    ...] para accelerando/ritardando (interpolado entre as marcas)."""
    return estudio.configurar(bpm or None, tom or None, escala or None,
                              console or None, nome or None, tempo_map)


@ferramenta
def resumo() -> str:
    """Estado atual: tracks, instrumentos, mix e onde cada uma tem material."""
    return estudio.resumo()


# --------------------------------------------------------------------------
# instrumentos e tracks
# --------------------------------------------------------------------------
@ferramenta
def listar_instrumentos(fonte: str = "ps1", busca: str = "") -> str:
    """fonte: ps1 (paleta da era 32 bits), chip (osciladores NES/GB),
    gm (banco GeneralUser GS inteiro), corda (guitarra/baixo por modelagem
    fisica — o melhor timbre de guitarra do app, use com amp), voz (voz
    cantada), cinema (impactos, risers). `busca` filtra pelo nome."""
    return estudio.listar_instrumentos(fonte, busca)


@ferramenta
def criar_track(nome: str, instrumento: str, volume: float = 0.8,
                pan: float = 0.0, reverb_send: float = 0.25) -> str:
    """Cria uma track. instrumento: 'ps1:baixo_slap', 'chip:NES Lead 25%',
    'gm:Trumpet', 'corda:Guitarra (chug pesado)', 'voz:Coro soprano (aah)',
    'cinema:Impacto grave'. O volume do instrumento e calibrado (-20 dBFS
    RMS numa nota tipica), entao volume 0.8 significa o mesmo em toda track."""
    return estudio.criar_track(nome, instrumento, volume, pan, reverb_send)


@ferramenta
def ajustar_track(nome: str, volume: float | None = None, pan: float | None = None,
                  reverb_send: float | None = None, instrumento: str = "",
                  mudo: bool | None = None, solo: bool | None = None,
                  naipe: int | None = None, efeitos: list[dict] | None = None,
                  automacao_volume: list[list[float]] | None = None) -> str:
    """Mixagem de uma track. naipe: N vozes em unissono desafinado (vira
    secao). efeitos: lista de inserts na ordem, ex.
    [{"tipo":"eq","modo":"highshelf","freq":6000,"ganho_db":3}]
    (modos: peak, lowshelf, highshelf, lowpass, highpass),
    {"tipo":"chorus"}, {"tipo":"amp","preset":"metal"} (presets crunch,
    rock, metal, limpo; aceita "pan" e ajustes como "ganho"), {"tipo":"comp"},
    {"tipo":"eco","tempo_s":0.3}, {"tipo":"largura","fator":1.4}.
    Lista vazia remove tudo. automacao_volume: [[compasso, 0..1], ...]."""
    return estudio.ajustar_track(nome, volume, pan, reverb_send,
                                 instrumento or None, mudo, solo, naipe,
                                 efeitos, automacao_volume)


# --------------------------------------------------------------------------
# escrita
# --------------------------------------------------------------------------
@ferramenta
def escrever(track: str, compasso: int, mml: str, substituir: bool = True,
             velocity: int = 96, humano: float = 0.0,
             tendencia_ms: float = 0.0) -> str:
    """Escreve notas em MML a partir do compasso (1 = primeiro). Exemplo:
    'o5 l8 e g a > c < b4 a | g4. e8 d2 |'. `!c` acento, `?c` fantasma.
    humano: espalhamento em ms (6-12 = tocado por gente; 0 = grade).
    tendencia_ms: negativo puxa a track para frente, positivo arrasta.
    Devolve extensao, notas fora da escala e avisos de compasso que nao fecha."""
    return estudio.escrever(track, compasso, mml, None, substituir,
                            velocity=velocity, humano=humano,
                            tendencia_ms=tendencia_ms)


def acompanhar(track: str, compasso: int, cifras: str, estilo: str = "bloco",
               oitava: int = 4, velocity: int = 90, vozes: int = 0,
               teto: str = "", humano: float = 0.0) -> str:
    return estudio.acompanhar(track, compasso, cifras, estilo, oitava, velocity,
                              vozes=vozes, teto=teto or None, humano=humano)


acompanhar.__doc__ = ("Gera acompanhamento a partir de cifras: 'Am7 | F G | "
                      "Dm/F E7 | %' (um compasso por '|'; '%' repete; '-' "
                      "segura). oitava 2 para baixo, 4 para acordes. vozes: quantas "
                      "notas no voicing (0 = automatico; menos vozes = menos "
                      "polifonia gasta no SPU). teto: nota mais aguda permitida (ex. "
                      "'B4') — deixe o acompanhamento abaixo da melodia. Estilos: "
                      + "; ".join(f"{k} ({v})"
                                  for k, v in ESTILOS_ACOMPANHAMENTO.items()))
acompanhar = ferramenta(acompanhar)


@ferramenta
def bateria(track: str, compasso: int, grade: str, repetir: int = 1,
            humano: float = 0.0, tendencia_ms: float = 0.0) -> str:
    """Bateria em grade, uma linha por peca, 16 passos por compasso:
    'k: x...x...x..x.x..\\ns: ....X.......X..o\\nh: x.x.x.x.x.x.x.x.'.
    X acento, x normal, o fantasma. Pecas: k s p(clap) rs h ph oh t1..t5
    c c2 r rb sp ch tb cb. `repetir` repete a grade N vezes.
    humano ~4-8 ms tira do grid; tendencia_ms -3 deixa a bateria 'puxando'."""
    return estudio.bateria(track, compasso, grade, repetir, humano=humano,
                           tendencia_ms=tendencia_ms)


@ferramenta
def ver(track: str, de_compasso: int = 1, ate_compasso: int = 0) -> str:
    """Mostra em MML (ou grade, se bateria) o que a track tem no trecho."""
    return estudio.ver(track, de_compasso, ate_compasso or None)


@ferramenta
def apagar(track: str, de_compasso: int, ate_compasso: int) -> str:
    """Apaga as notas da track no trecho."""
    return estudio.apagar(track, de_compasso, ate_compasso)


@ferramenta
def copiar(de_compasso: int, ate_compasso: int, destino: int,
           tracks: list[str] | None = None, vezes: int = 1) -> str:
    """Copia um trecho para outro compasso (todas as tracks, ou so as listadas).
    Use para refrao repetido e para fechar o loop da fase."""
    return estudio.copiar(de_compasso, ate_compasso, destino, tracks, vezes)


# --------------------------------------------------------------------------
# mix, analise, audio
# --------------------------------------------------------------------------
@ferramenta
def sala(preset: str = "studio_large", wet: float = 0.5) -> str:
    """Reverb do barramento. PS1 (hardware do SPU): room, studio_small,
    studio_medium, studio_large, hall, space_echo, half_echo. Convolucao:
    estudio, camara, salao, catedral, caverna, placa, epico. Ou 'nenhuma'."""
    return estudio.sala(preset, wet)


@ferramenta
def master(perfil: str = "jogo", ganho: float | None = None) -> str:
    """Cadeia de master: jogo, chiptune, rock, cinematico, trailer, cru, ou
    vazio para so limitador."""
    return estudio.master(perfil or None, ganho)


@ferramenta
def analisar(de_compasso: int = 1, ate_compasso: int = 0,
             secoes: list[list] | None = None, detectar_tom: bool = False) -> str:
    """Medidor de harmonia (0-100) com problemas por compasso e tracks que
    brigam. secoes: [["A", 1, 8], ["B", 9, 16]] da um laudo por trecho.
    Julga no tom do projeto; detectar_tom=True para musica que modula."""
    return estudio.analisar(de_compasso, ate_compasso or None, secoes,
                            detectar_tom)


@ferramenta
def renderizar(caminho: str, de_compasso: int = 0, ate_compasso: int = 0) -> str:
    """Gera .wav (ou .mp3 com ffmpeg) e mede: LUFS, pico real, LRA, correlacao,
    balanco espectral e, no PS1, vozes pedidas/roubadas."""
    return estudio.renderizar(caminho, de_compasso or None, ate_compasso or None)


@ferramenta
def tocar(de_compasso: int = 1) -> str:
    """Toca a musica no alto-falante do computador do usuario."""
    return estudio.tocar(de_compasso)


@ferramenta
def parar() -> str:
    """Para a reproducao."""
    return estudio.parar()


@ferramenta
def comparar_com_referencia(caminho_audio: str) -> str:
    """Analisa uma faixa de referencia (mp3/wav; precisa de ffmpeg) — tom,
    BPM, forma, loudness — e compara o balanco espectral com o render atual."""
    return estudio.comparar(caminho_audio)


@ferramenta
def adotar_referencia(caminho_audio: str) -> str:
    """Faz o master perseguir o balanco espectral da referencia (sem copiar
    nenhuma nota)."""
    return estudio.adotar_referencia(caminho_audio)


@ferramenta
def exportar_midi(caminho: str) -> str:
    """Exporta o arranjo como .mid."""
    return estudio.exportar_midi(caminho)


@ferramenta
def ajuda_notacao() -> str:
    """Referencia completa de MML, cifras e grade de bateria."""
    return _DOC_MML


def main():
    mcp.run()


if __name__ == "__main__":
    main()
