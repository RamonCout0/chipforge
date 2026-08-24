"""
chipforge.ui.sprite_stage
=========================
O Palco: um personagem animado que reage a intensidade da musica.

Quanto mais acao na faixa, maior o nivel e mais rapida a animacao — parado no
nivel 1, surtando no nivel 5. A intensidade vem de `core.intensity`, que olha
densidade de notas, polifonia, camadas e bateria, e nao volume: um trecho
baixinho e frenetico marca alto, um acorde longo fortissimo marca baixo.

Como colocar os SEUS desenhos
-----------------------------
    chipforge/assets/sprites/<personagem>/nivel1.png ... nivel5.png

Cada PNG e uma folha horizontal de quadros quadrados. O app deduz o lado do
quadro pela ALTURA da imagem: uma folha de 6 quadros de 64px fica 384x64.
Cada nivel pode ter uma quantidade diferente de quadros. Se preferir quadros
soltos, use uma pasta `nivel1/` com os arquivos em ordem alfabetica.

Crie quantos personagens quiser: cada subpasta vira uma opcao no seletor.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QRect, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPixmap
from PyQt6.QtWidgets import (QComboBox, QFileDialog, QFrame, QHBoxLayout,
                             QLabel, QPushButton, QVBoxLayout, QWidget)

from . import theme

PASTA_PADRAO = Path(__file__).resolve().parents[1] / "assets" / "sprites"
EXTENSOES = (".png", ".gif", ".webp", ".jpg", ".jpeg", ".bmp")


def _fatiar(folha: QPixmap) -> list[QPixmap]:
    """Corta a folha em quadros quadrados do tamanho da altura."""
    if folha.isNull():
        return []
    lado = folha.height()
    n = max(1, folha.width() // lado)
    return [folha.copy(QRect(i * lado, 0, lado, lado)) for i in range(n)]


def carregar_personagem(pasta: Path, n_niveis: int = 5) -> list[list[QPixmap]]:
    """Devolve uma lista de niveis, cada um com seus quadros."""
    niveis: list[list[QPixmap]] = []
    for i in range(1, n_niveis + 1):
        quadros: list[QPixmap] = []
        subpasta = pasta / f"nivel{i}"
        if subpasta.is_dir():
            for arq in sorted(subpasta.iterdir()):
                if arq.suffix.lower() in EXTENSOES:
                    px = QPixmap(str(arq))
                    if not px.isNull():
                        quadros.append(px)
        else:
            for ext in EXTENSOES:
                arq = pasta / f"nivel{i}{ext}"
                if arq.exists():
                    quadros = _fatiar(QPixmap(str(arq)))
                    break
        if not quadros and niveis:
            quadros = niveis[-1]          # nivel faltando reusa o anterior
        niveis.append(quadros)
    return niveis


def listar_personagens(raiz: Path = PASTA_PADRAO) -> list[Path]:
    if not raiz.exists():
        return []
    return sorted(p for p in raiz.iterdir()
                  if p.is_dir() and any(p.glob("nivel1*")))


class SpriteStage(QFrame):
    """Palco com o personagem, a barra de intensidade e o seletor."""

    personagemMudou = pyqtSignal(str)

    def __init__(self, n_niveis: int = 5):
        super().__init__()
        self.setObjectName("panel")
        self.n_niveis = n_niveis
        self.niveis: list[list[QPixmap]] = []
        self.nome = ""
        self.nivel = 1
        self.valor = 0.0
        self.quadro = 0
        self.tocando = False
        self.setMinimumHeight(210)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(8)

        topo = QHBoxLayout()
        titulo = QLabel("PALCO")
        titulo.setObjectName("h1")
        topo.addWidget(titulo)
        topo.addStretch(1)
        self.cmb = QComboBox()
        self.cmb.setMinimumWidth(120)
        self.cmb.currentIndexChanged.connect(self._trocar)
        topo.addWidget(self.cmb)
        self.btn = QPushButton("Pasta...")
        self.btn.setToolTip("Escolher outra pasta de sprites")
        self.btn.clicked.connect(self.escolher_pasta)
        topo.addWidget(self.btn)
        lay.addLayout(topo)

        self.tela = _Tela(self)
        lay.addWidget(self.tela, 1)

        self.dica = QLabel("")
        self.dica.setObjectName("dim")
        self.dica.setWordWrap(True)
        lay.addWidget(self.dica)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._avancar)
        self.timer.start(120)

        self.raiz = PASTA_PADRAO
        self.recarregar()

    # ------------------------------------------------------------------
    def recarregar(self):
        personagens = listar_personagens(self.raiz)
        self.cmb.blockSignals(True)
        self.cmb.clear()
        for p in personagens:
            self.cmb.addItem(p.name, str(p))
        self.cmb.blockSignals(False)
        if personagens:
            self._trocar(0)
        else:
            self.niveis = []
            self.dica.setText(
                f"Nenhum sprite em {self.raiz}. Crie uma pasta com "
                "nivel1.png ... nivel5.png (folhas horizontais de quadros "
                "quadrados) e clique em Pasta.")
            self.tela.update()

    def escolher_pasta(self):
        d = QFileDialog.getExistingDirectory(self, "Pasta de sprites",
                                             str(self.raiz))
        if d:
            escolhida = Path(d)
            # aceita tanto a raiz quanto a pasta de um personagem
            self.raiz = escolhida.parent if any(escolhida.glob("nivel1*")) \
                else escolhida
            self.recarregar()

    def _trocar(self, i: int):
        if i < 0 or i >= self.cmb.count():
            return
        pasta = Path(self.cmb.itemData(i))
        self.nome = pasta.name
        self.niveis = carregar_personagem(pasta, self.n_niveis)
        total = sum(len(n) for n in self.niveis)
        self.dica.setText(
            f"{self.nome}: {total} quadros em {len(self.niveis)} niveis. "
            "Nivel 1 = parado, nivel 5 = surto. Substitua os PNGs pelos seus.")
        self.quadro = 0
        self.tela.update()
        self.personagemMudou.emit(self.nome)

    # ------------------------------------------------------------------
    def set_estado(self, valor: float, nivel: int, tocando: bool = True):
        self.valor = max(0.0, min(1.0, float(valor)))
        self.nivel = max(1, min(self.n_niveis, int(nivel)))
        self.tocando = tocando
        # animacao acelera com a acao: 4 fps parado, ~19 fps no surto
        fps = 3 + self.nivel * 3.2
        self.timer.setInterval(int(1000 / fps))
        self.tela.update()

    def repousar(self):
        self.set_estado(0.0, 1, tocando=False)

    def _avancar(self):
        quadros = self.quadros_atuais()
        if quadros:
            self.quadro = (self.quadro + 1) % len(quadros)
            self.tela.update()

    def quadros_atuais(self) -> list[QPixmap]:
        if not self.niveis:
            return []
        return self.niveis[min(self.nivel, len(self.niveis)) - 1]


class _Tela(QWidget):
    """Area de desenho do palco."""

    def __init__(self, palco: SpriteStage):
        super().__init__()
        self.palco = palco
        self.setMinimumHeight(130)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect()

        # chao do palco: escurece embaixo e acende com a intensidade
        calor = self.palco.valor
        fundo = QColor(theme.BG)
        p.fillRect(r, fundo)
        halo = QColor(theme.ACCENT if calor < 0.6 else theme.BAD)
        halo.setAlpha(int(18 + 55 * calor))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(halo)
        raio = min(r.width(), r.height()) * (0.35 + 0.25 * calor)
        p.drawEllipse(QRectF(r.center().x() - raio, r.bottom() - raio * 0.75,
                             raio * 2, raio * 1.2))

        quadros = self.palco.quadros_atuais()
        if not quadros:
            p.setPen(QColor(theme.TEXT_DIM))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter,
                       "sem sprites\n(coloque nivel1.png ... nivel5.png)")
            return

        px = quadros[self.palco.quadro % len(quadros)]
        area_h = r.height() - 26
        escala = max(1, int(min(area_h / max(1, px.height()),
                                r.width() * 0.55 / max(1, px.width()))))
        alvo = px.scaled(px.width() * escala, px.height() * escala,
                         Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.FastTransformation)
        x = r.center().x() - alvo.width() / 2
        y = r.bottom() - 24 - alvo.height()
        p.drawPixmap(int(x), int(y), alvo)

        # medidor: pastilhas por nivel + barra continua
        larg = r.width() - 24
        base_y = r.bottom() - 14
        passo = larg / self.palco.n_niveis
        for i in range(self.palco.n_niveis):
            aceso = i < self.palco.nivel
            cor = QColor(theme.OK if i < 2 else theme.WARN if i < 4 else theme.BAD)
            cor.setAlpha(255 if aceso else 45)
            p.setBrush(cor)
            p.drawRoundedRect(QRectF(12 + i * passo, base_y,
                                     passo - 5, 6), 3, 3)
        f = QFont(); f.setPointSize(8); p.setFont(f)
        p.setPen(QColor(theme.TEXT_DIM))
        p.drawText(QRect(12, r.bottom() - 30, r.width() - 24, 14),
                   Qt.AlignmentFlag.AlignRight,
                   f"nivel {self.palco.nivel}/{self.palco.n_niveis}"
                   f"   {self.palco.valor*100:.0f}%")
