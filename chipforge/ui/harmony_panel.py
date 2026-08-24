"""
chipforge.ui.harmony_panel
==========================
O medidor. Um mostrador circular com o score, a tonalidade detectada, os
acordes lidos, as cinco metricas em barras e a lista de problemas com o
compasso exato — clicar num problema leva o piano roll ate ele.
"""
from __future__ import annotations

from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QListWidget,
                             QListWidgetItem, QVBoxLayout, QWidget)

from ..core.harmony import HarmonyReport, suggest_fix
from . import theme

METRIC_LABELS = {
    "consonancia": "Consonancia",
    "tonalidade": "Tonalidade",
    "conducao": "Conducao de vozes",
    "grave_limpo": "Grave limpo",
    "progressao": "Progressao",
}


class Gauge(QWidget):
    """Mostrador circular 0..100."""

    def __init__(self):
        super().__init__()
        self.value = 100.0
        self.color = theme.OK
        self.caption = "—"
        self.setMinimumHeight(150)

    def set(self, value: float, color: str, caption: str):
        self.value, self.color, self.caption = value, color, caption
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        side = min(self.width(), self.height()) - 16
        r = QRectF((self.width() - side) / 2, 6, side, side)
        p.setPen(QPen(QColor(theme.GRID), 11, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap))
        p.drawArc(r, 210 * 16, -240 * 16)
        p.setPen(QPen(QColor(self.color), 11, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap))
        p.drawArc(r, 210 * 16, int(-240 * 16 * self.value / 100))

        f = QFont(); f.setPointSize(max(14, side // 6)); f.setBold(True)
        p.setFont(f); p.setPen(QColor(theme.TEXT))
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, f"{self.value:.0f}")
        f2 = QFont(); f2.setPointSize(8)
        p.setFont(f2); p.setPen(QColor(theme.TEXT_DIM))
        p.drawText(QRectF(0, r.bottom() - side * 0.22, self.width(), 22),
                   Qt.AlignmentFlag.AlignCenter, self.caption)


class MetricBar(QWidget):
    def __init__(self, label: str):
        super().__init__()
        self.label = label
        self.value = 1.0
        self.setFixedHeight(26)

    def set(self, v: float):
        self.value = max(0.0, min(1.0, v))
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        f = QFont(); f.setPointSize(8); p.setFont(f)
        p.setPen(QColor(theme.TEXT_DIM))
        p.drawText(0, 0, 122, self.height(),
                   Qt.AlignmentFlag.AlignVCenter, self.label)
        p.setPen(QColor(theme.TEXT))
        p.drawText(self.width() - 34, 0, 34, self.height(),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                   f"{self.value*100:.0f}%")
        x0, x1 = 126, self.width() - 40
        y = self.height() / 2 - 3
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(theme.GRID))
        p.drawRoundedRect(QRectF(x0, y, x1 - x0, 6), 3, 3)
        col = theme.OK if self.value > 0.75 else (
            theme.WARN if self.value > 0.5 else theme.BAD)
        p.setBrush(QColor(col))
        p.drawRoundedRect(QRectF(x0, y, (x1 - x0) * self.value, 6), 3, 3)


class HarmonyPanel(QFrame):
    seekRequested = pyqtSignal(float)      # beat

    def __init__(self):
        super().__init__()
        self.setObjectName("panel")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        title = QLabel("MEDIDOR DE HARMONIA")
        title.setObjectName("h1")
        lay.addWidget(title)

        self.gauge = Gauge()
        lay.addWidget(self.gauge)

        self.key_lbl = QLabel("—")
        self.key_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.key_lbl)

        self.chords_lbl = QLabel("")
        self.chords_lbl.setObjectName("dim")
        self.chords_lbl.setWordWrap(False)
        self.chords_lbl.setFixedHeight(20)
        self.chords_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.chords_lbl)

        self.bars: dict[str, MetricBar] = {}
        for k, label in METRIC_LABELS.items():
            b = MetricBar(label)
            self.bars[k] = b
            lay.addWidget(b)

        lay.addWidget(QLabel("Pontos de atencao"))
        self.issues = QListWidget()
        self.issues.setMinimumHeight(90)
        self.issues.itemDoubleClicked.connect(self._goto)
        lay.addWidget(self.issues, 1)

        self.tips = QLabel("")
        self.tips.setWordWrap(True)
        self.tips.setObjectName("dim")
        lay.addWidget(self.tips)

    def _goto(self, item: QListWidgetItem):
        beat = item.data(Qt.ItemDataRole.UserRole)
        if beat is not None:
            self.seekRequested.emit(float(beat))

    def update_report(self, rep: HarmonyReport):
        self.gauge.set(rep.score, rep.color, rep.verdict)
        if rep.key:
            self.key_lbl.setText(
                f"<b>{rep.key.name}</b>  <span style='color:{theme.TEXT_DIM}'>"
                f"({rep.key.confidence*100:.0f}% de certeza)</span>")
        else:
            self.key_lbl.setText("—")
        seq = [c.name for _, c in rep.chords[:6]]
        self.chords_lbl.setText(
            (" → ".join(seq) + (" …" if len(rep.chords) > 6 else ""))
            or "sem acordes ainda")
        for k, b in self.bars.items():
            b.set(rep.metrics.get(k, 1.0))

        self.issues.clear()
        icon = {0: "•", 1: "▲", 2: "■"}
        col = {0: theme.TEXT_DIM, 1: theme.WARN, 2: theme.BAD}
        for i in rep.issues[:60]:
            quem = f"   [{i.culpados}]" if i.culpados else ""
            it = QListWidgetItem(
                f"{icon[i.severity]}  c.{i.beat/4+1:.2f}  {i.message}{quem}")
            it.setForeground(QColor(col[i.severity]))
            it.setData(Qt.ItemDataRole.UserRole, i.beat)
            self.issues.addItem(it)
        if not rep.issues:
            self.issues.addItem(QListWidgetItem("Nada travando. Segue o baile."))

        self.tips.setText("💡 " + "  ".join(suggest_fix(rep)[:2]))
