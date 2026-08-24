"""
chipforge.ui.piano_roll
=======================
O piano roll: onde a musica e desenhada.

Controles (a ideia e nao precisar de manual):
  clique esquerdo no vazio  -> cria nota (e toca preview)
  arrastar nota             -> move (segura Shift para so mover no tempo)
  arrastar a borda direita  -> muda a duracao
  clique direito            -> apaga
  Ctrl + arrastar           -> duplica
  roda do mouse             -> rolagem  |  Ctrl+roda: zoom horizontal
  Alt+roda                  -> zoom vertical
As linhas da escala ativa ficam iluminadas, e a tonica mais forte ainda —
assim da para compor "no olho" sem saber teoria.
"""
from __future__ import annotations

from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import (QBrush, QColor, QFont, QLinearGradient, QPainter, QPen)
from PyQt6.QtWidgets import QAbstractScrollArea, QScrollArea, QSizePolicy, QWidget

from ..core import theory as T
from ..core.project import Note, Pattern, Project
from . import theme

LOW_NOTE, HIGH_NOTE = 24, 108        # C1 .. C8
KEY_W = 62


def _is_black(pitch: int) -> bool:
    return (pitch % 12) in (1, 3, 6, 8, 10)


class KeyboardStrip(QWidget):
    notePressed = pyqtSignal(int)

    def __init__(self, roll: "PianoRollGrid"):
        super().__init__()
        self.roll = roll
        self.setFixedWidth(KEY_W)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def sizeHint(self):
        return self.roll.sizeHint().boundedTo(
            self.roll.sizeHint()).__class__(KEY_W, self.roll.total_height())

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        kh = self.roll.key_h
        f = QFont(); f.setPointSize(7); p.setFont(f)
        for pitch in range(LOW_NOTE, HIGH_NOTE):
            y = self.roll.pitch_to_y(pitch)
            black = _is_black(pitch)
            rect = QRect(0, y, KEY_W - 1, kh)
            p.fillRect(rect, QColor(theme.KEY_BLACK if black else theme.KEY_WHITE))
            p.setPen(QPen(QColor(theme.GRID), 1))
            p.drawLine(0, y + kh, KEY_W, y + kh)
            if pitch % 12 == 0:
                p.setPen(QColor(theme.TEXT_DIM))
                p.drawText(rect.adjusted(4, 0, -4, 0),
                           Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                           T.midi_to_name(pitch))
        p.setPen(QPen(QColor(theme.GRID_STRONG), 1))
        p.drawLine(KEY_W - 1, 0, KEY_W - 1, self.height())

    def mousePressEvent(self, e):
        self.notePressed.emit(self.roll.y_to_pitch(int(e.position().y())))


class PianoRollGrid(QWidget):
    notesChanged = pyqtSignal()
    previewNote = pyqtSignal(int)

    def __init__(self, project: Project):
        super().__init__()
        self.project = project
        self.pattern: Pattern | None = None
        self.key_h = 13
        self.beat_w = 58.0
        self.snap = 4                # 4 = semicolcheia (1/16)
        self.default_len_beats = 0.25
        self.playhead_tick = -1
        self.selected: set[int] = set()
        self._drag = None
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ---- geometria ----------------------------------------------------
    def total_height(self) -> int:
        return (HIGH_NOTE - LOW_NOTE) * self.key_h

    def total_width(self) -> int:
        ticks = self.pattern.length_ticks if self.pattern else self.project.ticks_per_bar * 4
        return int(ticks / self.project.ppq * self.beat_w) + 2

    def sizeHint(self):
        from PyQt6.QtCore import QSize
        return QSize(self.total_width(), self.total_height())

    def pitch_to_y(self, pitch: int) -> int:
        return (HIGH_NOTE - 1 - pitch) * self.key_h

    def y_to_pitch(self, y: int) -> int:
        return max(LOW_NOTE, min(HIGH_NOTE - 1, HIGH_NOTE - 1 - y // self.key_h))

    def tick_to_x(self, tick: int) -> float:
        return tick / self.project.ppq * self.beat_w

    def x_to_tick(self, x: float) -> int:
        return int(x / self.beat_w * self.project.ppq)

    @property
    def snap_ticks(self) -> int:
        return max(1, int(self.project.ppq / self.snap)) if self.snap else 1

    def quantize(self, tick: int) -> int:
        s = self.snap_ticks
        return (tick // s) * s

    def set_pattern(self, pattern: Pattern | None):
        self.pattern = pattern
        self.selected.clear()
        self.updateGeometry()
        self.resize(self.sizeHint())
        self.update()

    # ---- pintura -------------------------------------------------------
    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(theme.BG))
        if not self.pattern:
            p.setPen(QColor(theme.TEXT_DIM))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Selecione ou crie um pattern")
            return

        allowed = T.scale_pitch_classes(self.project.key_root, self.project.scale)
        w, h = self.width(), self.height()

        # faixas horizontais por nota
        for pitch in range(LOW_NOTE, HIGH_NOTE):
            y = self.pitch_to_y(pitch)
            pc = pitch % 12
            if pc == self.project.key_root:
                col = theme.ROOT_ROW
            elif pc in allowed:
                col = theme.SCALE_ROW
            else:
                col = "#141824" if _is_black(pitch) else theme.BG
            p.fillRect(0, y, w, self.key_h, QColor(col))
            p.setPen(QPen(QColor(theme.GRID), 1))
            p.drawLine(0, y + self.key_h, w, y + self.key_h)

        # grade vertical
        ppq = self.project.ppq
        bar_ticks = self.project.ticks_per_bar
        sub = self.snap_ticks
        t = 0
        end = self.pattern.length_ticks
        while t <= end:
            x = self.tick_to_x(t)
            if t % bar_ticks == 0:
                p.setPen(QPen(QColor(theme.GRID_STRONG), 2))
            elif t % ppq == 0:
                p.setPen(QPen(QColor(theme.GRID_STRONG), 1))
            else:
                p.setPen(QPen(QColor(theme.GRID), 1))
            p.drawLine(int(x), 0, int(x), h)
            t += sub

        # notas
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for i, n in enumerate(self.pattern.notes):
            x = self.tick_to_x(n.start)
            wid = max(3.0, self.tick_to_x(n.dur) - 1)
            y = self.pitch_to_y(n.pitch)
            r = QRect(int(x) + 1, y + 1, int(wid), self.key_h - 2)
            base = QColor(theme.NOTE_SEL if i in self.selected
                          else self.pattern.color)
            g = QLinearGradient(float(r.left()), float(r.top()), float(r.left()), float(r.bottom()))
            top = QColor(base); top.setAlpha(90 + int(n.velocity * 1.2))
            bot = QColor(base).darker(135); bot.setAlpha(90 + int(n.velocity * 1.2))
            g.setColorAt(0, top); g.setColorAt(1, bot)
            p.setBrush(QBrush(g))
            p.setPen(QPen(QColor(base).lighter(140), 1))
            p.drawRoundedRect(r, 3, 3)
            if n.pitch % 12 not in allowed and self.project.scale_lock is False:
                p.setPen(QPen(QColor(theme.BAD), 1, Qt.PenStyle.DotLine))
                p.drawRoundedRect(r, 3, 3)

        # playhead
        if self.playhead_tick >= 0:
            x = int(self.tick_to_x(self.playhead_tick))
            p.setPen(QPen(QColor(theme.PLAYHEAD), 2))
            p.drawLine(x, 0, x, h)

    # ---- interacao ------------------------------------------------------
    def _hit(self, pos: QPoint) -> tuple[int, Note, bool] | None:
        tick = self.x_to_tick(pos.x())
        pitch = self.y_to_pitch(pos.y())
        for i, n in enumerate(self.pattern.notes):
            if n.pitch == pitch and n.start <= tick < n.end:
                near_edge = self.tick_to_x(n.end) - pos.x() < 7
                return i, n, near_edge
        return None

    def mousePressEvent(self, e):
        if not self.pattern:
            return
        pos = e.position().toPoint()
        hit = self._hit(pos)
        mods = e.modifiers()

        if e.button() == Qt.MouseButton.RightButton:
            if hit:
                self.pattern.notes.pop(hit[0])
                self.selected.clear()
                self.notesChanged.emit()
                self.update()
            return

        if hit:
            i, n, edge = hit
            if mods & Qt.KeyboardModifier.ControlModifier:
                copy = Note(n.start, n.dur, n.pitch, n.velocity)
                self.pattern.notes.append(copy)
                i = len(self.pattern.notes) - 1
                n = copy
            self.selected = {i}
            self._drag = {"mode": "resize" if edge else "move", "index": i,
                          "t0": self.x_to_tick(pos.x()), "n0": Note(n.start, n.dur, n.pitch, n.velocity),
                          "lock_pitch": bool(mods & Qt.KeyboardModifier.ShiftModifier)}
        else:
            tick = self.quantize(self.x_to_tick(pos.x()))
            pitch = self.y_to_pitch(pos.y())
            if self.project.scale_lock:
                pitch = T.quantize_to_scale(pitch, self.project.key_root,
                                            self.project.scale)
            dur = int(self.default_len_beats * self.project.ppq)
            n = self.pattern.add(pitch, tick, dur)
            self.selected = {len(self.pattern.notes) - 1}
            self._drag = {"mode": "resize", "index": self.selected.copy().pop(),
                          "t0": tick + dur, "n0": Note(n.start, n.dur, n.pitch, n.velocity),
                          "lock_pitch": True}
            self.previewNote.emit(pitch)
            self.notesChanged.emit()
        self.update()

    def mouseMoveEvent(self, e):
        pos = e.position().toPoint()
        if not self._drag:
            hit = self._hit(pos) if self.pattern else None
            self.setCursor(Qt.CursorShape.SizeHorCursor if (hit and hit[2])
                           else Qt.CursorShape.ArrowCursor)
            return
        d = self._drag
        n = self.pattern.notes[d["index"]]
        n0: Note = d["n0"]
        dt = self.x_to_tick(pos.x()) - d["t0"]
        if d["mode"] == "resize":
            n.dur = max(self.snap_ticks,
                        self.quantize(n0.dur + dt + self.snap_ticks // 2))
        else:
            n.start = max(0, self.quantize(n0.start + dt))
            if not d["lock_pitch"]:
                pitch = self.y_to_pitch(pos.y())
                if self.project.scale_lock:
                    pitch = T.quantize_to_scale(pitch, self.project.key_root,
                                                self.project.scale)
                if pitch != n.pitch:
                    n.pitch = pitch
                    self.previewNote.emit(pitch)
        self.update()

    def mouseReleaseEvent(self, e):
        if self._drag:
            self._drag = None
            self.notesChanged.emit()
            self.update()

    def wheelEvent(self, e):
        mods = e.modifiers()
        if mods & Qt.KeyboardModifier.ControlModifier:
            f = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
            self.beat_w = max(14.0, min(320.0, self.beat_w * f))
        elif mods & Qt.KeyboardModifier.AltModifier:
            self.key_h = max(6, min(30, self.key_h + (1 if e.angleDelta().y() > 0 else -1)))
        else:
            e.ignore()
            return
        self.updateGeometry()
        self.resize(self.sizeHint())
        self.update()
        e.accept()

    def keyPressEvent(self, e):
        if not self.pattern:
            return
        if e.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self.selected:
            for i in sorted(self.selected, reverse=True):
                if i < len(self.pattern.notes):
                    self.pattern.notes.pop(i)
            self.selected.clear()
            self.notesChanged.emit()
            self.update()
        elif e.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down) and self.selected:
            step = 12 if e.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1
            step = step if e.key() == Qt.Key.Key_Up else -step
            for i in self.selected:
                self.pattern.notes[i].pitch = max(0, min(127, self.pattern.notes[i].pitch + step))
            self.notesChanged.emit()
            self.update()


class PianoRoll(QScrollArea):
    """Piano roll completo: teclado fixo + grade rolavel."""
    notesChanged = pyqtSignal()
    previewNote = pyqtSignal(int)

    def __init__(self, project: Project):
        super().__init__()
        self.grid = PianoRollGrid(project)
        self.keys = KeyboardStrip(self.grid)
        container = QWidget()
        from PyQt6.QtWidgets import QHBoxLayout
        lay = QHBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.keys)
        lay.addWidget(self.grid)
        lay.addStretch(0)
        self.setWidget(container)
        self.setWidgetResizable(False)
        self.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored)
        self.grid.notesChanged.connect(self.notesChanged)
        self.grid.previewNote.connect(self.previewNote)
        self.keys.notePressed.connect(self.previewNote)
        self._sync()

    def _sync(self):
        h = self.grid.total_height()
        self.keys.setFixedHeight(h)
        self.grid.setFixedSize(self.grid.total_width(), h)

    def set_pattern(self, pattern):
        self.grid.set_pattern(pattern)
        self._sync()
        # centraliza em C4
        v = self.verticalScrollBar()
        v.setValue(max(0, self.grid.pitch_to_y(72) - self.height() // 3))

    def set_playhead(self, tick: int):
        self.grid.playhead_tick = tick
        self.grid.update()

    def refresh(self):
        self._sync()
        self.grid.update()
        self.keys.update()
