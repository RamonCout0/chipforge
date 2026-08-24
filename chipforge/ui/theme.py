"""Paleta e stylesheet. Escuro, com sotaque de CRT verde/azul."""
from __future__ import annotations

BG          = "#12141c"
BG_PANEL    = "#191c26"
BG_RAISED   = "#222634"
GRID        = "#262b3a"
GRID_STRONG = "#39405a"
TEXT        = "#e6e9f2"
TEXT_DIM    = "#8b93ab"
ACCENT      = "#4aa3ff"
ACCENT_2    = "#c77dff"
OK          = "#3ddc84"
WARN        = "#ffcc33"
BAD         = "#ff4d5e"
KEY_WHITE   = "#e9ecf5"
KEY_BLACK   = "#1b1e28"
NOTE        = "#4aa3ff"
NOTE_SEL    = "#ffd166"
SCALE_ROW   = "#1a2130"
ROOT_ROW    = "#1f2b3d"
PLAYHEAD    = "#ff5d73"

TRACK_COLORS = ["#4aa3ff", "#3ddc84", "#ffcc33", "#c77dff",
                "#ff8f4a", "#4adede", "#ff6b9d", "#9dff6b"]

QSS = f"""
QWidget {{
    background: {BG};
    color: {TEXT};
    font-family: "Inter", "Segoe UI", "DejaVu Sans", sans-serif;
    font-size: 12px;
}}
QMainWindow, QDialog {{ background: {BG}; }}
QFrame#panel, QGroupBox {{
    background: {BG_PANEL};
    border: 1px solid {GRID};
    border-radius: 8px;
}}
QGroupBox {{ margin-top: 14px; padding-top: 10px; font-weight: 600; }}
QGroupBox::title {{
    subcontrol-origin: margin; left: 10px; padding: 0 4px;
    color: {TEXT_DIM};
}}
QPushButton {{
    background: {BG_RAISED};
    border: 1px solid {GRID_STRONG};
    border-radius: 6px;
    padding: 6px 12px;
}}
QPushButton:hover {{ background: #2b3145; border-color: {ACCENT}; }}
QPushButton:pressed {{ background: {ACCENT}; color: #08101c; }}
QPushButton:checked {{ background: {ACCENT}; color: #08101c; border-color: {ACCENT}; }}
QPushButton#transport {{ font-size: 15px; padding: 6px 14px; }}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background: {BG_RAISED};
    border: 1px solid {GRID_STRONG};
    border-radius: 6px;
    padding: 4px 8px;
    selection-background-color: {ACCENT};
}}
QComboBox QAbstractItemView {{
    background: {BG_PANEL}; border: 1px solid {GRID_STRONG};
    selection-background-color: {ACCENT}; outline: none;
}}
QListWidget, QTreeWidget, QTableWidget, QPlainTextEdit, QTextEdit {{
    background: {BG_PANEL};
    border: 1px solid {GRID};
    border-radius: 8px;
    outline: none;
}}
QListWidget::item {{ padding: 5px 8px; border-radius: 4px; }}
QListWidget::item:selected {{ background: {ACCENT}; color: #08101c; }}
QSlider::groove:horizontal {{ background: {GRID}; height: 4px; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {ACCENT}; width: 12px; margin: -5px 0; border-radius: 6px;
}}
QScrollBar:vertical, QScrollBar:horizontal {{
    background: {BG}; border: none;
}}
QScrollBar:vertical {{ width: 11px; }}
QScrollBar:horizontal {{ height: 11px; }}
QScrollBar::handle {{ background: {GRID_STRONG}; border-radius: 5px; min-height: 30px; min-width: 30px; }}
QScrollBar::handle:hover {{ background: {ACCENT}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QSplitter::handle {{ background: {GRID}; }}
QToolBar {{ background: {BG_PANEL}; border-bottom: 1px solid {GRID}; spacing: 6px; padding: 6px; }}
QStatusBar {{ background: {BG_PANEL}; border-top: 1px solid {GRID}; color: {TEXT_DIM}; }}
QLabel#h1 {{ font-size: 15px; font-weight: 700; }}
QLabel#dim {{ color: {TEXT_DIM}; }}
QTabWidget::pane {{ border: 1px solid {GRID}; border-radius: 8px; top: -1px; }}
QTabBar::tab {{
    background: transparent; padding: 7px 14px; color: {TEXT_DIM};
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QProgressBar {{
    background: {BG_RAISED}; border: 1px solid {GRID_STRONG};
    border-radius: 6px; text-align: center; height: 16px;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 5px; }}
QCheckBox::indicator {{
    width: 14px; height: 14px; border-radius: 4px;
    border: 1px solid {GRID_STRONG}; background: {BG_RAISED};
}}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
"""
