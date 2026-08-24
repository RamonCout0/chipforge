#!/usr/bin/env python3
"""Chipforge Studio — ponto de entrada."""
import sys

from PyQt6.QtWidgets import QApplication

from chipforge.ui.main_window import MainWindow
from chipforge.ui import theme


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Chipforge Studio")
    app.setStyleSheet(theme.QSS)
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
