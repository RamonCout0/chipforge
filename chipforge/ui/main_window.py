"""
chipforge.ui.main_window
========================
A janela. Layout em tres colunas: tracks/patterns | piano roll | medidor.
Embaixo, o editor de instrumento. Em cima, o transporte.

A regra de ouro da interface: nada de dialogo obrigatorio. Voce abre o app e
ja tem um projeto tocavel; tudo o mais e opcional.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QColor, QKeySequence
from PyQt6.QtWidgets import (QApplication, QComboBox, QCheckBox, QDoubleSpinBox,
                             QFileDialog, QFrame, QGroupBox, QHBoxLayout,
                             QInputDialog, QLabel, QListWidget, QListWidgetItem,
                             QMainWindow, QMessageBox, QProgressDialog,
                             QPushButton, QSlider, QSplitter, QVBoxLayout,
                             QWidget)

from ..audio import engine
from ..audio.instrument import FACTORY_BANK, Instrument, factory_by_name
from ..audio.oscillators import OSC_LABELS_PT, WAVETABLES
from ..core import harmony as H
from ..core import intensity as INT
from ..core import theory as T
from ..core.project import Project
from ..midiio import smf
from . import theme
from .harmony_panel import HarmonyPanel
from .piano_roll import PianoRoll
from .sprite_stage import SpriteStage


def demo_project() -> Project:
    """Projeto de boas-vindas: um loop de 2 compassos ja tocavel."""
    p = Project("Sem titulo", bpm=132, bit_depth=8)
    p.key_root, p.scale = 9, "natural_minor"      # La menor
    lead = p.new_track("Lead", factory_by_name("NES Lead 25%"))
    bass = p.new_track("Baixo", factory_by_name("NES Triangle Bass"))
    drums = p.new_track("Bateria", factory_by_name("Kick"))
    ppq = p.ppq

    pl = p.new_pattern("Lead A", bars=2)
    for i, n in enumerate([69, 72, 76, 72, 74, 72, 69, 67]):
        pl.add(n, i * ppq // 2, ppq // 2 - 20, 105)
    pb = p.new_pattern("Baixo A", bars=2)
    for i, n in enumerate([45, 45, 41, 43]):
        pb.add(n, i * ppq, ppq - 30, 115)
    pd = p.new_pattern("Bateria A", bars=2)
    for i in range(8):
        pd.add(36, i * ppq, 120, 118)
    p.place(pl, lead, 0); p.place(pb, bass, 0); p.place(pd, drums, 0)
    return p


class MainWindow(QMainWindow):
    def __init__(self, project: Project | None = None):
        super().__init__()
        self.project = project or demo_project()
        self.path: Path | None = None
        self.player = engine.RealtimePlayer(self.project.sample_rate)
        self._audio: np.ndarray | None = None
        self._analyze_song = False
        self.soundfont = None          # banco GM carregado, se houver
        self.curva = None               # curva de intensidade da musica

        self.setWindowTitle("Chipforge Studio")
        self.resize(1580, 940)
        self.setStyleSheet(theme.QSS)

        self._build_toolbar()
        self._build_menus()
        self._build_body()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.setInterval(33)

        self._reload_patterns()
        self._reload_tracks()
        self.statusBar().showMessage("Pronto. Espaco toca, Esc para.")

    # ------------------------------------------------------------------
    def _build_toolbar(self):
        tb = self.addToolBar("Transporte")
        tb.setMovable(False)

        self.btn_play = QPushButton("▶  Tocar")
        self.btn_play.setObjectName("transport")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_stop = QPushButton("■")
        self.btn_stop.setObjectName("transport")
        self.btn_stop.clicked.connect(self.stop)
        self.btn_loop = QPushButton("↻ Loop")
        self.btn_loop.setCheckable(True)
        self.btn_loop.setChecked(True)
        for b in (self.btn_play, self.btn_stop, self.btn_loop):
            tb.addWidget(b)
        tb.addSeparator()

        tb.addWidget(QLabel(" BPM "))
        self.spin_bpm = QDoubleSpinBox()
        self.spin_bpm.setRange(20, 400); self.spin_bpm.setDecimals(1)
        self.spin_bpm.setValue(self.project.bpm)
        self.spin_bpm.valueChanged.connect(self._set_bpm)
        tb.addWidget(self.spin_bpm)

        tb.addWidget(QLabel("  Bits "))
        self.cmb_bits = QComboBox()
        self.cmb_bits.addItems(["8 bits (NES)", "16 bits (SNES)", "24 bits (limpo)"])
        self.cmb_bits.setCurrentIndex({8: 0, 16: 1}.get(self.project.bit_depth, 2))
        self.cmb_bits.currentIndexChanged.connect(
            lambda i: setattr(self.project, "bit_depth", [8, 16, 24][i]))
        tb.addWidget(self.cmb_bits)

        tb.addSeparator()
        tb.addWidget(QLabel("  Tom "))
        self.cmb_key = QComboBox()
        self.cmb_key.addItems(T.SHARP_NAMES)
        self.cmb_key.setCurrentIndex(self.project.key_root)
        self.cmb_key.currentIndexChanged.connect(self._set_key)
        tb.addWidget(self.cmb_key)

        self.cmb_scale = QComboBox()
        for k in T.SCALES:
            self.cmb_scale.addItem(T.SCALE_LABELS_PT.get(k, k), k)
        self.cmb_scale.setCurrentIndex(
            list(T.SCALES).index(self.project.scale))
        self.cmb_scale.currentIndexChanged.connect(self._set_scale)
        tb.addWidget(self.cmb_scale)

        self.chk_lock = QCheckBox("Scale Lock")
        self.chk_lock.setToolTip("Prende toda nota nova na escala escolhida")
        self.chk_lock.toggled.connect(
            lambda v: (setattr(self.project, "scale_lock", v), self.roll.refresh()))
        tb.addWidget(self.chk_lock)

        tb.addSeparator()
        tb.addWidget(QLabel("  Snap "))
        self.cmb_snap = QComboBox()
        for label, v in [("1/1", 1), ("1/2", 2), ("1/4", 4), ("1/8", 8),
                         ("1/16", 16), ("tercina", 3), ("livre", 0)]:
            self.cmb_snap.addItem(label, v)
        self.cmb_snap.setCurrentIndex(2)
        self.cmb_snap.currentIndexChanged.connect(
            lambda: (setattr(self.roll.grid, "snap",
                             self.cmb_snap.currentData()), self.roll.refresh()))
        tb.addWidget(self.cmb_snap)

        tb.addSeparator()
        self.chk_song = QCheckBox("Analisar musica inteira")
        self.chk_song.toggled.connect(self._set_analyze_scope)
        tb.addWidget(self.chk_song)

    def _build_menus(self):
        m = self.menuBar()
        f = m.addMenu("&Arquivo")
        for label, slot, key in [
            ("Novo", self.new_project, QKeySequence.StandardKey.New),
            ("Abrir...", self.open_project, QKeySequence.StandardKey.Open),
            ("Salvar", self.save_project, QKeySequence.StandardKey.Save),
            ("Salvar como...", lambda: self.save_project(True), None),
        ]:
            a = QAction(label, self)
            if key:
                a.setShortcut(key)
            a.triggered.connect(slot)
            f.addAction(a)
        f.addSeparator()
        for label, slot in [("Importar MIDI...", self.import_midi),
                            ("Exportar MIDI...", self.export_midi),
                            ("Exportar WAV...", self.export_wav),
                            ("Exportar curva de intensidade (JSON)...",
                             self.export_intensity)]:
            a = QAction(label, self); a.triggered.connect(slot); f.addAction(a)

        t = m.addMenu("&Ferramentas")
        for label, slot in [
            ("Gerar progressao harmonica...", self.gen_progression),
            ("Quantizar pattern", self.quantize_pattern),
            ("Humanizar (tirar a rigidez)", self.humanize_pattern),
            ("Dobrar pattern (x2)", self.double_pattern),
            ("Transpor +12 / -12", self.transpose_dialog),
        ]:
            a = QAction(label, self); a.triggered.connect(slot); t.addAction(a)

        b = m.addMenu("&Biblioteca")
        for label, slot in [
            ("Baixar banco de instrumentos (GeneralUser GS, 28 MB)",
             self.download_bank),
            ("Abrir banco .sf2 do computador...", self.open_sf2),
            (None, None),
            ("Instrumento do banco na track atual...", self.pick_gm_instrument),
            (None, None),
            ("Ver catalogo", self.show_catalog),
            ("Importar SoundFont para a biblioteca...", self.import_sf2),
            ("Importar pasta de MIDIs...", self.import_midi_folder),
        ]:
            if label is None:
                b.addSeparator()
                continue
            a = QAction(label, self); a.triggered.connect(slot); b.addAction(a)

        c = m.addMenu("&Cinematografico")
        for label, slot in [
            ("Voz cantada na track atual...", self.pick_voz),
            ("Criar coro SATB (4 tracks)", self.criar_coro),
            ("Desenho de som na track atual...", self.pick_cinema),
            (None, None),
            ("Sala do projeto...", self.escolher_sala),
            ("Perfil de master...", self.escolher_master),
            (None, None),
            ("Analisar faixa de referencia...", self.analisar_referencia),
            ("Relatorio de master desta faixa", self.relatorio_de_master),
        ]:
            if label is None:
                c.addSeparator()
                continue
            a = QAction(label, self); a.triggered.connect(slot); c.addAction(a)

    def _build_body(self):
        split = QSplitter(Qt.Orientation.Horizontal)

        # ---- coluna esquerda ------------------------------------------
        left = QWidget(); lv = QVBoxLayout(left)
        lv.setContentsMargins(8, 8, 4, 8)

        gb_t = QGroupBox("Tracks")
        tv = QVBoxLayout(gb_t)
        self.list_tracks = QListWidget()
        self.list_tracks.currentRowChanged.connect(self._track_selected)
        tv.addWidget(self.list_tracks)
        row = QHBoxLayout()
        for label, slot in [("+ Track", self.add_track),
                            ("M", self.toggle_mute), ("S", self.toggle_solo)]:
            btn = QPushButton(label); btn.clicked.connect(slot); row.addWidget(btn)
        tv.addLayout(row)
        lv.addWidget(gb_t, 1)

        gb_p = QGroupBox("Patterns")
        pv = QVBoxLayout(gb_p)
        self.list_patterns = QListWidget()
        self.list_patterns.currentRowChanged.connect(self._pattern_selected)
        pv.addWidget(self.list_patterns)
        row2 = QHBoxLayout()
        for label, slot in [("+ Novo", self.add_pattern),
                            ("Duplicar", self.dup_pattern),
                            ("Apagar", self.del_pattern)]:
            btn = QPushButton(label); btn.clicked.connect(slot); row2.addWidget(btn)
        pv.addLayout(row2)
        lv.addWidget(gb_p, 1)
        left.setMinimumWidth(230)

        # ---- centro ----------------------------------------------------
        center = QWidget(); cv = QVBoxLayout(center)
        cv.setContentsMargins(4, 8, 4, 8)
        self.lbl_pattern = QLabel("—"); self.lbl_pattern.setObjectName("h1")
        cv.addWidget(self.lbl_pattern)
        self.roll = PianoRoll(self.project)
        self.roll.notesChanged.connect(self._on_notes_changed)
        self.roll.previewNote.connect(self._preview)
        cv.addWidget(self.roll, 1)
        cv.addWidget(self._build_instrument_editor())

        # ---- direita: palco em cima, medidor embaixo --------------------
        self.harmony = HarmonyPanel()
        self.harmony.seekRequested.connect(self._seek_beat)
        self.palco = SpriteStage()
        direita = QSplitter(Qt.Orientation.Vertical)
        direita.addWidget(self.palco)
        direita.addWidget(self.harmony)
        direita.setSizes([250, 640])
        direita.setMinimumWidth(330)

        split.addWidget(left); split.addWidget(center); split.addWidget(direita)
        split.setSizes([250, 930, 380])
        self.setCentralWidget(split)

    def _build_instrument_editor(self) -> QWidget:
        box = QGroupBox("Instrumento da track")
        lay = QHBoxLayout(box)

        col1 = QVBoxLayout()
        self.cmb_preset = QComboBox()
        for group, items in FACTORY_BANK.items():
            self.cmb_preset.addItem(f"— {group} —")
            idx = self.cmb_preset.count() - 1
            self.cmb_preset.model().item(idx).setEnabled(False)
            for inst in items:
                self.cmb_preset.addItem("   " + inst.name, inst.name)
        self.cmb_preset.currentIndexChanged.connect(self._preset_changed)
        col1.addWidget(QLabel("Preset")); col1.addWidget(self.cmb_preset)

        self.cmb_osc = QComboBox()
        for k, v in OSC_LABELS_PT.items():
            self.cmb_osc.addItem(v, k)
        self.cmb_osc.currentIndexChanged.connect(self._osc_changed)
        col1.addWidget(QLabel("Forma de onda")); col1.addWidget(self.cmb_osc)

        self.cmb_wave = QComboBox()
        self.cmb_wave.addItems(list(WAVETABLES))
        self.cmb_wave.currentTextChanged.connect(self._wave_changed)
        col1.addWidget(QLabel("Wavetable")); col1.addWidget(self.cmb_wave)
        lay.addLayout(col1)

        self.sliders: dict[str, QSlider] = {}
        specs = [("attack", "Attack", 0, 100), ("decay", "Decay", 0, 100),
                 ("sustain", "Sustain", 0, 100), ("release", "Release", 0, 100),
                 ("vibrato", "Vibrato", 0, 100), ("volume", "Volume", 0, 100),
                 ("pan", "Pan", -50, 50)]
        for key, label, lo, hi in specs:
            c = QVBoxLayout()
            lbl = QLabel(label); lbl.setObjectName("dim")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            s = QSlider(Qt.Orientation.Vertical)
            s.setRange(lo, hi); s.setMinimumHeight(60)
            s.valueChanged.connect(self._slider_changed)
            self.sliders[key] = s
            c.addWidget(lbl); c.addWidget(s, 0, Qt.AlignmentFlag.AlignHCenter)
            lay.addLayout(c)

        col3 = QVBoxLayout()
        self.chk_arp = QCheckBox("Arpejo (acorde em 1 canal)")
        self.chk_arp.toggled.connect(self._slider_changed)
        col3.addWidget(self.chk_arp)
        self.btn_preview = QPushButton("Ouvir nota (C4)")
        self.btn_preview.clicked.connect(lambda: self._preview(60))
        col3.addWidget(self.btn_preview)
        col3.addStretch(1)
        lay.addLayout(col3)
        box.setMinimumHeight(150); box.setMaximumHeight(240)
        return box

    # ------------------------------------------------------------------
    # estado / listas
    # ------------------------------------------------------------------
    @property
    def current_track(self):
        i = self.list_tracks.currentRow()
        return self.project.tracks[i] if 0 <= i < len(self.project.tracks) else None

    @property
    def current_pattern(self):
        i = self.list_patterns.currentRow()
        keys = list(self.project.patterns)
        return self.project.patterns[keys[i]] if 0 <= i < len(keys) else None

    def _reload_tracks(self):
        self.list_tracks.blockSignals(True)
        self.list_tracks.clear()
        for i, t in enumerate(self.project.tracks):
            flags = ("M" if t.mute else "") + ("S" if t.solo else "")
            it = QListWidgetItem(f"{t.name}   {flags}")
            it.setForeground(QColor(theme.TRACK_COLORS[i % len(theme.TRACK_COLORS)]))
            self.list_tracks.addItem(it)
        self.list_tracks.blockSignals(False)
        if self.project.tracks and self.list_tracks.currentRow() < 0:
            self.list_tracks.setCurrentRow(0)

    def _reload_patterns(self):
        self.list_patterns.blockSignals(True)
        self.list_patterns.clear()
        for pid, pat in self.project.patterns.items():
            self.list_patterns.addItem(f"{pat.name}  ({pat.length_ticks//self.project.ticks_per_bar} comp.)")
        self.list_patterns.blockSignals(False)
        if self.project.patterns and self.list_patterns.currentRow() < 0:
            self.list_patterns.setCurrentRow(0)
            self._pattern_selected(0)

    def _track_selected(self, _row):
        t = self.current_track
        if not t:
            return
        inst = t.instrument
        for w in (self.cmb_osc, self.cmb_wave, self.cmb_preset):
            w.blockSignals(True)
        i = self.cmb_osc.findData(inst.osc)
        if i >= 0:
            self.cmb_osc.setCurrentIndex(i)
        # Sampler, voz e desenho de som nao sao osciladores de chip: os
        # controles de forma de onda nao significam nada neles. Testar pelo
        # nome do oscilador (e nao por "e sample?") faz os tres casos cairem
        # na mesma regra — senao a interface mostraria um duty cycle enquanto
        # a track canta, e mexer no combo escreveria um campo que o render
        # ignora.
        especial = getattr(inst, "osc", "") not in OSC_LABELS_PT
        self._show_sampler_slot(inst if especial else None)
        for w in (self.cmb_osc, self.cmb_wave, self.chk_arp):
            w.setEnabled(not especial)
        self.sliders["vibrato"].setEnabled(
            not especial or hasattr(inst, "naipe"))
        self.cmb_wave.setCurrentText(inst.wavetable)
        j = self.cmb_preset.findData(inst.name)
        if j >= 0:
            self.cmb_preset.setCurrentIndex(j)
        for w in (self.cmb_osc, self.cmb_wave, self.cmb_preset):
            w.blockSignals(False)
        for s in self.sliders.values():
            s.blockSignals(True)
        self.sliders["attack"].setValue(int(min(1.0, inst.env.attack / 0.5) * 100))
        self.sliders["decay"].setValue(int(min(1.0, inst.env.decay / 1.0) * 100))
        self.sliders["sustain"].setValue(int(inst.env.sustain * 100))
        self.sliders["release"].setValue(int(min(1.0, inst.env.release / 1.5) * 100))
        self.sliders["vibrato"].setValue(int(min(100, inst.vibrato_depth_cents)))
        self.sliders["volume"].setValue(int(t.volume * 100))
        self.sliders["pan"].setValue(int(t.pan * 50))
        for s in self.sliders.values():
            s.blockSignals(False)
        self.chk_arp.blockSignals(True)
        self.chk_arp.setChecked(bool(inst.arpeggio))
        self.chk_arp.blockSignals(False)
        # seleciona o pattern que toca nessa track, se houver
        for c in self.project.arrangement:
            if c.track_id == t.id:
                keys = list(self.project.patterns)
                if c.pattern_id in keys:
                    self.list_patterns.setCurrentRow(keys.index(c.pattern_id))
                break

    def _pattern_selected(self, _row):
        pat = self.current_pattern
        self.roll.set_pattern(pat)
        self.lbl_pattern.setText(pat.name if pat else "—")
        self._analyze()

    # ------------------------------------------------------------------
    # edicao de instrumento
    # ------------------------------------------------------------------
    _SAMPLER_SLOT = "\u25b8 do banco: "

    def _show_sampler_slot(self, inst):
        """Mostra o instrumento de sample no topo do combo de presets.

        Sem isso o combo continuaria exibindo o ultimo preset chiptune enquanto
        a track toca uma flauta — a interface mentiria sobre o proprio estado.
        """
        cb = self.cmb_preset
        cb.blockSignals(True)
        has_slot = cb.count() and cb.itemText(0).startswith(self._SAMPLER_SLOT)
        if inst is None:
            if has_slot:
                cb.removeItem(0)
        else:
            tipo = ("canta" if hasattr(inst, "naipe")
                    else "desenho de som" if getattr(inst, "osc", "") == "cinema"
                    else "do banco")
            label = f"{self._SAMPLER_SLOT}{tipo}: {inst.name}"
            if has_slot:
                cb.setItemText(0, label)
            else:
                cb.insertItem(0, label)
            cb.setCurrentIndex(0)
        cb.blockSignals(False)

    def _preset_changed(self):
        name = self.cmb_preset.currentData()
        t = self.current_track
        if not name or not t:
            return
        inst = factory_by_name(name)
        if inst:
            t.instrument = inst
            self._track_selected(None)
            self._invalidate_audio()

    def _osc_changed(self):
        t = self.current_track
        if t:
            t.instrument.osc = self.cmb_osc.currentData()
            self._invalidate_audio()

    def _wave_changed(self, name):
        t = self.current_track
        if t:
            t.instrument.wavetable = name
            self._invalidate_audio()

    def _slider_changed(self):
        t = self.current_track
        if not t:
            return
        i = t.instrument
        i.env.attack = self.sliders["attack"].value() / 100 * 0.5
        i.env.decay = self.sliders["decay"].value() / 100 * 1.0
        i.env.sustain = self.sliders["sustain"].value() / 100
        i.env.release = self.sliders["release"].value() / 100 * 1.5
        if getattr(i, "osc", "") != "sample":
            i.vibrato_depth_cents = self.sliders["vibrato"].value()
        t.volume = self.sliders["volume"].value() / 100
        t.pan = self.sliders["pan"].value() / 50
        if getattr(i, "osc", "") == "sample":
            self._invalidate_audio()
            return
        if self.chk_arp.isChecked() and not i.arpeggio:
            minor = "minor" in self.project.scale or "phrygian" in self.project.scale
            i.arpeggio = (0, 3, 7) if minor else (0, 4, 7)
        elif not self.chk_arp.isChecked():
            i.arpeggio = ()
        self._invalidate_audio()

    def _preview(self, pitch: int):
        t = self.current_track
        if t:
            self.player.preview(t.instrument, pitch, 0.45)

    # ------------------------------------------------------------------
    # analise
    # ------------------------------------------------------------------
    def _set_analyze_scope(self, v: bool):
        self._analyze_song = v
        self._analyze()

    def _on_notes_changed(self):
        self._invalidate_audio()
        self._analyze()

    def _analyze(self):
        if self._analyze_song:
            events = self.project.harmony_events()
        else:
            pat = self.current_pattern
            events = self.project.pattern_events(pat) if pat else []
        key = T.KeyGuess(self.project.key_root,
                         "major" if self.project.scale == "major" else "natural_minor",
                         0.9) if self.project.scale_lock else None
        rep = H.analyze(events, key=key, beats_per_chord=1.0)
        self.harmony.update_report(rep)
        self.statusBar().showMessage(
            f"{rep.verdict} — {rep.score:.0f}/100   |   {len(events)} notas analisadas")

    def _seek_beat(self, beat: float):
        self.roll.set_playhead(int(beat * self.project.ppq))
        self.roll.horizontalScrollBar().setValue(
            max(0, int(self.roll.grid.tick_to_x(int(beat * self.project.ppq))) - 200))

    # ------------------------------------------------------------------
    # transporte
    # ------------------------------------------------------------------
    def _invalidate_audio(self):
        self._audio = None
        self.curva = None

    def _intensidade(self):
        if self.curva is None:
            self.curva = INT.calcular(self.project)
        return self.curva

    def _render(self) -> np.ndarray:
        if self._audio is None:
            self.statusBar().showMessage("Renderizando...")
            self._audio = engine.render_project(self.project)
            self.statusBar().showMessage("Pronto.")
        return self._audio

    def toggle_play(self):
        if self.timer.isActive():
            self.stop()
        else:
            self.player.sr = self.project.sample_rate
            self.player.play(self._render(), loop=self.btn_loop.isChecked())
            self.btn_play.setText("❚❚  Pausar")
            self.timer.start()

    def stop(self):
        self.timer.stop()
        self.player.stop()
        self.btn_play.setText("▶  Tocar")
        self.roll.set_playhead(-1)
        self.palco.repousar()

    def _tick(self):
        sec = self.player.position_seconds
        self.roll.set_playhead(self.project.seconds_to_ticks(sec))
        c = self._intensidade()
        self.palco.set_estado(c.em(sec), c.nivel_em(sec), tocando=True)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Space:
            self.toggle_play()
        elif e.key() == Qt.Key.Key_Escape:
            self.stop()
        else:
            super().keyPressEvent(e)

    def _set_bpm(self, v):
        self.project.bpm = v
        self._invalidate_audio()

    def _set_key(self, i):
        self.project.key_root = i
        self.roll.refresh(); self._analyze()

    def _set_scale(self, _i):
        self.project.scale = self.cmb_scale.currentData()
        self.roll.refresh(); self._analyze()

    # ------------------------------------------------------------------
    # tracks / patterns
    # ------------------------------------------------------------------
    def add_track(self):
        names = [i.name for g in FACTORY_BANK.values() for i in g]
        name, ok = QInputDialog.getItem(self, "Nova track", "Instrumento:", names, 0, False)
        if not ok:
            return
        t = self.project.new_track(name, factory_by_name(name))
        pat = self.project.new_pattern(f"{name} A", bars=2)
        self.project.place(pat, t, 0)
        self._reload_tracks(); self._reload_patterns()
        self.list_tracks.setCurrentRow(len(self.project.tracks) - 1)
        self._invalidate_audio()

    def toggle_mute(self):
        t = self.current_track
        if t:
            t.mute = not t.mute
            self._reload_tracks(); self._invalidate_audio()

    def toggle_solo(self):
        t = self.current_track
        if t:
            t.solo = not t.solo
            self._reload_tracks(); self._invalidate_audio()

    def add_pattern(self):
        t = self.current_track
        pat = self.project.new_pattern(bars=2)
        if t:
            self.project.place(pat, t, 0)
        self._reload_patterns()
        self.list_patterns.setCurrentRow(len(self.project.patterns) - 1)

    def dup_pattern(self):
        pat = self.current_pattern
        if not pat:
            return
        new = self.project.new_pattern(pat.name + " (copia)",
                                       bars=pat.length_ticks // self.project.ticks_per_bar)
        new.notes = [type(n)(n.start, n.dur, n.pitch, n.velocity) for n in pat.notes]
        self._reload_patterns()

    def del_pattern(self):
        pat = self.current_pattern
        if not pat:
            return
        self.project.patterns.pop(pat.id, None)
        self.project.arrangement = [c for c in self.project.arrangement
                                    if c.pattern_id != pat.id]
        self._reload_patterns(); self._invalidate_audio()

    # ------------------------------------------------------------------
    # ferramentas
    # ------------------------------------------------------------------
    def gen_progression(self):
        names = list(T.PROGRESSIONS)
        name, ok = QInputDialog.getItem(self, "Gerar progressao",
                                        "Escolha um esqueleto:", names, 0, False)
        if not ok:
            return
        pat = self.current_pattern
        if not pat:
            return
        degrees = T.PROGRESSIONS[name]
        mode = self.project.scale if self.project.scale in T.SCALES else "major"
        chords = T.build_progression(self.project.key_root, mode, degrees, octave=3)
        ppq = self.project.ppq
        pat.notes.clear()
        for i, ch in enumerate(chords):
            for pitch in ch:
                pat.add(pitch, i * ppq * 2, ppq * 2 - 20, 95)
        pat.length_ticks = max(pat.length_ticks, len(chords) * ppq * 2)
        self.roll.refresh(); self._on_notes_changed()

    def quantize_pattern(self):
        pat = self.current_pattern
        if not pat:
            return
        s = self.roll.grid.snap_ticks
        for n in pat.notes:
            n.start = int(round(n.start / s)) * s
            n.dur = max(s, int(round(n.dur / s)) * s)
        self.roll.refresh(); self._on_notes_changed()

    def humanize_pattern(self):
        import random
        pat = self.current_pattern
        if not pat:
            return
        for n in pat.notes:
            n.start = max(0, n.start + random.randint(-8, 8))
            n.velocity = max(40, min(127, n.velocity + random.randint(-12, 12)))
        self.roll.refresh(); self._on_notes_changed()

    def double_pattern(self):
        pat = self.current_pattern
        if not pat:
            return
        L = pat.length_ticks
        pat.notes += [type(n)(n.start + L, n.dur, n.pitch, n.velocity)
                      for n in list(pat.notes)]
        pat.length_ticks = L * 2
        self.roll.refresh(); self._on_notes_changed()

    def transpose_dialog(self):
        pat = self.current_pattern
        if not pat:
            return
        v, ok = QInputDialog.getInt(self, "Transpor", "Semitons:", 12, -36, 36)
        if ok:
            for n in pat.notes:
                n.pitch = max(0, min(127, n.pitch + v))
            self.roll.refresh(); self._on_notes_changed()

    # ------------------------------------------------------------------
    # arquivo
    # ------------------------------------------------------------------
    def new_project(self):
        self.stop()
        self.project = demo_project()
        self.path = None
        self.roll.grid.project = self.project
        self._reload_tracks(); self._reload_patterns(); self._invalidate_audio()

    def open_project(self):
        f, _ = QFileDialog.getOpenFileName(self, "Abrir projeto", "",
                                           "Chipforge (*.chip)")
        if not f:
            return
        self.stop()
        self.project = Project.load(f)
        self.path = Path(f)
        self.roll.grid.project = self.project
        self.spin_bpm.setValue(self.project.bpm)
        self._reload_tracks(); self._reload_patterns(); self._invalidate_audio()

    def save_project(self, as_new: bool = False):
        if as_new or not self.path:
            f, _ = QFileDialog.getSaveFileName(self, "Salvar projeto",
                                               self.project.name, "Chipforge (*.chip)")
            if not f:
                return
            self.path = Path(f)
        p = self.project.save(self.path)
        self.statusBar().showMessage(f"Salvo em {p}")

    def import_midi(self):
        f, _ = QFileDialog.getOpenFileName(self, "Importar MIDI", "",
                                           "MIDI (*.mid *.midi)")
        if not f:
            return
        try:
            mf = smf.read_midi(f)
            self.project = smf.midi_to_project(mf)
        except Exception as e:
            QMessageBox.warning(self, "Erro ao importar", str(e))
            return
        self.roll.grid.project = self.project
        self.spin_bpm.setValue(self.project.bpm)
        self._reload_tracks(); self._reload_patterns(); self._invalidate_audio()
        self._analyze()

    def export_midi(self):
        f, _ = QFileDialog.getSaveFileName(self, "Exportar MIDI",
                                           self.project.name, "MIDI (*.mid)")
        if f:
            smf.write_midi(smf.project_to_midi(self.project), f)
            self.statusBar().showMessage(f"MIDI exportado: {f}")

    def export_wav(self):
        f, _ = QFileDialog.getSaveFileName(self, "Exportar WAV",
                                           self.project.name, "WAV (*.wav)")
        if f:
            engine.export_wav(self._render(), f, self.project.sample_rate,
                              16 if self.project.bit_depth > 8 else 8)
            self.statusBar().showMessage(f"WAV exportado: {f}")

    def export_intensity(self):
        f, _ = QFileDialog.getSaveFileName(
            self, "Exportar curva de intensidade",
            self.project.name + "_intensidade", "JSON (*.json)")
        if not f:
            return
        caminho = self._intensidade().exportar_json(f, self.project.name)
        QMessageBox.information(
            self, "Exportado",
            f"{caminho}\n\nTempo -> nivel, para a engine do jogo tocar o mesmo "
            "sprite que voce ve aqui, sem precisar analisar audio em tempo real.")

    # ------------------------------------------------------------------
    # cinematografico
    # ------------------------------------------------------------------
    def _exige_track(self):
        t = self.current_track
        if not t:
            QMessageBox.information(self, "Sem track", "Selecione uma track antes.")
        return t

    def pick_voz(self):
        """Troca o instrumento da track por um cantor sintetizado."""
        from ..audio import voz as VOZ
        t = self._exige_track()
        if not t:
            return
        nomes = VOZ.nomes_de_preset()
        escolha, ok = QInputDialog.getItem(self, "Voz cantada",
                                           "Preset:", nomes, 0, False)
        if not ok or not escolha:
            return
        t.instrument = VOZ.voz(escolha)
        t.name = escolha
        # sem sala, voz sintetizada soa colada no microfone
        if not t.reverb_send:
            t.reverb_send = 0.45
        self._garantir_sala()
        self._reload_tracks()
        self._track_selected(None)
        self._invalidate_audio()
        self.statusBar().showMessage(f"{escolha} aplicado na track.")

    def criar_coro(self):
        """Cria as quatro tracks de um coro, cada naipe com sua dobra.

        Sao quatro tracks e nao uma porque naipe se equilibra separado: num
        coro de verdade o baixo quase sempre precisa de mais nivel que o
        soprano para ser ouvido no mesmo lugar.
        """
        from ..audio import voz as VOZ
        criadas = []
        for naipe, preset, pan in (
                ("Soprano", "Coro soprano (aah)", -0.45),
                ("Contralto", "Coro contralto (ooh)", 0.40),
                ("Tenor", "Coro tenor (aah)", -0.30),
                ("Baixo", "Coro baixo (ooh)", 0.28)):
            tr = self.project.new_track(f"Coro {naipe}",
                                        VOZ.voz(preset, pan=pan,
                                                seed=abs(hash(naipe)) % 999))
            tr.section = dict(VOZ.SECAO_CORO_INTIMO)
            tr.reverb_send = 0.75
            tr.volume = 0.46
            criadas.append(tr.name)
        self._garantir_sala()
        self._reload_tracks()
        self._invalidate_audio()
        QMessageBox.information(
            self, "Coro criado",
            "Quatro tracks novas:\n  " + "\n  ".join(criadas) +
            "\n\nCada naipe ja vem com tres gargantas (dobra de secao) e envio "
            "de sala. Escreva o acorde distribuido entre elas — o soprano leva "
            "a melodia, o baixo leva a fundamental.")

    def pick_cinema(self):
        from ..audio import cinema as CIN
        t = self._exige_track()
        if not t:
            return
        nomes = CIN.nomes_de_preset()
        escolha, ok = QInputDialog.getItem(self, "Desenho de som",
                                           "Preset:", nomes, 0, False)
        if not ok or not escolha:
            return
        t.instrument = CIN.instrumento(escolha)
        t.name = escolha
        self._reload_tracks()
        self._track_selected(None)
        self._invalidate_audio()
        self.statusBar().showMessage(f"{escolha} aplicado na track.")

    def _garantir_sala(self, nome: str = "salao"):
        """Liga a sala por convolucao se o projeto ainda estiver seco."""
        if not self.project.master_reverb:
            self.project.master_reverb = {"sala": nome, "wet": 0.9,
                                          "corte_grave": 150.0}

    def escolher_sala(self):
        from ..audio import espaco
        nomes = list(espaco.SALAS)
        rotulos = [espaco.SALAS_PT[k] for k in nomes] + ["(nenhuma — seco)"]
        atual = (self.project.master_reverb or {}).get("sala")
        idx = nomes.index(atual) if atual in nomes else len(nomes)
        escolha, ok = QInputDialog.getItem(
            self, "Sala do projeto",
            "Todos os canais com envio de reverb tocam nesta sala:",
            rotulos, idx, False)
        if not ok:
            return
        if escolha.startswith("(nenhuma"):
            self.project.master_reverb = None
        else:
            self.project.master_reverb = {
                "sala": nomes[rotulos.index(escolha)], "wet": 0.9,
                "corte_grave": 150.0}
        self._invalidate_audio()
        self.statusBar().showMessage(f"Sala: {escolha}")

    def escolher_master(self):
        from ..audio import master as MST
        nomes = list(MST.PERFIS) + ["(nenhum — so limitador de pico)"]
        descricao = {
            "cinematico": "trilha de jogo/filme, -14 LUFS",
            "trailer": "mais alto e mais brilhante, -11 LUFS",
            "jogo": "com folga para o mix do jogo, -16 LUFS",
            "chiptune": "sem excitador nem largura, so nivel",
            "cru": "processa mas nao persegue loudness",
        }
        rotulos = [f"{k} — {descricao.get(k, '')}" for k in MST.PERFIS] + [nomes[-1]]
        atual = getattr(self.project, "master_perfil", None)
        idx = list(MST.PERFIS).index(atual) if atual in MST.PERFIS else len(rotulos) - 1
        escolha, ok = QInputDialog.getItem(self, "Perfil de master",
                                           "Cadeia aplicada na exportacao:",
                                           rotulos, idx, False)
        if not ok:
            return
        self.project.master_perfil = (None if escolha.startswith("(nenhum")
                                      else escolha.split(" — ")[0])
        if self.project.master_perfil and self.project.bit_depth < 24:
            r = QMessageBox.question(
                self, "Bit depth",
                "O projeto esta em 8 ou 16 bits, e o bitcrush roda DEPOIS do "
                "master — o que desfaz boa parte do trabalho da cadeia.\n\n"
                "Passar para 24 bits?")
            if r == QMessageBox.StandardButton.Yes:
                self.project.bit_depth = 24
        self._invalidate_audio()
        self.statusBar().showMessage(f"Master: {self.project.master_perfil or 'nenhum'}")

    def analisar_referencia(self):
        """Mede uma faixa que voce quer alcancar e adota o perfil dela.

        O que sai daqui e forma e mixagem — andamento, tom, secoes, balanco
        espectral e loudness. Nao ha transcricao: o alvo e o EQUILIBRIO da
        faixa, nao as notas dela.
        """
        f, _ = QFileDialog.getOpenFileName(
            self, "Faixa de referencia", "",
            "Audio (*.mp3 *.wav *.flac *.ogg *.m4a);;Todos (*)")
        if not f:
            return
        prog = QProgressDialog("Medindo a referencia...", None, 0, 0, self)
        prog.setWindowTitle("Analise")
        prog.show()
        QApplication.processEvents()
        try:
            from ..analysis import reference as REF
            r = REF.analisar(f)
        except Exception as e:
            prog.close()
            QMessageBox.warning(
                self, "Nao consegui analisar",
                f"{e}\n\nEsta analise precisa do ffmpeg instalado e no PATH.")
            return
        prog.close()

        self.project.master_alvo = r.perfil_alvo()
        if not self.project.master_perfil:
            self.project.master_perfil = "cinematico"
        self._invalidate_audio()
        QMessageBox.information(
            self, "Referencia adotada",
            r.relatorio_pt() +
            "\n\nO balanco espectral acima virou o alvo do master deste "
            "projeto, e o perfil 'cinematico' foi ligado.")

    def relatorio_de_master(self):
        """Mede a propria faixa: LUFS, faixa de loudness e pico real."""
        from ..audio import master as MST
        audio = self._render()
        prog = QProgressDialog("Medindo...", None, 0, 0, self)
        prog.setWindowTitle("Master")
        prog.show()
        QApplication.processEvents()
        m = MST.medir(audio, self.project.sample_rate)
        prog.close()
        alvo = MST.PERFIS.get(getattr(self.project, "master_perfil", None) or "",
                              {}).get("alvo_lufs") or -14.0
        QMessageBox.information(self, "Relatorio de master",
                                m.relatorio_pt(alvo))

    # ------------------------------------------------------------------
    # biblioteca
    # ------------------------------------------------------------------
    # -- banco de instrumentos ------------------------------------------
    def _load_bank(self, path) -> bool:
        from ..audio.sampler import open_soundfont
        try:
            self.soundfont = open_soundfont(path)
        except Exception as e:
            QMessageBox.warning(self, "Banco invalido", str(e))
            return False
        self.statusBar().showMessage(
            f"Banco carregado: {self.soundfont.name} — "
            f"{len(self.soundfont.presets)} presets")
        return True

    def download_bank(self):
        from ..midiio import library as L
        dlg = QProgressDialog("Baixando GeneralUser GS...", "Cancelar", 0, 100, self)
        dlg.setWindowTitle("Biblioteca")
        dlg.setMinimumDuration(0)
        dlg.setValue(0)
        cancelled = {"v": False}

        def prog(got, total):
            if dlg.wasCanceled():
                cancelled["v"] = True
                raise RuntimeError("cancelado pelo usuario")
            dlg.setValue(int(got * 100 / total) if total else 0)
            dlg.setLabelText(f"Baixando GeneralUser GS...  {got/1e6:.1f} MB")
            QApplication.processEvents()

        try:
            sf2 = L.ensure("generaluser_gs", prog)
        except Exception as e:
            dlg.close()
            if not cancelled["v"]:
                QMessageBox.warning(
                    self, "Nao consegui baixar",
                    f"{e}\n\nAlternativa: baixe o .sf2 pelo navegador e use "
                    "Biblioteca -> Abrir banco .sf2 do computador.")
            return
        dlg.close()
        if sf2 and self._load_bank(sf2):
            QMessageBox.information(
                self, "Pronto",
                f"{self.soundfont.name}\n"
                f"{len(self.soundfont.presets)} presets, "
                f"{len(self.soundfont.samples)} samples.\n\n"
                "Use Biblioteca -> Instrumento do banco na track atual.")

    def open_sf2(self):
        f, _ = QFileDialog.getOpenFileName(self, "Abrir banco", "",
                                           "SoundFont (*.sf2 *.sf3)")
        if f:
            self._load_bank(f)

    def pick_gm_instrument(self):
        from ..audio.sampler import SamplerInstrument
        t = self.current_track
        if not t:
            QMessageBox.information(self, "Sem track", "Selecione uma track antes.")
            return
        if self.soundfont is None:
            r = QMessageBox.question(
                self, "Nenhum banco carregado",
                "Baixar o GeneralUser GS agora? (28 MB, licenca livre)")
            if r != QMessageBox.StandardButton.Yes:
                return
            self.download_bank()
            if self.soundfont is None:
                return
        nomes = self.soundfont.preset_names()
        escolha, ok = QInputDialog.getItem(
            self, "Instrumento do banco",
            "Digite para filtrar (banco:programa  nome):", nomes, 0, True)
        if not ok or not escolha:
            return
        alvo = escolha.split("  ", 1)[-1].strip()
        hits = [p for p in self.soundfont.presets if p.name == alvo] or \
            self.soundfont.find(alvo)
        if not hits:
            QMessageBox.warning(self, "Nao achei", f"Preset '{alvo}' inexistente.")
            return
        t.instrument = SamplerInstrument(self.soundfont, hits[0])
        t.name = hits[0].name
        self._reload_tracks()
        self._track_selected(None)
        self._invalidate_audio()
        self.statusBar().showMessage(f"{hits[0].name} aplicado na track.")

    def show_catalog(self):
        from ..midiio import library as L
        QMessageBox.information(self, "Biblioteca", L.catalog_summary())

    def import_sf2(self):
        from ..midiio import library as L
        f, _ = QFileDialog.getOpenFileName(self, "SoundFont", "", "SoundFont (*.sf2 *.sf3 *.zip)")
        if f:
            paths = L.import_local(f, "soundfont")
            QMessageBox.information(self, "Importado",
                                    "\n".join(str(p) for p in paths[:20]))

    def import_midi_folder(self):
        from ..midiio import library as L
        d = QFileDialog.getExistingDirectory(self, "Pasta com MIDIs")
        if d:
            n = 0
            for p in Path(d).rglob("*.mid"):
                L.import_local(p, "midi"); n += 1
            QMessageBox.information(self, "Importado", f"{n} arquivos MIDI.")
