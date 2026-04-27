from __future__ import annotations

"""
calibration.py
Image upload + drag-to-rank panel.
Emits calibration_ready(clips, scores) when the user clicks Start.

Fixes applied
─────────────
• _start() encoding loop moved into _EncoderWorker / QThread so the UI does
  not freeze during encode_image() calls.  A progress bar updates every step.

• _upload() now collects paths that failed to open and shows a warning label
  so the user knows which files were skipped.

• set_hub(hub) public method added so MainWindow can defer hub assignment until
  the background ModelLoader thread has finished.

• ThumbCard._drag_start (BUG FIX): initialised to QPoint() in __init__ so
  mouseMoveEvent never raises AttributeError before the first click.

• _EncoderWorker / _enc_thread cleanup (BUG FIX): thread.finished connects
  deleteLater() on both objects; instance variables are cleared to None.

• Encoding progress bar added beneath the start button.

• Finder drag-and-drop (NEW): images can be dragged directly from macOS Finder
  (or any OS file manager) into the rank grid.  RankGrid.dragEnterEvent now
  accepts hasUrls() drops in addition to internal card reorders.  A purple
  highlight border appears on the grid while files are being dragged over it.
  The shared _load_files() method handles both dialog picks and dropped paths.

• Fast file dialog (NEW): _upload() now uses DontUseNativeDialog + ReadOnly to
  bypass macOS thumbnail generation, which was the cause of Finder lag when
  the dialog opened.  The dialog also opens in the last-used image directory
  instead of defaulting to whatever macOS chose.

• Font (macOS): all "Consolas" references replaced with "Menlo" (the system
  monospace on macOS) to eliminate the 300–800 ms font-alias scan at startup.

• All print() calls replaced with logging.
"""

import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from PyQt6.QtCore    import Qt, pyqtSignal, QMimeData, QByteArray, QPoint, QObject, QThread
from PyQt6.QtGui     import (QPixmap, QColor, QPainter, QFont, QDrag,
                             QBrush, QPen, QPainterPath)
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QFileDialog, QScrollArea,
                             QFrame, QApplication, QProgressBar)

_log = logging.getLogger(__name__)

THUMB_SIZE = 80
MAX_IMAGES = 50
MONO       = "Menlo"   # system monospace on macOS; avoids Consolas alias scan

# Image extensions accepted for both dialog filter and Finder drops
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


# ── Encoder worker ────────────────────────────────────────────────────────────

class _EncoderWorker(QObject):
    """
    Encodes a list of (PIL image, score) pairs on a background thread.
    Emits progress(done, total) after each image, then
    finished(clips, mobilenets, scores).
    """
    progress = pyqtSignal(int, int)              # (done, total)
    finished = pyqtSignal(object, object, object) # (clips, mobilenets, scores)

    def __init__(self, hub, ordered: list, parent=None):
        super().__init__(parent)
        self._hub     = hub
        self._ordered = ordered

    def run(self):
        clips_list, mnvs_list, scores_list = [], [], []
        total = len(self._ordered)
        _log.info("Encoding %d calibration images …", total)
        for i, (pil, score) in enumerate(self._ordered):
            clip_emb, mnv_emb = self._hub.encode_image(pil)
            clips_list.append(clip_emb)
            mnvs_list.append(mnv_emb)
            scores_list.append(score)
            self.progress.emit(i + 1, total)

        clips     = np.stack(clips_list).astype(np.float32)
        mobilenets = np.stack(mnvs_list).astype(np.float32)
        scores    = np.array(scores_list, dtype=np.float32)
        _log.info("Encoding complete — %d embeddings ready.", total)
        self.finished.emit(clips, mobilenets, scores)


# ── Thumbnail card ────────────────────────────────────────────────────────────

class ThumbCard(QFrame):
    """Draggable image thumbnail with rank badge."""

    drag_started = pyqtSignal(int)   # emits card index

    def __init__(self, idx, pil_img, path, parent=None):
        super().__init__(parent)
        self.card_idx    = idx
        self.pil_img     = pil_img
        self.path        = path
        self._rank       = idx + 1
        self._dragging   = False
        self._drag_start = QPoint()   # BUG FIX: initialise so mouseMoveEvent is safe

        self.setFixedSize(THUMB_SIZE + 16, THUMB_SIZE + 32)
        self.setStyleSheet("QFrame { background: transparent; border: none; }")
        self._pixmap = self._make_pixmap(pil_img)

    @property
    def rank(self): return self._rank
    @rank.setter
    def rank(self, v): self._rank = v; self.update()

    def _make_pixmap(self, pil):
        img = pil.resize((THUMB_SIZE, THUMB_SIZE), PILImage.LANCZOS)
        data = img.convert("RGBA").tobytes("raw", "RGBA")
        from PyQt6.QtGui import QImage
        qi = QImage(data, THUMB_SIZE, THUMB_SIZE, QImage.Format.Format_RGBA8888)
        return QPixmap.fromImage(qi)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        path = QPainterPath()
        path.addRoundedRect(4, 0, THUMB_SIZE + 8, THUMB_SIZE + 28, 6, 6)
        p.fillPath(path, QColor(20, 18, 40, 200))

        p.drawPixmap(8, 2, self._pixmap)

        badge_color = {1: (255, 210, 10), 2: (180, 180, 200), 3: (200, 120, 60)
                       }.get(self._rank, (80, 70, 130))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(*badge_color, 220)))
        p.drawRoundedRect(8, THUMB_SIZE + 6, 24, 16, 4, 4)
        font = QFont(MONO, 8)
        font.setBold(True)
        p.setFont(font)
        p.setPen(QColor(10, 8, 20))
        p.drawText(8, THUMB_SIZE + 6, 24, 16, Qt.AlignmentFlag.AlignCenter,
                   str(self._rank))

        if self.underMouse():
            p.setPen(QPen(QColor(140, 100, 255, 160), 1.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(4, 0, THUMB_SIZE + 8, THUMB_SIZE + 28, 6, 6)

        p.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.pos()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.MouseButton.LeftButton):
            dist = (event.pos() - self._drag_start).manhattanLength()
            if dist > 10:
                self._start_drag()

    def _start_drag(self):
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData("application/x-card-index",
                     QByteArray(str(self.card_idx).encode()))
        drag.setMimeData(mime)

        preview = QPixmap(self.size())
        preview.fill(Qt.GlobalColor.transparent)
        self.render(preview)
        drag.setPixmap(preview)
        drag.setHotSpot(QPoint(self.width() // 2, self.height() // 2))
        drag.exec(Qt.DropAction.MoveAction)


# ── Rank grid ─────────────────────────────────────────────────────────────────

class RankGrid(QWidget):
    """
    Drag-and-drop grid that holds ThumbCards.

    Accepts two kinds of drops:
      • Internal card reorders  (application/x-card-index)
      • External file drops     (URLs from Finder / file manager)

    When files are dragged over the widget a purple highlight border is drawn
    to give the user clear visual feedback that the drop will be accepted.
    """
    order_changed = pyqtSignal()
    files_dropped = pyqtSignal(list)   # emits list[str] of local file paths

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._cards: list[ThumbCard] = []
        self._layout      = None
        self._drop_active = False   # True while a valid Finder drag is over us
        self._rebuild_layout()

    # ── Layout ────────────────────────────────────────────────────────────

    def _rebuild_layout(self):
        from PyQt6.QtWidgets import QGridLayout
        if self._layout is None:
            self._layout = QGridLayout(self)
            self._layout.setSpacing(6)
            self._layout.setContentsMargins(4, 4, 4, 4)
        else:
            live_cards = set(id(c) for c in self._cards)
            while self._layout.count():
                item = self._layout.takeAt(0)
                w = item.widget()
                if w:
                    if id(w) not in live_cards:
                        w.hide()
                        w.deleteLater()
                    else:
                        w.hide()
        for i, card in enumerate(self._cards):
            # Two columns — best (rank 1) top-left, fills top-to-bottom
            row, col = divmod(i, 2)
            self._layout.addWidget(card, row, col)
            card.show()
        for i, card in enumerate(self._cards):
            card.rank = i + 1

    def add_card(self, pil_img, path):
        idx  = len(self._cards)
        card = ThumbCard(idx, pil_img, path)
        self._cards.append(card)
        self._rebuild_layout()

    def clear_cards(self):
        self._cards = []
        self._rebuild_layout()

    def get_ordered_data(self):
        """Returns list of (pil_image, rank_score) in current display order."""
        n = len(self._cards)
        result = []
        for i, card in enumerate(self._cards):
            score = 1.0 - i / max(n - 1, 1)   # top-ranked → 1.0, last → 0.0
            result.append((card.pil_img, score))
        return result

    # ── Drop highlight ────────────────────────────────────────────────────

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._drop_active:
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            pen = QPen(QColor(140, 80, 255, 200), 2.5, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.setBrush(QBrush(QColor(100, 60, 200, 18)))
            p.drawRoundedRect(3, 3, self.width() - 6, self.height() - 6, 8, 8)
            p.end()

    # ── Drag events ───────────────────────────────────────────────────────

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-card-index"):
            event.acceptProposedAction()
        elif event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if any(Path(u.toLocalFile()).suffix.lower() in SUPPORTED_EXTS
                   for u in urls):
                self._drop_active = True
                self.update()
                event.acceptProposedAction()

    def dragMoveEvent(self, event):
        event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        # Clear the highlight when the drag leaves without dropping
        if self._drop_active:
            self._drop_active = False
            self.update()

    def dropEvent(self, event):
        if event.mimeData().hasFormat("application/x-card-index"):
            # ── Internal card reorder ──────────────────────────────────────
            raw      = event.mimeData().data("application/x-card-index").data()
            src_idx  = int(raw.decode())
            drop_pos = event.position().toPoint()
            target_idx = self._find_target(drop_pos)
            if target_idx is not None and target_idx != src_idx:
                card = self._cards.pop(src_idx)
                self._cards.insert(target_idx, card)
                for i, c in enumerate(self._cards):
                    c.card_idx = i
                self._rebuild_layout()
                self.order_changed.emit()
            event.acceptProposedAction()

        elif event.mimeData().hasUrls():
            # ── External Finder / file-manager drop ───────────────────────
            self._drop_active = False
            self.update()
            paths = [
                u.toLocalFile()
                for u in event.mimeData().urls()
                if Path(u.toLocalFile()).suffix.lower() in SUPPORTED_EXTS
            ]
            if paths:
                self.files_dropped.emit(paths)
            event.acceptProposedAction()

    def _find_target(self, pos):
        for i, card in enumerate(self._cards):
            if card.geometry().contains(pos):
                return i
        return len(self._cards) - 1


# ── Calibration panel ─────────────────────────────────────────────────────────

class CalibrationPanel(QWidget):
    """
    Full calibration panel.
    Signals:
      calibration_ready(clips_array, mobilenets_array, scores_array)
    """
    calibration_ready = pyqtSignal(object, object, object)  # clips, mobilenets, scores

    def __init__(self, hub, parent=None):
        super().__init__(parent)
        self._hub      = hub   # may be None until set_hub() is called
        self._paths    = []
        self._pil_imgs = []

        self._enc_thread: QThread | None       = None
        self._enc_worker: _EncoderWorker | None = None

        self.setStyleSheet("background: transparent;")
        self._build_ui()

    # ── Public API ────────────────────────────────────────────────────────

    def set_hub(self, hub):
        self._hub = hub
        self._refresh_start_btn()

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 8, 12, 8)
        root.setSpacing(8)

        # ── Header row ────────────────────────────────────────────────────
        hdr = QHBoxLayout()
        title = QLabel("CALIBRATION")
        title.setStyleSheet(
            f"color: #8878cc; font: bold 10px '{MONO}'; letter-spacing: 2px;")
        hdr.addWidget(title)
        hdr.addStretch()

        self._btn_upload = QPushButton("+ Upload Images")
        self._btn_upload.setFixedHeight(28)
        self._btn_upload.setStyleSheet(self._btn_style("#5540aa", "#7760cc"))
        self._btn_upload.clicked.connect(self._upload)
        hdr.addWidget(self._btn_upload)

        self._btn_clear = QPushButton("Clear")
        self._btn_clear.setFixedHeight(28)
        self._btn_clear.setStyleSheet(self._btn_style("#443355", "#664488"))
        self._btn_clear.clicked.connect(self._clear)
        hdr.addWidget(self._btn_clear)

        root.addLayout(hdr)

        # ── Hint label ───────────────────────────────────────────────────
        self._hint = QLabel(
            "Upload images or drag them in from Finder. "
            "Drag to reorder by preference. Top = most loved.")
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet(f"color: #605880; font: 9px '{MONO}';")
        root.addWidget(self._hint)

        # ── Scroll area with rank grid ────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea { background: transparent; border: 1px solid #2a2040; border-radius: 6px; }
            QScrollBar:vertical { background: #120e22; width: 8px; border-radius: 4px; }
            QScrollBar::handle:vertical { background: #4a3a80; border-radius: 4px; min-height: 20px; }
        """)
        self._grid = RankGrid()
        self._grid.order_changed.connect(lambda: None)
        self._grid.files_dropped.connect(self._load_files)   # Finder drop → loader
        scroll.setWidget(self._grid)
        scroll.setMinimumHeight(130)
        root.addWidget(scroll, stretch=1)

        # ── Start button ─────────────────────────────────────────────────
        self._btn_start = QPushButton("⬡  ENCODE IMAGES")
        self._btn_start.setFixedHeight(38)
        self._btn_start.setEnabled(False)
        self._btn_start.setStyleSheet(self._btn_style("#7730aa", "#9950dd", large=True))
        self._btn_start.clicked.connect(self._start)
        root.addWidget(self._btn_start)

        # ── Encoding progress bar (hidden until encoding starts) ──────────
        self._enc_bar = QProgressBar()
        self._enc_bar.setRange(0, 100)
        self._enc_bar.setValue(0)
        self._enc_bar.setTextVisible(True)
        self._enc_bar.setFormat("Encoding  %v / %m  images…")
        self._enc_bar.setFixedHeight(18)
        self._enc_bar.setVisible(False)
        self._enc_bar.setStyleSheet(f"""
            QProgressBar {{
                background: #160f28;
                border: 1px solid #2a1e4a;
                border-radius: 4px;
                color: #9977cc;
                font: 8px '{MONO}';
                text-align: center;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #5530aa, stop:1 #9950dd);
                border-radius: 3px;
            }}
        """)
        root.addWidget(self._enc_bar)

    @staticmethod
    def _btn_style(c1, c2, large=False):
        fs = 10 if large else 9
        return f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 {c2},stop:1 {c1});
                color: #ddd8ff;
                border: 1px solid {c2};
                border-radius: 5px;
                padding: 4px 14px;
                font: {'bold ' if large else ''}{fs}px '{MONO}';
                letter-spacing: {'2' if large else '1'}px;
            }}
            QPushButton:hover {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 {c2},stop:1 {c2}); }}
            QPushButton:disabled {{ background: #1e1830; color: #443355; border-color: #2a2040; }}
        """

    # ── Internal helpers ──────────────────────────────────────────────────

    def _refresh_start_btn(self):
        ready = self._hub is not None and len(self._paths) >= 5
        self._btn_start.setEnabled(ready)

    # ── Slots ─────────────────────────────────────────────────────────────

    def _upload(self):
        """Open a fast file dialog (no macOS thumbnail lag) and load images."""
        # Start in the folder of the last loaded image, or ~/Pictures as fallback
        if self._paths:
            start_dir = str(Path(self._paths[-1]).parent)
        else:
            pictures = Path.home() / "Pictures"
            start_dir = str(pictures if pictures.exists() else Path.home())

        # DontUseNativeDialog skips macOS thumbnail generation → no lag.
        # ReadOnly saves an extra permission check per file.
        dialog = QFileDialog(self, "Select Images", start_dir,
                             "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tiff)")
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dialog.setOption(QFileDialog.Option.ReadOnly, True)
        if not dialog.exec():
            return
        self._load_files(dialog.selectedFiles())

    def _load_files(self, files: list):
        """
        Shared loader used by both the file dialog (_upload) and Finder drops.
        Loads up to MAX_IMAGES total, logs failures, and updates the hint label.
        """
        failed = []
        for f in files[:MAX_IMAGES - len(self._paths)]:
            try:
                img = PILImage.open(f).convert("RGB")
                self._pil_imgs.append(img)
                self._paths.append(f)
                self._grid.add_card(img, f)
            except Exception as exc:
                _log.warning("Failed to open %s: %s", os.path.basename(f), exc)
                failed.append(os.path.basename(f))

        self._refresh_start_btn()

        msg = (f"{len(self._paths)} images loaded. "
               "Drag to reorder — top = most loved.")
        if failed:
            msg += f"  ⚠ Failed to load: {', '.join(failed)}"
        self._hint.setText(msg)

    def _clear(self):
        if self._enc_worker is not None:
            try:
                self._enc_worker.finished.disconnect(self._on_encode_done)
            except TypeError:
                pass

        self._enc_bar.setVisible(False)
        self._enc_bar.setValue(0)
        self._btn_start.setText("⬡  BEGIN OPTIMIZATION")

        self._paths    = []
        self._pil_imgs = []
        self._grid.clear_cards()
        self._refresh_start_btn()
        self._hint.setText(
            "Upload images or drag them in from Finder. "
            "Drag to reorder by preference. Top = most loved.")

    def _start(self):
        ordered = self._grid.get_ordered_data()
        if len(ordered) < 5 or self._hub is None:
            return

        n = len(ordered)
        self._btn_start.setEnabled(False)
        self._btn_start.setText("⏳  Encoding…")
        self._enc_bar.setRange(0, n)
        self._enc_bar.setValue(0)
        self._enc_bar.setFormat(f"Encoding  %v / {n}  images…")
        self._enc_bar.setVisible(True)

        self._enc_worker = _EncoderWorker(self._hub, ordered)
        self._enc_thread = QThread(self)
        self._enc_worker.moveToThread(self._enc_thread)

        self._enc_thread.started.connect(self._enc_worker.run)
        self._enc_worker.progress.connect(self._on_encode_progress)
        self._enc_worker.finished.connect(self._on_encode_done)

        self._enc_worker.finished.connect(self._enc_thread.quit)
        self._enc_thread.finished.connect(self._enc_worker.deleteLater)
        self._enc_thread.finished.connect(self._enc_thread.deleteLater)

        self._enc_thread.start()

    def _on_encode_progress(self, done: int, total: int):
        self._enc_bar.setValue(done)
        self._btn_start.setText(f"⏳  Encoding {done} / {total}…")

    def _on_encode_done(self, clips: np.ndarray,
                        mobilenets: np.ndarray,
                        scores: np.ndarray):
        self._enc_bar.setVisible(False)
        self._enc_bar.setValue(0)
        self._btn_start.setText("✓  ENCODED — run again?")
        self._refresh_start_btn()

        self._enc_worker = None
        self._enc_thread = None

        self.calibration_ready.emit(clips, mobilenets, scores)