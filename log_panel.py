"""
log_panel.py
Thread-safe real-time application log panel.

Usage
─────
    from log_panel import install_logging, LogPanel

    # Call once, after QApplication is created, before the window is shown.
    install_logging()

    # LogPanel() auto-subscribes; add it to any layout.
    panel = LogPanel()

All print() calls and Python logging from any thread will appear in every
LogPanel that has been instantiated.
"""

import sys
import html
import logging

from PyQt6.QtCore    import QObject, Qt, pyqtSignal
from PyQt6.QtGui     import QFont, QTextCursor
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                              QTextEdit, QPushButton, QLabel, QCheckBox)


# ── Thread-safe signal bridge ─────────────────────────────────────────────────
#
# _LogSignaler is created on the main thread.  Any background thread can safely
# call _signaler.log_line.emit() — Qt automatically queues the delivery to the
# main-thread slots that are connected.

class _LogSignaler(QObject):
    """Module-level singleton; safe to emit from any thread."""
    log_line = pyqtSignal(str, str)   # (message, level_name)


_signaler = _LogSignaler()


# ── Python logging → signal ────────────────────────────────────────────────────

class _QtLogHandler(logging.Handler):
    """Routes Python logging records to the signal bridge."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            _signaler.log_line.emit(msg, record.levelname)
        except Exception:   # never let a log handler crash the app
            pass


# ── stdout / stderr redirect ──────────────────────────────────────────────────

class _StreamRedirect:
    """
    Wraps an original stream (stdout or stderr).
    Each complete line (terminated by \\n) is forwarded to _signaler.
    Partial writes are buffered until a newline arrives.
    The original stream is always written to as well, so terminal output is
    preserved.
    """

    def __init__(self, original, level: str = "INFO") -> None:
        self._orig  = original
        self._level = level
        self._buf   = ""

    def write(self, text: str) -> None:
        self._orig.write(text)
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip()
            if line:
                _signaler.log_line.emit(line, self._level)

    def flush(self) -> None:
        self._orig.flush()

    def fileno(self) -> int:
        return self._orig.fileno()

    def isatty(self) -> bool:
        return False


# ── Public installer ──────────────────────────────────────────────────────────

def install_logging(level: int = logging.DEBUG) -> None:
    """
    Install the Qt log handler and redirect sys.stdout / sys.stderr.
    Call exactly once, after QApplication is created.
    """
    handler = _QtLogHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    # Avoid double-installing if called more than once
    if not any(isinstance(h, _QtLogHandler) for h in logging.root.handlers):
        logging.root.addHandler(handler)
        logging.root.setLevel(level)

    if not isinstance(sys.stdout, _StreamRedirect):
        sys.stdout = _StreamRedirect(sys.stdout, "INFO")
    if not isinstance(sys.stderr, _StreamRedirect):
        sys.stderr = _StreamRedirect(sys.stderr, "ERROR")


# ── LogPanel widget ───────────────────────────────────────────────────────────

class LogPanel(QWidget):
    """
    Dark-styled scrolling log window.

    Auto-subscribes to _signaler on construction; immediately starts showing
    all log output from any thread after install_logging() has been called.
    """

    # Rolling window — Qt's setMaximumBlockCount() handles pruning atomically.
    MAX_LINES = 4000

    _LEVEL_CSS: dict[str, str] = {
        "DEBUG":    "color:#4a5570;",
        "INFO":     "color:#8899b4;",
        "WARNING":  "color:#c8a040;",
        "ERROR":    "color:#cc4444;",
        "CRITICAL": "color:#ff2244;font-weight:bold;",
    }

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(80)
        self._auto_scroll = True
        self._build_ui()

        # QueuedConnection ensures delivery on the main thread even when
        # the signal is emitted from a background thread.
        _signaler.log_line.connect(
            self._append, Qt.ConnectionType.QueuedConnection)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 2, 4, 4)
        root.setSpacing(3)

        # ── header bar ───────────────────────────────────────────────────────
        hdr = QHBoxLayout()

        title = QLabel("APPLICATION LOG")
        title.setStyleSheet(
            "color:#443388; font:bold 9px 'Consolas'; letter-spacing:2px;")
        hdr.addWidget(title)
        hdr.addStretch()

        self._chk = QCheckBox("auto-scroll")
        self._chk.setChecked(True)
        self._chk.setStyleSheet(
            "color:#443366; font:8px 'Consolas'; spacing:3px;")
        self._chk.toggled.connect(self._on_autoscroll_toggled)
        hdr.addWidget(self._chk)

        btn_clear = QPushButton("Clear")
        btn_clear.setFixedSize(48, 18)
        btn_clear.setStyleSheet("""
            QPushButton {
                background:#0e0c1e; color:#443366;
                border:1px solid #1c1838; border-radius:3px;
                font:8px 'Consolas';
            }
            QPushButton:hover { background:#1c1838; color:#665599; }
        """)
        btn_clear.clicked.connect(self._clear)
        hdr.addWidget(btn_clear)

        root.addLayout(hdr)

        # ── text area ────────────────────────────────────────────────────────
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setFont(QFont("Consolas", 8))
        self._text.document().setMaximumBlockCount(self.MAX_LINES)
        self._text.setStyleSheet("""
            QTextEdit {
                background:#04040c;
                color:#8899b4;
                border:1px solid #10091e;
                border-radius:4px;
                padding:4px;
            }
            QScrollBar:vertical {
                background:#070512; width:8px; border-radius:4px;
            }
            QScrollBar::handle:vertical {
                background:#2c2248; border-radius:4px; min-height:18px;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical { height:0px; }
        """)
        root.addWidget(self._text)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_autoscroll_toggled(self, checked: bool) -> None:
        self._auto_scroll = checked

    def _clear(self) -> None:
        self._text.clear()

    def _append(self, text: str, level: str) -> None:
        css = self._LEVEL_CSS.get(level, "color:#8899b4;")
        esc = html.escape(text)
        self._text.append(
            f'<span style="font-family:Consolas;font-size:8pt;{css}">'
            f'{esc}</span>'
        )
        if self._auto_scroll:
            sb = self._text.verticalScrollBar()
            sb.setValue(sb.maximum())

    # ── Public helpers ────────────────────────────────────────────────────────

    def post(self, text: str, level: str = "INFO") -> None:
        """Programmatically inject a log line (useful for app-level events)."""
        _signaler.log_line.emit(text, level)