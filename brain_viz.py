"""
brain_viz.py  —  Yeo-7 Network Activation Panel
────────────────────────────────────────────────
Replaces the abstract dorsal-brain dot map with a clean, readable panel:

  • One horizontal bar per Yeo network, colour-matched to the network
  • Full network name + abbreviation label
  • Activation value shown as text on the right
  • "Aesthetic relevance" markers (★) on the two networks that drive the score
    (Limbic + DMN, per Vessel et al.)
  • Composite aesthetic score bar at the bottom with a sparkline history
  • All values animate smoothly via exponential interpolation

Nothing here requires the model to be loaded — it works in demo mode too.
"""

import math
import collections
import numpy as np

from PyQt6.QtCore    import QTimer, QRectF, Qt, QPointF
from PyQt6.QtGui     import (QPainter, QColor, QPen, QBrush,
                             QLinearGradient, QFont, QPainterPath)
from PyQt6.QtWidgets import QWidget

MONO = "Menlo"

# ── Network definitions (Yeo 7) ───────────────────────────────────────────────
NETWORKS = {
    1: dict(name="Visual",           short="VIS", color=(40,  110, 255), aesthetic=False),
    2: dict(name="Somatomotor",      short="SMN", color=(0,   195, 220), aesthetic=False),
    3: dict(name="Dorsal Attention", short="DAN", color=(20,  210,  80), aesthetic=False),
    4: dict(name="Ventral Attention",short="VAN", color=(170,  20, 255), aesthetic=False),
    5: dict(name="Limbic",           short="LIM", color=(255, 210,  10), aesthetic=True),
    6: dict(name="Frontoparietal",   short="FPN", color=(255, 100,  10), aesthetic=False),
    7: dict(name="Default Mode",     short="DMN", color=(255,  25,  60), aesthetic=True),
}

HISTORY_LEN = 80   # number of score samples kept for the sparkline


class BrainWidget(QWidget):
    """
    Readable network activation panel.

    Public API (same as the old BrainWidget so nothing else needs changing):
        update_activations(bold, yeo_assign)
        clear_activations()
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(340, 380)

        self._activations = {nid: 0.35 for nid in NETWORKS}
        self._target_act  = {nid: 0.35 for nid in NETWORKS}
        self._score_hist  = collections.deque(maxlen=HISTORY_LEN)
        self._t           = 0.0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(40)   # 25 fps

    # ── Public API ────────────────────────────────────────────────────────

    def update_activations(self, bold: np.ndarray, yeo_assign: np.ndarray):
        for nid in NETWORKS:
            mask = yeo_assign == nid
            if mask.sum() == 0:
                continue
            mean_val = float(np.mean(bold[mask]))
            act = 1.0 / (1.0 + math.exp(-mean_val * 0.8))
            self._target_act[nid] = act

        # Compute composite aesthetic score and push to history
        lim = self._activations[5]
        dmn = self._activations[7]
        vis = self._activations[1]
        score = 0.50 * lim + 0.35 * dmn + 0.15 * vis
        self._score_hist.append(score)

    def update_normalized(self, activations: dict):
        """
        Accept a {nid: value_in_0_1} dict and set those as the target
        activations directly, bypassing the sigmoid transform.
        Called by MainWindow._on_bold_update() with hub.normalized_yeo_means()
        so bars reflect real network differentiation instead of sigmoid≈0.5.
        """
        for nid, val in activations.items():
            if nid in self._target_act:
                self._target_act[nid] = float(np.clip(val, 0.0, 1.0))

        # Push a score history sample using the already-normalised values
        lim   = activations.get(5, self._activations[5])
        dmn   = activations.get(7, self._activations[7])
        vis   = activations.get(1, self._activations[1])
        score = 0.50 * lim + 0.35 * dmn + 0.15 * vis
        self._score_hist.append(score)

    def clear_activations(self):
        self._target_act = {nid: 0.15 for nid in NETWORKS}
        self._score_hist.clear()

    # ── Animation tick ────────────────────────────────────────────────────

    def _tick(self):
        self._t += 0.04
        for nid in NETWORKS:
            self._activations[nid] += (
                self._target_act[nid] - self._activations[nid]) * 0.10
        self.update()

    # ── Paint ─────────────────────────────────────────────────────────────

    def paintEvent(self, _):
        W, H = self.width(), self.height()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        bg = QLinearGradient(0, 0, 0, H)
        bg.setColorAt(0, QColor(6,  6, 14))
        bg.setColorAt(1, QColor(10, 8, 22))
        p.fillRect(0, 0, W, H, QBrush(bg))

        self._draw_title(p, W)

        # Layout: title takes 28px, sparkline at bottom 70px, rest for bars
        title_h   = 28
        spark_h   = 70
        bar_area_h = H - title_h - spark_h - 12
        bar_h      = bar_area_h / len(NETWORKS)

        for i, (nid, ndata) in enumerate(NETWORKS.items()):
            y = title_h + i * bar_h
            self._draw_network_row(p, nid, ndata, 0, y, W, bar_h)

        self._draw_sparkline(p, W, H - spark_h, W, spark_h)
        p.end()

    def _draw_title(self, p, W):
        font = QFont(MONO, 9)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2.5)
        p.setFont(font)
        p.setPen(QColor(120, 100, 200, 180))
        p.drawText(QRectF(0, 4, W, 22),
                   Qt.AlignmentFlag.AlignHCenter, "NETWORK ACTIVATIONS")

    def _draw_network_row(self, p, nid, ndata, rx, ry, rw, rh):
        """Draw one network row: label | bar | value."""
        act    = self._activations[nid]
        r, g, b = ndata["color"]
        pad    = 8
        label_w = 130    # fixed width for label column
        val_w   = 40     # fixed width for value on right
        bar_x   = rx + pad + label_w
        bar_w   = rw - pad * 2 - label_w - val_w
        bar_y   = ry + rh * 0.28
        bar_ht  = rh * 0.44
        cx      = rx + rw / 2
        cy      = ry + rh / 2

        # ── Subtle row separator ──────────────────────────────────────────
        p.setPen(QPen(QColor(30, 25, 55, 120), 0.5))
        p.drawLine(QPointF(rx + pad, ry + rh - 1),
                   QPointF(rx + rw - pad, ry + rh - 1))

        # ── Network name label ────────────────────────────────────────────
        alpha_label = int(100 + act * 140)
        font_name = QFont(MONO, 8)
        p.setFont(font_name)
        p.setPen(QColor(r, g, b, alpha_label))
        name_rect = QRectF(rx + pad, ry, label_w - 8, rh)
        p.drawText(name_rect, Qt.AlignmentFlag.AlignVCenter |
                   Qt.AlignmentFlag.AlignLeft, ndata["name"])

        # Aesthetic star marker
        if ndata["aesthetic"]:
            font_star = QFont(MONO, 7)
            p.setFont(font_star)
            p.setPen(QColor(r, g, b, 200))
            p.drawText(
                QRectF(rx + pad + label_w - 14, ry, 12, rh),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                "★"
            )

        # ── Bar track ────────────────────────────────────────────────────
        track_rect = QRectF(bar_x, bar_y, bar_w, bar_ht)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(25, 20, 50, 180)))
        p.drawRoundedRect(track_rect, 3, 3)

        # ── Bar fill ──────────────────────────────────────────────────────
        fill_w = max(4, bar_w * act)
        fill_rect = QRectF(bar_x, bar_y, fill_w, bar_ht)

        grad = QLinearGradient(bar_x, 0, bar_x + bar_w, 0)
        grad.setColorAt(0.0, QColor(r, g, b, int(act * 200 + 30)))
        grad.setColorAt(0.6, QColor(
            min(255, r + 40), min(255, g + 40), min(255, b + 40),
            int(act * 220 + 20)))
        grad.setColorAt(1.0, QColor(
            min(255, r + 80), min(255, g + 80), min(255, b + 80),
            int(act * 180)))
        p.setBrush(QBrush(grad))
        p.drawRoundedRect(fill_rect, 3, 3)

        # Leading edge glow
        if act > 0.1:
            glow = QLinearGradient(bar_x + fill_w - 10, 0,
                                   bar_x + fill_w + 4, 0)
            glow.setColorAt(0.0, QColor(r, g, b, 0))
            glow.setColorAt(1.0, QColor(min(255, r + 120),
                                         min(255, g + 120),
                                         min(255, b + 120), 180))
            p.setBrush(QBrush(glow))
            p.drawRoundedRect(
                QRectF(bar_x + fill_w - 10, bar_y, 14, bar_ht), 3, 3)

        # ── Activation value text ─────────────────────────────────────────
        font_val = QFont(MONO, 8)
        p.setFont(font_val)
        p.setPen(QColor(r, g, b, int(120 + act * 120)))
        val_rect = QRectF(bar_x + bar_w + 4, ry, val_w - 4, rh)
        p.drawText(val_rect,
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                   f"{act:.2f}")

    def _draw_sparkline(self, p, W, sy, sw, sh):
        """Draw composite aesthetic score + sparkline history at the bottom."""
        pad = 10

        # ── Section label ─────────────────────────────────────────────────
        font = QFont(MONO, 8)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.5)
        p.setFont(font)
        p.setPen(QColor(100, 80, 180, 160))
        p.drawText(QRectF(pad, sy + 4, sw - pad * 2, 16),
                   Qt.AlignmentFlag.AlignLeft, "AESTHETIC SCORE")

        # Compute current score
        lim   = self._activations[5]
        dmn   = self._activations[7]
        vis   = self._activations[1]
        score = 0.50 * lim + 0.35 * dmn + 0.15 * vis

        # ── Score text ────────────────────────────────────────────────────
        font_big = QFont(MONO, 14)
        font_big.setBold(True)
        p.setFont(font_big)
        # Colour shifts warm as score rises
        sr, sg, sb = (
            int(80  + score * 175),
            int(40  + score * 60),
            int(200 - score * 140),
        )
        p.setPen(QColor(sr, sg, sb, 220))
        p.drawText(QRectF(pad, sy + 18, 90, 26),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                   f"{score:+.4f}")

        # ── Contribution breakdown (small text) ──────────────────────────
        font_s = QFont(MONO, 7)
        p.setFont(font_s)
        breakdown_x = pad + 94
        items = [
            (f"LIM×0.50 = {0.50*lim:.3f}", NETWORKS[5]["color"]),
            (f"DMN×0.35 = {0.35*dmn:.3f}", NETWORKS[7]["color"]),
            (f"VIS×0.15 = {0.15*vis:.3f}", NETWORKS[1]["color"]),
        ]
        for j, (txt, col) in enumerate(items):
            p.setPen(QColor(*col, 160))
            p.drawText(
                QRectF(breakdown_x, sy + 18 + j * 11, W - breakdown_x - pad, 12),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                txt
            )

        # ── Sparkline ─────────────────────────────────────────────────────
        if len(self._score_hist) < 2:
            return

        hist   = list(self._score_hist)
        lo, hi = min(hist), max(hist)
        span   = max(hi - lo, 0.01)

        spark_x = pad
        spark_y = sy + 48
        spark_w = W - pad * 2
        spark_h = sh - 52

        # Track
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(20, 16, 40, 120)))
        p.drawRoundedRect(
            QRectF(spark_x, spark_y, spark_w, spark_h), 3, 3)

        # Line
        path = QPainterPath()
        n = len(hist)
        for i, v in enumerate(hist):
            nx = spark_x + (i / (n - 1)) * spark_w
            ny = spark_y + spark_h - ((v - lo) / span) * (spark_h - 4) - 2
            if i == 0:
                path.moveTo(nx, ny)
            else:
                path.lineTo(nx, ny)

        grad_line = QLinearGradient(spark_x, 0, spark_x + spark_w, 0)
        grad_line.setColorAt(0.0, QColor(80,  40, 200, 160))
        grad_line.setColorAt(0.5, QColor(200, 80,  40, 200))
        grad_line.setColorAt(1.0, QColor(255, 200, 20, 220))
        pen = QPen(QBrush(grad_line), 1.5)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)

        # Current value dot at end
        last_x = spark_x + spark_w
        last_y = spark_y + spark_h - ((hist[-1] - lo) / span) * (spark_h - 4) - 2
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(255, 200, 20, 220)))
        p.drawEllipse(QRectF(last_x - 3, last_y - 3, 6, 6))