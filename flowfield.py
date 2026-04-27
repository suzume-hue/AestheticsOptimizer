"""
flowfield.py  —  Starry Night-style Turbulent Flow Field Renderer
──────────────────────────────────────────────────────────────────
Replaces the geometric band art with a true flow field visualization:

  • Turbulent vector field built from superposed sine waves (no external
    noise library needed).  `angle_spread` controls turbulence intensity;
    `n_layers` controls streamline density.
  • Every Yeo network colour is visible simultaneously — colours are
    distributed across streamlines weighted by network activation.
  • Radial glow spots (like Starry Night's stars) appear at flow
    convergence points; their colour and brightness track the two
    aesthetic networks (Limbic=gold, DMN=red).
  • Dark background lets saturated streamline colours pop.

Module-level symbols required by main.py
─────────────────────────────────────────
    PARAM_NAMES    list[str]
    PARAM_BOUNDS   dict[str, tuple]
    DEFAULT_PARAMS dict[str, float]

    render_offscreen(params, seed, W, H, n_steps) → PIL.Image.Image

FlowFieldWidget public API  (unchanged)
──────────────────────────
    update_activations(bold, yeo_assign)
    update_bold_state(activations)
    clear_activations()
    set_params(params)
    set_score(score, iteration)
    set_optimizing(on: bool)
    new_seed()
"""

import math
import colorsys
import numpy as np

from PIL import Image as _PIL_Image

from PyQt6.QtCore    import Qt, QRectF, QPointF
from PyQt6.QtGui     import (QPainter, QColor, QPen, QBrush,
                             QLinearGradient, QRadialGradient, QFont,
                             QPainterPath, QPixmap)
from PyQt6.QtWidgets import QWidget


MONO = "Menlo"

NETWORKS = {
    1: dict(name="Visual",            short="VIS", color=(40,  110, 255)),
    2: dict(name="Somatomotor",       short="SMN", color=(0,   195, 220)),
    3: dict(name="Dorsal Attention",  short="DAN", color=(20,  210,  80)),
    4: dict(name="Ventral Attention", short="VAN", color=(170,  20, 255)),
    5: dict(name="Limbic",            short="LIM", color=(255, 210,  10)),
    6: dict(name="Frontoparietal",    short="FPN", color=(255, 100,  10)),
    7: dict(name="Default Mode",      short="DMN", color=(255,  25,  60)),
}


# ── Optimisable parameter space ───────────────────────────────────────────────
#
#   hue_rot       — rotate every network colour around the hue wheel (0→1 = 360°)
#   saturation    — colour saturation multiplier (dull ↔ vivid)
#   brightness    — overall luminance multiplier
#   n_layers      — streamline density: 2→sparse, 10→dense Starry Night
#   band_alpha    — stroke opacity  [0.10–0.90]
#   angle_spread  — flow turbulence: 0→parallel streams, 1→fully turbulent
#   band_width    — stroke pixel width  [0.30–1.50 → maps to ~0.5–4 px]
#   color_warmth  — blend all colours toward warm orange/red tones  [0–1]

PARAM_NAMES: list[str] = [
    "hue_rot",
    "saturation",
    "brightness",
    "n_layers",
    "band_alpha",
    "angle_spread",
    "band_width",
    "color_warmth",
]

DEFAULT_PARAMS: dict[str, float] = {
    "hue_rot":      0.00,
    "saturation":   1.10,
    "brightness":   1.00,
    "n_layers":     5.0,
    "band_alpha":   0.62,
    "angle_spread": 0.40,
    "band_width":   0.88,
    "color_warmth": 0.00,
    "seed":         42.0,   # always present; mutated during optimiser search
}

PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "hue_rot":      (0.00, 1.00),
    "saturation":   (0.30, 1.80),
    "brightness":   (0.40, 1.60),
    "n_layers":     (2.0,  10.0),
    "band_alpha":   (0.10, 0.90),
    "angle_spread": (0.00, 1.00),
    "band_width":   (0.30, 1.50),
    "color_warmth": (0.00, 1.00),
}


# ── Colour helpers ────────────────────────────────────────────────────────────

def _apply_params_to_color(r: int, g: int, b: int,
                            params: dict) -> tuple:
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    h = (h + params.get("hue_rot", 0.0)) % 1.0
    s = min(1.0, s * params.get("saturation", 1.0))
    v = min(1.0, v * params.get("brightness", 1.0))
    nr, ng, nb = colorsys.hsv_to_rgb(h, s, v)
    warmth = params.get("color_warmth", 0.0)
    if warmth > 0:
        wh = 0.08
        wr, wg, wb = colorsys.hsv_to_rgb(
            (h * (1 - warmth) + wh * warmth) % 1.0,
            min(1.0, s + warmth * (0.85 - s)),
            min(1.0, v),
        )
        nr += warmth * (wr - nr)
        ng += warmth * (wg - ng)
        nb += warmth * (wb - nb)
    return int(nr * 255), int(ng * 255), int(nb * 255)


def _sigmoid(x: float, scale: float = 0.8) -> float:
    return 1.0 / (1.0 + math.exp(-x * scale))


# ── Flow field helpers ────────────────────────────────────────────────────────

def _build_angle_field(rng: np.random.RandomState,
                        angle_spread: float,
                        grid_n: int = 48) -> np.ndarray:
    """
    Build a (grid_n, grid_n) angle field using superposed sine waves.
    Uses 5 harmonics at different scales and orientations so the field
    is smooth but genuinely turbulent — no external noise library needed.
    Returns angles in radians.
    """
    n_harmonics = 5
    # Spatial frequencies for x and y independently
    fx = rng.uniform(0.6, 2.8, n_harmonics)
    fy = rng.uniform(0.6, 2.8, n_harmonics)
    phases = rng.uniform(0.0, 2 * np.pi, n_harmonics)
    # Amplitude decays with frequency (smooth large-scale flow + fine detail)
    amps = np.exp(-np.arange(n_harmonics) * 0.35)
    amps = amps / amps.sum()

    xs = np.linspace(0.0, 1.0, grid_n)
    ys = np.linspace(0.0, 1.0, grid_n)
    XX, YY = np.meshgrid(xs, ys)   # (grid_n, grid_n)

    base_angle = rng.uniform(0.0, 2 * np.pi)
    field = np.full((grid_n, grid_n), base_angle, dtype=np.float64)

    for k in range(n_harmonics):
        field += (amps[k] * angle_spread * np.pi *
                  np.sin(fx[k] * XX * 2 * np.pi +
                         fy[k] * YY * 2 * np.pi + phases[k]))
    return field


def _sample_angle(field: np.ndarray, xn: float, yn: float) -> float:
    """
    Bilinear interpolation of the angle field at normalised position (xn, yn).
    xn, yn ∈ [0, 1].  Clamps gracefully at edges.
    """
    n = field.shape[0]
    xi = np.clip(xn * (n - 1), 0.0, n - 1.0)
    yi = np.clip(yn * (n - 1), 0.0, n - 1.0)
    xi0, yi0 = int(xi), int(yi)
    xi1 = min(xi0 + 1, n - 1)
    yi1 = min(yi0 + 1, n - 1)
    fx_  = xi - xi0
    fy_  = yi - yi0
    return float(
        field[yi0, xi0] * (1 - fx_) * (1 - fy_) +
        field[yi0, xi1] * fx_       * (1 - fy_) +
        field[yi1, xi0] * (1 - fx_) * fy_       +
        field[yi1, xi1] * fx_       * fy_
    )


# ── Core renderer ─────────────────────────────────────────────────────────────

def _render_card(painter: QPainter, W: int, H: int,
                 params: dict, activations: dict) -> None:
    """
    Starry Night-style turbulent flow field art.

    Each call:
      1. Builds a turbulent vector angle field from sine-wave superposition.
      2. Seeds N streamlines, traces each one through the field, draws as
         a smooth QPainterPath curve.
      3. Paints radial glow spots (stars/moons) at random positions.
      4. Overlays a subtle vignette and card border.

    All colours come from the 7 Yeo networks; their visual weight is
    proportional to their activation value, so aesthetically-activated
    networks (Limbic gold, DMN red) dominate the palette.
    """
    seed = int(params.get("seed", 42)) & 0xFFFF_FFFF
    rng  = np.random.RandomState(seed)

    # ── Parameters ────────────────────────────────────────────────────────
    hue_rot      = params.get("hue_rot",      0.00)
    saturation   = params.get("saturation",   1.10)
    brightness   = params.get("brightness",   1.00)
    n_layers_p   = params.get("n_layers",     5.0)
    band_alpha   = params.get("band_alpha",   0.62)
    angle_spread = params.get("angle_spread", 0.40)
    band_width   = params.get("band_width",   0.88)
    color_warmth = params.get("color_warmth", 0.00)

    # ── Card geometry ─────────────────────────────────────────────────────
    RATIO  = 0.635
    pad    = min(W, H) * 0.05
    card_h = H - 2 * pad
    card_w = min(card_h * RATIO, W - 2 * pad)
    card_h = card_w / RATIO
    card_x = W / 2 - card_w / 2
    card_y = H / 2 - card_h / 2
    card_rect = QRectF(card_x, card_y, card_w, card_h)
    RADIUS    = card_w * 0.07

    # ── Colour palette (all 7 networks, sorted by activation) ─────────────
    sorted_nets = sorted(activations.items(), key=lambda kv: kv[1], reverse=True)
    palette = []   # list of (r, g, b, activation_weight)
    for nid, act in sorted_nets:
        r0, g0, b0 = NETWORKS[nid]["color"]
        r, g, b    = _apply_params_to_color(r0, g0, b0, params)
        palette.append((r, g, b, max(0.05, float(act))))

    # Dominant network colour for background toning
    pr, pg, pb, _ = palette[0]
    ph, ps, pv     = colorsys.rgb_to_hsv(pr / 255, pg / 255, pb / 255)

    # ── Widget background (deep space black) ──────────────────────────────
    painter.fillRect(0, 0, W, H, QColor(3, 3, 10))

    # ── Card drop shadow ──────────────────────────────────────────────────
    painter.setPen(Qt.PenStyle.NoPen)
    for i in range(10, 0, -1):
        a = max(0, 50 - i * 5)
        painter.setBrush(QBrush(QColor(0, 0, 0, a)))
        painter.drawRoundedRect(
            QRectF(card_x + i * 1.5, card_y + i * 2.0, card_w, card_h),
            RADIUS, RADIUS)

    # ── Card background — very dark, colour-tinted ────────────────────────
    bg_v  = max(0.04, pv * 0.12)
    bg_v2 = max(0.08, pv * 0.20)
    bg_r,  bg_g,  bg_b  = (int(x * 255) for x in colorsys.hsv_to_rgb(ph, min(1.0, ps * 0.7), bg_v))
    bg_r2, bg_g2, bg_b2 = (int(x * 255) for x in colorsys.hsv_to_rgb((ph + 0.05) % 1.0,
                                                                        min(1.0, ps * 0.5), bg_v2))
    bg = QLinearGradient(card_x, card_y, card_x + card_w * 0.4, card_y + card_h)
    bg.setColorAt(0.0, QColor(bg_r2, bg_g2, bg_b2))
    bg.setColorAt(1.0, QColor(bg_r,  bg_g,  bg_b))
    painter.setBrush(QBrush(bg))
    painter.drawRoundedRect(card_rect, RADIUS, RADIUS)

    # ── Clip all flow content to card shape ───────────────────────────────
    painter.save()
    clip = QPainterPath()
    clip.addRoundedRect(card_rect, RADIUS, RADIUS)
    painter.setClipPath(clip)

    # ── Build turbulent angle field ───────────────────────────────────────
    angle_field = _build_angle_field(rng, angle_spread, grid_n=48)

    # ── Streamline parameters ─────────────────────────────────────────────
    n_streams  = max(80, int(n_layers_p * 60))   # 80–600
    step_size  = min(card_w, card_h) * 0.018
    n_steps    = int(22 + angle_spread * 18)     # 22–40 steps
    stroke_w   = band_width * 1.8 + 0.4          # ~0.9–3.1 px

    # Pre-build activation-weighted probability array for colour sampling
    act_weights = np.array([p[3] for p in palette], dtype=np.float64)
    act_weights = act_weights / act_weights.sum()

    painter.setBrush(Qt.BrushStyle.NoBrush)

    for _ in range(n_streams):
        # Seed position — uniform over card interior
        sx = card_x + rng.uniform(0.05, 0.95) * card_w
        sy = card_y + rng.uniform(0.05, 0.95) * card_h

        # Choose network colour, weighted by activation
        ci = int(rng.choice(len(palette), p=act_weights))
        cr, cg, cb, cact = palette[ci]

        # Slight per-streamline hue jitter for richness
        hj, sj, vj = colorsys.rgb_to_hsv(cr / 255, cg / 255, cb / 255)
        hj = (hj + rng.uniform(-0.03, 0.03)) % 1.0
        vj = min(1.0, vj * rng.uniform(0.75, 1.25))
        sj = min(1.0, sj * rng.uniform(0.85, 1.15))
        cr, cg, cb = (int(x * 255) for x in colorsys.hsv_to_rgb(hj, sj, vj))

        # Alpha: proportional to activation, clamped
        alpha = int(band_alpha * (80 + 140 * cact))
        alpha = max(18, min(210, alpha))

        # Trace streamline through the vector field
        path = QPainterPath()
        px, py = sx, sy
        path.moveTo(px, py)
        alive = True

        for step in range(n_steps):
            xn = (px - card_x) / card_w
            yn = (py - card_y) / card_h
            if not (0.0 <= xn <= 1.0 and 0.0 <= yn <= 1.0):
                alive = False
                break
            angle = _sample_angle(angle_field, xn, yn)
            # Tiny per-step jitter keeps lines from being mechanical
            angle += rng.uniform(-0.04, 0.04) * angle_spread
            dx = math.cos(angle) * step_size
            dy = math.sin(angle) * step_size
            px += dx
            py += dy
            path.lineTo(px, py)

        if path.elementCount() >= 3:
            pen = QPen(QColor(cr, cg, cb, alpha), stroke_w)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawPath(path)

    # ── Radial glow spots (stars / focal vortices) ────────────────────────
    n_glows = max(4, int(3 + n_layers_p * 1.2))
    for _ in range(n_glows):
        gx = card_x + rng.uniform(0.08, 0.92) * card_w
        gy = card_y + rng.uniform(0.08, 0.92) * card_h
        gr = min(card_w, card_h) * rng.uniform(0.035, 0.11)

        gi = int(rng.choice(len(palette), p=act_weights))
        gr_r, gr_g, gr_b, gr_act = palette[gi]

        inner_a = int(min(230, 120 + 160 * gr_act))
        mid_a   = int(min(80,  30  + 70  * gr_act))

        glow = QRadialGradient(gx, gy, gr)
        glow.setColorAt(0.00, QColor(min(255, gr_r + 80),
                                      min(255, gr_g + 80),
                                      min(255, gr_b + 80), inner_a))
        glow.setColorAt(0.35, QColor(gr_r, gr_g, gr_b, mid_a))
        glow.setColorAt(1.00, QColor(gr_r, gr_g, gr_b, 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(glow))
        painter.drawEllipse(QPointF(gx, gy), gr, gr)

    # ── Vignette (dark edges → focus on centre) ───────────────────────────
    cx = card_x + card_w / 2
    cy = card_y + card_h / 2
    vig_r = math.hypot(card_w, card_h) * 0.65
    vig = QRadialGradient(cx, cy, vig_r)
    vig.setColorAt(0.0, QColor(0, 0, 0, 0))
    vig.setColorAt(1.0, QColor(0, 0, 0, 130))
    painter.setBrush(QBrush(vig))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRoundedRect(card_rect, RADIUS, RADIUS)

    # ── Subtle top-half sheen ─────────────────────────────────────────────
    sheen = QLinearGradient(card_x, card_y, card_x, card_y + card_h * 0.45)
    sheen.setColorAt(0.0, QColor(255, 255, 255, 14))
    sheen.setColorAt(1.0, QColor(255, 255, 255, 0))
    painter.setBrush(QBrush(sheen))
    painter.drawRoundedRect(card_rect, RADIUS, RADIUS)

    painter.restore()

    # ── Card border ───────────────────────────────────────────────────────
    painter.setPen(QPen(QColor(255, 255, 255, 35), 0.8))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRoundedRect(card_rect, RADIUS, RADIUS)


# ── Offscreen renderer ────────────────────────────────────────────────────────

def render_offscreen(params: dict,
                     seed,
                     W: int = 256,
                     H: int = 256,
                     n_steps: int = 160) -> "_PIL_Image.Image":
    """
    Render the card art with *params* to a PIL Image.

    Called by OptimizerWorker._evaluate() for every candidate solution.
    seed and n_steps are kept for API compatibility (seed is read from
    params["seed"]; n_steps is unused in a static render).
    """
    neutral = {nid: 0.55 for nid in NETWORKS}

    pixmap  = QPixmap(W, H)
    pixmap.fill(QColor(5, 4, 12))

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    _render_card(painter, W, H, params, neutral)
    painter.end()

    qimage = pixmap.toImage().convertToFormat(
        pixmap.toImage().Format.Format_RGBA8888)
    ptr = qimage.bits()
    ptr.setsize(H * W * 4)
    arr = np.frombuffer(ptr, dtype=np.uint8).copy().reshape((H, W, 4))
    return _PIL_Image.fromarray(arr, "RGBA").convert("RGB")


# ── Widget ────────────────────────────────────────────────────────────────────

class FlowFieldWidget(QWidget):
    """
    Interactive single-card geometric art panel.

    State axes:
      _activations  — {nid: float in [0, 1]}  (BOLD predictions)
      _params       — visual parameter dict    (optimiser output)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(280, 440)
        self._activations: dict[int, float] = {nid: 0.30 for nid in NETWORKS}
        self._params:      dict             = dict(DEFAULT_PARAMS)
        self._score:       float            = 0.0
        self._iteration:   int              = 0
        self._optimizing:  bool             = False

    # ── Public API ────────────────────────────────────────────────────────

    def update_activations(self, bold: np.ndarray,
                           yeo_assign: np.ndarray) -> None:
        for nid in NETWORKS:
            mask = yeo_assign == nid
            if mask.sum() == 0:
                continue
            self._activations[nid] = _sigmoid(float(np.mean(bold[mask])))
        self.update()

    def update_bold_state(self, activations: dict[int, float]) -> None:
        self._activations.update(activations)
        self.update()

    def clear_activations(self) -> None:
        self._activations = {nid: 0.10 for nid in NETWORKS}
        self.update()

    def set_params(self, params: dict) -> None:
        self._params = dict(DEFAULT_PARAMS)
        self._params.update(params)
        self.update()

    def set_score(self, score: float, iteration: int) -> None:
        self._score     = score
        self._iteration = iteration
        self.update()

    def set_optimizing(self, on: bool) -> None:
        self._optimizing = on
        self.update()

    def new_seed(self) -> None:
        pass   # seed is managed externally by the optimiser

    # ── Paint ─────────────────────────────────────────────────────────────

    def paintEvent(self, _) -> None:
        W, H = self.width(), self.height()
        p    = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        _render_card(p, W, H, self._params, self._activations)
        self._draw_overlay(p, W)

        p.end()

    def _draw_overlay(self, p: QPainter, W: int) -> None:
        """Score / optimising badge — top-right corner."""
        if not self._optimizing and self._score == 0.0:
            return

        if self._optimizing:
            txt = f"OPTIMISING  iter {self._iteration}"
            col = QColor(160, 120, 255, 210)
        else:
            txt = f"BEST  {self._score:+.4f}  ({self._iteration} iter)"
            col = QColor(255, 200, 40, 210)

        PAD    = 8
        font   = QFont(MONO, 7)
        p.setFont(font)
        badge_w = len(txt) * 6 + PAD * 2
        rect    = QRectF(W - badge_w - PAD, PAD, badge_w, 18)

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(4, 3, 12, 190)))
        p.drawRoundedRect(rect, 3, 3)

        p.setPen(col)
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, txt)