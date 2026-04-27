"""
main.py
Neuroaesthetics Flow Optimizer — main entry point.

Layout
──────
┌─────────────────┬──────────────────────────┬──────────────────┐
│ CalibrationPanel│    FlowFieldWidget        │  BrainWidget     │
│  (380 px wide)  │    (fills remaining)      │  + controls      │
│                 │                           │  (360 px wide)   │
└─────────────────┴──────────────────────────┴──────────────────┘

Optimizer loop
──────────────
1. User uploads + ranks images → clicks ENCODE → calibration_ready(clips, scores)
2. User selects optimizer method from dropdown, clicks OPTIMISE
3. OptimizerWorker runs chosen backend in a QThread:
      for each candidate parameter vector:
          render_offscreen(params) → PIL image
          ModelHub.encode_image()  → (CLIP embedding, MobileNetV3 features)
          ModelHub.predict_bold()  → (1024,) BOLD signal
          ModelHub.aesthetic_score() → scalar
4. Signals push best params → FlowFieldWidget, BOLD → BrainWidget

Changes
───────
• Auto-install: missing packages (cma, optuna) are installed automatically
  on first run with a visible terminal message before any imports are attempted.

• Optimizer method selector: SidePanel now has a QComboBox to choose between
  CMA-ES, Bayesian (Optuna), Random Search, and Diffusion Model. Method is
  passed to OptimizerWorker at start time.

• Bayesian backend: OptimizerWorker._run_bayesian() uses Optuna's TPE
  sampler — a probabilistic model that learns which parameter regions are
  promising and focuses trials there. Needs far fewer evaluations than CMA-ES
  to find good regions when each evaluation is expensive (render + CLIP + BOLD).

• Diffusion backend (NEW): OptimizerWorker._run_diffusion() loads the trained
  conditional DDPM checkpoint (best_denoiser.pt from Notebook C), computes a
  personalised Yeo-7 target from the user's top-ranked calibration images, and
  runs DDIM sampling to generate informed candidate parameter vectors. Each
  candidate is then evaluated through the full render→CLIP→BOLD pipeline so
  scores are directly comparable to the other backends. Path is resolved from
  the DIFFUSION_CKPT env var, defaulting to diffusion_model/best_denoiser.pt
  next to main.py.

• Calibration button renamed: CalibrationPanel's action button now correctly
  says "ENCODE IMAGES" since it encodes the ranked images into embeddings,
  not start the optimizer.

• Font: all 'Consolas' replaced with 'Menlo' (macOS system monospace) to
  eliminate 300–800 ms font-alias scan at startup.

• All previous fixes retained (threading.Event stop, es.tell partial fix,
  signal throttling, 5s thread join timeout, seed mutation guard).
"""

# ── Preflight: install packages + download CLIP before GUI starts ─────────────
import subprocess
import sys
import importlib

# All packages the app needs. Heavy ones (torch, open_clip) are expected to
# already be installed — only the lightweight extras are auto-installed here.
_AUTO_INSTALL = {
    "cma":    "cma",
    "optuna": "optuna",
}

# All packages that must be present before the GUI can start. If any are
# missing the user gets a clear message pointing them to requirements.txt.
_REQUIRED_HEAVY = {
    "torch":       "pip install torch",
    "torchvision": "pip install torchvision",
    "open_clip":   "pip install open_clip_torch",
    "PyQt6":       "pip install PyQt6",
    "numpy":       "pip install numpy",
    "PIL":         "pip install Pillow",
}

def _ensure_packages():
    """Install lightweight missing packages silently."""
    for pip_name, import_name in _AUTO_INSTALL.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            print(f"[setup] Installing '{pip_name}'…", flush=True)
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pip_name],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            print(f"[setup] '{pip_name}' ready.", flush=True)

def _check_heavy_deps():
    """Warn clearly if a required heavy package is missing."""
    missing = []
    for mod, install_cmd in _REQUIRED_HEAVY.items():
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append((mod, install_cmd))
    if missing:
        print("\n[setup] ── Missing required packages ──────────────────────")
        for mod, cmd in missing:
            print(f"         {mod:20s}  →  {cmd}")
        print("[setup] Install them and re-run.\n", flush=True)
        sys.exit(1)

def _preflight_clip():
    """
    Check whether MobileCLIP2-S0 weights are already cached.
    If not, download them NOW in the terminal (with a visible progress bar)
    before the GUI starts — so the user can see exactly what is happening
    instead of the app silently hanging at 40%.
    """
    try:
        from pathlib import Path as _P
        cache = _P.home() / ".cache" / "huggingface" / "hub"
        model_dir = cache / "models--timm--MobileCLIP2-S0-OpenCLIP"
        blobs = model_dir / "blobs"

        # Check for a complete (non-.incomplete) file ≥ 100 MB
        already_cached = (
            blobs.exists() and
            any(
                f.suffix != ".incomplete" and f.stat().st_size > 100_000_000
                for f in blobs.iterdir()
            )
        )
        if already_cached:
            return  # nothing to do

        print("\n[preflight] MobileCLIP2-S0 weights not found in cache.")
        print("[preflight] Downloading now (~300 MB). This runs once.\n",
              flush=True)

        import os as _os
        _os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

        from huggingface_hub import snapshot_download
        snapshot_download(
            "timm/MobileCLIP2-S0-OpenCLIP",
            ignore_patterns=["*.incomplete"],
        )
        print("\n[preflight] Download complete.\n", flush=True)

    except Exception as exc:
        # Download failed — ModelHub will raise clearly when it tries to load
        print(f"[preflight] Could not pre-download CLIP weights: {exc}")
        print("[preflight] Continuing — will fail at model load if not cached.\n",
              flush=True)

# ── Run preflight before any Qt or project imports ────────────────────────────
_check_heavy_deps()
_ensure_packages()
_preflight_clip()

# ── Standard imports ──────────────────────────────────────────────────────────
import logging
import math
import os
import threading

_log = logging.getLogger(__name__)
from pathlib import Path
from typing import Optional

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PyQt6.QtCore    import Qt, QThread, pyqtSignal, QObject, QTimer
from PyQt6.QtGui     import QColor, QPalette, QFont
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget,
                              QHBoxLayout, QVBoxLayout, QComboBox,
                              QPushButton, QLabel, QSizePolicy,
                              QProgressBar, QFrame)

from flowfield   import FlowFieldWidget, render_offscreen, DEFAULT_PARAMS, PARAM_BOUNDS, PARAM_NAMES
from brain_viz   import BrainWidget
from calibration import CalibrationPanel
from inference   import ModelHub
from log_panel   import install_logging, LogPanel

MONO = "Menlo"

# ── Optimizer method constants ────────────────────────────────────────────────
METHOD_CMA       = "CMA-ES"
METHOD_BAYESIAN  = "Bayesian (Optuna)"
METHOD_RANDOM    = "Random Search"
METHOD_DIFFUSION = "Diffusion Model"
ALL_METHODS      = [METHOD_CMA, METHOD_BAYESIAN, METHOD_RANDOM, METHOD_DIFFUSION]


# ── Diffusion model architecture (ported from Notebook C) ─────────────────────
#
# These classes must stay in sync with Notebook C's training code.
# They are only used by _run_diffusion() inside OptimizerWorker.

class _SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0
        half  = dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, dtype=torch.float32) / (half - 1)
        )
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        t_f = t.float().unsqueeze(-1) * self.freqs
        emb = torch.cat([t_f.sin(), t_f.cos()], dim=-1)
        return self.proj(emb)


class _CondEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2), nn.LayerNorm(out_dim * 2),
            nn.SiLU(),
            nn.Linear(out_dim * 2, out_dim),
        )

    def forward(self, yeo):
        return self.net(yeo)


class _AdaLNResidualBlock(nn.Module):
    def __init__(self, dim, t_emb_dim, dropout=0.05):
        super().__init__()
        self.norm     = nn.LayerNorm(dim)
        self.fc1      = nn.Linear(dim, dim * 2)
        self.fc2      = nn.Linear(dim * 2, dim)
        self.dropout  = nn.Dropout(dropout)
        self.ada_proj = nn.Linear(t_emb_dim, dim * 2)

    def forward(self, x, t_emb):
        ada          = self.ada_proj(t_emb)
        scale, shift = ada.chunk(2, dim=-1)
        h = self.norm(x) * (1 + scale) + shift
        h = F.silu(self.fc1(h))
        h = self.dropout(self.fc2(h))
        return x + h


class _AestheticDenoiser(nn.Module):
    """
    Conditional denoiser.  Architecture must exactly match Notebook C's
    AestheticDenoiser so checkpoints load without key mismatches.
    """
    def __init__(self, cfg):
        super().__init__()
        P  = cfg["param_dim"]
        TD = cfg["t_emb_dim"]
        YD = cfg["yeo_cond_dim"]
        CD = cfg["cond_emb_dim"]
        HD = cfg["hidden_dim"]
        NB = cfg["n_blocks"]
        self.t_embed    = _SinusoidalTimeEmbed(TD)
        self.cond_enc   = _CondEncoder(YD, CD)
        self.input_proj = nn.Linear(P + TD + CD, HD)
        self.blocks     = nn.ModuleList([
            _AdaLNResidualBlock(HD, TD, dropout=0.05) for _ in range(NB)
        ])
        self.out_proj   = nn.Sequential(nn.LayerNorm(HD), nn.Linear(HD, P))

    def forward(self, theta_t, t, yeo):
        t_emb = self.t_embed(t)
        cond  = self.cond_enc(yeo)
        h     = self.input_proj(torch.cat([theta_t, t_emb, cond], dim=-1))
        for block in self.blocks:
            h = block(h, t_emb)
        return self.out_proj(h)


class _GaussianDiffusion:
    """
    Cosine noise schedule + DDIM reverse step.
    Must match Notebook C's GaussianDiffusion exactly.
    """
    def __init__(self, T: int):
        steps          = torch.arange(T + 1, dtype=torch.float64)
        s              = 0.008
        alphas_cumprod = torch.cos((steps / T + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas          = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas          = torch.clamp(betas, 0, 0.999).float()
        alphas         = 1.0 - betas
        self.T                   = T
        self.alphas_cumprod      = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

    def to(self, device):
        self.alphas_cumprod      = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        return self

    def ddim_step(self, x_t, t: int, t_prev: int, eps_pred):
        acp_t    = self.alphas_cumprod[t]
        acp_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=x_t.device)
        x0_pred  = (x_t - torch.sqrt(1.0 - acp_t) * eps_pred) / (torch.sqrt(acp_t) + 1e-8)
        x0_pred  = x0_pred.clamp(-1.5, 1.5)
        return torch.sqrt(acp_prev) * x0_pred + torch.sqrt(1.0 - acp_prev) * eps_pred


# ── Model loader (background thread) ─────────────────────────────────────────

class ModelLoader(QObject):
    progress = pyqtSignal(str, int)
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)

    def __init__(self, ckpt_path: str, yeo_path: str, parent=None):
        super().__init__(parent)
        self._ckpt_path = ckpt_path
        self._yeo_path  = yeo_path

    def run(self):
        import torch as _torch
        device = "cuda" if _torch.cuda.is_available() else "cpu"
        hub = ModelHub(
            ckpt_path=self._ckpt_path,
            yeo_path=self._yeo_path,
            device=device,
            progress_callback=self._on_load_progress,
        )
        self.progress.emit("Ready", 100)
        self.finished.emit(hub)

    def _on_load_progress(self, stage: str, pct: int):
        self.progress.emit(stage, pct)


# ── Optimizer worker ──────────────────────────────────────────────────────────

class OptimizerWorker(QObject):
    """
    Runs the chosen optimization backend in a dedicated QThread.

    Backends
    ────────
    CMA-ES        — Covariance Matrix Adaptation Evolution Strategy.
                    Good general-purpose optimizer; converges via tolx/tolfun.
                    Requires the `cma` package.

    Bayesian      — Optuna TPE (Tree-structured Parzen Estimator).
                    Builds a probabilistic model of the fitness landscape and
                    concentrates trials in promising regions.  Much more sample-
                    efficient than CMA-ES when each evaluation is expensive
                    (render + CLIP + BOLD).  Requires the `optuna` package.

    Random Search — Adaptive (1+λ) random walk.  Always available; useful as a
                    baseline or when neither cma nor optuna is installed.

    Diffusion     — Conditional DDPM sampler (Notebook C).
                    Loads the trained AestheticDenoiser checkpoint, computes a
                    personalised Yeo-7 conditioning vector from the user's
                    top-ranked calibration images, then runs DDIM sampling to
                    generate aesthetically informed parameter candidates.  Each
                    candidate is still evaluated through the full render→BOLD
                    pipeline so scores are on the same scale as other methods.
                    Falls back to Random Search if the checkpoint is not found.
    """

    params_ready = pyqtSignal(dict)
    bold_ready   = pyqtSignal(object, object)
    score_ready  = pyqtSignal(float, int)
    finished     = pyqtSignal()

    _POPSIZE     = 12
    _MAX_ITER    = 400
    _SIGMA0      = 0.45
    _SEED_MUTATE = 25

    # Diffusion-specific constants
    _DIFFUSION_ROUNDS    = 20   # number of DDIM sampling rounds
    _DIFFUSION_PER_ROUND = 16   # candidates generated per round
    _DIFFUSION_GUIDANCE  = 3.0  # CFG guidance scale (overridden by ckpt CFG if present)
    _DIFFUSION_STEPS     = 50   # DDIM steps (overridden by ckpt CFG if present)
    _DIFFUSION_TOP_K     = 5    # top-k calibration images used for Yeo target

    def __init__(self, hub: ModelHub,
                 calib_clips: np.ndarray,
                 calib_mobilenets: np.ndarray,
                 calib_scores: np.ndarray,
                 method: str = METHOD_CMA,
                 diffusion_ckpt_path: Optional[str] = None,
                 parent=None):
        super().__init__(parent)
        self._hub                  = hub
        self._calib_clips          = calib_clips
        self._calib_mobilenets     = calib_mobilenets
        self._calib_scores         = calib_scores
        self._method               = method
        self._diffusion_ckpt_path  = diffusion_ckpt_path
        self._stop_event           = threading.Event()

        self._lo = np.array([PARAM_BOUNDS[k][0] for k in PARAM_NAMES], np.float64)
        self._hi = np.array([PARAM_BOUNDS[k][1] for k in PARAM_NAMES], np.float64)

        # Shared mutable state updated by all backends
        self._best_score  = -np.inf
        self._best_params = {}

    def stop(self):
        self._stop_event.set()

    # ── Transformation helpers ────────────────────────────────────────────

    def _to_unit(self, x_nat):
        return (np.clip(x_nat, self._lo, self._hi) - self._lo) / (self._hi - self._lo)

    def _to_natural(self, x_unit):
        return self._lo + np.clip(x_unit, 0.0, 1.0) * (self._hi - self._lo)

    @staticmethod
    def _logit(u):
        u = np.clip(u, 1e-4, 1 - 1e-4)
        return np.log(u / (1.0 - u))

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def _decode_logit(self, x_raw):
        """Unbounded logit vector → parameter dict (used by CMA-ES)."""
        nat = self._to_natural(self._sigmoid(x_raw))
        return {k: float(v) for k, v in zip(PARAM_NAMES, nat)}

    def _decode_unit(self, x_unit):
        """[0,1] unit vector → parameter dict (used by Bayesian + Random)."""
        nat = self._to_natural(x_unit)
        return {k: float(v) for k, v in zip(PARAM_NAMES, nat)}

    # ── Single candidate evaluation ───────────────────────────────────────

    def _evaluate(self, params, seed):
        p = dict(params, seed=seed)
        pil           = render_offscreen(p, seed, W=256, H=256, n_steps=160)
        clip_emb, mn_emb = self._hub.encode_image(pil)
        bold          = self._hub.predict_bold(
            clip_emb, mn_emb,
            self._calib_clips, self._calib_mobilenets, self._calib_scores,
        )
        sc = self._hub.aesthetic_score(bold)
        return float(sc), bold

    def _check_improvement(self, sc, bold, params, seed, iteration):
        """Update best tracking and emit signals if improved."""
        if sc > self._best_score:
            self._best_score  = sc
            self._best_params = dict(params, seed=seed)
            self.params_ready.emit(self._best_params)
            self.bold_ready.emit(bold, self._hub.yeo_assign)
        if iteration % 5 == 0:
            self.score_ready.emit(self._best_score, iteration)

    # ── Main dispatcher ───────────────────────────────────────────────────

    def run(self):
        self._stop_event.clear()
        self._best_score  = -np.inf
        self._best_params = {}

        # ── Load normalization stats (all methods benefit) ────────────────
        # If the diffusion checkpoint is present, pull yeo_min/yeo_max from it
        # and apply to the hub.  This puts aesthetic_score() into [0,1] space
        # for the entire run, making scores meaningful and comparable to the
        # Notebook C training distribution regardless of which method is chosen.
        if (self._diffusion_ckpt_path and
                Path(self._diffusion_ckpt_path).exists()):
            try:
                _ckpt_stats = torch.load(
                    self._diffusion_ckpt_path,
                    map_location="cpu", weights_only=False)
                self._hub.set_normalization_stats(
                    np.array(_ckpt_stats["yeo_min"], dtype=np.float32).flatten(),
                    np.array(_ckpt_stats["yeo_max"], dtype=np.float32).flatten(),
                )
                del _ckpt_stats
                print("[Optimizer] Yeo normalization stats loaded from diffusion "
                      "checkpoint — aesthetic_score() is now in [0,1] space.",
                      flush=True)
            except Exception as _e:
                print(f"[Optimizer] Could not load normalization stats: {_e}. "
                      "Scores will be in raw z-score space.", flush=True)
        else:
            print("[Optimizer] No diffusion checkpoint found — "
                  "scores will be in raw z-score space.", flush=True)

        x0_nat  = np.array([DEFAULT_PARAMS[k] for k in PARAM_NAMES], np.float64)
        x0_unit = self._to_unit(x0_nat)
        x0      = self._logit(x0_unit)
        seed    = float(DEFAULT_PARAMS["seed"])

        if self._method == METHOD_DIFFUSION:
            self._run_diffusion(seed)
        elif self._method == METHOD_BAYESIAN:
            self._run_bayesian(seed)
        elif self._method == METHOD_CMA:
            try:
                import cma
                self._run_cma(x0, seed, cma)
            except ImportError:
                import logging
                logging.getLogger(__name__).warning(
                    "cma not available — falling back to Random Search")
                self._run_random_search(x0_unit, seed)
        else:
            self._run_random_search(x0_unit, seed)

        self.finished.emit()
        # Clear stats so hub is in a clean state for the next run
        self._hub.set_normalization_stats(None, None)

    # ── CMA-ES backend ────────────────────────────────────────────────────

    def _run_cma(self, x0, seed, cma_module):
        es = cma_module.CMAEvolutionStrategy(
            x0, self._SIGMA0,
            {"maxiter": self._MAX_ITER,
             "popsize": self._POPSIZE,
             "tolx":    1e-5,
             "tolfun":  1e-5,
             "verbose": -9}
        )
        iteration = 0
        while not es.stop() and not self._stop_event.is_set():
            solutions  = es.ask()
            fitnesses  = []
            for sol in solutions:
                if self._stop_event.is_set():
                    break
                params = self._decode_logit(sol)
                sc, bold = self._evaluate(params, seed)
                fitnesses.append(-sc)
                iteration += 1
                self._check_improvement(sc, bold, params, seed, iteration)

            if self._stop_event.is_set():
                break
            es.tell(solutions[:len(fitnesses)], fitnesses)

            if iteration > 0 and (iteration // self._POPSIZE) % self._SEED_MUTATE == 0:
                seed = float(np.random.randint(0, 99999))

    # ── Bayesian backend (Optuna TPE) ─────────────────────────────────────

    def _run_bayesian(self, seed):
        """
        Tree-structured Parzen Estimator via Optuna.

        TPE fits two density models — one over good parameter regions, one over
        bad — and samples candidates with high expected improvement.  It is much
        more sample-efficient than CMA-ES when evaluations are expensive, because
        it reuses information from all previous trials rather than maintaining
        only a population covariance.

        Each trial works directly in unit [0,1] space and maps to natural params,
        skipping the logit transform used by CMA-ES.
        """
        try:
            import optuna
        except ImportError:
            import logging
            logging.getLogger(__name__).warning(
                "optuna not available — falling back to Random Search")
            x0_nat  = np.array([DEFAULT_PARAMS[k] for k in PARAM_NAMES], np.float64)
            x0_unit = self._to_unit(x0_nat)
            self._run_random_search(x0_unit, seed)
            return

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        iteration   = [0]
        current_seed = [seed]

        def objective(trial):
            if self._stop_event.is_set():
                raise optuna.TrialPruned()

            x_unit = np.array([
                trial.suggest_float(name, 0.0, 1.0)
                for name in PARAM_NAMES
            ])
            params = self._decode_unit(x_unit)
            sc, bold = self._evaluate(params, current_seed[0])
            iteration[0] += 1
            self._check_improvement(sc, bold, params, current_seed[0], iteration[0])

            # Mutate seed periodically (same schedule as CMA-ES)
            if (iteration[0] > 0 and
                    iteration[0] % (self._SEED_MUTATE * self._POPSIZE) == 0):
                current_seed[0] = float(np.random.randint(0, 99999))

            return -sc   # optuna minimises

        def _stop_callback(study, trial):
            if self._stop_event.is_set():
                study.stop()

        n_trials = self._MAX_ITER * self._POPSIZE // 3
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=n_trials, callbacks=[_stop_callback])

    # ── Random search backend ─────────────────────────────────────────────

    def _run_random_search(self, x0_unit, seed):
        """
        Adaptive (1+λ) random walk in unit space.
        Shrinks step size on improvement, grows it on stagnation.
        Always available — no extra packages needed.
        """
        sigma     = self._SIGMA0
        x         = x0_unit.copy()
        iteration = 0

        for _ in range(self._MAX_ITER * self._POPSIZE):
            if self._stop_event.is_set():
                break
            candidate = np.clip(x + np.random.randn(len(x)) * sigma, 0.0, 1.0)
            params    = self._decode_unit(candidate)
            sc, bold  = self._evaluate(params, seed)
            iteration += 1

            if sc > self._best_score:
                x     = candidate
                sigma = max(0.04, sigma * 0.97)
            else:
                sigma = min(0.60, sigma * 1.005)

            self._check_improvement(sc, bold, params, seed, iteration)

            if (iteration > 0 and
                    iteration % (self._SEED_MUTATE * self._POPSIZE) == 0):
                seed = float(np.random.randint(0, 99999))

    # ── Diffusion backend ─────────────────────────────────────────────────

    def _run_diffusion(self, seed):
        """
        Conditional DDPM sampler from Notebook C used as an optimizer.

        Workflow
        ────────
        1. Load AestheticDenoiser checkpoint (best_denoiser.pt).
        2. Compute a personalised Yeo-7 conditioning target from the user's
           top-ranked calibration images by running their BOLD predictions
           through the same normalization used during training.
        3. Run _DIFFUSION_ROUNDS rounds of DDIM sampling, each yielding
           _DIFFUSION_PER_ROUND candidate parameter vectors.
        4. Evaluate every candidate through the full render→CLIP→BOLD pipeline
           and track the best score exactly like the other backends.

        Falls back to Random Search if the checkpoint is missing or fails.
        """
        ckpt_path = self._diffusion_ckpt_path
        if not ckpt_path or not Path(ckpt_path).exists():
            _log.warning(
                "Diffusion checkpoint not found at '%s' — falling back to "
                "Random Search.  Set DIFFUSION_CKPT env var or place "
                "best_denoiser.pt in diffusion_model/ next to main.py.",
                ckpt_path,
            )
            x0_nat  = np.array([DEFAULT_PARAMS[k] for k in PARAM_NAMES], np.float64)
            x0_unit = self._to_unit(x0_nat)
            self._run_random_search(x0_unit, seed)
            return

        try:
            device = self._hub.device
            ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)
        except Exception as exc:
            _log.error("Failed to load diffusion checkpoint: %s — falling back.", exc)
            x0_nat  = np.array([DEFAULT_PARAMS[k] for k in PARAM_NAMES], np.float64)
            x0_unit = self._to_unit(x0_nat)
            self._run_random_search(x0_unit, seed)
            return

        cfg_d   = ckpt["cfg"]
        yeo_min = np.array(ckpt["yeo_min"], dtype=np.float32).flatten()
        yeo_max = np.array(ckpt["yeo_max"], dtype=np.float32).flatten()

        # ── Apply normalization stats to hub immediately ──────────────────
        # This fixes aesthetic_score() for ALL evaluations in this run:
        # raw z-scored BOLD means are normalised to [0,1] using the exact
        # same statistics used during Notebook C training, so the optimizer
        # tracks scores in the same space the diffusion model was trained on.
        self._hub.set_normalization_stats(yeo_min, yeo_max)
        print("[Diffusion] Normalization stats applied to ModelHub.", flush=True)

        try:
            denoiser = _AestheticDenoiser(cfg_d).to(device).eval()
            denoiser.load_state_dict(ckpt["model_state"])
        except Exception as exc:
            _log.error("Denoiser architecture mismatch: %s — falling back.", exc)
            x0_nat  = np.array([DEFAULT_PARAMS[k] for k in PARAM_NAMES], np.float64)
            x0_unit = self._to_unit(x0_nat)
            self._run_random_search(x0_unit, seed)
            return

        diffusion  = _GaussianDiffusion(cfg_d["T"]).to(device)
        guidance   = cfg_d.get("guidance_scale", self._DIFFUSION_GUIDANCE)
        ddim_steps = cfg_d.get("ddim_steps",     self._DIFFUSION_STEPS)
        param_dim  = cfg_d["param_dim"]

        print("[Diffusion] Computing personalised Yeo-7 conditioning target…",
              flush=True)
        yeo_target = self._compute_yeo_target(yeo_min, yeo_max)
        print(f"[Diffusion] Yeo target: {np.round(yeo_target, 3)}", flush=True)

        iteration = 0
        for round_i in range(self._DIFFUSION_ROUNDS):
            if self._stop_event.is_set():
                break

            print(f"[Diffusion] Round {round_i + 1}/{self._DIFFUSION_ROUNDS} "
                  f"— sampling {self._DIFFUSION_PER_ROUND} candidates…",
                  flush=True)

            theta_samples = self._ddim_sample(
                denoiser, diffusion, yeo_target,
                n_samples=self._DIFFUSION_PER_ROUND,
                guidance_scale=guidance,
                ddim_steps=ddim_steps,
                param_dim=param_dim,
                device=device,
            )

            for i in range(self._DIFFUSION_PER_ROUND):
                if self._stop_event.is_set():
                    break
                params   = self._decode_unit(theta_samples[i].astype(np.float64))
                sc, bold = self._evaluate(params, seed)
                iteration += 1
                self._check_improvement(sc, bold, params, seed, iteration)

            if round_i % 5 == 4:
                seed = float(np.random.randint(0, 99999))

        print(f"[Diffusion] Done — {iteration} candidates, "
              f"best score: {self._best_score:+.4f}", flush=True)
        # Clear stats so other methods aren't affected if this worker is reused
        self._hub.set_normalization_stats(None, None)

    def _compute_yeo_target(self, yeo_min: np.ndarray,
                             yeo_max: np.ndarray) -> np.ndarray:
        """
        Personalised Yeo-7 conditioning vector from the user's top-ranked
        calibration images.  Mirrors compute_personalized_target() in Notebook C.
        Returns float32 array of shape (7,) in [0, 1].
        """
        top_k   = min(self._DIFFUSION_TOP_K, len(self._calib_scores))
        top_idx = np.argsort(self._calib_scores)[::-1][:top_k]

        bolds = []
        for i in top_idx:
            bold = self._hub.predict_bold(
                self._calib_clips[i],
                self._calib_mobilenets[i],
                self._calib_clips,
                self._calib_mobilenets,
                self._calib_scores,
            )
            bolds.append(bold)

        mean_bold  = np.mean(bolds, axis=0)
        yeo_assign = self._hub.yeo_assign
        yeo_m      = np.zeros(7, dtype=np.float32)
        for j, nid in enumerate([1, 2, 3, 4, 5, 6, 7]):
            mask = yeo_assign == nid
            if mask.sum() > 0:
                yeo_m[j] = float(mean_bold[mask].mean())

        yeo_n = (yeo_m - yeo_min) / (yeo_max - yeo_min + 1e-8)
        return np.clip(yeo_n, 0.0, 1.0).astype(np.float32)

    @torch.no_grad()
    def _ddim_sample(self, denoiser: _AestheticDenoiser,
                     diffusion: _GaussianDiffusion,
                     yeo_target: np.ndarray,
                     n_samples: int,
                     guidance_scale: float,
                     ddim_steps: int,
                     param_dim: int,
                     device) -> np.ndarray:
        """
        DDIM reverse diffusion with classifier-free guidance.
        Returns float32 array of shape (n_samples, param_dim) in [0, 1].
        Mirrors ddim_sample() from Notebook C.
        """
        yeo_t  = torch.FloatTensor(yeo_target).unsqueeze(0).expand(n_samples, -1).to(device)
        yeo_uc = torch.zeros_like(yeo_t)
        yeo_in = torch.cat([yeo_uc, yeo_t], dim=0)

        T         = diffusion.T
        timesteps = torch.linspace(T - 1, 0, ddim_steps + 1, dtype=torch.long).tolist()
        theta_t   = torch.randn(n_samples, param_dim, device=device)

        for i in range(ddim_steps):
            if self._stop_event.is_set():
                break
            t      = int(timesteps[i])
            t_prev = int(timesteps[i + 1])
            t_vec  = torch.full((n_samples,), t, device=device, dtype=torch.long)

            theta_double = torch.cat([theta_t, theta_t], dim=0)
            t_double     = torch.cat([t_vec,   t_vec],   dim=0)
            eps_double   = denoiser(theta_double, t_double, yeo_in)

            eps_uc, eps_c = eps_double.chunk(2, dim=0)
            eps_guided    = eps_uc + guidance_scale * (eps_c - eps_uc)
            theta_t       = diffusion.ddim_step(theta_t, t, t_prev, eps_guided)

        return theta_t.cpu().numpy().clip(0.0, 1.0).astype(np.float32)


# ── Side panel ────────────────────────────────────────────────────────────────

class SidePanel(QWidget):
    optimize_requested = pyqtSignal()
    stop_requested     = pyqtSignal()
    seed_requested     = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(360)
        self.setStyleSheet("background: transparent;")

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Brain widget ──────────────────────────────────────────────────
        self.brain = BrainWidget()
        self.brain.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Preferred)
        root.addWidget(self.brain)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #2a2040;")
        root.addWidget(sep)

        # ── Model loading progress bar ────────────────────────────────────
        self._load_bar = QProgressBar()
        self._load_bar.setRange(0, 100)
        self._load_bar.setValue(0)
        self._load_bar.setTextVisible(True)
        self._load_bar.setFormat("Loading: %p%")
        self._load_bar.setFixedHeight(16)
        self._load_bar.setVisible(True)
        self._load_bar.setStyleSheet(f"""
            QProgressBar {{
                background: #120e22;
                border: 1px solid #2a1e4a;
                border-radius: 3px;
                color: #8877cc;
                font: 8px '{MONO}';
                text-align: center;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #5530aa, stop:0.5 #8855dd, stop:1 #5530aa);
                border-radius: 2px;
            }}
        """)
        root.addWidget(self._load_bar)

        # ── Status label ──────────────────────────────────────────────────
        self._status = QLabel("Loading model…")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setWordWrap(True)
        self._status.setStyleSheet(
            f"color: #6655aa; font: 9px '{MONO}'; padding: 2px 4px;")
        root.addWidget(self._status)

        self._dot_count = 0
        self._dot_timer = QTimer(self)
        self._dot_timer.timeout.connect(self._pulse_status)
        self._dot_timer.start(500)

        # ── Optimizer method selector ─────────────────────────────────────
        method_row = QHBoxLayout()
        method_lbl = QLabel("METHOD")
        method_lbl.setStyleSheet(
            f"color: #554488; font: 8px '{MONO}'; letter-spacing: 1.5px;")
        method_row.addWidget(method_lbl)

        self._method_combo = QComboBox()
        self._method_combo.addItems(ALL_METHODS)
        self._method_combo.setCurrentText(METHOD_CMA)
        self._method_combo.setFixedHeight(26)
        self._method_combo.setStyleSheet(f"""
            QComboBox {{
                background: #14102a;
                color: #aaa0dd;
                border: 1px solid #3a2e60;
                border-radius: 4px;
                padding: 2px 8px;
                font: 9px '{MONO}';
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            QComboBox::down-arrow {{
                width: 8px;
                height: 8px;
            }}
            QComboBox QAbstractItemView {{
                background: #1a1430;
                color: #aaa0dd;
                border: 1px solid #3a2e60;
                selection-background-color: #4a3888;
                font: 9px '{MONO}';
            }}
        """)
        self._method_combo.currentTextChanged.connect(self._on_method_changed)
        method_row.addWidget(self._method_combo, stretch=1)
        root.addLayout(method_row)

        # ── Method description label ──────────────────────────────────────
        self._method_desc = QLabel(self._method_description(METHOD_CMA))
        self._method_desc.setWordWrap(True)
        self._method_desc.setStyleSheet(
            f"color: #443366; font: 8px '{MONO}'; padding: 0px 2px;")
        root.addWidget(self._method_desc)

        # ── Iteration progress bar ────────────────────────────────────────
        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setValue(0)
        self._bar.setTextVisible(True)
        self._bar.setFormat("Iteration: %v / %m")
        self._bar.setFixedHeight(18)
        self._bar.setStyleSheet(f"""
            QProgressBar {{
                background: #1a1430;
                border: 1px solid #2a1e4a;
                border-radius: 3px;
                color: #cc8866;
                font: 8px '{MONO}';
                text-align: center;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #ff193c, stop:0.5 #ff6400, stop:1 #ffd20a);
                border-radius: 2px;
            }}
        """)
        root.addWidget(self._bar)

        # ── Control buttons ───────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self._btn_opt = QPushButton("▶  OPTIMISE")
        self._btn_opt.setEnabled(False)
        self._btn_opt.setFixedHeight(36)
        self._btn_opt.setStyleSheet(self._btn_css("#5530aa", "#8855dd", bold=True))
        self._btn_opt.clicked.connect(self.optimize_requested)
        btn_row.addWidget(self._btn_opt)

        self._btn_stop = QPushButton("■  STOP")
        self._btn_stop.setEnabled(False)
        self._btn_stop.setFixedHeight(36)
        self._btn_stop.setStyleSheet(self._btn_css("#6a1a1a", "#cc3333", bold=True))
        self._btn_stop.clicked.connect(self.stop_requested)
        btn_row.addWidget(self._btn_stop)

        root.addLayout(btn_row)

        self._btn_seed = QPushButton("⟳  NEW SEED")
        self._btn_seed.setFixedHeight(28)
        self._btn_seed.setStyleSheet(self._btn_css("#1a2a44", "#335588"))
        self._btn_seed.clicked.connect(self.seed_requested)
        root.addWidget(self._btn_seed)

        root.addStretch()

    # ── Method selector ───────────────────────────────────────────────────

    @staticmethod
    def _method_description(method: str) -> str:
        return {
            METHOD_CMA:       "Covariance Matrix Adaptation. Learns parameter correlations. Converges via tolerance.",
            METHOD_BAYESIAN:  "Optuna TPE. Builds a probability model of the landscape. Most sample-efficient.",
            METHOD_RANDOM:    "Adaptive random walk. No extra packages needed. Good baseline.",
            METHOD_DIFFUSION: (
                "Conditional DDPM (Notebook C). Generates informed candidates "
                "from your calibration images via DDIM sampling. Requires "
                "diffusion_model/best_denoiser.pt."
            ),
        }.get(method, "")

    def _on_method_changed(self, method: str):
        self._method_desc.setText(self._method_description(method))

    def selected_method(self) -> str:
        return self._method_combo.currentText()

    # ── Animated loading dots ─────────────────────────────────────────────

    def _pulse_status(self):
        if self._load_bar.isVisible():
            dots = "." * ((self._dot_count % 3) + 1)
            base = self._status.text().rstrip(". ")
            self._status.setText(f"{base}{dots}")
            self._dot_count += 1

    # ── Public state setters ──────────────────────────────────────────────

    def set_status(self, text: str):
        self._dot_timer.stop()
        self._load_bar.setVisible(False)
        self._status.setText(text)

    def set_loading(self, text: str = "Loading model"):
        self._load_bar.setVisible(True)
        self._load_bar.setRange(0, 100)
        self._load_bar.setValue(0)
        self._load_bar.setFormat(f"{text}: %p%")
        self._status.setText(text)
        self._dot_count = 0
        self._dot_timer.start(500)

    def set_load_progress(self, stage: str, percentage: int):
        self._load_bar.setValue(percentage)
        self._load_bar.setFormat(f"{stage}: %p%")
        self._status.setText(f"{stage}…")

    def set_calibrated(self, yes: bool):
        self._btn_opt.setEnabled(yes)

    def set_running(self, yes: bool):
        self._btn_opt.setEnabled(not yes)
        self._btn_stop.setEnabled(yes)
        self._method_combo.setEnabled(not yes)   # lock method while running
        if yes:
            method = self.selected_method()
            if method == METHOD_BAYESIAN:
                total = (OptimizerWorker._MAX_ITER * OptimizerWorker._POPSIZE) // 3
            elif method == METHOD_DIFFUSION:
                total = (OptimizerWorker._DIFFUSION_ROUNDS
                         * OptimizerWorker._DIFFUSION_PER_ROUND)
            else:
                total = OptimizerWorker._MAX_ITER * OptimizerWorker._POPSIZE
            self._bar.setRange(0, total)
            self._bar.setValue(0)
            self._bar.setFormat("Iteration: %v / %m")

    def set_score(self, score: float, iteration: int):
        self._bar.setValue(iteration)
        max_iter     = self._bar.maximum()
        progress_pct = (iteration / max_iter * 100) if max_iter > 0 else 0
        self._status.setText(
            f"Progress: {progress_pct:.1f}%\n"
            f"Iteration {iteration} / {max_iter}\n"
            f"Best score: {score:+.4f}"
        )

    # ── Styling ───────────────────────────────────────────────────────────

    @staticmethod
    def _btn_css(dark: str, light: str, bold: bool = False) -> str:
        weight = "bold " if bold else ""
        return f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 {light}, stop:1 {dark});
                color: #ddd8ff;
                border: 1px solid {light};
                border-radius: 5px;
                padding: 4px 10px;
                font: {weight}9px '{MONO}';
                letter-spacing: 1.5px;
            }}
            QPushButton:hover   {{ background: {light}; }}
            QPushButton:pressed {{ background: {dark};  }}
            QPushButton:disabled {{
                background: #1a1428;
                color: #3a2d55;
                border-color: #2a2040;
            }}
        """


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neuroaesthetics Flow Optimizer")
        self.resize(1340, 840)
        self.setMinimumSize(900, 620)
        self.setStyleSheet("QMainWindow, QWidget#central { background: #07070e; }")

        self._hub: Optional[ModelHub]          = None
        self._calib_clips                      = None
        self._calib_mobilenets                 = None
        self._calib_scores                     = None
        self._opt_thread: Optional[QThread]    = None
        self._opt_worker: Optional[OptimizerWorker] = None

        self._build_ui()

        _repo     = Path(__file__).resolve().parent
        ckpt_path = os.environ.get("BOLD_CKPT", str(_repo / "model" / "best_model.pt"))
        yeo_path  = os.environ.get("YEO_PATH",  str(_repo / "preprocessed" / "yeo7_difumo1024_assignment.npy"))

        # Diffusion checkpoint — Notebook C saves to diffusion_model/best_denoiser.pt
        self._diffusion_ckpt = os.environ.get(
            "DIFFUSION_CKPT",
            str(_repo / "diffusion_model" / "best_denoiser.pt"),
        )
        if Path(self._diffusion_ckpt).exists():
            print(f"[App] Diffusion checkpoint found: {self._diffusion_ckpt}", flush=True)
        else:
            print(f"[App] Diffusion checkpoint NOT found at {self._diffusion_ckpt}. "
                  "Diffusion method will fall back to Random Search if selected.",
                  flush=True)

        self._start_model_loader(ckpt_path, yeo_path)

    # ── Model loading ─────────────────────────────────────────────────────

    def _start_model_loader(self, ckpt_path, yeo_path):
        self._load_thread = QThread(self)
        self._load_worker = ModelLoader(ckpt_path, yeo_path)
        self._load_worker.moveToThread(self._load_thread)
        self._load_thread.started.connect(self._load_worker.run)
        self._load_worker.progress.connect(self._on_load_progress)
        self._load_worker.finished.connect(self._on_hub_ready)
        self._load_worker.finished.connect(self._load_thread.quit)
        self._side.set_loading("Initializing")
        self._load_thread.start()

    def _on_load_progress(self, stage: str, percentage: int):
        self._side.set_load_progress(stage, percentage)

    def _on_hub_ready(self, hub: ModelHub):
        self._hub = hub
        self._calib.set_hub(hub)
        self._side.set_status(
            "Models loaded.\n"
            "Upload & rank images, encode,\nthen press OPTIMISE."
        )
        _log.info("ModelHub ready.")

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)

        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        panels = QWidget()
        layout = QHBoxLayout(panels)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)

        self._calib = CalibrationPanel(hub=None)
        self._calib.setFixedWidth(380)
        self._calib.calibration_ready.connect(self._on_calibration_ready)
        layout.addWidget(self._calib)

        layout.addWidget(self._divider())

        self._flow = FlowFieldWidget()
        self._flow.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        layout.addWidget(self._flow, stretch=1)

        layout.addWidget(self._divider())

        self._side = SidePanel()
        self._side.optimize_requested.connect(self._start_optimizer)
        self._side.stop_requested.connect(self._stop_optimizer)
        self._side.seed_requested.connect(self._flow.new_seed)
        layout.addWidget(self._side)

        outer.addWidget(panels, stretch=1)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: #1a1430; border: none;")
        outer.addWidget(sep)

        self._log_panel = LogPanel()
        self._log_panel.setFixedHeight(120)
        outer.addWidget(self._log_panel)

    @staticmethod
    def _divider():
        d = QFrame()
        d.setFrameShape(QFrame.Shape.VLine)
        d.setFixedWidth(1)
        d.setStyleSheet("background: #1e1a34; border: none;")
        return d

    # ── Calibration ───────────────────────────────────────────────────────

    def _on_calibration_ready(self, clips: np.ndarray,
                               mobilenets: np.ndarray,
                               scores: np.ndarray):
        self._calib_clips      = clips
        self._calib_mobilenets = mobilenets
        self._calib_scores     = scores
        self._side.set_calibrated(True)
        self._side.set_status(
            f"{len(clips)} images encoded.\n"
            "Select a method and press OPTIMISE."
        )
        self._side.brain.clear_activations()
        _log.info("Calibration ready — %d images.", len(clips))

    # ── Optimizer lifecycle ───────────────────────────────────────────────

    def _start_optimizer(self):
        if self._calib_clips is None:
            self._side.set_status("Encode images first!")
            return
        if self._opt_thread and self._opt_thread.isRunning():
            return

        method = self._side.selected_method()
        print(f"[App] Starting optimizer — method: {method}")

        self._opt_thread = QThread(self)
        self._opt_worker = OptimizerWorker(
            self._hub,
            self._calib_clips,
            self._calib_mobilenets,
            self._calib_scores,
            method=method,
            diffusion_ckpt_path=self._diffusion_ckpt,
        )
        self._opt_worker.moveToThread(self._opt_thread)

        self._opt_thread.started.connect(self._opt_worker.run)
        self._opt_worker.params_ready.connect(self._flow.set_params)
        self._opt_worker.score_ready.connect(self._on_score_update)
        self._opt_worker.bold_ready.connect(self._on_bold_update)
        self._opt_worker.finished.connect(self._on_optimizer_done)

        self._side.set_running(True)
        self._flow.set_optimizing(True)
        self._side.set_status(f"Optimising ({method})…")
        self._opt_thread.start()

    def _stop_optimizer(self):
        if self._opt_worker:
            self._opt_worker.stop()
        self._flow.set_optimizing(False)
        self._side.set_running(False)
        self._side.set_status("Stopped.")
        print("[App] Optimizer stop requested.")

    def _on_score_update(self, score: float, iteration: int):
        self._flow.set_score(score, iteration)
        self._side.set_score(score, iteration)

    def _on_bold_update(self, bold: np.ndarray, yeo_assign: np.ndarray):
        # Pass raw bold to brain widget (it has its own sigmoid transform),
        # but ALSO feed it normalized values via update_bold_state so the
        # bars reflect real differentiation instead of a flat sigmoid≈0.5.
        self._side.brain.update_activations(bold, yeo_assign)

        # Use hub's normalized_yeo_means() — in [0,1] if stats are loaded,
        # sigmoid fallback otherwise.  Both are better than raw z-scores.
        normed = self._hub.normalized_yeo_means(bold)
        activations = {nid: float(normed[j])
                       for j, nid in enumerate([1, 2, 3, 4, 5, 6, 7])}
        # Override BrainWidget bars with normalized values directly
        self._side.brain.update_normalized(activations)
        self._flow.update_bold_state(activations)

    def _on_optimizer_done(self):
        self._flow.set_optimizing(False)
        self._side.set_running(False)
        self._side.set_calibrated(True)
        self._side.set_status("Optimisation complete.")
        if self._opt_thread:
            self._opt_thread.quit()
            self._opt_thread.wait(5000)
        print("[App] Optimizer finished.")

    # ── Cleanup ───────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._stop_optimizer()
        if self._opt_thread and self._opt_thread.isRunning():
            self._opt_thread.quit()
            self._opt_thread.wait(3000)
        super().closeEvent(event)


# ── Dark palette ──────────────────────────────────────────────────────────────

def _apply_dark_palette(app: QApplication):
    pal = QPalette()
    def c(r, g, b): return QColor(r, g, b)
    pal.setColor(QPalette.ColorRole.Window,          c(8,  7,  18))
    pal.setColor(QPalette.ColorRole.WindowText,      c(200, 190, 240))
    pal.setColor(QPalette.ColorRole.Base,            c(12, 10, 26))
    pal.setColor(QPalette.ColorRole.AlternateBase,   c(18, 15, 36))
    pal.setColor(QPalette.ColorRole.ToolTipBase,     c(24, 20, 48))
    pal.setColor(QPalette.ColorRole.ToolTipText,     c(200, 190, 240))
    pal.setColor(QPalette.ColorRole.Text,            c(200, 190, 240))
    pal.setColor(QPalette.ColorRole.BrightText,      c(240, 230, 255))
    pal.setColor(QPalette.ColorRole.Button,          c(20, 16, 42))
    pal.setColor(QPalette.ColorRole.ButtonText,      c(200, 190, 240))
    pal.setColor(QPalette.ColorRole.Link,            c(120, 90, 220))
    pal.setColor(QPalette.ColorRole.Highlight,       c(80,  50, 170))
    pal.setColor(QPalette.ColorRole.HighlightedText, c(230, 220, 255))
    for role, col in [
        (QPalette.ColorRole.WindowText, c(80, 70, 110)),
        (QPalette.ColorRole.Text,       c(80, 70, 110)),
        (QPalette.ColorRole.ButtonText, c(70, 60, 100)),
    ]:
        pal.setColor(QPalette.ColorGroup.Disabled, role, col)
    app.setPalette(pal)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(sys.argv)
    app.setApplicationName("Neuroaesthetics Flow Optimizer")
    app.setApplicationDisplayName("Neuroaesthetics")
    app.setStyle("Fusion")

    install_logging()
    _apply_dark_palette(app)

    font = QFont(MONO, 10)
    app.setFont(font)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()