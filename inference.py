"""
inference.py
Loads MobileCLIP2-S0 (open_clip) + MobileNetV3-Small (torchvision) and the
trained CalibratedBoldPredictor checkpoint.

Image encoding returns BOTH embeddings:
    clip      (512,)  — L2-normalised semantic embedding
    mobilenet (176,)  — multi-layer GAP features (hooks on features[1,3,6,10])

These are concatenated to a 688-d vector before being passed to the predictor,
matching the Notebook-B training setup exactly.

Demo mode has been removed entirely.  If any model fails to load the app
raises immediately with a clear error rather than silently substituting
random predictions.  This ensures every prediction is real.
"""

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as tv_transforms

_log = logging.getLogger(__name__)


# ── MobileNetV3-Small feature extractor ──────────────────────────────────────

class MNV3FeatureExtractor(nn.Module):
    """
    Wraps MobileNetV3-Small and exposes multi-layer GAP features.

    Hooked layers and channels (features[1,3,6,10] → 16+24+40+96 = 176 d):
        features[1]  : 16-ch — edges, colour boundaries
        features[3]  : 24-ch — early texture / colour regions
        features[6]  : 40-ch — mid-level form (actual torchvision output; not 48)
        features[10] : 96-ch — complex shapes before high-level collapse

    NOTE: torchvision reports 40 channels at features[6], not 48 as listed in
    the Notebook-A comment, giving total dim=176 (confirmed at training time).
    """

    HOOK_LAYERS = [1, 3, 6, 10]

    def __init__(self, device: str):
        super().__init__()
        self.device       = device
        self._activations = {}
        self._handles     = []

        weights = tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        base    = tv_models.mobilenet_v3_small(weights=weights)
        self.features = base.features
        del base

        self._gap = nn.AdaptiveAvgPool2d(1)

        for layer_idx in self.HOOK_LAYERS:
            handle = self.features[layer_idx].register_forward_hook(
                self._make_hook(layer_idx)
            )
            self._handles.append(handle)

    def _make_hook(self, idx):
        def _hook(module, inp, out):
            self._activations[idx] = self._gap(out).squeeze(-1).squeeze(-1)
        return _hook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 224, 224) → (B, total_dim)"""
        self._activations.clear()
        stop = max(self.HOOK_LAYERS) + 1
        with torch.no_grad():
            for i, layer in enumerate(self.features):
                x = layer(x)
                if i >= stop:
                    break
        parts = [self._activations[l] for l in self.HOOK_LAYERS]
        return torch.cat(parts, dim=1)

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ── Model architecture ────────────────────────────────────────────────────────

class FourierRankEmbed(nn.Module):
    def __init__(self, n_fourier, d_model):
        super().__init__()
        freqs = torch.arange(1, n_fourier + 1, dtype=torch.float32) * math.pi
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(
            nn.Linear(2 * n_fourier, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, score):
        s = score * self.freqs
        feats = torch.cat([s.sin(), s.cos()], dim=-1)
        return self.proj(feats)


class CalibratedBoldPredictor(nn.Module):
    """
    Predicts whole-brain BOLD (1024 DiFuMo parcels) for a new image,
    personalised by (image, aesthetic-rank) calibration pairs.

    Input per image: clip (512) + mobilenet (176) = 688-d vector.

    Must stay in sync with Notebook B's CalibratedBoldPredictor.
    cfg keys consumed: img_feat_dim, bold_dim, d_model, n_attn_heads,
    n_attn_layers, n_fourier_feats, mlp_hidden, mlp_n_layers, dropout.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        D  = cfg["d_model"]
        I  = cfg["img_feat_dim"]        # 688  (clip 512 + mobilenet 176)
        B  = cfg["bold_dim"]            # 1024
        H  = cfg["n_attn_heads"]
        L  = cfg["n_attn_layers"]
        F_ = cfg["n_fourier_feats"]
        MH = cfg["mlp_hidden"]
        NL = cfg.get("mlp_n_layers", 3)
        P  = cfg["dropout"]

        # Shared image projector: cat(clip, mobilenet) → d_model
        self.img_proj    = nn.Sequential(nn.Linear(I, D), nn.LayerNorm(D))
        self.rank_embed  = FourierRankEmbed(n_fourier=F_, d_model=D)
        self.prior_token = nn.Parameter(torch.randn(1, 1, D) * 0.02)

        self.cross_layers = nn.ModuleList([
            nn.ModuleDict({
                "attn":   nn.MultiheadAttention(D, H, dropout=P, batch_first=True),
                "ln_q":   nn.LayerNorm(D),
                "ln_kv":  nn.LayerNorm(D),
                "ffn":    nn.Sequential(nn.Linear(D, D * 4), nn.GELU(),
                                        nn.Dropout(P), nn.Linear(D * 4, D)),
                "ln_ffn": nn.LayerNorm(D),
            }) for _ in range(L)
        ])

        self.predictor = self._build_predictor(
            in_dim=D * 2, hidden=MH, out_dim=B, n_layers=NL, dropout=P)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _build_predictor(in_dim: int, hidden: int, out_dim: int,
                         n_layers: int, dropout: float) -> nn.Sequential:
        if n_layers < 2:
            raise ValueError(f"mlp_n_layers must be >= 2, got {n_layers}")
        layers = [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden),
                  nn.GELU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden, out_dim))
        return nn.Sequential(*layers)

    def _project_image(self,
                       clips:      torch.Tensor,
                       mobilenets: torch.Tensor) -> torch.Tensor:
        return self.img_proj(torch.cat([clips, mobilenets], dim=-1))

    def _build_kv(self,
                  calib_clips:      torch.Tensor,   # (B, N, 512)
                  calib_mobilenets: torch.Tensor,   # (B, N, 176)
                  calib_scores:     torch.Tensor    # (B, N, 1)
                 ) -> torch.Tensor:                  # (B, N+1, D)
        img_feats  = self._project_image(calib_clips, calib_mobilenets)
        rank_feats = self.rank_embed(calib_scores)
        calib_tok  = img_feats + rank_feats
        B     = calib_clips.shape[0]
        prior = self.prior_token.expand(B, -1, -1)
        return torch.cat([prior, calib_tok], dim=1)

    def forward(self,
                new_clip:         torch.Tensor,   # (B, 512)
                new_mobilenet:    torch.Tensor,   # (B, 176)
                calib_clips:      torch.Tensor,   # (B, N, 512)
                calib_mobilenets: torch.Tensor,   # (B, N, 176)
                calib_scores:     torch.Tensor,   # (B, N, 1)
                pad_mask:         torch.Tensor    # (B, N) bool; True = pad
               ) -> torch.Tensor:                 # (B, 1024)
        B = new_clip.shape[0]

        query_feat = self._project_image(new_clip, new_mobilenet)  # (B, D)
        query_seq  = query_feat.unsqueeze(1)                        # (B, 1, D)

        kv_tokens   = self._build_kv(calib_clips, calib_mobilenets, calib_scores)
        prior_valid = torch.zeros(B, 1, dtype=torch.bool, device=pad_mask.device)
        full_mask   = torch.cat([prior_valid, pad_mask], dim=1)

        x = query_seq
        for layer in self.cross_layers:
            q_n = layer["ln_q"](x)
            k_n = layer["ln_kv"](kv_tokens)
            ao, _ = layer["attn"](q_n, k_n, k_n, key_padding_mask=full_mask)
            x = x + ao
            x = x + layer["ffn"](layer["ln_ffn"](x))

        context   = x.squeeze(1)
        combined  = torch.cat([query_feat, context], dim=-1)
        return self.predictor(combined)


# ── ModelHub ──────────────────────────────────────────────────────────────────

class ModelHub:
    """
    Holds the CLIP encoder, the MobileNetV3-Small encoder, and the BOLD predictor.
    All three must load successfully — there is no demo / random fallback.

    Parameters
    ──────────
    ckpt_path          : path to the CalibratedBoldPredictor .pt checkpoint
    yeo_path           : path to the yeo7_difumo1024_assignment.npy file
    device             : 'cpu' or 'cuda'
    progress_callback  : optional callable(stage: str, pct: int)
    """

    CLIP_MODEL_ID  = "MobileCLIP2-S0"
    MNV3_IMG_SIZE  = 224
    MNV3_MEAN      = [0.485, 0.456, 0.406]
    MNV3_STD       = [0.229, 0.224, 0.225]

    def __init__(self,
                 ckpt_path: Optional[str] = None,
                 yeo_path:  Optional[str] = None,
                 device: str = "cpu",
                 progress_callback=None):
        self.device   = device
        self.clip_dim = 512
        self.bold_dim = 1024
        self._progress_cb = progress_callback

        # Per-network normalisation stats from Notebook C training data.
        # Shape (7,) float32.  None until set_normalization_stats() is called.
        # When set, aesthetic_score() normalises Yeo means to [0,1] before
        # computing the weighted score — matching the training distribution exactly.
        self.yeo_min: Optional[np.ndarray] = None
        self.yeo_max: Optional[np.ndarray] = None

        # ── Yeo-7 assignment ──────────────────────────────────────────────
        self._report("Loading Yeo-7 assignment", 10)
        if not yeo_path or not Path(yeo_path).exists():
            raise FileNotFoundError(
                f"Yeo-7 assignment not found at: {yeo_path}\n"
                "Expected: preprocessed/yeo7_difumo1024_assignment.npy")
        self.yeo_assign = np.load(yeo_path).astype(np.int8)
        _log.info("Yeo-7 assignment loaded from %s", yeo_path)

        # ── CLIP encoder ──────────────────────────────────────────────────
        self._report("Loading CLIP encoder", 25)
        self._load_clip()

        # ── MobileNetV3-Small encoder ─────────────────────────────────────
        self._report("Loading MobileNetV3-Small", 55)
        self._load_mobilenet()

        # ── BOLD predictor ────────────────────────────────────────────────
        self._report("Loading BOLD predictor", 75)
        self._load_bold(ckpt_path)

        self._report("Ready", 100)

    def _report(self, stage: str, pct: int):
        if self._progress_cb:
            self._progress_cb(stage, pct)

    # ── CLIP ──────────────────────────────────────────────────────────────

    def _load_clip(self):
        import time
        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "open_clip_torch is not installed.\n"
                "Run:  pip install open_clip_torch")

        # Remove stale file locks (can block download after a crash)
        cache_locks = Path.home() / ".cache" / "huggingface" / "hub" / ".locks"
        if cache_locks.exists():
            now = time.time()
            for lock_file in cache_locks.rglob("*.lock"):
                try:
                    if now - lock_file.stat().st_mtime > 60:
                        _log.warning("Removing stale lock: %s", lock_file)
                        lock_file.unlink(missing_ok=True)
                except Exception as e:
                    _log.debug("Could not remove lock %s: %s", lock_file, e)

        max_retries = 3
        last_exc    = None
        for attempt in range(max_retries):
            try:
                tags = open_clip.list_pretrained_tags_by_model(self.CLIP_MODEL_ID)
                tag  = tags[0] if tags else "openai"
                _log.info("Loading CLIP model (attempt %d/%d) tag=%s …",
                          attempt + 1, max_retries, tag)
                self._clip, _, self._clip_preproc = open_clip.create_model_and_transforms(
                    self.CLIP_MODEL_ID, pretrained=tag)
                self._clip = self._clip.to(self.device).eval()
                _log.info("CLIP loaded: %s (%s)", self.CLIP_MODEL_ID, tag)
                return
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    _log.warning("CLIP attempt %d failed (%s), retrying in %ds …",
                                 attempt + 1, exc, wait)
                    time.sleep(wait)

        raise RuntimeError(
            f"Failed to load CLIP model '{self.CLIP_MODEL_ID}' after "
            f"{max_retries} attempts.\nLast error: {last_exc}\n"
            "Make sure you have an internet connection or the model is cached.")

    # ── MobileNetV3-Small ─────────────────────────────────────────────────

    def _load_mobilenet(self):
        try:
            self._mnv3 = MNV3FeatureExtractor(self.device).to(self.device).eval()
            # Confirm actual output dim by running a dummy forward
            with torch.no_grad():
                dummy = torch.zeros(1, 3, self.MNV3_IMG_SIZE, self.MNV3_IMG_SIZE,
                                    device=self.device)
                out = self._mnv3(dummy)
            self.mobilenet_dim = out.shape[1]
            _log.info("MobileNetV3-Small loaded  dim=%d", self.mobilenet_dim)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load MobileNetV3-Small.\nError: {exc}\n"
                "Make sure torchvision is installed: pip install torchvision")

        # Standard ImageNet preprocessing (matches torchvision MNV3 weights)
        self._mnv3_preproc = tv_transforms.Compose([
            tv_transforms.Resize(256,
                interpolation=tv_transforms.InterpolationMode.BILINEAR),
            tv_transforms.CenterCrop(self.MNV3_IMG_SIZE),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=self.MNV3_MEAN, std=self.MNV3_STD),
        ])

    # ── Encoding ─────────────────────────────────────────────────────────

    def encode_image(self, pil_image) -> tuple[np.ndarray, np.ndarray]:
        """
        PIL Image → (clip_emb, mobilenet_feats)
            clip_emb       : (512,)  float32, L2-normalised
            mobilenet_feats: (176,)  float32
        """
        # CLIP
        with torch.no_grad():
            clip_t = self._clip_preproc(pil_image).unsqueeze(0).to(self.device)
            clip_e = self._clip.encode_image(clip_t)
            clip_e = clip_e / clip_e.norm(dim=-1, keepdim=True)
        clip_np = clip_e.squeeze(0).cpu().float().numpy()

        # MobileNetV3
        with torch.no_grad():
            mnv3_t = self._mnv3_preproc(pil_image).unsqueeze(0).to(self.device)
            mnv3_e = self._mnv3(mnv3_t)
        mnv3_np = mnv3_e.squeeze(0).cpu().float().numpy()

        return clip_np, mnv3_np

    def encode_batch(self, pil_images: list) -> tuple[np.ndarray, np.ndarray]:
        """List of PIL Images → (clips (N,512), mobilenets (N,176)) float32."""
        clips, mnvs = zip(*[self.encode_image(img) for img in pil_images])
        return np.stack(clips), np.stack(mnvs)

    # ── BOLD predictor ────────────────────────────────────────────────────

    def _load_bold(self, ckpt_path):
        if not ckpt_path or not Path(ckpt_path).exists():
            raise FileNotFoundError(
                f"BOLD predictor checkpoint not found at: {ckpt_path}\n"
                "Expected: model/best_model.pt\n"
                "Set the BOLD_CKPT environment variable to override the path.")
        try:
            ckpt       = torch.load(ckpt_path, map_location=self.device)
            cfg        = ckpt["cfg"]
            self._bold = CalibratedBoldPredictor(cfg)
            self._bold.load_state_dict(ckpt["model_state"])
            self._bold = self._bold.to(self.device).eval()
            _log.info("BOLD predictor loaded from %s  (epoch %d)",
                      ckpt_path, ckpt.get("epoch", -1))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load BOLD checkpoint from {ckpt_path}.\n"
                f"Error: {exc}")

    def predict_bold(self,
                     new_clip:         np.ndarray,
                     new_mobilenet:    np.ndarray,
                     calib_clips:      np.ndarray,
                     calib_mobilenets: np.ndarray,
                     calib_scores:     np.ndarray) -> np.ndarray:
        """
        Predict BOLD activations for one new image.
        new_clip         : (512,)
        new_mobilenet    : (176,)
        calib_clips      : (N, 512)
        calib_mobilenets : (N, 176)
        calib_scores     : (N,)
        Returns          : (1024,) float32
        """
        N  = len(calib_clips)
        tc  = torch.FloatTensor(new_clip).unsqueeze(0).to(self.device)
        tmn = torch.FloatTensor(new_mobilenet).unsqueeze(0).to(self.device)
        cc  = torch.FloatTensor(calib_clips).unsqueeze(0).to(self.device)
        cmn = torch.FloatTensor(calib_mobilenets).unsqueeze(0).to(self.device)
        cs  = torch.FloatTensor(calib_scores).view(1, N, 1).to(self.device)
        pm  = torch.zeros(1, N, dtype=torch.bool).to(self.device)

        with torch.no_grad():
            pred = self._bold(tc, tmn, cc, cmn, cs, pm)
        return pred.squeeze(0).cpu().numpy()

    # ── Normalization stats ────────────────────────────────────────────────

    def set_normalization_stats(self,
                                yeo_min: Optional[np.ndarray],
                                yeo_max: Optional[np.ndarray]):
        """
        Provide per-network min/max from Notebook C's training dataset so that
        aesthetic_score() can normalise raw Yeo means to [0, 1] before computing
        the weighted score.

        Call with the yeo_min / yeo_max arrays stored in best_denoiser.pt:
            hub.set_normalization_stats(
                np.array(ckpt["yeo_min"]).flatten(),
                np.array(ckpt["yeo_max"]).flatten(),
            )

        Pass (None, None) to revert to raw z-score mode.
        """
        if yeo_min is None or yeo_max is None:
            self.yeo_min = None
            self.yeo_max = None
            _log.info("Normalization stats cleared — aesthetic_score uses raw z-scores.")
        else:
            self.yeo_min = np.array(yeo_min, dtype=np.float32).flatten()
            self.yeo_max = np.array(yeo_max, dtype=np.float32).flatten()
            _log.info("Normalization stats set from Notebook C checkpoint.")

    # ── Yeo-7 means ───────────────────────────────────────────────────────

    def yeo_means(self, bold: np.ndarray) -> np.ndarray:
        """
        Compute per-network mean activation from a (1024,) BOLD vector.
        Returns float32 array of shape (7,) in raw BOLD (z-score) units.
        Network order: [VIS, SMN, DAN, VAN, LIM, FPN, DMN] (network IDs 1-7).
        """
        means = np.zeros(7, dtype=np.float32)
        for j, nid in enumerate([1, 2, 3, 4, 5, 6, 7]):
            mask = self.yeo_assign == nid
            if mask.sum() > 0:
                means[j] = float(bold[mask].mean())
        return means

    def normalized_yeo_means(self, bold: np.ndarray) -> np.ndarray:
        """
        Compute per-network means and normalise to [0, 1] using the Notebook C
        training statistics.  If normalization stats are not loaded, returns
        a sigmoid-squashed version of the raw means instead (graceful fallback).
        Returns float32 array of shape (7,).
        """
        raw = self.yeo_means(bold)
        if self.yeo_min is not None and self.yeo_max is not None:
            normed = (raw - self.yeo_min) / (self.yeo_max - self.yeo_min + 1e-8)
            return np.clip(normed, 0.0, 1.0).astype(np.float32)
        else:
            # Fallback: sigmoid so display values are in (0,1) even without stats
            return (1.0 / (1.0 + np.exp(-raw * 0.8))).astype(np.float32)

    # ── Aesthetic score ───────────────────────────────────────────────────

    def aesthetic_score(self, bold: np.ndarray) -> float:
        """
        Weighted combination of Limbic (50%) + DMN (35%) + Visual (15%).
        Matches Notebook C's _aesthetic_score_np() exactly.

        If normalization stats are loaded (via set_normalization_stats), Yeo
        means are min-max normalised to [0, 1] before weighting — this is the
        same transformation applied during Notebook C training, so the returned
        score is directly comparable to training scores (expected ≈ 0.5 mean,
        range ≈ 0.2–0.85).

        Without stats, the raw z-scored Yeo means are used — values will be
        near 0 and lack the [0,1] reference frame the diffusion model expects,
        but relative rankings between candidates remain valid.
        """
        normed = self.normalized_yeo_means(bold)
        # normed indices: 0=VIS, 1=SMN, 2=DAN, 3=VAN, 4=LIM, 5=FPN, 6=DMN
        limbic = float(normed[4])
        dmn    = float(normed[6])
        visual = float(normed[0])
        return float(0.50 * limbic + 0.35 * dmn + 0.15 * visual)