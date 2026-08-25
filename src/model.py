"""SDFN: graph-conditioned latent diffusion for probabilistic forecasting."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────────────────
#  Utilities
# ────────────────────────────────────────────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal embedding for diffusion timestep *t*."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        emb = math.log(10_000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb                                       # (B, dim)


class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al., 2022).

    Normalises each sample along the time axis and stores statistics so
    the inverse transform can be applied to the model output.
    """

    def __init__(self, n_vars: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # No learnable affine — keeps things simple & avoids
        # discrepancies between normalising *x* and *y*.
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) — computes stats along T, stores for denorm."""
        self._mean = x.mean(dim=1, keepdim=True).detach()
        self._std = (x.var(dim=1, keepdim=True, unbiased=False)
                     + self.eps).sqrt().detach()
        return (x - self._mean) / self._std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, *, D) — reverses normalization."""
        return x * self._std + self._mean


# ────────────────────────────────────────────────────────────────────────────
#  Context Encoder
# ────────────────────────────────────────────────────────────────────────────

class ContextEncoder(nn.Module):
    """Encode historical time series into per-node context.

    Pipeline::

        x ∈ (B,T,D)
        → per-variable temporal Conv1D (shared weights)
        → pool over time
        → cross-variable multi-head self-attention
        → context ∈ (B,D,d_model)
    """

    def __init__(
        self,
        d_model: int = 128,
        n_temporal_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Per-variable temporal feature extraction (weight-shared)
        self.input_proj = nn.Linear(1, d_model)
        self.temporal_blocks = nn.ModuleList()
        for _ in range(n_temporal_layers):
            self.temporal_blocks.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=7, padding=3),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            ))
        self.temporal_norm = nn.LayerNorm(d_model)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        # Cross-variable attention
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads=n_heads, batch_first=True, dropout=dropout,
        )
        self.cross_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # --- temporal encoding (shared across variables) ----------------
        x_flat = x.transpose(1, 2).reshape(B * D, T, 1)    # (BD, T, 1)
        h = self.input_proj(x_flat)                          # (BD, T, dm)
        h = h.transpose(1, 2)                                # (BD, dm, T)
        for block in self.temporal_blocks:
            h = h + block(h)                                 # residual
        h = self.temporal_norm(h.transpose(1, 2))            # (BD, T, dm)
        h = h.transpose(1, 2)                                # (BD, dm, T)
        h = self.temporal_pool(h).squeeze(-1)                # (BD, dm)
        h = h.view(B, D, self.d_model)                       # (B, D, dm)

        # --- cross-variable attention -----------------------------------
        h_attn, _ = self.cross_attn(h, h, h)
        h = self.cross_norm(h + h_attn)
        h = self.ffn_norm(h + self.ffn(h))
        return h                                             # (B, D, dm)


# ────────────────────────────────────────────────────────────────────────────
#  Target Encoder / Decoder  (latent per-node representation of the future)
# ────────────────────────────────────────────────────────────────────────────

class TargetEncoder(nn.Module):
    """Compress forecast target  y ∈ (B,H,D) → z₀ ∈ (B,D,d_latent).

    Each variable's H-step future is independently mapped to d_latent
    via a shared MLP.
    """

    def __init__(self, horizon: int, d_latent: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(horizon, d_latent * 2),
            nn.GELU(),
            nn.Linear(d_latent * 2, d_latent),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y.transpose(1, 2))               # (B, D, d_latent)


class TargetDecoder(nn.Module):
    """Expand latent  z₀ ∈ (B,D,d_latent) → ŷ ∈ (B,H,D)."""

    def __init__(self, d_latent: int, horizon: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent, d_latent * 2),
            nn.GELU(),
            nn.Linear(d_latent * 2, horizon),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).transpose(1, 2)               # (B, H, D)


# ────────────────────────────────────────────────────────────────────────────
#  Graph Learner
# ────────────────────────────────────────────────────────────────────────────

class GraphLearner(nn.Module):
    """Learn dynamic adjacency A ∈ (B,D,D) from per-node context.

    * Scaled dot-product between learned query/key projections
    * Top-k sparsification per row  (symmetric mask)
    * Optional integration with a pre-computed static adjacency
      (e.g. road-network graph for METR-LA / PEMS-BAY)
    """

    def __init__(
        self,
        d_model: int,
        n_vars: int,
        top_k: int = 20,
        static_adj: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.top_k = min(top_k, n_vars - 1)

        d_k = d_model // 2
        self.query = nn.Linear(d_model, d_k)
        self.key   = nn.Linear(d_model, d_k)

        if static_adj is not None:
            self.register_buffer("static_adj", static_adj)
            self.mix = nn.Parameter(torch.tensor(0.0))   # sigmoid → 0.5
        else:
            self.static_adj = None

    def forward(self, node_emb: torch.Tensor) -> torch.Tensor:
        """node_emb: (B, D, d_model) → A: (B, D, D)."""
        Q = self.query(node_emb)                          # (B, D, dk)
        K = self.key(node_emb)
        d_k = Q.shape[-1]
        eye = torch.eye(self.n_vars, device=node_emb.device)

        A_logits = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
        A_logits = (A_logits + A_logits.transpose(1, 2)) / 2   # symmetry
        # Exclude self-connections before top-k selection so each row keeps
        # up to k inter-node edges.
        A_logits = A_logits.masked_fill(eye.unsqueeze(0).bool(), -1e9)

        # top-k sparsification
        if self.top_k < self.n_vars - 1:
            _, topk_idx = torch.topk(A_logits, self.top_k, dim=-1)
            mask = torch.zeros_like(A_logits).scatter_(-1, topk_idx, 1.0)
            mask = ((mask + mask.transpose(1, 2)) > 0).float()
            A_logits = A_logits * mask + (1 - mask) * (-1e9)

        A = torch.sigmoid(A_logits)
        # remove self-loops
        A = A * (1.0 - eye)

        # mix with static adjacency when available
        if self.static_adj is not None:
            w = torch.sigmoid(self.mix)
            static = self.static_adj.unsqueeze(0).expand_as(A)
            static = static * (1.0 - eye)
            A = w * static + (1 - w) * A
            A = A * (1.0 - eye)

        return A


# ────────────────────────────────────────────────────────────────────────────
#  Scalable Laplacian-Conditioned Noise
# ────────────────────────────────────────────────────────────────────────────

class ScalableLaplacianNoise(nn.Module):
    r"""Sample noise from a graph-Laplacian-conditioned Gaussian:

    .. math::

        \varepsilon \sim \mathcal{N}\!\bigl(0,\;(L_G + \lambda I)^{-1}\bigr)

    Uses **eigen-decomposition** instead of Cholesky-of-inverse, giving
    stable :math:`O(D^2 k)` sampling after an :math:`O(D^3)` pre-compute:

    .. math::

        L_{\text{reg}} = U\Lambda U^\top
        \;\;\Rightarrow\;\;
        \varepsilon = U\,\mathrm{diag}(1/\!\sqrt{\lambda_i})\,z,
        \quad z\sim\mathcal{N}(0,I)

    For large *D* (> ``max_rank``), only the *k* smallest Laplacian
    eigenvalues (= largest covariance modes) are kept, giving a low-rank
    approximation that dramatically cuts cost.
    """

    def __init__(self, lambda_reg: float = 0.1,
                 max_rank: Optional[int] = None):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.max_rank = max_rank

    def forward(self, A: torch.Tensor, d_feat: int) -> torch.Tensor:
        """
        Args:
            A:      (B, D, D) adjacency matrix
            d_feat: independent feature dimensions to sample per node
        Returns:
            eps:    (B, D, d_feat) structured noise
        """
        B, D, _ = A.shape
        device = A.device

        # Graph Laplacian  L = D_deg − A
        deg = A.sum(dim=-1)
        L = torch.diag_embed(deg) - A
        L_reg = L + self.lambda_reg * torch.eye(D, device=device)

        # Batched symmetric eigen-decomposition (ascending order)
        eigenvalues, eigvecs = torch.linalg.eigh(L_reg)  # (B,D), (B,D,D)
        eigenvalues = eigenvalues.clamp(min=1e-6)

        # optional low-rank truncation
        k = D
        if self.max_rank is not None and self.max_rank < D:
            k = self.max_rank
            eigenvalues = eigenvalues[:, :k]
            eigvecs = eigvecs[:, :, :k]

        # ε = U diag(1/√λ) z ,  z ~ N(0,I)
        inv_sqrt = (1.0 / eigenvalues).sqrt()           # (B, k)
        z = torch.randn(B, k, d_feat, device=device)
        scaled = z * inv_sqrt.unsqueeze(-1)              # (B, k, d_feat)
        eps = torch.bmm(eigvecs, scaled)                 # (B, D, d_feat)
        return eps

    @staticmethod
    def standard_noise(B: int, D: int, d_feat: int,
                       device: torch.device) -> torch.Tensor:
        """Fallback isotropic Gaussian (for debugging / ablation)."""
        return torch.randn(B, D, d_feat, device=device)


# ────────────────────────────────────────────────────────────────────────────
#  GNN Denoiser
# ────────────────────────────────────────────────────────────────────────────

class GraphConvBlock(nn.Module):
    """Message-passing GCN layer with residual + LayerNorm.

    .. math::

        h' = \\text{Norm}\\bigl(h + \\sigma(\\bar A\\,W_{\\text{msg}}h
                                              + W_{\\text{self}}h)\\bigr)
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.W_msg  = nn.Linear(d_model, d_model)
        self.W_self = nn.Linear(d_model, d_model)
        self.norm   = nn.LayerNorm(d_model)
        self.drop   = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor,
                A_norm: torch.Tensor) -> torch.Tensor:
        """h: (B,D,dm), A_norm: (B,D,D) row-normalised adjacency."""
        h_msg  = torch.bmm(A_norm, self.W_msg(h))
        h_self = self.W_self(h)
        return self.norm(self.drop(F.gelu(h_msg + h_self)))


class GNNDenoiser(nn.Module):
    """Multi-layer GNN noise predictor with FiLM time conditioning.

    Each layer applies::

        scale, shift = MLP(time_emb)
        h_cond = h * (1 + scale) + shift        # FiLM
        h = h + GraphConvBlock(h_cond, A)        # residual GCN
    """

    def __init__(
        self,
        d_latent: int,
        d_cond: int,
        d_model: int = 128,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # diffusion-time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

        # input projection: concat(z_t, context) → d_model
        self.input_proj = nn.Linear(d_latent + d_cond, d_model)

        # GNN + FiLM layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.ModuleDict({
                "gcn":        GraphConvBlock(d_model, dropout),
                "film_scale": nn.Linear(d_model, d_model),
                "film_shift": nn.Linear(d_model, d_model),
            })
            # init FiLM to identity transform (scale≈0, shift≈0)
            nn.init.zeros_(layer["film_scale"].weight)
            nn.init.zeros_(layer["film_scale"].bias)
            nn.init.zeros_(layer["film_shift"].weight)
            nn.init.zeros_(layer["film_shift"].bias)
            self.layers.append(layer)

        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_latent),
        )

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        A_norm: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_t:     (B, D, d_latent) noisy latent
            t:       (B,) diffusion timestep indices
            A_norm:  (B, D, D) row-normalised adjacency
            context: (B, D, d_cond) per-node context
        Returns:
            eps_pred: (B, D, d_latent)
        """
        t_emb = self.time_mlp(t)                          # (B, dm)

        h = self.input_proj(
            torch.cat([z_t, context], dim=-1)              # (B,D,d_lat+d_cond)
        )                                                  # (B, D, dm)

        for layer in self.layers:
            scale = layer["film_scale"](t_emb).unsqueeze(1)  # (B,1,dm)
            shift = layer["film_shift"](t_emb).unsqueeze(1)
            h_cond = h * (1.0 + scale) + shift
            h = h + layer["gcn"](h_cond, A_norm)          # residual

        return self.out_proj(self.out_norm(h))


# ────────────────────────────────────────────────────────────────────────────
#  SDFN  (main model)
# ────────────────────────────────────────────────────────────────────────────

class SDFN(nn.Module):
    """Structural Diffusion Forecasting Networks.

    Parameters
    ----------
    n_vars : int
        Number of variables / sensor nodes *D*.
    horizon : int
        Forecast length *H*.
    d_model : int
        Hidden size for encoders & denoiser.
    d_latent : int
        Per-node latent dimension for diffusion.
    n_diff_steps : int
        Number of diffusion timesteps *T*.
    n_gnn_layers : int
        Depth of the GNN denoiser.
    top_k : int
        Sparsity of learned adjacency (edges per node).
    lambda_noise : float
        Regularisation added to graph Laplacian for numerical stability.
    lambda_recon : float
        Weight of the target auto-encoder reconstruction loss.
    lambda_sparse : float
        Weight of the L1 graph-sparsity penalty.
    lambda_smooth : float
        Weight of the Laplacian-smoothness penalty on latent z₀.
    max_rank : int | None
        Low-rank approximation rank for Laplacian noise sampling.
        ``None`` = full rank (exact).  Set e.g. 50 for *D* > 500.
    static_adj : Tensor | None
        Pre-computed adjacency *(D, D)* (e.g. road network).
    beta_schedule : str
        ``'linear'`` or ``'cosine'``.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        n_vars: int,
        horizon: int = 96,
        d_model: int = 128,
        d_latent: int = 64,
        n_diff_steps: int = 200,
        n_gnn_layers: int = 4,
        top_k: int = 20,
        lambda_noise: float = 0.1,
        lambda_recon: float = 1.0,
        lambda_sparse: float = 0.01,
        lambda_smooth: float = 0.1,
        max_rank: Optional[int] = None,
        static_adj: Optional[torch.Tensor] = None,
        beta_schedule: str = "cosine",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.horizon = horizon
        self.d_latent = d_latent
        self.n_diff_steps = n_diff_steps
        self.lambda_recon = lambda_recon
        self.lambda_sparse = lambda_sparse
        self.lambda_smooth = lambda_smooth

        # ── sub-modules ─────────────────────────────────────────────────
        self.revin = RevIN(n_vars)

        self.context_encoder = ContextEncoder(
            d_model=d_model, dropout=dropout,
        )
        self.target_encoder = TargetEncoder(
            horizon=horizon, d_latent=d_latent,
        )
        self.target_decoder = TargetDecoder(
            d_latent=d_latent, horizon=horizon,
        )
        self.graph_learner = GraphLearner(
            d_model=d_model, n_vars=n_vars, top_k=top_k,
            static_adj=static_adj,
        )
        self.noise_sampler = ScalableLaplacianNoise(
            lambda_reg=lambda_noise, max_rank=max_rank,
        )
        self.denoiser = GNNDenoiser(
            d_latent=d_latent, d_cond=d_model, d_model=d_model,
            n_layers=n_gnn_layers, dropout=dropout,
        )

        # ── diffusion schedule ──────────────────────────────────────────
        if beta_schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, n_diff_steps)
        elif beta_schedule == "cosine":
            betas = self._cosine_beta_schedule(n_diff_steps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)

    # ------------------------------------------------------------------ #
    #  Static helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cosine_beta_schedule(n_steps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule (Nichol & Dhariwal, 2021)."""
        t = torch.arange(n_steps + 1, dtype=torch.float64) / n_steps
        alpha_bar = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return betas.clamp(max=0.999).float()

    def _normalize_adj(self, A: torch.Tensor) -> torch.Tensor:
        """Row-normalise adjacency: Ā = D⁻¹A."""
        return A / (A.sum(dim=-1, keepdim=True) + 1e-6)

    # ------------------------------------------------------------------ #
    #  Training forward                                                    #
    # ------------------------------------------------------------------ #

    def forward(
        self, x: torch.Tensor, y: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass.

        Args:
            x: (B, T, D) historical input
            y: (B, H, D) forecast target

        Returns:
            Dict with keys ``total``, ``diffusion``, ``reconstruction``,
            ``sparse``, ``smooth``.
        """
        B = x.shape[0]
        device = x.device

        # ── normalise ───────────────────────────────────────────────────
        x = self.revin.normalize(x)
        y_norm = (y - self.revin._mean) / self.revin._std

        # ── encode context & learn graph ────────────────────────────────
        context = self.context_encoder(x)                 # (B,D,dm)
        A = self.graph_learner(context)                   # (B,D,D)
        A_norm = self._normalize_adj(A)

        # ── latent encoding of target ───────────────────────────────────
        z0 = self.target_encoder(y_norm)                  # (B,D,dl)

        # ── reconstruction loss (autoencoder pathway) ───────────────────
        y_recon = self.target_decoder(z0)                 # (B,H,D)
        loss_recon = F.mse_loss(y_recon, y_norm)

        # ── forward diffusion ───────────────────────────────────────────
        t = torch.randint(0, self.n_diff_steps, (B,), device=device)
        abar = self.alpha_cumprod[t].view(B, 1, 1)

        eps = self.noise_sampler(A, self.d_latent)        # (B,D,dl)
        z_t = abar.sqrt() * z0 + (1 - abar).sqrt() * eps

        # ── denoise ─────────────────────────────────────────────────────
        eps_pred = self.denoiser(z_t, t, A_norm, context)
        loss_diff = F.mse_loss(eps_pred, eps)

        # ── graph regularisation ────────────────────────────────────────
        # L1 sparsity
        loss_sparse = A.abs().mean()

        # Laplacian smoothness: tr(z₀ᵀ L z₀) — connected nodes ↔ similar latents
        L = torch.diag_embed(A.sum(-1)) - A               # (B,D,D)
        # (B. dl, D) @ (B, D, D) @ (B, D, dl) → trace of (B, dl, dl)
        smooth = torch.bmm(z0.transpose(1, 2), torch.bmm(L, z0))
        loss_smooth = smooth.diagonal(dim1=-2, dim2=-1).mean()

        # ── aggregate ──────────────────────────────────────────────────
        loss_total = (
            loss_diff
            + self.lambda_recon  * loss_recon
            + self.lambda_sparse * loss_sparse
            + self.lambda_smooth * loss_smooth
        )

        return {
            "total":          loss_total,
            "diffusion":      loss_diff.detach(),
            "reconstruction": loss_recon.detach(),
            "sparse":         loss_sparse.detach(),
            "smooth":         loss_smooth.detach(),
        }

    # ------------------------------------------------------------------ #
    #  Inference  (reverse diffusion)                                      #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        n_samples: int = 1,
        sample_steps: int | None = None,
        use_ddim: bool = False,
        ddim_eta: float = 0.0,
    ) -> torch.Tensor:
        """Generate forecasts via reverse diffusion.

        Args:
            x:            (B, T, D) history
            n_samples:    number of independent forecast draws
            sample_steps: reverse-process steps (default = n_diff_steps)
            use_ddim:     use DDIM (faster & optionally deterministic)
            ddim_eta:     DDIM stochasticity (0 = deterministic)

        Returns:
            (n_samples, B, H, D) if n_samples > 1 else (B, H, D)
        """
        self.eval()
        B, T, D = x.shape
        device = x.device

        x = self.revin.normalize(x)
        context = self.context_encoder(x)
        A = self.graph_learner(context)
        A_norm = self._normalize_adj(A)

        steps = self.n_diff_steps if sample_steps is None else min(
            sample_steps, self.n_diff_steps,
        )

        samples = []
        for _ in range(n_samples):
            z = torch.randn(B, D, self.d_latent, device=device)
            if use_ddim:
                z = self._ddim_sample(z, steps, A, A_norm, context, ddim_eta)
            else:
                z = self._ddpm_sample(z, steps, A, A_norm, context)

            y_hat = self.target_decoder(z)                # (B, H, D)
            y_hat = self.revin.denormalize(y_hat)
            samples.append(y_hat)

        if n_samples == 1:
            return samples[0]
        return torch.stack(samples, dim=0)

    # ---- reverse samplers -------------------------------------------

    def _ddpm_sample(
        self, z: torch.Tensor, steps: int,
        A: torch.Tensor, A_norm: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        for t_idx in reversed(range(steps)):
            B = z.shape[0]
            t = torch.full((B,), t_idx, device=z.device, dtype=torch.long)

            eps_pred = self.denoiser(z, t, A_norm, context)

            beta = self.betas[t_idx]
            alpha = self.alphas[t_idx]
            abar = self.alpha_cumprod[t_idx]

            z = (1.0 / alpha.sqrt()) * (
                z - (beta / (1 - abar).sqrt()) * eps_pred
            )

            if t_idx > 0:
                noise = self.noise_sampler(A, self.d_latent)
                z = z + beta.sqrt() * noise

        return z

    def _ddim_sample(
        self, z: torch.Tensor, steps: int,
        A: torch.Tensor, A_norm: torch.Tensor,
        context: torch.Tensor, eta: float = 0.0,
    ) -> torch.Tensor:
        # build sub-sequence of timesteps
        step_size = max(self.n_diff_steps // steps, 1)
        ts = list(range(0, self.n_diff_steps, step_size))[:steps]
        ts = list(reversed(ts))

        for i, t_idx in enumerate(ts):
            B = z.shape[0]
            t = torch.full((B,), t_idx, device=z.device, dtype=torch.long)

            eps_pred = self.denoiser(z, t, A_norm, context)

            abar_t = self.alpha_cumprod[t_idx]
            abar_prev = (
                self.alpha_cumprod[ts[i + 1]] if i + 1 < len(ts)
                else torch.tensor(1.0, device=z.device)
            )

            # predicted z₀
            z0_pred = (z - (1 - abar_t).sqrt() * eps_pred) / abar_t.sqrt()

            sigma = eta * (
                (1 - abar_prev) / (1 - abar_t) * (1 - abar_t / abar_prev)
            ).clamp(min=0).sqrt()

            dir_zt = (1 - abar_prev - sigma ** 2).clamp(min=0).sqrt() * eps_pred
            z = abar_prev.sqrt() * z0_pred + dir_zt

            if sigma > 0 and i + 1 < len(ts):
                noise = self.noise_sampler(A, self.d_latent)
                z = z + sigma * noise

        return z
