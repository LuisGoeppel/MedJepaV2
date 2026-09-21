"""LeJEPA invariance and SIGReg objectives, including distributed variants."""

from __future__ import annotations
import math
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from .config import TrainConfig
from .training_utils import get_world_size, is_distributed


def invariance_loss(proj: torch.Tensor) -> torch.Tensor:
    # proj: [B, V, D]. Make all views of the same image agree.
    return (proj.mean(dim=1, keepdim=True) - proj).square().mean()


def normalize_projection_for_loss(proj: torch.Tensor, mode: str = "sqrt_dim") -> torch.Tensor:
    """Normalize projector output before the SSL loss.

    - none: old behavior.
    - unit: unit L2 norm per projected view.
    - sqrt_dim: unit L2 norm scaled by sqrt(D), matching a standard-normal
      per-vector norm scale more closely for SIGReg's target distribution.
    """
    if mode == "none":
        return proj
    z = F.normalize(proj, dim=-1, eps=1e-6)
    if mode == "sqrt_dim":
        z = z * math.sqrt(float(proj.size(-1)))
    elif mode != "unit":
        raise ValueError(f"Unknown projection_normalization mode: {mode}")
    return z


class PooledViewsSIGReg(nn.Module):
    """Stable v6 pooled-view SIGReg ablation.

    Input: proj [B, V, D].
    It regularizes the pooled local set [B*V, D].
    """

    def __init__(self, knots: int = 17, num_projections: int = 256):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.num_projections = int(num_projections)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        z = proj.reshape(-1, proj.size(-1))  # [B*V, D]
        dim = z.size(-1)
        n = z.size(0)
        A = torch.randn(dim, self.num_projections, device=z.device, dtype=z.dtype)
        A = A / A.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12)
        x_t = (z @ A).unsqueeze(-1) * self.t.to(z.device, dtype=z.dtype)  # [N, P, K]
        err = (x_t.cos().mean(dim=0) - self.phi.to(z.device, dtype=z.dtype)).square()
        err = err + x_t.sin().mean(dim=0).square()
        statistic = (err @ self.weights.to(z.device, dtype=z.dtype)) * n
        return statistic.mean()


class AuthorDDPPooledViewsSIGReg(nn.Module):
    """DDP/global ECF version of the stable pooled-view SIGReg.

    Input: proj [B, V, D].

    This keeps the empirically successful pooled-view idea, but uses the authors'
    DDP communication pattern: compute an empirical characteristic function locally,
    all-reduce the ECF across ranks, and scale by the global number of projected
    vectors B * V * world_size.

    This is useful as a bridge between:
      - pooled_views: old v6 local pooled regularizer that worked empirically
      - author_ddp_per_view: reference-style per-view regularizer that produced
        cramped encoder embeddings in our Hologic/Lorad experiments
    """

    def __init__(self, knots: int = 17, num_projections: int = 1024, normalize_by_n: bool = False):
        super().__init__()
        self.num_projections = int(num_projections)
        self.normalize_by_n = bool(normalize_by_n)

        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        if proj.ndim != 3:
            raise ValueError(f"AuthorDDPPooledViewsSIGReg expects proj [B,V,D], got shape {tuple(proj.shape)}")

        z = proj.reshape(-1, proj.size(-1))  # local [B*V, D]
        dim = z.size(-1)
        local_n = int(z.size(0))
        world_size = get_world_size()

        A = torch.randn(dim, self.num_projections, device=z.device, dtype=z.dtype)
        if is_distributed():
            dist.broadcast(A, src=0)
        A = A / A.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12)

        t = self.t.to(z.device, dtype=z.dtype)
        phi = self.phi.to(z.device, dtype=z.dtype)
        weights = self.weights.to(z.device, dtype=z.dtype)

        x_t = (z @ A).unsqueeze(-1) * t  # [N, P, K]
        ecf_real = x_t.cos().mean(dim=0)
        ecf_imag = x_t.sin().mean(dim=0)
        ecf = torch.stack((ecf_real, ecf_imag), dim=0)  # [2, P, K]

        if is_distributed():
            dist.all_reduce(ecf, op=dist.ReduceOp.AVG)

        err = (ecf[0] - phi).square() + ecf[1].square()  # [P, K]
        if self.normalize_by_n:
            statistic = err @ weights
        else:
            statistic = (err @ weights) * local_n * world_size
        return statistic.mean()


class AuthorDDPPerViewSIGReg(nn.Module):
    """Author-style DDP-compatible per-view SIGReg.

    Input: proj [V, B, D].

    It computes empirical characteristic functions over the batch dimension,
    all-reduces these ECF statistics across DDP ranks, and scales the statistic
    by the global batch size B * world_size.

    Note: all ranks must use the same random projection matrix A for the ECF
    coordinates to be compatible. To make this independent of RNG state, v7
    broadcasts A from rank 0 before computing the ECF.
    """

    def __init__(self, knots: int = 17, num_projections: int = 1024, normalize_by_n: bool = False):
        super().__init__()
        self.num_projections = int(num_projections)
        self.normalize_by_n = bool(normalize_by_n)

        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        if proj.ndim != 3:
            raise ValueError(f"AuthorDDPPerViewSIGReg expects proj [V,B,D], got shape {tuple(proj.shape)}")

        dim = proj.size(-1)
        local_b = int(proj.size(-2))
        world_size = get_world_size()

        A = torch.randn(dim, self.num_projections, device=proj.device, dtype=proj.dtype)
        if is_distributed():
            # Rank 0 projection coordinates define the shared sketch basis.
            dist.broadcast(A, src=0)
        A = A / A.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12)

        t = self.t.to(proj.device, dtype=proj.dtype)
        phi = self.phi.to(proj.device, dtype=proj.dtype)
        weights = self.weights.to(proj.device, dtype=proj.dtype)

        # [V,B,D] @ [D,P] -> [V,B,P], then [V,B,P,K].
        x_t = (proj @ A).unsqueeze(-1) * t

        # Mean over batch dimension B. For [V,B,P,K], dim=-3 is B.
        ecf_real = x_t.cos().mean(dim=-3)
        ecf_imag = x_t.sin().mean(dim=-3)
        ecf = torch.stack((ecf_real, ecf_imag), dim=0)  # [2,V,P,K]

        if is_distributed():
            dist.all_reduce(ecf, op=dist.ReduceOp.AVG)

        err = (ecf[0] - phi).square() + ecf[1].square()  # [V,P,K]
        if self.normalize_by_n:
            statistic = err @ weights
        else:
            statistic = (err @ weights) * local_b * world_size
        return statistic.mean()


class LeJEPALoss(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.lambda_sigreg = float(cfg.lambda_sigreg)
        self.projection_normalization = cfg.projection_normalization
        self.sigreg_mode = cfg.sigreg_mode

        if cfg.sigreg_mode == "author_ddp_per_view":
            self.sigreg = AuthorDDPPerViewSIGReg(
                knots=cfg.sigreg_knots,
                num_projections=cfg.sigreg_num_projections,
                normalize_by_n=cfg.sigreg_normalize_by_n,
            )
        elif cfg.sigreg_mode == "author_ddp_pooled_views":
            self.sigreg = AuthorDDPPooledViewsSIGReg(
                knots=cfg.sigreg_knots,
                num_projections=cfg.sigreg_num_projections,
                normalize_by_n=cfg.sigreg_normalize_by_n,
            )
        elif cfg.sigreg_mode == "pooled_views":
            self.sigreg = PooledViewsSIGReg(
                knots=cfg.sigreg_knots,
                num_projections=cfg.sigreg_num_projections,
            )
        else:
            raise ValueError(f"Unknown SIGReg mode: {cfg.sigreg_mode}")

    def forward(self, proj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # v5/v6 stable behavior:
        # - invariance on normalized projections
        # - SIGReg on raw projections
        proj_inv = normalize_projection_for_loss(proj, self.projection_normalization)
        inv = invariance_loss(proj_inv)

        if self.sigreg_mode == "author_ddp_per_view":
            # Network returns [B,V,D]; author code expects [V,B,D].
            sig = self.sigreg(proj.permute(1, 0, 2).contiguous())
        else:
            sig = self.sigreg(proj)

        total = sig * self.lambda_sigreg + inv * (1.0 - self.lambda_sigreg)
        return total, inv, sig

    def normalize_for_diagnostics(self, proj: torch.Tensor) -> torch.Tensor:
        return normalize_projection_for_loss(proj, self.projection_normalization)
