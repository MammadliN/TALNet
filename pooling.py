"""PyTorch pooling operators: AutoPool, PowerPool, and BetaExpPool."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_feature_param(name: str, module: nn.Module, feature_dim: int, init: str) -> nn.Parameter:
    """Create a learnable parameter of shape (1, feature_dim) with init {zeros, ones}."""
    if init not in {"zeros", "ones"}:
        raise ValueError(f"{name}: init must be 'zeros' or 'ones', got {init!r}")
    if init == "zeros":
        p = nn.Parameter(torch.zeros(1, feature_dim))
    else:
        p = nn.Parameter(torch.ones(1, feature_dim))
    module.register_parameter(name, p)
    return p


def _broadcast_feature_param(p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Broadcast a (1, C) parameter to x.shape for elementwise ops with feature dim last."""
    view_shape = [1] * (x.ndim - 1) + [x.size(-1)]
    return p.view(*view_shape)


def _apply_mask_to_logits(
    logits: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Mask logits so that masked positions receive ~-inf before softmax."""
    if mask is None:
        return logits
    if mask.dtype != torch.bool:
        mask = mask.bool()
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


def _masked_weighted_sum(
    x: torch.Tensor,
    weights: torch.Tensor,
    axis: int,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Compute sum(x * weights) along axis, optionally masking (zeroing masked positions)."""
    if mask is not None:
        if mask.dtype != torch.bool:
            mask = mask.bool()
        x = x.masked_fill(~mask, 0.0)
        weights = weights.masked_fill(~mask, 0.0)
    return torch.sum(x * weights, dim=axis)


class AutoPool1D(nn.Module):
    """Learnable softmax pooling: weights = softmax(alpha * x)."""

    def __init__(self, axis: int = 1, init: str = "zeros") -> None:
        super().__init__()
        self.axis = axis
        self.init = init
        self.alpha: Optional[nn.Parameter] = None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError("AutoPool1D expects x.ndim >= 2")
        if self.alpha is None:
            self.alpha = _make_feature_param("alpha", self, x.size(-1), self.init)

        alpha_b = _broadcast_feature_param(self.alpha, x)
        logits = _apply_mask_to_logits(alpha_b * x, mask)
        weights = F.softmax(logits, dim=self.axis)
        return _masked_weighted_sum(x, weights, axis=self.axis, mask=mask)


class PowerPool1D(nn.Module):
    """Power pooling: sum(x^(n+1))/sum(x^n) with learnable n >= 0."""

    def __init__(self, axis: int = 1, init_n: float = 1.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.axis = axis
        self.eps = eps
        self.n_raw: Optional[nn.Parameter] = None
        self.init_n = float(init_n)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError("PowerPool1D expects x.ndim >= 2")
        if self.n_raw is None:
            n0 = torch.tensor(self.init_n)
            u0 = torch.log(torch.expm1(n0)).clamp_min(-20.0)
            self.n_raw = _make_feature_param("n_raw", self, x.size(-1), init="zeros")
            with torch.no_grad():
                self.n_raw.fill_(u0.item())

        n = F.softplus(self.n_raw)
        n_b = _broadcast_feature_param(n, x)
        x_pos = x.clamp_min(self.eps)
        w = torch.pow(x_pos, n_b)
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.bool()
            w = w.masked_fill(~mask, 0.0)
            x_pos = x_pos.masked_fill(~mask, 0.0)

        num = torch.sum(x_pos * w, dim=self.axis)
        den = torch.sum(w, dim=self.axis).clamp_min(self.eps)
        return num / den


class BetaExpPool1D(nn.Module):
    """Pooling with weights y^beta * exp(alpha * y), alpha,beta >= 0."""

    def __init__(
        self,
        axis: int = 1,
        init_alpha: float = 0.0,
        init_beta: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.axis = axis
        self.eps = eps
        self.init_alpha = float(init_alpha)
        self.init_beta = float(init_beta)
        self.alpha_raw: Optional[nn.Parameter] = None
        self.beta_raw: Optional[nn.Parameter] = None

    @staticmethod
    def _inv_softplus(v: float) -> float:
        t = torch.tensor(v)
        u = torch.log(torch.expm1(t)).clamp_min(-20.0)
        return float(u.item())

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError("BetaExpPool1D expects x.ndim >= 2")
        if self.alpha_raw is None or self.beta_raw is None:
            feature_dim = x.size(-1)
            self.alpha_raw = _make_feature_param("alpha_raw", self, feature_dim, init="zeros")
            self.beta_raw = _make_feature_param("beta_raw", self, feature_dim, init="zeros")
            with torch.no_grad():
                self.alpha_raw.fill_(self._inv_softplus(self.init_alpha))
                self.beta_raw.fill_(self._inv_softplus(self.init_beta))

        alpha = F.softplus(self.alpha_raw)
        beta = F.softplus(self.beta_raw)
        alpha_b = _broadcast_feature_param(alpha, x)
        beta_b = _broadcast_feature_param(beta, x)
        x_pos = x.clamp_min(self.eps)
        w = torch.pow(x_pos, beta_b) * torch.exp(alpha_b * x_pos)
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.bool()
            w = w.masked_fill(~mask, 0.0)
            x_pos = x_pos.masked_fill(~mask, 0.0)

        num = torch.sum(x_pos * w, dim=self.axis)
        den = torch.sum(w, dim=self.axis).clamp_min(self.eps)
        return num / den
