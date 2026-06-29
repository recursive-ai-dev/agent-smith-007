"""
AdamW Optimizer — Adaptive Moment Estimation with Weight Decay
==============================================================
Pure-Python implementation for NanoTensor.

Algorithm:
  m_t = β₁ m_{t-1} + (1 - β₁) g_t
  v_t = β₂ v_{t-1} + (1 - β₂) g_t²
  m̂_t = m_t / (1 - β₁ᵗ)
  v̂_t = v_t / (1 - β₂ᵗ)
  θ_t = θ_t - η (m̂_t / (√v̂_t + ε) + λ θ_t)
"""

import math
import logging
from typing import List, Dict, Optional, Any

from ..tensor import NanoTensor

logger = logging.getLogger(__name__)


class AdamOptimizer:
    """
    AdamW optimizer with built-in warmup and gradient clipping.
    """

    def __init__(
        self,
        params: List[NanoTensor],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        max_grad_norm: Optional[float] = 1.0,
    ):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm

        self.t = 0
        # State: m and v for each parameter
        self.m: List[List[float]] = [[0.0] * len(p.data) for p in self.params]
        self.v: List[List[float]] = [[0.0] * len(p.data) for p in self.params]

    def _get_lr(self) -> float:
        """Apply linear warmup if configured."""
        if self.t < self.warmup_steps:
            return self.lr * (self.t + 1) / self.warmup_steps
        return self.lr

    def clip_gradients(self):
        """Global gradient clipping by norm."""
        if self.max_grad_norm is None:
            return

        total_norm_sq = 0.0
        for p in self.params:
            if p.grad:
                total_norm_sq += sum(g * g for g in p.grad)

        total_norm = math.sqrt(total_norm_sq)
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)

        if clip_coef < 1.0:
            for p in self.params:
                if p.grad:
                    for i in range(len(p.grad)):
                        p.grad[i] *= clip_coef

    def step(self):
        """Perform a single optimization step."""
        self.t += 1
        lr = self._get_lr()

        # 1. Gradient clipping
        self.clip_gradients()

        for i, p in enumerate(self.params):
            if not p.grad:
                continue

            for j in range(len(p.data)):
                g = p.grad[j]

                # 2. Update moments
                self.m[i][j] = self.beta1 * self.m[i][j] + (1.0 - self.beta1) * g
                self.v[i][j] = self.beta2 * self.v[i][j] + (1.0 - self.beta2) * (g * g)

                # 3. Bias correction
                m_hat = self.m[i][j] / (1.0 - self.beta1 ** self.t)
                v_hat = self.v[i][j] / (1.0 - self.beta2 ** self.t)

                # 4. Weight decay and parameter update
                update = m_hat / (math.sqrt(v_hat) + self.eps)
                p.data[j] -= lr * (update + self.weight_decay * p.data[j])

    def zero_grad(self):
        """Reset gradients for all parameters."""
        for p in self.params:
            p.zero_grad()

    def state_dict(self) -> Dict[str, Any]:
        """Snapshot optimizer state."""
        return {
            "t": self.t,
            "m": [list(mi) for mi in self.m],
            "v": [list(vi) for vi in self.v],
            "lr": self.lr,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Restore optimizer state."""
        self.t = state["t"]
        self.lr = state["lr"]
        for i in range(len(self.params)):
            self.m[i] = list(state["m"][i])
            self.v[i] = list(state["v"][i])
