"""
Adam Optimiser (AdamW variant)
==============================
Implements the full Adam algorithm with:
  • Per-parameter first-moment (m̂) and second-moment (v̂) estimates
  • Bias-correction at every step:  m̂_t = m_t / (1 − β₁^t)
                                     v̂_t = v_t / (1 − β₂^t)
  • Decoupled weight decay (AdamW):  θ ← θ − λ θ  applied *before* the
    gradient step, independent of the adaptive scaling
  • Global gradient norm clipping before stepping
  • Linear warm-up schedule for learning rate

Mathematics (per parameter θ with gradient g):
    m_t  = β₁ m_{t−1} + (1−β₁) g_t
    v_t  = β₂ v_{t−1} + (1−β₂) g_t²
    m̂_t = m_t  / (1 − β₁^t)
    v̂_t = v_t  / (1 − β₂^t)
    θ_t  = θ_{t−1} − α_t [m̂_t / (√v̂_t + ε) + λ θ_{t−1}]

Reference: Kingma & Ba 2015 (Adam); Loshchilov & Hutter 2019 (AdamW).
"""

import math
from typing import List, Dict, Any

from ..tensor import NanoTensor


class AdamOptimizer:
    """
    Adam / AdamW optimiser operating directly on NanoTensor parameters.

    Parameters
    ----------
    params        : list of NanoTensor (all learnable parameters of the model)
    lr            : base learning rate α
    beta1         : exponential decay for first moment  (default 0.9)
    beta2         : exponential decay for second moment (default 0.999)
    eps           : numerical stability term (default 1e-8)
    weight_decay  : decoupled L2 coefficient λ (default 0.01)
    warmup_steps  : number of steps over which LR linearly ramps from 0 to lr
    grad_clip     : maximum global gradient L2 norm (0 = no clipping)
    """

    def __init__(
        self,
        params: List[NanoTensor],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        grad_clip: float = 1.0,
    ):
        self.params       = [p for p in params if p.requires_grad]
        self.lr           = lr
        self.beta1        = beta1
        self.beta2        = beta2
        self.eps          = eps
        self.weight_decay = weight_decay
        self.warmup_steps = max(warmup_steps, 1)
        self.grad_clip    = grad_clip

        self.step_count: int = 0

        # First and second moment estimates, one list per parameter
        self._m: List[List[float]] = [
            [0.0] * len(p.data) for p in self.params
        ]
        self._v: List[List[float]] = [
            [0.0] * len(p.data) for p in self.params
        ]

    # ── Utilities ────────────────────────────────────────────────────────

    def _current_lr(self) -> float:
        """Linear warm-up: lr scales from 0 → lr over warmup_steps."""
        if self.step_count <= self.warmup_steps:
            return self.lr * (self.step_count / self.warmup_steps)
        return self.lr

    def global_grad_norm(self) -> float:
        """L2 norm over all parameter gradients: ‖g‖₂."""
        sq_sum = 0.0
        for p in self.params:
            if p.grad:
                sq_sum += sum(g * g for g in p.grad)
        return math.sqrt(sq_sum)

    def clip_gradients(self) -> float:
        """
        In-place global gradient norm clipping.
        Scales all gradients by min(1, grad_clip / ‖g‖₂).
        Returns the pre-clip norm (useful for logging).
        """
        norm = self.global_grad_norm()
        if self.grad_clip > 0 and norm > self.grad_clip:
            scale = self.grad_clip / (norm + 1e-12)
            for p in self.params:
                if p.grad:
                    p.grad = [g * scale for g in p.grad]
        return norm

    # ── Core step ────────────────────────────────────────────────────────

    def step(self) -> Dict[str, Any]:
        """
        Execute one Adam update step.

        Returns a dict of diagnostic scalars:
          grad_norm     : pre-clip gradient L2 norm
          effective_lr  : learning rate after warm-up schedule
          max_update    : maximum absolute parameter change in this step
        """
        self.step_count += 1
        alpha = self._current_lr()

        # Gradient clipping
        grad_norm = self.clip_gradients()

        # Bias-correction denominators (computed once per step)
        bc1 = 1.0 - self.beta1 ** self.step_count
        bc2 = 1.0 - self.beta2 ** self.step_count

        max_update = 0.0

        for idx, p in enumerate(self.params):
            if p.grad is None:
                continue

            m = self._m[idx]
            v = self._v[idx]
            n = len(p.data)

            for i in range(n):
                g = p.grad[i]

                # ── First moment:  m_t = β₁ m_{t−1} + (1−β₁) g
                m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * g

                # ── Second moment: v_t = β₂ v_{t−1} + (1−β₂) g²
                v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * g * g

                # ── Bias-corrected estimates
                m_hat = m[i] / bc1
                v_hat = v[i] / bc2

                # ── Adaptive step size
                adaptive = m_hat / (math.sqrt(v_hat) + self.eps)

                # ── Decoupled weight decay: θ ← θ − λ θ
                decay = self.weight_decay * p.data[i]

                # ── Parameter update: θ ← θ − α (adaptive + decay)
                delta = alpha * (adaptive + decay)
                p.data[i] -= delta
                max_update = max(max_update, abs(delta))

        return {
            "grad_norm":    grad_norm,
            "effective_lr": alpha,
            "max_update":   max_update,
            "step":         self.step_count,
        }

    def register_param(self, param: NanoTensor):
        """
        Register a new parameter after construction.
        Appends to self.params and initialises its moment vectors to zero.
        Call this whenever a new learnable NanoTensor is created (e.g. a
        newly registered GSAR symbol embedding) so that it receives updates.
        """
        if not param.requires_grad:
            return
        if any(p is param for p in self.params):
            return  # already registered — prevent duplicate updates
        self.params.append(param)
        self._m.append([0.0] * len(param.data))
        self._v.append([0.0] * len(param.data))

    def zero_grad(self):
        """Zero all parameter gradients (and Kahan error accumulators)."""
        for p in self.params:
            p.zero_grad()

    # ── Serialisation ─────────────────────────────────────────────────

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step_count": self.step_count,
            "m": [list(mi) for mi in self._m],
            "v": [list(vi) for vi in self._v],
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
        }

    def load_state_dict(self, sd: Dict[str, Any]):
        self.step_count   = sd["step_count"]
        self._m           = [list(mi) for mi in sd["m"]]
        self._v           = [list(vi) for vi in sd["v"]]
        self.lr           = sd["lr"]
        self.beta1        = sd["beta1"]
        self.beta2        = sd["beta2"]
        self.eps          = sd["eps"]
        self.weight_decay = sd["weight_decay"]
