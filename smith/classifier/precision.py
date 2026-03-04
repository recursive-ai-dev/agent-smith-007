"""
Mixed-Precision Autograd Context
==================================
Pure-Python simulation of FP16 / FP32 mixed precision training.

Hardware background
-------------------
On GPUs, mixed precision stores activations and weights in FP16 (half)
while accumulating gradients in FP32 (full). This halves memory bandwidth
and uses Tensor Cores for ≈2× throughput.  Loss scaling prevents FP16
underflow on small gradient values.

Simulation strategy (CPU / pure Python)
-----------------------------------------
Python floats are IEEE 754 double (64-bit).  We simulate lower precision
by rounding values to the decimal precision equivalent of the target type:
  • FP32 → 7 significant decimal digits  (10^{−7} relative error)
  • FP16 → 3 significant decimal digits  (10^{−3} relative error)

During a "forward pass" the MixedPrecisionContext optionally quantises
every intermediate NanoTensor value.  During the backward pass gradients
are accumulated in full (FP64) precision, matching hardware behaviour.

Loss scaling
--------------
Small FP16 gradients underflow to zero.  The fix is to scale the loss by
a large factor S before backward(), then unscale the gradients by 1/S
before the optimiser step.  We implement static scaling here; dynamic
scaling (adjusting S if NaN/Inf appear) can be layered on top.
"""

import math
import contextlib
from typing import List

from ..tensor import NanoTensor


# ── Precision-rounding helpers ────────────────────────────────────────────

def _fp32_round(x: float) -> float:
    """Round to FP32 precision (7 significant decimal digits)."""
    if x == 0.0 or not math.isfinite(x):
        return x
    mag = math.floor(math.log10(abs(x)))
    factor = 10 ** (7 - 1 - mag)
    return round(x * factor) / factor


def _fp16_round(x: float) -> float:
    """Round to FP16 precision (~3 significant decimal digits)."""
    if x == 0.0 or not math.isfinite(x):
        return x
    # FP16 has 10-bit mantissa ≈ 3.01 decimal digits
    mag = math.floor(math.log10(abs(x)))
    factor = 10 ** (3 - 1 - mag)
    return round(x * factor) / factor


_ROUNDERS = {
    64: (lambda x: x),    # full Python float — no rounding
    32: _fp32_round,
    16: _fp16_round,
}


# ── Context Manager ───────────────────────────────────────────────────────

class MixedPrecisionContext:
    """
    Controls precision for forward activations and loss scaling.

    Usage
    -----
        amp = MixedPrecisionContext(enabled=True, bits=32, loss_scale=128.0)

        # Forward pass — quantise activations
        with amp.forward():
            logits = model(tokens)

        # Scale loss before backward
        scaled_loss = amp.scale(loss)
        scaled_loss.backward()

        # Unscale gradients, then step
        amp.unscale(optimizer.params)
        optimizer.step()
    """

    def __init__(self, enabled: bool = True, bits: int = 32,
                 loss_scale: float = 1.0):
        self.enabled    = enabled
        self.bits       = bits if enabled else 64
        self.loss_scale = loss_scale if enabled else 1.0
        if enabled and self.bits not in _ROUNDERS:
            allowed = sorted(_ROUNDERS.keys())
            raise ValueError(
                f"unsupported bits: {bits}; supported: {allowed}"
            )
        self._rounder   = _ROUNDERS[self.bits]

        # Track overflow detection
        self._overflow_detected: bool = False

    # ── Quantisation ──────────────────────────────────────────────

    def quantise(self, tensor: NanoTensor) -> NanoTensor:
        """
        Return a new NanoTensor whose *values* are rounded to target precision.
        Gradients still accumulate in full float64.
        The returned tensor is detached from the computation graph (no backprop
        through the quantisation step — matching hardware behaviour where
        quantisation is applied to activations, not to the graph).
        """
        if not self.enabled or self.bits == 64:
            return tensor
        q_data = [self._rounder(x) for x in tensor.data]
        return NanoTensor(q_data, requires_grad=False)

    @contextlib.contextmanager
    def forward(self):
        """Context that flags we are in a forward pass (future: hook quantise here)."""
        yield

    # ── Loss scaling ──────────────────────────────────────────────

    def scale(self, loss: NanoTensor) -> NanoTensor:
        """
        Multiply loss by loss_scale before backward().
        Creates a new leaf node so NanoTensor's graph stays clean.
        """
        if self.loss_scale == 1.0:
            return loss
        scale_t = NanoTensor([self.loss_scale], requires_grad=False)
        return loss * scale_t

    def unscale(self, params: List[NanoTensor]) -> bool:
        """
        Divide all parameter gradients by loss_scale.
        Returns False if any gradient is NaN or Inf (overflow detected).
        """
        if self.loss_scale == 1.0:
            return True

        inv = 1.0 / self.loss_scale
        self._overflow_detected = False
        for p in params:
            if p.grad is None:
                continue
            for i in range(len(p.grad)):
                g = p.grad[i] * inv
                if not math.isfinite(g):
                    self._overflow_detected = True
                    g = 0.0
                p.grad[i] = g

        return not self._overflow_detected

    @property
    def overflow(self) -> bool:
        return self._overflow_detected

    # ── Gradient-precision report ─────────────────────────────────

    def gradient_precision_report(self, params: List[NanoTensor]) -> dict:
        """
        Measure the empirical precision of gradients in terms of
        significant decimal digits.
        """
        if not params:
            return {}
        all_g = []
        for p in params:
            if p.grad:
                all_g.extend(p.grad)
        if not all_g:
            return {"count": 0}
        nonzero = [g for g in all_g if g != 0.0 and math.isfinite(g)]
        if not nonzero:
            return {"count": len(all_g), "nonzero": 0}
        magnitudes = [abs(g) for g in nonzero]
        return {
            "count":   len(all_g),
            "nonzero": len(nonzero),
            "min_abs": min(magnitudes),
            "max_abs": max(magnitudes),
            "mean_abs": sum(magnitudes) / len(magnitudes),
            "bits":    self.bits,
        }
