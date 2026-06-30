"""
Mixed Precision Management
==========================
Simulates 16-bit (half) and 32-bit (float) mixed training.
Includes Dynamic Loss Scaling to prevent gradient underflow.
"""

import math
import logging
from typing import List, Optional

from ..tensor import NanoTensor

logger = logging.getLogger(__name__)


class MixedPrecisionManager:
    """
    Manages loss scaling and simulated fp16 casts.
    """

    def __init__(
        self,
        init_scale: float = 2.0 ** 16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 100,
    ):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval

        self._steps_since_last_scale = 0
        self._has_inf_or_nan = False

    def scale_loss(self, loss: NanoTensor) -> NanoTensor:
        """Apply current scale to the loss scalar."""
        return loss * self.scale

    def unscale_gradients(self, params: List[NanoTensor]):
        """Check for NaN/Inf and unscale gradients."""
        self._has_inf_or_nan = False
        inv_scale = 1.0 / self.scale

        for p in params:
            if not p.grad:
                continue

            for i in range(len(p.grad)):
                g = p.grad[i]
                if not math.isfinite(g):
                    self._has_inf_or_nan = True
                    break
                p.grad[i] *= inv_scale

            if self._has_inf_or_nan:
                break

    def update(self):
        """Update scale factor based on gradient validity."""
        if self._has_inf_or_nan:
            self.scale *= self.backoff_factor
            self._steps_since_last_scale = 0
            logger.debug("Gradient overflow detected. Scaling back to %s", self.scale)
        else:
            self._steps_since_last_scale += 1
            if self._steps_since_last_scale >= self.growth_interval:
                self.scale *= self.growth_factor
                self._steps_since_last_scale = 0
                logger.debug("Stable training. Increasing scale to %s", self.scale)

    @property
    def should_skip_step(self) -> bool:
        """True if current gradients contain non-finite values."""
        return self._has_inf_or_nan
