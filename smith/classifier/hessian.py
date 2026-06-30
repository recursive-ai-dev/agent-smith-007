"""
Hessian Computation
===================
Computes the Hessian matrix H for curvature analysis.
H[i, j] = ∂²f / (∂x_i ∂x_j)
Uses finite differences of the Jacobian for numerical stability.
"""

import math
import logging
from typing import Callable, List, Optional
from .jacobian import jacobian
from ..tensor import NanoTensor

logger = logging.getLogger(__name__)

def hessian(
    forward_fn: Callable[[NanoTensor], NanoTensor],
    x_data: List[float],
    eps: float = 1e-4
) -> List[List[float]]:
    """
    Compute the Hessian matrix H using finite differences of the Jacobian.
    Assumes forward_fn returns a scalar NanoTensor.
    """
    n = len(x_data)
    H = [[0.0] * n for _ in range(n)]

    def get_grad(x_vals: List[float]) -> List[float]:
        # Wrap in a function that returns a scalar result
        def inner_fn(xt: NanoTensor):
            return forward_fn(xt)

        # Get Jacobian row (gradient)
        J = jacobian(inner_fn, x_vals)
        return J[0]

    logger.debug("Computing Hessian for %d parameters...", n)

    for j in range(n):
        x_plus = list(x_data)
        x_minus = list(x_data)
        x_plus[j] += eps
        x_minus[j] -= eps

        try:
            g_plus = get_grad(x_plus)
            g_minus = get_grad(x_minus)

            for i in range(n):
                # Central difference approximation of the gradient's derivative
                H[i][j] = (g_plus[i] - g_minus[i]) / (2.0 * eps)
        except Exception as e:
            logger.error("Hessian step failure at index %d: %s", j, e)
            raise

    return H
