"""
Jacobian Computation
====================
Computes the Jacobian matrix J of a vector-valued function f: ℝⁿ → ℝᵐ
at a given input point, using reverse-mode automatic differentiation
(one backward pass per output dimension).

Definition
----------
    J[i, j] = ∂f_i / ∂x_j     for i ∈ {0…m−1}, j ∈ {0…n−1}
"""

import math
import logging
from typing import Callable, List, Tuple, Optional, Union, Dict, Any

from ..tensor import NanoTensor

logger = logging.getLogger(__name__)

# ── Full Jacobian via reverse-mode AD ────────────────────────────────────────

def jacobian(
    forward_fn: Callable[[NanoTensor], Union[NanoTensor, List[NanoTensor]]],
    x_data: List[float],
    output_indices: Optional[List[int]] = None,
) -> List[List[float]]:
    """
    Compute the Jacobian of forward_fn w.r.t. input x using reverse-mode AD.

    Parameters
    ----------
    forward_fn    : callable taking a NanoTensor and returning one or more NanoTensors.
    x_data        : base point values.
    output_indices: indices of the output vector to differentiate.

    Returns
    -------
    J : Jacobian matrix where J[i][j] = ∂f_i / ∂x_j.
    """
    n = len(x_data)

    # First pass to build graph and determine output size
    x_init = NanoTensor(x_data, requires_grad=True)
    res_init = forward_fn(x_init)

    if isinstance(res_init, NanoTensor):
        outputs = [res_init]
    else:
        outputs = list(res_init)

    m = len(outputs)
    if output_indices is None:
        output_indices = list(range(m))

    J: List[List[float]] = []

    for i in output_indices:
        # Create fresh input for each row to reset gradients
        x = NanoTensor(x_data, requires_grad=True)
        res = forward_fn(x)

        row_outputs = [res] if isinstance(res, NanoTensor) else list(res)

        if i >= len(row_outputs):
            raise IndexError(f"Output index {i} out of range for result of size {len(row_outputs)}")

        target = row_outputs[i]
        if len(target.data) != 1:
            raise ValueError(f"Jacobian target must be a scalar output; got size {len(target.data)}")

        # Standard NanoTensor backward builds the gradient for the entire graph w.r.t. this output
        target.backward()

        # Snapshot the input gradients as the Jacobian row
        row = list(x.grad) if x.grad else [0.0] * n
        J.append(row)

    return J


# ── Numerical Jacobian (finite differences) ─────────────────────────────────

def numerical_jacobian(
    scalar_fn: Callable[[List[float]], float],
    x_data: List[float],
    eps: float = 1e-4,
) -> List[float]:
    """
    Gradient of a scalar function via central finite differences.
    """
    n = len(x_data)
    grad = []
    for j in range(n):
        x_plus  = list(x_data)
        x_minus = list(x_data)
        x_plus[j]  += eps
        x_minus[j] -= eps
        g = (scalar_fn(x_plus) - scalar_fn(x_minus)) / (2.0 * eps)
        grad.append(g)
    return grad


# ── Jacobian-Vector Product (forward-mode approximation) ───────────────────

def jvp(
    scalar_fn: Callable[[List[float]], float],
    x_data: List[float],
    v: List[float],
    eps: float = 1e-4,
) -> float:
    """
    Directional derivative (Jacobian-vector product) Jv.
    """
    n = len(x_data)
    if len(v) != n:
        raise ValueError(f"v must have length {n}")

    x_plus  = [x_data[i] + eps * v[i] for i in range(n)]
    x_minus = [x_data[i] - eps * v[i] for i in range(n)]
    return (scalar_fn(x_plus) - scalar_fn(x_minus)) / (2.0 * eps)


# ── Gauss-Newton Hessian approximation (JᵀJ) ────────────────────────────────

def gauss_newton_hessian(
    residual_fn: Callable[[List[float]], List[float]],
    x_data: List[float],
    eps: float = 1e-4,
) -> List[List[float]]:
    """
    Compute the Gauss-Newton Hessian approximation G = JᵀJ.
    Useful for least-squares problems and cheap curvature estimation.
    """
    n = len(x_data)
    r0 = residual_fn(x_data)
    m = len(r0)

    # Build full numerical Jacobian J [m x n]
    J: List[List[float]] = []
    for i in range(m):
        # Local scalar function for row i
        def row_fn(x): return residual_fn(x)[i]
        J.append(numerical_jacobian(row_fn, x_data, eps=eps))

    # G = J^T * J [n x n]
    G = [[0.0] * n for _ in range(n)]
    for j1 in range(n):
        for j2 in range(j1, n):
            # dot(J[:, j1], J[:, j2])
            val = sum(J[i][j1] * J[i][j2] for i in range(m))
            G[j1][j2] = val
            G[j2][j1] = val # symmetric

    return G


# ── Statistics ───────────────────────────────────────────────────────────────

def jacobian_stats(J: List[List[float]]) -> Dict[str, Any]:
    """Analyze Jacobian properties."""
    if not J or not J[0]:
        return {"rows": 0, "cols": 0}

    rows = len(J)
    cols = len(J[0])

    flat = [val for row in J for val in row]
    frob_norm = math.sqrt(sum(x*x for x in flat))
    abs_vals = [abs(x) for x in flat]

    return {
        "rows": rows,
        "cols": cols,
        "frobenius_norm": frob_norm,
        "max_abs": max(abs_vals) if abs_vals else 0.0,
        "min_abs": min(abs_vals) if abs_vals else 0.0,
        "sparsity": flat.count(0.0) / len(flat) if flat else 0.0
    }
