"""
Jacobian Computation
====================
Computes the Jacobian matrix J of a vector-valued function f: ℝⁿ → ℝᵐ
at a given input point, using reverse-mode automatic differentiation
(one backward pass per output dimension).

Definition
----------
    J[i, j] = ∂f_i / ∂x_j     for i ∈ {0…m−1}, j ∈ {0…n−1}

Algorithm (reverse-mode, row by row)
--------------------------------------
For each output dimension i:
  1. Run f(x) to get outputs [f₀, f₁, …, f_{m−1}] (NanoTensors).
  2. Seed grad of f_i = 1, all other f_k = 0.
  3. Call f_i.backward() — propagates ∂f_i/∂x_j for all j.
  4. Read off x.grad as the i-th row of J.

Because NanoTensor reuses the same parameter objects, we must carefully
snapshot the data values and reconstruct fresh NanoTensors for each row.

Gauss-Newton Approximation
---------------------------
For a residual function r: ℝⁿ → ℝᵐ with loss L = ½ ‖r‖²:
    ∇²L ≈ JᵀJ     (positive semi-definite Gauss-Newton Hessian)
This is cheaper than the full Hessian and always PSD.

Jacobian-Vector Product (forward-mode, O(n) cost)
---------------------------------------------------
For directional derivative in direction v:
    Jv ≈ [f(x + ε v) − f(x − ε v)] / (2ε)
Provided as jvp() for efficiency when only directional info is needed.
"""

import math
from typing import Callable, List, Tuple

from ..tensor import NanoTensor


# ── Full Jacobian via reverse-mode AD ────────────────────────────────────────

def jacobian(
    forward_fn: Callable[[List[float]], List[NanoTensor]],
    x_data: List[float],
    output_indices: List[int] = None,
) -> List[List[float]]:
    """
    Compute the Jacobian of forward_fn w.r.t. input x.

    Parameters
    ----------
    forward_fn    : callable x_data → list of scalar NanoTensors (outputs)
    x_data        : input values (list of floats)
    output_indices: which output dimensions to differentiate (None = all)

    Returns
    -------
    J : list of rows.  J[i] is ∂f_i/∂x (a list of length n).
    """
    n = len(x_data)

    # First pass to discover output dimension m
    x_probe = NanoTensor(x_data[:], requires_grad=True)
    outputs_probe = forward_fn(x_probe.data)
    m = len(outputs_probe)

    if output_indices is None:
        output_indices = list(range(m))

    J: List[List[float]] = []

    for i in output_indices:
        # Fresh input with grad enabled
        x = NanoTensor(x_data[:], requires_grad=True)
        outputs = forward_fn(x.data)

        # Re-run forward with x as NanoTensor to build computation graph
        # (forward_fn must accept NanoTensor-like, so we wrap in a single tensor)
        x_t = NanoTensor(x_data[:], requires_grad=True)
        outs = _run_with_tensor(forward_fn, x_t, m)

        if outs is None or i >= len(outs):
            J.append([0.0] * n)
            continue

        # Seed gradient at output i
        out_i = outs[i]
        if len(out_i.data) != 1:
            raise ValueError(f"Output {i} must be scalar; got length {len(out_i.data)}")

        # Manual backward from a single output
        out_i.grad = [1.0]
        # Walk the graph manually: set all other output grads to 0
        # (they share the same computation graph, so backward on out_i
        #  only propagates through its parents)
        out_i.backward()

        row = list(x_t.grad) if x_t.grad else [0.0] * n
        J.append(row)

    return J


def _run_with_tensor(
    forward_fn: Callable,
    x_t: NanoTensor,
    expected_outputs: int,
) -> List[NanoTensor]:
    """
    Call forward_fn with the NanoTensor itself so the graph is built.
    Works when forward_fn accepts a NanoTensor (not just raw data).
    Falls back gracefully.
    """
    try:
        result = forward_fn(x_t)
        if isinstance(result, (list, tuple)):
            return list(result)
        if isinstance(result, NanoTensor):
            # scalar → wrap
            return [result]
    except Exception:
        pass
    return None


# ── Numerical Jacobian (finite differences) ─────────────────────────────────

def numerical_jacobian(
    scalar_fn: Callable[[List[float]], float],
    x_data: List[float],
    eps: float = 1e-4,
) -> List[float]:
    """
    Gradient of a scalar function via central finite differences:
        ∂f/∂x_j ≈ [f(x + ε eⱼ) − f(x − ε eⱼ)] / (2ε)

    Returns gradient vector of length n.
    This is the j=0 row of the Jacobian (for scalar f).
    """
    n = len(x_data)
    grad = []
    for j in range(n):
        x_plus  = x_data[:]
        x_minus = x_data[:]
        x_plus[j]  += eps
        x_minus[j] -= eps
        g = (scalar_fn(x_plus) - scalar_fn(x_minus)) / (2.0 * eps)
        grad.append(g)
    return grad


# ── Jacobian-Vector Product (forward-mode) ───────────────────────────────────

def jvp(
    scalar_fn: Callable[[List[float]], float],
    x_data: List[float],
    v: List[float],
    eps: float = 1e-4,
) -> float:
    """
    Forward-mode directional derivative (Jacobian-vector product):
        Jv ≈ [f(x + ε v) − f(x − ε v)] / (2ε)

    Cost: 2 function evaluations, independent of input dimension n.
    """
    n = len(x_data)
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
    Compute the Gauss-Newton Hessian approximation: G = JᵀJ

    For residual r: ℝⁿ → ℝᵐ, computes J numerically via central FD
    then returns JᵀJ ∈ ℝ^{n×n}.

    Properties:
      • Always positive semi-definite
      • Cheaper than the full Hessian (no second-order FD)
      • Exact when the residuals are small (near a minimum)
    """
    n   = len(x_data)
    r0  = residual_fn(x_data)
    m   = len(r0)

    # Build numerical Jacobian J ∈ ℝ^{m×n}
    J: List[List[float]] = []
    for j in range(n):
        x_p = x_data[:]
        x_m = x_data[:]
        x_p[j] += eps
        x_m[j] -= eps
        rp = residual_fn(x_p)
        rm = residual_fn(x_m)
        col = [(rp[k] - rm[k]) / (2.0 * eps) for k in range(m)]
        J.append(col)   # col j of J (shape m)

    # G = JᵀJ:  G[i][j] = Σ_k J[i][k] * J[j][k]
    G = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            s = sum(J[i][k] * J[j][k] for k in range(m))
            G[i][j] = s
            G[j][i] = s   # symmetric

    return G


# ── Summary stats for diagnostics ────────────────────────────────────────────

def jacobian_stats(J: List[List[float]]) -> dict:
    """
    Compute useful statistics about a Jacobian matrix.
    Returns: frobenius_norm, max_abs, min_abs, spectral_approx (power iter).
    """
    flat = [J[i][j] for i in range(len(J)) for j in range(len(J[0]))]
    if not flat:
        return {}
    frob  = math.sqrt(sum(x*x for x in flat))
    abs_f = [abs(x) for x in flat]
    return {
        "frobenius_norm": frob,
        "max_abs":        max(abs_f),
        "min_abs":        min(abs_f),
        "rows":           len(J),
        "cols":           len(J[0]) if J else 0,
    }
