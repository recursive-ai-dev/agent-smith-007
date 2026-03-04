"""
Hessian Computation
====================
Computes second-order partial derivatives of a scalar loss function.

Definition
----------
    H[i, j] = ∂²L / ∂θᵢ ∂θⱼ

Implementations provided
-------------------------
1. Full Hessian via central finite differences (FD)
   Cost: O(n²) function evaluations, suitable for small parameter subsets.

2. Hessian-Vector Product (HVP) via second-order FD
   Cost: O(1) function evaluations per query direction.
   H v ≈ [∇L(θ + ε v) − ∇L(θ − ε v)] / (2ε)

3. Diagonal Hessian estimate (cheap approximation)
   H[i,i] ≈ [L(θ + ε eᵢ) − 2L(θ) + L(θ − ε eᵢ)] / ε²
   O(n) evaluations — useful for adaptive preconditioners.

4. Curvature analysis utilities
   - Eigenvalue bounds (power iteration / Gershgorin circles)
   - Condition number estimate

Mathematical note
-----------------
At a local minimum the Hessian is positive semi-definite (PSD).
Negative eigenvalues indicate saddle points; large condition numbers
imply slow convergence of first-order methods.

The Adam optimiser implicitly approximates the diagonal Hessian via
the second-moment estimate v_t ≈ diag(H) / (1 − β₂), so the diagonal
Hessian here can be compared against Adam's v_t for interpretability.
"""

import math
from typing import Callable, List, Optional, Tuple


# ── Gradient helper ──────────────────────────────────────────────────────────

def _numerical_gradient(
    loss_fn: Callable[[List[float]], float],
    theta: List[float],
    eps: float = 1e-4,
) -> List[float]:
    """Central-difference gradient: ∂L/∂θᵢ ≈ [L(θ+εeᵢ) − L(θ−εeᵢ)] / (2ε)."""
    n = len(theta)
    g = []
    for i in range(n):
        t_p = theta[:]
        t_m = theta[:]
        t_p[i] += eps
        t_m[i] -= eps
        g.append((loss_fn(t_p) - loss_fn(t_m)) / (2.0 * eps))
    return g


# ── Full Hessian ─────────────────────────────────────────────────────────────

def hessian(
    loss_fn: Callable[[List[float]], float],
    theta: List[float],
    eps: float = 1e-4,
    indices: Optional[List[int]] = None,
) -> List[List[float]]:
    """
    Full Hessian via central finite differences.

    H[i,j] = [L(θ + ε eᵢ + ε eⱼ) − L(θ + ε eᵢ − ε eⱼ)
               − L(θ − ε eᵢ + ε eⱼ) + L(θ − ε eᵢ − ε eⱼ)] / (4ε²)

    For i == j we use the simpler second-order stencil:
    H[i,i] = [L(θ + ε eᵢ) − 2L(θ) + L(θ − ε eᵢ)] / ε²

    Parameters
    ----------
    loss_fn : scalar loss function
    theta   : parameter vector (list of floats)
    eps     : finite-difference step size
    indices : subset of parameter indices to differentiate (None = all)

    Returns
    -------
    H : square matrix as list-of-lists, shape (|indices|, |indices|)
    """
    if indices is None:
        indices = list(range(len(theta)))
    k   = len(indices)
    L0  = loss_fn(theta)
    H   = [[0.0] * k for _ in range(k)]

    for a, i in enumerate(indices):
        for b, j in enumerate(indices):
            if i == j:
                # Diagonal: second-order central difference
                t_p = theta[:]
                t_m = theta[:]
                t_p[i] += eps
                t_m[i] -= eps
                H[a][b] = (loss_fn(t_p) - 2.0 * L0 + loss_fn(t_m)) / (eps * eps)
            elif b > a:
                # Off-diagonal: mixed partial via 4-point stencil
                t_pp = theta[:]
                t_pm = theta[:]
                t_mp = theta[:]
                t_mm = theta[:]
                t_pp[i] += eps;  t_pp[j] += eps
                t_pm[i] += eps;  t_pm[j] -= eps
                t_mp[i] -= eps;  t_mp[j] += eps
                t_mm[i] -= eps;  t_mm[j] -= eps
                h_ij = (loss_fn(t_pp) - loss_fn(t_pm)
                        - loss_fn(t_mp) + loss_fn(t_mm)) / (4.0 * eps * eps)
                H[a][b] = h_ij
                H[b][a] = h_ij   # symmetry: ∂²L/∂θᵢ∂θⱼ = ∂²L/∂θⱼ∂θᵢ

    return H


# ── Hessian-Vector Product ────────────────────────────────────────────────────

def hvp(
    loss_fn: Callable[[List[float]], float],
    theta: List[float],
    v: List[float],
    eps: float = 1e-4,
) -> List[float]:
    """
    Hessian-vector product:  H(θ) v

    Uses the gradient-difference formula:
        H v ≈ [∇L(θ + ε v) − ∇L(θ − ε v)] / (2ε)

    Cost: 2n function evaluations (one forward per gradient component).
    This is the workhorse for large models where the full Hessian is
    impractical (e.g. conjugate-gradient Newton-Krylov solvers).
    """
    if len(v) != len(theta):
        raise ValueError(
            f"hvp: v must have same length as theta "
            f"(got len(v)={len(v)}, len(theta)={len(theta)})"
        )
    if not hasattr(v, '__getitem__'):
        raise TypeError(
            f"hvp: v must be an indexable sequence, got {type(v).__name__}"
        )
    n = len(theta)
    theta_p = [theta[i] + eps * v[i] for i in range(n)]
    theta_m = [theta[i] - eps * v[i] for i in range(n)]
    g_p = _numerical_gradient(loss_fn, theta_p, eps=eps * 0.1)
    g_m = _numerical_gradient(loss_fn, theta_m, eps=eps * 0.1)
    return [(g_p[i] - g_m[i]) / (2.0 * eps) for i in range(n)]


# ── Diagonal Hessian ─────────────────────────────────────────────────────────

def diagonal_hessian(
    loss_fn: Callable[[List[float]], float],
    theta: List[float],
    eps: float = 1e-4,
) -> List[float]:
    """
    Efficient diagonal of the Hessian via per-parameter second-order FD:
        H[i,i] = [L(θ + ε eᵢ) − 2 L(θ) + L(θ − ε eᵢ)] / ε²

    Cost: 2n + 1 function evaluations.
    The diagonal gives a simple preconditioner / curvature estimate.
    """
    n  = len(theta)
    L0 = loss_fn(theta)
    diag = []
    for i in range(n):
        t_p = theta[:]
        t_m = theta[:]
        t_p[i] += eps
        t_m[i] -= eps
        h_ii = (loss_fn(t_p) - 2.0 * L0 + loss_fn(t_m)) / (eps * eps)
        diag.append(h_ii)
    return diag


# ── Eigenvalue bounds ─────────────────────────────────────────────────────────

def gershgorin_bounds(H: List[List[float]]) -> Tuple[float, float]:
    """
    Gershgorin circle theorem: every eigenvalue λ of H satisfies
        |λ − H[i,i]| ≤ Σ_{j≠i} |H[i,j]|
    so λ ∈ [H[i,i] − Rᵢ, H[i,i] + Rᵢ] for some i.

    Returns (lower_bound, upper_bound) on the eigenvalue spectrum.
    O(n²) cost — no iterative solver required.
    """
    n = len(H)
    if n == 0 or any(len(row) != n for row in H):
        raise ValueError("H must be a non-empty square matrix")
    lo, hi = float('inf'), float('-inf')
    for i in range(n):
        r = sum(abs(H[i][j]) for j in range(n) if j != i)
        lo = min(lo, H[i][i] - r)
        hi = max(hi, H[i][i] + r)
    return lo, hi


def power_iteration(
    H: List[List[float]],
    max_iter: int = 50,
    tol: float = 1e-6,
) -> Tuple[float, List[float]]:
    """
    Power iteration to estimate the largest eigenvalue of a symmetric H.

    Returns (lambda_max, eigenvector).
    Convergence guaranteed for symmetric matrices (all real eigenvalues).
    """
    n = len(H)
    if n == 0 or any(len(row) != n for row in H):
        raise ValueError("H must be a non-empty square matrix")
    # Random initialisation
    import random
    v = [random.gauss(0, 1) for _ in range(n)]
    norm_v = math.sqrt(sum(x*x for x in v))
    if norm_v < 1e-30:
        raise ValueError("cannot normalize zero vector")
    v = [x / norm_v for x in v]

    lam = 0.0
    for _ in range(max_iter):
        # w = H v
        w = [sum(H[i][j] * v[j] for j in range(n)) for i in range(n)]
        lam_new = sum(v[i] * w[i] for i in range(n))
        norm_w = math.sqrt(sum(x*x for x in w))
        if norm_w < 1e-30:
            break
        v = [x / norm_w for x in w]
        if abs(lam_new - lam) < tol:
            lam = lam_new
            break
        lam = lam_new

    return lam, v


# ── Curvature summary ─────────────────────────────────────────────────────────

def curvature_summary(
    loss_fn: Callable[[List[float]], float],
    theta: List[float],
    sample_size: int = 8,
    eps: float = 1e-4,
) -> dict:
    """
    Lightweight curvature diagnostics for a model at given parameter values.

    Samples `sample_size` parameters uniformly, computes their 2D sub-Hessian,
    and returns interpretable statistics.

    Returns dict with keys:
      diag_hessian_sample  : diagonal H values for sampled params
      mean_curvature       : mean of diagonal H (overall curvature)
      max_curvature        : max diagonal — steepest direction
      min_curvature        : min diagonal — flattest direction
      gershgorin_lo/hi     : eigenvalue bounds for sub-Hessian
      condition_number_est : max_curvature / (|min_curvature| + eps)
    """
    n    = len(theta)
    step = max(1, n // sample_size)
    idx  = list(range(0, n, step))[:sample_size]

    # Diagonal Hessian for all sampled parameters
    L0   = loss_fn(theta)
    diag = []
    for i in idx:
        t_p = theta[:]
        t_m = theta[:]
        t_p[i] += eps
        t_m[i] -= eps
        h_ii = (loss_fn(t_p) - 2.0 * L0 + loss_fn(t_m)) / (eps * eps)
        diag.append(h_ii)

    # Sub-Hessian for Gershgorin bounds
    sub_H = hessian(loss_fn, theta, eps=eps, indices=idx)
    lo, hi = gershgorin_bounds(sub_H)

    mean_c = sum(diag) / len(diag) if diag else 0.0
    max_c  = max(diag) if diag else 0.0
    min_c  = min(diag) if diag else 0.0
    cond   = abs(max_c) / (abs(min_c) + 1e-30)

    return {
        "diag_hessian_sample":  diag,
        "sampled_param_indices": idx,
        "mean_curvature":        mean_c,
        "max_curvature":         max_c,
        "min_curvature":         min_c,
        "gershgorin_lo":         lo,
        "gershgorin_hi":         hi,
        "condition_number_est":  cond,
    }
