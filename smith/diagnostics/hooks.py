"""
Diagnostic Hooks
================
Pluggable callbacks for training-time monitoring of:
  • Forward-pass activations (distribution, norms)
  • Gradient norms and statistics per parameter
  • Loss trajectory
  • Jacobian and Hessian snapshots (periodic, on small subsets)
  • GSAR symbol-registry growth
  • SEP consistency-weight distributions

Usage
-----
    diag = DiagnosticsManager(model, config)
    # register with model
    model.register_forward_hook(diag.on_activation)
    # call after each step
    diag.after_step(step, loss, optimizer, model)
    # print summary
    print(diag.report())
"""

import math
import os
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional

from ..classifier.model import AgentSmith
from ..classifier.adam import AdamOptimizer
from ..classifier.config import AgentSmithConfig
from ..classifier.jacobian import numerical_jacobian, jacobian_stats
from ..classifier.hessian import curvature_summary


# ─────────────────────────────────────────────────────────────────────────────
# Activation statistics
# ─────────────────────────────────────────────────────────────────────────────

def _vec_stats(values: List[float]) -> dict:
    """Mean, std, min, max of a numeric list."""
    if not values:
        return {}
    n    = len(values)
    mu   = sum(values) / n
    var  = sum((x - mu) ** 2 for x in values) / max(n - 1, 1)
    return {
        "mean": mu,
        "std":  math.sqrt(var),
        "min":  min(values),
        "max":  max(values),
        "norm": math.sqrt(sum(x * x for x in values)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# DiagnosticsManager
# ─────────────────────────────────────────────────────────────────────────────

class DiagnosticsManager:
    """
    Central hub for all diagnostic hooks.

    Parameters
    ----------
    model    : AgentSmith instance (the model being trained)
    config   : AgentSmithConfig
    log_dir  : directory for persisting JSON logs (None = no file output)
    """

    def __init__(
        self,
        model: AgentSmith,
        config: AgentSmithConfig,
        log_dir: Optional[str] = None,
    ):
        self.model    = model
        self.config   = config
        self.log_dir  = log_dir

        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # ── History buffers ──────────────────────────────────────────
        self.loss_history:        List[float]        = []
        self.grad_norm_history:   List[float]        = []
        self.lr_history:          List[float]        = []
        self.activation_history:  Dict[str, list]    = defaultdict(list)
        self.jacobian_history:    List[dict]          = []
        self.hessian_history:     List[dict]          = []
        self.gsar_history:        List[dict]          = []
        self.sep_history:         List[dict]          = []

        # ── Step counters ────────────────────────────────────────────
        self._step: int = 0

    # ── Forward-activation hook ──────────────────────────────────────────

    def on_activation(self, layer_name: str, data: Any):
        """
        Called by model._fire_forward_hooks(layer_name, data).
        Records summary statistics for each named activation.
        """
        if isinstance(data, list):
            if data and isinstance(data[0], list):
                # list-of-vectors: flatten to one list for stats
                flat = [x for row in data for x in row]
            else:
                flat = [float(x) for x in data]
        elif isinstance(data, (int, float)):
            flat = [float(data)]
        else:
            return

        stats = _vec_stats(flat)
        stats["step"] = self._step
        self.activation_history[layer_name].append(stats)

    # ── Per-step callback ─────────────────────────────────────────────────

    def after_step(
        self,
        step: int,
        loss: float,
        optimizer: AdamOptimizer,
        sep_explanation: Optional[dict] = None,
    ):
        """
        Call after every optimiser step.

        Records loss, gradient norms, GSAR stats, SEP weight distribution,
        and (periodically) Jacobian / Hessian diagnostics.
        """
        self._step = step

        # ── Loss ──────────────────────────────────────────────────
        self.loss_history.append(loss)

        # ── Optimizer diagnostics ─────────────────────────────────
        self.grad_norm_history.append(optimizer.global_grad_norm())
        self.lr_history.append(optimizer._current_lr())

        # ── Gradient statistics per parameter (sampled) ───────────
        if step % self.config.grad_stats_freq == 0:
            self._log_gradient_stats(step)

        # ── GSAR ──────────────────────────────────────────────────
        gsar_s = self.model.gsar.stats()
        gsar_s["step"] = step
        self.gsar_history.append(gsar_s)

        # ── SEP ───────────────────────────────────────────────────
        if sep_explanation:
            self.sep_history.append({
                "step":               step,
                "dominant_chunk":     sep_explanation.get("dominant_chunk"),
                "consistency_weights":sep_explanation.get("consistency_weights"),
                "num_chunks":         sep_explanation.get("num_chunks"),
            })

        # ── Jacobian (periodic) ───────────────────────────────────
        if step > 0 and step % self.config.jacobian_freq == 0:
            self._log_jacobian(step)

        # ── Hessian (periodic) ────────────────────────────────────
        if step > 0 and step % self.config.hessian_freq == 0:
            self._log_hessian(step)

    # ── Gradient statistics ───────────────────────────────────────────────

    def _log_gradient_stats(self, step: int):
        params = self.model.parameters()
        all_grads: List[float] = []
        for p in params:
            if p.grad:
                all_grads.extend(p.grad)
        if not all_grads:
            return
        stats       = _vec_stats(all_grads)
        stats["step"]        = step
        stats["num_params"]  = len(all_grads)
        stats["zero_frac"]   = sum(1 for g in all_grads if g == 0.0) / len(all_grads)
        # Log to activation history under a special key
        self.activation_history["_gradients"].append(stats)

    # ── Jacobian snapshot ─────────────────────────────────────────────────

    def _log_jacobian(self, step: int):
        """
        Compute Jacobian of the SEP global classifier's output w.r.t.
        a small sample of its weights.  Uses numerical central differences.
        """
        # Sample first few params of global_clf for tractability
        clf  = self.model.sep.global_clf
        w    = clf.weight.data
        n    = min(8, len(w))    # only the first 8 weights
        idx  = list(range(n))
        w_snapshot = w[:]

        def loss_from_w(w_vals):
            # Temporarily set weights, run a simple dot with fixed input
            for i, ii in enumerate(idx):
                clf.weight.data[ii] = w_vals[i]
            # Probe with a zero input vector
            dummy = [0.5] * self.model.config.d_model
            from ..tensor import NanoTensor
            dummy_t = NanoTensor(dummy, requires_grad=False)
            out = clf(dummy_t)
            return out.data[0]    # scalar output

        try:
            g = numerical_jacobian(loss_from_w, [w_snapshot[i] for i in idx])
            stats = jacobian_stats([g]) if g else {}
            stats["step"] = step
            self.jacobian_history.append(stats)
        except Exception:
            pass
        finally:
            # Restore weights
            for i, ii in enumerate(idx):
                clf.weight.data[ii] = w_snapshot[ii]

    # ── Hessian snapshot ──────────────────────────────────────────────────

    def _log_hessian(self, step: int):
        """
        Curvature summary for a small subset of the global classifier weights.
        """
        clf     = self.model.sep.global_clf
        w       = clf.weight.data
        sample  = min(6, len(w))
        idx     = list(range(sample))
        w_snap  = w[:]

        def scalar_loss(w_vals):
            for i, ii in enumerate(idx):
                clf.weight.data[ii] = w_vals[i]
            from ..tensor import NanoTensor
            dummy_t = NanoTensor([0.5] * self.model.config.d_model, requires_grad=False)
            out = clf(dummy_t)
            # Simple squared output as loss proxy
            return sum(v * v for v in out.data)

        try:
            theta = [w_snap[i] for i in idx]
            summary = curvature_summary(scalar_loss, theta, sample_size=sample)
            summary["step"] = step
            self.hessian_history.append(summary)
        except Exception:
            pass
        finally:
            for i, ii in enumerate(idx):
                clf.weight.data[ii] = w_snap[ii]

    # ── Summary report ────────────────────────────────────────────────────

    def report(self, last_n: int = 20) -> str:
        """Return a formatted diagnostic summary."""
        lines = ["═" * 60, "  AGENT SMITH — DIAGNOSTICS REPORT", "═" * 60]

        # Loss
        if self.loss_history:
            recent = self.loss_history[-last_n:]
            lines.append(
                f"  Loss   (last {len(recent)} steps): "
                f"min={min(recent):.4f}  max={max(recent):.4f}  "
                f"last={recent[-1]:.4f}"
            )

        # Gradient norm
        if self.grad_norm_history:
            recent = self.grad_norm_history[-last_n:]
            lines.append(
                f"  ‖∇‖₂  (last {len(recent)} steps): "
                f"min={min(recent):.4e}  max={max(recent):.4e}  "
                f"last={recent[-1]:.4e}"
            )

        # GSAR
        if self.gsar_history:
            last_g = self.gsar_history[-1]
            lines.append(
                f"  GSAR  registered_symbols={last_g.get('registered_symbols', 0)}  "
                f"windows_seen={last_g.get('total_windows_seen', 0)}"
            )

        # Jacobian
        if self.jacobian_history:
            last_j = self.jacobian_history[-1]
            lines.append(
                f"  Jacobian (step {last_j.get('step')})  "
                f"frob={last_j.get('frobenius_norm', 'n/a'):.4e}  "
                f"max={last_j.get('max_abs', 'n/a'):.4e}"
            )

        # Hessian
        if self.hessian_history:
            last_h = self.hessian_history[-1]
            lines.append(
                f"  Hessian (step {last_h.get('step')})  "
                f"κ≈{last_h.get('condition_number_est', 'n/a'):.2f}  "
                f"mean_curv={last_h.get('mean_curvature', 'n/a'):.4e}"
            )

        # Activation summaries
        for layer, history in self.activation_history.items():
            if not history:
                continue
            s = history[-1]
            lines.append(
                f"  Act/{layer:20s}  "
                f"mean={s.get('mean', 0):.4f}  "
                f"std={s.get('std', 0):.4f}  "
                f"norm={s.get('norm', 0):.4f}"
            )

        lines.append("═" * 60)
        return "\n".join(lines)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, filename: str = "diagnostics.json"):
        """Persist all history to JSON."""
        if not self.log_dir:
            return
        path = os.path.join(self.log_dir, filename)
        data = {
            "loss":       self.loss_history,
            "grad_norm":  self.grad_norm_history,
            "lr":         self.lr_history,
            "jacobian":   self.jacobian_history,
            "hessian":    self.hessian_history,
            "gsar":       self.gsar_history,
            "sep":        self.sep_history,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
