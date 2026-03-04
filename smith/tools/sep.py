"""
SEP — Self-Explanatory Perception
===================================
SEP processes an input sequence in *fixed-size chunks*, makes a preliminary
class prediction after each chunk, and then compares predictions across
chunks before committing to a final output.

Motivation
----------
A spurious correlation is a statistical association in training data that
does not reflect a genuine causal relationship.  A model that relies on
a spurious feature will make confident but fragile predictions.  SEP
detects and suppresses this by checking *consistency*: if every chunk of
a document independently agrees on a label, that agreement is evidence of
a genuine signal.  If only an early chunk drives the prediction and later
chunks disagree, the early-chunk prediction is likely spurious.

Algorithm
---------
Given a sequence of T hidden-state vectors h₁…h_T (each in ℝ^d_model):

1. Split into chunks:  C_k = h_{(k−1)K+1} … h_{min(kK, T)}

2. Per-chunk pooled representation:
       r_k = mean(C_k)   ∈ ℝ^d_model

3. Per-chunk preliminary logits via a lightweight classifier W_chunk:
       ℓ_k = W_chunk r_k + b_chunk   ∈ ℝ^C

4. Consistency weight for chunk k:
       cos(k, j) = (ℓ_k · ℓ_j) / (‖ℓ_k‖ ‖ℓ_j‖ + ε)
       cons_k = (1/(K−1)) Σ_{j≠k} cos(k, j)   ∈ [−1, 1]
       w_k = softmax( cons_k )   ∈ [0, 1],  Σ_k w_k = 1

   High w_k → chunk k's prediction is consistent with others → trustworthy.

5. Spurious-corrected logits:
       ℓ_k* = ℓ_k − λ · (Σ_{j≠k} w_j · ℓ_j)
   (penalise chunks whose predictions are dominated by others)

6. Final logits:
       ℓ_final = W_final(mean(h₁…h_T))  +  Σ_k w_k · ℓ_k*

7. Explanation dictionary (returned alongside logits):
       chunk_logits         : raw per-chunk predictions
       consistency_weights  : w_k values
       spurious_penalty     : λ Σ_{j≠k} w_j ℓ_j per chunk
       chunk_contributions  : w_k · ℓ_k*  (additive contributions to final)
       dominant_chunk       : argmax(w_k)
"""

import math
from typing import List, Optional, Dict, Any

from ..tensor import NanoTensor
from ..classifier.layers import Linear


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dot(a: List[float], b: List[float]) -> float:
    return sum(a[i] * b[i] for i in range(len(a)))


def _l2(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _cosine(a: List[float], b: List[float]) -> float:
    return _dot(a, b) / (_l2(a) * _l2(b) + 1e-12)


def _softmax(vals: List[float]) -> List[float]:
    mv = max(vals)
    e  = [math.exp(v - mv) for v in vals]
    s  = sum(e)
    return [ei / s for ei in e]


def _mean_pool(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    d = len(vectors[0])
    result = [0.0] * d
    for v in vectors:
        for i in range(d):
            result[i] += v[i]
    n = len(vectors)
    return [x / n for x in result]


# ─────────────────────────────────────────────────────────────────────────────
# SEP Module
# ─────────────────────────────────────────────────────────────────────────────

class SEP:
    """
    Self-Explanatory Perception module.

    Parameters
    ----------
    d_model      : hidden dimension (from transformer encoder)
    num_classes  : number of output classes
    chunk_size   : tokens per processing chunk  (K)
    lambda_      : spurious-correlation penalty coefficient
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        chunk_size: int = 32,
        lambda_: float = 0.15,
    ):
        self.d_model     = d_model
        self.num_classes = num_classes
        self.chunk_size  = chunk_size
        self.lambda_     = lambda_

        # Lightweight per-chunk classifier (shared across all chunks)
        self.chunk_clf  = Linear(d_model, num_classes)

        # Full-sequence classifier (applied to global mean pool)
        self.global_clf = Linear(d_model, num_classes)

    # ── Core forward ────────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: List[NanoTensor],
    ) -> tuple:  # (NanoTensor [num_classes], dict)
        """
        Parameters
        ----------
        hidden_states : list of T NanoTensors, each ∈ ℝ^d_model

        Returns
        -------
        logits      : NanoTensor [num_classes]
        explanation : dict with per-chunk diagnostics
        """
        T = len(hidden_states)
        K = self.chunk_size

        # ── 1. Split into chunks ──────────────────────────────────────
        chunks: List[List[NanoTensor]] = []
        start = 0
        while start < T:
            chunks.append(hidden_states[start: start + K])
            start += K

        num_chunks = len(chunks)

        # ── 2. Per-chunk pooled representation + preliminary logits ───
        chunk_reprs:  List[NanoTensor] = []
        chunk_logits: List[NanoTensor] = []

        for chunk in chunks:
            # Mean pool: r_k = mean(h) — use NanoTensor ops for backprop
            r_k = chunk[0]
            for h in chunk[1:]:
                r_k = r_k + h
            r_k = r_k * NanoTensor(
                [1.0 / len(chunk)] * self.d_model, requires_grad=False
            )
            chunk_reprs.append(r_k)
            chunk_logits.append(self.chunk_clf(r_k))

        # ── 3. Consistency weights (scalar, non-differentiable) ───────
        # Use .data snapshots to avoid polluting the backward graph
        logit_data = [cl.data for cl in chunk_logits]

        if num_chunks == 1:
            weights     = [1.0]
            cons_scores = [1.0]
        else:
            cos_matrix = [
                [_cosine(logit_data[i], logit_data[j]) for j in range(num_chunks)]
                for i in range(num_chunks)
            ]
            cons_scores = []
            for i in range(num_chunks):
                others = [cos_matrix[i][j] for j in range(num_chunks) if j != i]
                cons_scores.append(sum(others) / len(others))
            weights = _softmax(cons_scores)

        # ── 4. Spurious-corrected chunk logits ────────────────────────
        corrected_logits: List[NanoTensor] = []

        for k in range(num_chunks):
            # Penalty = λ · Σ_{j≠k} w_j · ℓ_j
            penalty_data = [0.0] * self.num_classes
            for j in range(num_chunks):
                if j == k:
                    continue
                wj = weights[j]
                for c in range(self.num_classes):
                    penalty_data[c] += self.lambda_ * wj * logit_data[j][c]

            penalty_t = NanoTensor(penalty_data, requires_grad=False)
            corrected_logits.append(chunk_logits[k] - penalty_t)

        # ── 5. Global representation logits ──────────────────────────
        # Sum all hidden states then divide (backprop-friendly mean pool)
        global_sum = hidden_states[0]
        for h in hidden_states[1:]:
            global_sum = global_sum + h
        global_mean = global_sum * NanoTensor(
            [1.0 / T] * self.d_model, requires_grad=False
        )
        global_logits = self.global_clf(global_mean)

        # ── 6. Final logits = global + Σ_k w_k · ℓ_k* ───────────────
        weighted_sum_logits = None
        for k in range(num_chunks):
            wk_t = NanoTensor([weights[k]] * self.num_classes, requires_grad=False)
            contribution = corrected_logits[k] * wk_t
            weighted_sum_logits = (
                contribution if weighted_sum_logits is None
                else weighted_sum_logits + contribution
            )

        final_logits = global_logits + weighted_sum_logits

        # ── 7. Build explanation dict ─────────────────────────────────
        chunk_contrib_data = []
        for k in range(num_chunks):
            contribution = [weights[k] * corrected_logits[k].data[c]
                            for c in range(self.num_classes)]
            chunk_contrib_data.append(contribution)

        explanation: Dict[str, Any] = {
            "num_chunks":           num_chunks,
            "chunk_size":           K,
            "chunk_logits":         logit_data,
            "consistency_scores":   cons_scores,
            "consistency_weights":  weights,
            "dominant_chunk":       weights.index(max(weights)),
            "chunk_contributions":  chunk_contrib_data,
            "global_logits":        global_logits.data[:],
            "lambda":               self.lambda_,
        }

        return final_logits, explanation

    # ── Explanation rendering ────────────────────────────────────────────

    @staticmethod
    def render_explanation(explanation: Dict[str, Any]) -> str:
        """Return a human-readable explanation summary."""
        lines = ["── SEP Self-Explanation ──────────────────────"]
        nc    = explanation["num_chunks"]
        lines.append(
            f"  Sequence split into {nc} chunk(s) "
            f"of ~{explanation['chunk_size']} tokens each."
        )
        w = explanation["consistency_weights"]
        dc = explanation["dominant_chunk"]
        lines.append(f"  Dominant chunk: #{dc}  (weight={w[dc]:.3f})")
        for k in range(nc):
            lines.append(
                f"  Chunk {k:2d}:  consistency={explanation['consistency_scores'][k]:.3f} "
                f"  weight={w[k]:.3f}"
            )
        lines.append("──────────────────────────────────────────────")
        return "\n".join(lines)

    # ── Parameters ───────────────────────────────────────────────────────

    def parameters(self) -> List[NanoTensor]:
        return self.chunk_clf.parameters() + self.global_clf.parameters()

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()
