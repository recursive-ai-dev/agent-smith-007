"""
SEP — Self-Explanatory Perception
=================================
Chunked prediction head with Spurious Correlation Detection.

SEP divides the hidden state sequence into chunks, computes a prediction
for each, and then measures the consistency of these predictions.
A 'Consistency Mask' is used to down-weight chunks that diverge from the
global consensus, effectively suppressing spurious local signals.
"""

import math
from typing import List, Tuple, Dict, Any

from ..tensor import NanoTensor
from ..classifier.layers import Linear, LayerNorm


class SEP:
    """
    Self-explanatory output module.
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        chunk_size: int = 16,
        lambda_: float = 0.1
    ):
        self.d_model = d_model
        self.num_classes = num_classes
        self.chunk_size = chunk_size
        self.lambda_ = lambda_  # Consistency penalty

        # Output projection (shared across chunks)
        self.projector = Linear(d_model, num_classes, bias=True)
        # Prediction refinement norm
        self.norm = LayerNorm(num_classes)

    def forward(self, hidden_states: List[NanoTensor]) -> Tuple[NanoTensor, Dict[str, Any]]:
        """
        Computes weighted global logits and explanation metadata.
        """
        T = len(hidden_states)
        if T == 0:
            return NanoTensor([0.0] * self.num_classes), {}

        # 1. Chunk hidden states
        chunks = [
            hidden_states[i : i + self.chunk_size]
            for i in range(0, T, self.chunk_size)
        ]

        # 2. Compute local predictions (mean of states in chunk projected)
        chunk_logits: List[NanoTensor] = []
        for c in chunks:
            # Mean state in chunk
            c_mean = c[0]
            for i in range(1, len(c)):
                c_mean = c_mean + c[i]
            c_mean = c_mean / len(c)

            chunk_logits.append(self.projector(c_mean))

        # 3. Compute global consensus
        global_logits = chunk_logits[0]
        for i in range(1, len(chunk_logits)):
            global_logits = global_logits + chunk_logits[i]
        global_logits = global_logits / len(chunk_logits)

        # 4. Consistency Scoring (how much does each chunk agree with global?)
        # score_i = exp(-lambda * ||logits_i - global||^2)
        scores = []
        for cl in chunk_logits:
            diff_sq = sum((cl.data[j] - global_logits.data[j])**2 for j in range(self.num_classes))
            s = math.exp(-self.lambda_ * diff_sq)
            scores.append(s)

        # Re-normalize scores to sum to 1.0
        sum_s = sum(scores) + 1e-12
        weights = [s / sum_s for s in scores]

        # 5. Final weighted logits
        final_logits = chunk_logits[0] * weights[0]
        for i in range(1, len(chunk_logits)):
            final_logits = final_logits + (chunk_logits[i] * weights[i])

        # 6. Metadata for explanation
        explanation = {
            "chunk_weights": weights,
            "chunk_predictions": [cl.data.index(max(cl.data)) for cl in chunk_logits],
            "consensus_score": sum_s / len(scores),
        }

        return final_logits, explanation

    @staticmethod
    def render_explanation(exp: Dict[str, Any]) -> str:
        """Format explanation metadata for humans."""
        if not exp: return "No signal detected."

        weights = exp.get("chunk_weights", [])
        preds = exp.get("chunk_predictions", [])

        lines = ["SEP Signal Analysis:"]
        for i, (w, p) in enumerate(zip(weights, preds)):
            bar = "█" * int(w * 20)
            lines.append(f"  Chunk {i:02}: [{bar:<20}] Class {p} (Weight: {w:.3f})")

        score = exp.get("consensus_score", 0.0)
        lines.append(f"Consensus Robustness: {score:.4f}")
        return "\n".join(lines)

    def parameters(self) -> List[NanoTensor]:
        return self.projector.parameters() + self.norm.parameters()
