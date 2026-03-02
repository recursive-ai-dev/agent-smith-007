"""
Gated Recurrent Unit (GRU) - High-Fidelity Refactor
Includes Risk-Weighted Token Sampling for confidence < 90%.
"""

import math
import random
from typing import List, Dict, Optional, Tuple, Any

from .tensor import NanoTensor
from .pattern_matcher import PatternMatcher
from .database import SymbolicDB


class GatedRecurrentUnit:
    """
    GRU implementation with algebraic symbolism and risk-weighted sampling.
    """

    def __init__(self, vocab_size: int, hidden_size: int, db: SymbolicDB):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.db = db

        scale = 0.1
        self.params = {
            'W_zx': NanoTensor([random.uniform(-scale, scale) for _ in range(vocab_size * hidden_size)]),
            'W_zh': NanoTensor([random.uniform(-scale, scale) for _ in range(hidden_size * hidden_size)]),
            'b_z': NanoTensor([0.0] * hidden_size),
            'W_rx': NanoTensor([random.uniform(-scale, scale) for _ in range(vocab_size * hidden_size)]),
            'W_rh': NanoTensor([random.uniform(-scale, scale) for _ in range(hidden_size * hidden_size)]),
            'b_r': NanoTensor([0.0] * hidden_size),
            'W_hx': NanoTensor([random.uniform(-scale, scale) for _ in range(vocab_size * hidden_size)]),
            'W_hh': NanoTensor([random.uniform(-scale, scale) for _ in range(hidden_size * hidden_size)]),
            'b_h': NanoTensor([0.0] * hidden_size),
            'W_hy': NanoTensor([random.uniform(-scale, scale) for _ in range(hidden_size * vocab_size)]),
            'b_y': NanoTensor([0.0] * vocab_size)
        }

        self.pattern_matcher = PatternMatcher()

    def _embed(self, token_id: int) -> NanoTensor:
        """Branchless one-hot embedding."""
        embedding = [0.0] * self.vocab_size
        # Correctly find index using algebraic logic
        for i in range(self.vocab_size):
            diff = abs(float(i - (token_id % self.vocab_size)))
            is_equal = 1.0 - min(1.0, diff)
            embedding[i] = is_equal
        return NanoTensor(embedding, requires_grad=False)

    def forward(self, inputs: List[int], h_prev: Optional[NanoTensor] = None) -> Tuple[NanoTensor, NanoTensor]:
        """Forward pass with proper hidden state handling."""
        h = h_prev if h_prev else NanoTensor([0.0] * self.hidden_size, requires_grad=False)

        for token_id in inputs:
            x = self._embed(token_id)

            # Reset gate
            r_t = (self.params['W_rx'].matmul(x) + self.params['W_rh'].matmul(h) + self.params['b_r']).sigmoid()

            # Update gate
            z_t = (self.params['W_zx'].matmul(x) + self.params['W_zh'].matmul(h) + self.params['b_z']).sigmoid()

            # Candidate hidden state (using high-precision gelu)
            h_tilde = (self.params['W_hx'].matmul(x) + self.params['W_hh'].matmul(r_t * h) + self.params['b_h']).gelu()

            # Final hidden state: h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t
            one = NanoTensor([1.0] * self.hidden_size, requires_grad=False)
            h = ((one + (z_t * -1.0)) * h) + (z_t * h_tilde)

        logits = self.params['W_hy'].matmul(h) + self.params['b_y']
        return logits, h

    def sample_risk_weighted(self, logits: NanoTensor, temperature: float = 1.0) -> int:
        """
        Risk-Weighted Token Sampler.
        For confidence < 90%, it allows creative divergence based on risk/reward metrics.
        """
        scaled_logits = [l / (temperature + 1e-12) for l in logits.data]
        max_l = max(scaled_logits)
        exp_l = [math.exp(l - max_l) for l in scaled_logits]
        sum_e = sum(exp_l)
        probs = [e / (sum_e + 1e-12) for e in exp_l]

        # Determine confidence (max probability)
        confidence = max(probs)

        if confidence < 0.90:
            # Apply creative divergence: slightly boost lower probability tokens based on 'reward'
            # Reward heuristic: favor tokens with unique ASCII signatures (imaginary creative metric)
            reward_multiplier = [(1.0 + (i % 10) / 100.0) for i in range(len(probs))]
            probs = [p * r for p, r in zip(probs, reward_multiplier)]
            # Re-normalize
            sum_p = sum(probs)
            probs = [p / sum_p for p in probs]

        r = random.random()
        cumsum = 0.0
        for i, p in enumerate(probs):
            cumsum += p
            if r <= cumsum:
                return i
        return len(probs) - 1

    def generate(self, seed: str, length: int, temperature: float = 0.5) -> str:
        token_ids = [ord(c) % self.vocab_size for c in seed]
        h = None
        generated = list(seed)

        for _ in range(length):
            logits, h = self.forward(token_ids[-1:], h)
            # Detach h for generation to prevent gradient accumulation in long chains
            h = NanoTensor(h.data, requires_grad=False)

            next_id = self.sample_risk_weighted(logits, temperature=temperature)
            token_ids.append(next_id)
            generated.append(chr(next_id % 128))

        return "".join(generated)

    def zero_grad(self):
        for p in self.params.values():
            p.zero_grad()
