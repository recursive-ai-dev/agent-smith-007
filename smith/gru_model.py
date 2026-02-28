"""
Gated Recurrent Unit (GRU) - Advanced Recurrent Neural Network

Implements a GRU using only branchless operations and pattern matching
for control flow, replacing the simpler SymbolicRNN.
"""

import math
import random
from typing import List, Dict, Optional, Tuple

from .tensor import NanoTensor
from .pattern_matcher import PatternMatcher
from .database import SymbolicDB


class GatedRecurrentUnit:
    """
    A Gated Recurrent Unit (GRU) implementation using algebraic symbolism.
    This model uses gating mechanisms for improved long-range dependency handling.
    """

    def __init__(self, vocab_size: int, hidden_size: int, db: SymbolicDB):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.db = db

        # Initialize parameters for GRU gates (update, reset, candidate)
        scale = 0.1
        self.params = {
            # Update gate
            'W_zx': NanoTensor([random.uniform(-scale, scale) for _ in range(vocab_size * hidden_size)]),
            'W_zh': NanoTensor([random.uniform(-scale, scale) for _ in range(hidden_size * hidden_size)]),
            'b_z': NanoTensor([0.0] * hidden_size),
            # Reset gate
            'W_rx': NanoTensor([random.uniform(-scale, scale) for _ in range(vocab_size * hidden_size)]),
            'W_rh': NanoTensor([random.uniform(-scale, scale) for _ in range(hidden_size * hidden_size)]),
            'b_r': NanoTensor([0.0] * hidden_size),
            # Candidate hidden state
            'W_hx': NanoTensor([random.uniform(-scale, scale) for _ in range(vocab_size * hidden_size)]),
            'W_hh': NanoTensor([random.uniform(-scale, scale) for _ in range(hidden_size * hidden_size)]),
            'b_h': NanoTensor([0.0] * hidden_size),
            # Output layer
            'W_hy': NanoTensor([random.uniform(-scale, scale) for _ in range(hidden_size * vocab_size)]),
            'b_y': NanoTensor([0.0] * vocab_size)
        }

        self.pattern_matcher = PatternMatcher()

    def _embed(self, token_id: int) -> NanoTensor:
        """One-hot encoding using branchless selection"""
        embedding = [0.0] * self.vocab_size
        for i in range(self.vocab_size):
            diff_abs = abs(float(i - token_id))
            is_equal = 1.0 - float(diff_abs > 0.0)
            embedding[i] = is_equal
        return NanoTensor(embedding)

    def forward(self, inputs: List[int], h_prev: Optional[NanoTensor] = None) -> Tuple[NanoTensor, NanoTensor]:
        """
        Forward pass for the GRU model.
        Returns logits and the final hidden state.
        """
        if h_prev is None:
            h = NanoTensor([0.0] * self.hidden_size)
        else:
            h = h_prev

        for token_id in inputs:
            x = self._embed(token_id)

            # 1. Reset gate
            r_t_raw = self.params['W_rx'].matmul(x) + self.params['W_rh'].matmul(h) + self.params['b_r']
            r_t = r_t_raw.sigmoid()

            # 2. Update gate
            z_t_raw = self.params['W_zx'].matmul(x) + self.params['W_zh'].matmul(h) + self.params['b_z']
            z_t = z_t_raw.sigmoid()

            # 3. Candidate hidden state
            # Non-standard GELU activation is used here instead of the typical tanh.
            # This is an experimental choice for exploring alternative activation dynamics.
            h_tilde_raw = self.params['W_hx'].matmul(x) + self.params['W_hh'].matmul(r_t * h) + self.params['b_h']
            h_tilde = h_tilde_raw.gelu()

            # 4. Final hidden state
            # h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t
            one_minus_z = NanoTensor([1.0] * self.hidden_size) + (z_t * NanoTensor([-1.0] * self.hidden_size))
            h = (one_minus_z * h) + (z_t * h_tilde)

        # Compute output logits
        logits = self.params['W_hy'].matmul(h) + self.params['b_y']

        return logits, h

    def sample(self, logits: NanoTensor, temperature: float = 1.0) -> int:
        """Branchless sampling from logits"""
        scaled_logits = [l / temperature for l in logits.data]
        max_logit = max(scaled_logits)
        exp_logits = [math.exp(l - max_logit) for l in scaled_logits]
        sum_exp = sum(exp_logits)
        probs = [e / (sum_exp + 1e-8) for e in exp_logits]

        r = random.random()
        cumsum = 0.0
        selected = len(probs) - 1
        for i, p in enumerate(probs):
            cumsum += p
            is_selected = 1.0 if cumsum > r and selected == len(probs) - 1 else 0.0
            selected = int(is_selected * i + (1.0 - is_selected) * selected)
        return selected

    def generate(self, seed: str, length: int, temperature: float = 0.5) -> str:
        """Generate text using the GRU model"""
        config = self.pattern_matcher.match_generation_mode(f"sample_{temperature}")
        token_ids = [ord(c) % self.vocab_size for c in seed]

        h = None
        generated = list(seed)

        for _ in range(length):
            logits, h = self.forward(token_ids[-1:], h)
            next_id = self.sample(logits, temperature=config["temperature"])
            token_ids.append(next_id)
            char_code = next_id % 128
            generated.append(chr(char_code))

        return "".join(generated)

    def get_params(self) -> Dict[str, NanoTensor]:
        """Get model parameters"""
        return self.params

    def zero_grad(self):
        """Reset all parameter gradients"""
        for p in self.params.values():
            p.zero_grad()
