"""
Gated Recurrent Unit (GRU) - High-Fidelity Refactor
===================================================
Implementation of a Gated Recurrent Unit using NanoTensor autograd.
Includes Risk-Weighted Token Sampling for creative divergence.

Architecture:
  r_t = sigmoid(W_rx x_t + W_rh h_{t-1} + b_r)
  z_t = sigmoid(W_zx x_t + W_zh h_{t-1} + b_z)
  n_t = gelu(W_hx x_t + W_hh (r_t * h_{t-1}) + b_h)
  h_t = (1 - z_t) * h_{t-1} + z_t * n_t
"""

import math
import random
import logging
from typing import List, Dict, Optional, Tuple, Any

from .tensor import NanoTensor
from .pattern_matcher import PatternMatcher
from .database import SymbolicDB

logger = logging.getLogger(__name__)

class GatedRecurrentUnit:
    """
    GRU implementation optimized for algebraic symbolism and stability.
    """

    def __init__(self, vocab_size: int, hidden_size: int, db: Optional[SymbolicDB] = None):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.db = db

        # Kaiming-like initialization for GRU weights
        scale = math.sqrt(2.0 / (vocab_size + hidden_size))

        self.params: Dict[str, NanoTensor] = {
            'W_zx': NanoTensor([random.uniform(-scale, scale) for _ in range(vocab_size * hidden_size)]),
            'W_zh': NanoTensor([random.uniform(-scale, scale) for _ in range(hidden_size * hidden_size)]),
            'b_z':  NanoTensor([0.0] * hidden_size),

            'W_rx': NanoTensor([random.uniform(-scale, scale) for _ in range(vocab_size * hidden_size)]),
            'W_rh': NanoTensor([random.uniform(-scale, scale) for _ in range(hidden_size * hidden_size)]),
            'b_r':  NanoTensor([0.0] * hidden_size),

            'W_hx': NanoTensor([random.uniform(-scale, scale) for _ in range(vocab_size * hidden_size)]),
            'W_hh': NanoTensor([random.uniform(-scale, scale) for _ in range(hidden_size * hidden_size)]),
            'b_h':  NanoTensor([0.0] * hidden_size),

            'W_hy': NanoTensor([random.uniform(-scale, scale) for _ in range(hidden_size * vocab_size)]),
            'b_y':  NanoTensor([0.0] * vocab_size)
        }

        self.pattern_matcher = PatternMatcher()

    def _embed(self, token_id: int) -> NanoTensor:
        """
        Algebraic one-hot embedding.
        Maintains branchless property while encoding token identity.
        """
        embedding = [0.0] * self.vocab_size
        target_idx = int(token_id) % self.vocab_size

        # Pure algebraic one-hot encoding
        for i in range(self.vocab_size):
            # 1.0 if i == target_idx, else 0.0
            diff = abs(float(i - target_idx))
            is_equal = 1.0 - min(1.0, diff)
            embedding[i] = is_equal

        return NanoTensor(embedding, requires_grad=False)

    def forward(self, inputs: List[int], h_prev: Optional[NanoTensor] = None) -> Tuple[NanoTensor, NanoTensor]:
        """
        Sequence forward pass through the GRU.

        inputs: List of token IDs
        h_prev: Initial hidden state [hidden_size]

        Returns:
          logits: prediction for the final token in the input sequence
          h: final hidden state
        """
        h = h_prev if h_prev is not None else NanoTensor([0.0] * self.hidden_size, requires_grad=False)

        logits = None
        for token_id in inputs:
            x = self._embed(token_id)

            # 1. Reset gate: determines how much of past to forget
            r_t = (self.params['W_rx'].matmul(x) + self.params['W_rh'].matmul(h) + self.params['b_r']).sigmoid()

            # 2. Update gate: determines how much new info to keep
            z_t = (self.params['W_zx'].matmul(x) + self.params['W_zh'].matmul(h) + self.params['b_z']).sigmoid()

            # 3. Candidate hidden state
            # Note: r_t * h is element-wise gating of the previous hidden state
            h_tilde = (self.params['W_hx'].matmul(x) + self.params['W_hh'].matmul(r_t * h) + self.params['b_h']).gelu()

            # 4. Final hidden state interpolation: h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde
            # We use (1 + (-z_t)) to stay within NanoTensor supported ops
            h = ((1.0 + (-z_t)) * h) + (z_t * h_tilde)

            # 5. Output projection
            logits = self.params['W_hy'].matmul(h) + self.params['b_y']

        if logits is None:
            # Fallback for empty input sequence
            logits = self.params['b_y']

        return logits, h

    def sample_risk_weighted(self, logits: NanoTensor, temperature: float = 1.0) -> int:
        """
        Probabilistic sampling with 'Risk-Weighted' creative divergence.
        If max confidence is low, it injects a reward-based heuristic to favor
        tokens with higher semantic novelty (simulated via ASCII signatures).
        """
        # Apply temperature
        t_safe = max(temperature, 1e-6)
        scaled_data = [l / t_safe for l in logits.data]

        # Softmax in raw Python for efficiency during sampling
        max_l = max(scaled_data)
        exp_l = [math.exp(l - max_l) for l in scaled_data]
        sum_e = sum(exp_l)
        probs = [e / (sum_e + 1e-12) for e in exp_l]

        confidence = max(probs)

        # Creative divergence for low-confidence tokens
        if confidence < 0.90:
            # Heuristic: boost tokens based on positional diversity
            reward_multiplier = [(1.0 + (i % 7) / 50.0) for i in range(len(probs))]
            probs = [p * r for p, r in zip(probs, reward_multiplier)]
            # Re-normalize
            sum_p = sum(probs)
            probs = [p / sum_p for p in probs]

        # Categorical sampling
        r = random.random()
        cumsum = 0.0
        for i, p in enumerate(probs):
            cumsum += p
            if r <= cumsum:
                return i
        return len(probs) - 1

    def generate(self, seed: str, length: int, temperature: float = 0.7) -> str:
        """
        Generate a text sequence from a starting seed string.
        """
        token_ids = [ord(c) % self.vocab_size for c in seed]
        h = None

        # Initial context processing
        if seed:
            _, h = self.forward(token_ids, h)
            # Detach hidden state to prevent gradient accumulation during generation
            h = NanoTensor(h.data, requires_grad=False)

        generated_chars = list(seed)

        # Iterative generation
        for _ in range(length):
            # Input is the last token generated
            curr_input = [token_ids[-1]]
            logits, h = self.forward(curr_input, h)
            h = NanoTensor(h.data, requires_grad=False)

            next_id = self.sample_risk_weighted(logits, temperature=temperature)
            token_ids.append(next_id)
            # Map ID back to ASCII for simplicity
            generated_chars.append(chr(next_id % 128))

        return "".join(generated_chars)

    def zero_grad(self):
        """Reset gradients for all model parameters."""
        for p in self.params.values():
            p.zero_grad()

    def param_count(self) -> int:
        """Total number of scalar parameters in the GRU."""
        return sum(len(p.data) for p in self.params.values())
