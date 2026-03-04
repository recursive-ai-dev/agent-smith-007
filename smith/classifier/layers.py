"""
Neural-Network Layers for AgentSmith
=====================================
All layers are pure-Python / NanoTensor.  No external dependencies.

Implemented:
  Linear          — affine transform y = Wx + b  (Kaiming init)
  LayerNorm       — LN(x) = γ(x−μ)/σ + β  (exact analytical gradient)
  ScaledDotProductAttention — Attention(Q,K,V) = softmax(QKᵀ/√d_k) V
  MultiHeadAttention        — h parallel attention heads + output projection
  FeedForward               — 2-layer MLP with GELU
  TransformerBlock          — pre-norm: LN → Attn → residual → LN → FFN → residual
  TokenEmbedding            — learnable lookup table (vocab_size × d_model)
  PositionalEncoding        — fixed sinusoidal positional bias
"""

import math
import random
from typing import List, Optional

from ..tensor import NanoTensor


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _kaiming_uniform(fan_in: int, fan_out: int) -> List[float]:
    """Kaiming uniform initialisation for weights: U(−a, a), a = √(2/fan_in)."""
    a = math.sqrt(2.0 / fan_in)
    return [random.uniform(-a, a) for _ in range(fan_out * fan_in)]


def _zeros(n: int) -> List[float]:
    return [0.0] * n


# ─────────────────────────────────────────────────────────────────────────────
# Linear layer
# ─────────────────────────────────────────────────────────────────────────────

class Linear:
    """
    Fully-connected linear transform:  y = W x + b

    Weight matrix W stored row-major as a flat NanoTensor of length
    (out_features × in_features).  NanoTensor.matmul(W, x) interprets
    W as a matrix and computes the matrix-vector product automatically.

    Gradient derivation:
        ∂L/∂x = Wᵀ ∂L/∂y      (propagated by NanoTensor.matmul backward)
        ∂L/∂W = ∂L/∂y ⊗ x     (idem)
        ∂L/∂b = ∂L/∂y          (idem via __add__ backward)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = NanoTensor(_kaiming_uniform(in_features, out_features))
        self.bias = NanoTensor(_zeros(out_features)) if bias else None

    def __call__(self, x: NanoTensor) -> NanoTensor:
        if len(x.data) != self.in_features:
            raise ValueError(
                f"Linear: expected in_features={self.in_features}, "
                f"got {len(x.data)}"
            )
        out = self.weight.matmul(x)
        if self.bias is not None:
            out = out + self.bias
        return out

    def parameters(self) -> List[NanoTensor]:
        return [self.weight] + ([self.bias] if self.bias is not None else [])

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def param_count(self) -> int:
        n = self.in_features * self.out_features
        if self.bias is not None:
            n += self.out_features
        return n


# ─────────────────────────────────────────────────────────────────────────────
# Layer Normalisation
# ─────────────────────────────────────────────────────────────────────────────

class LayerNorm:
    """
    Layer Normalisation:  LN(x) = γ ⊙ (x − μ)/σ + β

    μ = mean(x),  σ = sqrt(var(x) + ε)

    Backward (normalisation step, before scale/shift):
        Let x̂_i = (x_i − μ) / σ
        ∂L/∂x_i = (1/σ) [∂L/∂x̂_i
                         − mean_j(∂L/∂x̂_j)
                         − x̂_i · mean_j(∂L/∂x̂_j · x̂_j)]
    The scale (γ) and shift (β) gradients flow through NanoTensor __mul__/__add__.
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        self.d_model = d_model
        self.eps = eps
        self.gamma = NanoTensor([1.0] * d_model)   # scale
        self.beta = NanoTensor([0.0] * d_model)    # shift

    def __call__(self, x: NanoTensor) -> NanoTensor:
        n = len(x.data)
        mu = sum(x.data) / n
        var = sum((xi - mu) ** 2 for xi in x.data) / n
        sigma = math.sqrt(var + self.eps)

        x_hat_data = [(xi - mu) / sigma for xi in x.data]

        # Build NanoTensor with custom backward for the normalisation step
        x_hat = NanoTensor(x_hat_data[:], _parents=(x,), _op='ln_norm')

        def _backward_norm():
            if not x.requires_grad:
                return
            n_ = len(x.data)
            g = x_hat.grad
            g_sum = sum(g)
            g_xhat_dot = sum(g[i] * x_hat_data[i] for i in range(n_))
            for i in range(n_):
                dx = (g[i] - g_sum / n_ - x_hat_data[i] * g_xhat_dot / n_) / sigma
                x._accumulate_grad(i, dx)

        x_hat._backward = _backward_norm

        # Scale and shift (NanoTensor handles their gradients automatically)
        return x_hat * self.gamma + self.beta

    def parameters(self) -> List[NanoTensor]:
        return [self.gamma, self.beta]

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()


# ─────────────────────────────────────────────────────────────────────────────
# Scaled Dot-Product Attention (single head)
# ─────────────────────────────────────────────────────────────────────────────

class ScaledDotProductAttention:
    """
    Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V

    For a *single* query vector q ∈ ℝ^d_k, key list K=[k₁…k_T] and
    value list V=[v₁…v_T], returns context ∈ ℝ^d_v.

    The computation graph is:
        scores[j]   = dot(q, k_j) · scale          (NanoTensor.matmul)
        α           = softmax(scores)               (NanoTensor.softmax)
        context     = weighted_sum(α, V)            (NanoTensor.weighted_sum)

    All backward flows are handled automatically by NanoTensor.
    """

    def __init__(self, d_k: int):
        self._scale = NanoTensor([1.0 / math.sqrt(d_k)], requires_grad=False)

    def __call__(
        self,
        query: NanoTensor,
        keys: List[NanoTensor],
        values: List[NanoTensor],
    ) -> NanoTensor:
        seq_len = len(keys)

        # 1. Compute raw scores: s_j = q · k_j
        raw_scores = [query.matmul(kj) for kj in keys]  # list of [1] tensors

        # 2. Concatenate into [seq_len] tensor
        scores = raw_scores[0]
        for s in raw_scores[1:]:
            scores = scores.concat(s)

        # 3. Scale
        scale_t = NanoTensor([self._scale.data[0]] * seq_len, requires_grad=False)
        scores_scaled = scores * scale_t

        # 4. Softmax → attention weights [seq_len]
        attn_weights = scores_scaled.softmax()

        # 5. Weighted sum of values → context [d_v]
        return NanoTensor.weighted_sum(attn_weights, values)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Head Attention
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadAttention:
    """
    MultiHead(Q, K, V) = Concat(head₁…head_h) W_O
    head_i = Attention(Q W_Qi, K W_Ki, V W_Vi)

    Input : list of T NanoTensors, each ∈ ℝ^d_model
    Output: list of T NanoTensors, each ∈ ℝ^d_model

    Dimension bookkeeping:
        W_Q / W_K : d_k × d_model  (per head)
        W_V       : d_v × d_model  (per head)
        W_O       : d_model × (num_heads · d_v)
    """

    def __init__(self, d_model: int, num_heads: int, d_k: int, d_v: int):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = [Linear(d_model, d_k, bias=False) for _ in range(num_heads)]
        self.W_K = [Linear(d_model, d_k, bias=False) for _ in range(num_heads)]
        self.W_V = [Linear(d_model, d_v, bias=False) for _ in range(num_heads)]
        self.W_O = Linear(num_heads * d_v, d_model, bias=True)

        self._attn = ScaledDotProductAttention(d_k)

    def __call__(self, hidden_states: List[NanoTensor]) -> List[NanoTensor]:
        T = len(hidden_states)

        # Pre-project keys and values for every head (shared across query positions)
        all_keys   = [[self.W_K[h](tok) for tok in hidden_states]
                      for h in range(self.num_heads)]
        all_values = [[self.W_V[h](tok) for tok in hidden_states]
                      for h in range(self.num_heads)]

        outputs = []
        for tok in hidden_states:
            head_contexts = []
            for h in range(self.num_heads):
                q = self.W_Q[h](tok)
                ctx = self._attn(q, all_keys[h], all_values[h])
                head_contexts.append(ctx)

            # Concatenate heads → [num_heads · d_v]
            concat = head_contexts[0]
            for hc in head_contexts[1:]:
                concat = concat.concat(hc)

            # Project back to d_model
            outputs.append(self.W_O(concat))

        return outputs

    def parameters(self) -> List[NanoTensor]:
        params: List[NanoTensor] = []
        for h in range(self.num_heads):
            params.extend(self.W_Q[h].parameters())
            params.extend(self.W_K[h].parameters())
            params.extend(self.W_V[h].parameters())
        params.extend(self.W_O.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()


# ─────────────────────────────────────────────────────────────────────────────
# Feed-Forward Network
# ─────────────────────────────────────────────────────────────────────────────

class FeedForward:
    """
    Position-wise FFN:  FFN(x) = W₂ · GELU(W₁ x + b₁) + b₂

    GELU approximation: 0.5 x (1 + tanh(√(2/π)(x + 0.044715 x³)))
    Exact gradient computed by NanoTensor.gelu backward.
    """

    def __init__(self, d_model: int, d_ff: int):
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)

    def __call__(self, x: NanoTensor) -> NanoTensor:
        return self.fc2(self.fc1(x).gelu())

    def parameters(self) -> List[NanoTensor]:
        return self.fc1.parameters() + self.fc2.parameters()

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Block (pre-norm)
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock:
    """
    Pre-norm Transformer encoder block:

        h′  = h + MHA(LN(h))       # self-attention with residual
        h″  = h′ + FFN(LN(h′))     # feed-forward with residual

    Pre-norm (LN before sub-layer) gives more stable gradients than post-norm.
    """

    def __init__(self, d_model: int, num_heads: int, d_k: int, d_v: int, d_ff: int,
                 eps: float = 1e-5):
        self.attn  = MultiHeadAttention(d_model, num_heads, d_k, d_v)
        self.ff    = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model, eps)
        self.norm2 = LayerNorm(d_model, eps)

    def __call__(self, hidden_states: List[NanoTensor]) -> List[NanoTensor]:
        # ── Self-attention branch ──────────────────────────────────
        normed   = [self.norm1(h) for h in hidden_states]
        attn_out = self.attn(normed)
        hidden_states = [h + a for h, a in zip(hidden_states, attn_out)]

        # ── Feed-forward branch ────────────────────────────────────
        out = []
        for h in hidden_states:
            out.append(h + self.ff(self.norm2(h)))
        return out

    def parameters(self) -> List[NanoTensor]:
        return (self.attn.parameters() + self.ff.parameters()
                + self.norm1.parameters() + self.norm2.parameters())

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()


# ─────────────────────────────────────────────────────────────────────────────
# Token Embedding + Sinusoidal Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────

class TokenEmbedding:
    """
    Learnable lookup table: token_id → NanoTensor ∈ ℝ^d_model.
    Initialised with N(0, d_model^{-0.5}).
    """

    def __init__(self, vocab_size: int, d_model: int):
        self.vocab_size = vocab_size
        self.d_model = d_model
        scale = math.sqrt(1.0 / d_model)
        # One NanoTensor per vocabulary entry
        self._table: List[NanoTensor] = [
            NanoTensor([random.gauss(0.0, scale) for _ in range(d_model)])
            for _ in range(vocab_size)
        ]

    def __call__(self, token_id: int) -> NanoTensor:
        token_id = token_id % self.vocab_size
        return self._table[token_id]

    def parameters(self) -> List[NanoTensor]:
        return list(self._table)

    def zero_grad(self):
        for p in self._table:
            p.zero_grad()


class PositionalEncoding:
    """
    Fixed sinusoidal positional encoding (Vaswani et al. 2017):

        PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
        PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})

    Added to token embeddings as a non-learnable bias (requires_grad=False).
    """

    def __init__(self, d_model: int, max_len: int = 512):
        self.d_model = d_model
        self._cache: List[NanoTensor] = []
        for pos in range(max_len):
            pe = []
            for i in range(d_model):
                angle = pos / (10000.0 ** (2 * (i // 2) / d_model))
                pe.append(math.sin(angle) if i % 2 == 0 else math.cos(angle))
            self._cache.append(NanoTensor(pe, requires_grad=False))

    def __call__(self, pos: int) -> NanoTensor:
        return self._cache[pos]
