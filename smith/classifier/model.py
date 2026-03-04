"""
AgentSmith — Multi-Domain Classification Model
================================================
Full forward pass:

  tokens → TokenEmbedding + PositionalEncoding
         → GSAR compression   (compresses frequent n-grams to symbols)
         → N × TransformerBlock   (self-attention + FFN)
         → SEP module   (chunked prediction + spurious-correlation detection)
         → softmax probabilities  +  SEP explanation

Architecture is parameterised entirely by AgentSmithConfig.  The model
is domain-agnostic in structure; the 12-way classification targets are
listed in config.domains.

Parameter count (default config):
  TokenEmbedding   : vocab_size × d_model   = 4096 × 128  = 524 288
  PositionalEnc    : non-learnable           = 0
  per TransformerBlock:
    MHA (4 heads × 3 projections × d_k×d_model + W_O):
      = 4 × (32×128 + 32×128 + 32×128) + (128×128+128) = 49 280
    FFN (d_model→d_ff→d_model + biases):
      = 128×256+256 + 256×128+128       = 66 176
    LayerNorm ×2:  4 × d_model          =    512
    Block total                          ≈ 115 968
  3 blocks                               = 347 904
  GSAR symbol embeddings (up to 512):    = 512 × 128 = 65 536
  SEP classifiers (2 × d_model×C+C):    = 2×(128×12+12) = 3 096
  ─────────────────────────────────────────────────────
  Total (approx.)                        ≈ 940 824  (~1M params)

Scale d_model to 1024 and num_layers to 24 for ~1 billion parameters.
"""

import hashlib
import logging
import math
import re
from typing import List, Optional, Tuple, Dict, Any

from ..tensor import NanoTensor

logger = logging.getLogger(__name__)
from .config import AgentSmithConfig
from .layers import (
    TokenEmbedding,
    PositionalEncoding,
    TransformerBlock,
    LayerNorm,
    Linear,
)
from ..tools.gsar import GSAR
from ..tools.sep import SEP


# ─────────────────────────────────────────────────────────────────────────────
# Simple word-level tokeniser (no external deps)
# ─────────────────────────────────────────────────────────────────────────────

class Tokenizer:
    """
    Hash-based word tokeniser.
    Maps any Unicode word to an integer ID in [2, vocab_size−1].
    ID 0 = PAD, ID 1 = UNK (never emitted by hash, reserved for forced assign).
    """

    PAD = 0
    UNK = 1

    def __init__(self, vocab_size: int, max_len: int = 128):
        self.vocab_size = vocab_size
        self.max_len    = max_len

    def encode(self, text: str) -> List[int]:
        """Lowercase, split on Unicode word chars, deterministic-hash each token."""
        tokens = re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)
        ids = []
        for tok in tokens[: self.max_len]:
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            h = int.from_bytes(digest, "little") % (self.vocab_size - 2) + 2
            ids.append(h)
        # Pad if necessary
        ids = ids + [self.PAD] * (self.max_len - len(ids))
        return ids

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(t) for t in texts]


# ─────────────────────────────────────────────────────────────────────────────
# AgentSmith Model
# ─────────────────────────────────────────────────────────────────────────────

class AgentSmith:
    """
    Trainable multi-domain text classifier.

    Components
    ----------
    • TokenEmbedding + sinusoidal PositionalEncoding
    • GSAR layer (symbolic n-gram compression)
    • N × TransformerBlock (pre-norm MHA + FFN)
    • Final LayerNorm
    • SEP module (chunked perception, spurious-correlation filtering)
    • Output: class logits + softmax probabilities + SEP explanation
    """

    def __init__(self, config: AgentSmithConfig):
        self.config = config

        # ── Embedding ──────────────────────────────────────────────────
        self.token_emb = TokenEmbedding(config.vocab_size, config.d_model)
        self.pos_enc   = PositionalEncoding(config.d_model, config.max_seq_len)

        # ── GSAR ───────────────────────────────────────────────────────
        self.gsar = GSAR(
            d_model            = config.d_model,
            vocab_size         = config.vocab_size,
            window_sizes       = config.gsar_window_sizes,
            priority_threshold = config.gsar_priority_threshold,
            temperature        = config.gsar_temperature,
            min_freq           = config.gsar_min_freq,
            max_symbols        = config.gsar_max_symbols,
            blend_alpha        = config.gsar_blend_alpha,
        )

        # ── Transformer stack ──────────────────────────────────────────
        self.blocks = [
            TransformerBlock(
                d_model   = config.d_model,
                num_heads = config.num_heads,
                d_k       = config.d_k,
                d_v       = config.d_v,
                d_ff      = config.d_ff,
                eps       = config.layer_norm_eps,
            )
            for _ in range(config.num_layers)
        ]
        self.final_norm = LayerNorm(config.d_model, config.layer_norm_eps)

        # ── SEP ────────────────────────────────────────────────────────
        self.sep = SEP(
            d_model     = config.d_model,
            num_classes = config.num_classes,
            chunk_size  = config.sep_chunk_size,
            lambda_     = config.sep_spurious_lambda,
        )

        # ── Tokeniser ──────────────────────────────────────────────────
        self.tokenizer = Tokenizer(config.vocab_size, config.max_seq_len)

        # ── Hook registry (for diagnostics) ───────────────────────────
        # Each hook: callable(layer_name, tensor_data) → None
        self._forward_hooks:  List[callable] = []
        self._backward_hooks: List[callable] = []

    # ── Forward pass ────────────────────────────────────────────────────

    def forward(
        self,
        token_ids: List[int],
        use_gsar: bool = True,
    ) -> Tuple[NanoTensor, NanoTensor, Dict[str, Any]]:
        """
        Parameters
        ----------
        token_ids : integer token IDs, length ≤ max_seq_len
        use_gsar  : whether to apply GSAR compression (default True)

        Returns
        -------
        logits      : NanoTensor [num_classes]  (raw pre-softmax)
        probs       : NanoTensor [num_classes]  (softmax probabilities)
        diagnostics : dict with intermediate tensors and SEP explanation
        """
        T = min(len(token_ids), self.config.max_seq_len)
        token_ids = token_ids[:T]

        # ── 1. Embedding + positional encoding ──────────────────────
        if use_gsar and self.gsar._registry:
            # GSAR-compressed embeddings (variable length ≤ T)
            emb_list, is_sym, patterns = self.gsar.compress(
                token_ids, self.token_emb
            )
            # Add positional encoding to each position
            hidden = [emb_list[i] + self.pos_enc(i) for i in range(len(emb_list))]
        else:
            # Standard embedding + positional
            hidden = [
                self.token_emb(token_ids[i]) + self.pos_enc(i)
                for i in range(T)
            ]
            is_sym   = [False] * T
            patterns = [None] * T

        # Fire forward hook: embeddings
        self._fire_forward_hooks("embeddings", [h.data for h in hidden])

        # ── 2. Transformer stack ───────────────────────────────────
        for layer_idx, block in enumerate(self.blocks):
            hidden = block(hidden)
            self._fire_forward_hooks(
                f"block_{layer_idx}", [h.data for h in hidden]
            )

        # ── 3. Final layer norm ────────────────────────────────────
        hidden = [self.final_norm(h) for h in hidden]

        # ── 4. SEP module ──────────────────────────────────────────
        logits, sep_explanation = self.sep.forward(hidden)
        self._fire_forward_hooks("sep_logits", logits.data[:])

        # ── 5. Softmax probabilities ───────────────────────────────
        probs = logits.softmax()

        # ── Diagnostics bundle ─────────────────────────────────────
        diagnostics = {
            "sep": sep_explanation,
            "gsar_compressed_len": len(hidden),
            "gsar_original_len":   T,
            "gsar_compression_ratio": len(hidden) / max(T, 1),
            "gsar_symbol_positions": [i for i, s in enumerate(is_sym) if s],
            "predicted_class": probs.data.index(max(probs.data)),
            "predicted_domain": self.config.domains[probs.data.index(max(probs.data))],
        }

        return logits, probs, diagnostics

    def __call__(self, token_ids: List[int], use_gsar: bool = True):
        return self.forward(token_ids, use_gsar=use_gsar)

    # ── Convenience: predict from raw text ────────────────────────────

    def predict(self, text: str) -> Dict[str, Any]:
        """
        End-to-end inference from raw text.

        Returns dict with:
          label     : predicted domain name
          class_id  : predicted class index
          probs     : list of class probabilities
          confidence: max probability
          explanation: SEP explanation string
          diagnostics: full diagnostics dict
        """
        token_ids = self.tokenizer.encode(text)
        logits, probs, diagnostics = self.forward(token_ids)

        pred_id = probs.data.index(max(probs.data))
        return {
            "label":       self.config.domains[pred_id],
            "class_id":    pred_id,
            "probs":       probs.data[:],
            "confidence":  max(probs.data),
            "explanation": SEP.render_explanation(diagnostics["sep"]),
            "diagnostics": diagnostics,
        }

    # ── Cross-entropy loss ────────────────────────────────────────────

    def cross_entropy_loss(
        self,
        logits: NanoTensor,
        target_class: int,
    ) -> NanoTensor:
        """
        Cross-entropy loss:
            L = −log softmax(logits)[target]
              = −logits[target] + log(Σ_j exp(logits[j]))

        Numerically stable via log-sum-exp:
            L = −logits[target] + max + log(Σ_j exp(logits[j] − max))
        """
        C    = len(logits.data)
        maxv = max(logits.data)

        # Σ exp(logits[j] − max)
        exp_shifted = [math.exp(logits.data[j] - maxv) for j in range(C)]
        sum_exp     = sum(exp_shifted)
        log_sum_exp = maxv + math.log(sum_exp + 1e-30)

        # −logits[target]
        neg_target  = logits.extract(target_class, target_class + 1)
        neg_target  = neg_target * NanoTensor([-1.0], requires_grad=False)

        # log_sum_exp as a constant NanoTensor (no gradient)
        lse_t = NanoTensor([log_sum_exp], requires_grad=False)

        loss = neg_target + lse_t

        # Backprop gradient through neg_target only (lse_t carries the
        # full softmax gradient implicitly):
        # ∂L/∂logits[j] = softmax(logits)[j] − 1(j == target)
        # This is achieved by neg_target.backward() + a custom correction.
        # We build a proper graph node that captures the full analytic gradient.

        # ── Re-implement with full graph for correctness ──────────────
        # Build a NanoTensor that represents the full cross-entropy,
        # with the correct analytical backward.
        loss_val = log_sum_exp - logits.data[target_class]
        loss_nt  = NanoTensor([loss_val], _parents=(logits,), _op='xent')

        def _backward_xent():
            if logits.requires_grad:
                probs = [e / (sum_exp + 1e-30) for e in exp_shifted]
                for j in range(C):
                    grad = probs[j] - (1.0 if j == target_class else 0.0)
                    logits._accumulate_grad(j, grad * loss_nt.grad[0])

        loss_nt._backward = _backward_xent
        return loss_nt

    # ── Parameter management ──────────────────────────────────────────

    def parameters(self) -> List[NanoTensor]:
        params: List[NanoTensor] = []
        params.extend(self.token_emb.parameters())
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.final_norm.parameters())
        params.extend(self.sep.parameters())
        params.extend(self.gsar.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def param_count(self) -> int:
        return sum(len(p.data) for p in self.parameters())

    # ── Hooks ─────────────────────────────────────────────────────────

    def register_forward_hook(self, fn: callable):
        """Register a hook called as fn(layer_name, activation_data)."""
        self._forward_hooks.append(fn)

    def register_backward_hook(self, fn: callable):
        """Register a hook called after backward as fn(layer_name, grad_data)."""
        self._backward_hooks.append(fn)

    def _fire_forward_hooks(self, name: str, data):
        for fn in list(self._forward_hooks):
            try:
                fn(name, data)
            except Exception as e:
                logger.error("forward hook %s failed for %s: %s", fn, name, e, exc_info=True)

    def _fire_backward_hooks(self, name: str, grad_input, grad_output):
        for fn in list(self._backward_hooks):
            try:
                fn(grad_input, grad_output)
            except Exception as e:
                logger.error("backward hook %s failed for %s: %s", fn, name, e, exc_info=True)

    def backward(self, loss):
        """Run backward pass on loss and fire all registered backward hooks."""
        loss.backward()
        self._fire_backward_hooks("loss", None, None)

    # ── Checkpoint (plain JSON-serialisable) ──────────────────────────

    def state_dict(self) -> dict:
        """Serialise all parameter values."""
        return {
            str(i): list(p.data)
            for i, p in enumerate(self.parameters())
        }

    def load_state_dict(self, sd: dict):
        """Restore parameter values from state_dict."""
        params = self.parameters()
        missing = [str(i) for i in range(len(params)) if str(i) not in sd]
        if missing:
            raise RuntimeError(
                f"load_state_dict: checkpoint is missing parameters: {missing}"
            )
        for i, p in enumerate(params):
            key = str(i)
            saved = list(sd[key])
            if len(saved) != len(p.data):
                raise ValueError(
                    f"load_state_dict: parameter {key} size mismatch -- "
                    f"expected {len(p.data)}, got {len(saved)}"
                )
            for j in range(len(p.data)):
                p.data[j] = saved[j]

    # ── String representation ─────────────────────────────────────────

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"AgentSmith(\n"
            f"  d_model={cfg.d_model}, num_layers={cfg.num_layers}, "
            f"num_heads={cfg.num_heads}, d_ff={cfg.d_ff}\n"
            f"  vocab_size={cfg.vocab_size}, num_classes={cfg.num_classes}\n"
            f"  total_params={self.param_count():,}\n"
            f"  gsar_symbols={len(self.gsar._registry)}, "
            f"sep_chunk_size={cfg.sep_chunk_size}\n"
            f")"
        )
