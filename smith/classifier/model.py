"""
AgentSmith — Multi-Domain Classification Model
================================================
Full forward pass:

  tokens → TokenEmbedding + PositionalEncoding
         → GSAR compression   (compresses frequent n-grams to symbols)
         → N × TransformerBlock   (self-attention + FFN)
         → SEP module   (chunked prediction + spurious-correlation detection)
         → softmax probabilities  +  SEP explanation

Architecture is parameterised entirely by AgentSmithConfig.
"""

import hashlib
import logging
import math
import re
from typing import List, Optional, Tuple, Dict, Any, Callable, Union

from ..tensor import NanoTensor
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

logger = logging.getLogger(__name__)


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
            # Use BLAKE2b for fast deterministic hashing
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            h = int.from_bytes(digest, "little") % (self.vocab_size - 2) + 2
            ids.append(h)
        # Pad sequence to max_len
        if len(ids) < self.max_len:
            ids.extend([self.PAD] * (self.max_len - len(ids)))
        return ids[:self.max_len]

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode multiple texts in a batch."""
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
    • GSAR: General Symbolic Arrays Reasoning (compression)
    • TransformerBlocks: Encoder stack
    • SEP: Self-Explanatory Perception (output & explanation)
    """

    def __init__(self, config: AgentSmithConfig):
        self.config = config

        # ── 1. Embeddings ──────────────────────────────────────────────
        self.token_emb = TokenEmbedding(config.vocab_size, config.d_model)
        self.pos_enc   = PositionalEncoding(config.d_model, config.max_seq_len)

        # ── 2. GSAR Compression ────────────────────────────────────────
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

        # ── 3. Transformer Stack ───────────────────────────────────────
        self.blocks = [
            TransformerBlock(
                d_model        = config.d_model,
                num_heads      = config.num_heads,
                d_k            = config.d_k,
                d_v            = config.d_v,
                d_ff           = config.d_ff,
                eps            = config.layer_norm_eps,
            )
            for _ in range(config.num_layers)
        ]

        # ── 4. Final Head ──────────────────────────────────────────────
        self.final_norm = LayerNorm(config.d_model, eps=config.layer_norm_eps)

        self.sep = SEP(
            d_model     = config.d_model,
            num_classes = config.num_classes,
            chunk_size  = config.sep_chunk_size,
            lambda_     = config.sep_spurious_lambda,
        )

        # ── Tokeniser ──────────────────────────────────────────────────
        self.tokenizer = Tokenizer(config.vocab_size, config.max_seq_len)

        # ── Hook registry (for diagnostics) ───────────────────────────
        self._forward_hooks:  List[Callable[[str, Any], None]] = []
        self._backward_hooks: List[Callable[[Any, Any], None]] = []

    # ── Forward pass ────────────────────────────────────────────────────

    def forward(
        self,
        token_ids: List[int],
        use_gsar: bool = True,
    ) -> Tuple[NanoTensor, NanoTensor, Dict[str, Any]]:
        """
        Run the full model forward pass.

        Returns
        -------
        logits      : NanoTensor [num_classes]
        probs       : NanoTensor [num_classes]
        diagnostics : dict with intermediate state and explanations
        """
        T = min(len(token_ids), self.config.max_seq_len)
        token_ids = token_ids[:T]

        # ── 1. Embedding + positional encoding ──────────────────────
        if use_gsar and self.gsar._registry:
            # GSAR-compressed embeddings
            emb_list, is_sym, patterns = self.gsar.compress(
                token_ids, self.token_emb
            )
            # Add positional encoding
            hidden = [emb_list[i] + self.pos_enc(i) for i in range(len(emb_list))]
        else:
            # Standard embedding + positional
            hidden = [
                self.token_emb(token_ids[i]) + self.pos_enc(i)
                for i in range(len(token_ids))
            ]
            is_sym   = [False] * len(token_ids)
            patterns = [None] * len(token_ids)

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
        self._fire_forward_hooks("sep_logits", logits.data)

        # ── 5. Softmax probabilities ───────────────────────────────
        probs = logits.softmax()

        # ── Diagnostics bundle ─────────────────────────────────────
        pred_class = probs.data.index(max(probs.data))
        diagnostics = {
            "sep": sep_explanation,
            "gsar_compressed_len": len(hidden),
            "gsar_original_len":   T,
            "gsar_compression_ratio": len(hidden) / max(T, 1),
            "gsar_symbol_positions": [i for i, s in enumerate(is_sym) if s],
            "predicted_class": pred_class,
            "predicted_domain": self.config.domains[pred_class],
        }

        return logits, probs, diagnostics

    def __call__(self, token_ids: List[int], use_gsar: bool = True):
        return self.forward(token_ids, use_gsar=use_gsar)

    # ── Convenience: predict from raw text ────────────────────────────

    def predict(self, text: str) -> Dict[str, Any]:
        """End-to-end inference from raw text."""
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
        Numerically stable cross-entropy loss with analytic backward pass.
        L = −logits[target] + log(Σ exp(logits))
        """
        C    = len(logits.data)
        if not 0 <= target_class < C:
            raise ValueError(f"target_class {target_class} out of range for {C} classes")

        maxv = max(logits.data)
        exp_shifted = [math.exp(xi - maxv) for xi in logits.data]
        sum_exp     = sum(exp_shifted)
        log_sum_exp = maxv + math.log(sum_exp + 1e-30)

        loss_val = log_sum_exp - logits.data[target_class]
        loss_nt  = NanoTensor([loss_val], _parents=(logits,), _op='xent')

        def _backward_xent():
            if logits.requires_grad:
                probs = [e / (sum_exp + 1e-30) for e in exp_shifted]
                for j in range(C):
                    # ∂L/∂logits[j] = softmax(logits)[j] − 1(j == target)
                    grad = (probs[j] - (1.0 if j == target_class else 0.0)) * loss_nt.grad[0]
                    logits._accumulate_grad(j, grad)

        loss_nt._backward = _backward_xent
        return loss_nt

    # ── Parameter management ──────────────────────────────────────────

    def parameters(self) -> List[NanoTensor]:
        """Aggregate all learnable parameters in the model."""
        params: List[NanoTensor] = []
        params.extend(self.token_emb.parameters())
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.final_norm.parameters())
        params.extend(self.sep.parameters())
        params.extend(self.gsar.parameters())
        return params

    def zero_grad(self):
        """Zero all gradients in the model's parameters."""
        for p in self.parameters():
            p.zero_grad()

    def param_count(self) -> int:
        """Total number of scalar parameters in the model."""
        return sum(len(p.data) for p in self.parameters())

    # ── Hooks ─────────────────────────────────────────────────────────

    def register_forward_hook(self, fn: Callable[[str, Any], None]):
        self._forward_hooks.append(fn)

    def register_backward_hook(self, fn: Callable[[Any, Any], None]):
        self._backward_hooks.append(fn)

    def _fire_forward_hooks(self, name: str, data):
        for fn in list(self._forward_hooks):
            try:
                fn(name, data)
            except Exception as e:
                logger.error("forward hook failed for %s: %s", name, e)

    def _fire_backward_hooks(self, name: str, grad_input, grad_output):
        for fn in list(self._backward_hooks):
            try:
                fn(grad_input, grad_output)
            except Exception as e:
                logger.error("backward hook failed for %s: %s", name, e)

    def backward(self, loss: NanoTensor):
        """Run backward pass and fire registered hooks."""
        loss.backward()
        self._fire_backward_hooks("loss", None, None)

    # ── Checkpoint (plain JSON-serialisable) ──────────────────────────

    def state_dict(self) -> Dict[str, List[float]]:
        """Snapshot all parameter values into a dictionary."""
        return {
            str(i): list(p.data)
            for i, p in enumerate(self.parameters())
        }

    def load_state_dict(self, sd: Dict[str, List[float]]):
        """Restore parameter values from a state_dict snapshot."""
        params = self.parameters()
        for i, p in enumerate(params):
            key = str(i)
            if key not in sd:
                raise RuntimeError(f"load_state_dict: missing key {key}")

            saved = list(sd[key])
            if len(saved) != len(p.data):
                raise ValueError(f"load_state_dict: size mismatch for key {key}")

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
