"""
GSAR — General Symbolic Arrays Reasoning
==========================================
GSAR assigns a *priority rating* to observed token n-gram combinations.
High-priority combinations (frequently co-occurring sequences) are
registered in a symbol vocabulary and, during inference, are compressed
into a single symbolic embedding rather than being re-parsed individually.

Core idea
---------
Natural language contains enormous amounts of repetition at the phrase
level.  Once the model has confidently associated a particular phrase
pattern with a semantic concept, spending attention on every individual
token in that phrase is wasteful.  GSAR short-circuits this by:

  1. Monitoring token n-gram co-occurrence counts across training batches.
  2. Computing a priority score  P(w) = σ(T · freq(w))  where
       freq(w) = count(w) / total_windows
       σ       = sigmoid
       T       = temperature (sharpness of discrimination)
  3. Registering patterns with P(w) ≥ priority_threshold as *symbols*.
  4. At forward time, scanning each input sequence for registered patterns.
     A matched high-priority window is replaced by its symbol embedding:
       emb_out = α · emb_symbol + (1−α) · mean(emb_tokens)
     where α = priority_score (blend coefficient — smooth interpolation).
  5. The symbol embeddings are learnable parameters, updated via backprop.

Symbol embeddings
-----------------
Each registered symbol s has a learnable embedding vector e_s ∈ ℝ^d_model.
Initialised as the mean of the constituent token embeddings at registration
time, then fine-tuned through normal gradient descent.

Effect on sequence length
--------------------------
Compressing a k-gram to one symbol reduces sequence length by (k−1).
Shorter sequences reduce attention complexity from O(L²) to O(L′²).
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ..tensor import NanoTensor


# ─────────────────────────────────────────────────────────────────────────────
# SymbolEntry
# ─────────────────────────────────────────────────────────────────────────────

class SymbolEntry:
    """Metadata for a single registered symbolic pattern."""

    __slots__ = ("symbol_id", "pattern", "priority", "count", "window_size")

    def __init__(
        self,
        symbol_id: int,
        pattern: Tuple[int, ...],
        priority: float,
        count: int,
    ):
        self.symbol_id   = symbol_id
        self.pattern     = pattern
        self.priority    = priority
        self.count       = count
        self.window_size = len(pattern)

    def __repr__(self):
        return (
            f"SymbolEntry(id={self.symbol_id}, "
            f"pattern={self.pattern}, priority={self.priority:.3f}, "
            f"count={self.count})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# GSAR  (main class)
# ─────────────────────────────────────────────────────────────────────────────

class GSAR:
    """
    General Symbolic Arrays Reasoning module.

    Parameters
    ----------
    d_model           : embedding dimension (must match rest of model)
    vocab_size        : base token vocabulary size
    window_sizes      : list of n-gram sizes to monitor  (e.g. [2, 3, 4])
    priority_threshold: minimum P(w) to register a symbol
    temperature       : sigmoid temperature T
    min_freq          : minimum raw count before registration is attempted
    max_symbols       : upper bound on symbol vocabulary size
    blend_alpha       : fixed blending weight; if None, uses live priority
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        window_sizes: List[int] = None,
        priority_threshold: float = 0.60,
        temperature: float = 8.0,
        min_freq: int = 3,
        max_symbols: int = 512,
        blend_alpha: Optional[float] = 0.85,
    ):
        self.d_model            = d_model
        self.vocab_size         = vocab_size
        self.window_sizes       = sorted(window_sizes or [2, 3, 4], reverse=True)
        self.priority_threshold = priority_threshold
        self.temperature        = temperature
        self.min_freq           = min_freq
        self.max_symbols        = max_symbols
        self.blend_alpha        = blend_alpha

        # ── Co-occurrence statistics (non-differentiable, updated each batch)
        self._counts: Dict[Tuple[int, ...], int] = defaultdict(int)
        self._total_windows: int = 0

        # ── Symbol registry: pattern → SymbolEntry
        self._registry: Dict[Tuple[int, ...], SymbolEntry] = {}

        # ── Learnable symbol embeddings: list[NanoTensor ∈ ℝ^d_model]
        #    Grown dynamically as symbols are registered.
        self._sym_embeddings: List[NanoTensor] = []

    # ── Statistics update ────────────────────────────────────────────────

    def update_statistics(self, sequences: List[List[int]]) -> int:
        """
        Ingest a batch of tokenised sequences and update co-occurrence counts.

        Parameters
        ----------
        sequences : list of token-ID lists (one per sample in the batch)

        Returns the number of newly registered symbols.
        """
        for seq in sequences:
            for ws in self.window_sizes:
                for i in range(len(seq) - ws + 1):
                    w = tuple(seq[i: i + ws])
                    self._counts[w] += 1
                    self._total_windows += 1

        return self._register_new_symbols()

    def _compute_priority(self, count: int) -> float:
        """P(w) = σ(T · count / total_windows)."""
        if self._total_windows == 0:
            return 0.0
        freq = count / self._total_windows
        x    = self.temperature * freq
        # Numerically stable sigmoid
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        else:
            ex = math.exp(x)
            return ex / (1.0 + ex)

    def _register_new_symbols(self) -> int:
        """Scan counts and register patterns that cross the threshold."""
        newly_registered = 0
        # Sort by count descending; register the most frequent first
        for pattern, count in sorted(self._counts.items(), key=lambda kv: -kv[1]):
            if len(self._registry) >= self.max_symbols:
                break
            if pattern in self._registry:
                # Refresh priority if count grew
                self._registry[pattern].priority = self._compute_priority(count)
                self._registry[pattern].count    = count
                continue
            if count < self.min_freq:
                continue
            p = self._compute_priority(count)
            if p >= self.priority_threshold:
                sym_id = len(self._sym_embeddings)
                entry  = SymbolEntry(sym_id, pattern, p, count)
                self._registry[pattern] = entry
                # Initialise embedding to zeros (will be set from token embs below)
                self._sym_embeddings.append(
                    NanoTensor([0.0] * self.d_model)
                )
                newly_registered += 1

        return newly_registered

    def initialise_symbol_embedding(
        self,
        pattern: Tuple[int, ...],
        token_embeddings_fn,
    ):
        """
        Set the symbol embedding to the mean of constituent token embeddings
        (called once when the symbol is first used in a forward pass).

        token_embeddings_fn : callable(token_id) → NanoTensor [d_model]
        """
        if pattern not in self._registry:
            return
        entry = self._registry[pattern]
        if entry.symbol_id >= len(self._sym_embeddings):
            return
        mean_data = [0.0] * self.d_model
        for tok_id in pattern:
            emb = token_embeddings_fn(tok_id)
            for d in range(self.d_model):
                mean_data[d] += emb.data[d]
        n = len(pattern)
        mean_data = [x / n for x in mean_data]
        self._sym_embeddings[entry.symbol_id] = NanoTensor(mean_data)

    # ── Forward: compress a single token sequence ────────────────────────

    def compress(
        self,
        token_ids: List[int],
        embed_fn,
    ) -> Tuple[List[NanoTensor], List[bool], List[Optional[Tuple[int, ...]]]]:
        """
        Scan `token_ids` and replace high-priority n-grams with their
        symbol embedding.  Tokens not part of any registered pattern pass
        through as normal embeddings.

        Parameters
        ----------
        token_ids : integer token IDs for one sequence
        embed_fn  : callable(token_id) → NanoTensor [d_model]
                    (the model's token embedding lookup)

        Returns
        -------
        embeddings : list of NanoTensor, each [d_model]
                     (length ≤ len(token_ids) due to compression)
        is_symbol  : bool flag per position (True = compressed symbol)
        patterns   : which pattern was matched (None for plain tokens)
        """
        embeddings: List[NanoTensor]         = []
        is_symbol:  List[bool]               = []
        patterns:   List[Optional[Tuple]]    = []

        i = 0
        L = len(token_ids)

        while i < L:
            matched = False

            # Try longest window first (greedy leftmost)
            for ws in self.window_sizes:
                if i + ws > L:
                    continue
                window = tuple(token_ids[i: i + ws])
                if window not in self._registry:
                    continue

                entry = self._registry[window]
                if entry.priority < self.priority_threshold:
                    continue

                # ── Retrieve or initialise symbol embedding
                sym_emb = self._sym_embeddings[entry.symbol_id]
                if all(v == 0.0 for v in sym_emb.data):
                    self.initialise_symbol_embedding(window, embed_fn)
                    sym_emb = self._sym_embeddings[entry.symbol_id]

                # ── Blend: α·sym + (1−α)·mean(token_embs)
                alpha   = self.blend_alpha if self.blend_alpha is not None else entry.priority
                tok_embs = [embed_fn(tid) for tid in window]
                mean_data = [
                    sum(te.data[d] for te in tok_embs) / ws
                    for d in range(self.d_model)
                ]

                blended_data = [
                    alpha * sym_emb.data[d] + (1.0 - alpha) * mean_data[d]
                    for d in range(self.d_model)
                ]

                # Build blended NanoTensor with gradient path through sym_emb
                # (token embeddings also contribute through (1−α) branch)
                mean_nt  = NanoTensor(mean_data, requires_grad=False)  # detached
                alpha_t  = NanoTensor([alpha]   * self.d_model, requires_grad=False)
                nalpha_t = NanoTensor([1.0 - alpha] * self.d_model, requires_grad=False)
                blended  = sym_emb * alpha_t + mean_nt * nalpha_t

                embeddings.append(blended)
                is_symbol.append(True)
                patterns.append(window)
                i += ws
                matched = True
                break

            if not matched:
                embeddings.append(embed_fn(token_ids[i]))
                is_symbol.append(False)
                patterns.append(None)
                i += 1

        return embeddings, is_symbol, patterns

    # ── Diagnostics ──────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return a summary of the symbol registry state."""
        if not self._registry:
            return {"registered_symbols": 0, "total_windows_seen": self._total_windows}

        priorities = [e.priority for e in self._registry.values()]
        counts     = [e.count    for e in self._registry.values()]
        return {
            "registered_symbols":   len(self._registry),
            "total_windows_seen":   self._total_windows,
            "mean_priority":        sum(priorities) / len(priorities),
            "max_priority":         max(priorities),
            "min_priority":         min(priorities),
            "mean_count":           sum(counts) / len(counts),
            "max_count":            max(counts),
            "window_size_dist":     self._window_size_distribution(),
        }

    def _window_size_distribution(self) -> dict:
        dist = defaultdict(int)
        for e in self._registry.values():
            dist[e.window_size] += 1
        return dict(dist)

    def parameters(self) -> List[NanoTensor]:
        """Return symbol embeddings as trainable parameters."""
        return list(self._sym_embeddings)

    def zero_grad(self):
        for p in self._sym_embeddings:
            p.zero_grad()
