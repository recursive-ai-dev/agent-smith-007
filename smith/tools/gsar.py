"""
GSAR — General Symbolic Arrays Reasoning
========================================
Implements semantic n-gram compression.

GSAR scans for high-priority token sequences (patterns) and replaces them
with a single symbolic embedding. This effectively 'chunks' the input,
allowing the Transformer to process longer logical contexts with less compute.
"""

import math
import logging
from collections import Counter
from typing import List, Tuple, Dict, Optional, Any

from ..tensor import NanoTensor
from ..classifier.layers import TokenEmbedding

logger = logging.getLogger(__name__)


class GSAR:
    """
    Symbolic compression engine.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        window_sizes: List[int] = [2, 3, 4],
        priority_threshold: float = 0.1,
        temperature: float = 0.5,
        min_freq: int = 5,
        max_symbols: int = 500,
        blend_alpha: float = 0.8,
    ):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.window_sizes = window_sizes
        self.priority_threshold = priority_threshold
        self.temperature = temperature
        self.min_freq = min_freq
        self.max_symbols = max_symbols
        self.blend_alpha = blend_alpha

        # Symbolic registry: tuple(token_ids) -> NanoTensor
        self._registry: Dict[Tuple[int, ...], NanoTensor] = {}
        # Frequency tracking
        self._counts: Counter = Counter()

    def update_patterns(self, corpus: List[List[int]]):
        """Mine new patterns from a batch of sequences."""
        new_counts = Counter()
        for seq in corpus:
            for w in self.window_sizes:
                for i in range(len(seq) - w + 1):
                    pattern = tuple(seq[i : i + w])
                    # Ignore patterns containing PAD or UNK
                    if 0 in pattern or 1 in pattern:
                        continue
                    new_counts[pattern] += 1

        self._counts.update(new_counts)

        # Filter and promote patterns to symbols
        candidates = [
            (p, c) for p, c in self._counts.items()
            if c >= self.min_freq and p not in self._registry
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)

        for pattern, _ in candidates[:self.max_symbols - len(self._registry)]:
            # Initialise symbolic embedding as mean of constituents
            # This is a 'prior' that gradients will later refine
            self._registry[pattern] = NanoTensor(
                [0.0] * self.d_model, requires_grad=True
            )
            # Metadata for initialisation (will be used once in first compress)
            self._registry[pattern].metadata['needs_init'] = True

    def compress(
        self,
        token_ids: List[int],
        embedding_layer: TokenEmbedding
    ) -> Tuple[List[NanoTensor], List[bool], List[Optional[Tuple[int, ...]]]]:
        """
        Greedily replace patterns with symbols.
        Returns: (compressed_embeddings, is_symbol_mask, patterns_found)
        """
        out_embs: List[NanoTensor] = []
        is_sym: List[bool] = []
        patterns: List[Optional[Tuple[int, ...]]] = []

        idx = 0
        while idx < len(token_ids):
            match_found = False
            # Check longest windows first
            for w in sorted(self.window_sizes, reverse=True):
                if idx + w <= len(token_ids):
                    pattern = tuple(token_ids[idx : idx + w])
                    if pattern in self._registry:
                        sym_emb = self._registry[pattern]

                        # Lazy initialisation of symbol embedding
                        if sym_emb.metadata.get('needs_init'):
                            constituent_embs = [embedding_layer(tid) for tid in pattern]
                            mean_data = [
                                sum(e.data[d] for e in constituent_embs) / w
                                for d in range(self.d_model)
                            ]
                            for d in range(self.d_model):
                                sym_emb.data[d] = mean_data[d]
                            sym_emb.metadata['needs_init'] = False

                        out_embs.append(sym_emb)
                        is_sym.append(True)
                        patterns.append(pattern)
                        idx += w
                        match_found = True
                        break

            if not match_found:
                # No symbol found, use raw token embedding
                out_embs.append(embedding_layer(token_ids[idx]))
                is_sym.append(False)
                patterns.append(None)
                idx += 1

        return out_embs, is_sym, patterns

    def parameters(self) -> List[NanoTensor]:
        """Return all active symbolic embeddings."""
        return list(self._registry.values())
