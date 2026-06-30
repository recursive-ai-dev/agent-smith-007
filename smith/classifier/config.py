"""
AgentSmith Configuration
========================
Structured parameterisation for the model and its subsystems.
"""

from dataclasses import dataclass, field
from typing import List

# Standard domains for classification
DOMAINS = ["security", "administrative", "standard", "restricted", "emergency"]

@dataclass
class AgentSmithConfig:
    """
    Hyperparameters and structural settings for the AgentSmith model.
    """
    vocab_size: int = 50000
    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 8
    d_k: int = 16
    d_v: int = 16
    d_ff: int = 512
    max_seq_len: int = 128
    num_classes: int = 5
    domains: List[str] = field(default_factory=lambda: list(DOMAINS))
    layer_norm_eps: float = 1e-5

    # GSAR Settings: N-gram compression priorities
    gsar_window_sizes: List[int] = field(default_factory=lambda: [2, 3, 4])
    gsar_priority_threshold: float = 0.1
    gsar_temperature: float = 0.5
    gsar_min_freq: int = 5
    gsar_max_symbols: int = 500
    gsar_blend_alpha: float = 0.8

    # SEP Settings: Spurious correlation detection
    sep_chunk_size: int = 16
    sep_spurious_lambda: float = 0.1
