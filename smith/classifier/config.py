"""
AgentSmith Configuration
========================
Single source-of-truth for every hyperparameter in the classifier stack.
All numeric defaults are chosen so the model trains on CPU in reasonable
time; scale d_model / num_heads / num_layers for larger deployments.
"""

from dataclasses import dataclass, field
from typing import List

# ── Domain taxonomy ──────────────────────────────────────────────────
DOMAINS: List[str] = [
    "science_technology",   # 0
    "mathematics",          # 1
    "medicine_health",      # 2
    "law_legal",            # 3
    "finance_economics",    # 4
    "literature_arts",      # 5
    "history_politics",     # 6
    "philosophy_ethics",    # 7
    "engineering",          # 8
    "natural_sciences",     # 9
    "computer_science",     # 10
    "social_sciences",      # 11
]


@dataclass
class AgentSmithConfig:
    # ── Vocabulary ───────────────────────────────────────────────────
    vocab_size: int = 4096          # hash-mapped word tokens + specials
    pad_id: int = 0
    unk_id: int = 1

    # ── Transformer architecture ─────────────────────────────────────
    d_model: int = 128              # token embedding / hidden dimension
    num_heads: int = 4              # parallel attention heads
    d_k: int = 32                   # key / query dimension per head
    d_v: int = 32                   # value dimension per head
    num_layers: int = 3             # stacked TransformerBlocks
    d_ff: int = 256                 # inner FFN dimension  (≈ 2× d_model)
    max_seq_len: int = 128          # hard cap on tokenised input length
    num_classes: int = 12           # one per domain above
    layer_norm_eps: float = 1e-5

    # ── Domain labels ────────────────────────────────────────────────
    domains: List[str] = field(default_factory=lambda: list(DOMAINS))

    # ── Adam optimiser ───────────────────────────────────────────────
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    adam_eps: float = 1e-8
    weight_decay: float = 0.01      # decoupled L2 (AdamW style)

    # ── Training loop ────────────────────────────────────────────────
    num_epochs: int = 10
    grad_clip: float = 1.0          # global gradient norm clipping
    warmup_steps: int = 100         # linear LR warm-up

    # ── Mixed-precision simulation ───────────────────────────────────
    # Pure-Python floats are 64-bit; we simulate FP32/FP16 by rounding
    # intermediate activations to the corresponding decimal precision.
    use_amp: bool = True
    amp_bits: int = 32              # 16 → simulate FP16, 32 → simulate FP32
    loss_scale: float = 128.0       # loss scaling factor for FP16 stability

    # ── GSAR (General Symbolic Arrays Reasoning) ─────────────────────
    gsar_window_sizes: List[int] = field(default_factory=lambda: [2, 3, 4])
    gsar_priority_threshold: float = 0.60   # min priority to compress
    gsar_temperature: float = 8.0           # sharpness of priority sigmoid
    gsar_min_freq: int = 3                  # occurrences before registration
    gsar_max_symbols: int = 512             # symbol-vocabulary capacity
    gsar_blend_alpha: float = 0.85          # symbol vs. mean-embed blend

    # ── SEP (Self-Explanatory Perception) ────────────────────────────
    sep_chunk_size: int = 32        # tokens per processing chunk
    sep_spurious_lambda: float = 0.15   # penalty weight for correlated chunks

    # ── Diagnostics ──────────────────────────────────────────────────
    jacobian_freq: int = 50         # steps between Jacobian snapshots
    hessian_freq: int = 200         # steps between Hessian snapshots
    grad_stats_freq: int = 10       # steps between gradient-norm logging

    # ── Checkpointing ────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    checkpoint_freq: int = 200

    def __post_init__(self):
        if self.num_heads * self.d_k > self.d_model:
            raise ValueError(
                "num_heads x d_k must be <= d_model "
                f"({self.num_heads}x{self.d_k}={self.num_heads*self.d_k} > {self.d_model})"
            )
        if len(self.domains) != self.num_classes:
            raise ValueError(
                f"len(domains)={len(self.domains)} != num_classes={self.num_classes}"
            )
        if not isinstance(self.sep_chunk_size, int) or self.sep_chunk_size <= 0:
            raise ValueError(
                f"sep_chunk_size must be an int > 0, got {self.sep_chunk_size!r}"
            )
        if not self.gsar_window_sizes:
            raise ValueError("gsar_window_sizes must be a non-empty iterable of ints > 0")
        for w in self.gsar_window_sizes:
            if not isinstance(w, int) or w <= 0:
                raise ValueError(
                    f"gsar_window_sizes must contain ints > 0, got {w!r}"
                )
