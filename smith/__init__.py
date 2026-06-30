"""
Smith - Algebraic Symbolism Text Generation System
==================================================
A lightweight AI model implementation demonstrating:
- Branchless operations using algebraic primitives
- Structural pattern matching for control flow
- Self-hosted database for model persistence
- Custom autograd implementation (NanoTensor)
- Safetensor checkpointing for model serialization
- Symbo-Logic Hierarchical Semantic Weighting System (HSWS)
"""

from .tensor import NanoTensor
from .pattern_matcher import PatternMatcher, Token
from .database import SymbolicDB
from .gru_model import GatedRecurrentUnit
from .trainer import Trainer

# Alias for backward compatibility with earlier versions
SymbolicRNN = GatedRecurrentUnit

try:
    from .checkpoint import SafetensorCheckpoint
except ImportError:
    SafetensorCheckpoint = None

from .hsws import Concept, Subconcept, Betaconcept, SimpleSemanticEngine

try:
    from .stiv import (
        STIV,
        STIVConfig,
        STIVState,
        Validator,
        ValidatorConfig,
        DomainError,
        TrafficCorpusBuilder,
    )
    _stiv_available = True
except ImportError:
    _stiv_available = False
    STIV = STIVConfig = STIVState = Validator = ValidatorConfig = None
    DomainError = TrafficCorpusBuilder = None

# ── AgentSmith classifier
from .classifier import AgentSmith, AgentSmithConfig, DOMAINS

__version__ = "0.3.0"
__all__ = [
    "AgentSmith",
    "AgentSmithConfig",
    "Betaconcept",
    "Concept",
    "DOMAINS",
    "GatedRecurrentUnit",
    "SymbolicRNN",
    "NanoTensor",
    "PatternMatcher",
    "SimpleSemanticEngine",
    "Subconcept",
    "SymbolicDB",
    "Token",
    "Trainer",
    "reset_global_state",
]

if SafetensorCheckpoint is not None:
    __all__.append("SafetensorCheckpoint")

if _stiv_available:
    __all__.extend([
        "DomainError",
        "STIV",
        "STIVConfig",
        "STIVState",
        "TrafficCorpusBuilder",
        "Validator",
        "ValidatorConfig",
    ])

def reset_global_state():
    """Reset global state for deterministic execution runs."""
    NanoTensor._global_index = 0
