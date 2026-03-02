"""
Smith - Algebraic Symbolism Text Generation System

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

try:
    from .checkpoint import SafetensorCheckpoint
except ImportError:
    SafetensorCheckpoint = None

from .hsws import Concept, Subconcept, Betaconcept, SimpleSemanticEngine
from .stiv import (
    STIV,
    STIVConfig,
    STIVState,
    Validator,
    ValidatorConfig,
    DomainError,
    TrafficCorpusBuilder,
)

__version__ = "0.1.0"
__all__ = [
    "NanoTensor",
    "PatternMatcher",
    "Token",
    "SymbolicDB",
    "GatedRecurrentUnit",
    "Trainer",
    "Concept",
    "Subconcept",
    "Betaconcept",
    "SimpleSemanticEngine",
    "STIV",
    "STIVConfig",
    "STIVState",
    "Validator",
    "ValidatorConfig",
    "DomainError",
    "TrafficCorpusBuilder",
]

if SafetensorCheckpoint is not None:
    __all__.append("SafetensorCheckpoint")
