"""
Pattern Matcher - Structural Pattern Matching Engine

Uses Python 3.10+ pattern matching to replace nested conditionals
with elegant, declarative pattern-based dispatch.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Token:
    """Symbolic token representation for pattern matching"""
    value: str
    id: int
    type: str = "char"


class PatternMatcher:
    """
    Structural Pattern Matching engine for text processing.
    Replaces nested conditionals with symbolic pattern-based dispatch.
    """
    
    @staticmethod
    def match_token_pattern(token: Token) -> Dict[str, Any]:
        """
        Use SPM to classify tokens and extract features.
        This replaces if-elif-else chains with symbolic patterns.
        """
        match token:
            case Token(value=c, type="char") if c.isalpha():
                return {"category": "letter", "is_vowel": c.lower() in "aeiou"}
            case Token(value=c, type="char") if c.isdigit():
                return {"category": "digit", "value": int(c)}
            case Token(value=c, type="char") if c.isspace():
                return {"category": "whitespace"}
            case Token(value=c, type="char"):
                return {"category": "punctuation"}
            case Token(type="special", value=v):
                return {"category": "special", "raw": v}
            case _:
                return {"category": "unknown"}
    
    @staticmethod
    def match_generation_mode(mode: str) -> Dict[str, Any]:
        """
        Pattern match on generation strategy.
        Transforms control flow into symbolic dispatch.
        """
        match mode.split("_"):
            case ["greedy"]:
                return {"strategy": "max_probability", "temperature": 0.0}
            case ["sample", temp]:
                return {"strategy": "probabilistic", "temperature": float(temp)}
            case ["topk", k, temp]:
                return {"strategy": "topk", "k": int(k), "temperature": float(temp)}
            case _:
                return {"strategy": "greedy", "temperature": 0.0}
    
    @staticmethod
    def match_model_action(state: Dict[str, Any]) -> str:
        """
        Pattern match on model state to determine next action.
        """
        match state:
            case {"loss": loss, "epoch": epoch} if loss < 0.1:
                return "save_and_evaluate"
            case {"loss": loss, "epoch": epoch} if epoch > 1000:
                return "early_stop"
            case {"grad_norm": g} if g > 10.0:
                return "reduce_lr"
            case _:
                return "continue_training"
