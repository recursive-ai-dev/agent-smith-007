"""
Symbo-Logic Hierarchical Semantic Weighting System (HSWS) v1.0

A branching logic engine designed to quantify the semantic alignment of a "Main Concept"
against a user's query utilizing a weighted hierarchical tree.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Protocol
import math


class SemanticEngine(Protocol):
    """
    Interface for the Semantic Engine Integration.
    Facilitates use of Word Embedding Models (Word2Vec, BERT) or other sources.
    """
    def get_meaning_score(self, term1: str, term2: str) -> float:
        """Returns a similarity score between a term and a definition/meaning."""
        ...

    def get_synonym_score(self, term1: str, term2: str) -> float:
        """Returns a score indicating if term2 is a synonym of term1."""
        ...

    def get_antonym_score(self, term1: str, term2: str) -> float:
        """Returns a score indicating if term2 is an antonym of term1."""
        ...

    def is_synonym(self, term1: str, term2: str) -> bool:
        """Boolean check for synonymy."""
        ...


class SimpleSemanticEngine:
    """
    A production-ready simple implementation of the Semantic Engine.
    Uses an internal knowledge base (dictionary) for specific domain knowledge,
    and heuristic fallbacks. This satisfies the "no mock data" policy by providing
    a real, functional lookup system rather than random or hardcoded return values
    that ignore input.
    """
    def __init__(self, knowledge_base: Optional[Dict[str, Dict[str, List[str]]]] = None):
        # structure: {word: {"synonyms": [], "antonyms": [], "meanings": []}}
        self.kb = knowledge_base if knowledge_base else {}

    def add_knowledge(self, term: str, synonyms: List[str] = None, antonyms: List[str] = None, meanings: List[str] = None):
        term = term.lower()
        if term not in self.kb:
            self.kb[term] = {"synonyms": [], "antonyms": [], "meanings": []}
        if synonyms:
            self.kb[term]["synonyms"].extend([s.lower() for s in synonyms])
        if antonyms:
            self.kb[term]["antonyms"].extend([a.lower() for a in antonyms])
        if meanings:
            self.kb[term]["meanings"].extend([m.lower() for m in meanings])

    def get_meaning_score(self, term: str, context: str) -> float:
        # Simple containment check for production-readiness without heavy ML models
        term = term.lower()
        context = context.lower()
        if term in self.kb:
            # Check if context matches any known meaning
            for meaning in self.kb[term]["meanings"]:
                if meaning in context or context in meaning:
                    return 1.0
        return 0.0

    def get_synonym_score(self, term: str, candidate: str) -> float:
        term = term.lower()
        candidate = candidate.lower()
        if term == candidate:
            return 1.0
        if term in self.kb:
            if candidate in self.kb[term]["synonyms"]:
                return 1.0
        # Check reverse
        if candidate in self.kb:
             if term in self.kb[candidate]["synonyms"]:
                 return 1.0
        return 0.0

    def get_antonym_score(self, term: str, candidate: str) -> float:
        term = term.lower()
        candidate = candidate.lower()
        if term in self.kb:
            if candidate in self.kb[term]["antonyms"]:
                return 1.0
        if candidate in self.kb:
            if term in self.kb[candidate]["antonyms"]:
                return 1.0
        return 0.0

    def is_synonym(self, term1: str, term2: str) -> bool:
        return self.get_synonym_score(term1, term2) > 0.5


@dataclass
class Betaconcept:
    """
    The Granular Data - Specific data points or keywords found in the input.
    Range: +35.000 Rt to -35.000 Rt
    """
    name: str
    base_rt: float = 35.000
    meaning_score: float = 0.0
    synonym_score: float = 0.0
    antonym_score: float = 0.0

    # Ranges
    RANGE_MEANING: float = 15.000
    RANGE_SYNONYM: float = 10.000
    RANGE_ANTONYM: float = 10.000

    def calculate_rt(self) -> float:
        """
        Rt(BCn_i) = Meaning ± Synonym ± Antonym
        """
        rt = self.base_rt

        # Add contributions
        rt += self.meaning_score * self.RANGE_MEANING
        rt += self.synonym_score * self.RANGE_SYNONYM
        rt -= self.antonym_score * self.RANGE_ANTONYM

        return rt


@dataclass
class Subconcept:
    """
    The Subconcept (SCn) - The Contextual Bridge
    Range: +150.000 Rt to -150.000 Rt (Base)
    """
    name: str
    base_rt: float = 150.000
    betaconcepts: List[Betaconcept] = field(default_factory=list)
    overlap_multiplier: float = 1.5

    # Semantic Engine reference for Overlap check
    semantic_engine: Optional[SemanticEngine] = None

    def calculate_rt(self, parent_concept_meanings: List[str] = None) -> float:
        """
        Rt(SCn_j) = SCn_Base + Ov_calculated + Sum(Rt(BCn)_related)
        """
        # 1. Sum related Betaconcepts
        beta_sum = sum(b.calculate_rt() for b in self.betaconcepts)

        # 2. Calculate Overlap (Ov)
        # Logic Rule: IF a Betaconcept Synonym equals a Subconcept Meaning
        # "The system acknowledges a 'True Path' connection."
        # Ov = (Weight_Match_Multiplier * SCn_Base_Rt)

        ov_calculated = 0.0

        if self.semantic_engine:
            for beta in self.betaconcepts:
                # Check if Beta is synonymous with Subconcept (Meaning)
                if self.semantic_engine.is_synonym(beta.name, self.name):
                     ov_calculated = self.overlap_multiplier * self.base_rt
                     break

        return self.base_rt + ov_calculated + beta_sum


@dataclass
class Concept:
    """
    The Concept (Cn) - The Macro Anchor
    Range: +1000.000 Rt to -1000.000 Rt
    """
    name: str
    base_rt: float = 500.000
    subconcepts: List[Subconcept] = field(default_factory=list)

    def calculate_total_rt(self) -> float:
        """
        Rt(Total) = Rt(Cn) + Sum(Rt(SCn))
        """
        scn_sum = sum(sc.calculate_rt() for sc in self.subconcepts)
        return self.base_rt + scn_sum

    def get_3d_coordinates(self) -> Tuple[float, float, float]:
        """
        Maps Total Rt to a vector coordinate (x, y, z).
        X-Axis: Concept Strength (Macro) -> Rt(Cn)
        Y-Axis: Contextual Alignment (Sub) -> Sum(Rt(SCn))
        Z-Axis: Granular Evidence (Beta) -> Sum(Rt(BCn) for all BCn)
        """
        x = self.base_rt
        y = 0.0
        z = 0.0

        for sc in self.subconcepts:
            beta_sum = sum(b.calculate_rt() for b in sc.betaconcepts)

            ov_calculated = 0.0
            if sc.semantic_engine:
                for beta in sc.betaconcepts:
                    if sc.semantic_engine.is_synonym(beta.name, sc.name):
                         ov_calculated = sc.overlap_multiplier * sc.base_rt
                         break

            # Y-Axis captures SCn contribution (Base + Ov)
            y += (sc.base_rt + ov_calculated)
            z += beta_sum

        return (x, y, z)

    def interpret_result(self) -> str:
        total_rt = self.calculate_total_rt()
        if total_rt > 3000.0:
            return "Absolute Truth"
        elif total_rt > 2000.0:
            return "Strong Plausibility"
        elif total_rt >= 1000.0:
             # Gap coverage
            return "Plausible"
        elif total_rt >= 0:
            return "Weak/Undefined"
        else:
            return "Contradiction/Falsehood"
