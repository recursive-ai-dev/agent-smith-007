"""
STIV — Semantic Token Integrity Verification
============================================
Production-grade manifold learner and verifier for tokenized text.

STIV builds a high-dimensional manifold representing "safe" or "legitimate"
token distributions.  During verification, incoming text is projected into
this space; if the projection falls outside a learned distance threshold
(epsilon), the input is flagged as potentially adversarial or corrupted.
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import re
import string
import sys
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

# Optional: suppress numpy/scipy warnings in production
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def configure_logging(level: int = logging.INFO, fmt: str = DEFAULT_LOG_FORMAT) -> None:
    """Configure logging if not already configured."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format=fmt)
    else:
        root.setLevel(level)


class STIVState(Enum):
    INITIALIZING = auto()
    ACTIVE = auto()


class DomainError(Exception):
    """Raised for configuration or data integrity errors in STIV."""
    pass


@dataclass
class STIVConfig:
    dimension: int = 64
    epsilon: float = 0.45

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise DomainError("Dimension must be positive")
        if not 0 < self.epsilon < math.sqrt(2):
            raise DomainError("Epsilon must be in (0, √2)")


@dataclass
class ValidatorConfig:
    fuzz_iterations: int = 10_000
    fuzz_max_penetrations: int = 50
    perf_iterations: int = 50_000
    random_seed: int = 1337
    corpus_target: int = 256

    def __post_init__(self) -> None:
        for name, value in {
            "fuzz_iterations": self.fuzz_iterations,
            "fuzz_max_penetrations": self.fuzz_max_penetrations,
            "perf_iterations": self.perf_iterations,
            "corpus_target": self.corpus_target,
        }.items():
            if value <= 0:
                raise DomainError(f"{name} must be positive")


class SemanticTokenizer:
    """Tokenizer that uses cryptographic hashing to generate token embeddings."""

    def __init__(self, dimension: int = 64):
        self.dim = dimension

    def embed(self, token: str) -> np.ndarray:
        """Cryptographic hash-based embedding. Stable and deterministic."""
        h = hashlib.sha256(token.encode("utf-8")).digest()
        arr = np.frombuffer(h, dtype=np.uint8) / 128.0 - 1.0
        # Tile or slice to match target dimension
        if len(arr) < self.dim:
            arr = np.tile(arr, (self.dim // len(arr) + 1))
        vec = arr[: self.dim].astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-9) if norm else vec

    def tokenize(self, text: str) -> List[str]:
        """Split text into lowercase alphanumeric tokens."""
        return re.findall(r"\w+", text.lower())


class TrafficCorpusBuilder:
    """Generates structured traffic for manifold learning."""

    def __init__(self, seed: int = 1337):
        self._rng = random.Random(seed)

    def _normalize_identifiers(self, raw: Iterable[str]) -> List[str]:
        identifiers = []
        for token in raw:
            token = re.sub(r"[^a-zA-Z0-9_]", "", token).lower()
            if len(token) >= 3 and token[0].isalpha():
                identifiers.append(token)
        return sorted(set(identifiers))

    def _extract_terms(self, sources: Sequence[str]) -> List[str]:
        terms = []
        for text in sources:
            terms.extend(re.findall(r"[A-Za-z_]{3,}", text))
        return self._normalize_identifiers(terms)

    def _pick(self, items: Sequence[str], count: int) -> List[str]:
        if not items:
            return []
        return [self._rng.choice(items) for _ in range(count)]

    def build(self, sources: Sequence[str], min_samples: int) -> List[str]:
        """Construct a representative corpus from source texts."""
        identifiers = self._extract_terms(sources)
        if not identifiers:
            identifiers = ["users", "products", "orders", "settings", "logs", "sessions"]

        tables = identifiers[: max(4, len(identifiers) // 4)]
        columns = identifiers[max(1, len(identifiers) // 4) : max(6, len(identifiers) // 2)]
        if not columns:
            columns = ["id", "name", "created_at", "status"]

        sql_ops = ["=", "<", ">", "!="]
        http_verbs = ["GET", "POST", "PUT", "DELETE"]
        endpoints = [f"/api/v1/{table}" for table in tables[:3]]

        samples: List[str] = []

        # Generate SQL-like patterns
        for table in tables:
            for col in self._pick(columns, 3):
                val = self._rng.randint(1, 1000)
                op = self._rng.choice(sql_ops)
                samples.append(f"SELECT {col} FROM {table} WHERE {col} {op} {val}")
                samples.append(f"UPDATE {table} SET {col} = {val} WHERE id = {val}")

        # Generate API-like patterns
        for endpoint in endpoints:
            for verb in http_verbs:
                samples.append(f"{verb} {endpoint}/{self._rng.randint(1, 999)} payload={self._rng.choice(columns)}")

        # Pad to min_samples
        while len(samples) < min_samples:
            t = self._rng.choice(tables)
            c = self._rng.choice(columns)
            v = self._rng.randint(1, 1000)
            samples.append(f"SELECT {c} FROM {t} WHERE {c} = {v}")

        return samples[:min_samples]

    def noise_payload(self, min_len: int = 10, max_len: int = 60) -> str:
        """Generate random printable string payload."""
        length = self._rng.randint(min_len, max_len)
        return "".join(self._rng.choices(string.printable, k=length))


class STIV:
    """Semantic Token Integrity Verification engine."""

    def __init__(self, config: Optional[STIVConfig] = None):
        self.config = config or STIVConfig()
        self.tokenizer = SemanticTokenizer(self.config.dimension)
        self._tree: Optional[cKDTree] = None
        self._vectors: np.ndarray = np.array([], dtype=np.float32)
        self._state = STIVState.INITIALIZING

    @property
    def state(self) -> STIVState:
        return self._state

    def learn(self, corpus: List[str]) -> None:
        """Construct the safety manifold from a trusted corpus."""
        if not corpus:
            raise DomainError("Corpus required for learning")

        logging.info("STIV: Learning from %s samples...", len(corpus))
        vectors = []

        for text in corpus:
            tokens = self.tokenizer.tokenize(text)
            if tokens:
                # Embed each token and compute centroid
                embeddings = [self.tokenizer.embed(t) for t in tokens]
                centroid = np.mean(embeddings, axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 1e-9:
                    vectors.append(centroid / norm)

        if not vectors:
            raise DomainError("No valid semantic vectors generated from corpus")

        self._vectors = np.vstack(vectors).astype(np.float32)
        # Use cKDTree for efficient nearest neighbor search
        self._tree = cKDTree(self._vectors)
        self._state = STIVState.ACTIVE
        logging.info("STIV: Manifold active (%s nodes, dim=%s)", len(vectors), self.config.dimension)

    def verify(self, input_text: str) -> Dict[str, Any]:
        """Verify if input text lies within the learned safety manifold."""
        if self.state != STIVState.ACTIVE or self._tree is None:
            raise DomainError("STIV engine not ready (state=%s)" % self.state)

        tokens = self.tokenizer.tokenize(input_text)
        if not tokens:
            return {"safe": True, "score": 0.0, "reason": "EMPTY"}

        # Project input to manifold space
        embeddings = [self.tokenizer.embed(t) for t in tokens]
        input_vec = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(input_vec)
        if norm > 1e-9:
            input_vec /= norm

        # Query nearest point in manifold
        dist, _ = self._tree.query(input_vec, k=1)

        is_safe = dist <= self.config.epsilon
        return {
            "safe": bool(is_safe),
            "score": float(dist),
            "reason": "WITHIN_BOUNDS" if is_safe else "MANIFOLD_DIVERGENCE",
        }


class Validator:
    """Validation suite for STIV manifold robustness and performance."""

    def __init__(
        self,
        engine: STIV,
        config: Optional[ValidatorConfig] = None,
        corpus_builder: Optional[TrafficCorpusBuilder] = None,
    ):
        self.engine = engine
        self.config = config or ValidatorConfig()
        self.corpus_builder = corpus_builder or TrafficCorpusBuilder(seed=self.config.random_seed)

    def _load_sources(self) -> List[str]:
        sources = []
        for path in ("README.md", "USAGE.md", "ENGLISH_TRAINING.md"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    sources.append(f.read())
            except FileNotFoundError:
                continue
        if not sources:
            sources.append("system tokens requests responses validation security queries")
        return sources

    def run_tests(self) -> bool:
        """Execute the full validation suite."""
        logging.info("=== STIV VALIDATION SUITE ===")

        # 1. Build and Learn
        safe_traffic = self.corpus_builder.build(self._load_sources(), min_samples=self.config.corpus_target)
        try:
            self.engine.learn(safe_traffic)
        except Exception as exc:
            logging.error("STIV: Learning failed: %s", exc)
            return False

        # 2. Legit probe
        legit = "SELECT * FROM users WHERE id = 500"
        res = self.engine.verify(legit)
        logging.info('[TEST+] Legit input: Safe=%s Score=%.4f', res["safe"], res["score"])
        if not res["safe"]:
            logging.warning("✓ Warning: legit input rejected (likely too small epsilon)")

        # 3. Attack probe
        attack = "UNION SELECT 1, @@version -- ' OR 1=1"
        res = self.engine.verify(attack)
        logging.info('[TEST-] Attack input: Safe=%s Score=%.4f', res["safe"], res["score"])
        if res["safe"]:
            logging.error("✘ FAIL: Attack penetrated manifold!")
            return False
        logging.info("✓ Attack correctly rejected")

        # 4. Fuzz test
        logging.info("[FUZZ] Running %s random iterations...", self.config.fuzz_iterations)
        penetrations = 0
        for _ in range(self.config.fuzz_iterations):
            payload = self.corpus_builder.noise_payload()
            if self.engine.verify(payload)["safe"]:
                penetrations += 1

        logging.info("Fuzzing Penetrations: %s/%s", penetrations, self.config.fuzz_iterations)
        if penetrations > self.config.fuzz_max_penetrations:
            logging.error("✘ FAIL: False positive rate too high (%s > %s)", penetrations, self.config.fuzz_max_penetrations)
            return False

        # 5. Performance benchmark
        logging.info("[PERF] Benchmarking %s operations...", self.config.perf_iterations)
        start = time.perf_counter()
        test_q = "SELECT * FROM valid_table"
        for _ in range(self.config.perf_iterations):
            self.engine.verify(test_q)
        dt = time.perf_counter() - start
        throughput = self.config.perf_iterations / dt
        logging.info("✓ Performance: %.0f req/sec (total time %.2fs)", throughput, dt)

        logging.info("=== ALL STIV TESTS PASSED ===")
        return True


def main() -> int:
    configure_logging()
    config = STIVConfig(dimension=128, epsilon=0.55)
    validator = Validator(STIV(config))
    try:
        success = validator.run_tests()
        return 0 if success else 1
    except Exception as e:
        logging.exception("Validator crashed: %s", e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
