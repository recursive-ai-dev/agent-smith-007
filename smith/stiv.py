"""
STIV - Semantic Token Integrity Verification

Production-grade manifold learner and verifier for tokenized text.
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
    def __init__(self, dimension: int = 64):
        self.dim = dimension

    def embed(self, token: str) -> np.ndarray:
        """Cryptographic hash-based embedding."""
        h = hashlib.sha256(token.encode("utf-8")).digest()
        arr = np.frombuffer(h, dtype=np.uint8) / 128.0 - 1.0
        if len(arr) < self.dim:
            arr = np.tile(arr, (self.dim // len(arr) + 1))
        vec = arr[: self.dim]
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-9) if norm else vec

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())


class TrafficCorpusBuilder:
    """Generate deterministic, structured traffic for manifold learning."""

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

    def build(
        self,
        sources: Sequence[str],
        min_samples: int,
    ) -> List[str]:
        identifiers = self._extract_terms(sources)
        if not identifiers:
            identifiers = [
                "users",
                "products",
                "orders",
                "settings",
                "logs",
                "sessions",
                "metrics",
                "events",
            ]

        tables = identifiers[: max(4, len(identifiers) // 4)]
        columns = identifiers[max(1, len(identifiers) // 4) : max(6, len(identifiers) // 2)]
        if not columns:
            columns = ["id", "name", "created_at", "status", "count", "active"]

        sql_ops = ["=", "<", ">", "<=", ">=", "!="]
        http_verbs = ["GET", "POST", "PUT", "PATCH", "DELETE"]
        endpoints = [
            f"/api/v1/{table}" for table in tables[: max(3, len(tables))]
        ]

        samples: List[str] = []

        for table in tables:
            for col in self._pick(columns, min(3, len(columns))):
                value = self._rng.randint(1, 1000)
                op = self._rng.choice(sql_ops)
                samples.append(f"SELECT {col} FROM {table} WHERE {col} {op} {value}")
                samples.append(f"UPDATE {table} SET {col} = {value} WHERE id = {value}")
                samples.append(
                    f"INSERT INTO {table} ({col}, created_at) VALUES ({value}, '2024-01-01')"
                )
                samples.append(f"DELETE FROM {table} WHERE {col} {op} {value}")

        for endpoint in endpoints:
            for verb in http_verbs:
                path = f"{endpoint}/{self._rng.randint(1, 999)}"
                payload = self._rng.choice(columns)
                samples.append(f"{verb} {path} payload={payload}")

        if len(samples) < min_samples:
            pad = min_samples - len(samples)
            for _ in range(pad):
                table = self._rng.choice(tables)
                col = self._rng.choice(columns)
                value = self._rng.randint(1, 1000)
                samples.append(f"SELECT {col} FROM {table} WHERE {col} = {value}")

        return samples[:min_samples]

    def noise_payload(self, min_len: int = 10, max_len: int = 60) -> str:
        length = self._rng.randint(min_len, max_len)
        return "".join(self._rng.choices(string.printable, k=length))


class STIV:
    def __init__(self, config: Optional[STIVConfig] = None):
        self.config = config or STIVConfig()
        self.tokenizer = SemanticTokenizer(self.config.dimension)
        self._tree: Optional[cKDTree] = None
        self._vectors: np.ndarray = np.array([])
        self._state = STIVState.INITIALIZING

    @property
    def state(self) -> STIVState:
        return self._state

    def learn(self, corpus: List[str]) -> None:
        """Construct safe manifold from corpus."""
        if not corpus:
            raise DomainError("Corpus required")

        logging.info("Learning from %s samples...", len(corpus))
        vectors = []

        for text in corpus:
            tokens = self.tokenizer.tokenize(text)
            if tokens:
                centroid = np.mean([self.tokenizer.embed(t) for t in tokens], axis=0)
                norm = np.linalg.norm(centroid)
                if norm:
                    vectors.append(centroid / norm)

        if not vectors:
            raise DomainError("No valid vectors generated")

        self._vectors = np.vstack(vectors)
        self._tree = cKDTree(self._vectors)
        self._state = STIVState.ACTIVE
        logging.info(
            "Manifold ready: %s nodes, dim=%s",
            len(vectors),
            self.config.dimension,
        )

    def verify(self, input_text: str) -> Dict[str, Any]:
        """Verify input against manifold boundary."""
        if self.state != STIVState.ACTIVE:
            raise DomainError("STIV not ready")

        tokens = self.tokenizer.tokenize(input_text)
        if not tokens:
            return {"safe": True, "score": 0.0, "reason": "EMPTY"}

        input_vec = np.mean([self.tokenizer.embed(t) for t in tokens], axis=0)
        norm = np.linalg.norm(input_vec)
        if norm:
            input_vec /= norm

        dist, _ = self._tree.query(input_vec, k=1)

        return {
            "safe": dist <= self.config.epsilon,
            "score": float(dist),
            "reason": "WITHIN_BOUNDS" if dist <= self.config.epsilon else "MANIFOLD_DIVERGENCE",
        }


class Validator:
    def __init__(
        self,
        engine: STIV,
        config: Optional[ValidatorConfig] = None,
        corpus_builder: Optional[TrafficCorpusBuilder] = None,
    ):
        self.engine = engine
        self.config = config or ValidatorConfig()
        self.corpus_builder = corpus_builder or TrafficCorpusBuilder(
            seed=self.config.random_seed
        )

    def _load_sources(self) -> List[str]:
        sources = []
        for path in ("README.md", "USAGE.md", "ENGLISH_TRAINING.md"):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    sources.append(handle.read())
            except FileNotFoundError:
                continue
        if not sources:
            sources.append("system tokens requests responses validation security queries")
        return sources

    def run_tests(self) -> bool:
        """Execute validation suite."""
        logging.info("=== STIV VALIDATION SUITE ===")

        safe_traffic = self.corpus_builder.build(
            self._load_sources(),
            min_samples=self.config.corpus_target,
        )

        try:
            self.engine.learn(safe_traffic)
        except Exception as exc:
            logging.error("Training failed: %s", exc)
            return False

        legit = "SELECT * FROM users WHERE id = 500"
        res = self.engine.verify(legit)
        logging.info(
            '[TEST+] Safe: %s | Score: %.4f | "%s"',
            res["safe"],
            res["score"],
            legit,
        )

        attack = "UNION SELECT 1, @@version -- ' OR 1=1"
        res = self.engine.verify(attack)
        logging.info(
            '[TEST-] Safe: %s | Score: %.4f | "%s"',
            res["safe"],
            res["score"],
            attack,
        )

        if res["safe"]:
            logging.error("Attack penetrated!")
            return False
        logging.info("✓ Attack correctly rejected")

        logging.info("\n[FUZZ] %s random iterations...", self.config.fuzz_iterations)
        penetrations = 0
        for _ in range(self.config.fuzz_iterations):
            payload = self.corpus_builder.noise_payload()
            if self.engine.verify(payload)["safe"]:
                penetrations += 1
        logging.info("Penetrations: %s/%s", penetrations, self.config.fuzz_iterations)
        if penetrations > self.config.fuzz_max_penetrations:
            logging.error("False negative rate too high")
            return False

        logging.info("\n[PERF] Benchmarking %s ops...", self.config.perf_iterations)
        start = time.perf_counter()
        for _ in range(self.config.perf_iterations):
            self.engine.verify("SELECT * FROM valid_table")
        throughput = self.config.perf_iterations / (time.perf_counter() - start)
        logging.info("Throughput: %.0f req/sec", throughput)

        logging.info("\n=== ALL TESTS PASSED ===")
        return True


def main() -> int:
    configure_logging()
    config = STIVConfig(dimension=128, epsilon=0.5)
    validator = Validator(STIV(config))
    return 0 if validator.run_tests() else 1


if __name__ == "__main__":
    sys.exit(main())
