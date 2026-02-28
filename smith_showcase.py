"""
Smith Showcase - Production-Ready Algebraic Symbolism Text Generation System

A single-script, production-grade implementation of the Smith concept.
Features:
- Full Autograd (NanoTensor) with no broken gradients.
- Branchless operations using algebraic primitives.
- Structural Pattern Matching.
- Self-hosted Symbolic Database.
- Symbolic RNN with working backpropagation.
"""

import argparse
import hashlib
import json
import logging
import math
import random
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# ============================================================================
# NANO TENSOR WITH FULL AUTOGRAD
# ============================================================================

class NanoTensor:
    """
    Lightweight tensor with automatic differentiation.
    Uses branchless algebraic primitives for all conditional logic.
    Supports full backward pass through algebraic compositions.
    """

    def __init__(self, data, _parents=(), _op=None, requires_grad=True):
        if isinstance(data, (int, float)):
            data = [float(data)]
        self.data = [float(x) for x in data] if isinstance(data, (list, tuple)) else data
        self.grad = [0.0] * len(self.data)
        self._backward = lambda: None
        self._parents = _parents
        self._op = _op
        self.requires_grad = requires_grad
        self.shape = (len(self.data),)

    def _broadcast_to(self, target_len: int) -> 'NanoTensor':
        if len(self.data) == 1 and target_len > 1:
            # Broadcast scalar
            new_data = self.data * target_len
            # We don't track this as an op for simplicity, but we handle gradient accumulation
            out = NanoTensor(new_data, _parents=(self,), _op='broadcast')
            def _backward():
                if self.requires_grad:
                    self.grad[0] += sum(out.grad)
            out._backward = _backward
            return out
        return self

    # --- Basic Operations ---

    def __add__(self, other):
        other = other if isinstance(other, NanoTensor) else NanoTensor(other)
        # Handle simple broadcasting if one is scalar (length 1)
        if len(self.data) != len(other.data):
             if len(self.data) == 1: self = self._broadcast_to(len(other.data))
             elif len(other.data) == 1: other = other._broadcast_to(len(self.data))

        out = NanoTensor([a + b for a, b in zip(self.data, other.data)],
                         _parents=(self, other), _op='+')

        def _backward():
            if self.requires_grad:
                for i in range(len(self.grad)):
                    self.grad[i] += out.grad[i]
            if other.requires_grad:
                for i in range(len(other.grad)):
                    other.grad[i] += out.grad[i]
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, NanoTensor) else NanoTensor(other)
        if len(self.data) != len(other.data):
             if len(self.data) == 1: self = self._broadcast_to(len(other.data))
             elif len(other.data) == 1: other = other._broadcast_to(len(self.data))

        out = NanoTensor([a * b for a, b in zip(self.data, other.data)],
                         _parents=(self, other), _op='*')

        def _backward():
            if self.requires_grad:
                for i in range(len(self.grad)):
                    self.grad[i] += other.data[i] * out.grad[i]
            if other.requires_grad:
                for i in range(len(other.grad)):
                    other.grad[i] += self.data[i] * out.grad[i]
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __pow__(self, power):
        # Assuming power is a float/int constant for simplicity, or we implement full pow
        # For this showcase, we support constant power
        assert isinstance(power, (int, float)), "Power must be scalar constant for now"
        out = NanoTensor([x ** power for x in self.data], _parents=(self,), _op=f'**{power}')

        def _backward():
            if self.requires_grad:
                for i in range(len(self.grad)):
                    self.grad[i] += (power * (self.data[i] ** (power - 1))) * out.grad[i]
        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, NanoTensor) else NanoTensor(other)
        return self * (other ** -1.0)

    def exp(self):
        out = NanoTensor([math.exp(x) for x in self.data], _parents=(self,), _op='exp')

        def _backward():
            if self.requires_grad:
                for i in range(len(self.grad)):
                    self.grad[i] += out.data[i] * out.grad[i] # d(e^x)/dx = e^x
        out._backward = _backward
        return out

    def log(self):
        out = NanoTensor([math.log(x) for x in self.data], _parents=(self,), _op='log')

        def _backward():
            if self.requires_grad:
                for i in range(len(self.grad)):
                    self.grad[i] += (1.0 / self.data[i]) * out.grad[i]
        out._backward = _backward
        return out

    def sum(self):
        out = NanoTensor([sum(self.data)], _parents=(self,), _op='sum')

        def _backward():
            if self.requires_grad:
                for i in range(len(self.grad)):
                    self.grad[i] += out.grad[0]
        out._backward = _backward
        return out

    def abs(self):
        out = NanoTensor([abs(x) for x in self.data], _parents=(self,), _op='abs')

        def _backward():
            if self.requires_grad:
                for i in range(len(self.grad)):
                    # d(|x|)/dx = sign(x)
                    sign = 1.0 if self.data[i] > 0 else (-1.0 if self.data[i] < 0 else 0.0)
                    self.grad[i] += sign * out.grad[i]
        out._backward = _backward
        return out

    def max(self):
        # Note: True max for autograd selects the gradient of the max element
        m = max(self.data)
        out = NanoTensor([m], _parents=(self,), _op='max')

        # We need to know WHICH element was max to route gradient
        # If multiple are equal to max, we can distribute or pick one.
        # For simplicity, we pick the first one (or all).
        # Better to distribute gradient among all max elements? Standard is usually just one or split.
        # Let's route to all that match max

        def _backward():
            if self.requires_grad:
                # Count how many hit max
                count = 0
                for x in self.data:
                    if x == m: count += 1

                for i in range(len(self.grad)):
                    if self.data[i] == m:
                        self.grad[i] += out.grad[0] / count
        out._backward = _backward
        return out

    def matmul(self, other):
        other = other if isinstance(other, NanoTensor) else NanoTensor(other)
        m = len(self.data)
        n = len(other.data)

        # Case 1: Dot product
        if m == n:
            # We can reuse element-wise mul + sum
            return (self * other).sum()

        # Case 2: Matrix-Vector (m flattened matrix, n vector)
        if n > 0 and m % n == 0:
            rows = m // n
            y = []
            for i in range(rows):
                base = i * n
                # This logic replicates the row slice dot product
                # We can construct the result graph node by node, but that creates many nodes.
                # Optimized 'matmul' node is better.
                acc = 0.0
                for j in range(n):
                    acc += self.data[base + j] * other.data[j]
                y.append(acc)

            out = NanoTensor(y, _parents=(self, other), _op='matmul_mv')

            def _backward():
                if self.requires_grad:
                    for i in range(rows):
                        base = i * n
                        gi = out.grad[i]
                        for j in range(n):
                            self.grad[base + j] += other.data[j] * gi
                if other.requires_grad:
                    for j in range(n):
                        acc = 0.0
                        for i in range(rows):
                            acc += self.data[i * n + j] * out.grad[i]
                        other.grad[j] += acc
            out._backward = _backward
            return out

        raise ValueError(f"Incompatible dimensions for matmul: {m} and {n}")

    def backward(self):
        topo = []
        visited = set()
        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for parent in t._parents:
                    build_topo(parent)
                topo.append(t)
        build_topo(self)

        self.grad[0] = 1.0
        for t in reversed(topo):
            t._backward()

    def zero_grad(self):
        self.grad = [0.0] * len(self.grad)

    def __repr__(self):
        data_str = str(self.data[:5]) + ("..." if len(self.data) > 5 else "")
        grad_str = str([f"{g:.3f}" for g in self.grad[:5]]) + ("..." if len(self.grad) > 5 else "")
        return f"NanoTensor(data={data_str}, grad={grad_str})"

    # --- Static Algebraic Primitives (Branchless) ---
    # These return raw floats, not tensors, used for things like selecting embeddings
    # or control flow emulation. If differentiation is needed through them, we use Tensor ops.

    @staticmethod
    def _sign(x: float) -> float:
        return float((x > 0) - (x < 0))

    @staticmethod
    def _max(a: float, b: float) -> float:
        return (a + b + abs(a - b)) / 2.0

    @staticmethod
    def _if_else(condition: float, true_val: float, false_val: float) -> float:
        # condition > 0 -> sign(condition) = 1
        mask = (NanoTensor._sign(condition) + 1.0) / 2.0
        return mask * true_val + (1.0 - mask) * false_val

# ============================================================================
# PATTERN MATCHER
# ============================================================================

@dataclass
class Token:
    value: str
    id: int
    type: str = "char"

class PatternMatcher:
    @staticmethod
    def match_generation_mode(mode: str) -> Dict[str, Any]:
        match mode.split("_"):
            case ["greedy"]:
                return {"strategy": "max_probability", "temperature": 0.0}
            case ["sample", temp]:
                return {"strategy": "probabilistic", "temperature": float(temp)}
            case ["topk", k, temp]:
                return {"strategy": "topk", "k": int(k), "temperature": float(temp)}
            case _:
                return {"strategy": "greedy", "temperature": 0.0}

# ============================================================================
# DATABASE
# ============================================================================

class SymbolicDB:
    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        cursor = self.conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS model_params (id INTEGER PRIMARY KEY, param_key TEXT, data BLOB)")
        cursor.execute("CREATE TABLE IF NOT EXISTS training_log (epoch INTEGER, loss REAL, grad_norm REAL)")
        self.conn.commit()

    def save_params(self, params: Dict[str, NanoTensor]) -> str:
        param_key = f"params_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
        serialized = json.dumps({k: v.data for k, v in params.items()}).encode()
        self.conn.execute("INSERT INTO model_params (param_key, data) VALUES (?, ?)", (param_key, serialized))
        self.conn.commit()
        return param_key

    def log_training(self, epoch: int, loss: float, grad_norm: float):
        self.conn.execute("INSERT INTO training_log (epoch, loss, grad_norm) VALUES (?, ?, ?)", (epoch, loss, grad_norm))
        self.conn.commit()

    def close(self):
        self.conn.close()

# ============================================================================
# SYMBOLIC RNN
# ============================================================================

class SymbolicRNN:
    def __init__(self, vocab_size: int, hidden_size: int, db: SymbolicDB):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.db = db

        scale = 0.1
        self.params = {
            'Wxh': NanoTensor([random.uniform(-scale, scale) for _ in range(vocab_size * hidden_size)]),
            'Whh': NanoTensor([random.uniform(-scale, scale) for _ in range(hidden_size * hidden_size)]),
            'Why': NanoTensor([random.uniform(-scale, scale) for _ in range(hidden_size * vocab_size)]),
            'bh': NanoTensor([0.0] * hidden_size),
            'by': NanoTensor([0.0] * vocab_size)
        }
        self.pattern_matcher = PatternMatcher()

    def _embed(self, token_id: int) -> NanoTensor:
        # One-hot embedding
        embedding = [0.0] * self.vocab_size
        embedding[token_id] = 1.0
        return NanoTensor(embedding, requires_grad=False)

    def _tanh_branchless(self, x: NanoTensor) -> NanoTensor:
        """
        Branchless tanh approximation using algebraic primitives.
        We use Softsign: x / (1 + |x|) which is bounded [-1, 1] and algebraic.
        The original Pade approximation x(27+x^2)/(27+9x^2) is NOT bounded for large x
        (it grows as x/9), leading to exploding gradients in RNNs.
        """
        # Softsign: x / (1 + |x|)
        return x / (x.abs() + 1.0)

    def forward(self, inputs: List[int], hprev: Optional[NanoTensor] = None) -> Tuple[NanoTensor, NanoTensor]:
        if hprev is None:
            h = NanoTensor([0.0] * self.hidden_size, requires_grad=True) # h0 usually no grad w.r.t prev, but for trunc BPTT it might matter?
            # Actually h0 is usually constant 0.
            h.requires_grad = True # If we want to learn initial state? But let's say no for now.
        else:
            h = hprev

        for token_id in inputs:
            x = self._embed(token_id)

            # h = tanh(Wxh @ x + Whh @ h + bh)
            Wxh_h = self.params['Wxh'].matmul(x)
            Whh_h = self.params['Whh'].matmul(h)
            h_raw = Wxh_h + Whh_h + self.params['bh']

            h = self._tanh_branchless(h_raw)

        logits = self.params['Why'].matmul(h) + self.params['by']
        return logits, h

    def generate(self, seed: str, length: int, temperature: float = 0.5) -> str:
        config = self.pattern_matcher.match_generation_mode(f"sample_{temperature}")
        token_ids = [ord(c) % self.vocab_size for c in seed]
        h = None
        generated = list(seed)

        # Warmup
        if token_ids:
            _, h = self.forward(token_ids[:-1])
            last_logits, h = self.forward([token_ids[-1]], h)
        else:
             # Should not happen with non-empty seed
             return ""

        curr_token_id = token_ids[-1]

        for _ in range(length):
            logits, h = self.forward([curr_token_id], h)

            # Sampling logic (simple version for showcase)
            temp = config["temperature"]
            if temp == 0:
                # Argmax
                next_id = 0
                max_val = -float('inf')
                for i, val in enumerate(logits.data):
                    if val > max_val:
                        max_val = val
                        next_id = i
            else:
                # Sample
                probs = [math.exp(v / temp) for v in logits.data]
                total = sum(probs)
                r = random.uniform(0, total)
                upto = 0
                next_id = len(probs) - 1
                for i, p in enumerate(probs):
                    upto += p
                    if upto > r:
                        next_id = i
                        break

            curr_token_id = next_id
            generated.append(chr(next_id % 128))

        return "".join(generated)

    def zero_grad(self):
        for p in self.params.values():
            p.zero_grad()

# ============================================================================
# TRAINER
# ============================================================================

class Trainer:
    def __init__(self, model: SymbolicRNN, learning_rate: float = 0.01):
        self.model = model
        self.learning_rate = learning_rate

    def compute_loss(self, logits: NanoTensor, target: int) -> NanoTensor:
        """
        Symbolic Cross Entropy Loss = -log(softmax(logits)[target])
        = -log(exp(logits[target]) / sum(exp(logits)))
        = -logits[target] + log(sum(exp(logits)))
        """
        # We need to extract the target logit symbolically
        # We can multiply logits by a one-hot vector to get the scalar target logit
        target_mask = [0.0] * len(logits.data)
        target_mask[target] = 1.0
        target_mask_t = NanoTensor(target_mask, requires_grad=False)

        # Extract target logit: sum(logits * mask)
        target_logit = (logits * target_mask_t).sum()

        # LogSumExp with numerical stability (subtract max)
        # We use raw max of data (non-differentiable selection for stability constant)
        max_logit = max(logits.data)

        # Shifted logits for stability
        # We assume max_logit is constant for backprop purposes (detached)
        # But wait, max(logits) IS dependent on logits.
        # In standard frameworks, this shift doesn't affect gradient of LogSumExp.
        # log(sum(exp(x - c))) + c = log(sum(exp(x) * exp(-c))) + c
        # = log(exp(-c) * sum(exp(x))) + c = -c + log(sum(exp(x))) + c = log(sum(exp(x)))
        # So we can just shift logits and add the constant back?
        # Actually, if we just shift for calculation, we can do:
        # shifted = logits - max_logit (where max_logit is constant scalar)
        shifted_logits = logits + (-max_logit)
        exp_logits = shifted_logits.exp()
        sum_exp = exp_logits.sum()
        log_sum_exp = sum_exp.log() + max_logit

        # Loss = log_sum_exp - target_logit
        loss = log_sum_exp - target_logit
        return loss

    def train_step(self, inputs: List[int], targets: List[int]):
        self.model.zero_grad()
        h = None
        losses = []

        for i in range(len(inputs)):
            logits, h = self.model.forward([inputs[i]], h)
            loss = self.compute_loss(logits, targets[i])
            losses.append(loss)

        # Accumulate gradients once for the whole sequence (BPTT)
        # This avoids double-counting gradients on shared history nodes
        if losses:
            total_loss_tensor = losses[0]
            for l in losses[1:]:
                total_loss_tensor = total_loss_tensor + l

            total_loss_tensor.backward()
            avg_loss = total_loss_tensor.data[0] / len(inputs)
        else:
            avg_loss = 0.0

        # Update
        total_norm_sq = 0.0
        for param in self.model.params.values():
            p_norm = sum(g*g for g in param.grad)
            total_norm_sq += p_norm

        grad_norm = math.sqrt(total_norm_sq)

        # Gradient Clipping
        clip_threshold = 5.0
        scale = 1.0
        if grad_norm > clip_threshold:
            scale = clip_threshold / (grad_norm + 1e-8)

        # Simple SGD with clipping
        for param in self.model.params.values():
            for j in range(len(param.data)):
                param.data[j] -= self.learning_rate * (param.grad[j] * scale)

        return avg_loss, grad_norm

# ============================================================================
# SHOWCASE CONFIGURATION AND UTILITIES
# ============================================================================

@dataclass
class ShowcaseConfig:
    db_path: str = "showcase.db"
    hidden_size: int = 32
    vocab_size: int = 128
    learning_rate: float = 0.05
    epochs: int = 100
    seed: int = 1337
    text: Optional[str] = None
    text_path: Optional[str] = None
    max_chars: Optional[int] = 2000
    generate_seed: str = "Hello"
    generate_length: int = 50
    temperature: float = 0.7
    log_interval: int = 20
    verify_gradients: bool = True
    sample_every: int = 50
    log_level: str = "INFO"


def setup_logger(level: str) -> logging.Logger:
    logger = logging.getLogger("smith_showcase")
    if logger.handlers:
        return logger
    numeric_level = resolve_log_level(level)
    logger.setLevel(numeric_level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def resolve_log_level(level: str) -> int:
    numeric_level = logging.getLevelName(level.upper())
    if isinstance(numeric_level, int):
        return numeric_level
    return logging.INFO


def seed_everything(seed: int, logger: logging.Logger) -> None:
    random.seed(seed)
    logger.debug("Random seed set to %d", seed)


def validate_positive_int(name: str, value: int, min_value: int = 1) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    return value


def validate_temperature(value: float, min_value: float = 0.01, max_value: float = 5.0) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"temperature must be numeric, got {type(value).__name__}")
    if value < min_value or value > max_value:
        raise ValueError(f"temperature must be between {min_value} and {max_value}, got {value}")
    return float(value)


def read_text_from_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Training text file not found: {path}")
    return path.read_text(encoding="utf-8")


def normalize_text_for_vocab(text: str, vocab_size: int) -> str:
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    max_char = vocab_size - 1
    normalized_chars = []
    for char in text:
        code = ord(char)
        if 0 <= code <= max_char:
            normalized_chars.append(char)
        else:
            normalized_chars.append("?")
    return "".join(normalized_chars)


def load_training_text(config: ShowcaseConfig, logger: logging.Logger) -> str:
    text = None
    if config.text_path:
        text = read_text_from_file(Path(config.text_path))
        logger.info("Loaded training text from %s", config.text_path)
    elif config.text:
        text = config.text
        logger.info("Loaded training text from CLI input")
    else:
        default_path = Path(__file__).resolve().parent / "README.md"
        text = read_text_from_file(default_path)
        logger.info("Loaded training text from %s", default_path)

    if config.max_chars is not None and len(text) > config.max_chars:
        text = text[:config.max_chars]
        logger.info("Trimmed training text to %d characters", config.max_chars)

    text = normalize_text_for_vocab(text, config.vocab_size)
    if len(text) < 2:
        raise ValueError("Training text must contain at least 2 characters after normalization")
    return text


def ensure_db_directory(db_path: str) -> None:
    db_parent = Path(db_path).expanduser().resolve().parent
    db_parent.mkdir(parents=True, exist_ok=True)


def verify_gradient_flow(model: 'SymbolicRNN', token_ids: List[int], logger: logging.Logger) -> bool:
    model.zero_grad()
    logits, _ = model.forward([token_ids[0]])
    loss = logits.sum()
    loss.backward()
    wxh_grad = sum(abs(g) for g in model.params['Wxh'].grad)
    logger.info("Gradient on Wxh (input weights): %.6f", wxh_grad)
    if wxh_grad == 0:
        logger.error("No gradient flow to Wxh detected.")
        return False
    return True


def run_training_loop(
    trainer: Trainer,
    token_ids: List[int],
    db: SymbolicDB,
    config: ShowcaseConfig,
    logger: logging.Logger,
) -> None:
    start_time = time.time()
    for epoch in range(config.epochs):
        inputs = token_ids[:-1]
        targets = token_ids[1:]

        avg_loss, grad_norm = trainer.train_step(inputs, targets)

        if epoch % config.log_interval == 0:
            logger.info("Epoch %d | Loss: %.4f | Grad: %.4f", epoch, avg_loss, grad_norm)
            db.log_training(epoch, avg_loss, grad_norm)

        if config.sample_every and epoch % config.sample_every == 0 and epoch != 0:
            sample = trainer.model.generate(config.generate_seed, length=40, temperature=config.temperature)
            logger.info("Sample @ epoch %d: %s", epoch, sample)

    logger.info("Training finished in %.2fs", time.time() - start_time)


def run_showcase(config: ShowcaseConfig, logger: logging.Logger) -> None:
    validate_positive_int("hidden_size", config.hidden_size)
    validate_positive_int("vocab_size", config.vocab_size)
    validate_positive_int("epochs", config.epochs, min_value=1)
    validate_positive_int("generate_length", config.generate_length)
    validate_positive_int("log_interval", config.log_interval)
    if config.learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {config.learning_rate}")
    if config.max_chars is not None:
        validate_positive_int("max_chars", config.max_chars)
    if config.sample_every is not None and config.sample_every < 0:
        raise ValueError(f"sample_every must be >= 0, got {config.sample_every}")
    validate_temperature(config.temperature)

    ensure_db_directory(config.db_path)
    seed_everything(config.seed, logger)
    text = load_training_text(config, logger)
    token_ids = [ord(c) for c in text]

    logger.info("Training on %d characters", len(text))

    db = SymbolicDB(config.db_path)
    model = SymbolicRNN(vocab_size=config.vocab_size, hidden_size=config.hidden_size, db=db)
    trainer = Trainer(model, learning_rate=config.learning_rate)

    if config.verify_gradients:
        logger.info("Verifying autograd integrity...")
        if not verify_gradient_flow(model, token_ids, logger):
            db.close()
            raise RuntimeError("Gradient verification failed")

    logger.info("Starting training...")
    run_training_loop(trainer, token_ids, db, config, logger)

    logger.info("Generating text (seed: %r)...", config.generate_seed)
    generated = model.generate(config.generate_seed, length=config.generate_length, temperature=config.temperature)
    logger.info("Output: %s", generated)

    key = db.save_params(model.params)
    logger.info("Model saved with key: %s", key)
    db.close()


def run_self_tests(logger: logging.Logger) -> None:
    logger.info("Running self-tests...")

    temperature_values = list(range(-1, 13))
    for value in temperature_values:
        try:
            validate_temperature(value)
            if value <= 0 or value > 5:
                raise AssertionError(f"temperature validation accepted invalid value: {value}")
        except ValueError:
            if 0 < value <= 5:
                raise AssertionError(f"temperature validation rejected valid value: {value}")

    for value in range(-1, 13):
        try:
            validate_positive_int("hidden_size", value)
            if value <= 0:
                raise AssertionError(f"validate_positive_int accepted invalid value: {value}")
        except ValueError:
            if value > 0:
                raise AssertionError(f"validate_positive_int rejected valid value: {value}")

    sample_text = "Café μ-test"
    normalized = normalize_text_for_vocab(sample_text, 128)
    if "?" not in normalized:
        raise AssertionError("normalize_text_for_vocab failed to replace out-of-range characters")

    db_path = "self_test_showcase.db"
    config = ShowcaseConfig(
        db_path=db_path,
        epochs=2,
        hidden_size=8,
        vocab_size=128,
        learning_rate=0.1,
        max_chars=200,
        log_interval=1,
        sample_every=1,
        generate_length=10,
        temperature=0.9,
    )
    try:
        run_showcase(config, logger)
    finally:
        db_file = Path(db_path)
        if db_file.exists():
            db_file.unlink()
    logger.info("Self-tests completed successfully.")

# ============================================================================
# MAIN SHOWCASE
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smith Production Showcase")
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--db-path", type=str, help="SQLite DB path")
    parser.add_argument("--hidden-size", type=int, help="Hidden state size")
    parser.add_argument("--vocab-size", type=int, help="Vocabulary size")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--epochs", type=int, help="Training epochs")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--text", type=str, help="Training text content")
    parser.add_argument("--text-path", type=str, help="Training text file path")
    parser.add_argument("--max-chars", type=int, help="Max characters to train on")
    parser.add_argument("--generate-seed", type=str, help="Seed text for generation")
    parser.add_argument("--generate-length", type=int, help="Generated text length")
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument("--log-interval", type=int, help="Epoch logging interval")
    parser.add_argument("--sample-every", type=int, help="Epoch interval for sample generation")
    parser.add_argument("--no-verify-gradients", action="store_true", help="Disable gradient verification")
    parser.add_argument("--log-level", type=str, help="Logging level")
    parser.add_argument("--self-test", action="store_true", help="Run self-tests and exit")
    return parser.parse_args()


def load_config_file(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    raw = config_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object")
    return data


def build_config_from_args(args: argparse.Namespace) -> ShowcaseConfig:
    base_config = ShowcaseConfig()
    if args.config:
        overrides = load_config_file(args.config)
        for key, value in overrides.items():
            if hasattr(base_config, key):
                setattr(base_config, key, value)
            else:
                raise ValueError(f"Unknown config key: {key}")

    if args.db_path is not None:
        base_config.db_path = args.db_path
    if args.hidden_size is not None:
        base_config.hidden_size = args.hidden_size
    if args.vocab_size is not None:
        base_config.vocab_size = args.vocab_size
    if args.learning_rate is not None:
        base_config.learning_rate = args.learning_rate
    if args.epochs is not None:
        base_config.epochs = args.epochs
    if args.seed is not None:
        base_config.seed = args.seed
    if args.text is not None:
        base_config.text = args.text
    if args.text_path is not None:
        base_config.text_path = args.text_path
    if args.max_chars is not None:
        base_config.max_chars = args.max_chars
    if args.generate_seed is not None:
        base_config.generate_seed = args.generate_seed
    if args.generate_length is not None:
        base_config.generate_length = args.generate_length
    if args.temperature is not None:
        base_config.temperature = args.temperature
    if args.log_interval is not None:
        base_config.log_interval = args.log_interval
    if args.sample_every is not None:
        base_config.sample_every = args.sample_every
    base_config.verify_gradients = not args.no_verify_gradients
    if args.log_level is not None:
        base_config.log_level = args.log_level
    return base_config


def main() -> int:
    args = parse_args()
    logger = setup_logger(args.log_level or ShowcaseConfig().log_level)
    logger.info("=== Smith Production Showcase ===")

    if args.self_test:
        run_self_tests(logger)
        return 0

    config = build_config_from_args(args)
    resolved_level = resolve_log_level(config.log_level)
    if logger.level != resolved_level:
        logger.setLevel(resolved_level)
    run_showcase(config, logger)
    logger.info("Showcase complete.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
