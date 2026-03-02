"""
Trainer - High-Fidelity Refactor
Includes Unified Loss-Logit Bridge and Training Progress Validation.
"""

import random
import math
import time
from typing import List, Optional, Callable, Dict, Any, Tuple

from .gru_model import GatedRecurrentUnit
from .database import SymbolicDB
from .tensor import NanoTensor


class Trainer:
    """
    Refactored Training interface for GatedRecurrentUnit.
    Handles stable training loop and progress verification.
    """
    
    def __init__(
        self,
        model: GatedRecurrentUnit,
        learning_rate: float = 0.01,
        clip_grad: float = 5.0,
        verbose: bool = True
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.clip_grad = clip_grad
        self.verbose = verbose
        self.training_history = []
    
    def compute_loss(self, logits: NanoTensor, target: int) -> NanoTensor:
        """
        Compute cross-entropy loss with stable softmax gradient accumulation.
        """
        max_l = max(logits.data)
        exp_l = [math.exp(l - max_l) for l in logits.data]
        sum_e = sum(exp_l)
        
        target_prob = exp_l[target] / (sum_e + 1e-12)
        loss_val = -math.log(target_prob + 1e-12)
        
        loss = NanoTensor([loss_val], _parents=(logits,), _op='loss')
        
        def _backward():
            if logits.requires_grad:
                probs = [e / (sum_e + 1e-12) for e in exp_l]
                for i in range(len(logits.grad)):
                    # Softmax-Log-Likelihood gradient
                    grad_val = (probs[i] - (1.0 if i == target else 0.0)) * loss.grad[0]
                    logits.grad[i] += grad_val
        
        loss._backward = _backward
        return loss
    
    def clip_gradients(self) -> float:
        total_norm = 0.0
        for p in self.model.params.values():
            if p.grad:
                total_norm += sum(g*g for g in p.grad)
        
        grad_norm = math.sqrt(total_norm)
        if grad_norm > self.clip_grad:
            scale = self.clip_grad / (grad_norm + 1e-12)
            for p in self.model.params.values():
                if p.grad:
                    p.grad = [g * scale for g in p.grad]
        return grad_norm
    
    def update_parameters(self):
        for p in self.model.params.values():
            if p.grad:
                for i in range(len(p.data)):
                    p.data[i] -= self.learning_rate * p.grad[i]
    
    def train_step(self, inputs: List[int], targets: List[int]) -> Tuple[float, float]:
        self.model.zero_grad()
        total_loss = 0.0
        h = None
        
        for i in range(len(inputs)):
            # Forward pass
            logits, h = self.model.forward([inputs[i]], h)
            
            # Loss computation
            loss = self.compute_loss(logits, targets[i])
            total_loss += loss.data[0]
            
            # Backward pass (AtomicGraph engine)
            loss.backward()

            # Detach hidden state for long sequences (truncated BPTT)
            h = NanoTensor(h.data, requires_grad=False)
        
        gnorm = self.clip_gradients()
        self.update_parameters()
        return total_loss / len(inputs), gnorm

    def train(
        self,
        text: str,
        epochs: int,
        seq_length: int = 25,
        save_every: int = 100,
        eval_every: int = 50,
        callback: Optional[Callable] = None
    ):
        token_ids = [ord(c) % self.model.vocab_size for c in text]
        start_t = time.time()
        
        for epoch in range(1, epochs + 1):
            e_start = time.time()
            s_idx = random.randint(0, max(0, len(token_ids) - seq_length - 1))
            inputs = token_ids[s_idx : s_idx + seq_length]
            targets = token_ids[s_idx + 1 : s_idx + seq_length + 1]
            
            loss, gnorm = self.train_step(inputs, targets)
            
            self.training_history.append({
                'epoch': epoch,
                'loss': loss,
                'grad_norm': gnorm,
                'time': time.time() - e_start
            })
            
            if epoch % eval_every == 0 and self.verbose:
                elapsed = time.time() - start_t
                sample = self.model.generate(text[:5], length=20, temperature=0.7)
                print(f"Epoch {epoch}/{epochs} | Loss: {loss:.4f} | GNorm: {gnorm:.4f} | Time: {elapsed:.2f}s")
                print(f"Sample: {repr(sample)}")
                if callback:
                    callback(epoch, loss, sample)

            if epoch % save_every == 0:
                self.model.db.save_params(self.model.params)

    def get_history(self) -> List[Dict[str, Any]]:
        return self.training_history
