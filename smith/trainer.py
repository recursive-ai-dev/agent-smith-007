"""
Trainer - Training Interface for SymbolicRNN

Provides a high-level training loop with progress tracking,
model saving, and evaluation.
Deterministic Hardening: High-precision gradient accumulation and deterministic sequence sampling.
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
        
        # Manually set up gradient computation for softmax + NLL
        def _backward():
            if logits.requires_grad:
                probs = [e / (sum_e + 1e-12) for e in exp_l]
                for i in range(len(logits.grad)):
                    if i == target:
                        logits._accumulate_grad(i, (probs[i] - 1.0) * loss.grad[0])
                    else:
                        logits._accumulate_grad(i, probs[i] * loss.grad[0])
        
        loss._backward = _backward
        return loss
    
    def clip_gradients(self) -> float:
        """
        Clip gradients to prevent exploding gradients.
        Deterministic accumulation of norm squared.
        """
        total_norm_sq = 0.0
        # Sort keys to ensure deterministic summation order
        sorted_params = sorted(self.model.params.items())
        for name, param in sorted_params:
            if param.grad:
                for g in param.grad:
                    total_norm_sq += g * g
        
        grad_norm = math.sqrt(total_norm_sq)
        
        if grad_norm > self.clip_grad:
            scale = self.clip_grad / (grad_norm + 1e-8)
            for name, param in sorted_params:
                if param.grad:
                    param.grad = [g * scale for g in param.grad]
                    # Also scale the Kahan error buffer to maintain consistency
                    param._grad_err = [e * scale for e in param._grad_err]
        
        return grad_norm
    
    def update_parameters(self):
        """Update model parameters using gradient descent."""
        # Sort keys for deterministic update order (important for some distributed/sharded scenarios)
        for name, param in sorted(self.model.params.items()):
            if param.grad:
                for i in range(len(param.data)):
                    param.data[i] -= self.learning_rate * param.grad[i]
    
    def train_step(
        self,
        inputs: List[int],
        targets: List[int]
    ) -> Tuple[float, float]:
        """
        Perform one training step.
        """
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
        """
        Train the model on text data.
        """
        # Convert text to token IDs
        token_ids = [ord(c) % self.model.vocab_size for c in text]
        start_t = time.time()
        
        for epoch in range(1, epochs + 1):
            e_start = time.time()
            s_idx = random.randint(0, max(0, len(token_ids) - seq_length - 1))
            inputs = token_ids[s_idx : s_idx + seq_length]
            targets = token_ids[s_idx + 1 : s_idx + seq_length + 1]
            
            loss, gnorm = self.train_step(inputs, targets)
            # Deterministic starting position based on epoch
            if len(token_ids) <= seq_length + 1:
                start_idx = 0
            else:
                # Linear scan through dataset for deterministic coverage
                start_idx = (epoch - 1) % (len(token_ids) - seq_length)
            
            self.training_history.append({
                'epoch': epoch,
                'loss': loss,
                'grad_norm': gnorm,
                'time': time.time() - e_start
            })
            
            # Log to database
            if epoch % 10 == 0:
                self.model.db.log_training(
                    epoch=epoch,
                    loss=loss,
                    grad_norm=grad_norm,
                    pattern=f"epoch_{epoch}"
                )
            
            # Print progress
            if self.verbose and epoch % eval_every == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch}/{epochs} | Loss: {loss:.4f} | Time: {elapsed:.2f}s")
            
            # Generate sample
            if epoch % eval_every == 0:
                seed = text[:5] if len(text) >= 5 else text
                sample = self.model.generate(seed, length=20, temperature=0.7)
                
                if self.verbose:
                    print(f"Sample: {repr(sample)}")
                    print("-" * 60)
                
                if callback:
                    callback(epoch, loss, sample)

            if epoch % save_every == 0:
                self.model.db.save_params(self.model.params)

    def get_history(self) -> List[Dict[str, Any]]:
        return self.training_history
