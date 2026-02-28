"""
Trainer - Training Interface for SymbolicRNN

Provides a high-level training loop with progress tracking,
model saving, and evaluation.
"""

import math
import time
from typing import List, Optional, Callable, Dict, Any, Tuple

from .gru_model import GatedRecurrentUnit
from .database import SymbolicDB
from .tensor import NanoTensor


class Trainer:
    """
    Training interface for SymbolicRNN model.
    Handles training loop, optimization, and progress tracking.
    """
    
    def __init__(
        self,
        model: GatedRecurrentUnit,
        learning_rate: float = 0.01,
        clip_grad: float = 5.0,
        verbose: bool = True
    ):
        """
        Initialize trainer.
        
        Args:
            model: SymbolicRNN model to train
            learning_rate: Learning rate for gradient descent
            clip_grad: Gradient clipping threshold
            verbose: Whether to print training progress
        """
        self.model = model
        self.learning_rate = learning_rate
        self.clip_grad = clip_grad
        self.verbose = verbose
        self.training_history = []
    
    def compute_loss(self, logits: NanoTensor, target: int) -> NanoTensor:
        """
        Compute cross-entropy loss using branchless operations.
        
        Args:
            logits: Model output logits
            target: Target token ID
            
        Returns:
            Loss value as NanoTensor
        """
        # Apply softmax to get probabilities
        max_logit = max(logits.data)
        exp_logits = [math.exp(l - max_logit) for l in logits.data]
        sum_exp = sum(exp_logits)
        
        # Get probability of target
        target_prob = exp_logits[target] / (sum_exp + 1e-8)
        
        # Negative log likelihood
        loss_val = -math.log(target_prob + 1e-8)
        
        # Create loss tensor with gradient
        loss = NanoTensor([loss_val])
        
        # Manually set up gradient computation for softmax + NLL
        # This is simplified for the minimal implementation
        def _backward():
            if logits.requires_grad:
                # Gradient of softmax + NLL
                probs = [e / (sum_exp + 1e-8) for e in exp_logits]
                for i in range(len(logits.grad)):
                    if i == target:
                        logits.grad[i] += (probs[i] - 1.0) * loss.grad[0]
                    else:
                        logits.grad[i] += probs[i] * loss.grad[0]
        
        loss._backward = _backward
        loss._parents = (logits,)
        
        return loss
    
    def clip_gradients(self) -> float:
        """
        Clip gradients to prevent exploding gradients.
        
        Returns:
            Gradient norm before clipping
        """
        total_norm = 0.0
        for param in self.model.params.values():
            if param.grad:
                for g in param.grad:
                    total_norm += g * g
        
        grad_norm = math.sqrt(total_norm)
        
        if grad_norm > self.clip_grad:
            scale = self.clip_grad / (grad_norm + 1e-8)
            for param in self.model.params.values():
                if param.grad:
                    param.grad = [g * scale for g in param.grad]
        
        return grad_norm
    
    def update_parameters(self):
        """Update model parameters using gradient descent."""
        for param in self.model.params.values():
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
        
        Args:
            inputs: List of input token IDs
            targets: List of target token IDs
            
        Returns:
            Tuple of (Average loss for this step, Gradient norm)
        """
        self.model.zero_grad()
        
        total_loss = 0.0
        h = None
        
        # Process sequence
        for i in range(len(inputs)):
            # Forward pass
            logits, h = self.model.forward([inputs[i]], h)
            
            # Compute loss
            loss = self.compute_loss(logits, targets[i])
            total_loss += loss.data[0]
            
            # Backward pass
            loss.backward()
        
        # Clip gradients
        grad_norm = self.clip_gradients()
        
        # Update parameters
        self.update_parameters()
        
        return total_loss / len(inputs), grad_norm
    
    def train(
        self,
        text: str,
        epochs: int,
        seq_length: int = 25,
        save_every: int = 100,
        eval_every: int = 50,
        callback: Optional[Callable[[int, float, str], None]] = None
    ):
        """
        Train the model on text data.
        
        Args:
            text: Training text
            epochs: Number of training epochs
            seq_length: Length of training sequences
            save_every: Save model every N epochs
            eval_every: Evaluate model every N epochs
            callback: Optional callback function(epoch, loss, sample_text)
        """
        # Convert text to token IDs
        token_ids = [ord(c) % self.model.vocab_size for c in text]
        
        if self.verbose:
            print(f"Training on {len(text)} characters ({len(token_ids)} tokens)")
            print(f"Vocabulary size: {self.model.vocab_size}")
            print(f"Hidden size: {self.model.hidden_size}")
            print(f"Epochs: {epochs}")
            print("-" * 60)
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # Sample random starting position
            if len(token_ids) <= seq_length + 1:
                # If text is too short, just use the whole thing
                start_idx = 0
            else:
                start_idx = epoch % (len(token_ids) - seq_length - 1)
            
            # Get input and target sequences
            inputs = token_ids[start_idx:start_idx + seq_length]
            targets = token_ids[start_idx + 1:start_idx + seq_length + 1]
            
            # Train step
            loss, grad_norm = self.train_step(inputs, targets)
            
            # Record history
            self.training_history.append({
                'epoch': epoch,
                'loss': loss,
                'grad_norm': grad_norm,
                'time': time.time() - epoch_start
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
                # Generate a short sample
                seed = text[:5] if len(text) >= 5 else text
                sample = self.model.generate(seed, length=20, temperature=0.7)
                
                if self.verbose:
                    print(f"Sample: {repr(sample)}")
                    print("-" * 60)
                
                # Call callback if provided
                if callback:
                    callback(epoch, loss, sample)
            
            # Save checkpoint
            if epoch % save_every == 0:
                param_key = self.model.db.save_params(self.model.params)
                if self.verbose:
                    print(f"✓ Saved checkpoint: {param_key}")
        
        total_time = time.time() - start_time
        if self.verbose:
            print(f"\nTraining complete in {total_time:.2f}s")
            print(f"Final loss: {self.training_history[-1]['loss']:.4f}")
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get training history."""
        return self.training_history
