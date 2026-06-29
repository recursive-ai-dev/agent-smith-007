"""
Pipeline Trainer — Production training loop
==========================================
Coordinates data loading, forward/backward passes,
optimisation steps, and diagnostic logging.
"""

import json
import logging
import os
import time
from typing import List, Dict, Any, Optional

from ..classifier.model import AgentSmith
from ..classifier.adam import AdamOptimizer
from ..classifier.precision import MixedPrecisionManager
from .data import DataLoader

logger = logging.getLogger(__name__)


class PipelineTrainer:
    """
    Standardized trainer for AgentSmith.
    """

    def __init__(
        self,
        model: AgentSmith,
        optimizer: AdamOptimizer,
        dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        amp: bool = False,
        log_dir: str = "logs",
    ):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.amp_manager = MixedPrecisionManager() if amp else None
        self.log_dir = log_dir

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.history: List[Dict[str, Any]] = []
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch: int) -> float:
        """Run one full epoch over the training data."""
        self.optimizer.zero_grad()
        total_loss = 0.0
        start_time = time.time()

        for i, (text, label) in enumerate(self.dataloader):
            token_ids = self.model.tokenizer.encode(text)

            # Forward
            logits, probs, diag = self.model.forward(token_ids)
            loss = self.model.cross_entropy_loss(logits, label)

            # Scaled Backward
            if self.amp_manager:
                scaled_loss = self.amp_manager.scale_loss(loss)
                scaled_loss.backward()
                self.amp_manager.unscale_gradients(self.model.parameters())
            else:
                loss.backward()

            # Optimizer step
            if not self.amp_manager or not self.amp_manager.should_skip_step:
                self.optimizer.step()
                if self.amp_manager:
                    self.amp_manager.update()

            self.optimizer.zero_grad()
            total_loss += loss.data[0]

            if i % 10 == 0:
                elapsed = time.time() - start_time
                logger.info(
                    "Epoch %d | Batch %d/%d | Loss: %.4f | Speed: %.2f ms/sample",
                    epoch, i, len(self.dataloader), loss.data[0],
                    (elapsed / (i+1)) * 1000
                )

        return total_loss / len(self.dataloader)

    def validate(self) -> Dict[str, float]:
        """Compute metrics on validation set."""
        if not self.val_dataloader:
            return {}

        total_loss = 0.0
        correct = 0
        total = 0

        for text, label in self.val_dataloader:
            token_ids = self.model.tokenizer.encode(text)
            logits, probs, _ = self.model.forward(token_ids)
            loss = self.model.cross_entropy_loss(logits, label)

            total_loss += loss.data[0]
            pred = probs.data.index(max(probs.data))
            if pred == label:
                correct += 1
            total += 1

        return {
            "val_loss": total_loss / total,
            "val_accuracy": correct / total
        }

    def fit(self, epochs: int):
        """Full training procedure."""
        logger.info("Starting training: %d epochs, total params: %d",
                    epochs, self.model.param_count())

        for epoch in range(epochs):
            train_loss = self.train_epoch(epoch)
            metrics = self.validate()

            summary = {
                "epoch": epoch,
                "train_loss": train_loss,
                **metrics
            }
            self.history.append(summary)

            logger.info("Epoch %d complete: %s", epoch, summary)

            # Save best model
            val_loss = metrics.get("val_loss", train_loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model.json")

        # Final diagnostics export
        with open(os.path.join(self.log_dir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

    def save_checkpoint(self, filename: str):
        """Persist model and optimizer state."""
        path = os.path.join(self.log_dir, filename)
        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.model.config.__dict__
        }
        with open(path, "w") as f:
            json.dump(checkpoint, f)
        logger.info("Saved checkpoint to %s", path)
