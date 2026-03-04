"""
Training Pipeline
==================
Full training loop for AgentSmith.

Implements:
  • Epoch / step loop over a DataLoader
  • Mixed-precision (simulated) forward + scaled backward
  • Adam gradient step with warmup scheduling
  • GSAR statistics update after every batch
  • Diagnostic hook callbacks (loss, grads, Jacobian, Hessian)
  • Periodic evaluation on validation set (accuracy + avg loss)
  • Checkpoint save / restore
  • Clean console logging

Mathematics of the training update
------------------------------------
For each sample (x, y):
  1. Tokenise x → token_ids
  2. Forward:  logits, probs, diag = model(token_ids)
  3. Loss:     L = CrossEntropy(logits, y)
  4. (AMP)     L_scaled = S · L          where S = loss_scale
  5. Backward: L_scaled.backward()
  6. Unscale:  g ← g / S
  7. Clip:     g ← g · min(1, clip / ‖g‖)
  8. Adam:     θ ← θ − α [m̂ / (√v̂ + ε) + λθ]
"""

import math
import os
import json
import time
from typing import Optional, Tuple

from ..tensor import NanoTensor
from ..classifier.config import AgentSmithConfig
from ..classifier.model import AgentSmith
from ..classifier.adam import AdamOptimizer
from ..classifier.precision import MixedPrecisionContext
from ..pipeline.data import DataLoader, Dataset
from ..diagnostics.hooks import DiagnosticsManager


class Trainer:
    """
    Orchestrates the complete training workflow.

    Parameters
    ----------
    model      : AgentSmith instance
    config     : AgentSmithConfig
    log_dir    : directory for checkpoints and diagnostic logs
    """

    def __init__(
        self,
        model: AgentSmith,
        config: AgentSmithConfig,
        log_dir: str = "logs",
    ):
        self.model  = model
        self.config = config
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # ── Optimiser ────────────────────────────────────────────────
        self.optimizer = AdamOptimizer(
            params        = model.parameters(),
            lr            = config.lr,
            beta1         = config.beta1,
            beta2         = config.beta2,
            eps           = config.adam_eps,
            weight_decay  = config.weight_decay,
            warmup_steps  = config.warmup_steps,
            grad_clip     = config.grad_clip,
        )

        # ── Mixed precision ───────────────────────────────────────────
        self.amp = MixedPrecisionContext(
            enabled    = config.use_amp,
            bits       = config.amp_bits,
            loss_scale = config.loss_scale,
        )

        # ── Diagnostics ───────────────────────────────────────────────
        self.diag = DiagnosticsManager(
            model   = model,
            config  = config,
            log_dir = log_dir,
        )
        model.register_forward_hook(self.diag.on_activation)

        # ── State ─────────────────────────────────────────────────────
        self.global_step:  int   = 0
        self.best_val_acc: float = 0.0
        self.train_losses: list  = []

    # ── Core training step ───────────────────────────────────────────────

    def _train_step(
        self,
        text: str,
        label: int,
    ) -> Tuple[float, dict]:
        """
        Single sample forward + backward + optimiser step.

        Returns (loss_value: float, diagnostics: dict).
        """
        # ── 1. Tokenise ──────────────────────────────────────────
        token_ids = self.model.tokenizer.encode(text)

        # ── 2. Zero gradients ────────────────────────────────────
        self.optimizer.zero_grad()

        # ── 3. Forward pass (mixed precision) ────────────────────
        with self.amp.forward():
            logits, probs, diag = self.model.forward(token_ids, use_gsar=True)

        # ── 4. Cross-entropy loss ─────────────────────────────────
        loss = self.model.cross_entropy_loss(logits, label)

        # ── 5. Loss scaling (AMP) ─────────────────────────────────
        scaled_loss = self.amp.scale(loss)

        # ── 6. Backward ───────────────────────────────────────────
        scaled_loss.backward()

        # ── 7. Unscale gradients ──────────────────────────────────
        ok = self.amp.unscale(self.model.parameters())
        if not ok:
            # Gradient overflow → skip this step
            return float('nan'), diag

        # ── 8. Adam step (includes clipping and warmup) ───────────
        opt_stats = self.optimizer.step()

        loss_val = loss.data[0]
        return loss_val, {**diag, **opt_stats}

    # ── Evaluation ───────────────────────────────────────────────────────

    @staticmethod
    def _argmax(v: list) -> int:
        return v.index(max(v))

    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate accuracy and average loss on a dataset.

        Returns (accuracy: float, avg_loss: float).
        """
        correct = 0
        total   = 0
        total_loss = 0.0

        for text, label in loader:
            token_ids = self.model.tokenizer.encode(text)
            logits, probs, _ = self.model.forward(token_ids, use_gsar=False)
            pred = self._argmax(probs.data)
            correct += int(pred == label)
            total   += 1
            loss = self.model.cross_entropy_loss(logits, label)
            total_loss += loss.data[0]

        acc      = correct / max(total, 1)
        avg_loss = total_loss / max(total, 1)
        return acc, avg_loss

    # ── Main training loop ───────────────────────────────────────────────

    def train(
        self,
        train_loader:  DataLoader,
        val_loader:    Optional[DataLoader] = None,
        num_epochs:    Optional[int] = None,
        verbose:       bool = True,
    ) -> dict:
        """
        Run the full training loop.

        Parameters
        ----------
        train_loader  : DataLoader over training samples
        val_loader    : optional DataLoader for validation
        num_epochs    : override config.num_epochs if provided
        verbose       : print per-epoch summary

        Returns
        -------
        history : dict with training statistics
        """
        epochs = num_epochs if num_epochs is not None else self.config.num_epochs
        history: dict = {
            "train_loss": [],
            "val_acc":    [],
            "val_loss":   [],
        }

        # ── Collect all token sequences for GSAR warm-up ─────────
        all_sequences = []
        for text, _ in train_loader.dataset.samples:
            ids = self.model.tokenizer.encode(text)
            all_sequences.append(ids)

        # Initial GSAR statistics pass (2 warm-up passes before first epoch)
        self.model.gsar.update_statistics(all_sequences)
        self.model.gsar.update_statistics(all_sequences)

        if verbose:
            print(f"\n{'═'*65}")
            print(f"  AGENT SMITH  — Training starts")
            print(f"  Model params : {self.model.param_count():,}")
            print(f"  GSAR symbols : {len(self.model.gsar._registry)}")
            print(f"  Epochs       : {epochs}")
            print(f"  Samples/epoch: {len(train_loader)}")
            print(f"{'═'*65}\n")

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            epoch_losses = []
            epoch_seqs   = []

            for text, label in train_loader:
                self.global_step += 1

                # Training step
                loss_val, step_diag = self._train_step(text, label)

                if math.isnan(loss_val):
                    continue
                epoch_losses.append(loss_val)
                self.train_losses.append(loss_val)

                # Collect token sequence for GSAR update
                epoch_seqs.append(self.model.tokenizer.encode(text))

                # Diagnostic callbacks
                sep_expl = step_diag.get("sep")
                self.diag.after_step(
                    step        = self.global_step,
                    loss        = loss_val,
                    optimizer   = self.optimizer,
                    sep_explanation = sep_expl,
                )

                # Periodic console log
                if verbose and self.global_step % 20 == 0:
                    gn = self.diag.grad_norm_history[-1] if self.diag.grad_norm_history else 0
                    print(
                        f"  step={self.global_step:5d}  "
                        f"loss={loss_val:.4f}  "
                        f"‖∇‖={gn:.3e}  "
                        f"lr={self.optimizer._current_lr():.2e}"
                    )

            # Update GSAR with epoch sequences
            new_syms = self.model.gsar.update_statistics(epoch_seqs)

            avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            history["train_loss"].append(avg_loss)

            # ── Validation ─────────────────────────────────────
            val_acc, val_loss = 0.0, 0.0
            if val_loader is not None:
                val_acc, val_loss = self.evaluate(val_loader)
                history["val_acc"].append(val_acc)
                history["val_loss"].append(val_loss)

                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self._save_checkpoint("best_model.json")

            elapsed = time.time() - epoch_start

            if verbose:
                syms = len(self.model.gsar._registry)
                print(
                    f"\n  Epoch {epoch:3d}/{epochs}  "
                    f"train_loss={avg_loss:.4f}  "
                    + (f"val_acc={val_acc:.3f}  val_loss={val_loss:.4f}  " if val_loader else "")
                    + f"gsar_syms={syms}  "
                    f"new_syms={new_syms}  "
                    f"elapsed={elapsed:.1f}s\n"
                )

            # Periodic checkpoint
            if self.global_step % self.config.checkpoint_freq == 0:
                self._save_checkpoint(f"ckpt_step{self.global_step}.json")

        # Final diagnostics save
        self.diag.save("diagnostics.json")
        if verbose:
            print(self.diag.report())

        return history

    # ── Checkpoint helpers ────────────────────────────────────────────────

    def _save_checkpoint(self, filename: str):
        path = os.path.join(self.log_dir, filename)
        ckpt = {
            "model":     self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step":      self.global_step,
        }
        with open(path, "w") as f:
            json.dump(ckpt, f)

    def load_checkpoint(self, path: str):
        with open(path) as f:
            ckpt = json.load(f)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = ckpt.get("step", 0)
        print(f"Loaded checkpoint from {path}  (step={self.global_step})")
