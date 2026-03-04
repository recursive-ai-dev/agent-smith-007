#!/usr/bin/env python3
"""
run_classifier.py — AgentSmith Entry Point
============================================
Trains the AgentSmith multi-domain text classifier and demonstrates
all components: GSAR, SEP, Adam, mixed precision, Jacobian, Hessian.

Usage
-----
    python run_classifier.py                      # full training run
    python run_classifier.py --epochs 3 --demo    # quick demo
    python run_classifier.py --predict "your text here"

No external dependencies required.  Pure Python 3.11+ stdlib only.
"""

import argparse
import sys
import os

# ── Add project root to path ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smith.classifier.config import AgentSmithConfig, DOMAINS
from smith.classifier.model import AgentSmith
from smith.pipeline.data import Dataset, DataLoader
from smith.pipeline.trainer import Trainer
from smith.classifier.jacobian import numerical_jacobian, jacobian_stats
from smith.classifier.hessian import diagonal_hessian, curvature_summary
from smith.classifier.precision import MixedPrecisionContext


# ─────────────────────────────────────────────────────────────────────────────
# Demo: single-sample prediction with full diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def demo_prediction(model: AgentSmith, text: str):
    print(f"\n{'─'*65}")
    print(f"  TEXT: {text[:80]}")
    print(f"{'─'*65}")
    result = model.predict(text)
    print(f"  Predicted domain : {result['label']}")
    print(f"  Class ID         : {result['class_id']}")
    print(f"  Confidence       : {result['confidence']:.4f}")
    print(f"\n  Class probabilities:")
    probs_sorted = sorted(enumerate(result['probs']), key=lambda x: -x[1])
    for class_id, prob in probs_sorted[:5]:
        marker = " ←" if class_id == result['class_id'] else ""
        print(f"    [{class_id:2d}] {DOMAINS[class_id]:25s}  {prob:.4f}{marker}")
    print(f"\n{result['explanation']}")


# ─────────────────────────────────────────────────────────────────────────────
# Demo: Jacobian & Hessian analysis
# ─────────────────────────────────────────────────────────────────────────────

def demo_calculus(model: AgentSmith):
    print(f"\n{'═'*65}")
    print("  JACOBIAN & HESSIAN DIAGNOSTICS")
    print(f"{'═'*65}")

    # ── Jacobian: gradient of the first class logit w.r.t. chunk_clf weights
    clf = model.sep.chunk_clf
    w   = clf.weight.data
    n   = min(6, len(w))

    def logit0_from_w(w_vals):
        for i in range(n):
            clf.weight.data[i] = w_vals[i]
        from smith.tensor import NanoTensor
        dummy = NanoTensor([0.3] * model.config.d_model, requires_grad=False)
        out = clf(dummy)
        return out.data[0]

    w_snap = w[:n]
    grad   = numerical_jacobian(logit0_from_w, w_snap)
    stats  = jacobian_stats([grad])
    # Restore
    for i in range(n):
        clf.weight.data[i] = w_snap[i]

    print(f"\n  Jacobian (∂logit₀/∂w₀…w_{n-1}):")
    print(f"    Frobenius norm : {stats.get('frobenius_norm', 0):.6f}")
    print(f"    Max |∂f/∂wᵢ|  : {stats.get('max_abs', 0):.6f}")
    print(f"    Min |∂f/∂wᵢ|  : {stats.get('min_abs', 0):.6f}")
    for i, g in enumerate(grad):
        print(f"    ∂logit₀/∂w{i}  = {g:+.6f}")

    # ── Hessian diagonal: curvature of ‖logits‖² w.r.t. same weights
    def quad_loss(w_vals):
        for i in range(n):
            clf.weight.data[i] = w_vals[i]
        from smith.tensor import NanoTensor
        dummy = NanoTensor([0.3] * model.config.d_model, requires_grad=False)
        out = clf(dummy)
        return sum(v * v for v in out.data)

    diag_h = diagonal_hessian(quad_loss, w_snap)
    # Restore
    for i in range(n):
        clf.weight.data[i] = w_snap[i]

    print(f"\n  Hessian diagonal (∂²‖logits‖²/∂wᵢ²):")
    for i, h in enumerate(diag_h):
        print(f"    H[{i},{i}] = {h:+.6f}")

    # Curvature summary
    summary = curvature_summary(quad_loss, w_snap, sample_size=n)
    # Restore
    for i in range(n):
        clf.weight.data[i] = w_snap[i]

    print(f"\n  Curvature summary:")
    print(f"    Mean curvature     : {summary['mean_curvature']:.6f}")
    print(f"    Max curvature      : {summary['max_curvature']:.6f}")
    print(f"    Condition number ≈ : {summary['condition_number_est']:.2f}")
    print(f"    Gershgorin bounds  : [{summary['gershgorin_lo']:.4f}, {summary['gershgorin_hi']:.4f}]")


# ─────────────────────────────────────────────────────────────────────────────
# Demo: GSAR status
# ─────────────────────────────────────────────────────────────────────────────

def demo_gsar(model: AgentSmith):
    print(f"\n{'═'*65}")
    print("  GSAR — SYMBOLIC ARRAY REASONING STATUS")
    print(f"{'═'*65}")
    stats = model.gsar.stats()
    print(f"  Registered symbols   : {stats.get('registered_symbols', 0)}")
    print(f"  Total n-gram windows : {stats.get('total_windows_seen', 0)}")
    print(f"  Mean priority score  : {stats.get('mean_priority', 0):.4f}")
    print(f"  Window size dist     : {stats.get('window_size_dist', {})}")
    # Show a few registered patterns
    registry = model.gsar._registry
    if registry:
        print(f"\n  Top-5 symbols by count:")
        top = sorted(registry.values(), key=lambda e: -e.count)[:5]
        for e in top:
            # Decode pattern hashes back to printable form
            print(
                f"    pattern={e.pattern}  count={e.count}  "
                f"priority={e.priority:.3f}  sym_id={e.symbol_id}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Demo: mixed-precision report
# ─────────────────────────────────────────────────────────────────────────────

def demo_precision(model: AgentSmith, config: AgentSmithConfig):
    print(f"\n{'═'*65}")
    print("  MIXED-PRECISION CONTEXT")
    print(f"{'═'*65}")
    amp = MixedPrecisionContext(
        enabled    = config.use_amp,
        bits       = config.amp_bits,
        loss_scale = config.loss_scale,
    )
    report = amp.gradient_precision_report(model.parameters())
    print(f"  AMP enabled  : {amp.enabled}")
    print(f"  Target bits  : {amp.bits}")
    print(f"  Loss scale   : {amp.loss_scale}")
    print(f"  Grad report  : {report}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AgentSmith classifier")
    parser.add_argument("--epochs",  type=int,   default=5,    help="Training epochs")
    parser.add_argument("--lr",      type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--demo",    action="store_true",      help="Run quick demo only")
    parser.add_argument("--predict", type=str,   default=None, help="Classify a text string")
    args = parser.parse_args()

    # ── Configuration ──────────────────────────────────────────────
    config = AgentSmithConfig(
        d_model      = 64,
        num_heads    = 2,
        d_k          = 16,
        d_v          = 16,
        num_layers   = 2,
        d_ff         = 128,
        vocab_size   = 2048,
        max_seq_len  = 64,
        num_classes  = 12,
        lr           = args.lr,
        num_epochs   = args.epochs,
        use_amp      = True,
        amp_bits     = 32,
        loss_scale   = 64.0,
        warmup_steps = 30,
        gsar_min_freq= 2,
        sep_chunk_size= 16,
        jacobian_freq = 20,
        hessian_freq  = 60,
        grad_stats_freq = 5,
    )

    # ── Build model ────────────────────────────────────────────────
    print("Building AgentSmith...")
    model = AgentSmith(config)
    print(model)

    # ── Predict-only mode ──────────────────────────────────────────
    if args.predict:
        demo_prediction(model, args.predict)
        return

    # ── Load synthetic dataset ─────────────────────────────────────
    dataset = Dataset.from_synthetic().shuffle(seed=42)
    train_ds, val_ds = dataset.split(val_fraction=0.15)
    train_loader = DataLoader(train_ds, shuffle=True,  seed=0)
    val_loader   = DataLoader(val_ds,   shuffle=False, seed=1)

    print(f"\nDataset: {len(train_ds)} train  /  {len(val_ds)} val  (synthetic)")
    print(f"Class distribution: {train_loader.class_distribution()}\n")

    if args.demo:
        # ── Quick demo: no training ─────────────────────────────
        print("── Demo mode (no training) ──────────────────────────────")
        demo_prediction(model, "Quantum entanglement and superposition in qubits")
        demo_prediction(model, "The court issued a writ of habeas corpus")
        demo_prediction(model, "Portfolio diversification reduces systematic risk")
        demo_calculus(model)
        demo_gsar(model)
        demo_precision(model, config)
        return

    # ── Train ──────────────────────────────────────────────────────
    trainer = Trainer(model, config, log_dir="logs")
    history = trainer.train(
        train_loader = train_loader,
        val_loader   = val_loader,
        verbose      = True,
    )

    # ── Post-training demos ────────────────────────────────────────
    print("\n\n── Post-training inference ─────────────────────────────────")
    demo_texts = [
        "Quantum entanglement allows instantaneous correlation between particles",
        "The court ruled that the contract was void for lack of consideration",
        "The Riemann hypothesis remains unproven after more than a century",
        "Gradient descent minimises the loss function iteratively",
        "The French Revolution began with the storming of the Bastille in 1789",
        "Photosynthesis converts solar energy into chemical energy in plants",
    ]
    for text in demo_texts:
        demo_prediction(model, text)

    demo_calculus(model)
    demo_gsar(model)
    demo_precision(model, config)

    print(f"\n\n  Training complete.")
    print(f"  Best validation accuracy : {trainer.best_val_acc:.4f}")
    print(f"  Final train loss         : {history['train_loss'][-1]:.4f}")
    print(f"  Logs saved to            : {trainer.log_dir}/")


if __name__ == "__main__":
    main()
