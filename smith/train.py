"""
Main Training Script

Command-line interface for training the SymbolicRNN model.
"""

import argparse
import sys
from pathlib import Path

from .database import SymbolicDB
from .gru_model import GatedRecurrentUnit
from .trainer import Trainer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train SymbolicRNN text generation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training arguments
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Training text (string or path to file)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="Hidden layer size"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=128,
        help="Vocabulary size (ASCII=128, extended=256)"
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=25,
        help="Training sequence length"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate"
    )
    parser.add_argument(
        "--clip-grad",
        type=float,
        default=5.0,
        help="Gradient clipping threshold"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="symbolic_ai.db",
        help="Database path for model persistence"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=50,
        help="Evaluate and print sample every N epochs"
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="Hello",
        help="Seed text for generation samples"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress training output"
    )
    
    args = parser.parse_args()
    
    # Load training text
    text_path = Path(args.text)
    if text_path.exists() and text_path.is_file():
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Loaded training text from {text_path} ({len(text)} characters)")
    else:
        text = args.text
        print(f"Using provided text ({len(text)} characters)")
    
    if len(text) < 10:
        print("Error: Training text must be at least 10 characters")
        sys.exit(1)
    
    # Initialize components
    print("\nInitializing model...")
    db = SymbolicDB(args.db_path)
    model = GatedRecurrentUnit(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        db=db
    )
    
    trainer = Trainer(
        model=model,
        learning_rate=args.learning_rate,
        clip_grad=args.clip_grad,
        verbose=not args.quiet
    )
    
    # Train
    print("\nStarting training...")
    print("=" * 60)
    
    trainer.train(
        text=text,
        epochs=args.epochs,
        seq_length=args.seq_length,
        save_every=args.save_every,
        eval_every=args.eval_every
    )
    
    # Final generation
    print("\n" + "=" * 60)
    print("Final generation samples:")
    print("-" * 60)
    
    for temp in [0.3, 0.7, 1.0]:
        sample = model.generate(args.seed, length=50, temperature=temp)
        print(f"\nTemperature {temp}:")
        print(repr(sample))
    
    # Save final model
    final_key = db.save_params(model.params)
    print(f"\n✓ Final model saved: {final_key}")
    
    db.close()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
