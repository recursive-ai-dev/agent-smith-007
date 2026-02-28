#!/usr/bin/env python3
"""
English Language Training Script

Teaches agent-smith basic English at a mildly-conversational level
with safetensor checkpointing for persistence.
"""

import argparse
from pathlib import Path

from smith import SymbolicDB, SymbolicRNN, Trainer, SafetensorCheckpoint
from smith.english_data import get_full_training_corpus, get_conversational_corpus


def train_english(
    epochs: int = 2000,
    hidden_size: int = 128,
    learning_rate: float = 0.01,
    checkpoint_every: int = 200,
    db_path: str = "english_model.db",
    checkpoint_dir: str = "checkpoints",
    use_safetensors: bool = True,
    resume_from: str = None
):
    """
    Train agent-smith on basic conversational English.
    
    Args:
        epochs: Number of training epochs
        hidden_size: Size of hidden layer
        learning_rate: Learning rate for training
        checkpoint_every: Save checkpoint every N epochs
        db_path: Path to database file
        checkpoint_dir: Directory for safetensor checkpoints
        use_safetensors: Whether to use safetensor checkpointing
        resume_from: Checkpoint name to resume from
    """
    print("=" * 70)
    print("Agent-Smith English Language Training")
    print("=" * 70)
    print("Teaching basic conversational English with persistence...")
    print()
    
    # Get training data
    training_text = get_full_training_corpus()
    
    print(f"Training corpus size: {len(training_text)} characters")
    print(f"Vocabulary size: 128 (ASCII)")
    print(f"Hidden size: {hidden_size}")
    print(f"Epochs: {epochs}")
    print(f"Safetensor checkpointing: {'Enabled' if use_safetensors else 'Disabled'}")
    print()
    
    # Initialize database
    db = SymbolicDB(db_path)
    print(f"✓ Database initialized: {db_path}")
    
    # Initialize safetensor checkpoint manager
    checkpoint_manager = None
    if use_safetensors:
        checkpoint_manager = SafetensorCheckpoint(checkpoint_dir)
        print(f"✓ Safetensor checkpoint manager initialized: {checkpoint_dir}")
    
    # Create or load model
    vocab_size = 128
    model = SymbolicRNN(vocab_size=vocab_size, hidden_size=hidden_size, db=db)
    
    # Resume from checkpoint if specified
    if resume_from and checkpoint_manager:
        try:
            print(f"\nLoading checkpoint: {resume_from}")
            params, metadata = checkpoint_manager.load_checkpoint(resume_from)
            
            # Deconstruct safetensor and load into model
            for key, tensor in params.items():
                if key in model.params:
                    model.params[key].data = tensor.data
            
            print(f"✓ Checkpoint loaded successfully")
            print(f"  Metadata: {metadata}")
        except Exception as e:
            print(f"⚠ Could not load checkpoint: {e}")
            print("  Starting from scratch...")
    
    print(f"✓ Model created (vocab={vocab_size}, hidden={hidden_size})")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        learning_rate=learning_rate,
        clip_grad=5.0,
        verbose=True
    )
    print("✓ Trainer initialized")
    
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70 + "\n")
    
    # Custom callback for safetensor checkpointing
    def checkpoint_callback(epoch, loss, sample):
        if use_safetensors and epoch % checkpoint_every == 0:
            checkpoint_name = f"english_epoch_{epoch}"
            try:
                # Construct safetensor from model parameters
                checkpoint_path = checkpoint_manager.save_checkpoint(
                    model.params,
                    checkpoint_name,
                    epoch=epoch,
                    loss=loss,
                    training_type="english_conversational",
                    corpus_size=len(training_text)
                )
                print(f"✓ Safetensor checkpoint saved: {checkpoint_name}")
                print(f"  Path: {checkpoint_path}")
            except Exception as e:
                print(f"⚠ Error saving checkpoint: {e}")
    
    # Train the model
    trainer.train(
        text=training_text,
        epochs=epochs,
        seq_length=30,
        save_every=checkpoint_every,  # Still save to DB
        eval_every=100,
        callback=checkpoint_callback
    )
    
    print("\n" + "=" * 70)
    print("Training complete! Testing conversational ability...")
    print("=" * 70 + "\n")
    
    # Test generation with various prompts
    test_prompts = [
        "Hello",
        "How are you",
        "What is your name",
        "Thank you",
        "I am",
    ]
    
    print("Generating conversational responses:")
    print("-" * 70)
    
    for prompt in test_prompts:
        for temp in [0.5, 0.7]:
            generated = model.generate(prompt, length=50, temperature=temp)
            print(f"\nPrompt (T={temp}): {repr(prompt)}")
            print(f"Response: {repr(generated)}")
    
    # Save final checkpoint
    if use_safetensors:
        final_checkpoint = "english_final"
        final_path = checkpoint_manager.save_checkpoint(
            model.params,
            final_checkpoint,
            epoch=epochs,
            loss=trainer.training_history[-1]['loss'],
            training_type="english_conversational",
            corpus_size=len(training_text),
            status="completed"
        )
        print(f"\n✓ Final safetensor checkpoint saved: {final_checkpoint}")
        print(f"  Path: {final_path}")
        
        # List all checkpoints
        print(f"\nAvailable checkpoints:")
        for cp_name in checkpoint_manager.list_checkpoints():
            print(f"  - {cp_name}")
    
    # Save to database as well
    final_db_key = db.save_params(model.params)
    print(f"\n✓ Model also saved to database: {final_db_key}")
    
    # Show training history
    history = trainer.get_history()
    if history:
        print(f"\n✓ Training summary:")
        print(f"  Total epochs: {len(history)}")
        print(f"  Initial loss: {history[0]['loss']:.4f}")
        print(f"  Final loss: {history[-1]['loss']:.4f}")
        print(f"  Improvement: {(history[0]['loss'] - history[-1]['loss']):.4f}")
    
    db.close()
    print("\n✓ Training session complete!")
    print("=" * 70)


def main():
    """Main entry point for English training script."""
    parser = argparse.ArgumentParser(
        description="Train agent-smith on basic conversational English"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=2000,
        help="Number of training epochs (default: 2000)"
    )
    
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="Hidden layer size (default: 128)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01)"
    )
    
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=200,
        help="Save checkpoint every N epochs (default: 200)"
    )
    
    parser.add_argument(
        "--db-path",
        type=str,
        default="english_model.db",
        help="Database path (default: english_model.db)"
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Checkpoint directory (default: checkpoints)"
    )
    
    parser.add_argument(
        "--no-safetensors",
        action="store_true",
        help="Disable safetensor checkpointing"
    )
    
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Resume training from specified checkpoint"
    )
    
    args = parser.parse_args()
    
    train_english(
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        checkpoint_every=args.checkpoint_every,
        db_path=args.db_path,
        checkpoint_dir=args.checkpoint_dir,
        use_safetensors=not args.no_safetensors,
        resume_from=args.resume_from
    )


if __name__ == "__main__":
    main()
