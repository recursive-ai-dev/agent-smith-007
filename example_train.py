#!/usr/bin/env python3
"""
Example Training Script

Demonstrates how to use the Smith library to train a text generation model.
"""

from smith import SymbolicDB, SymbolicRNN, Trainer


def main():
    """Train a simple model on example text."""
    
    # Training data - simple repeating pattern for quick demo
    training_text = """Hello world! This is a test. Hello world! This is a test. 
    The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.
    Machine learning is fun! Machine learning is fun!
    Python is great for AI. Python is great for AI."""
    
    print("Smith - Algebraic Symbolism Text Generation")
    print("=" * 60)
    
    # Initialize database
    db = SymbolicDB("example_model.db")
    print("✓ Database initialized")
    
    # Create model
    vocab_size = 128  # ASCII characters
    hidden_size = 32   # Small for quick demo
    model = SymbolicRNN(vocab_size=vocab_size, hidden_size=hidden_size, db=db)
    print(f"✓ Model created (vocab={vocab_size}, hidden={hidden_size})")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        learning_rate=0.1,
        clip_grad=5.0,
        verbose=True
    )
    print("✓ Trainer initialized")
    
    print("\nStarting training...")
    print("-" * 60)
    
    # Train the model
    trainer.train(
        text=training_text,
        epochs=200,
        seq_length=20,
        save_every=100,
        eval_every=50
    )
    
    print("\n" + "=" * 60)
    print("Generating samples with different temperatures...")
    print("-" * 60)
    
    # Generate samples
    seed = "Hello"
    for temperature in [0.3, 0.7, 1.0]:
        generated = model.generate(seed, length=50, temperature=temperature)
        print(f"\nTemperature {temperature}:")
        print(f"  {repr(generated)}")
    
    # Save final model
    final_key = db.save_params(model.params)
    print(f"\n✓ Final model saved with key: {final_key}")
    
    # Show training history
    history = trainer.get_history()
    if history:
        print(f"\n✓ Training history: {len(history)} epochs")
        print(f"  Initial loss: {history[0]['loss']:.4f}")
        print(f"  Final loss: {history[-1]['loss']:.4f}")
    
    db.close()
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
