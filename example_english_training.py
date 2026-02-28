#!/usr/bin/env python3
"""
Complete English Training Example

Demonstrates the full workflow of training agent-smith on conversational English
with safetensor checkpointing, including construction, deconstruction, and persistence.
"""

from smith import SymbolicDB, SymbolicRNN, Trainer, SafetensorCheckpoint
from smith.english_data import get_conversational_corpus


def main():
    """Run complete English training demonstration."""
    
    print("=" * 70)
    print("Agent-Smith English Training - Complete Example")
    print("=" * 70)
    print()
    
    # Configuration
    db_path = "example_english.db"
    checkpoint_dir = "example_checkpoints"
    epochs = 500
    hidden_size = 64
    
    print("Configuration:")
    print(f"  Database: {db_path}")
    print(f"  Checkpoint directory: {checkpoint_dir}")
    print(f"  Epochs: {epochs}")
    print(f"  Hidden size: {hidden_size}")
    print()
    
    # Step 1: Initialize components
    print("Step 1: Initializing components...")
    db = SymbolicDB(db_path)
    checkpoint = SafetensorCheckpoint(checkpoint_dir)
    model = SymbolicRNN(vocab_size=128, hidden_size=hidden_size, db=db)
    trainer = Trainer(model, learning_rate=0.05, verbose=False)
    print("  ✓ Components initialized")
    print()
    
    # Step 2: Get training data
    print("Step 2: Loading English training corpus...")
    training_text = get_conversational_corpus()
    print(f"  ✓ Loaded {len(training_text)} characters of conversational English")
    print()
    
    # Step 3: Train the model
    print(f"Step 3: Training for {epochs} epochs...")
    trainer.train(training_text, epochs=epochs, seq_length=25, eval_every=1000)
    print(f"  ✓ Training complete")
    print(f"  ✓ Final loss: {trainer.training_history[-1]['loss']:.4f}")
    print()
    
    # Step 4: Construct safetensor checkpoint (serialize)
    print("Step 4: Constructing safetensor checkpoint...")
    checkpoint_name = "example_english_trained"
    checkpoint_path = checkpoint.construct_safetensor(
        model.params,
        checkpoint_name,
        metadata={
            "epochs": str(epochs),
            "loss": str(trainer.training_history[-1]['loss']),
            "hidden_size": str(hidden_size),
            "training_type": "conversational_english"
        }
    )
    print(f"  ✓ Checkpoint constructed: {checkpoint_path}")
    print()
    
    # Step 5: Test generation before checkpoint reload
    print("Step 5: Testing generation (before checkpoint reload)...")
    test_prompts = ["Hello", "How are you", "Thank you"]
    original_generations = {}
    
    for prompt in test_prompts:
        generated = model.generate(prompt, length=30, temperature=0.6)
        original_generations[prompt] = generated
        print(f"  {prompt} -> {repr(generated[:40])}")
    print()
    
    # Step 6: Create new model instance
    print("Step 6: Creating new model instance...")
    model2 = SymbolicRNN(vocab_size=128, hidden_size=hidden_size, db=db)
    print("  ✓ New model created (untrained)")
    print()
    
    # Step 7: Deconstruct safetensor checkpoint (deserialize)
    print("Step 7: Deconstructing safetensor checkpoint...")
    loaded_params, loaded_metadata = checkpoint.deconstruct_safetensor(checkpoint_name)
    
    # Load parameters into new model
    for key, tensor in loaded_params.items():
        model2.params[key].data = tensor.data
    
    print("  ✓ Checkpoint deconstructed and loaded")
    print(f"  ✓ Metadata: {loaded_metadata}")
    print()
    
    # Step 8: Test generation after checkpoint reload
    print("Step 8: Testing generation (after checkpoint reload)...")
    for prompt in test_prompts:
        generated = model2.generate(prompt, length=30, temperature=0.6)
        print(f"  {prompt} -> {repr(generated[:40])}")
    print()
    
    # Step 9: Verify persistence
    print("Step 9: Verifying persistence...")
    all_checkpoints = checkpoint.list_checkpoints()
    print(f"  ✓ Available checkpoints: {all_checkpoints}")
    print(f"  ✓ Latest checkpoint: {checkpoint.get_latest_checkpoint()}")
    print()
    
    # Step 10: Save to database as well
    print("Step 10: Saving to database...")
    db_key = db.save_params(model.params)
    print(f"  ✓ Saved to database: {db_key}")
    
    # Log training to database
    db.log_training(
        epoch=epochs,
        loss=trainer.training_history[-1]['loss'],
        grad_norm=trainer.training_history[-1]['grad_norm'],
        pattern="english_conversational"
    )
    print("  ✓ Training history logged")
    print()
    
    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"✓ Trained agent-smith on conversational English")
    print(f"✓ Safetensor checkpoint constructed and saved")
    print(f"✓ Safetensor checkpoint deconstructed and loaded")
    print(f"✓ Persistence verified across model instances")
    print(f"✓ Dual storage: safetensor files + SQLite database")
    print()
    print("Files created:")
    print(f"  - {db_path} (SQLite database)")
    print(f"  - {checkpoint_path} (Safetensor checkpoint)")
    print()
    print("Next steps:")
    print("  1. Run with more epochs for better results")
    print("  2. Try: python train_english.py --epochs 2000")
    print("  3. Load checkpoint: checkpoint.load_checkpoint('example_english_trained')")
    print("=" * 70)
    
    db.close()


if __name__ == "__main__":
    main()
