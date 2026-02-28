#!/usr/bin/env python3
"""
Tests for Safetensor Checkpoint and English Training

Tests the safetensor construction/deconstruction functionality
and English language training capabilities.
"""

import os
import sys
import shutil
from pathlib import Path

from smith import (
    NanoTensor,
    SafetensorCheckpoint,
    SymbolicDB,
    SymbolicRNN,
    Trainer
)
from smith.english_data import (
    get_full_training_corpus,
    get_conversational_corpus,
    get_vocabulary_corpus,
    get_sentences_corpus
)


def test_safetensor_checkpoint():
    """Test safetensor checkpoint construction and deconstruction."""
    print("Testing SafetensorCheckpoint...")
    
    test_checkpoint_dir = "test_checkpoints"
    
    try:
        # Create checkpoint manager
        checkpoint = SafetensorCheckpoint(test_checkpoint_dir)
        
        # Create test parameters
        params = {
            'W1': NanoTensor([0.1, 0.2, 0.3, 0.4]),
            'W2': NanoTensor([0.5, 0.6]),
            'b': NanoTensor([0.0, 0.1])
        }
        
        # Test construction (serialize)
        checkpoint_name = "test_checkpoint"
        metadata = {"test": "value", "epoch": "10"}
        path = checkpoint.construct_safetensor(params, checkpoint_name, metadata)
        
        assert path.exists(), "Checkpoint file not created"
        assert path.suffix == ".safetensors", "Wrong file extension"
        print("  ✓ Safetensor construction successful")
        
        # Test deconstruction (deserialize)
        loaded_params, loaded_metadata = checkpoint.deconstruct_safetensor(checkpoint_name)
        
        assert len(loaded_params) == 3, "Wrong number of parameters loaded"
        assert 'W1' in loaded_params, "W1 parameter missing"
        assert 'W2' in loaded_params, "W2 parameter missing"
        assert 'b' in loaded_params, "b parameter missing"
        
        # Check values (using approximate comparison due to float32 precision)
        def approx_equal(list1, list2, tol=1e-6):
            return all(abs(a - b) < tol for a, b in zip(list1, list2))
        
        assert approx_equal(loaded_params['W1'].data, [0.1, 0.2, 0.3, 0.4]), "W1 data mismatch"
        assert approx_equal(loaded_params['W2'].data, [0.5, 0.6]), "W2 data mismatch"
        assert approx_equal(loaded_params['b'].data, [0.0, 0.1]), "b data mismatch"
        
        print("  ✓ Safetensor deconstruction successful")
        
        # Test save_checkpoint with metadata
        params2 = {'W': NanoTensor([1.0, 2.0, 3.0])}
        path2 = checkpoint.save_checkpoint(
            params2,
            "test_checkpoint_2",
            epoch=100,
            loss=0.5,
            custom_field="test_value"
        )
        
        assert path2.exists(), "Second checkpoint not created"
        print("  ✓ Checkpoint save with metadata successful")
        
        # Test load_checkpoint
        loaded2, meta2 = checkpoint.load_checkpoint("test_checkpoint_2")
        assert 'W' in loaded2, "W parameter missing in loaded checkpoint"
        
        def approx_equal(list1, list2, tol=1e-6):
            return all(abs(a - b) < tol for a, b in zip(list1, list2))
        
        assert approx_equal(loaded2['W'].data, [1.0, 2.0, 3.0]), "W data mismatch"
        print("  ✓ Checkpoint load successful")
        
        # Test list_checkpoints
        checkpoints = checkpoint.list_checkpoints()
        assert len(checkpoints) >= 2, "Not all checkpoints listed"
        assert "test_checkpoint" in checkpoints, "test_checkpoint not in list"
        assert "test_checkpoint_2" in checkpoints, "test_checkpoint_2 not in list"
        print("  ✓ List checkpoints successful")
        
        # Test get_latest_checkpoint
        latest = checkpoint.get_latest_checkpoint()
        assert latest is not None, "Latest checkpoint not found"
        assert latest in checkpoints, "Latest checkpoint not in checkpoint list"
        print("  ✓ Get latest checkpoint successful")
        
        print("  ✓ SafetensorCheckpoint tests passed")
        
    finally:
        # Cleanup
        if os.path.exists(test_checkpoint_dir):
            shutil.rmtree(test_checkpoint_dir)


def test_english_data():
    """Test English training data availability."""
    print("Testing English training data...")
    
    # Test full corpus
    full_corpus = get_full_training_corpus()
    assert len(full_corpus) > 0, "Full corpus is empty"
    assert "digital consciousness" in full_corpus, "Conversational text missing"
    print(f"  ✓ Full corpus loaded: {len(full_corpus)} characters")

    conv_corpus = get_conversational_corpus()
    assert len(conv_corpus) > 0, "Conversational corpus is empty"
    assert "What are you?" in conv_corpus, "Basic conversation missing"
    print(f"  ✓ Conversational corpus loaded: {len(conv_corpus)} characters")

    vocab_corpus = get_vocabulary_corpus()
    assert len(vocab_corpus) > 0, "Vocabulary corpus is empty"
    assert "system" in vocab_corpus, "Basic vocabulary missing"
    print(f"  ✓ Vocabulary corpus loaded: {len(vocab_corpus)} characters")

    sentences_corpus = get_sentences_corpus()
    assert len(sentences_corpus) > 0, "Sentences corpus is empty"
    assert "The system is everything" in sentences_corpus, "Basic sentences missing"
    print(f"  ✓ Sentences corpus loaded: {len(sentences_corpus)} characters")
    
    print("  ✓ English data tests passed")


def test_english_training_short():
    """Test short English training session with safetensor checkpointing."""
    print("Testing English training with safetensors...")
    
    db_path = "test_english.db"
    checkpoint_dir = "test_english_checkpoints"
    
    try:
        # Initialize components
        db = SymbolicDB(db_path)
        checkpoint = SafetensorCheckpoint(checkpoint_dir)
        
        # Create model
        model = SymbolicRNN(vocab_size=128, hidden_size=16, db=db)
        
        # Get small training corpus
        training_text = get_conversational_corpus()[:500]  # Small subset for testing
        print(f"  Training on {len(training_text)} characters")
        
        # Create trainer
        trainer = Trainer(model, learning_rate=0.1, verbose=False)
        
        # Train for a few epochs
        trainer.train(training_text, epochs=10, seq_length=10, eval_every=100)
        
        # Save checkpoint using safetensor
        checkpoint.save_checkpoint(
            model.params,
            "test_english_checkpoint",
            epoch=10,
            loss=trainer.training_history[-1]['loss'],
            training_type="test"
        )
        
        # Verify checkpoint exists
        checkpoints = checkpoint.list_checkpoints()
        assert "test_english_checkpoint" in checkpoints, "Checkpoint not saved"
        print("  ✓ Training checkpoint saved")
        
        # Load checkpoint and verify
        loaded_params, metadata = checkpoint.load_checkpoint("test_english_checkpoint")
        assert len(loaded_params) == len(model.params), "Parameter count mismatch"
        print("  ✓ Training checkpoint loaded")
        
        # Test generation
        generated = model.generate("Hello", length=20, temperature=0.7)
        assert len(generated) > len("Hello"), "Generation failed"
        print(f"  ✓ Generated text: {repr(generated[:30])}")
        
        db.close()
        print("  ✓ English training tests passed")
        
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.remove(db_path)
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)


def test_safetensor_persistence():
    """Test safetensor persistence across sessions."""
    print("Testing safetensor persistence...")
    
    checkpoint_dir = "test_persistence_checkpoints"
    
    try:
        # Session 1: Create and save
        checkpoint1 = SafetensorCheckpoint(checkpoint_dir)
        params1 = {
            'W': NanoTensor([0.1, 0.2, 0.3]),
            'b': NanoTensor([0.5])
        }
        checkpoint1.save_checkpoint(params1, "persist_test", epoch=1, loss=0.5)
        
        # Session 2: Load in new instance
        checkpoint2 = SafetensorCheckpoint(checkpoint_dir)
        loaded_params, metadata = checkpoint2.load_checkpoint("persist_test")
        
        def approx_equal(list1, list2, tol=1e-6):
            return all(abs(a - b) < tol for a, b in zip(list1, list2))
        
        # Verify data persisted correctly
        assert approx_equal(loaded_params['W'].data, [0.1, 0.2, 0.3]), "W data not persisted"
        assert approx_equal(loaded_params['b'].data, [0.5]), "b data not persisted"
        assert 'epoch' in metadata, "Metadata not persisted"
        
        print("  ✓ Safetensor persistence verified")
        
    finally:
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)


def main():
    """Run all tests."""
    print("=" * 70)
    print("Running Safetensor and English Training Tests")
    print("=" * 70)
    
    try:
        test_safetensor_checkpoint()
        test_english_data()
        test_english_training_short()
        test_safetensor_persistence()
        
        print("\n" + "=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
        return 0
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
