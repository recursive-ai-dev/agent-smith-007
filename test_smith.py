#!/usr/bin/env python3
"""
Basic tests for Smith package

Verifies core functionality of all modules.
"""

import math
import os
import sys
from smith import (
    NanoTensor,
    PatternMatcher,
    Token,
    SymbolicDB,
    GatedRecurrentUnit,
    Trainer,
    STIV,
    STIVConfig,
    Validator,
    ValidatorConfig,
)
from smith.stiv import TrafficCorpusBuilder, DomainError


def test_tensor():
    """Test NanoTensor basic operations."""
    print("Testing NanoTensor...")
    
    # Create tensors
    a = NanoTensor([1.0, 2.0])
    b = NanoTensor([3.0, 4.0])
    
    # Operations
    c = a + b
    assert c.data == [4.0, 6.0], "Addition failed"
    
    d = a * b
    assert d.data == [3.0, 8.0], "Multiplication failed"
    
    # Backward pass
    loss = d.sum()
    loss.backward()
    assert a.grad == [3.0, 4.0], "Gradient computation failed"
    
    print("  ✓ NanoTensor tests passed")


def test_pattern_matcher():
    """Test PatternMatcher functionality."""
    print("Testing PatternMatcher...")
    
    matcher = PatternMatcher()
    
    # Token matching
    token = Token(value='A', id=65)
    result = matcher.match_token_pattern(token)
    assert result['category'] == 'letter', "Token matching failed"
    assert result['is_vowel'] == True, "Vowel detection failed"
    
    # Generation mode matching
    config = matcher.match_generation_mode("sample_0.7")
    assert config['strategy'] == 'probabilistic', "Mode matching failed"
    assert config['temperature'] == 0.7, "Temperature parsing failed"
    
    print("  ✓ PatternMatcher tests passed")


def test_database():
    """Test SymbolicDB operations."""
    print("Testing SymbolicDB...")
    
    db_path = "test_temp.db"
    
    try:
        db = SymbolicDB(db_path)
        
        # Save params
        params = {
            'W': NanoTensor([0.1, 0.2, 0.3]),
            'b': NanoTensor([0.0])
        }
        key = db.save_params(params)
        assert key is not None, "Save params failed"
        
        # Load params
        loaded = db.load_params(key)
        assert loaded is not None, "Load params failed"
        assert loaded['W'] == [0.1, 0.2, 0.3], "Loaded data mismatch"
        
        # Log training
        db.log_training(1, 0.5, 1.0, "test")
        history = db.get_training_history()
        assert len(history) == 1, "Training log failed"
        
        db.close()
        print("  ✓ SymbolicDB tests passed")
        
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_model():
    """Test GatedRecurrentUnit basic functionality."""
    print("Testing GatedRecurrentUnit...")
    
    db_path = "test_model.db"
    
    try:
        db = SymbolicDB(db_path)
        model = GatedRecurrentUnit(vocab_size=128, hidden_size=16, db=db)
        
        # Forward pass
        inputs = [72, 101, 108]  # "Hel"
        logits, hidden = model.forward(inputs)
        
        assert len(logits.data) == 128, "Output size mismatch"
        assert len(hidden.data) == 16, "Hidden size mismatch"
        
        # Generate text
        text = model.generate("Hi", length=10, temperature=0.5)
        assert len(text) >= 12, "Generation length incorrect"  # "Hi" + 10 chars
        
        db.close()
        print("  ✓ SymbolicRNN tests passed")
        
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_trainer():
    """Test Trainer basic functionality."""
    print("Testing Trainer...")
    
    db_path = "test_trainer.db"
    
    try:
        db = SymbolicDB(db_path)
        model = GatedRecurrentUnit(vocab_size=128, hidden_size=8, db=db)
        trainer = Trainer(model, learning_rate=0.1, verbose=False)
        
        # Quick training
        text = "Hello world!"
        trainer.train(text, epochs=5, seq_length=5, eval_every=10)
        
        history = trainer.get_history()
        assert len(history) == 5, "Training history mismatch"
        assert 'loss' in history[0], "Loss not recorded"
        
        db.close()
        print("  ✓ Trainer tests passed")
        
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_stiv():
    """Test STIV manifold training and verification."""
    print("Testing STIV...")

    for value in range(-1, 13):
        if value <= 0:
            try:
                STIVConfig(dimension=value, epsilon=0.5)
                raise AssertionError("Invalid dimension accepted")
            except DomainError:
                pass
        else:
            STIVConfig(dimension=value, epsilon=0.5)

    for value in range(-1, 13):
        epsilon = float(value)
        if 0 < epsilon < math.sqrt(2):
            STIVConfig(dimension=16, epsilon=epsilon)
        else:
            try:
                STIVConfig(dimension=16, epsilon=epsilon)
                raise AssertionError("Invalid epsilon accepted")
            except DomainError:
                pass

    config = STIVConfig(dimension=32, epsilon=0.6)
    builder = TrafficCorpusBuilder(seed=42)
    corpus = builder.build(
        sources=["users orders logs metrics systems access security"],
        min_samples=64,
    )
    engine = STIV(config)
    engine.learn(corpus)

    safe_query = "SELECT logs FROM access WHERE logs = 1"
    safe_result = engine.verify(safe_query)
    assert safe_result["safe"] is True, f"Safe query rejected: {safe_query}"

    attack_result = engine.verify("UNION SELECT 1, @@version -- ' OR 1=1")
    assert attack_result["safe"] is False, "Attack not rejected"

    validator = Validator(
        engine=engine,
        config=ValidatorConfig(
            fuzz_iterations=200,
            fuzz_max_penetrations=50,
            perf_iterations=500,
            corpus_target=64,
        ),
        corpus_builder=builder,
    )
    assert validator.run_tests() is True, "Validator suite failed"

    print("  ✓ STIV tests passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Running Smith Package Tests")
    print("=" * 60)
    
    try:
        test_tensor()
        test_pattern_matcher()
        test_database()
        test_model()
        test_trainer()
        test_stiv()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
