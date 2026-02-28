# Smith - Algebraic Symbolism Text Generation System

A lightweight AI text generation system demonstrating advanced software engineering principles:
- **Branchless operations** using algebraic primitives
- **Structural pattern matching** for elegant control flow
- **Self-hosted database** for model persistence
- **Custom autograd** implementation (NanoTensor)
- **Safetensor checkpointing** for efficient model serialization

## 🎯 Features

- **NanoTensor**: Lightweight tensor with automatic differentiation
- **PatternMatcher**: Structural pattern matching engine for declarative logic
- **SymbolicDB**: Self-hosted SQLite database for model persistence
- **SymbolicRNN**: Character-level RNN using only branchless operations
- **Trainer**: High-level training interface with progress tracking
- **SafetensorCheckpoint**: Efficient checkpoint management with construct/deconstruct capabilities
- **English Language Training**: Pre-built conversational English training corpus and scripts
- **STIV**: Semantic Token Integrity Verification for manifold-based safety gating

## 🗣️ New: English Language Training

Train agent-smith on basic conversational English with persistent checkpointing:

```bash
# Quick start - train on conversational English
python train_english.py --epochs 2000 --hidden-size 128

# Resume from checkpoint
python train_english.py --resume-from english_epoch_2000 --epochs 3000
```

See [ENGLISH_TRAINING.md](ENGLISH_TRAINING.md) for complete documentation.

## 📦 Installation

**Requirements:**
- Python 3.10 or higher (required for structural pattern matching)

```bash
# Clone the repository
git clone https://github.com/recursive-ai-dev/agent-smith.git
cd agent-smith

# Install dependencies
pip install -r requirements.txt
```

**Dependencies:**
- `safetensors>=0.4.0` - Required for checkpoint persistence
- `numpy` - Required for safetensor support
- `scipy` - Required for STIV manifold indexing
- Other dependencies are optional for visualization

## 🚀 Quick Start

### Using the Python API

```python
from smith import SymbolicDB, SymbolicRNN, Trainer

# Initialize components
db = SymbolicDB("my_model.db")
model = SymbolicRNN(vocab_size=128, hidden_size=64, db=db)
trainer = Trainer(model, learning_rate=0.01)

# Train on text
text = "Your training text here..."
trainer.train(text, epochs=1000, seq_length=25)

# Generate text
generated = model.generate("Hello", length=100, temperature=0.7)
print(generated)

# Save model
model.db.save_params(model.params)
db.close()
```

### Using the Command Line

```bash
# Train on a text string
python -m smith.train --text "Hello world! This is a test." --epochs 500

# Train on a file
python -m smith.train --text data.txt --epochs 1000 --hidden-size 128

# Full options
python -m smith.train \
    --text training.txt \
    --epochs 1000 \
    --hidden-size 64 \
    --vocab-size 128 \
    --seq-length 25 \
    --learning-rate 0.01 \
    --db-path my_model.db
```

### Running the Example

```bash
# Run the example training script
python example_train.py
```

## 📚 Module Overview

### `smith.tensor` - NanoTensor

Lightweight tensor with automatic differentiation:

```python
from smith import NanoTensor

# Create tensors
a = NanoTensor([1.0, 2.0, 3.0])
b = NanoTensor([4.0, 5.0, 6.0])

# Operations
c = a * b  # Element-wise multiplication
d = c.sum()  # Sum all elements

# Backward pass
d.backward()
print(a.grad)  # Gradients
```

### `smith.pattern_matcher` - PatternMatcher

Structural pattern matching for declarative control flow:

```python
from smith import PatternMatcher, Token

matcher = PatternMatcher()

# Token classification
token = Token(value='A', id=65)
result = matcher.match_token_pattern(token)
# {'category': 'letter', 'is_vowel': True}

# Generation mode parsing
config = matcher.match_generation_mode("sample_0.7")
# {'strategy': 'probabilistic', 'temperature': 0.7}
```

### `smith.database` - SymbolicDB

Self-hosted database for model persistence:

```python
from smith import SymbolicDB

db = SymbolicDB("model.db")

# Save parameters
param_key = db.save_params(model.params)

# Load parameters
params = db.load_params(param_key)

# Log training
db.log_training(epoch=1, loss=0.5, grad_norm=2.3, pattern="pattern_1")

# Get history
history = db.get_training_history()

db.close()
```

### `smith.model` - SymbolicRNN

Character-level RNN with branchless operations:

```python
from smith import SymbolicRNN, SymbolicDB

db = SymbolicDB("model.db")
model = SymbolicRNN(vocab_size=128, hidden_size=64, db=db)

# Generate text
text = model.generate(seed="Hello", length=100, temperature=0.7)

# Forward pass
logits, hidden = model.forward([72, 101, 108, 108, 111])  # "Hello"
```

### `smith.trainer` - Trainer

High-level training interface:

```python
from smith import Trainer

trainer = Trainer(
    model=model,
    learning_rate=0.01,
    clip_grad=5.0,
    verbose=True
)

trainer.train(
    text="Your training text...",
    epochs=1000,
    seq_length=25,
    save_every=100,
    eval_every=50
)

# Get training history
history = trainer.get_history()
```

### `smith.stiv` - STIV

Manifold-based verifier for tokenized text:

```python
from smith import STIV, STIVConfig, Validator

config = STIVConfig(dimension=128, epsilon=0.5)
engine = STIV(config)
validator = Validator(engine)
validator.run_tests()
```

## 🎓 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TEXT GENERATION SYSTEM                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │NanoTensor    │  │PatternMatcher│  │SymbolicDB    │       │
│  │(Autograd)    │  │(SPM Engine)  │  │(Self-hosted) │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                   │                   │               │
│  ┌──────▼──────────────────▼──────────────────▼──────┐       │
│  │         SymbolicRNN (Branchless Logic)             │       │
│  └──────┬──────────────────┬──────────────────┬──────┘       │
│         │                  │                  │               │
│  ┌──────▼──────┐  ┌───────▼──────┐  ┌───────▼──────┐       │
│  │Training Loop│  │Text Generation│  │Evaluation   │       │
│  │(Symbolic)   │  │(Pattern-based)│  │(Database)   │       │
│  └─────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

## 🧪 Key Concepts

### Branchless Operations

Traditional approach:
```python
if x > 0:
    return x
else:
    return 0
```

Algebraic symbolism approach:
```python
return (x + abs(x)) / 2.0  # Branchless ReLU
```

### Pattern Matching

Traditional approach:
```python
if mode == "greedy":
    temperature = 0.0
elif mode.startswith("sample_"):
    temperature = float(mode.split("_")[1])
else:
    temperature = 0.0
```

Pattern matching approach:
```python
match mode.split("_"):
    case ["greedy"]:
        return {"temperature": 0.0}
    case ["sample", temp]:
        return {"temperature": float(temp)}
    case _:
        return {"temperature": 0.0}
```

## 🔧 Advanced Usage

### Custom Training Callback

```python
def my_callback(epoch, loss, sample):
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss={loss:.4f}")
        print(f"Sample: {sample}")

trainer.train(
    text=training_text,
    epochs=1000,
    callback=my_callback
)
```

### Loading Saved Models

```python
db = SymbolicDB("model.db")
model = SymbolicRNN(vocab_size=128, hidden_size=64, db=db)

# Load specific checkpoint
params_data = db.load_params("params_abc12345")
if params_data:
    for key, data in params_data.items():
        model.params[key].data = data
```

## 📊 Performance Tips

1. **Vocabulary Size**: Use 128 for ASCII, 256 for extended ASCII
2. **Hidden Size**: Start with 32-64 for quick experiments, 128-256 for better quality
3. **Sequence Length**: 25-50 works well for most texts
4. **Learning Rate**: 0.01-0.1 depending on data size
5. **Training Duration**: 500-1000 epochs for small texts, more for larger datasets

## 🤝 Contributing

Contributions are welcome! This project demonstrates:
- Clean modular architecture
- Type hints and documentation
- Educational code structure
- Minimal dependencies

## 📝 License

See LICENSE file for details.

## 🎯 Educational Goals

This project teaches:
1. **Algebraic Symbolism**: Transform imperative code into mathematical operations
2. **Pattern Matching**: Replace conditionals with declarative patterns
3. **Branchless Computing**: Build GPU-friendly, differentiable operations
4. **Self-Hosted Systems**: Implement custom databases and computation graphs
5. **Autograd Implementation**: Understand how PyTorch/TensorFlow work internally

## 🚀 From Notebook to Production

This project started as `smith.ipynb` - an educational Jupyter notebook. It has been modularized into a production-ready Python package with:
- ✅ Clear module separation
- ✅ Reusable components
- ✅ Command-line interface
- ✅ Comprehensive documentation
- ✅ Example scripts
- ✅ Training interface

Perfect for learning, experimentation, and building upon!
