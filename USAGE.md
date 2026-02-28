# Smith Usage Guide

This guide demonstrates how to use the Smith package for training text generation models.

## Quick Start Examples

### 1. Using the Python API (Recommended)

```python
from smith import SymbolicDB, SymbolicRNN, Trainer

# Initialize database
db = SymbolicDB("my_model.db")

# Create model
model = SymbolicRNN(
    vocab_size=128,    # ASCII character set
    hidden_size=64,    # Hidden layer size
    db=db
)

# Create trainer
trainer = Trainer(
    model=model,
    learning_rate=0.01,
    clip_grad=5.0,
    verbose=True
)

# Prepare training text
with open("training_data.txt", "r") as f:
    text = f.read()

# Train the model
trainer.train(
    text=text,
    epochs=1000,
    seq_length=25,
    save_every=100,
    eval_every=50
)

# Generate text
generated = model.generate("Hello", length=100, temperature=0.7)
print(generated)

# Clean up
db.close()
```

### 2. Using the Command Line Interface

```bash
# Basic usage
python -m smith.train --text "Your training text here" --epochs 500

# Train on a file
python -m smith.train --text training_data.txt --epochs 1000

# With custom parameters
python -m smith.train \
    --text data.txt \
    --epochs 2000 \
    --hidden-size 128 \
    --learning-rate 0.01 \
    --seq-length 30 \
    --db-path my_model.db \
    --eval-every 100
```

### 3. Using the Example Script

```bash
python example_train.py
```

## Module-by-Module Usage

### NanoTensor - Automatic Differentiation

```python
from smith import NanoTensor

# Create tensors
a = NanoTensor([1.0, 2.0, 3.0])
b = NanoTensor([4.0, 5.0, 6.0])

# Operations
c = a + b              # [5.0, 7.0, 9.0]
d = a * b              # [4.0, 10.0, 18.0]
e = d.sum()            # [32.0]

# Backward pass
e.backward()

# Check gradients
print(a.grad)  # [4.0, 5.0, 6.0]
print(b.grad)  # [1.0, 2.0, 3.0]

# Matrix operations
W = NanoTensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])  # 2x3 matrix (row-major)
x = NanoTensor([1.0, 2.0, 3.0])
y = W.matmul(x)  # Matrix-vector product
```

### PatternMatcher - Declarative Control Flow

```python
from smith import PatternMatcher, Token

matcher = PatternMatcher()

# Token classification
token = Token(value='A', id=65, type="char")
result = matcher.match_token_pattern(token)
# {'category': 'letter', 'is_vowel': True}

# Generation mode parsing
config = matcher.match_generation_mode("sample_0.7")
# {'strategy': 'probabilistic', 'temperature': 0.7}

config = matcher.match_generation_mode("topk_5_1.0")
# {'strategy': 'topk', 'k': 5, 'temperature': 1.0}

# Model action decisions
state = {"loss": 0.05, "epoch": 100}
action = matcher.match_model_action(state)
# 'save_and_evaluate'
```

### SymbolicDB - Model Persistence

```python
from smith import SymbolicDB, NanoTensor

db = SymbolicDB("model.db")

# Save parameters
params = {
    'W1': NanoTensor([0.1, 0.2, 0.3]),
    'b1': NanoTensor([0.0])
}
key = db.save_params(params)
print(f"Saved with key: {key}")

# Load parameters
loaded = db.load_params(key)
print(loaded)

# Log training metrics
db.log_training(
    epoch=1,
    loss=0.5,
    grad_norm=2.3,
    pattern="training_step"
)

# Get training history
history = db.get_training_history()
for entry in history:
    print(f"Epoch {entry['epoch']}: loss={entry['loss']}")

# Store generated text
db.store_generation(
    seed="Hello",
    text="Hello world!",
    config={"temperature": 0.7},
    quality=0.85
)

# Get best generation
best = db.get_best_generation()
print(best)

db.close()
```

### SymbolicRNN - Text Generation Model

```python
from smith import SymbolicRNN, SymbolicDB

db = SymbolicDB("model.db")
model = SymbolicRNN(
    vocab_size=128,
    hidden_size=64,
    db=db
)

# Forward pass
token_ids = [72, 101, 108, 108, 111]  # "Hello"
logits, hidden = model.forward(token_ids)

# Generate text
text = model.generate(
    seed="Hello",
    length=50,
    temperature=0.7
)
print(text)

# Sample from logits
next_token = model.sample(logits, temperature=0.5)

db.close()
```

### Trainer - Training Interface

```python
from smith import Trainer, SymbolicRNN, SymbolicDB

db = SymbolicDB("model.db")
model = SymbolicRNN(vocab_size=128, hidden_size=64, db=db)

trainer = Trainer(
    model=model,
    learning_rate=0.01,
    clip_grad=5.0,
    verbose=True
)

# Custom callback for monitoring
def my_callback(epoch, loss, sample):
    if epoch % 100 == 0:
        print(f"Custom: Epoch {epoch}, Loss: {loss:.4f}")
        print(f"Sample: {sample}")

# Train with callback
trainer.train(
    text="Your training text...",
    epochs=1000,
    seq_length=25,
    save_every=100,
    eval_every=50,
    callback=my_callback
)

# Get training history
history = trainer.get_history()
print(f"Trained for {len(history)} epochs")
print(f"Final loss: {history[-1]['loss']:.4f}")

db.close()
```

### STIV - Semantic Token Integrity Verification

```python
from smith import STIV, STIVConfig, Validator

config = STIVConfig(dimension=128, epsilon=0.5)
engine = STIV(config)
validator = Validator(engine)
validator.run_tests()
```

## Advanced Usage

### Custom Temperature Sampling

```python
# Lower temperature = more focused, repetitive
text = model.generate("Hello", length=50, temperature=0.3)

# Medium temperature = balanced
text = model.generate("Hello", length=50, temperature=0.7)

# Higher temperature = more random, creative
text = model.generate("Hello", length=50, temperature=1.2)
```

### Loading Saved Models

```python
from smith import SymbolicDB, SymbolicRNN, NanoTensor

db = SymbolicDB("model.db")
model = SymbolicRNN(vocab_size=128, hidden_size=64, db=db)

# Load specific checkpoint
params_data = db.load_params("params_abc12345")
if params_data:
    for key, data in params_data.items():
        model.params[key].data = data
    print("Model loaded successfully")

# Generate with loaded model
text = model.generate("Hello", length=100, temperature=0.7)
print(text)

db.close()
```

### Incremental Training

```python
# Initial training
trainer.train(text1, epochs=500)

# Continue training on new data
trainer.train(text2, epochs=500)

# Get complete history
full_history = trainer.get_history()
```

## Tips and Best Practices

### Vocabulary Size
- **128**: Use for ASCII text (English)
- **256**: Use for extended ASCII (other languages)

### Hidden Size
- **16-32**: Quick experiments, low memory
- **64-128**: Good balance of quality and speed
- **256-512**: High quality, slower training

### Learning Rate
- **0.001-0.01**: Conservative, stable training
- **0.01-0.1**: Faster convergence
- **0.1+**: Aggressive, may be unstable

### Sequence Length
- **10-20**: Short-term dependencies
- **25-50**: Balanced (recommended)
- **50-100**: Long-term dependencies, slower

### Training Duration
- **100-500**: Quick test/demo
- **1000-2000**: Small datasets
- **5000+**: Larger datasets, better quality

## Troubleshooting

### High Loss / Poor Generation
- Increase `hidden_size`
- Increase `epochs`
- Decrease `learning_rate`
- Increase `seq_length`

### Slow Training
- Decrease `hidden_size`
- Decrease `seq_length`
- Use shorter training text

### Out of Memory
- Decrease `hidden_size`
- Decrease `vocab_size`
- Train on smaller text chunks

## Examples

See the provided example files:
- `example_train.py`: Complete training example
- `test_smith.py`: Unit tests showing all features
- `smith.ipynb`: Original notebook with detailed explanations
