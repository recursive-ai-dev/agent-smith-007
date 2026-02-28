# English Language Training for Agent-Smith

This module teaches agent-smith basic English at a mildly-conversational level using safetensor checkpointing for persistence.

## Features

- **Conversational English Training**: Comprehensive dataset covering basic conversations, vocabulary, and sentence structures
- **Safetensor Checkpointing**: Efficient serialization/deserialization of model parameters
- **Persistent Learning**: Model state is saved and can be resumed across sessions
- **Dual Persistence**: Both safetensor files and SQLite database storage

## Quick Start

### Basic Training

Train agent-smith on conversational English:

```bash
python train_english.py --epochs 2000 --hidden-size 128
```

### Custom Training

```bash
python train_english.py \
    --epochs 5000 \
    --hidden-size 256 \
    --learning-rate 0.01 \
    --checkpoint-every 200 \
    --checkpoint-dir my_checkpoints \
    --db-path my_english_model.db
```

### Resume Training

Resume from a previous checkpoint:

```bash
python train_english.py \
    --resume-from english_epoch_2000 \
    --epochs 3000
```

## Python API

### Training with Safetensor Checkpointing

```python
from smith import SymbolicDB, SymbolicRNN, Trainer, SafetensorCheckpoint
from smith.english_data import get_full_training_corpus

# Initialize components
db = SymbolicDB("english_model.db")
checkpoint = SafetensorCheckpoint("checkpoints")
model = SymbolicRNN(vocab_size=128, hidden_size=128, db=db)

# Get training data
training_text = get_full_training_corpus()

# Train
trainer = Trainer(model, learning_rate=0.01)
trainer.train(training_text, epochs=2000, seq_length=30)

# Save with safetensor
checkpoint.save_checkpoint(
    model.params,
    "english_trained",
    epoch=2000,
    loss=trainer.training_history[-1]['loss']
)

db.close()
```

### Loading and Using a Trained Model

```python
from smith import SymbolicDB, SymbolicRNN, SafetensorCheckpoint

# Load checkpoint
checkpoint = SafetensorCheckpoint("checkpoints")
params, metadata = checkpoint.load_checkpoint("english_trained")

# Create model and load parameters
db = SymbolicDB("english_model.db")
model = SymbolicRNN(vocab_size=128, hidden_size=128, db=db)

# Deconstruct safetensor and load into model
for key, tensor in params.items():
    model.params[key].data = tensor.data

# Generate conversational text
response = model.generate("Hello, how are you?", length=100, temperature=0.7)
print(response)

db.close()
```

## Safetensor Checkpointing

### Construction (Serialization)

The `SafetensorCheckpoint` class provides `construct_safetensor` to serialize model parameters:

```python
from smith import SafetensorCheckpoint, NanoTensor

checkpoint = SafetensorCheckpoint("checkpoints")

# Construct safetensor from parameters
params = {
    'W': NanoTensor([0.1, 0.2, 0.3]),
    'b': NanoTensor([0.0])
}

path = checkpoint.construct_safetensor(
    params,
    checkpoint_name="my_checkpoint",
    metadata={"epoch": "100", "loss": "0.5"}
)
```

### Deconstruction (Deserialization)

Use `deconstruct_safetensor` to deserialize:

```python
# Deconstruct safetensor to parameters
params, metadata = checkpoint.deconstruct_safetensor("my_checkpoint")

# Access reconstructed tensors
print(params['W'].data)  # [0.1, 0.2, 0.3]
print(metadata)  # {'epoch': '100', 'loss': '0.5', ...}
```

### Managing Checkpoints

```python
# List all checkpoints
checkpoints = checkpoint.list_checkpoints()
print(checkpoints)  # ['my_checkpoint', 'english_final', ...]

# Get latest checkpoint
latest = checkpoint.get_latest_checkpoint()
print(latest)  # 'english_final'

# Load checkpoint with metadata
params, metadata = checkpoint.load_checkpoint(latest)
```

## Training Data

The English training corpus includes:

1. **Conversational Dialogues**: Real-world conversation patterns
2. **Basic Vocabulary**: Common words and phrases
3. **Sentence Structures**: Grammar patterns and sentence construction

### Accessing Training Data

```python
from smith.english_data import (
    get_full_training_corpus,
    get_conversational_corpus,
    get_vocabulary_corpus,
    get_sentences_corpus
)

# Full corpus (all data combined)
full_text = get_full_training_corpus()

# Just conversations
conversations = get_conversational_corpus()

# Just vocabulary
vocabulary = get_vocabulary_corpus()

# Just sentences
sentences = get_sentences_corpus()
```

## Training Parameters

### Recommended Settings

For best conversational results:

- **Epochs**: 2000-5000
- **Hidden Size**: 128-256
- **Learning Rate**: 0.01
- **Sequence Length**: 30-50
- **Temperature**: 0.5-0.7 for generation

### Quick Experimentation

For fast testing:

- **Epochs**: 100-500
- **Hidden Size**: 32-64
- **Learning Rate**: 0.1
- **Sequence Length**: 20

## Examples

### Example 1: Train and Save

```python
from smith import SymbolicDB, SymbolicRNN, Trainer, SafetensorCheckpoint
from smith.english_data import get_conversational_corpus

db = SymbolicDB("english.db")
checkpoint = SafetensorCheckpoint()
model = SymbolicRNN(vocab_size=128, hidden_size=128, db=db)
trainer = Trainer(model, learning_rate=0.01)

trainer.train(get_conversational_corpus(), epochs=1000)
checkpoint.save_checkpoint(model.params, "trained", epoch=1000, loss=0.5)

db.close()
```

### Example 2: Load and Continue Training

```python
from smith import SymbolicDB, SymbolicRNN, Trainer, SafetensorCheckpoint
from smith.english_data import get_full_training_corpus

checkpoint = SafetensorCheckpoint()
params, metadata = checkpoint.load_checkpoint("trained")

db = SymbolicDB("english.db")
model = SymbolicRNN(vocab_size=128, hidden_size=128, db=db)

# Load parameters
for key, tensor in params.items():
    model.params[key].data = tensor.data

# Continue training
trainer = Trainer(model, learning_rate=0.01)
trainer.train(get_full_training_corpus(), epochs=2000)

db.close()
```

### Example 3: Generate Conversation

```python
from smith import SymbolicDB, SymbolicRNN, SafetensorCheckpoint

checkpoint = SafetensorCheckpoint()
params, _ = checkpoint.load_checkpoint("english_final")

db = SymbolicDB("english.db")
model = SymbolicRNN(vocab_size=128, hidden_size=128, db=db)

for key, tensor in params.items():
    model.params[key].data = tensor.data

# Generate responses
prompts = ["Hello", "How are you", "What is your name"]
for prompt in prompts:
    response = model.generate(prompt, length=50, temperature=0.7)
    print(f"{prompt} -> {response}")

db.close()
```

## File Structure

After training, you'll have:

```
.
├── english_model.db              # SQLite database
├── checkpoints/                  # Safetensor checkpoints
│   ├── english_epoch_200.safetensors
│   ├── english_epoch_400.safetensors
│   └── english_final.safetensors
└── train_english.py             # Training script
```

## Technical Details

### Safetensor Format

- Uses `safetensors` library for secure, efficient serialization
- Stores model parameters as numpy arrays (float32)
- Includes metadata (epoch, loss, training type, etc.)
- Platform-independent binary format

### Persistence Strategy

1. **Safetensor Files**: Efficient binary checkpoints with metadata
2. **SQLite Database**: Training history, generations, and fallback storage
3. **Dual Storage**: Both formats for flexibility and redundancy

### Tensor Construction/Deconstruction

The implementation provides explicit construction and deconstruction:

- **Construction**: `NanoTensor` → numpy array → safetensor file
- **Deconstruction**: safetensor file → numpy array → `NanoTensor`

This ensures the tensor format can be properly serialized and deserialized across sessions.

## Testing

Run the test suite:

```bash
# Test safetensor functionality and English training
python test_english_safetensor.py

# Run all tests
python test_smith.py
```

## Troubleshooting

### Model Generates Gibberish

- Increase training epochs (try 2000-5000)
- Increase hidden size (try 128-256)
- Check that checkpoint loaded correctly

### Checkpoint Not Found

- Verify checkpoint directory exists
- Check checkpoint name (without .safetensors extension)
- List available checkpoints: `checkpoint.list_checkpoints()`

### Out of Memory

- Decrease hidden size (try 64 or 32)
- Decrease batch/sequence length
- Train on smaller corpus subset

## Requirements

- Python 3.10+
- safetensors >= 0.4.0
- numpy (required for safetensors)

Install dependencies:

```bash
pip install -r requirements.txt
```

## License

See main project LICENSE file.
