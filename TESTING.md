# TESTING.md

## Testing Strategy

The showcase includes a built-in self-test mode that exercises:
- Parameter validation (including edge cases)
- Temperature range checks (values from -1 to 12)
- Text normalization for out-of-range characters
- A short end-to-end training and generation run using real repo data (README.md)

These tests avoid mock data and validate the system under both valid and invalid inputs.

## Self-Test Coverage

### Validation Edge Cases
- **Temperature validation:** values from **-1 to 12** are tested to confirm rejection outside the allowed range and acceptance inside it.
- **Positive integer validation:** values from **-1 to 12** are tested to confirm only positive values are accepted.

### Text Normalization
- Input containing extended Unicode (e.g., `Café μ-test`) validates that out-of-range characters are replaced deterministically.

### End-to-End Run
- A brief 2-epoch training run uses the real `README.md` corpus (trimmed to `max_chars=200`) with deterministic seeding.
- A sample generation is produced to confirm output path correctness.

### STIV Validation Coverage
- **Config bounds:** `dimension` and `epsilon` are tested for values from **-1 to 12** to validate error handling and acceptable ranges.
- **Manifold training:** Deterministic corpus generation is used to train the manifold, confirming centroid normalization and index build.
- **Safety verification:** Safe SQL strings are accepted and explicit injection patterns are rejected.
- **Fuzz/perf bounds:** Fuzzing and performance loops are executed with bounded iterations to ensure reproducible safety checks.

## How to Run

```bash
python smith_showcase.py --self-test
```

```bash
python test_smith.py
```

## Expected Behavior
- The script logs validation checks and completes a short training run.
- The self-test exits with a success status code (0).
- A temporary SQLite DB (`self_test_showcase.db`) is created and cleaned up automatically.

## Colab Training Script Validation Coverage

### Dataset Integrity Checks
- Validates the dataset root exists and contains at least one class folder.
- Validates that at least one supported text file is discovered in the dataset.
- Deterministic class-to-index mapping via sorted directory names.

### Training Loop and Metrics
- Confirms GPU availability selection by printing the chosen device.
- Tracks training loss per epoch and computes validation accuracy.
- Saves a checkpoint only when validation accuracy improves.

### Suggested Colab Runs (No Mock Data)
```bash
python colab_train_agent_smith.py
```

Expected behavior: the script indexes Drive data, reports detected classes, and trains with tqdm progress bars.
