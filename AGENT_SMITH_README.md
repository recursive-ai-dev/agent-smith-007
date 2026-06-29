# Agent Smith: Matrix Anomaly Detection Autoencoder

Before the rise of modern LLMs within the Matrix, Agent Smith would have started as a classic machine learning model designed to identify and eliminate bugs, glitches, and unauthorized entities (like Redpills). Fundamentally, this makes him an **Anomaly Detection Autoencoder**.

## Concept
1. **The Matrix is highly structured:** Normal Matrix code is predictable. An Autoencoder can easily learn to compress and reconstruct it with minimal error.
2. **Redpills (Anomalies) break the rules:** Entities like Neo behave outside the system's learned parameters. When Smith tries to reconstruct their data, the reconstruction error (Loss) is exceptionally high.
3. **Purge Protocol:** If the loss exceeds a certain threshold, Smith flags the data as an anomaly and initiates a purge.
4. **Replication (Cloning):** True to his nature, Smith can copy himself over existing programs. We implemented this via Safetensor checkpointing (`clone` and `load_clone`), leveraging the custom `smith` framework.

## How It Works
The script `agent_smith.py` implements a minimal viable product (MVP) of this concept using the repository's native framework (`GatedRecurrentUnit`, `NanoTensor`, `SymbolicDB`, `SafetensorCheckpoint`).

1. **Architecture:** A recurrent autoencoder extending `GatedRecurrentUnit` with an additional `W_recon` reconstruction layer.
2. **Training:** Learns to reconstruct standard Matrix traffic, driving down the anomaly score.
3. **Evaluation:** Tested against a "Normal" sequence and a highly divergent "Neo/Redpill" signal, initiating a purge when detecting the latter.

## Usage
To run Agent Smith:
```bash
python3 agent_smith.py
```

### Expected Output
```text
Initializing Agent Smith (Anomaly Detection Autoencoder)...
Training Agent Smith on normal Matrix data...
Epoch 0 Loss: 17.0840
...
Cloning Agent Smith...
Evaluating anomalies...
Normal Activity Loss: 0.0088
Anomaly (Neo) Loss:   82.4023
Anomaly detected! Initiating purge.
```
