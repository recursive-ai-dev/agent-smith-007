import math
import random
from typing import List, Tuple
from smith.tensor import NanoTensor
from smith.database import SymbolicDB
from smith.gru_model import GatedRecurrentUnit
from smith.checkpoint import SafetensorCheckpoint

# Anomaly Detection Autoencoder version of Agent Smith, using the custom framework.
class AgentSmithAutoencoder(GatedRecurrentUnit):
    def __init__(self, vocab_size: int, hidden_size: int, db: SymbolicDB):
        super().__init__(vocab_size, hidden_size, db)

        # Additional weights for reconstruction (Decoder part)
        scale = 0.1
        self.params['W_recon'] = NanoTensor([random.uniform(-scale, scale) for _ in range(hidden_size * vocab_size)])
        self.params['b_recon'] = NanoTensor([0.0] * vocab_size)

    def reconstruct(self, inputs: List[int]) -> Tuple[NanoTensor, NanoTensor]:
        # Process the input to get the final hidden state (Encoding)
        _, final_h = super().forward(inputs)

        # Decode: attempt to reconstruct the input sequence (for simplicity, we'll just reconstruct a representation)
        reconstruction_logits = self.params['W_recon'].matmul(final_h) + self.params['b_recon']
        return reconstruction_logits, final_h

    def compute_anomaly_score(self, target_embedding: NanoTensor, reconstruction_logits: NanoTensor) -> NanoTensor:
        # Simple MSE-like anomaly score based on logits and target embedding
        diff = target_embedding - reconstruction_logits
        squared = diff * diff
        return squared.sum()

    def clone(self, checkpoint_path="smith_clone"):
        checkpoint = SafetensorCheckpoint()
        # Save the params using safetensors
        return checkpoint.save_checkpoint(self.params, checkpoint_path, epoch=100, loss=0.0)

    def load_clone(self, checkpoint_path="smith_clone"):
        checkpoint = SafetensorCheckpoint()
        params, _metadata = checkpoint.load_checkpoint(checkpoint_path)

        expected_keys = set(self.params)
        loaded_keys = set(params)
        if loaded_keys != expected_keys:
            raise ValueError(f"Checkpoint parameter mismatch: expected {expected_keys}, got {loaded_keys}")

        for key, current in self.params.items():
            loaded = params[key]
            if len(loaded.data) != len(current.data):
                raise ValueError(
                    f"Checkpoint tensor size mismatch for {key}: expected {len(current.data)}, got {len(loaded.data)}"
                )
            current.data = loaded.data[:]

def train_smith():
    print("Initializing Agent Smith (Anomaly Detection Autoencoder)...")
    db = SymbolicDB(":memory:")
    smith = AgentSmithAutoencoder(vocab_size=128, hidden_size=32, db=db)

    # "Normal Matrix Data"
    normal_sequence = [ord(c) for c in "Normal routine matrix code 1011001"]
    target_emb = NanoTensor([float(c) / 128.0 for c in normal_sequence[:32]] + [0.0]*(128-len(normal_sequence[:32])), requires_grad=False)

    # Training Loop
    print("Training Agent Smith on normal Matrix data...")
    learning_rate = 0.01

    for epoch in range(100):
        smith.zero_grad()
        recon_logits, _ = smith.reconstruct(normal_sequence)

        loss = smith.compute_anomaly_score(target_emb, recon_logits)
        loss.backward()

        # Simple SGD update
        for key, param in smith.params.items():
            if param.requires_grad:
                for i in range(len(param.data)):
                    param.data[i] -= learning_rate * param.grad[i]

        if epoch % 20 == 0:
            print(f"Epoch {epoch} Loss: {loss.data[0]:.4f}")

    print("Cloning Agent Smith...")
    smith.clone("smith_v1")

    print("Evaluating anomalies...")
    # New Smith instance loading from clone
    new_smith = AgentSmithAutoencoder(vocab_size=128, hidden_size=32, db=db)
    new_smith.load_clone("smith_v1")

    # The anomaly (Redpill / Neo)
    neo_sequence = [ord(c) for c in "THERE IS NO SPOON 9999999"]
    neo_emb = NanoTensor([0.9] * 128, requires_grad=False) # highly divergent

    normal_recon, _ = new_smith.reconstruct(normal_sequence)
    neo_recon, _ = new_smith.reconstruct(neo_sequence)

    normal_loss = new_smith.compute_anomaly_score(target_emb, normal_recon)
    neo_loss = new_smith.compute_anomaly_score(neo_emb, neo_recon)

    print(f"Normal Activity Loss: {normal_loss.data[0]:.4f}")
    print(f"Anomaly (Neo) Loss:   {neo_loss.data[0]:.4f}")

    if neo_loss.data[0] > normal_loss.data[0] * 2:
        print("Anomaly detected! Initiating purge.")
    else:
        print("Matrix is stable.")

if __name__ == "__main__":
    train_smith()
