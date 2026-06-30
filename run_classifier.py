"""
AgentSmith Pipeline Orchestrator
================================
Main entry point for auditing and running the classifier.
"""

import logging
import sys
from smith.classifier.model import AgentSmith
from smith.classifier.config import AgentSmithConfig, DOMAINS
from smith.classifier.adam import AdamOptimizer
from smith.pipeline.data import DataLoader, generate_dummy_data
from smith.pipeline.trainer import PipelineTrainer
from smith import reset_global_state

# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("run_classifier")

def main():
    # 0. Reset global state for determinism
    reset_global_state()
    logger.info("Environment initialized.")

    # 1. Config & Model
    config = AgentSmithConfig(
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=128,
        num_classes=len(DOMAINS),
        max_seq_len=32,
        gsar_window_sizes=[2, 3]
    )
    model = AgentSmith(config)
    logger.info("Model built: %s", model)

    # 2. Data Preparation
    texts, labels = generate_dummy_data(samples=100)
    # Split 80/20
    split = 80
    train_dl = DataLoader(texts[:split], labels[:split])
    val_dl = DataLoader(texts[split:], labels[split:])
    logger.info("Data loaded: %d train, %d val samples", len(train_dl), len(val_dl))

    # 3. GSAR Pattern Mining (Warmup)
    logger.info("Mining patterns for GSAR...")
    tokenized_train = [model.tokenizer.encode(t) for t in texts[:split]]
    model.gsar.update_patterns(tokenized_train)
    logger.info("GSAR active with %d symbols.", len(model.gsar._registry))

    # 4. Optimization Setup
    optimizer = AdamOptimizer(
        model.parameters(),
        lr=2e-4,
        weight_decay=0.01,
        warmup_steps=10
    )

    # 5. Training
    trainer = PipelineTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=train_dl,
        val_dataloader=val_dl,
        amp=True
    )

    logger.info("Starting training loop...")
    trainer.fit(epochs=5)

    # 6. Inference / Audit Verification
    test_text = "The user is requesting an administrative access token for system logs."
    logger.info("Testing inference on: '%s'", test_text)
    result = model.predict(test_text)

    logger.info("Inference Result:")
    logger.info("  Predicted Label: %s", result["label"])
    logger.info("  Confidence:      %.4f", result["confidence"])
    logger.info("\n%s", result["explanation"])

    logger.info("Pipeline Execution Success.")

if __name__ == "__main__":
    main()
