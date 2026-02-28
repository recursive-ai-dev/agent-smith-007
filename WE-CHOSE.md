# WE-CHOSE.md

## Three-Perspective Planning Summary

### 1) CEO Perspective (Strategic Value & Reliability)
**Focus:** Provide a stable, production-ready entry point that can be shipped, audited, and repeated.

**Key priorities:**
- Deterministic configuration and execution
- Operational transparency through structured logging
- Minimal operational risk (validated parameters, gradient checks)

**Outcome of this perspective:**
The implementation added a full configuration layer (CLI + JSON), consistent logging, and deterministic seeding. This makes the system viable for repeatable demos and integrations.

### 2) Junior Developer Perspective (Maintainability & Clarity)
**Focus:** Improve readability and reduce cognitive load while keeping the script single-file and executable.

**Key priorities:**
- Clear separation of concerns (config, validation, training, generation)
- Reusable helpers for validation and normalization
- Safe defaults without hidden behavior

**Outcome of this perspective:**
The script now has a clear, modular structure with isolated functions for each responsibility. The execution flow is readable and easy to extend without rewriting core logic.

### 3) End Customer Perspective (Usability & Trust)
**Focus:** Provide a script that runs successfully out of the box and explains what it is doing.

**Key priorities:**
- Simple CLI with explicit options
- Sensible defaults that run without extra setup
- Verification that training is correct (gradient integrity)

**Outcome of this perspective:**
Users can run the showcase immediately, observe progress via logs, and trust that the system verifies autograd correctness before training.

## Final Decision: Combined Approach

The final implementation blends all three perspectives:
- **CEO:** deterministic configuration and logging for production-grade reliability.
- **Junior Dev:** structured, modular functions for maintainability.
- **End Customer:** a straightforward CLI with sensible defaults and built-in correctness checks.

This combination provides a high-quality, production-ready script that is both trustworthy and easy to operate.

## STIV Perspective Mapping

### CEO Perspective (Operational Safety & Auditability)
**Why this path:** The STIV module needed deterministic validation and measurable safety boundaries.

**Decision:** Add `ValidatorConfig` with explicit limits and provide structured logging so the manifold verification is auditable and reproducible.

### Junior Developer Perspective (Readable Integration)
**Why this path:** The package needed a clean, discoverable integration point with minimal cognitive overhead.

**Decision:** Expose STIV classes through `smith/__init__.py` and keep domain validation localized in dataclasses for clarity.

### End Customer Perspective (Ready-to-Run Experience)
**Why this path:** Users require an immediately runnable validation flow for training readiness.

**Decision:** Provide a default `main()` entry in `smith/stiv.py` and a deterministic corpus generator that removes reliance on hard-coded mock samples.

## Colab Training Script Perspective Mapping

### CEO Perspective (Reliable Delivery in Colab)
**Why this path:** A training workflow that depends on Drive and GPU selection must be deterministic and auditable.

**Decision:** Centralized configuration in a dataclass, mandated Drive mount instructions, and explicit device selection to guarantee predictable runs.

### Junior Developer Perspective (Maintainable Data Pipeline)
**Why this path:** A custom dataset with mixed file types needs clear boundaries and obvious extension points.

**Decision:** Implemented a dedicated `TextFolderDataset` with extension filtering, deterministic class mapping, and explicit validation errors.

### End Customer Perspective (Usable and Trustworthy Training)
**Why this path:** Users need clear feedback on class discovery, training progress, and best checkpoints.

**Decision:** Added tqdm progress bars, per-epoch loss/accuracy reporting, and best-model checkpointing to Drive.
