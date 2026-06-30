"""
Checkpoint Serialization
========================
Handles model serialization using the industry-standard safetensors format.
Falls back to JSON for zero-dependency environments.
"""

import os
import json
import logging
from typing import Dict, List, Any

try:
    from safetensors import safe_open
    from safetensors.numpy import save_file
    import numpy as np
    _has_safetensors = True
except ImportError:
    _has_safetensors = False

logger = logging.getLogger(__name__)

class SafetensorCheckpoint:
    """
    Zero-copy serialization for model parameters.
    """
    @staticmethod
    def save(path: str, state_dict: Dict[str, List[float]]):
        """Save parameters to disk."""
        if not _has_safetensors:
            logger.warning("safetensors not installed, falling back to JSON serialization")
            with open(path + ".json", "w") as f:
                json.dump(state_dict, f)
            return

        try:
            tensors = {k: np.array(v, dtype=np.float32) for k, v in state_dict.items()}
            save_file(tensors, path)
            logger.info("Checkpoint successfully saved: %s", path)
        except Exception as e:
            logger.error("Serialization failed: %s", e)
            raise

    @staticmethod
    def load(path: str) -> Dict[str, List[float]]:
        """Load parameters from disk."""
        if not os.path.exists(path) and os.path.exists(path + ".json"):
             with open(path + ".json", "r") as f:
                return json.load(f)
        
        if not _has_safetensors:
            raise ImportError("safetensors required to load .safetensors files")

        try:
            result = {}
            with safe_open(path, framework="np") as f:
                for key in f.keys():
                    result[key] = f.get_tensor(key).tolist()
            return result
        except Exception as e:
            logger.error("Deserialization failed: %s", e)
            raise
