"""
Safetensor Checkpoint Manager

Provides safetensor-based checkpointing for model persistence.
Implements both construction (serialization) and deconstruction (deserialization)
of tensors to/from the safetensor format.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Any
import numpy as np

try:
    from safetensors.numpy import save_file, load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warning: safetensors not available. Install with: pip install safetensors")

from .tensor import NanoTensor


class SafetensorCheckpoint:
    """
    Safetensor-based checkpoint manager for model persistence.
    Handles construction (tensor -> safetensor) and deconstruction (safetensor -> tensor).
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors is required. Install with: pip install safetensors")
    
    def construct_safetensor(
        self,
        params: Dict[str, NanoTensor],
        checkpoint_name: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> Path:
        """
        Construct safetensor from model parameters (serialize).
        
        Args:
            params: Dictionary of parameter name to NanoTensor
            checkpoint_name: Name for this checkpoint
            metadata: Optional metadata to store with checkpoint
            
        Returns:
            Path to saved checkpoint file
        """
        # Convert NanoTensors to numpy arrays for safetensor serialization
        tensor_dict = {}
        shapes = {}
        
        for name, tensor in params.items():
            # Convert to numpy array
            np_array = np.array(tensor.data, dtype=np.float32)
            tensor_dict[name] = np_array
            shapes[name] = list(tensor.shape) if hasattr(tensor, 'shape') else [len(tensor.data)]
        
        # Prepare metadata
        checkpoint_metadata = {
            "format": "safetensor",
            "version": "1.0",
            "shapes": json.dumps(shapes)
        }
        
        if metadata:
            checkpoint_metadata.update({k: str(v) for k, v in metadata.items()})
        
        # Save to safetensor file
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.safetensors"
        save_file(tensor_dict, checkpoint_path, metadata=checkpoint_metadata)
        
        return checkpoint_path
    
    def deconstruct_safetensor(
        self,
        checkpoint_name: str
    ) -> tuple[Dict[str, NanoTensor], Dict[str, Any]]:
        """
        Deconstruct safetensor to model parameters (deserialize).
        
        Args:
            checkpoint_name: Name of checkpoint to load
            
        Returns:
            Tuple of (parameters dict, metadata dict)
        """
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.safetensors"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load from safetensor file
        tensor_dict = load_file(checkpoint_path)
        
        # Load metadata
        import safetensors
        with open(checkpoint_path, 'rb') as f:
            metadata = safetensors.safe_open(checkpoint_path, framework="numpy").metadata()
        
        # Convert numpy arrays back to NanoTensors
        params = {}
        for name, np_array in tensor_dict.items():
            # Convert to list for NanoTensor
            tensor_data = np_array.flatten().tolist()
            params[name] = NanoTensor(tensor_data, requires_grad=True)
        
        return params, metadata or {}
    
    def save_checkpoint(
        self,
        params: Dict[str, NanoTensor],
        checkpoint_name: str,
        epoch: int,
        loss: float,
        **extra_metadata
    ) -> Path:
        """
        Save a complete checkpoint with metadata.
        
        Args:
            params: Model parameters
            checkpoint_name: Name for checkpoint
            epoch: Current epoch number
            loss: Current loss value
            **extra_metadata: Additional metadata to store
            
        Returns:
            Path to saved checkpoint
        """
        metadata = {
            "epoch": str(epoch),
            "loss": str(loss),
            **{k: str(v) for k, v in extra_metadata.items()}
        }
        
        return self.construct_safetensor(params, checkpoint_name, metadata)
    
    def load_checkpoint(
        self,
        checkpoint_name: str
    ) -> tuple[Dict[str, NanoTensor], Dict[str, Any]]:
        """
        Load a complete checkpoint.
        
        Args:
            checkpoint_name: Name of checkpoint to load
            
        Returns:
            Tuple of (parameters, metadata)
        """
        return self.deconstruct_safetensor(checkpoint_name)
    
    def list_checkpoints(self) -> list[str]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint names (without .safetensors extension)
        """
        checkpoints = []
        for path in self.checkpoint_dir.glob("*.safetensors"):
            checkpoints.append(path.stem)
        return sorted(checkpoints)
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get the name of the most recent checkpoint.
        
        Returns:
            Name of latest checkpoint, or None if no checkpoints exist
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        
        # Get the most recently modified checkpoint
        latest = max(
            self.checkpoint_dir.glob("*.safetensors"),
            key=lambda p: p.stat().st_mtime
        )
        return latest.stem
