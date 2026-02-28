"""
Symbolic Database - Self-Hosted Model Persistence

SQLite-based database for storing model states, training history,
and generated text with symbolic representations.
"""

import json
import sqlite3
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

from .tensor import NanoTensor


class SymbolicDB:
    """
    Self-hosted database using SQLite with algebraic operations.
    Stores model states, training history, and generated text symbolically.
    """
    
    def __init__(self, db_path: str = "symbolic_ai.db"):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self._init_schema()
    
    def _init_schema(self):
        """Initialize database schema with symbolic representations"""
        cursor = self.conn.cursor()
        
        # Store model parameters as symbolic expressions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_params (
                id INTEGER PRIMARY KEY,
                param_key TEXT UNIQUE,
                data BLOB,
                checksum TEXT,
                timestamp REAL
            )
        """)
        
        # Store training history with pattern metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_log (
                id INTEGER PRIMARY KEY,
                epoch INTEGER,
                loss REAL,
                grad_norm REAL,
                pattern_signature TEXT,
                timestamp REAL
            )
        """)
        
        # Store generated text samples
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS generations (
                id INTEGER PRIMARY KEY,
                seed_text TEXT,
                generated_text TEXT,
                config JSON,
                quality_score REAL,
                timestamp REAL
            )
        """)
        
        self.conn.commit()
    
    def save_params(self, params: Dict[str, NanoTensor]) -> str:
        """Save model parameters with algebraic checksum"""
        cursor = self.conn.cursor()
        param_key = f"params_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
        
        # Serialize tensor data
        serialized = json.dumps({k: v.data for k, v in params.items()}).encode()
        checksum = hashlib.md5(serialized).hexdigest()
        
        cursor.execute(
            "INSERT INTO model_params (param_key, data, checksum, timestamp) VALUES (?, ?, ?, ?)",
            (param_key, serialized, checksum, time.time())
        )
        self.conn.commit()
        return param_key
    
    def load_params(self, param_key: str) -> Optional[Dict[str, List[float]]]:
        """Load model parameters by key"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT data FROM model_params WHERE param_key = ?", (param_key,))
        result = cursor.fetchone()
        if result:
            return json.loads(result[0].decode())
        return None
    
    def log_training(self, epoch: int, loss: float, grad_norm: float, pattern: str):
        """Log training metrics with pattern metadata"""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO training_log (epoch, loss, grad_norm, pattern_signature, timestamp) VALUES (?, ?, ?, ?, ?)",
            (epoch, loss, grad_norm, pattern, time.time())
        )
        self.conn.commit()
    
    def store_generation(self, seed: str, text: str, config: Dict[str, Any], quality: float):
        """Store generated text with configuration"""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO generations (seed_text, generated_text, config, quality_score, timestamp) VALUES (?, ?, ?, ?, ?)",
            (seed, text, json.dumps(config), quality, time.time())
        )
        self.conn.commit()
    
    def get_best_generation(self) -> Optional[Dict[str, Any]]:
        """Retrieve highest quality generation using symbolic query"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT seed_text, generated_text, config, quality_score FROM generations "
            "ORDER BY quality_score DESC LIMIT 1"
        )
        result = cursor.fetchone()
        if result:
            return {
                "seed": result[0],
                "text": result[1],
                "config": json.loads(result[2]),
                "quality": result[3]
            }
        return None
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Retrieve training history for analysis"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT epoch, loss, grad_norm, pattern_signature, timestamp FROM training_log ORDER BY epoch")
        results = cursor.fetchall()
        return [{
            "epoch": r[0],
            "loss": r[1],
            "grad_norm": r[2],
            "pattern": r[3],
            "timestamp": r[4]
        } for r in results]
    
    def close(self):
        """Close database connection"""
        self.conn.close()
