"""
Symbolic Database
=================
SQLite persistence for model checkpoints and GSAR symbolic registries.
Ensures model state is recoverable across sessions.
"""

import sqlite3
import json
import logging
from typing import Optional, Any, Dict

logger = logging.getLogger(__name__)

class SymbolicDB:
    """
    SQLite-backed storage for model weights and structural metadata.
    """
    def __init__(self, db_path: str = "smith_archive.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        id TEXT PRIMARY KEY,
                        metadata TEXT,
                        weights BLOB
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS registry (
                        key TEXT PRIMARY KEY,
                        value TEXT
                    )
                """)
            logger.info("Database initialized at %s", self.db_path)
        except sqlite3.Error as e:
            logger.error("Database initialization failed: %s", e)
            raise

    def save_checkpoint(self, checkpoint_id: str, metadata: Dict[str, Any], weights: bytes):
        """Save a binary weights blob with JSON metadata."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO checkpoints (id, metadata, weights) VALUES (?, ?, ?)",
                    (checkpoint_id, json.dumps(metadata), weights)
                )
        except sqlite3.Error as e:
            logger.error("Failed to save checkpoint %s: %s", checkpoint_id, e)

    def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve checkpoint metadata and weights."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT metadata, weights FROM checkpoints WHERE id = ?", (checkpoint_id,))
                row = cursor.fetchone()
                if row:
                    return {"metadata": json.loads(row[0]), "weights": row[1]}
        except sqlite3.Error as e:
            logger.error("Failed to load checkpoint %s: %s", checkpoint_id, e)
        return None

    def close(self):
        """No-op for connect-per-call pattern, included for API completeness."""
        pass
