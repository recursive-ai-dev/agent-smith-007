"""
Data Utilities
==============
Provides streamlined data loading and synthetic data generation for testing.
"""

import random
from typing import List, Tuple, Generator

class DataLoader:
    """
    Minimalist iterator for AgentSmith datasets.
    """
    def __init__(self, texts: List[str], labels: List[int], batch_size: int = 1):
        self.data = list(zip(texts, labels))
        self.batch_size = batch_size

    def __iter__(self) -> Generator[Tuple[str, int], None, None]:
        data_copy = list(self.data)
        random.shuffle(data_copy)
        for i in range(0, len(data_copy), self.batch_size):
            # Currently NanoTensor expects single-sample updates (SGD)
            yield data_copy[i]

    def __len__(self) -> int:
        return len(self.data)

def generate_dummy_data(samples: int = 100) -> Tuple[List[str], List[int]]:
    """
    Generates structured multi-domain synthetic text data.
    """
    templates = [
        ("The system requested administrative access for user {id}.", 1),
        ("Security breach detected in sector {id}.", 0),
        ("Standard maintenance completed for node {id}.", 2),
        ("Restricted file access attempted by {id}.", 3),
        ("Emergency shutdown initiated by terminal {id}.", 4)
    ]
    texts = []
    labels = []
    for _ in range(samples):
        tmpl, lbl = random.choice(templates)
        texts.append(tmpl.format(id=random.randint(100, 999)))
        labels.append(lbl)
    return texts, labels
