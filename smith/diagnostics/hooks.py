"""
Diagnostic Hooks
================
Utilities for tracing and logging intermediate model states.
"""

import logging
from typing import Any, List

logger = logging.getLogger(__name__)

def logging_hook(name: str, data: Any):
    """
    Standard hook to trace intermediate states without disrupting the graph.
    """
    if isinstance(data, list):
        if not data:
            summary = "empty"
        else:
            # Stats for list of floats (embedding hidden states)
            try:
                mean = sum(data) / len(data)
                summary = f"len={len(data)}, mean={mean:.4f}"
            except (TypeError, ZeroDivisionError):
                summary = f"len={len(data)}"
    else:
        summary = str(data)[:50] + "..." if len(str(data)) > 50 else str(data)

    logger.debug("TRACER [%s]: %s", name, summary)

def gradient_check_hook(name: str, grad: List[float]):
    """
    Specific hook for monitoring gradient health (exploding/vanishing).
    """
    if not grad: return
    import math
    norm = math.sqrt(sum(g*g for g in grad))
    if norm > 100.0:
        logger.warning("High gradient detected in %s: norm=%.4f", name, norm)
    elif norm < 1e-7:
        logger.debug("Near-zero gradient in %s: norm=%.4f", name, norm)
