"""Global seed utility for reproducibility across all libraries."""

import logging
import random

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """Set global random seed for reproducibility across all libraries.

    Args:
        seed: Integer seed value. Use the same value for every experiment run.
    """
    random.seed(seed)
    logger.info("Seeded: random (seed=%d)", seed)

    try:
        import numpy as np

        np.random.seed(seed)
        logger.info("Seeded: numpy (seed=%d)", seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info("Seeded: torch (seed=%d)", seed)
    except ImportError:
        pass
