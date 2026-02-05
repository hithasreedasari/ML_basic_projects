from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def get_data(test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = load_iris()
    x = data.data.astype(np.float32)
    y = data.target.astype(np.int64)
    return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)


def ensure_models_dir(path: str = "models") -> Path:
    models_dir = Path(path)
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir
