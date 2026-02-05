from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import train_mlp, train_sklearn


def test_train_sklearn_smoke() -> None:
    train_sklearn.main()
    assert Path("models/sklearn_iris.joblib").exists()


def test_train_mlp_smoke() -> None:
    train_mlp.main()
    assert Path("models/mlp_iris.joblib").exists()
