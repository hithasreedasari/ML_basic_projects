from __future__ import annotations

from pathlib import Path

from src import train_mlp, train_sklearn


def test_train_sklearn_smoke() -> None:
    train_sklearn.main()
    assert Path("models/sklearn_iris.joblib").exists()


def test_train_mlp_smoke() -> None:
    train_mlp.main()
    assert Path("models/mlp_iris.joblib").exists()
