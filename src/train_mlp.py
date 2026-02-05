from __future__ import annotations

import sys
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils import ensure_models_dir, get_data


def main() -> None:
    x_train, x_test, y_train, y_test = get_data()

    model = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    acc = accuracy_score(y_test, preds)
    print(f"mlp accuracy: {acc:.3f}")

    models_dir = ensure_models_dir()
    joblib.dump(model, models_dir / "mlp_iris.joblib")


if __name__ == "__main__":
    main()
