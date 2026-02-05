from __future__ import annotations

import sys
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils import ensure_models_dir, get_data


def main() -> None:
    x_train, x_test, y_train, y_test = get_data()

    model = LogisticRegression(max_iter=200)
    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    acc = accuracy_score(y_test, preds)
    print(f"sklearn accuracy: {acc:.3f}")

    models_dir = ensure_models_dir()
    joblib.dump(model, models_dir / "sklearn_iris.joblib")


if __name__ == "__main__":
    main()
