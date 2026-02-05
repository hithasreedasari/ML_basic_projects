# Minimal ML Starter

Small, beginner-friendly ML scaffold that shows:
- a scikit-learn baseline on Iris
- a small scikit-learn MLP on Iris
- quick smoke tests
- GitHub Actions CI

## Project structure

```
README.md
requirements.txt
.gitignore
src/
  __init__.py
  utils.py
  train_sklearn.py
  train_mlp.py
tests/
  test_smoke.py
.github/
  workflows/
    ci.yml
```

## Quickstart (local)

1. Create and activate a venv:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

2. Install deps:

```bash
pip install -r requirements.txt
```

3. Run training:

```bash
python -m src.train_sklearn
python -m src.train_mlp
```

4. Run tests:

```bash
pytest -q
```

## Notes

- Models are saved under `models/` (`.joblib` files).
- This is intentionally small so it runs quickly on CPU.
