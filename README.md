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

## Repo AI assistant (OpenAI API)

1. Add your API key to environment:

```bash
# PowerShell
$env:OPENAI_API_KEY="your_api_key_here"
```

Optional model override:

```bash
$env:OPENAI_MODEL="gpt-4o-mini"
```

Optional OpenAI-compatible endpoint override (self-hosted or third-party):

```bash
$env:OPENAI_BASE_URL="https://your-endpoint.example.com/v1"
```

2. Ask a question about this repo:

```bash
python -m src.repo_assistant "How do I train the sklearn baseline?"
```

Without an inline question, it opens an interactive prompt:

```bash
python -m src.repo_assistant
```

## Public GitHub Q&A bot (`/ask`)

This repo also includes a GitHub Actions bot that answers questions in issues.

How to enable:

1. Go to `Settings -> Secrets and variables -> Actions`.
2. Create a repository secret named `OPENAI_API_KEY`.
3. Optional for non-OpenAI endpoints: add `OPENAI_BASE_URL`.
4. Optional model override: add `OPENAI_MODEL`.
5. Push this workflow file: `.github/workflows/repo-assistant.yml`.

How visitors use it:

1. Open any issue in this repo.
2. Add a comment starting with `/ask`.
3. Example:

```text
/ask How do I run the sklearn baseline training?
```

The bot will reply in the same issue thread.

Example self-hosted/OpenAI-compatible setup:

1. `OPENAI_BASE_URL`: your server URL ending with `/v1`
2. `OPENAI_API_KEY`: token expected by that server
3. `OPENAI_MODEL`: model id served by that endpoint
4. Optional `OPENAI_FALLBACK_MODEL`: backup model id
5. Optional `OPENAI_MODEL_CANDIDATES`: comma-separated backup model ids

Low-memory Ollama tip:

- If you see memory errors, use small models such as `llama3.2:1b` and set:
  - `OPENAI_MODEL=llama3.2:1b`
  - `OPENAI_FALLBACK_MODEL=tinyllama:latest`

## Notes

- Models are saved under `models/` (`.joblib` files).
- This is intentionally small so it runs quickly on CPU.
