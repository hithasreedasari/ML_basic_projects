import argparse
import os
from pathlib import Path

DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".idea",
    ".venv",
    "__pycache__",
    ".pytest_cache",
}
DEFAULT_EXCLUDE_SUFFIXES = {
    ".pyc",
    ".joblib",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
}


def build_repo_context(repo_root: Path, max_chars: int = 120_000) -> str:
    sections = []
    total_chars = 0

    for path in sorted(repo_root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in DEFAULT_EXCLUDE_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in DEFAULT_EXCLUDE_SUFFIXES:
            continue
        if path.name.startswith(".") and path.name not in {".gitignore"}:
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        except OSError:
            continue

        rel_path = path.relative_to(repo_root)
        section = f"\n\n### FILE: {rel_path}\n{text}"
        if total_chars + len(section) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 0:
                sections.append(section[:remaining])
            break
        sections.append(section)
        total_chars += len(section)

    return "".join(sections).strip()


def ask_repo_assistant(question: str, repo_context: str, model: str) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: openai. Install with `pip install -r requirements.txt`."
        ) from exc

    base_url = os.getenv("OPENAI_BASE_URL")
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=base_url if base_url else None,
    )
    system_prompt = (
        "You are a software engineering assistant. "
        "Answer using only the provided repository context when possible. "
        "If context is missing, say what is missing."
    )
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Repository context:\n{repo_context}\n\n"
                    f"Question:\n{question}"
                ),
            },
        ],
    )
    return response.output_text.strip()


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    default_model = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

    parser = argparse.ArgumentParser(
        description="Ask questions about this repo using OpenAI."
    )
    parser.add_argument(
        "question",
        nargs="*",
        help='Question to ask. Example: "How do I run training?"',
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Path to repo root (default: current directory).",
    )
    parser.add_argument(
        "--model",
        default=default_model,
        help="Model name (default: OPENAI_MODEL or gpt-4o-mini).",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=120_000,
        help="Maximum repo context characters to send to the API.",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Add it to your environment or .env file."
        )

    question = " ".join(args.question).strip()
    if not question:
        question = input("Ask about this repo: ").strip()
    if not question:
        raise SystemExit("Question is required.")

    repo_root = Path(args.repo_root).resolve()
    repo_context = build_repo_context(repo_root, max_chars=args.max_context_chars)
    if not repo_context:
        raise SystemExit("No readable text files found for repo context.")

    answer = ask_repo_assistant(question, repo_context, args.model)
    print(answer)


if __name__ == "__main__":
    main()
