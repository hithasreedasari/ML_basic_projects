import json
import os
import time
from pathlib import Path
from urllib import error, request

from openai import OpenAI, RateLimitError


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
ASK_PREFIX = "/ask"


def build_repo_context(repo_root: Path, max_chars: int) -> str:
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
        except (UnicodeDecodeError, OSError):
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


def extract_question(comment_body: str) -> str:
    if not comment_body:
        return ""
    stripped = comment_body.strip()
    if not stripped.lower().startswith(ASK_PREFIX):
        return ""
    return stripped[len(ASK_PREFIX) :].strip()


def post_issue_comment(repo: str, issue_number: int, token: str, body: str) -> None:
    url = f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments"
    payload = json.dumps({"body": body}).encode("utf-8")
    req = request.Request(
        url=url,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        },
    )
    with request.urlopen(req) as resp:
        if resp.status < 200 or resp.status >= 300:
            raise RuntimeError(f"GitHub API returned status {resp.status}")


def answer_question(question: str, repo_context: str, model: str) -> str:
    base_url = os.getenv("OPENAI_BASE_URL")
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=base_url if base_url else None,
    )
    system_prompt = (
        "You are a repository assistant. Answer using the provided repository context. "
        "Be concise and practical. If information is missing, explicitly say so."
    )
    for attempt in range(3):
        try:
            resp = client.responses.create(
                model=model,
                max_output_tokens=400,
                input=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Repository context:\n{repo_context}\n\n"
                            f"Question from GitHub issue comment:\n{question}"
                        ),
                    },
                ],
            )
            return resp.output_text.strip()
        except RateLimitError:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    return ""


def is_memory_related_error(exc: Exception) -> bool:
    details = str(exc).lower()
    return (
        "requires more system memory" in details
        or "out of memory" in details
        or "not enough memory" in details
    )


def is_model_missing_error(exc: Exception) -> bool:
    details = str(exc).lower()
    return "model" in details and ("not found" in details or "does not exist" in details)


def build_model_candidates(primary_model: str) -> list[str]:
    candidates = [primary_model]

    fallback_model = os.getenv("OPENAI_FALLBACK_MODEL", "").strip()
    if fallback_model:
        candidates.append(fallback_model)

    extra_candidates = os.getenv("OPENAI_MODEL_CANDIDATES", "").strip()
    if extra_candidates:
        candidates.extend([m.strip() for m in extra_candidates.split(",") if m.strip()])

    base_url = (os.getenv("OPENAI_BASE_URL") or "").lower()
    if "127.0.0.1:11434" in base_url or "localhost:11434" in base_url:
        # Common lightweight Ollama model names for low-memory machines.
        candidates.extend([
            "llama3.2:1b",
            "qwen2.5:0.5b",
            "qwen2.5:1.5b",
            "tinyllama:latest",
            "phi3:mini",
        ])

    deduped = []
    seen = set()
    for model in candidates:
        if model and model not in seen:
            deduped.append(model)
            seen.add(model)
    return deduped


def generate_answer_with_fallback(question: str, repo_context: str, model: str) -> str:
    candidates = build_model_candidates(model)
    last_exc: Exception | None = None

    for candidate in candidates:
        context_variant = repo_context
        for _ in range(3):
            try:
                return answer_question(question, context_variant, model=candidate)
            except Exception as exc:
                last_exc = exc
                if is_memory_related_error(exc):
                    if len(context_variant) > 3000:
                        # Reduce prompt size before switching models.
                        context_variant = context_variant[: max(3000, len(context_variant) // 2)]
                        continue
                    break
                if is_model_missing_error(exc):
                    break
                raise

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("No model candidates available")


def main() -> None:
    event_path = os.getenv("GITHUB_EVENT_PATH")
    github_token = os.getenv("GITHUB_TOKEN")
    openai_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    max_context_chars = int(os.getenv("MAX_CONTEXT_CHARS", "30000"))

    if not event_path or not github_token or not openai_key:
        raise SystemExit("Missing required env vars: GITHUB_EVENT_PATH, GITHUB_TOKEN, OPENAI_API_KEY")

    os.environ["OPENAI_API_KEY"] = openai_key

    with open(event_path, "r", encoding="utf-8") as f:
        event = json.load(f)

    comment = event.get("comment", {})
    comment_body = comment.get("body", "")
    comment_user = (comment.get("user", {}) or {}).get("login", "")
    if comment_user.endswith("[bot]"):
        return

    issue = event.get("issue", {})
    if "pull_request" in issue:
        return

    question = extract_question(comment_body)
    if not question:
        return

    repo = (event.get("repository", {}) or {}).get("full_name")
    issue_number = issue.get("number")
    if not repo or not issue_number:
        raise SystemExit("Missing repository or issue number in event payload")

    repo_root = Path(".").resolve()
    repo_context = build_repo_context(repo_root, max_chars=max_context_chars)
    if not repo_context:
        post_issue_comment(
            repo,
            issue_number,
            github_token,
            "I could not read repository files for context.",
        )
        return

    try:
        answer = generate_answer_with_fallback(question, repo_context, model=model)
    except Exception as exc:
        details = str(exc).strip()
        if not details:
            details = repr(exc)
        if len(details) > 400:
            details = details[:400] + "..."
        post_issue_comment(
            repo,
            issue_number,
            github_token,
            (
                "I hit an error while generating an answer.\n\n"
                f"- Type: `{type(exc).__name__}`\n"
                f"- Details: `{details}`"
            ),
        )
        return

    response_body = (
        "### Repo Assistant\n\n"
        f"**Question:** {question}\n\n"
        f"{answer}\n\n"
        "_Ask another question by commenting with `/ask ...`._"
    )
    try:
        post_issue_comment(repo, issue_number, github_token, response_body)
    except error.HTTPError as exc:
        raise SystemExit(f"Failed posting comment to GitHub: {exc.code}") from exc


if __name__ == "__main__":
    main()
