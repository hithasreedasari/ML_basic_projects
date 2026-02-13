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
        answer = answer_question(question, repo_context, model=model)
    except Exception as exc:
        post_issue_comment(
            repo,
            issue_number,
            github_token,
            f"I hit an error while generating an answer: `{type(exc).__name__}`.",
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
