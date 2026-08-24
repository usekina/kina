"""Allow only authorized Aoi identities to modify aoi_kinabot_app/."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROTECTED_PREFIX = "aoi_kinabot_app/"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout


def load_allowed_emails(path: Path) -> set[str]:
    return {
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def changed_files(commit: str) -> list[str]:
    return [
        line.strip()
        for line in git("diff-tree", "--no-commit-id", "--name-only", "-r", "--root", commit).splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("revision_range", help="Git revision range, for example origin/main..HEAD")
    parser.add_argument(
        "--allowed-emails",
        type=Path,
        default=Path(".github/aoi-app-maintainer-emails.txt"),
    )
    args = parser.parse_args()

    allowed = load_allowed_emails(args.allowed_emails)
    violations: list[tuple[str, str, str]] = []
    commits = [line for line in git("rev-list", "--reverse", args.revision_range).splitlines() if line]
    for commit in commits:
        if not any(path.startswith(PROTECTED_PREFIX) for path in changed_files(commit)):
            continue
        author_email = git("show", "-s", "--format=%ae", commit).strip().lower()
        committer_email = git("show", "-s", "--format=%ce", commit).strip().lower()
        if author_email not in allowed:
            violations.append((commit, "author", author_email))
        if committer_email not in allowed:
            violations.append((commit, "committer", committer_email))

    if not violations:
        print("Aoi-maintained application ownership policy passed.")
        return 0

    print(f"Only authorized Aoi identities may modify {PROTECTED_PREFIX}", file=sys.stderr)
    for commit, role, email in violations:
        print(f"- {commit[:12]} {role}: {email}", file=sys.stderr)
    print(
        "External contributors may propose changes in issues or patches; Aoi must independently review and submit accepted implementation commits.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
