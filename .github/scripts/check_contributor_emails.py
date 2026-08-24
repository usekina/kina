"""Reject commits attributed to blocked employer email domains."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def load_blocked_domains(path: Path) -> set[str]:
    return {
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def commit_identities(revision_range: str) -> list[tuple[str, str, str]]:
    result = subprocess.run(
        [
            "git",
            "log",
            revision_range,
            "--format=%H%x09%ae%x09%ce",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    identities: list[tuple[str, str, str]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        commit, author_email, committer_email = line.split("\t", maxsplit=2)
        identities.append((commit, author_email, committer_email))
    return identities


def email_domain(email: str) -> str:
    return email.rsplit("@", maxsplit=1)[-1].lower() if "@" in email else ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("revision_range", help="Git revision range, for example origin/main..HEAD")
    parser.add_argument(
        "--blocked-domains",
        type=Path,
        default=Path(".github/blocked-contributor-email-domains.txt"),
    )
    args = parser.parse_args()

    blocked = load_blocked_domains(args.blocked_domains)
    violations: list[tuple[str, str, str]] = []
    for commit, author_email, committer_email in commit_identities(args.revision_range):
        for role, email in (("author", author_email), ("committer", committer_email)):
            if email_domain(email) in blocked:
                violations.append((commit, role, email))

    if not violations:
        print("Contributor email policy passed.")
        return 0

    print("Contributor email policy failed:", file=sys.stderr)
    for commit, role, email in violations:
        print(f"- {commit[:12]} {role}: {email}", file=sys.stderr)
    print(
        "Use a personal or authorized project email and recreate the affected commit(s).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
