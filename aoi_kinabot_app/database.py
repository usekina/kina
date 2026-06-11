"""SQLite database helpers for KinaBot V1."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from config import DATABASE_PATH


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_connection(db_path: Path = DATABASE_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email_hash TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                last_active_at TEXT
            );

            CREATE TABLE IF NOT EXISTS verification_codes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email_hash TEXT NOT NULL,
                code_hash TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                used_at TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS consent_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                consent_version TEXT NOT NULL,
                accepted_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS test_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_date TEXT NOT NULL,
                session_number INTEGER NOT NULL,
                app_version TEXT NOT NULL,
                consent_version TEXT NOT NULL,
                scoring_model_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS feature_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_session_id INTEGER NOT NULL,
                feature_name TEXT NOT NULL,
                raw_metric TEXT,
                score REAL NOT NULL,
                explanation TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (test_session_id) REFERENCES test_sessions(id)
            );
            """
        )


def upsert_user(email_hash: str) -> int:
    now = utc_now_iso()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO users (email_hash, created_at, last_active_at)
            VALUES (?, ?, ?)
            ON CONFLICT(email_hash)
            DO UPDATE SET last_active_at = excluded.last_active_at
            """,
            (email_hash, now, now),
        )
        row = conn.execute(
            "SELECT id FROM users WHERE email_hash = ?",
            (email_hash,),
        ).fetchone()
        return int(row["id"])


def record_consent(user_id: int, consent_version: str) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO consent_events (user_id, consent_version, accepted_at)
            VALUES (?, ?, ?)
            """,
            (user_id, consent_version, utc_now_iso()),
        )


def save_verification_code(email_hash: str, code_hash: str, expires_at: str) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO verification_codes
                (email_hash, code_hash, expires_at, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (email_hash, code_hash, expires_at, utc_now_iso()),
        )


def find_active_code(email_hash: str, code_hash: str, now_iso: str) -> sqlite3.Row | None:
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT id, expires_at, used_at
            FROM verification_codes
            WHERE email_hash = ?
              AND code_hash = ?
              AND used_at IS NULL
              AND expires_at >= ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (email_hash, code_hash, now_iso),
        ).fetchone()


def mark_code_used(code_id: int) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE verification_codes SET used_at = ? WHERE id = ?",
            (utc_now_iso(), code_id),
        )


def count_tests_today(user_id: int, session_date: str) -> int:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM test_sessions
            WHERE user_id = ? AND session_date = ?
            """,
            (user_id, session_date),
        ).fetchone()
        return int(row["count"])


def create_test_session(
    user_id: int,
    session_date: str,
    session_number: int,
    app_version: str,
    consent_version: str,
    scoring_model_version: str,
) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO test_sessions
                (user_id, session_date, session_number, app_version,
                 consent_version, scoring_model_version, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                session_date,
                session_number,
                app_version,
                consent_version,
                scoring_model_version,
                utc_now_iso(),
            ),
        )
        return int(cursor.lastrowid)


def save_feature_scores(test_session_id: int, scores: Iterable[dict]) -> None:
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO feature_scores
                (test_session_id, feature_name, raw_metric, score, explanation, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    test_session_id,
                    item["feature_name"],
                    item.get("raw_metric"),
                    float(item["score"]),
                    item["explanation"],
                    utc_now_iso(),
                )
                for item in scores
            ],
        )


def get_user_scores(user_id: int) -> list[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT
                ts.created_at,
                ts.session_number,
                fs.feature_name,
                fs.score
            FROM feature_scores fs
            JOIN test_sessions ts ON ts.id = fs.test_session_id
            WHERE ts.user_id = ?
            ORDER BY ts.created_at ASC, fs.feature_name ASC
            """,
            (user_id,),
        ).fetchall()
