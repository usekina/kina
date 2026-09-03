"""SQLite database helpers for KinaBot V1."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from config import DATABASE_PATH
from local_time import utc_iso_to_local_date


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    db_path = db_path or DATABASE_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = [row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email_hash TEXT NOT NULL UNIQUE,
                email TEXT,
                display_name TEXT,
                age_range TEXT,
                gender TEXT,
                primary_language TEXT,
                country_region TEXT,
                timezone_name TEXT,
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
                withdrawn_at TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS test_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_date TEXT NOT NULL,
                session_number INTEGER NOT NULL,
                session_type TEXT,
                language TEXT,
                duration_seconds REAL,
                timezone_name TEXT,
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

            CREATE TABLE IF NOT EXISTS habit_checkins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                checkin_date TEXT NOT NULL,
                habit_name TEXT NOT NULL,
                completed INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, checkin_date, habit_name),
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            """
        )
        ensure_column(conn, "users", "email", "TEXT")
        ensure_column(conn, "users", "display_name", "TEXT")
        ensure_column(conn, "users", "age_range", "TEXT")
        ensure_column(conn, "users", "gender", "TEXT")
        ensure_column(conn, "users", "primary_language", "TEXT")
        ensure_column(conn, "users", "country_region", "TEXT")
        ensure_column(conn, "users", "timezone_name", "TEXT")
        ensure_column(conn, "test_sessions", "session_type", "TEXT")
        ensure_column(conn, "test_sessions", "language", "TEXT")
        ensure_column(conn, "test_sessions", "duration_seconds", "REAL")
        ensure_column(conn, "test_sessions", "timezone_name", "TEXT")
        ensure_column(conn, "consent_events", "withdrawn_at", "TEXT")
        ensure_column(conn, "test_sessions", "idempotency_key", "TEXT")
        _repair_legacy_session_number_duplicates(conn)
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_sessions_user_date_number "
            "ON test_sessions(user_id, session_date, session_number)"
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_feature_scores_session_name "
            "ON feature_scores(test_session_id, feature_name)"
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_sessions_idempotency_key "
            "ON test_sessions(idempotency_key) WHERE idempotency_key IS NOT NULL"
        )


def _repair_legacy_session_number_duplicates(conn: sqlite3.Connection) -> None:
    """Normalize legacy numbering before adding the production uniqueness index."""
    rows = conn.execute(
        """
        SELECT id, user_id, session_date
        FROM test_sessions
        ORDER BY user_id, session_date, created_at, id
        """
    ).fetchall()
    counters: dict[tuple[int, str], int] = {}
    for row in rows:
        key = (int(row["user_id"]), str(row["session_date"]))
        counters[key] = counters.get(key, 0) + 1
        conn.execute(
            "UPDATE test_sessions SET session_number = ? WHERE id = ?",
            (counters[key], int(row["id"])),
        )


def upsert_user(email_hash: str, email: str | None = None) -> int:
    now = utc_now_iso()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO users (email_hash, email, created_at, last_active_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(email_hash)
            DO UPDATE SET
                email = COALESCE(excluded.email, users.email),
                last_active_at = excluded.last_active_at
            """,
            (email_hash, email, now, now),
        )
        row = conn.execute(
            "SELECT id FROM users WHERE email_hash = ?",
            (email_hash,),
        ).fetchone()
        return int(row["id"])


def find_user_id_by_email_hash(email_hash: str) -> int | None:
    """Resolve a pseudonymous key without creating a participant record."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id FROM users WHERE email_hash = ?", (email_hash,)
        ).fetchone()
        return int(row["id"]) if row else None


def delete_user_research_data(user_id: int) -> bool:
    """Delete one participant's local records in an explicit transaction."""
    with get_connection() as conn:
        if not conn.execute("SELECT 1 FROM users WHERE id = ?", (user_id,)).fetchone():
            return False
        session_ids = [
            int(row["id"])
            for row in conn.execute(
                "SELECT id FROM test_sessions WHERE user_id = ?", (user_id,)
            ).fetchall()
        ]
        if session_ids:
            placeholders = ",".join("?" for _ in session_ids)
            conn.execute(
                f"DELETE FROM feature_scores WHERE test_session_id IN ({placeholders})",
                session_ids,
            )
        email_hash = conn.execute(
            "SELECT email_hash FROM users WHERE id = ?", (user_id,)
        ).fetchone()["email_hash"]
        conn.execute("DELETE FROM verification_codes WHERE email_hash = ?", (email_hash,))
        conn.execute("DELETE FROM habit_checkins WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM consent_events WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM test_sessions WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        return True


def get_user_profile(user_id: int) -> sqlite3.Row | None:
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT id, email, display_name, age_range, gender, primary_language,
                   country_region, timezone_name
            FROM users
            WHERE id = ?
            """,
            (user_id,),
        ).fetchone()


def update_user_profile(
    user_id: int,
    display_name: str | None,
    age_range: str | None,
    gender: str | None,
    primary_language: str | None,
    country_region: str | None,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE users
            SET display_name = ?, age_range = ?, gender = ?, primary_language = ?,
                country_region = ?, last_active_at = ?
            WHERE id = ?
            """,
            (
                display_name,
                age_range,
                gender,
                primary_language,
                country_region,
                utc_now_iso(),
                user_id,
            ),
        )


def record_consent(user_id: int, consent_version: str) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO consent_events (user_id, consent_version, accepted_at)
            SELECT ?, ?, ?
            WHERE NOT EXISTS (
                SELECT 1
                FROM consent_events
                WHERE user_id = ? AND consent_version = ? AND withdrawn_at IS NULL
            )
            """,
            (user_id, consent_version, utc_now_iso(), user_id, consent_version),
        )


def has_active_consent(user_id: int, consent_version: str) -> bool:
    with get_connection() as conn:
        return bool(conn.execute(
            "SELECT 1 FROM consent_events WHERE user_id = ? AND consent_version = ? "
            "AND withdrawn_at IS NULL LIMIT 1", (user_id, consent_version)
        ).fetchone())


def withdraw_research_consent(user_id: int) -> int:
    """Stop future research use while retaining the consent audit event."""
    with get_connection() as conn:
        cursor = conn.execute(
            "UPDATE consent_events SET withdrawn_at = ? "
            "WHERE user_id = ? AND withdrawn_at IS NULL",
            (utc_now_iso(), user_id),
        )
        return int(cursor.rowcount)


def export_user_data(user_id: int) -> dict:
    """Return a participant's stored data in a portable JSON-compatible shape."""
    with get_connection() as conn:
        user = conn.execute(
            "SELECT id, email, display_name, age_range, gender, primary_language, "
            "country_region, timezone_name, created_at, last_active_at "
            "FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if not user:
            return {}
        sessions = conn.execute(
            "SELECT * FROM test_sessions WHERE user_id = ? ORDER BY created_at ASC, id ASC",
            (user_id,),
        ).fetchall()
        session_ids = [int(row["id"]) for row in sessions]
        scores = []
        if session_ids:
            placeholders = ",".join("?" for _ in session_ids)
            scores = conn.execute(
                f"SELECT * FROM feature_scores WHERE test_session_id IN ({placeholders}) "
                "ORDER BY test_session_id ASC, id ASC", session_ids
            ).fetchall()
        consents = conn.execute(
            "SELECT consent_version, accepted_at, withdrawn_at FROM consent_events "
            "WHERE user_id = ? ORDER BY accepted_at ASC", (user_id,)
        ).fetchall()
        habits = conn.execute(
            "SELECT checkin_date, habit_name, completed, created_at, updated_at "
            "FROM habit_checkins WHERE user_id = ? ORDER BY checkin_date ASC, id ASC",
            (user_id,),
        ).fetchall()
        return {
            "profile": dict(user),
            "sessions": [dict(row) for row in sessions],
            "feature_scores": [dict(row) for row in scores],
            "consent_events": [dict(row) for row in consents],
            "habit_checkins": [dict(row) for row in habits],
        }


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


def assign_timezone_to_legacy_sessions(user_id: int, timezone_name: str) -> int:
    """Rebucket legacy UTC-dated sessions once using the user's browser zone."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE users SET timezone_name = ?, last_active_at = ? WHERE id = ?",
            (timezone_name, utc_now_iso(), user_id),
        )
        rows = conn.execute(
            """
            SELECT id, created_at
            FROM test_sessions
            WHERE user_id = ?
              AND (timezone_name IS NULL OR timezone_name = '')
            ORDER BY created_at ASC, id ASC
            """,
            (user_id,),
        ).fetchall()
        for row in rows:
            local_date = utc_iso_to_local_date(row["created_at"], timezone_name)
            conn.execute(
                """
                UPDATE test_sessions
                SET session_date = ?, timezone_name = ?
                WHERE id = ?
                """,
                (local_date, timezone_name, row["id"]),
            )

        dated_rows = conn.execute(
            """
            SELECT id, session_date
            FROM test_sessions
            WHERE user_id = ?
            ORDER BY session_date ASC, created_at ASC, id ASC
            """,
            (user_id,),
        ).fetchall()
        counters: dict[str, int] = {}
        for row in dated_rows:
            session_date = str(row["session_date"])
            counters[session_date] = counters.get(session_date, 0) + 1
            conn.execute(
                "UPDATE test_sessions SET session_number = ? WHERE id = ?",
                (counters[session_date], row["id"]),
            )
        return len(rows)


def create_test_session(
    user_id: int,
    session_date: str,
    session_number: int,
    app_version: str,
    consent_version: str,
    scoring_model_version: str,
    session_type: str | None = None,
    language: str | None = None,
    duration_seconds: float | None = None,
    timezone_name: str | None = None,
    idempotency_key: str | None = None,
) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO test_sessions
                (user_id, session_date, session_number, session_type, language,
                 duration_seconds, timezone_name, app_version, consent_version,
                scoring_model_version, idempotency_key, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                session_date,
                session_number,
                session_type,
                language,
                duration_seconds,
                timezone_name,
                app_version,
                consent_version,
                scoring_model_version,
                idempotency_key,
                utc_now_iso(),
            ),
        )
        return int(cursor.lastrowid)


class DailyLimitReached(ValueError):
    """Raised when an atomic completion would exceed the daily limit."""


def complete_test_session(
    *,
    user_id: int,
    session_date: str,
    app_version: str,
    consent_version: str,
    scoring_model_version: str,
    scores: Iterable[dict],
    max_tests_per_day: int,
    session_type: str | None = None,
    language: str | None = None,
    duration_seconds: float | None = None,
    timezone_name: str | None = None,
    idempotency_key: str | None = None,
) -> tuple[int, int, bool]:
    """Atomically enforce quota and persist one complete scored session.

    Returns (session_id, session_number, already_existed). Transcription stays
    outside this short write transaction; all database writes commit or roll back.
    """
    score_rows = list(scores)
    with get_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        if idempotency_key:
            existing = conn.execute(
                "SELECT id, session_number FROM test_sessions WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if existing:
                return int(existing["id"]), int(existing["session_number"]), True

        count = conn.execute(
            "SELECT COUNT(*) AS count FROM test_sessions WHERE user_id = ? AND session_date = ?",
            (user_id, session_date),
        ).fetchone()["count"]
        if int(count) >= max_tests_per_day:
            raise DailyLimitReached("Daily reflection limit reached.")

        session_number = int(count) + 1
        now = utc_now_iso()
        cursor = conn.execute(
            """
            INSERT INTO test_sessions
                (user_id, session_date, session_number, session_type, language,
                 duration_seconds, timezone_name, app_version, consent_version,
                 scoring_model_version, idempotency_key, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id, session_date, session_number, session_type, language,
                duration_seconds, timezone_name, app_version, consent_version,
                scoring_model_version, idempotency_key, now,
            ),
        )
        session_id = int(cursor.lastrowid)
        conn.executemany(
            """
            INSERT INTO feature_scores
                (test_session_id, feature_name, raw_metric, score, explanation, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    session_id,
                    item["feature_name"],
                    item.get("raw_metric"),
                    float(item["score"]),
                    item["explanation"],
                    now,
                )
                for item in score_rows
            ],
        )
        return session_id, session_number, False


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
    """Return every score stored for one account across languages and versions."""
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT
                ts.id AS session_id,
                ts.created_at,
                ts.session_date,
                ts.session_number,
                ts.session_type,
                ts.language,
                ts.duration_seconds,
                ts.app_version,
                ts.consent_version,
                ts.scoring_model_version,
                fs.feature_name,
                fs.score
            FROM feature_scores fs
            JOIN test_sessions ts ON ts.id = fs.test_session_id
            WHERE ts.user_id = ?
            ORDER BY ts.created_at ASC, fs.feature_name ASC
            """,
            (user_id,),
        ).fetchall()


def save_habit_checkins(user_id: int, checkin_date: str, habits: dict[str, bool]) -> None:
    now = utc_now_iso()
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO habit_checkins
                (user_id, checkin_date, habit_name, completed, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, checkin_date, habit_name)
            DO UPDATE SET completed = excluded.completed, updated_at = excluded.updated_at
            """,
            [
                (user_id, checkin_date, name, int(completed), now, now)
                for name, completed in habits.items()
            ],
        )


def get_user_habit_checkins(user_id: int) -> list[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT checkin_date, habit_name, completed, updated_at
            FROM habit_checkins
            WHERE user_id = ?
            ORDER BY checkin_date ASC, habit_name ASC
            """,
            (user_id,),
        ).fetchall()


def get_admin_metrics() -> dict:
    with get_connection() as conn:
        total_users = conn.execute("SELECT COUNT(*) AS count FROM users").fetchone()["count"]
        total_tests = conn.execute("SELECT COUNT(*) AS count FROM test_sessions").fetchone()["count"]
        total_scores = conn.execute("SELECT COUNT(*) AS count FROM feature_scores").fetchone()["count"]
        active_today = conn.execute(
            "SELECT COUNT(DISTINCT user_id) AS count FROM test_sessions WHERE session_date = ?",
            (datetime.now(timezone.utc).date().isoformat(),),
        ).fetchone()["count"]
        return {
            "total_users": int(total_users),
            "total_tests": int(total_tests),
            "total_scores": int(total_scores),
            "active_users_today": int(active_today),
        }


def list_admin_users() -> list[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT id, email, display_name, age_range, gender, primary_language,
                   country_region, timezone_name, created_at, last_active_at
            FROM users
            ORDER BY created_at DESC
            """
        ).fetchall()


def list_admin_test_records() -> list[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT
                u.email,
                u.display_name,
                u.age_range,
                u.gender,
                u.primary_language,
                u.country_region,
                ts.created_at,
                ts.session_number,
                ts.session_type,
                ts.language,
                ts.duration_seconds,
                ts.timezone_name,
                fs.feature_name,
                fs.score,
                fs.raw_metric
            FROM feature_scores fs
            JOIN test_sessions ts ON ts.id = fs.test_session_id
            JOIN users u ON u.id = ts.user_id
            ORDER BY ts.created_at DESC, fs.feature_name ASC
            """
        ).fetchall()


def list_research_records() -> list[sqlite3.Row]:
    """Return a de-identified, analysis-ready longitudinal dataset."""
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT
                printf('P%06d', u.id) AS participant_id,
                u.age_range,
                u.gender,
                u.primary_language,
                u.country_region,
                ts.id AS session_id,
                ts.session_date,
                ts.created_at,
                ts.session_number,
                ts.session_type,
                ts.language,
                ts.duration_seconds,
                ts.timezone_name,
                ts.app_version,
                ts.consent_version,
                ts.scoring_model_version,
                fs.feature_name,
                fs.raw_metric,
                fs.score
            FROM feature_scores fs
            JOIN test_sessions ts ON ts.id = fs.test_session_id
            JOIN users u ON u.id = ts.user_id
            ORDER BY u.id ASC, ts.created_at ASC, fs.feature_name ASC
            """
        ).fetchall()
