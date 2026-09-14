"""Transactional, append-only application writes; not a tamper-proof audit database."""

import csv
import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .types import Crossing


def utc_now():
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class Store:
    def __init__(self, path: str | Path, camera_label: str, settings: dict):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=5)
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY, camera_label TEXT NOT NULL, started_at TEXT NOT NULL,
                ended_at TEXT, settings_json TEXT NOT NULL, status TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS crossings (
                session_id TEXT NOT NULL REFERENCES sessions(id), sequence INTEGER NOT NULL,
                track_id INTEGER NOT NULL, direction TEXT NOT NULL CHECK(direction IN ('in','out')),
                frame_index INTEGER NOT NULL, media_seconds REAL NOT NULL,
                observed_at TEXT NOT NULL,
                x REAL NOT NULL, y REAL NOT NULL,
                PRIMARY KEY(session_id, sequence)
            );
            CREATE TABLE IF NOT EXISTS snapshots (
                session_id TEXT NOT NULL REFERENCES sessions(id), frame_index INTEGER NOT NULL,
                observed_at TEXT NOT NULL, metrics_json TEXT NOT NULL,
                PRIMARY KEY(session_id, frame_index)
            );
            CREATE INDEX IF NOT EXISTS ix_crossings_time ON crossings(observed_at);
        """)
        self.session_id = str(uuid.uuid4())
        with self.connection:
            self.connection.execute(
                "INSERT INTO sessions VALUES (?,?,?,NULL,?,?)",
                (self.session_id, camera_label, utc_now(), json.dumps(settings), "running"),
            )

    def record(self, events: list[Crossing], frame_index: int, metrics: dict | None = None):
        with self.connection:
            self.connection.executemany(
                """
                INSERT INTO crossings VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(session_id,sequence) DO NOTHING
            """,
                [
                    (
                        self.session_id,
                        e.sequence,
                        e.track_id,
                        e.direction,
                        e.frame_index,
                        e.media_seconds,
                        utc_now(),
                        e.x,
                        e.y,
                    )
                    for e in events
                ],
            )
            if metrics is not None:
                self.connection.execute(
                    """
                    INSERT INTO snapshots VALUES (?,?,?,?)
                    ON CONFLICT(session_id,frame_index) DO UPDATE SET
                    metrics_json=excluded.metrics_json, observed_at=excluded.observed_at
                """,
                    (self.session_id, frame_index, utc_now(), json.dumps(metrics)),
                )

    def close(self, status="completed"):
        try:
            with self.connection:
                self.connection.execute(
                    "UPDATE sessions SET ended_at=?,status=? WHERE id=?",
                    (utc_now(), status, self.session_id),
                )
        finally:
            self.connection.close()


def export_csv(database: str | Path, output: str | Path) -> int:
    database = Path(database).resolve()
    if not database.is_file():
        raise FileNotFoundError("Counting database not found")
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    try:
        cursor = connection.execute("""
            SELECT c.session_id,s.camera_label,c.sequence,c.track_id,c.direction,
                   c.frame_index,c.media_seconds,c.observed_at,c.x,c.y
            FROM crossings c JOIN sessions s ON s.id=c.session_id
            ORDER BY s.started_at,c.sequence
        """)
        count = 0
        # Exclusive creation protects pre-existing user files.
        with Path(output).open("x", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([column[0] for column in cursor.description])
            for row in cursor:
                # Prevent spreadsheet formula injection through a user camera label.
                safe = [
                    "'" + value
                    if isinstance(value, str)
                    and value.startswith(("=", "+", "-", "@", "\t", "\r", "\n"))
                    else value
                    for value in row
                ]
                writer.writerow(safe)
                count += 1
        return count
    finally:
        connection.close()
