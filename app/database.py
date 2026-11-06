"""
database.py — SQLite persistence layer
Tables:
  images   – master record per uploaded image
  versions – every edit creates a new version row
"""

import sqlite3
import os
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "image_editor.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist yet."""
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS images (
                id          TEXT PRIMARY KEY,
                filename    TEXT NOT NULL,
                filepath    TEXT NOT NULL,
                thumbnail   TEXT,
                caption     TEXT,
                embedding   BLOB,          -- JSON-serialised float list
                uploaded_at TEXT NOT NULL,
                provider    TEXT           -- which AI provider captioned it
            );

            CREATE TABLE IF NOT EXISTS versions (
                id           TEXT PRIMARY KEY,
                image_id     TEXT NOT NULL REFERENCES images(id),
                version_num  INTEGER NOT NULL,
                filepath     TEXT NOT NULL,
                thumbnail    TEXT,
                prompt       TEXT,
                provider     TEXT,
                model        TEXT,
                created_at   TEXT NOT NULL,
                notes        TEXT
            );
            """
        )


def insert_image(row: dict) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO images
              (id, filename, filepath, thumbnail, caption, embedding, uploaded_at, provider)
            VALUES (:id,:filename,:filepath,:thumbnail,:caption,:embedding,:uploaded_at,:provider)
            """,
            row,
        )


def update_image_caption(image_id: str, caption: str, embedding: str, provider: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE images SET caption=?, embedding=?, provider=? WHERE id=?",
            (caption, embedding, provider, image_id),
        )


def insert_version(row: dict) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO versions
              (id, image_id, version_num, filepath, thumbnail, prompt, provider, model, created_at, notes)
            VALUES (:id,:image_id,:version_num,:filepath,:thumbnail,:prompt,:provider,:model,:created_at,:notes)
            """,
            row,
        )


def get_all_images() -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM images ORDER BY uploaded_at DESC"
        ).fetchall()


def get_image(image_id: str) -> sqlite3.Row | None:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM images WHERE id=?", (image_id,)
        ).fetchone()


def get_versions(image_id: str) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM versions WHERE image_id=? ORDER BY version_num ASC",
            (image_id,),
        ).fetchall()


def next_version_num(image_id: str) -> int:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT MAX(version_num) as mx FROM versions WHERE image_id=?",
            (image_id,),
        ).fetchone()
        return (row["mx"] or 0) + 1


def delete_image(image_id: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM versions WHERE image_id=?", (image_id,))
        conn.execute("DELETE FROM images WHERE id=?", (image_id,))
