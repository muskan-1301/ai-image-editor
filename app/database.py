"""
database.py — SQLite persistence layer (Week 3 Enhanced)

Tables:
  images   — master record per uploaded image (enhanced with metadata)
  versions — every edit creates a new version row (enhanced tracking)
  searches — search history log (new in Week 3)
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
    """Create tables and migrate existing schema if needed."""
    with get_conn() as conn:
        # Core tables
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS images (
                id          TEXT PRIMARY KEY,
                filename    TEXT NOT NULL,
                filepath    TEXT NOT NULL,
                thumbnail   TEXT,
                caption     TEXT,
                embedding   BLOB,
                uploaded_at TEXT NOT NULL,
                provider    TEXT,
                tags        TEXT,
                width       INTEGER,
                height      INTEGER,
                file_size   INTEGER
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

            CREATE TABLE IF NOT EXISTS searches (
                id          TEXT PRIMARY KEY,
                query       TEXT NOT NULL,
                result_count INTEGER,
                top_score   REAL,
                searched_at TEXT NOT NULL
            );
        """)

        # Week 3 migration: add new columns to existing tables if missing
        existing_img_cols = [
            r[1] for r in conn.execute("PRAGMA table_info(images)").fetchall()
        ]
        migrations = {
            "tags":      "ALTER TABLE images ADD COLUMN tags TEXT",
            "width":     "ALTER TABLE images ADD COLUMN width INTEGER",
            "height":    "ALTER TABLE images ADD COLUMN height INTEGER",
            "file_size": "ALTER TABLE images ADD COLUMN file_size INTEGER",
        }
        for col, sql in migrations.items():
            if col not in existing_img_cols:
                conn.execute(sql)


def insert_image(row: dict) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO images
              (id, filename, filepath, thumbnail, caption, embedding,
               uploaded_at, provider, tags, width, height, file_size)
            VALUES
              (:id,:filename,:filepath,:thumbnail,:caption,:embedding,
               :uploaded_at,:provider,:tags,:width,:height,:file_size)
            """,
            {
                "id": row["id"],
                "filename": row["filename"],
                "filepath": row["filepath"],
                "thumbnail": row.get("thumbnail"),
                "caption": row.get("caption", ""),
                "embedding": row.get("embedding", ""),
                "uploaded_at": row["uploaded_at"],
                "provider": row.get("provider"),
                "tags": row.get("tags", ""),
                "width": row.get("width"),
                "height": row.get("height"),
                "file_size": row.get("file_size"),
            },
        )


def update_image_caption(image_id: str, caption: str, embedding: str, provider: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE images SET caption=?, embedding=?, provider=? WHERE id=?",
            (caption, embedding, provider, image_id),
        )


def update_image_tags(image_id: str, tags: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE images SET tags=? WHERE id=?", (tags, image_id))


def insert_version(row: dict) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO versions
              (id, image_id, version_num, filepath, thumbnail,
               prompt, provider, model, created_at, notes)
            VALUES
              (:id,:image_id,:version_num,:filepath,:thumbnail,
               :prompt,:provider,:model,:created_at,:notes)
            """,
            row,
        )


def log_search(search_id: str, query: str, result_count: int,
               top_score: float, searched_at: str) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO searches (id, query, result_count, top_score, searched_at)
            VALUES (?,?,?,?,?)
            """,
            (search_id, query, result_count, top_score, searched_at),
        )


def get_all_images() -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM images ORDER BY uploaded_at DESC"
        ).fetchall()


def get_images_without_embeddings() -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM images WHERE embedding IS NULL OR embedding = ''"
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


def get_search_history(limit: int = 10) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM searches ORDER BY searched_at DESC LIMIT ?",
            (limit,),
        ).fetchall()


def get_library_stats() -> dict:
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) as n FROM images").fetchone()["n"]
        indexed = conn.execute(
            "SELECT COUNT(*) as n FROM images WHERE embedding IS NOT NULL AND embedding != ''"
        ).fetchone()["n"]
        edits = conn.execute("SELECT COUNT(*) as n FROM versions").fetchone()["n"]
        searches = conn.execute("SELECT COUNT(*) as n FROM searches").fetchone()["n"]
        return {
            "total_images": total,
            "indexed_images": indexed,
            "total_edits": edits,
            "total_searches": searches,
        }


def delete_image(image_id: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM versions WHERE image_id=?", (image_id,))
        conn.execute("DELETE FROM images WHERE id=?", (image_id,))
