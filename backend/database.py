import os
import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:1234@localhost:5432/rag_db")


def get_connection():
    return psycopg2.connect(DATABASE_URL)


# ── Table Creation ─────────────────────────────────────────────────────────────

def create_tables():
    conn = get_connection()
    try:
        cur = conn.cursor()

        # Users table — stores registered accounts
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            SERIAL      PRIMARY KEY,
                username      TEXT        UNIQUE NOT NULL,
                email         TEXT        UNIQUE NOT NULL,
                password_hash TEXT        NOT NULL,
                created_at    TIMESTAMP   DEFAULT NOW()
            );
        """)

        # Document sessions — now linked to a user
        cur.execute("""
            CREATE TABLE IF NOT EXISTS document_sessions (
                id          TEXT        PRIMARY KEY,
                user_id     INTEGER     REFERENCES users(id) ON DELETE CASCADE,
                files       TEXT[],
                doc_count   INTEGER     NOT NULL,
                chunk_count INTEGER     NOT NULL,
                created_at  TIMESTAMP   DEFAULT NOW()
            );
        """)

        # Messages — linked to a session (user is implicit via session)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id          SERIAL      PRIMARY KEY,
                session_id  TEXT        REFERENCES document_sessions(id) ON DELETE CASCADE,
                role        TEXT        CHECK (role IN ('user', 'assistant')),
                content     TEXT        NOT NULL,
                created_at  TIMESTAMP   DEFAULT NOW()
            );
        """)

        # Migration: add user_id to existing tables that pre-date the auth system
        cur.execute("""
            ALTER TABLE document_sessions
            ADD COLUMN IF NOT EXISTS user_id INTEGER REFERENCES users(id) ON DELETE CASCADE;
        """)

        conn.commit()
        print("[DB] Tables ready.")
    finally:
        conn.close()


# ── User CRUD ──────────────────────────────────────────────────────────────────

def create_user(username: str, email: str, password_hash: str) -> dict:
    """Insert a new user and return the created row."""
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            INSERT INTO users (username, email, password_hash)
            VALUES (%s, %s, %s)
            RETURNING id, username, email, created_at;
        """, (username, email, password_hash))
        conn.commit()
        return dict(cur.fetchone())
    finally:
        conn.close()


def get_user_by_username(username: str) -> dict | None:
    """Return a user row by username, or None if not found."""
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM users WHERE username = %s;", (username,))
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> dict | None:
    """Return a user row by ID, or None if not found."""
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM users WHERE id = %s;", (user_id,))
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ── Session CRUD ───────────────────────────────────────────────────────────────

def save_session(session_id: str, user_id: int, files: list[str], doc_count: int, chunk_count: int):
    """Save a new document session linked to a specific user."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO document_sessions (id, user_id, files, doc_count, chunk_count)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING;
        """, (session_id, user_id, files, doc_count, chunk_count))
        conn.commit()
    finally:
        conn.close()


def get_sessions_for_user(user_id: int) -> list[dict]:
    """Return all sessions belonging to a specific user."""
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT * FROM document_sessions
            WHERE user_id = %s
            ORDER BY created_at DESC;
        """, (user_id,))
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def get_all_sessions() -> list[dict]:
    """Return ALL sessions (used at startup to restore in-memory state)."""
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM document_sessions ORDER BY created_at DESC;")
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def delete_session_db(session_id: str):
    """Delete a session and cascade-delete its messages."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM document_sessions WHERE id = %s;", (session_id,))
        conn.commit()
    finally:
        conn.close()


# ── Message CRUD ───────────────────────────────────────────────────────────────

def save_message(session_id: str, role: str, content: str):
    """INSERT one message (user or assistant) linked to a session."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO messages (session_id, role, content)
            VALUES (%s, %s, %s);
        """, (session_id, role, content))
        conn.commit()
    finally:
        conn.close()


def get_messages(session_id: str) -> list[dict]:
    """SELECT all messages for a session, ordered by time."""
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT role, content, created_at
            FROM messages
            WHERE session_id = %s
            ORDER BY created_at ASC;
        """, (session_id,))
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def session_exists(session_id: str) -> bool:
    """Check if a session_id exists in the database."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM document_sessions WHERE id = %s;", (session_id,))
        return cur.fetchone() is not None
    finally:
        conn.close()
