

import os
import psycopg2
from psycopg2.extras import RealDictCursor

# ── Connection string from .env ────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:1234@localhost:5432/rag_db")


def get_connection():
   
    return psycopg2.connect(DATABASE_URL)


def create_tables():
    
    conn = get_connection()
    try:
        cur = conn.cursor()

       
        cur.execute("""
            CREATE TABLE IF NOT EXISTS document_sessions (
                id          TEXT PRIMARY KEY,
                files       TEXT[],
                doc_count   INTEGER     NOT NULL,
                chunk_count INTEGER     NOT NULL,
                created_at  TIMESTAMP   DEFAULT NOW()
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id          SERIAL      PRIMARY KEY,
                session_id  TEXT        REFERENCES document_sessions(id) ON DELETE CASCADE,
                role        TEXT        CHECK (role IN ('user', 'assistant')),
                content     TEXT        NOT NULL,
                created_at  TIMESTAMP   DEFAULT NOW()
            );
        """)

        conn.commit()
        print("[DB] Tables ready.")

    finally:
       
        conn.close()




def save_session(session_id: str, files: list[str], doc_count: int, chunk_count: int):
   
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO document_sessions (id, files, doc_count, chunk_count)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING;
        """, (session_id, files, doc_count, chunk_count))
        conn.commit()
    finally:
        conn.close()


def get_all_sessions() -> list[dict]:
  
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM document_sessions ORDER BY created_at DESC;")
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def delete_session_db(session_id: str):
    
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM document_sessions WHERE id = %s;", (session_id,))
        conn.commit()
    finally:
        conn.close()


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
