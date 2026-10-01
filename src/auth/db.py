"""
File: src/auth/db.py
Purpose: Initialises the local SQLite database and defines the full schema.
         Also provides delete_account() for complete, irreversible user data removal.
Why we need it: All persistent data (user accounts, diagnosis history, session tokens)
                lives in a single .db file. This module guarantees the schema exists
                before any other module tries to read or write data.
"""

import sqlite3
import os
import logging

# The database lives inside the data/ directory (which is gitignored)
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "app.db")


def get_connection() -> sqlite3.Connection:
    """
    Opens and returns a SQLite connection with row_factory set to Row,
    so columns are accessible by name (row["username"]) rather than index.
    Why we need it: Centralises connection settings — every module calls this
                    instead of hardcoding the path or settings themselves.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # Safe for multi-thread Streamlit access
    return conn


def init_db() -> None:
    """
    Creates all tables if they do not already exist.
    Safe to call on every app startup — uses CREATE TABLE IF NOT EXISTS.
    Why we need it: The DB file may not exist on first run. This function
                    bootstraps the schema automatically with no manual setup.

    Tables:
        users             — registered user accounts
        diagnosis_history — every prediction result tied to a user
        sessions          — persistent login tokens (30-day expiry)
    """
    conn = get_connection()
    cursor = conn.cursor()

    # --- users ---
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT    UNIQUE NOT NULL,
            full_name   TEXT    NOT NULL,
            email       TEXT    UNIQUE NOT NULL,
            phone       TEXT    NOT NULL,
            dob         TEXT,
            gender      TEXT,
            password_hash TEXT  NOT NULL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # --- diagnosis_history ---
    # probabilities stored as a JSON string (10-class probability dict)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS diagnosis_history (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER NOT NULL,
            prediction    TEXT    NOT NULL,
            confidence    REAL    NOT NULL,
            probabilities TEXT    NOT NULL,
            symptom_text  TEXT,
            image_used    INTEGER DEFAULT 0,
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # --- sessions ---
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL UNIQUE,
            token       TEXT    UNIQUE NOT NULL,
            expires_at  TIMESTAMP NOT NULL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()


# ==============================================================================
# ACCOUNT DELETION
# ==============================================================================

def delete_account(user_id: int) -> bool:
    """
    Permanently and irreversibly deletes every row associated with a user.
    Why we need it: Full right-to-erasure — after this call, no trace of the
                    user exists in any table. The operation is wrapped in an
                    explicit transaction so it's atomic: either all three
                    deletes succeed together, or none do.

    Deletion order matters here:
        1. diagnosis_history — has a FOREIGN KEY referencing users(id)
        2. sessions          — has a FOREIGN KEY referencing users(id)
        3. users             — the parent row; must be deleted last

    Args:
        user_id: The integer primary key from the users table.

    Returns:
        True if deletion succeeded, False on any DB error.
    """
    conn = get_connection()
    try:
        # Python's sqlite3 auto-transaction mode issues an implicit BEGIN before
        # the first DML statement. All three DELETEs run in the same transaction
        # and are committed atomically. An explicit BEGIN is NOT used here because
        # it conflicts with the auto-transaction management and causes an
        # OperationalError: "cannot start a transaction within a transaction".
        conn.execute("DELETE FROM diagnosis_history WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        logging.info(f"Account deleted: user_id={user_id}")
        return True
    except Exception as e:
        conn.rollback()
        logging.error(f"Account deletion failed for user_id={user_id}: {e}")
        return False
    finally:
        conn.close()
