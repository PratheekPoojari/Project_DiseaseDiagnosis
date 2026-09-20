"""
File: src/auth/db.py
Purpose: Initialises the local SQLite database and defines the full schema.
Why we need it: All persistent data (user accounts, diagnosis history, session tokens)
                lives in a single .db file. This module guarantees the schema exists
                before any other module tries to read or write data.
"""

import sqlite3
import os

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
