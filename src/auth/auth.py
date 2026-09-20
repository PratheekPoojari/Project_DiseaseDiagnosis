"""
File: src/auth/auth.py
Purpose: Handles all authentication logic: signup, login, logout, and persistent
         session token management (30-day auto-login via a local token file).
Why we need it: Keeps all auth concerns isolated from the UI layer (app.py).
                Passwords are never stored in plain text — PBKDF2-HMAC-SHA256
                with a random salt is used, which is stdlib and academically sound.
"""

import hashlib
import secrets
import json
import os
from datetime import datetime, timedelta

from src.auth.db import get_connection

# The session token is persisted in this file so the user stays logged in
# even after the browser/app is closed (Option A — one machine, one user).
SESSION_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "session.json"
)
SESSION_EXPIRY_DAYS = 30


# ==============================================================================
# PASSWORD HASHING
# ==============================================================================

def _hash_password(password: str) -> str:
    """
    Hashes a plain-text password with PBKDF2-HMAC-SHA256 and a random salt.
    Why we need it: Never store raw passwords. PBKDF2 with 260,000 iterations
                    is the NIST-recommended minimum as of 2023.
    Returns a storable string in the format "salt_hex:dk_hex".
    """
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 260_000
    )
    return f"{salt}:{dk.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    """
    Re-derives the key for a candidate password and compares it to the stored hash.
    Why we need it: The stored hash contains the salt — we need to re-derive
                    using the same parameters to verify without storing the password.
    Uses secrets.compare_digest to prevent timing attacks.
    """
    try:
        salt, dk_hex = stored_hash.split(":", 1)
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt.encode("utf-8"), 260_000
        )
        return secrets.compare_digest(dk.hex(), dk_hex)
    except Exception:
        return False


# ==============================================================================
# SIGNUP
# ==============================================================================

def signup(username: str, full_name: str, email: str, phone: str,
           dob: str, gender: str, password: str) -> dict:
    """
    Registers a new user. Returns {"success": True, "user": {...}} or
    {"success": False, "error": "..."}.
    Why we need it: Encapsulates all validation and DB insertion for new accounts.
    """
    if not username.strip() or not password.strip():
        return {"success": False, "error": "Username and password cannot be empty."}

    conn = get_connection()
    try:
        password_hash = _hash_password(password)
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO users (username, full_name, email, phone, dob, gender, password_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (username.strip(), full_name.strip(), email.strip(),
             phone.strip(), dob, gender, password_hash)
        )
        conn.commit()
        user_id = cursor.lastrowid
        return {
            "success": True,
            "user": {
                "id": user_id,
                "username": username.strip(),
                "full_name": full_name.strip(),
                "email": email.strip(),
                "phone": phone.strip(),
            }
        }
    except Exception as e:
        error_msg = str(e)
        if "UNIQUE constraint failed: users.username" in error_msg:
            return {"success": False, "error": "Username already taken."}
        if "UNIQUE constraint failed: users.email" in error_msg:
            return {"success": False, "error": "An account with this email already exists."}
        return {"success": False, "error": f"Signup failed: {error_msg}"}
    finally:
        conn.close()


# ==============================================================================
# LOGIN
# ==============================================================================

def login(username: str, password: str) -> dict:
    """
    Verifies credentials against the DB. Returns the user dict on success,
    or {"success": False, "error": "..."} on failure.
    Why we need it: Centralises login so app.py never touches raw DB queries.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username.strip(),))
        row = cursor.fetchone()

        if row is None or not _verify_password(password, row["password_hash"]):
            return {"success": False, "error": "Invalid username or password."}

        return {
            "success": True,
            "user": {
                "id": row["id"],
                "username": row["username"],
                "full_name": row["full_name"],
                "email": row["email"],
                "phone": row["phone"],
            }
        }
    finally:
        conn.close()


# ==============================================================================
# SESSION MANAGEMENT
# ==============================================================================

def create_session(user_id: int) -> str:
    """
    Generates a secure random token, stores it in the DB with a 30-day expiry,
    and writes it to the local session.json file for auto-login persistence.
    Why we need it: This is the mechanism that keeps the user logged in across
                    app restarts without asking for credentials every time.
    Returns the token string.
    """
    token = secrets.token_hex(32)
    expires_at = datetime.utcnow() + timedelta(days=SESSION_EXPIRY_DAYS)

    conn = get_connection()
    try:
        # REPLACE handles the case where a session already exists for this user
        conn.execute(
            """INSERT OR REPLACE INTO sessions (user_id, token, expires_at)
               VALUES (?, ?, ?)""",
            (user_id, token, expires_at.isoformat())
        )
        conn.commit()
    finally:
        conn.close()

    # Persist to file for next app startup
    os.makedirs(os.path.dirname(SESSION_FILE), exist_ok=True)
    with open(SESSION_FILE, "w") as f:
        json.dump({
            "user_id": user_id,
            "token": token,
            "expires_at": expires_at.isoformat()
        }, f)

    return token


def load_session() -> dict | None:
    """
    Reads the local session file and validates the token against the DB.
    Returns the full user dict if the session is still valid, or None if
    it is missing, expired, or tampered with.
    Why we need it: This is called once at app startup to auto-login the user
                    without showing the login screen.
    """
    if not os.path.exists(SESSION_FILE):
        return None

    try:
        with open(SESSION_FILE) as f:
            session_data = json.load(f)

        expires_at = datetime.fromisoformat(session_data["expires_at"])
        if datetime.utcnow() > expires_at:
            _clear_session_file()
            return None

        conn = get_connection()
        try:
            cursor = conn.cursor()
            # Validate token exists in DB and matches user_id
            cursor.execute(
                "SELECT * FROM sessions WHERE token = ? AND user_id = ?",
                (session_data["token"], session_data["user_id"])
            )
            session_row = cursor.fetchone()
            if session_row is None:
                _clear_session_file()
                return None

            # Fetch user data
            cursor.execute("SELECT * FROM users WHERE id = ?", (session_data["user_id"],))
            user_row = cursor.fetchone()
            if user_row is None:
                _clear_session_file()
                return None

            return {
                "id": user_row["id"],
                "username": user_row["username"],
                "full_name": user_row["full_name"],
                "email": user_row["email"],
                "phone": user_row["phone"],
            }
        finally:
            conn.close()

    except Exception:
        _clear_session_file()
        return None


def logout(user_id: int) -> None:
    """
    Deletes the session from the DB and removes the local session file.
    Why we need it: Clean logout — after this call, load_session() will
                    return None and the user will see the login screen.
    """
    conn = get_connection()
    try:
        conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        conn.commit()
    finally:
        conn.close()
    _clear_session_file()


def _clear_session_file() -> None:
    """Removes the session.json file silently if it exists."""
    try:
        if os.path.exists(SESSION_FILE):
            os.remove(SESSION_FILE)
    except Exception:
        pass
