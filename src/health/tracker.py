"""
File: src/health/tracker.py
Purpose: Saves each diagnosis to the user's history and computes a health trend
         by analysing the last 5 results.
Why we need it: The health tracking feature is what makes this app genuinely useful
                over time — a user can see if their condition is improving or worsening
                across multiple visits to the app.
"""

import json
from datetime import datetime
from src.auth.db import get_connection


# ==============================================================================
# SAVE
# ==============================================================================

def save_diagnosis(user_id: int, result: dict, symptom_text: str, image_used: bool) -> None:
    """
    Persists a single fusion result to the diagnosis_history table.
    Why we need it: Every prediction a logged-in user runs gets stored here
                    so we can compute trends and send follow-up notifications.

    Args:
        user_id:      The logged-in user's DB id.
        result:       The fusion result dict from fuse_predictions().
        symptom_text: The raw symptom string the user entered (may be empty).
        image_used:   True if an image was part of this diagnosis.
    """
    conn = get_connection()
    try:
        raw_probs = result.get("probabilities", {}) or {}
        sanitized_probs = {str(k): float(v) for k, v in raw_probs.items()}

        conn.execute(
            """INSERT INTO diagnosis_history
               (user_id, prediction, confidence, probabilities, symptom_text, image_used)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                result.get("prediction", "unknown"),
                float(result.get("confidence", 0.0)),
                json.dumps(sanitized_probs),
                symptom_text.strip() if symptom_text else "",
                1 if image_used else 0,
            )
        )
        conn.commit()
    finally:
        conn.close()


# ==============================================================================
# HISTORY
# ==============================================================================

def get_history(user_id: int, limit: int = 5) -> list[dict]:
    """
    Retrieves the N most recent diagnoses for a user, newest first.
    Why we need it: The trend analyser and the history display in the UI both
                    consume this list. Defaulting to 5 matches our trend window.

    Returns a list of dicts with keys:
        prediction, confidence, probabilities (dict), symptom_text,
        image_used (bool), created_at (str).
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT prediction, confidence, probabilities, symptom_text,
                      image_used, created_at
               FROM diagnosis_history
               WHERE user_id = ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (user_id, limit)
        )
        rows = cursor.fetchall()
        history = []
        for row in rows:
            history.append({
                "prediction":   row["prediction"],
                "confidence":   row["confidence"],
                "probabilities": json.loads(row["probabilities"]),
                "symptom_text": row["symptom_text"],
                "image_used":   bool(row["image_used"]),
                "created_at":   row["created_at"],
            })
        return history
    finally:
        conn.close()


# ==============================================================================
# TREND ANALYSIS
# ==============================================================================

def compute_trend(history: list[dict]) -> dict:
    """
    Analyses the last N diagnoses (up to 5) and returns a trend assessment.
    Why we need it: Showing the user whether their condition is improving, worsening,
                    or stable is the core value of long-term health tracking.

    Trend logic:
        - If the top predicted condition has CHANGED  → "new_finding"
        - If the condition is consistent AND confidence is FALLING  → "improving"
          Rationale: lower model confidence = less pronounced symptoms = getting better.
        - If the condition is consistent AND confidence is RISING   → "worsening"
        - If the confidence is roughly flat (within ±5%)            → "stable"
        - If there is only 1 result in history                      → "first_entry"

    Returns a dict with keys: status, condition, delta (confidence change), message.
    """
    if len(history) < 2:
        return {
            "status": "first_entry",
            "condition": history[0]["prediction"] if history else None,
            "delta": 0.0,
            "message": "First recorded diagnosis. Check back after your next session to see a trend."
        }

    # Check whether the top prediction is consistent across all entries in the window
    conditions = [h["prediction"] for h in history]
    if len(set(conditions)) > 1:
        return {
            "status": "new_finding",
            "condition": history[0]["prediction"],
            "previous_condition": history[1]["prediction"],
            "delta": 0.0,
            "message": (
                f"Your most recent diagnosis ({history[0]['prediction'].replace('_',' ').title()}) "
                f"differs from your previous result "
                f"({history[1]['prediction'].replace('_',' ').title()}). "
                "This may indicate a new or evolving condition — consult a dermatologist."
            )
        }

    # Consistent condition — measure confidence slope (newest vs oldest in window)
    condition = history[0]["prediction"].replace("_", " ").title()
    newest_conf = history[0]["confidence"]
    oldest_conf = history[-1]["confidence"]
    delta = newest_conf - oldest_conf  # positive = getting worse, negative = improving

    STABILITY_THRESHOLD = 0.05  # ±5% is considered stable

    if delta < -STABILITY_THRESHOLD:
        status = "improving"
        message = (
            f"✅ {condition}: Your condition appears to be **improving**. "
            f"Model confidence has dropped by {abs(delta)*100:.1f}% over your last {len(history)} sessions "
            f"— fewer detectable symptoms."
        )
    elif delta > STABILITY_THRESHOLD:
        status = "worsening"
        message = (
            f"⚠️ {condition}: Your condition appears to be **worsening**. "
            f"Model confidence has risen by {delta*100:.1f}% over your last {len(history)} sessions "
            f"— symptoms are becoming more pronounced. Please consult a doctor."
        )
    else:
        status = "stable"
        message = (
            f"📊 {condition}: Your condition is **stable**. "
            f"Confidence has changed by less than {STABILITY_THRESHOLD*100:.0f}% over "
            f"your last {len(history)} sessions."
        )

    return {
        "status": status,
        "condition": condition,
        "delta": delta,
        "history_length": len(history),
        "message": message,
    }
