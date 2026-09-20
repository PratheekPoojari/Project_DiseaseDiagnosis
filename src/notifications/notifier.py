"""
File: src/notifications/notifier.py
Purpose: Handles all outbound notifications — email (smtplib/Gmail) and SMS (Fast2SMS
         Quick SMS API) — and schedules follow-up reminders using APScheduler.
Why we need it: The follow-up notification module is a required feature in the project
                synopsis. It reminds users to check their condition after 3 and 7 days
                (or at short intervals in demo mode for demonstration purposes).

Demo mode is activated by passing `--demo` as a CLI argument:
    streamlit run app.py -- --demo
In demo mode, all intervals are measured in seconds rather than days.
"""

import os
import sys
import smtplib
import requests
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
from dotenv import load_dotenv

load_dotenv()

# Suppress APScheduler's noisy default logging to keep the Streamlit console clean
logging.getLogger("apscheduler").setLevel(logging.WARNING)

# ==============================================================================
# DEMO MODE DETECTION
# ==============================================================================
# Activated by: streamlit run app.py -- --demo
# In demo mode, follow-up intervals are in seconds. Normal mode uses days.
DEMO_MODE = "--demo" in sys.argv
DEMO_INTERVAL_SECONDS = 30   # Each "day" becomes 30 seconds in demo mode


# ==============================================================================
# SCHEDULER SINGLETON
# ==============================================================================

_scheduler: BackgroundScheduler | None = None


def get_scheduler() -> BackgroundScheduler:
    """
    Returns the global APScheduler BackgroundScheduler, starting it if needed.
    Why we need it: Only one scheduler instance should exist per process.
                    Streamlit reruns the script on every interaction, so we keep
                    the scheduler alive as a module-level singleton.
    """
    global _scheduler
    if _scheduler is None or not _scheduler.running:
        _scheduler = BackgroundScheduler()
        _scheduler.start()
    return _scheduler


# ==============================================================================
# EMAIL
# ==============================================================================

def send_email(to_email: str, subject: str, body: str) -> bool:
    """
    Sends a plain-text email via Gmail SMTP using App Password authentication.
    Why we need it: Email is the primary notification channel for follow-up reminders.

    Requires .env variables:
        EMAIL_ADDRESS     — your Gmail address
        EMAIL_APP_PASSWORD — a Gmail App Password (NOT your regular password).
                            Enable at: myaccount.google.com → Security → App Passwords.

    Returns True on success, False on failure (errors are logged, not raised,
    so a notification failure never crashes the main app).
    """
    email_address = os.getenv("EMAIL_ADDRESS")
    app_password = os.getenv("EMAIL_APP_PASSWORD")

    if not email_address or not app_password:
        logging.warning("Email not configured. Set EMAIL_ADDRESS and EMAIL_APP_PASSWORD in .env")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"Skin Diagnosis System <{email_address}>"
        msg["To"] = to_email
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(email_address, app_password)
            server.sendmail(email_address, to_email, msg.as_string())

        logging.info(f"Email sent to {to_email}: {subject}")
        return True

    except Exception as e:
        logging.error(f"Email send failed to {to_email}: {e}")
        return False


from twilio.rest import Client

# ==============================================================================
# SMS — Twilio Trial API
# ==============================================================================

def send_sms(to_phone: str, message: str) -> bool:
    """
    Sends an SMS via the Twilio API.
    Why we need it: SMS is a secondary notification channel — more immediate
                    than email for time-sensitive health reminders.

    Requires .env variables:
        TWILIO_ACCOUNT_SID
        TWILIO_AUTH_TOKEN
        TWILIO_PHONE_NUMBER — the Twilio sender number
        
    Note: On a Twilio trial account, you can only send messages to the phone 
    number you verified during signup.

    Returns True on success, False on failure.
    """
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_phone = os.getenv("TWILIO_PHONE_NUMBER")

    if not account_sid or not auth_token or not from_phone:
        logging.warning("SMS not configured. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER in .env")
        return False

    # Ensure phone number is E.164 format (starts with + and country code)
    phone = to_phone.strip()
    if phone.startswith("0"):
        phone = phone[1:]
    if not phone.startswith("+"):
        if phone.startswith("91") and len(phone) == 12:
            phone = "+" + phone
        elif len(phone) == 10:
            phone = "+91" + phone
        else:
            # Fallback for unexpected formats
            phone = "+" + phone

    try:
        client = Client(account_sid, auth_token)
        message_instance = client.messages.create(
            body=message,
            from_=from_phone,
            to=phone
        )
        logging.info(f"SMS sent to {phone}. SID: {message_instance.sid}")
        return True

    except Exception as e:
        logging.error(f"Twilio SMS send failed to {phone}: {e}")
        return False


# ==============================================================================
# NOTIFICATION MESSAGE BUILDERS
# ==============================================================================

def _build_followup_message(user_name: str, condition: str, day: int) -> tuple[str, str]:
    """
    Builds the subject and body for a follow-up notification.
    Why we need it: Consistent, friendly message templates rather than
                    ad-hoc strings scattered throughout the scheduler.

    Returns (subject, body) tuple.
    """
    condition_clean = condition.replace("_", " ").title()
    subject = f"[Skin Diagnosis] Day-{day} Follow-Up: {condition_clean}"
    body = (
        f"Hello {user_name},\n\n"
        f"This is your Day-{day} follow-up reminder from the Skin Disease Diagnosis System.\n\n"
        f"Your last diagnosed condition was: {condition_clean}\n\n"
        f"Please open the app and run a new diagnosis to track whether your condition "
        f"has improved, stayed the same, or worsened. Early detection of changes can "
        f"make a significant difference in treatment outcomes.\n\n"
        f"Reminder: This system is for academic/demonstrative purposes only. "
        f"Always consult a qualified dermatologist for a confirmed diagnosis.\n\n"
        f"— Skin Diagnosis System"
    )
    return subject, body


def _build_sms_message(user_name: str, condition: str, day: int) -> str:
    """Builds a concise SMS body for a follow-up reminder."""
    condition_clean = condition.replace("_", " ").title()
    return (
        f"Hi {user_name}, Day-{day} reminder: Open the Skin Diagnosis app to "
        f"check your {condition_clean} progress. Consult a doctor for medical advice."
    )


# ==============================================================================
# SCHEDULER
# ==============================================================================

def schedule_followup(
    user_id: int,
    user_name: str,
    email: str,
    phone: str,
    condition: str,
) -> None:
    """
    Schedules two follow-up notifications after a diagnosis:
        - Day 3 (or 3 × DEMO_INTERVAL_SECONDS seconds in demo mode)
        - Day 7 (or 7 × DEMO_INTERVAL_SECONDS seconds in demo mode)

    Each notification fires both an email and an SMS.
    Why we need it: The project synopsis specifies time-delayed post-prediction
                    follow-ups. APScheduler's DateTrigger lets us fire a one-shot
                    job at an exact future datetime.

    Jobs are tagged with user_id so existing jobs can be replaced if the user
    runs another diagnosis before the previous reminders have fired.
    """
    scheduler = get_scheduler()

    # Remove any existing pending jobs for this user (new diagnosis resets the clock)
    for existing_job in scheduler.get_jobs():
        if existing_job.id.startswith(f"followup_{user_id}_"):
            existing_job.remove()

    now = datetime.utcnow()

    if DEMO_MODE:
        intervals = {
            3: now + timedelta(seconds=3 * DEMO_INTERVAL_SECONDS),
            7: now + timedelta(seconds=7 * DEMO_INTERVAL_SECONDS),
        }
        mode_label = f"DEMO ({DEMO_INTERVAL_SECONDS}s units)"
    else:
        intervals = {
            3: now + timedelta(days=3),
            7: now + timedelta(days=7),
        }
        mode_label = "NORMAL (day units)"

    for day, fire_at in intervals.items():
        subject, email_body = _build_followup_message(user_name, condition, day)
        sms_body = _build_sms_message(user_name, condition, day)

        # Capture loop variables in default args to avoid closure pitfall
        def _send_notification(
            _email=email, _phone=phone,
            _subject=subject, _email_body=email_body, _sms_body=sms_body
        ):
            send_email(_email, _subject, _email_body)
            send_sms(_phone, _sms_body)

        scheduler.add_job(
            func=_send_notification,
            trigger=DateTrigger(run_date=fire_at),
            id=f"followup_{user_id}_day{day}",
            replace_existing=True,
            misfire_grace_time=60,  # Fire even if up to 60s late
        )

    logging.info(
        f"Scheduled Day-3 and Day-7 follow-ups for user {user_id} | Mode: {mode_label}"
    )
