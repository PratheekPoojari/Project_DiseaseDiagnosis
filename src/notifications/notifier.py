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
            # Twilio trial accounts only accept predefined template names as the body.
            # The email channel carries the full personalized follow-up message.
            # To send custom SMS bodies, upgrade to a paid Twilio account.
            body="sms_appointment_reminders",
            from_=from_phone,
            to=phone
        )
        logging.info(f"SMS sent to {phone}. SID: {message_instance.sid}")
        return True

    except Exception as e:
        logging.error(f"Twilio SMS send failed to {phone}: {e}")
        return False


# ==============================================================================
# WHATSAPP — Twilio WhatsApp Business API
# ==============================================================================

def send_whatsapp(to_phone: str, message: str) -> bool:
    """
    Sends a WhatsApp message via the Twilio WhatsApp API.
    Requires .env variables:
        TWILIO_ACCOUNT_SID
        TWILIO_AUTH_TOKEN
        TWILIO_WHATSAPP_NUMBER (default: whatsapp:+14155238886 for Twilio sandbox)
    """
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    whatsapp_from = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")

    if not account_sid or not auth_token:
        logging.warning("WhatsApp not configured. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN in .env")
        return False

    if not whatsapp_from.startswith("whatsapp:"):
        whatsapp_from = f"whatsapp:{whatsapp_from}"

    phone = to_phone.strip()
    if phone.startswith("0"):
        phone = phone[1:]
    if not phone.startswith("+"):
        if phone.startswith("91") and len(phone) == 12:
            phone = "+" + phone
        elif len(phone) == 10:
            phone = "+91" + phone
        else:
            phone = "+" + phone

    whatsapp_to = f"whatsapp:{phone}"

    try:
        client = Client(account_sid, auth_token)
        message_instance = client.messages.create(
            body=message,
            from_=whatsapp_from,
            to=whatsapp_to
        )
        logging.info(f"WhatsApp sent to {whatsapp_to}. SID: {message_instance.sid}")
        return True

    except Exception as e:
        logging.error(f"Twilio WhatsApp send failed to {whatsapp_to}: {e}")
        return False


# ==============================================================================
# NOTIFICATION MESSAGE BUILDERS
# ==============================================================================

def _build_followup_message(user_name: str, condition: str, day: int) -> tuple[str, str]:
    """
    Builds the subject and body for a follow-up email notification.
    """
    condition_clean = condition.replace("_", " ").title()
    subject = f"[Skin Diagnosis] Day-{day} Follow-Up: {condition_clean}"
    body = (
        f"Hello {user_name},\n\n"
        f"This is your scheduled Day-{day} follow-up check-in from the Skin Disease Diagnosis System.\n\n"
        f"Your evaluated condition was: {condition_clean}\n\n"
        f"Please inspect the affected skin area: has the lesion improved, remained stable, or worsened? "
        f"You can log back into the app at any time to run a follow-up screening and check your longitudinal health trend.\n\n"
        f"Reminder: This automated system is for demonstrative/academic purposes only. "
        f"Always consult a qualified dermatologist for clinical evaluation and prescription treatment.\n\n"
        f"— Multimodal Skin Disease Diagnosis System"
    )
    return subject, body


def _build_sms_message(user_name: str, condition: str, day: int) -> str:
    """Builds a concise SMS body for a follow-up reminder."""
    condition_clean = condition.replace("_", " ").title()
    return (
        f"Hi {user_name}, Day-{day} reminder: Check your {condition_clean} progress "
        f"in the Skin Diagnosis app. Consult a dermatologist for confirmed medical guidance."
    )


def _build_whatsapp_message(user_name: str, condition: str, day: int) -> str:
    """Builds a structured WhatsApp reminder message."""
    condition_clean = condition.replace("_", " ").title()
    return (
        f"🩺 *Skin Disease Diagnosis — Day {day} Follow-Up*\n\n"
        f"Hello *{user_name}*,\n\n"
        f"This is your Day-{day} check-in regarding your recent evaluation for *{condition_clean}*.\n\n"
        f"• *Action:* Check your affected skin area.\n"
        f"• *Has it changed?* Note any change in size, border, redness, or itching.\n"
        f"• *App Tracker:* Log into the portal anytime to test again and view your 5-session health trend.\n\n"
        f"⚠️ _Academic Screening Notice: Always seek in-person medical care from a licensed dermatologist._"
    )


# ==============================================================================
# SCHEDULER (DAYS 3, 7, 14, 21, 30 + DAY 1 IN DEMO MODE)
# ==============================================================================

FOLLOWUP_SCHEDULE_DAYS = [3, 7, 14, 21, 30]
DEMO_SCHEDULE_DAYS = [1, 3, 7, 14, 21, 30]


def trigger_immediate_test_notification(
    user_name: str,
    email: str,
    phone: str,
    condition: str = "Fungal Infection"
) -> dict:
    """
    Synchronously triggers an immediate test follow-up across Email, SMS, and WhatsApp.
    Returns status and diagnostic messages for all three channels.
    """
    subject, email_body = _build_followup_message(user_name, condition, 1)
    sms_body = _build_sms_message(user_name, condition, 1)
    whatsapp_body = _build_whatsapp_message(user_name, condition, 1)

    # 1. Email via Gmail SMTP
    email_ok = send_email(email, subject, email_body)
    email_msg = f"Delivered to {email}" if email_ok else "Failed: Check EMAIL_ADDRESS and EMAIL_APP_PASSWORD in .env"

    # 2. SMS via Twilio
    sms_ok = send_sms(phone, sms_body)
    sms_msg = f"Delivered to {phone}" if sms_ok else "Failed: Ensure recipient phone is verified in your Twilio Console"

    # 3. WhatsApp via Twilio
    wa_ok = send_whatsapp(phone, whatsapp_body)
    wa_msg = f"Delivered to WhatsApp {phone}" if wa_ok else "Twilio Trial: Recipient must join Twilio sandbox by sending 'join <sandbox-code>' to +1 415 523 8886, or requires approved ContentSid."

    return {
        "email": email_ok,
        "email_msg": email_msg,
        "sms": sms_ok,
        "sms_msg": sms_msg,
        "whatsapp": wa_ok,
        "whatsapp_msg": wa_msg
    }


def schedule_followup(
    user_id: int,
    user_name: str,
    email: str,
    phone: str,
    condition: str,
) -> None:
    """
    Schedules follow-up notifications after diagnosis:
        - Normal Mode: Days 3, 7, 14, 21, 30
        - Demo Mode: Days 1, 3, 7, 14, 21, 30 (Day 1 fires after 1 x DEMO_INTERVAL_SECONDS)

    At each checkpoint, one Email, one SMS, and one WhatsApp message are dispatched.
    """
    scheduler = get_scheduler()

    # Remove any existing pending jobs for this user (new diagnosis resets the clock)
    for existing_job in scheduler.get_jobs():
        if existing_job.id.startswith(f"followup_{user_id}_"):
            existing_job.remove()

    now = datetime.utcnow()
    target_days = DEMO_SCHEDULE_DAYS if DEMO_MODE else FOLLOWUP_SCHEDULE_DAYS

    if DEMO_MODE:
        intervals = {
            day: now + timedelta(seconds=day * DEMO_INTERVAL_SECONDS)
            for day in target_days
        }
        mode_label = f"DEMO ({DEMO_INTERVAL_SECONDS}s per day unit; Day 1 fires at {DEMO_INTERVAL_SECONDS}s)"
    else:
        intervals = {
            day: now + timedelta(days=day)
            for day in target_days
        }
        mode_label = "NORMAL (Day units: 3, 7, 14, 21, 30)"

    for day, fire_at in intervals.items():
        subject, email_body = _build_followup_message(user_name, condition, day)
        sms_body = _build_sms_message(user_name, condition, day)
        whatsapp_body = _build_whatsapp_message(user_name, condition, day)

        # Capture loop variables in default args to avoid closure pitfall
        def _send_notification(
            _email=email, _phone=phone,
            _subject=subject, _email_body=email_body,
            _sms_body=sms_body, _whatsapp_body=whatsapp_body
        ):
            send_email(_email, _subject, _email_body)
            send_sms(_phone, _sms_body)
            send_whatsapp(_phone, _whatsapp_body)

        scheduler.add_job(
            func=_send_notification,
            trigger=DateTrigger(run_date=fire_at),
            id=f"followup_{user_id}_day{day}",
            replace_existing=True,
            misfire_grace_time=60,  # Fire even if up to 60s late
        )

    logging.info(
        f"Scheduled follow-ups (Email + SMS + WhatsApp) for user {user_id} | Mode: {mode_label}"
    )
