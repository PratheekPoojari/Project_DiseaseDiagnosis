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


def _build_test_message(user_name: str) -> tuple[str, str]:
    """Builds a neutral system delivery verification email without any disease diagnosis."""
    subject = "[Skin Diagnosis System] Delivery Test Notification"
    body = (
        f"Hello {user_name},\n\n"
        f"This is an automated delivery test verifying that your registered email address "
        f"is active and properly connected to the Skin Disease Diagnosis System.\n\n"
        f"All notification channels are operational. You will receive follow-up check-ins "
        f"here whenever you complete a clinical diagnosis in the application.\n\n"
        f"— Multimodal Skin Disease Diagnosis System"
    )
    return subject, body


def _build_sms_test_message(user_name: str) -> str:
    """Builds a neutral SMS body for system channel testing."""
    return f"Hi {user_name}, this is a delivery test from the Skin Diagnosis System. Outbound SMS verified."


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
    condition: str | None = None
) -> dict:
    """
    Synchronously triggers an immediate test follow-up across Email and SMS.
    If condition is None, dispatches a neutral system delivery verification.
    If condition is provided, dispatches a diagnosis-specific follow-up reminder.
    """
    if condition and condition.strip() and condition != "System Test":
        subject, email_body = _build_followup_message(user_name, condition, 1)
        sms_body = _build_sms_message(user_name, condition, 1)
    else:
        subject, email_body = _build_test_message(user_name)
        sms_body = _build_sms_test_message(user_name)

    # 1. Email via Gmail SMTP (fully automatic)
    email_ok = send_email(email, subject, email_body)
    email_msg = f"Delivered to {email}" if email_ok else "Failed: Check EMAIL_ADDRESS and EMAIL_APP_PASSWORD in .env"

    # 2. SMS via Twilio (semi-automatic — requires one-time phone OTP verification at signup)
    sms_ok = send_sms(phone, sms_body)
    sms_msg = f"Delivered to {phone}" if sms_ok else "Failed: Ensure recipient phone was verified via OTP during signup"

    # 3. WhatsApp — Future Scope
    wa_ok = False
    wa_msg = "WhatsApp: Future scope — requires approved Twilio WhatsApp Business API."

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

        # Capture loop variables in default args to avoid closure pitfall
        def _send_notification(
            _email=email, _phone=phone,
            _subject=subject, _email_body=email_body,
            _sms_body=sms_body
        ):
            send_email(_email, _subject, _email_body)
            send_sms(_phone, _sms_body)
            # WhatsApp: future scope — not dispatched here.

        scheduler.add_job(
            func=_send_notification,
            trigger=DateTrigger(run_date=fire_at),
            id=f"followup_{user_id}_day{day}",
            replace_existing=True,
            misfire_grace_time=60,  # Fire even if up to 60s late
        )

    logging.info(
        f"Scheduled follow-ups (Email + SMS) for user {user_id} | Mode: {mode_label}"
    )


def cancel_user_followups(user_id: int) -> None:
    """
    Cancels all scheduled follow-up reminder jobs for a specific user from APScheduler.
    Must be called during account deletion to immediately halt all future automated notifications.
    """
    scheduler = get_scheduler()
    for job in list(scheduler.get_jobs()):
        if job.id.startswith(f"followup_{user_id}_"):
            try:
                job.remove()
                logging.info(f"Removed follow-up job {job.id} for user {user_id}")
            except Exception as e:
                logging.warning(f"Failed to remove job {job.id}: {e}")


def purge_orphaned_followup_jobs() -> None:
    """
    Scans all jobs in APScheduler and removes any jobs belonging to users
    who no longer exist in the users database table.
    Guarantees no ghost notifications are sent for deleted accounts.
    """
    try:
        from src.auth.db import get_connection
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users")
        active_user_ids = {row["id"] for row in cursor.fetchall()}
        conn.close()

        scheduler = get_scheduler()
        for job in list(scheduler.get_jobs()):
            if job.id.startswith("followup_"):
                parts = job.id.split("_")
                if len(parts) >= 3 and parts[1].isdigit():
                    job_user_id = int(parts[1])
                    if job_user_id not in active_user_ids:
                        job.remove()
                        logging.info(f"Purged orphaned followup job {job.id} for non-existent user {job_user_id}")
    except Exception as e:
        logging.warning(f"Failed to purge orphaned followup jobs: {e}")


# ==============================================================================
# TWILIO TRIAL HELPERS — Phone Verification & WhatsApp Sandbox
# ==============================================================================

def _normalize_e164(phone: str) -> str:
    """Converts a 10-digit Indian number or partial E.164 to full E.164 (+91XXXXXXXXXX)."""
    clean = phone.strip()
    if clean.startswith("0"):
        clean = clean[1:]
    if not clean.startswith("+"):
        if clean.startswith("91") and len(clean) == 12:
            clean = "+" + clean
        elif len(clean) == 10:
            clean = "+91" + clean
        else:
            clean = "+" + clean
    return clean


def send_verification_otp(phone: str, channel: str = "sms") -> dict:
    """
    Sends a phone verification OTP via the Twilio Verify API.

    Uses Twilio Verify (client.verify.v2.services) instead of client.messages.create
    because Verify:
      - Works on Twilio Free Trial accounts without Verified Caller ID restrictions
      - Uses Twilio-managed message templates (bypasses Error 572006 template restriction)
      - Does NOT require the recipient number to be pre-registered on the account
      - Handles its own OTP lifecycle (generation, expiry, rate limiting)

    The generated OTP is NOT returned — Twilio owns it. Verification happens by calling
    check_verification_otp() which submits the user-entered code to Twilio for validation.

    Args:
        phone: 10-digit Indian number or international E.164 string.
        channel: 'sms' only (voice call removed — not supported on trial accounts).

    Returns:
        {
            "success": bool,
            "code": None,            # Twilio Verify owns the OTP — we never see it
            "channel": str,
            "trial_restricted": bool,
            "message": str
        }
    """
    account_sid   = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token    = os.getenv("TWILIO_AUTH_TOKEN")
    service_sid   = os.getenv("TWILIO_VERIFY_SERVICE_SID")

    if not account_sid or not auth_token or not service_sid:
        return {
            "success": False,
            "code": None,
            "channel": channel,
            "trial_restricted": False,
            "message": "Twilio Verify not configured. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_VERIFY_SERVICE_SID in .env."
        }

    clean_phone = _normalize_e164(phone)

    try:
        client = Client(account_sid, auth_token)
        verification = client.verify.v2.services(service_sid).verifications.create(
            to=clean_phone,
            channel="sms"      # always SMS for now; voice call future scope
        )
        logging.info(f"Twilio Verify OTP sent to {clean_phone}. Status: {verification.status}")
        return {
            "success": True,
            "code": None,          # Twilio owns the OTP
            "channel": "sms",
            "trial_restricted": False,
            "message": f"Verification code sent via SMS to {clean_phone}. Enter it below."
        }

    except Exception as e:
        err_str = str(e)
        logging.error(f"Twilio Verify send failed to {clean_phone}: {err_str}")
        return {
            "success": False,
            "code": None,
            "channel": "sms",
            "trial_restricted": False,
            "message": f"Failed to send OTP: {err_str}"
        }


def check_verification_otp(phone: str, code: str) -> dict:
    """
    Verifies a user-submitted OTP code against the Twilio Verify service.

    Twilio Verify manages OTP expiry (10 minutes) and invalid-attempt tracking
    automatically. We simply submit the code and get approved/denied back.

    Args:
        phone: The phone number that received the OTP (E.164 or 10-digit Indian).
        code: The 6-digit code the user typed in.

    Returns:
        {
            "approved": bool,
            "message": str
        }
    """
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token  = os.getenv("TWILIO_AUTH_TOKEN")
    service_sid = os.getenv("TWILIO_VERIFY_SERVICE_SID")

    if not account_sid or not auth_token or not service_sid:
        return {"approved": False, "message": "Twilio Verify not configured."}

    clean_phone = _normalize_e164(phone)

    try:
        client = Client(account_sid, auth_token)
        check = client.verify.v2.services(service_sid).verification_checks.create(
            to=clean_phone,
            code=code.strip()
        )
        approved = (check.status == "approved")
        logging.info(f"Verify check for {clean_phone}: {check.status}")
        return {
            "approved": approved,
            "message": "Phone verified successfully." if approved else "Incorrect or expired code."
        }

    except Exception as e:
        err_str = str(e)
        logging.error(f"Twilio Verify check failed for {clean_phone}: {err_str}")
        return {"approved": False, "message": f"Verification check failed: {err_str}"}

def request_phone_verification(phone: str) -> dict:
    """
    Programmatically triggers a Twilio verification call/SMS to a new user's
    phone number so that it gets added to the trial account's Verified Caller IDs.

    Why we need it: Twilio trial accounts can only send SMS/WhatsApp to verified
                    numbers. Instead of manually adding numbers in the Twilio
                    console, this initiates the process via the REST API:
                        1. Twilio calls the user's phone and reads them a 6-digit code
                           (or sends it via SMS, depending on Twilio account settings).
                        2. The user enters that code into the app.
                        3. The app calls check_phone_verification() to complete it.
                    Once confirmed, Twilio adds the number to Verified Caller IDs and
                    that number can receive SMS from your trial account.

    Returns:
        {
            "success": bool,
            "validation_code": str | None,   # 6-digit code shown in the Twilio call
            "message": str                   # human-readable status or error
        }
    """
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token  = os.getenv("TWILIO_AUTH_TOKEN")

    if not account_sid or not auth_token:
        return {
            "success": False,
            "validation_code": None,
            "message": "Twilio not configured. Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN in .env"
        }

    # Normalise to E.164 (same logic as send_sms)
    phone = phone.strip()
    if phone.startswith("0"):
        phone = phone[1:]
    if not phone.startswith("+"):
        if phone.startswith("91") and len(phone) == 12:
            phone = "+" + phone
        elif len(phone) == 10:
            phone = "+91" + phone
        else:
            phone = "+" + phone

    try:
        client = Client(account_sid, auth_token)
        validation = client.validation_requests.create(
            phone_number=phone,
            friendly_name=f"Skin Diagnosis App — {phone}",
        )
        logging.info(f"Verification call triggered for {phone}. Code: {validation.validation_code}")
        return {
            "success": True,
            "validation_code": validation.validation_code,
            "message": (
                f"Twilio is calling {phone}. "
                f"When prompted, enter the code shown on your screen on your phone keypad."
            )
        }
    except Exception as e:
        logging.error(f"Phone verification request failed for {phone}: {e}")
        return {
            "success": False,
            "validation_code": None,
            "message": f"Verification failed: {e}"
        }


def is_phone_verified(phone: str) -> bool:
    """
    Queries Twilio's OutgoingCallerIds list to check if a phone number has been
    successfully verified and can receive SMS on this trial account.
    """
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token  = os.getenv("TWILIO_AUTH_TOKEN")
    if not account_sid or not auth_token:
        return False

    phone = phone.strip()
    if phone.startswith("0"):
        phone = phone[1:]
    if not phone.startswith("+"):
        if phone.startswith("91") and len(phone) == 12:
            phone = "+" + phone
        elif len(phone) == 10:
            phone = "+91" + phone
        else:
            phone = "+" + phone

    try:
        client = Client(account_sid, auth_token)
        caller_ids = [item.phone_number for item in client.outgoing_caller_ids.list()]
        return phone in caller_ids
    except Exception as e:
        logging.error(f"Error checking caller ID verification for {phone}: {e}")
        return False


def get_whatsapp_sandbox_info() -> dict:
    """
    Returns the WhatsApp sandbox join instructions so the app can display them
    to a new user during or after signup.

    Why we need it: WhatsApp's policy (enforced by Meta, not Twilio) requires
                    recipients to explicitly opt in by texting the sandbox join
                    code to the sandbox number. There is no API to bypass this —
                    not on trial, not on paid accounts with an approved sender.
                    The only thing we can do is show the instructions clearly.

    Returns a dict with sandbox_number and join_code from .env, so the
    instructions reflect the actual sandbox this account is paired with.
    """
    sandbox_number = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
    join_code      = os.getenv("TWILIO_WHATSAPP_JOIN_CODE", "<your-sandbox-code>")
    # Strip the "whatsapp:" prefix for display
    display_number = sandbox_number.replace("whatsapp:", "")
    return {
        "sandbox_number": display_number,
        "join_code": join_code,
        "instructions": (
            f"To receive WhatsApp notifications, send the message  "
            f"**join {join_code}**  to  **{display_number}**  on WhatsApp. "
            f"This is a one-time opt-in required by WhatsApp's sandbox policy."
        )
    }
