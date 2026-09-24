"""
File: app.py
Purpose: The main Streamlit application for the Multimodal Skin Disease Diagnosis System.
         Handles: persistent auth gate (login/signup), model inference (NLP + CNN + fusion),
         health tracking with trend analysis, result export, and follow-up notifications.

Run normally:   streamlit run app.py
Run demo mode:  streamlit run app.py -- --demo
  (Demo mode fires notifications in seconds instead of days for demonstration.)
"""

import streamlit as st
import os
import sys
import tempfile
import speech_recognition as sr
import pandas as pd
from gtts import gTTS
from datetime import datetime
import io
import calendar
import re
import string

# Core pipeline
from src.nlp.predict import load_model as load_nlp_model, predict_symptom
from src.dip.predict import load_model as load_dip_model, predict_image
from src.fusion.fuse import fuse_predictions
from src.fusion.narrative import generate_specialist_narrative

# Supporting modules
from src.nlp.file_parser import extract_text_from_file
from src.fusion.data_export import export_txt, export_csv, export_pdf, export_docx
from src.auth.db import init_db
from src.auth.auth import signup, login, create_session, load_session, logout
from src.health.tracker import save_diagnosis, get_history, compute_trend
from src.notifications.notifier import schedule_followup, trigger_immediate_test_notification, DEMO_MODE


# ==============================================================================
# STARTUP — DB init (runs once; safe to call every time)
# ==============================================================================
init_db()


# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="Skin Disease Diagnosis",
    page_icon="🩺",
    layout="wide"
)


# ==============================================================================
# SESSION STATE BOOTSTRAP
# ==============================================================================
# On every cold start, try to restore a previous session from the token file.
if "user" not in st.session_state:
    restored_user = load_session()
    st.session_state["user"] = restored_user  # None if no valid session

if "auth_mode" not in st.session_state:
    st.session_state["auth_mode"] = "login"

# This key persists diagnosis results across Streamlit reruns so they don't
# flash and vanish when the script re-executes (the core bug in the old version).
if "current_diagnosis" not in st.session_state:
    st.session_state["current_diagnosis"] = None

# Store the symptom text and image flag that produced the current diagnosis
if "last_symptom_text" not in st.session_state:
    st.session_state["last_symptom_text"] = ""
if "last_image_used" not in st.session_state:
    st.session_state["last_image_used"] = False

# Demo mode settings — defaults from CLI flag, overridable in sidebar
if "demo_mode" not in st.session_state:
    st.session_state["demo_mode"] = DEMO_MODE
if "demo_interval" not in st.session_state:
    st.session_state["demo_interval"] = 30
if "symptom_input_text" not in st.session_state:
    st.session_state["symptom_input_text"] = ""


# ==============================================================================
# MODEL CACHING — loaded once into RAM regardless of reruns
# ==============================================================================
@st.cache_resource
def initialize_models():
    """
    Loads both the NLP (Bio_ClinicalBERT) and CNN (ConvNeXt-Base) models once.
    @st.cache_resource persists the return value across all Streamlit reruns.
    """
    nlp_model = load_nlp_model()
    dip_model, dip_classes, dip_device = load_dip_model()
    return nlp_model, dip_model, dip_classes, dip_device


nlp_model, dip_model, dip_classes, dip_device = initialize_models()


# ==============================================================================
# HELPERS
# ==============================================================================

def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Converts raw audio bytes (from st.audio_input) into text via Google Web Speech API.
    Returns an empty string if transcription fails.
    """
    recognizer = sr.Recognizer()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        with sr.AudioFile(tmp_path) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        st.warning("⚠️ Could not reach the speech recognition service. Check your internet.")
        return ""
    finally:
        os.remove(tmp_path)


def synthesize_speech(text: str) -> bytes:
    """
    Converts a plain-text result summary into an MP3 audio clip via gTTS.
    Returns raw MP3 bytes for st.audio().
    """
    tts = gTTS(text=text, lang="en", slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


def check_password_strength(password: str) -> tuple:
    """
    Evaluates password strength on a 0-5 scale.
    Returns (score, label, color) for UI display.

    Criteria checked:
        1. Length >= 8 characters
        2. Contains at least one lowercase letter
        3. Contains at least one uppercase letter
        4. Contains at least one digit
        5. Contains at least one special character
    """
    if not password:
        return (0, "", "gray")

    score = 0
    if len(password) >= 8:
        score += 1
    if re.search(r"[a-z]", password):
        score += 1
    if re.search(r"[A-Z]", password):
        score += 1
    if re.search(r"\d", password):
        score += 1
    if re.search(r"[!@#$%^&*()_+\-=\[\]{}|;:,.<>?/~`]", password):
        score += 1

    if score <= 2:
        return (score, "Weak 🔴", "red")
    elif score <= 3:
        return (score, "Medium 🟡", "orange")
    elif score <= 4:
        return (score, "Good 🟢", "green")
    else:
        return (score, "Strong 💪", "green")


# ==============================================================================
# AUTH GATE — if not logged in, show login/signup and stop
# ==============================================================================

if st.session_state["user"] is None:

    st.title("🩺 Skin Disease Diagnosis System")
    if st.session_state["demo_mode"]:
        st.info("🧪 **Demo Mode Active** — notifications fire every 30 seconds per day.")

    tab_login, tab_signup = st.tabs(["🔑 Log In", "📝 Sign Up"])

    # ---- LOGIN TAB ----
    with tab_login:
        st.subheader("Welcome back")
        with st.form("login_form"):
            username_in = st.text_input("Username")
            password_in = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log In", use_container_width=True, type="primary")

        if submitted:
            if not username_in or not password_in:
                st.error("Please fill in both fields.")
            else:
                with st.spinner("Verifying credentials..."):
                    result = login(username_in, password_in)
                if result["success"]:
                    st.session_state["user"] = result["user"]
                    create_session(result["user"]["id"])
                    st.success(f"Welcome back, {result['user']['full_name']}!")
                    st.rerun()
                else:
                    st.error(result["error"])

    # ---- SIGNUP TAB ----
    # NOTE: This section does NOT use st.form() on purpose.
    # Streamlit forms suppress all reruns until submission, which prevents
    # the live password strength meter from updating as the user types.
    # Regular widgets trigger reruns on change, enabling real-time feedback.
    # Model loading is cached via @st.cache_resource so reruns are cheap.
    with tab_signup:
        st.subheader("Create your account")

        col_a, col_b = st.columns(2)
        with col_a:
            su_username  = st.text_input("Username *", key="su_username")
            su_full_name = st.text_input("Full Name *", key="su_full_name")
            su_email     = st.text_input("Email *", key="su_email")
            su_phone     = st.text_input("Phone Number * (10-digit Indian)", key="su_phone")

        with col_b:
            # Enhanced DOB: 3 dropdowns spanning 100+ years instead of
            # the decade-limited st.date_input calendar widget.
            st.markdown("**Date of Birth**")
            dob_col_y, dob_col_m, dob_col_d = st.columns(3)
            current_year = datetime.now().year
            with dob_col_y:
                su_dob_year = st.selectbox(
                    "Year",
                    options=list(range(current_year, 1919, -1)),
                    index=20,  # Default to ~20 years ago
                    key="su_dob_year",
                )
            with dob_col_m:
                su_dob_month = st.selectbox(
                    "Month",
                    options=list(range(1, 13)),
                    format_func=lambda m: calendar.month_name[m],
                    key="su_dob_month",
                )
            with dob_col_d:
                # Dynamically calculate max days for the selected month/year
                max_day = calendar.monthrange(su_dob_year, su_dob_month)[1]
                su_dob_day = st.selectbox(
                    "Day",
                    options=list(range(1, max_day + 1)),
                    key="su_dob_day",
                )

            su_gender = st.selectbox(
                "Gender",
                ["Prefer not to say", "Male", "Female", "Other"],
                key="su_gender",
            )

            # Password taskbar row: label on the left, live strength badge at the end of the taskbar
            pw_bar_l, pw_bar_r = st.columns([1, 1])
            curr_pass = st.session_state.get("su_pass", "")
            pw_score, pw_label, pw_color = check_password_strength(curr_pass) if curr_pass else (0, "Not Entered", "gray")

            with pw_bar_l:
                st.markdown("**Password ***")
            with pw_bar_r:
                st.markdown(
                    f"<div style='text-align:right;'><span id='pw_live_badge' style='font-size:12px; font-weight:700; "
                    f"color:{pw_color}; border:1px solid {pw_color}; border-radius:4px; padding:2px 8px;'>"
                    f"Strength: {pw_label}</span></div>",
                    unsafe_allow_html=True
                )

            st.caption(
                "Guidelines: Minimum 8 characters with uppercase, lowercase, numbers, and symbols."
            )
            su_pass = st.text_input(
                "Password *", type="password", label_visibility="collapsed",
                key="su_pass",
            )

            su_pass2 = st.text_input(
                "Confirm Password *", type="password", key="su_pass2",
            )

        # Regular button instead of form_submit_button
        su_submitted = st.button(
            "Create Account", use_container_width=True, type="primary",
            key="signup_btn",
        )

        if su_submitted:
            errors = []
            if not su_username:   errors.append("Username is required.")
            if not su_full_name:  errors.append("Full name is required.")
            if not su_email:      errors.append("Email is required.")
            if not su_phone:      errors.append("Phone number is required.")
            if not su_pass:       errors.append("Password is required.")
            if su_pass != su_pass2: errors.append("Passwords do not match.")
            if su_phone and (not su_phone.isdigit() or len(su_phone) != 10):
                errors.append("Phone must be a 10-digit number.")

            # Enforce minimum password strength
            if su_pass:
                pw_score, _, _ = check_password_strength(su_pass)
                if pw_score < 3:
                    errors.append(
                        "Password is too weak. It must be at least 8 characters "
                        "with uppercase, lowercase, digits, and a special character."
                    )

            if errors:
                for e in errors:
                    st.error(e)
            else:
                # Construct DOB string from the 3 dropdowns
                dob_str = f"{su_dob_year}-{su_dob_month:02d}-{su_dob_day:02d}"

                with st.spinner("Creating your account..."):
                    result = signup(
                        username=su_username,
                        full_name=su_full_name,
                        email=su_email,
                        phone=su_phone,
                        dob=dob_str,
                        gender=su_gender,
                        password=su_pass,
                    )
                if result["success"]:
                    st.session_state["user"] = result["user"]
                    create_session(result["user"]["id"])
                    st.success(f"Account created! Welcome, {result['user']['full_name']}!")
                    st.rerun()
                else:
                    st.error(result["error"])

    # Stop rendering the rest of the app until the user is logged in
    st.stop()


# ==============================================================================
# MAIN APP — only reached if user is logged in
# ==============================================================================

user = st.session_state["user"]

# ==============================================================================
# SIDEBAR — Settings & Demo Mode Controls
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Settings")

    # Demo mode toggle
    st.session_state["demo_mode"] = st.toggle(
        "🧪 Demo Mode (fast notifications)",
        value=st.session_state["demo_mode"],
        help="When enabled, follow-up notifications fire in seconds instead of days."
    )

    if st.session_state["demo_mode"]:
        st.session_state["demo_interval"] = st.slider(
            "Demo interval (seconds per 'day')",
            min_value=10,
            max_value=120,
            value=st.session_state["demo_interval"],
            step=10,
            help="Each 'day' in notification scheduling equals this many seconds."
        )
        st.caption(
            f"⚡ **Schedule:** Day 1: **{st.session_state['demo_interval']}s**, "
            f"Day 3: **{3 * st.session_state['demo_interval']}s**, "
            f"Day 7: **{7 * st.session_state['demo_interval']}s**"
        )

    st.divider()

    # Manual notification testing
    st.markdown("**Outbound Channel Test**")
    st.caption("Verify your Email, SMS & WhatsApp delivery instantly:")
    if st.button("🚀 Test Notifications Now", use_container_width=True, key="sb_test_notif"):
        with st.spinner("Testing channels..."):
            cond = "Fungal Infection"
            if st.session_state.get("current_diagnosis") and st.session_state["current_diagnosis"].get("prediction"):
                cond = st.session_state["current_diagnosis"]["prediction"]
            res_test = trigger_immediate_test_notification(
                user_name=user["full_name"],
                email=user["email"],
                phone=user["phone"],
                condition=cond
            )
        if res_test["email"]:
            st.success(f"📧 Email: {res_test['email_msg']}")
        else:
            st.error(f"📧 Email: {res_test['email_msg']}")

        if res_test["sms"]:
            st.success(f"📱 SMS: {res_test['sms_msg']}")
        else:
            st.error(f"📱 SMS: {res_test['sms_msg']}")

        if res_test["whatsapp"]:
            st.success(f"💬 WhatsApp: {res_test['whatsapp_msg']}")
        else:
            st.warning(f"💬 WhatsApp: {res_test['whatsapp_msg']}")

    st.divider()

    # Clear diagnosis button
    if st.session_state["current_diagnosis"] is not None:
        if st.button("🗑️ Clear Current Results", use_container_width=True):
            st.session_state["current_diagnosis"] = None
            st.session_state["last_symptom_text"] = ""
            st.session_state["last_image_used"] = False
            st.rerun()

    st.divider()

    # Logout
    if st.button("🚪 Logout", use_container_width=True):
        logout(user["id"])
        st.session_state["user"] = None
        st.session_state["current_diagnosis"] = None
        st.rerun()


# ---- Top bar: user greeting ----
st.title("🩺 Multimodal Skin Disease Diagnosis System")

if st.session_state["demo_mode"]:
    st.info(
        f"🧪 **Demo Mode Active** — follow-up notifications fire every "
        f"**{st.session_state['demo_interval']}s** per \"day\". "
        f"Adjust in the sidebar."
    )

st.markdown(
    f"> **⚠️ Academic Notice:** For demonstrative purposes only. "
    f"Not a substitute for professional medical advice.  \n"
    f"> Logged in as **{user['full_name']}** (`{user['username']}`)"
)

st.divider()


# ==============================================================================
# HEALTH HISTORY — shown before the diagnosis form if history exists
# ==============================================================================
history = get_history(user["id"], limit=5)

if history:
    trend = compute_trend(history)
    with st.expander("📊 Your Health Trend (last sessions)", expanded=False):
        status_icon = {
            "improving":    "✅",
            "worsening":    "⚠️",
            "stable":       "📊",
            "new_finding":  "🔄",
            "first_entry":  "📋",
        }.get(trend["status"], "ℹ️")

        st.markdown(f"### {status_icon} {trend['message']}")

        # Confidence-over-time chart
        history_df = pd.DataFrame([
            {
                "Session": f"#{i+1}  {h['created_at'][:10]}",
                "Condition": h["prediction"].replace("_", " ").title(),
                "Confidence (%)": round(h["confidence"] * 100, 2),
            }
            for i, h in enumerate(reversed(history))
        ])
        st.line_chart(history_df, x="Session", y="Confidence (%)", use_container_width=True)

        # History table
        st.dataframe(history_df, use_container_width=True, hide_index=True)

    st.divider()


# ==============================================================================
# INPUT COLUMNS
# ==============================================================================
col1, col2 = st.columns(2)

with col1:
    st.header("📝 1. Symptom Description")
    st.caption("Enter patient symptoms below. Use the toolbar icons to dictate via voice or attach medical files.")

    # Toolbar row: Taskbar with label, mic icon popover, and '+' attach file popover
    tb_col1, tb_col2, tb_col3 = st.columns([6, 1, 1])

    with tb_col1:
        st.markdown("**Clinical Notes**")

    with tb_col2:
        with st.popover("🎙️", help="Voice Dictation (Google Speech-to-Text)"):
            st.markdown("##### 🎙️ Voice Dictation")
            st.caption("Click the microphone below to record and transcribe symptoms:")
            audio_value = st.audio_input("Record Symptoms", key="voice_audio_input", label_visibility="collapsed")
            if audio_value is not None:
                with st.spinner("Transcribing audio..."):
                    transcribed = transcribe_audio(audio_value.getvalue())
                if transcribed:
                    st.success(f'✅ Transcribed: *"{transcribed}"*')
                    if transcribed not in st.session_state["symptom_input_text"]:
                        if st.session_state["symptom_input_text"].strip():
                            st.session_state["symptom_input_text"] += " " + transcribed.strip()
                        else:
                            st.session_state["symptom_input_text"] = transcribed.strip()
                        st.rerun()
                else:
                    st.warning("Could not transcribe speech. Please speak clearly into your microphone.")

    with tb_col3:
        with st.popover("➕", help="Attach Medical Records (.txt, .csv, .pdf, .docx)"):
            st.markdown("##### 📁 Attach Medical File")
            st.caption("Supported formats: `.txt`, `.csv`, `.pdf`, `.docx`")
            uploaded_doc = st.file_uploader(
                "Upload Document",
                type=["txt", "csv", "pdf", "docx"],
                key="doc_uploader",
                label_visibility="collapsed"
            )
            if uploaded_doc is not None:
                try:
                    with st.spinner("Extracting clinical text..."):
                        extracted = extract_text_from_file(uploaded_doc.getvalue(), uploaded_doc.name)
                    if extracted.strip():
                        st.success(f"✅ Extracted text from {uploaded_doc.name}")
                        if extracted.strip() not in st.session_state["symptom_input_text"]:
                            if st.session_state["symptom_input_text"].strip():
                                st.session_state["symptom_input_text"] += "\n\n" + extracted.strip()
                            else:
                                st.session_state["symptom_input_text"] = extracted.strip()
                            st.rerun()
                    else:
                        st.warning("No readable symptom text found in file.")
                except Exception as e:
                    st.error(f"Error parsing file: {e}")

    # Main text input area on the same taskbar view
    symptom_text = st.text_area(
        "Patient Symptoms",
        value=st.session_state["symptom_input_text"],
        height=160,
        placeholder="E.g., I have intensely itchy, red scaly plaques on my elbows and scalp that have been peeling for three weeks...",
        label_visibility="collapsed",
        key="symptom_text_widget",
    )
    # Synchronize manual edits back to session state
    st.session_state["symptom_input_text"] = symptom_text

    # Word count and clear shortcut
    meta_c1, meta_c2 = st.columns([3, 1])
    with meta_c1:
        word_count = len(symptom_text.split()) if symptom_text.strip() else 0
        st.caption(f"📝 {word_count} words entered | Clinical NLP ready")
    with meta_c2:
        if symptom_text.strip():
            if st.button("🗑️ Clear", key="clear_symptom_btn", use_container_width=True):
                st.session_state["symptom_input_text"] = ""
                st.rerun()

with col2:
    st.header("📸 2. Affected Skin Area")
    st.caption("Upload a high-resolution photograph or capture one live using your camera.")

    tab_upload, tab_camera = st.tabs(["🖼️ Upload Photograph", "📷 Camera Capture"])
    img_from_upload = None
    img_from_camera = None

    with tab_upload:
        img_from_upload = st.file_uploader(
            "Upload lesion photo (.jpg, .jpeg, .png)",
            type=["jpg", "jpeg", "png"],
            key="dip_tab_uploader",
            help="High-resolution dermoscopy or clinical macro photograph"
        )

    with tab_camera:
        img_from_camera = st.camera_input(
            "Take a picture of the affected skin area",
            key="dip_tab_camera",
            help="Use your laptop webcam or mobile device rear camera"
        )

    uploaded_image = img_from_upload if img_from_upload is not None else img_from_camera

    if uploaded_image is not None:
        st.image(
            uploaded_image,
            caption="Lesion photograph ready for ConvNeXt-Base neural analysis",
            use_container_width=True
        )


# ==============================================================================
# DIAGNOSIS PRE-FLIGHT SUMMARY & TRIGGER
# ==============================================================================
st.divider()

# Pre-flight modality badges
pf_c1, pf_c2, pf_c3 = st.columns(3)
with pf_c1:
    has_text = bool(symptom_text.strip())
    st.markdown(f"**NLP Branch:** {'🟢 Ready' if has_text else '⚪ Not Provided'}")
    if has_text:
        words = symptom_text.strip().split()
        st.caption(f"{len(words)} words recorded")
with pf_c2:
    has_img = uploaded_image is not None
    st.markdown(f"**DIP Branch:** {'🟢 Ready' if has_img else '⚪ Not Provided'}")
    if has_img:
        st.caption("Lesion image loaded & preprocessed")
with pf_c3:
    if has_text and has_img:
        mode_str = "⚡ **Multimodal (60% Image / 40% Text)**"
    elif has_img:
        mode_str = "🖼️ **DIP Image Only (ConvNeXt-Base)**"
    elif has_text:
        mode_str = "📝 **Clinical NLP Only (Bio_ClinicalBERT)**"
    else:
        mode_str = "⚠️ **Awaiting Input**"
    st.markdown(f"**Execution Mode:**\n{mode_str}")

st.write("")
center_col = st.columns([1, 2, 1])[1]

with center_col:
    run_clicked = st.button(
        "🔍 Run Multimodal Diagnosis",
        use_container_width=True,
        type="primary"
    )

if run_clicked:
    if not symptom_text.strip() and uploaded_image is None:
        st.error("Please provide at least a symptom description OR an image.")
    else:
        with st.spinner("Running deep learning models..."):

            text_result  = None
            image_result = None

            # --- 1. NLP Branch ---
            if symptom_text.strip():
                text_result = predict_symptom(nlp_model, symptom_text)

            # --- 2. DIP Branch ---
            if uploaded_image is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp.write(uploaded_image.getvalue())
                    tmp_path = tmp.name
                image_result = predict_image(dip_model, dip_classes, dip_device, tmp_path)
                os.remove(tmp_path)

            # --- 3. Late Fusion (60% image / 40% text) ---
            final_result = fuse_predictions(text_result, image_result)

        # Store results in session state so they persist across reruns
        st.session_state["current_diagnosis"] = final_result
        st.session_state["last_symptom_text"] = symptom_text
        st.session_state["last_image_used"] = (uploaded_image is not None)

        # Save to history and schedule notifications only on new diagnosis
        if final_result.get("status") != "error":
            save_diagnosis(
                user_id=user["id"],
                result=final_result,
                symptom_text=symptom_text,
                image_used=(uploaded_image is not None),
            )

            # Use the sidebar demo_mode settings for notification scheduling
            if st.session_state["demo_mode"]:
                # Override the module-level interval with the user's slider value
                from src.notifications import notifier
                notifier.DEMO_MODE = True
                notifier.DEMO_INTERVAL_SECONDS = st.session_state["demo_interval"]

            schedule_followup(
                user_id=user["id"],
                user_name=user["full_name"],
                email=user["email"],
                phone=user["phone"],
                condition=final_result["prediction"],
            )


# ==============================================================================
# RESULTS DISPLAY — reads from session state, persists across reruns
# ==============================================================================

diagnosis = st.session_state["current_diagnosis"]

if diagnosis is not None:
    st.divider()

    if diagnosis["status"] == "error":
        st.error(f"Error: {diagnosis['message']}")
    else:
        st.success("✅ Analysis Complete!")

        saved_symptom = st.session_state.get("last_symptom_text", "")
        image_was_used = st.session_state.get("last_image_used", False)
        narrative = generate_specialist_narrative(diagnosis, saved_symptom, image_was_used)

        # ---- Clinical Status Card (Refined, Non-Overpowering) ----
        stat_col1, stat_col2 = st.columns([3, 1])
        with stat_col1:
            st.markdown(f"### 🩺 {narrative['formal_name']}")
            st.caption(f"Taxonomy: **{narrative['category']}** | Analysis: **{'Multimodal (Image + Symptoms)' if image_was_used and saved_symptom else ('DIP Dermoscopy' if image_was_used else 'Clinical NLP Text')}**")
        with stat_col2:
            st.markdown(
                f"<div style='text-align:right; padding:8px 12px; background-color:{narrative['badge_bg']}; "
                f"border:1px solid {narrative['badge_color']}; border-radius:8px; display:inline-block;'>"
                f"<span style='font-size:16px; font-weight:700; color:{narrative['badge_color']};'>{narrative['confidence_str']} Confidence</span><br/>"
                f"<small style='color:#555;'>{narrative['certainty_label']}</small>"
                f"</div>",
                unsafe_allow_html=True
            )

        # ---- Urgency / Triage Notice ----
        if "Urgent" in narrative["urgency"]:
            st.error(f"🚨 **Clinical Triage Advisory:** {narrative['urgency']}")
        elif "Prompt" in narrative["urgency"] or "Timely" in narrative["urgency"]:
            st.warning(f"⚠️ **Clinical Triage Advisory:** {narrative['urgency']}")
        else:
            st.info(f"ℹ️ **Clinical Triage Advisory:** {narrative['urgency']}")

        if diagnosis["status"] == "low_confidence":
            st.warning("⚠️ **Low Statistical Concordance:** Clinical presentation does not cleanly align with single-disease benchmarks. Consider secondary differentials below.")

        # ---- Specialist Consultation Narrative Container ----
        with st.container(border=True):
            st.markdown("#### Specialist Consultation Assessment")
            st.markdown(narrative["lead_paragraph"])

            col_diag_a, col_diag_b = st.columns(2)
            with col_diag_a:
                st.markdown("##### Morphological & Clinical Indicators Observed")
                for finding in narrative["visual_findings"]:
                    st.markdown(f"• {finding}")

            with col_diag_b:
                st.markdown("##### Recommended Next Clinical Actions")
                for action in narrative["clinical_actions"]:
                    st.markdown(f"1. {action}")

            st.markdown("##### Differential Diagnosis Analysis")
            st.markdown(narrative["differential_analysis"])

            if narrative.get("red_flags"):
                rf_list = "\n".join([f"• **{rf}**" for rf in narrative["red_flags"]])
                st.error(f"**🚨 Red-Flag Symptoms Requiring Immediate Emergency Care:**\n\n{rf_list}")

        # ---- Audio Readout (Specialist Spoken Consultation) ----
        try:
            audio_bytes_tts = synthesize_speech(narrative["tts_script"])
            st.markdown("#### 🔊 Spoken Consultation Summary")
            st.audio(audio_bytes_tts, format="audio/mp3", autoplay=False)
        except Exception:
            st.caption("Audio consultation unavailable — check internet connection.")

        # ---- Class Probabilities (Tucked into a clean expander) ----
        with st.expander("📊 Detailed Model Probability Distribution", expanded=False):
            probs = diagnosis["probabilities"]
            df_probs = pd.DataFrame({
                "Condition": [k.replace("_", " ").title() for k in probs],
                "Probability (%)": [v * 100 for v in probs.values()]
            }).sort_values(by="Probability (%)", ascending=False)
            st.bar_chart(df_probs, x="Condition", y="Probability (%)")
            st.dataframe(df_probs, use_container_width=True, hide_index=True)

        # --- Notification confirmation ---
        if st.session_state["demo_mode"]:
            interval = st.session_state["demo_interval"]
            interval_str = f"at Day 1, 3, 7, 14, 21, and 30 (Day 1 fires at {interval}s in demo mode)"
        else:
            interval_str = "at Day 3, 7, 14, 21, and 30"

        st.info(
            f"📬 **Follow-up Notifications Scheduled:** {interval_str} "
            f"via **Email**, **SMS**, and **WhatsApp** to **{user['email']}** and **{user['phone']}**."
        )

        if st.button("🚀 Send Test Follow-Up Now (Email, SMS & WhatsApp)", use_container_width=True, key="res_test_notif"):
            with st.spinner("Dispatching live notifications..."):
                test_res = trigger_immediate_test_notification(
                    user_name=user["full_name"],
                    email=user["email"],
                    phone=user["phone"],
                    condition=diagnosis["prediction"]
                )
            col_nt1, col_nt2, col_nt3 = st.columns(3)
            with col_nt1:
                if test_res["email"]:
                    st.success(f"📧 **Email**\n\n{test_res['email_msg']}")
                else:
                    st.error(f"📧 **Email**\n\n{test_res['email_msg']}")
            with col_nt2:
                if test_res["sms"]:
                    st.success(f"📱 **SMS**\n\n{test_res['sms_msg']}")
                else:
                    st.error(f"📱 **SMS**\n\n{test_res['sms_msg']}")
            with col_nt3:
                if test_res["whatsapp"]:
                    st.success(f"💬 **WhatsApp**\n\n{test_res['whatsapp_msg']}")
                else:
                    st.warning(f"💬 **WhatsApp**\n\n{test_res['whatsapp_msg']}")

        # ==========================================================
        # EXPORT / DOWNLOAD
        # ==========================================================
        st.markdown("### 📥 Download Report")
        st.caption("Save your diagnosis result in your preferred format.")

        saved_symptom = st.session_state["last_symptom_text"]
        fname_base = f"diagnosis_{user['username']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dl_col1, dl_col2, dl_col3, dl_col4 = st.columns(4)

        with dl_col1:
            st.download_button(
                label="⬇️ TXT",
                data=export_txt(diagnosis, saved_symptom),
                file_name=f"{fname_base}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with dl_col2:
            st.download_button(
                label="⬇️ CSV",
                data=export_csv(diagnosis, saved_symptom),
                file_name=f"{fname_base}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl_col3:
            st.download_button(
                label="⬇️ PDF",
                data=export_pdf(diagnosis, saved_symptom),
                file_name=f"{fname_base}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        with dl_col4:
            st.download_button(
                label="⬇️ DOCX",
                data=export_docx(diagnosis, saved_symptom),
                file_name=f"{fname_base}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )
