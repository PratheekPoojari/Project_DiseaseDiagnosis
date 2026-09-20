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

# Core pipeline
from src.nlp.predict import load_model as load_nlp_model, predict_symptom
from src.dip.predict import load_model as load_dip_model, predict_image
from src.fusion.fuse import fuse_predictions

# New modules
from src.nlp.file_parser import extract_text_from_file
from src.fusion.data_export import export_txt, export_csv, export_pdf, export_docx
from src.auth.db import init_db
from src.auth.auth import signup, login, create_session, load_session, logout
from src.health.tracker import save_diagnosis, get_history, compute_trend
from src.notifications.notifier import schedule_followup, DEMO_MODE

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
    st.session_state["auth_mode"] = "login"   # or "signup"


# ==============================================================================
# MODEL CACHING — loaded once into RAM regardless of reruns
# ==============================================================================
@st.cache_resource
def initialize_models():
    """
    Loads both the NLP (TF-IDF + SVM) and CNN (EfficientNet-B0) models once.
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


# ==============================================================================
# AUTH GATE — if not logged in, show login/signup and stop
# ==============================================================================

if st.session_state["user"] is None:

    st.title("🩺 Skin Disease Diagnosis System")
    if DEMO_MODE:
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
    with tab_signup:
        st.subheader("Create your account")
        with st.form("signup_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                su_username  = st.text_input("Username *")
                su_full_name = st.text_input("Full Name *")
                su_email     = st.text_input("Email *")
                su_phone     = st.text_input("Phone Number * (10-digit Indian)")
            with col_b:
                su_dob    = st.date_input("Date of Birth", min_value=datetime(1900, 1, 1))
                su_gender = st.selectbox("Gender", ["Prefer not to say", "Male", "Female", "Other"])
                su_pass   = st.text_input("Password *", type="password")
                su_pass2  = st.text_input("Confirm Password *", type="password")

            su_submitted = st.form_submit_button("Create Account", use_container_width=True, type="primary")

        if su_submitted:
            errors = []
            if not su_username:  errors.append("Username is required.")
            if not su_full_name: errors.append("Full name is required.")
            if not su_email:     errors.append("Email is required.")
            if not su_phone:     errors.append("Phone number is required.")
            if not su_pass:      errors.append("Password is required.")
            if su_pass != su_pass2: errors.append("Passwords do not match.")
            if su_phone and (not su_phone.isdigit() or len(su_phone) != 10):
                errors.append("Phone must be a 10-digit number.")

            if errors:
                for e in errors:
                    st.error(e)
            else:
                with st.spinner("Creating your account..."):
                    result = signup(
                        username=su_username,
                        full_name=su_full_name,
                        email=su_email,
                        phone=su_phone,
                        dob=str(su_dob),
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

# ---- Top bar: user greeting + logout ----
top_col1, top_col2 = st.columns([5, 1])
with top_col1:
    st.title("🩺 Multimodal Skin Disease Diagnosis System")
    if DEMO_MODE:
        st.info("🧪 **Demo Mode Active** — follow-up notifications fire every 30 seconds per \"day\".")
    st.markdown(
        f"> **⚠️ Academic Notice:** For demonstrative purposes only. Not a substitute for professional medical advice.  \n"
        f"> Logged in as **{user['full_name']}** (`{user['username']}`)"
    )
with top_col2:
    st.write("")
    st.write("")
    if st.button("🚪 Logout", use_container_width=True):
        logout(user["id"])
        st.session_state["user"] = None
        st.rerun()

st.divider()

# ==============================================================================
# HEALTH HISTORY — shown before the diagnosis form if history exists
# ==============================================================================
history = get_history(user["id"], limit=5)

if history:
    trend = compute_trend(history)
    with st.expander("📊 Your Health Trend (last sessions)", expanded=True):
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
    st.write("Type your symptoms, upload a file, OR record them using your microphone.")

    # --- File Input ---
    uploaded_text_file = st.file_uploader(
        "Upload symptom notes (.txt, .csv, .pdf, .docx)",
        type=["txt", "csv", "pdf", "docx"]
    )
    file_text = ""
    if uploaded_text_file is not None:
        try:
            with st.spinner("Extracting text..."):
                file_text = extract_text_from_file(
                    uploaded_text_file.getvalue(), uploaded_text_file.name
                )
            if file_text.strip():
                st.success(f"✅ Extracted text from {uploaded_text_file.name}")
            else:
                st.warning("No usable text found in the uploaded file.")
        except Exception as e:
            st.error(f"Error parsing file: {e}")

    # --- Voice Input (STT) ---
    audio_value = st.audio_input("🎙️ Record your symptoms")
    transcribed_text = ""
    if audio_value is not None:
        with st.spinner("Transcribing your voice..."):
            transcribed_text = transcribe_audio(audio_value.getvalue())
        if transcribed_text:
            st.success(f"✅ Transcribed: *\"{transcribed_text}\"*")
        else:
            st.warning("Could not transcribe. Please try again or type manually.")

    # Combine sources
    default_text = ""
    if file_text.strip():
        default_text += file_text + "\n\n"
    if transcribed_text.strip():
        default_text += transcribed_text

    symptom_text = st.text_area(
        "Symptoms",
        value=default_text.strip(),
        height=120,
        placeholder="E.g., I have extremely itchy, red patches on my elbows that have been peeling for three days..."
    )

with col2:
    st.header("📸 2. Affected Skin Area")
    st.write("Upload a photo OR use your camera.")

    img_source = st.radio("Image Source", ["Upload a File", "Use Camera"], horizontal=True)
    uploaded_image = None
    if img_source == "Upload a File":
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    else:
        uploaded_image = st.camera_input("Take a picture of the affected area")

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Image ready for analysis.", use_container_width=True)

# ==============================================================================
# DIAGNOSIS EXECUTION
# ==============================================================================
st.divider()
center_col = st.columns([1, 2, 1])[1]

with center_col:
    if st.button("🔍 Run Multimodal Diagnosis", use_container_width=True, type="primary"):
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

            # ==========================================================
            # RESULTS DISPLAY
            # ==========================================================
            if final_result["status"] == "error":
                st.error(f"Error: {final_result['message']}")
            else:
                st.success("Analysis Complete!")

                clean_name = final_result["prediction"].replace("_", " ").title()
                confidence = final_result["confidence"] * 100

                st.markdown(
                    f"<h2 style='text-align:center;color:#1E88E5;'>Predicted: {clean_name}</h2>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<h3 style='text-align:center;'>Confidence: {confidence:.2f}%</h3>",
                    unsafe_allow_html=True
                )

                # --- TTS ---
                tts_text = (
                    f"Analysis complete. The system predicts {clean_name} "
                    f"with a confidence of {confidence:.0f} percent. "
                    f"Please consult a medical professional for a confirmed diagnosis."
                )
                try:
                    audio_bytes_tts = synthesize_speech(tts_text)
                    st.markdown("#### 🔊 Audio Summary")
                    st.audio(audio_bytes_tts, format="audio/mp3", autoplay=True)
                except Exception:
                    st.caption("Audio readout unavailable — check internet connection.")

                if final_result["status"] == "low_confidence":
                    st.warning("⚠️ Low confidence: Input may not clearly match our trained conditions.")

                # --- Probability Bar Chart ---
                st.markdown("### Class Probabilities")
                probs = final_result["probabilities"]
                df_probs = pd.DataFrame({
                    "Condition": [k.replace("_", " ").title() for k in probs],
                    "Probability (%)": [v * 100 for v in probs.values()]
                }).sort_values(by="Probability (%)", ascending=False)
                st.bar_chart(df_probs, x="Condition", y="Probability (%)")

                # ==========================================================
                # SAVE TO HISTORY + SCHEDULE NOTIFICATIONS
                # ==========================================================
                save_diagnosis(
                    user_id=user["id"],
                    result=final_result,
                    symptom_text=symptom_text,
                    image_used=(uploaded_image is not None),
                )

                schedule_followup(
                    user_id=user["id"],
                    user_name=user["full_name"],
                    email=user["email"],
                    phone=user["phone"],
                    condition=final_result["prediction"],
                )

                interval_str = (
                    f"every {30}s (demo)" if DEMO_MODE
                    else "at Day 3 and Day 7"
                )
                st.info(
                    f"📬 Follow-up notifications scheduled {interval_str} "
                    f"to **{user['email']}** and **{user['phone']}**."
                )

                # ==========================================================
                # EXPORT / DOWNLOAD
                # ==========================================================
                st.markdown("### 📥 Download Report")
                st.caption("Save your diagnosis result in your preferred format.")

                fname_base = f"diagnosis_{user['username']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                dl_col1, dl_col2, dl_col3, dl_col4 = st.columns(4)

                with dl_col1:
                    st.download_button(
                        label="⬇️ TXT",
                        data=export_txt(final_result, symptom_text),
                        file_name=f"{fname_base}.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
                with dl_col2:
                    st.download_button(
                        label="⬇️ CSV",
                        data=export_csv(final_result, symptom_text),
                        file_name=f"{fname_base}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                with dl_col3:
                    st.download_button(
                        label="⬇️ PDF",
                        data=export_pdf(final_result, symptom_text),
                        file_name=f"{fname_base}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                with dl_col4:
                    st.download_button(
                        label="⬇️ DOCX",
                        data=export_docx(final_result, symptom_text),
                        file_name=f"{fname_base}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True,
                    )

                # Trigger a rerun after save so the trend chart above refreshes
                st.rerun()
