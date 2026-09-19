"""
File: app.py
Purpose: The main Streamlit application serving as the UI for the Disease Diagnosis system.
         It handles loading models into memory, taking user input (text/image/camera/voice), 
         calling the prediction branches, running the fusion layer, and displaying results.
"""

import streamlit as st
import os
import tempfile
import speech_recognition as sr
import pandas as pd
from gtts import gTTS
import io

# Import our core pipeline scripts
from src.nlp.predict import load_model as load_nlp_model, predict_symptom
from src.dip.predict import load_model as load_dip_model, predict_image
from src.fusion.fuse import fuse_predictions

# ==============================================================================
# CONFIG & CACHING
# ==============================================================================
st.set_page_config(
    page_title="Multimodal Disease Diagnosis",
    page_icon="🩺",
    layout="wide"
)

# @st.cache_resource ensures models are only loaded into RAM once when the app starts,
# not every single time a user clicks a button.
@st.cache_resource
def initialize_models():
    with st.spinner("Loading NLP Model (TF-IDF + SVM)..."):
        nlp_model = load_nlp_model()

    with st.spinner("Loading Image CNN (EfficientNet-B0)..."):
        dip_model, dip_classes, dip_device = load_dip_model()

    return nlp_model, dip_model, dip_classes, dip_device

nlp_model, dip_model, dip_classes, dip_device = initialize_models()


def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Takes raw audio bytes from st.audio_input and returns the transcribed text string.
    Why we need it: st.audio_input gives us an audio file object; we need to pipe it 
                    through Google's free Speech Recognition API to convert it to text.
    Returns an empty string if transcription fails.
    """
    recognizer = sr.Recognizer()

    # Save audio bytes to a temp .wav file so SpeechRecognition can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        tmp_audio.write(audio_bytes)
        tmp_audio_path = tmp_audio.name

    try:
        with sr.AudioFile(tmp_audio_path) as source:
            audio_data = recognizer.record(source)
        # Use Google's free web speech API for transcription (requires internet)
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        # Audio was too quiet or unclear
        return ""
    except sr.RequestError:
        # Google API was unreachable
        st.warning("⚠️ Could not reach the speech recognition service. Check your internet connection.")
        return ""
    finally:
        os.remove(tmp_audio_path)


def synthesize_speech(text: str) -> bytes:
    """
    Converts a plain-text result summary into an MP3 audio clip using Google TTS.
    Why we need it: Gives the user an audio readout of the diagnosis result,
                    improving accessibility and making the app feel truly multimodal.
    Returns raw MP3 bytes that Streamlit can play directly in the browser.
    """
    tts = gTTS(text=text, lang='en', slow=False)
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer.read()


# ==============================================================================
# UI LAYOUT
# ==============================================================================
st.title("🩺 Multimodal Skin Disease Diagnosis System")
st.markdown("> **⚠️ Academic Notice:** This system is developed strictly for academic and demonstrative purposes. It does not replace professional medical consultation.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.header("📝 1. Symptom Description")
    st.write("Type your symptoms OR record them using your microphone.")

    # --- Voice Input (STT) ---
    audio_value = st.audio_input("🎙️ Record your symptoms")

    # If the user records audio, transcribe it and pre-fill the text area
    transcribed_text = ""
    if audio_value is not None:
        with st.spinner("Transcribing your voice..."):
            transcribed_text = transcribe_audio(audio_value.getvalue())
        if transcribed_text:
            st.success(f"✅ Transcribed: *\"{transcribed_text}\"*")
        else:
            st.warning("Could not transcribe audio. Please try again or type your symptoms manually.")

    # The text_area uses the transcribed text as the default value if available,
    # but the user can still edit it or type from scratch.
    symptom_text = st.text_area(
        "Symptoms",
        value=transcribed_text,
        height=120,
        placeholder="E.g., I have extremely itchy, red patches on my elbows that have been peeling for three days..."
    )

with col2:
    st.header("📸 2. Affected Skin Area")
    st.write("Upload a photo OR use your camera.")

    # Allow user to choose input method
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

                text_result = None
                image_result = None

                # --- 1. NLP Branch ---
                if symptom_text.strip():
                    text_result = predict_symptom(nlp_model, symptom_text)

                # --- 2. DIP Branch ---
                if uploaded_image is not None:
                    # Save the uploaded file temporarily so our predict script can read it
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                        tmp_file.write(uploaded_image.getvalue())
                        tmp_path = tmp_file.name

                    image_result = predict_image(dip_model, dip_classes, dip_device, tmp_path)

                    # Clean up temp file
                    os.remove(tmp_path)

                # --- 3. Late Fusion ---
                # Passes results to fuse module (weights: 60% Image, 40% Text default)
                final_result = fuse_predictions(text_result, image_result)

                # ==============================================================================
                # RESULTS DISPLAY
                # ==============================================================================
                if final_result["status"] == "error":
                    st.error(f"Error: {final_result['message']}")
                else:
                    st.success("Analysis Complete!")

                    # Big Prediction Text
                    clean_name = final_result['prediction'].replace("_", " ").title()
                    confidence = final_result['confidence'] * 100

                    st.markdown(f"<h2 style='text-align: center; color: #1E88E5;'>Predicted: {clean_name}</h2>", unsafe_allow_html=True)
                    st.markdown(f"<h3 style='text-align: center;'>Confidence: {confidence:.2f}%</h3>", unsafe_allow_html=True)

                    # --- TTS: Read the result summary aloud ---
                    # Build a natural-language summary sentence for the voice readout
                    tts_text = (
                        f"Analysis complete. "
                        f"The system predicts {clean_name} "
                        f"with a confidence of {confidence:.0f} percent. "
                        f"Please consult a medical professional for a confirmed diagnosis."
                    )
                    try:
                        audio_bytes = synthesize_speech(tts_text)
                        st.markdown("#### 🔊 Audio Summary")
                        st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                    except Exception:
                        st.caption("Audio readout unavailable — check internet connection.")

                    if final_result["status"] == "low_confidence":
                        st.warning("⚠️ Low confidence warning: The provided input may not clearly match our trained conditions.")

                    # Display probabilities as a nice bar chart
                    st.markdown("### Class Probabilities")
                    probs = final_result["probabilities"]
                    # Format for Streamlit bar chart
                    df_probs = pd.DataFrame({
                        "Condition": [k.replace("_", " ").title() for k in probs.keys()],
                        "Probability (%)": [v * 100 for v in probs.values()]
                    }).sort_values(by="Probability (%)", ascending=False)

                    st.bar_chart(df_probs, x="Condition", y="Probability (%)")
